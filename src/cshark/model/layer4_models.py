"""Layer 4: transcription from sequence + epigenome state + 3D contacts.

Layers 1-3 of C.Shark go sequence -> CTCF/ATAC -> (RAD21) -> Hi-C.  Layer 4 is
the read-out that consumes *all* of it and predicts transcription.

Why a separate layer rather than another head on layer 3
-------------------------------------------------------
Layer 3 already emits 1D tracks, but its 1D decoder branches off the shared
latent and never looks at the contact map it just predicted -- the 3D structure
is an output, not a feature.  For transcription that is the wrong way round: an
enhancer regulates a promoter *because* they contact, so the contact map should
be an input to the expression read-out.  :class:`HierarchicalExpressionModel`
makes it one, in either of two ways (``hic_mode``):

``bias``  (default)
    The predicted contact map enters the transformer as an additive attention
    bias, ``logits += w_h * log1p(A) + b_h`` with a learned per-head scale.  This
    is the same trick AlphaGenome uses to let its 2D pair embeddings steer the 1D
    trunk, and it keeps the map differentiable, so the transcription loss
    back-propagates into layer 3's Hi-C head.
``propagate``
    Explicit message passing: each latent bin additionally sees a
    contact-weighted average of every other bin, ``L' = L @ A_norm``.  More
    literally "enhancer signal reaches the promoter along contacts", and cheaper
    than attention, but it cannot modulate *what* is exchanged the way attention
    can.  ``both`` runs the two together; ``none`` is the ablation that measures
    what the contact map is worth.

Because a bin here is 4,096 bp and the contact map is 512x512 over the same
2,097,152 bp window, contact bins index latent bins one-to-one -- no resampling.

Heads (what to supervise, and why each one earns its place)
-----------------------------------------------------------
``coverage``       stranded RNA-seq (and optionally CAGE / PRO-cap) at 64 bp over
                   the whole window.  The primary target.
``splice_class``   5-way per-base donor/acceptor/background at 1 bp over a
                   central crop.  This is the head that matters most for RNA:
                   coverage steps by an order of magnitude at every exon
                   boundary, and without splice supervision the coverage head has
                   to discover exon structure with no gradient pointing at it.
                   Targets come free from the annotation (no new experiment).
``splice_usage``   per-tissue sigmoid at annotated sites -- graded usage rather
                   than presence.  Needs junction-derived PSI, so it is optional.
``junction``       donor x acceptor counts, predicted by indexing the 1 bp
                   embedding at candidate sites (AlphaGenome's formulation).
                   Optional; needs junction counts.
``gene_expression`` one scalar per annotated gene, pooled over the gene body.
                   Cheap, and it optimises the quantity that downstream analysis
                   actually uses instead of leaving it to post-hoc aggregation.

The 1 bp branch is deliberately *not* just an upsampled trunk latent: a 4,096x
downsampled representation cannot resolve a GT dinucleotide, so a SpliceAI-style
dilated convolution over the raw one-hot sequence runs alongside it and the two
are fused.  Long-range context comes from the trunk, the motif from the sequence
branch.
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

import cshark.model.blocks as blocks
from cshark.data.splice_features import NUM_SPLICE_CLASSES


class DilatedResBlock1D(nn.Module):
    """Length-preserving dilated residual block.

    ``blocks.ResBlockDilated1D`` pads by ``dil``, which only preserves length for
    kernel size 3.  Splice motifs want SpliceAI-width kernels (11) over several
    dilations, so this pads by ``dil * (size - 1) // 2`` and stays 'same' for any
    odd kernel.
    """

    def __init__(self, size=11, hidden=64, dil=1, dropout=0.0):
        super().__init__()
        pad = dil * (size - 1) // 2
        self.res = nn.Sequential(
            nn.Conv1d(hidden, hidden, size, padding=pad, dilation=dil),
            nn.BatchNorm1d(hidden), nn.ReLU(),
            nn.Dropout(dropout) if dropout else nn.Identity(),
            nn.Conv1d(hidden, hidden, size, padding=pad, dilation=dil),
            nn.BatchNorm1d(hidden))
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.relu(self.res(x) + x)


# ---------------------------------------------------------------------------
# Hi-C conditioning
# ---------------------------------------------------------------------------
class ContactAttentionBias(nn.Module):
    """Turns a contact map into a per-head additive attention bias.

    The map is log1p'd first: contact frequency spans several orders of
    magnitude and decays as a power law with distance, so the raw value would
    saturate every head on the diagonal.  ``distance_bias`` optionally adds a
    learned function of |i-j| so the head can separate "close in 3D" from
    "close on the sequence", which is the entire distinction that makes a
    contact map informative beyond linear proximity.
    """

    def __init__(self, num_heads=8, use_distance_bias=True, max_bins=512):
        super().__init__()
        self.num_heads = num_heads
        self.scale = nn.Parameter(torch.ones(num_heads))
        self.bias = nn.Parameter(torch.zeros(num_heads))
        self.use_distance_bias = use_distance_bias
        if use_distance_bias:
            # 32 log-spaced distance buckets, learned per head.
            self.n_buckets = 32
            self.distance_embed = nn.Parameter(torch.zeros(num_heads, self.n_buckets))
            self.register_buffer('bucket_of', self._make_buckets(max_bins),
                                 persistent=False)

    def _make_buckets(self, n):
        idx = torch.arange(n)
        d = (idx[None, :] - idx[:, None]).abs().float()
        b = torch.log1p(d) / math.log(n + 1.0) * (self.n_buckets - 1)
        return b.round().long().clamp(0, self.n_buckets - 1)

    def forward(self, contact_map):
        """``(B, N, N)`` contact map -> ``(B * num_heads, N, N)`` additive bias."""
        B, N, _ = contact_map.shape
        a = torch.log1p(torch.clamp(contact_map, min=0.0))
        # Standardise per sample so the bias magnitude does not depend on the
        # (arbitrary) normalisation of whichever cooler produced the map.
        a = (a - a.mean(dim=(1, 2), keepdim=True)) / (a.std(dim=(1, 2), keepdim=True) + 1e-6)
        bias = a.unsqueeze(1) * self.scale.view(1, -1, 1, 1) + self.bias.view(1, -1, 1, 1)
        if self.use_distance_bias:
            bucket = self.bucket_of[:N, :N]
            bias = bias + self.distance_embed[:, bucket].unsqueeze(0)
        return bias.reshape(B * self.num_heads, N, N)


class ContactPropagation(nn.Module):
    """Contact-weighted message passing over the latent bins."""

    def __init__(self, latent_dim, rounds=1, dropout=0.1):
        super().__init__()
        self.rounds = rounds
        self.mix = nn.ModuleList([
            nn.Sequential(nn.Conv1d(latent_dim * 2, latent_dim, 1),
                          nn.BatchNorm1d(latent_dim), nn.ReLU(), nn.Dropout(dropout))
            for _ in range(rounds)])

    def forward(self, latent, contact_map):
        """``latent`` ``(B, D, N)``; ``contact_map`` ``(B, N, N)``."""
        a = torch.clamp(contact_map, min=0.0)
        # Row-normalise into a transition matrix: each bin receives a weighted
        # average of its contacts, so the update is scale-free in the map.
        a = a / (a.sum(dim=-1, keepdim=True) + 1e-6)
        for mix in self.mix:
            messages = torch.bmm(latent, a.transpose(1, 2))     # (B, D, N)
            latent = latent + mix(torch.cat([latent, messages], dim=1))
        return latent


class ContactConditionedTransformer(nn.Module):
    """Transformer over latent bins, optionally biased by the contact map."""

    def __init__(self, latent_dim, layers=8, num_heads=8, dim_feedforward=1024,
                 dropout=0.1, hic_mode='bias', propagate_rounds=1, max_bins=512):
        super().__init__()
        self.hic_mode = hic_mode
        self.num_heads = num_heads
        self.pos_encoder = blocks.PositionalEncoding(latent_dim, dropout=dropout,
                                                     max_len=max_bins)
        layer = blocks.TransformerLayer(latent_dim, nhead=num_heads, dropout=dropout,
                                        dim_feedforward=dim_feedforward,
                                        batch_first=True)
        self.encoder = blocks.TransformerEncoder(layer, layers, record_attn=False)
        self.attn_bias = (ContactAttentionBias(num_heads, max_bins=max_bins)
                          if hic_mode in ('bias', 'both') else None)
        self.propagate = (ContactPropagation(latent_dim, rounds=propagate_rounds)
                          if hic_mode in ('propagate', 'both') else None)

    def forward(self, latent, contact_map=None):
        """``latent`` ``(B, D, N)`` -> ``(B, D, N)``."""
        if self.propagate is not None and contact_map is not None:
            latent = self.propagate(latent, contact_map)
        x = latent.transpose(1, 2)                                # (B, N, D)
        x = self.pos_encoder(x)
        mask = None
        if self.attn_bias is not None and contact_map is not None:
            mask = self.attn_bias(contact_map).to(x.dtype)
        out = self.encoder(x, mask=mask)
        if isinstance(out, tuple):
            out = out[0]
        return out.transpose(1, 2)


# ---------------------------------------------------------------------------
# 1 bp branch
# ---------------------------------------------------------------------------
class SequenceMotifEncoder(nn.Module):
    """SpliceAI-style dilated residual stack on the raw one-hot sequence.

    The trunk latent is 4,096x downsampled, which cannot represent a GT/AG
    dinucleotide at all, so the 1 bp heads need a path that never leaves base
    resolution.  Dilations double each block, giving a receptive field of a few
    kb -- enough for the branch-point/polypyrimidine context that decides a real
    acceptor -- while the trunk supplies everything longer range.
    """

    def __init__(self, in_channels=5, hidden=64, num_blocks=8, filter_size=11):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv1d(in_channels, hidden, filter_size, padding='same'),
            nn.BatchNorm1d(hidden), nn.ReLU())
        self.blocks = nn.Sequential(*[
            DilatedResBlock1D(filter_size, hidden=hidden, dil=2 ** i)
            for i in range(num_blocks)])

    def forward(self, seq_onehot):
        """``(B, S, C)`` one-hot -> ``(B, hidden, S)``."""
        return self.blocks(self.stem(seq_onehot.transpose(1, 2).float()))


class LatentUpsampler(nn.Module):
    """Upsample the trunk latent to 1 bp, halving channels as length doubles."""

    def __init__(self, latent_dim, out_channels=64, num_upsample=12,
                 min_channels=32, filter_size=3):
        super().__init__()
        layers = []
        c = latent_dim
        for _ in range(num_upsample):
            nxt = max(min_channels, c // 2)
            layers.append(nn.Sequential(
                nn.ConvTranspose1d(c, nxt, kernel_size=2, stride=2),
                nn.BatchNorm1d(nxt), nn.ReLU(),
                nn.Conv1d(nxt, nxt, filter_size, padding='same'),
                nn.BatchNorm1d(nxt), nn.ReLU()))
            c = nxt
        self.layers = nn.Sequential(*layers)
        self.proj = nn.Conv1d(c, out_channels, 1)

    def forward(self, latent):
        return self.proj(self.layers(latent))


class SpliceHeads(nn.Module):
    """1 bp splice-site classification and (optionally) per-track usage."""

    def __init__(self, in_channels, num_usage_tracks=0, hidden=96, num_blocks=4,
                 filter_size=11):
        super().__init__()
        self.fuse = nn.Sequential(
            nn.Conv1d(in_channels, hidden, 1), nn.BatchNorm1d(hidden), nn.ReLU())
        self.blocks = nn.Sequential(*[
            DilatedResBlock1D(filter_size, hidden=hidden, dil=2 ** i)
            for i in range(num_blocks)])
        self.hidden = hidden
        self.to_class = nn.Conv1d(hidden, NUM_SPLICE_CLASSES, 1)
        self.to_usage = (nn.Conv1d(hidden, num_usage_tracks, 1)
                         if num_usage_tracks > 0 else None)

    def forward(self, fused):
        h = self.blocks(self.fuse(fused))
        out = {'splice_class_logits': self.to_class(h).transpose(1, 2),
               'splice_embedding': h}
        if self.to_usage is not None:
            out['splice_usage_logits'] = self.to_usage(h).transpose(1, 2)
        return out


class JunctionHead(nn.Module):
    """Donor x acceptor junction counts from the 1 bp embedding.

    Follows AlphaGenome: project the 1 bp embedding, gather it at candidate
    donor and acceptor positions, apply a learned per-tissue scale/offset and a
    rotary encoding of genomic position, then score every donor-acceptor pair by
    dot product.  Encoding position rotationally is what lets the head express
    "this donor pairs with the acceptor 4 kb downstream" without ever seeing an
    absolute coordinate.
    """

    def __init__(self, in_channels, hidden=64, num_tissues=1, max_distance=2 ** 21):
        super().__init__()
        self.hidden = hidden
        self.num_tissues = num_tissues
        self.max_distance = max_distance
        self.proj = nn.Conv1d(in_channels, hidden, 1)
        self.donor_scale = nn.Parameter(torch.ones(num_tissues, hidden))
        self.donor_offset = nn.Parameter(torch.zeros(num_tissues, hidden))
        self.acceptor_scale = nn.Parameter(torch.ones(num_tissues, hidden))
        self.acceptor_offset = nn.Parameter(torch.zeros(num_tissues, hidden))

    def _rope(self, x, positions):
        """``x`` ``(B, P, T, H)``, ``positions`` ``(B, P)`` -> rotated ``x``."""
        H = x.shape[-1]
        half = H // 2
        inv_freq = torch.exp(
            -math.log(self.max_distance) * torch.arange(half, device=x.device) / half)
        ang = positions.float()[:, :, None] * inv_freq[None, None, :]   # (B, P, H/2)
        cos, sin = torch.cos(ang)[:, :, None, :], torch.sin(ang)[:, :, None, :]
        x1, x2 = x[..., :half], x[..., half:2 * half]
        return torch.cat([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1)

    def forward(self, embedding_1bp, donor_pos, acceptor_pos):
        """``embedding_1bp`` ``(B, C, S)``; positions ``(B, P)`` -> ``(B, P, P, T)``."""
        h = self.proj(embedding_1bp)                                # (B, H, S)
        B = h.shape[0]
        bidx = torch.arange(B, device=h.device).unsqueeze(1)

        def gather(pos, scale, offset):
            e = h[bidx, :, pos.clamp(min=0)]                        # (B, P, H)
            e = e[:, :, None, :] * scale[None, None] + offset[None, None]
            return self._rope(e, pos)                               # (B, P, T, H)

        d = gather(donor_pos, self.donor_scale, self.donor_offset)
        a = gather(acceptor_pos, self.acceptor_scale, self.acceptor_offset)
        # (B, P_d, T, H) x (B, P_a, T, H) -> (B, P_d, P_a, T)
        logits = torch.einsum('bdth,bath->bdat', d, a) / math.sqrt(self.hidden)
        return F.softplus(logits)


class GeneExpressionHead(nn.Module):
    """One scalar per annotated gene, mean-pooled over its latent bins.

    Optimising gene-level expression directly matters because that is the number
    downstream analysis uses; leaving it to post-hoc aggregation of a coverage
    track makes the model's errors and the analysis's aggregation interact in
    ways nobody audits.  Strand is passed in because a gene's expression should
    read from the matching RNA strand.
    """

    def __init__(self, latent_dim, hidden=256, num_outputs=1):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(latent_dim + 2, hidden), nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(hidden, hidden // 2), nn.ReLU(),
            nn.Linear(hidden // 2, num_outputs))

    def forward(self, latent, gene_bins, gene_strand, gene_mask):
        """``latent`` ``(B, D, N)``; ``gene_bins`` ``(B, G, 2)`` bin start/end;
        ``gene_strand`` ``(B, G)`` in {+1, -1}; ``gene_mask`` ``(B, G)`` bool.
        Returns ``(B, G, num_outputs)``."""
        B, D, N = latent.shape
        idx = torch.arange(N, device=latent.device)[None, None, :]
        lo = gene_bins[:, :, 0:1]
        hi = torch.clamp(gene_bins[:, :, 1:2], min=lo + 1)
        span = ((idx >= lo) & (idx < hi)).to(latent.dtype)          # (B, G, N)
        span = span * gene_mask.unsqueeze(-1).to(latent.dtype)
        pooled = torch.einsum('bgn,bdn->bgd', span, latent)
        pooled = pooled / torch.clamp(span.sum(-1, keepdim=True), min=1.0)
        length = torch.log1p((hi - lo).to(latent.dtype))            # (B, G, 1)
        feats = torch.cat([pooled, gene_strand.unsqueeze(-1).to(latent.dtype), length], -1)
        return self.mlp(feats)


# ---------------------------------------------------------------------------
# Layer 4
# ---------------------------------------------------------------------------
class HierarchicalExpressionModel(nn.Module):
    """Sequence + epigenome + contacts -> transcription.

    Parameters mirror the layer-3 model where they mean the same thing, so a
    layer-4 checkpoint can be configured from the same hparams.
    """

    def __init__(self,
                 num_input_tracks,
                 num_coverage_tracks,
                 window=2_097_152,
                 mat_size=512,
                 target_1d_size=32768,
                 latent_dim=512,
                 num_blocks=11,
                 seq_filter_size=15,
                 epi_filter_size=5,
                 hic_mode='bias',
                 transformer_layers=8,
                 num_heads=8,
                 propagate_rounds=1,
                 splice_crop=131_072,
                 num_splice_usage_tracks=0,
                 junction_head=False,
                 num_junction_tissues=1,
                 gene_expression_head=True,
                 motif_channels=64,
                 upsample_channels=64,
                 grad_checkpoint=False,
                 mask_track_idx=(),
                 mask_prob=1.0,
                 coverage_track_means=None):
        super().__init__()
        # Recompute activations in the backward pass instead of storing them.
        # The encoder runs over 2,097,152 positions and the 1 bp branch over the
        # crop, so their stored activations dominate layer 4's memory; trading
        # ~30% more compute for them is what makes a batch larger than 1 fit.
        self.grad_checkpoint = grad_checkpoint
        self.window = window
        self.mat_size = mat_size
        self.target_1d_size = target_1d_size
        self.latent_dim = latent_dim
        self.hic_mode = hic_mode
        self.splice_crop = int(splice_crop)
        self.num_coverage_tracks = num_coverage_tracks
        self.num_splice_usage_tracks = num_splice_usage_tracks

        # -- input masking ---------------------------------------------------
        # Zeroing a channel rather than removing it keeps the channel count (and
        # so the parameter count) identical to an unmasked run, which is what
        # makes masked and unmasked checkpoints directly comparable and lets the
        # channel-ablation diagnostic keep working.  The tracks are ln(1+x), so
        # zero is literally "no signal" -- a value the model already sees in
        # untranscribed regions, not an out-of-distribution sentinel.
        self.mask_prob = float(mask_prob)
        mask = torch.ones(num_input_tracks, dtype=torch.float32)
        for i in mask_track_idx:
            mask[int(i)] = 0.0
        # persistent=False: derived from hparams at construction, so a checkpoint
        # written before this existed still loads with strict=True.
        self.register_buffer('input_track_mask', mask, persistent=False)
        self.masked_track_idx = tuple(int(i) for i in mask_track_idx)

        self.encoder = blocks.EncoderSplit(
            num_input_tracks, hidden=latent_dim, output_size=latent_dim,
            num_blocks=num_blocks, num_bases=5,
            epi_filter_size=epi_filter_size, seq_filter_size=seq_filter_size)
        # EncoderSplit downsamples by 2 in conv_start and by 2 in every ConvBlock,
        # so the factor is 2**(num_blocks + 1): 4,096 bp per latent bin at the
        # default num_blocks=11, i.e. 512 bins -- exactly the Hi-C map size, which
        # is what lets contact bins index latent bins one-to-one.
        self.downsample = 2 ** (num_blocks + 1)
        self.latent_bins = window // self.downsample
        self.bp_per_bin = self.downsample
        if mat_size and self.latent_bins != mat_size:
            print(f'[layer4] NOTE: {self.latent_bins} latent bins vs contact map '
                  f'size {mat_size}; the map will be resampled to match.')

        self.trunk = ContactConditionedTransformer(
            latent_dim, layers=transformer_layers, num_heads=num_heads,
            hic_mode=hic_mode, propagate_rounds=propagate_rounds,
            max_bins=self.latent_bins)

        # -- coverage (whole window, 64 bp) --------------------------------
        n_up = int(math.log2(target_1d_size // self.latent_bins))
        self.coverage_decoder = blocks.Decoder1D(
            num_target_tracks=num_coverage_tracks, latent_dim=latent_dim,
            target_length=target_1d_size, num_upsample_blocks=n_up)
        means = (torch.as_tensor(coverage_track_means, dtype=torch.float32)
                 if coverage_track_means is not None
                 else torch.ones(num_coverage_tracks))
        self.register_buffer('coverage_track_means', means)

        # -- 1 bp branch (central crop) -------------------------------------
        self.splice_enabled = self.splice_crop > 0
        if self.splice_enabled:
            if self.splice_crop % self.bp_per_bin:
                raise ValueError(f'splice_crop {self.splice_crop} must divide by '
                                 f'{self.bp_per_bin} bp per latent bin')
            self.crop_bins = self.splice_crop // self.bp_per_bin
            self.motif_encoder = SequenceMotifEncoder(
                in_channels=5, hidden=motif_channels)
            self.latent_upsampler = LatentUpsampler(
                latent_dim, out_channels=upsample_channels,
                num_upsample=int(math.log2(self.bp_per_bin)))
            self.splice_heads = SpliceHeads(
                in_channels=motif_channels + upsample_channels,
                num_usage_tracks=num_splice_usage_tracks)
            self.junction = (JunctionHead(self.splice_heads.hidden,
                                          num_tissues=num_junction_tissues)
                             if junction_head else None)
        else:
            self.junction = None

        self.gene_head = (GeneExpressionHead(latent_dim)
                          if gene_expression_head else None)

    # ------------------------------------------------------------------
    def apply_input_mask(self, tracks):
        """Zero the masked input channels of ``tracks`` ``(B, L, K)``.

        Applied inside ``forward`` rather than by the caller so that every entry
        point -- training, both evaluation scripts, the visualisation callback --
        gets the same treatment.  A mask applied at only some call sites would
        feed the model at inference a channel it was never trained on, which is
        silent and would look like a modelling result.

        ``mask_prob >= 1`` (the default) masks always, in training and eval, and
        is the setting that removes a channel from the model's reach.  A value in
        (0, 1) masks only during training, as dropout-style regularisation: the
        channel stays available at inference and the model has merely learnt not
        to depend on it.
        """
        if tracks is None or not self.masked_track_idx:
            return tracks
        if self.mask_prob >= 1.0:
            return tracks * self.input_track_mask.to(tracks.dtype)
        if not self.training:
            return tracks                      # regularisation only
        keep = (torch.rand(tracks.shape[0], len(self.masked_track_idx),
                           device=tracks.device) >= self.mask_prob).to(tracks.dtype)
        out = tracks.clone()
        for j, i in enumerate(self.masked_track_idx):
            out[:, :, i] = out[:, :, i] * keep[:, j:j + 1]
        return out

    def crop_slice(self):
        """``(start, end)`` bp of the central crop the 1 bp heads cover."""
        c = self.window // 2
        return c - self.splice_crop // 2, c + self.splice_crop // 2

    def forward(self, seq_onehot, tracks=None, contact_map=None,
                gene_bins=None, gene_strand=None, gene_mask=None,
                donor_pos=None, acceptor_pos=None, need_splice=True):
        """
        Parameters
        ----------
        seq_onehot : ``(B, L, 5)`` ATCGN one-hot, ``L == window``.
        tracks : ``(B, L, K)`` epigenome tracks in ``ln(1+x)`` space, or None.
        contact_map : ``(B, N, N)`` predicted or experimental Hi-C, or None.
        gene_bins / gene_strand / gene_mask : gene-level head inputs.
        donor_pos / acceptor_pos : ``(B, P)`` candidate splice sites, crop-relative.

        Returns a dict; ``coverage`` is in model space (softplus), so
        ``losses_alphagenome.scale_targets`` maps targets into it and
        ``unscale_predictions`` maps predictions back out.
        """
        tracks = self.apply_input_mask(tracks)
        x = seq_onehot if tracks is None else torch.cat([seq_onehot, tracks], dim=2)
        x = x.transpose(1, 2).float()
        if self.grad_checkpoint and self.training and x.requires_grad:
            latent = checkpoint(self.encoder, x, use_reentrant=False)
        else:
            latent = self.encoder(x)                                # (B, D, N)
        if latent.shape[-1] != self.latent_bins:
            raise RuntimeError(
                f'encoder produced {latent.shape[-1]} latent bins, expected '
                f'{self.latent_bins}. The 1 bp crop and the contact-map indexing '
                f'both assume {self.bp_per_bin} bp per bin; fix the downsample '
                f'factor rather than letting the branches misalign.')
        if contact_map is not None and contact_map.shape[-1] != self.latent_bins:
            contact_map = F.interpolate(
                contact_map.unsqueeze(1), size=(self.latent_bins, self.latent_bins),
                mode='bilinear', align_corners=False).squeeze(1)
        latent = self.trunk(latent, contact_map)

        out = {'latent': latent}
        out['coverage'] = F.softplus(self.coverage_decoder(latent))  # (B, T1d, C)

        if self.splice_enabled and need_splice:
            lo_bin = (self.latent_bins - self.crop_bins) // 2
            crop_latent = latent[:, :, lo_bin:lo_bin + self.crop_bins]
            lo_bp, hi_bp = self.crop_slice()
            seq_crop = seq_onehot[:, lo_bp:hi_bp, :]
            if self.grad_checkpoint and self.training:
                up = checkpoint(self.latent_upsampler, crop_latent,
                                use_reentrant=False)
                motif = checkpoint(self.motif_encoder, seq_crop,
                                   use_reentrant=False)
            else:
                up = self.latent_upsampler(crop_latent)              # (B, U, crop)
                motif = self.motif_encoder(seq_crop)
            if up.shape[-1] != motif.shape[-1]:
                up = F.interpolate(up, size=motif.shape[-1], mode='linear',
                                   align_corners=False)
            fused = torch.cat([up, motif], dim=1)
            # The splice head is the single largest activation store in layer 4:
            # ``hidden`` channels x the full 1 bp crop, through several dilated
            # residual blocks each keeping its own input.  Checkpointing it is
            # what the earlier version missed -- OOMs land here, inside its
            # batchnorms, precisely because it was the one part left uncovered.
            if self.grad_checkpoint and self.training:
                out.update(checkpoint(self.splice_heads, fused,
                                      use_reentrant=False))
            else:
                out.update(self.splice_heads(fused))
            if self.junction is not None and donor_pos is not None:
                out['junction_counts'] = self.junction(
                    out['splice_embedding'], donor_pos, acceptor_pos)

        if self.gene_head is not None and gene_bins is not None:
            out['gene_expression'] = self.gene_head(
                latent, gene_bins, gene_strand, gene_mask)
        return out

    def unscale_coverage(self, coverage, resolution=1, apply_squashing=True):
        """Model space -> experimental scale, for inference.

        ``resolution=1`` matches training, where the targets are bin means.
        """
        from cshark.training.losses_alphagenome import unscale_predictions
        return unscale_predictions(coverage, self.coverage_track_means,
                                   resolution=resolution,
                                   apply_squashing=apply_squashing)
