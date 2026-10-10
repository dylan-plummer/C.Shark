"""Full hierarchical pipeline (layer 1 -> layer 3) for allele-specific benchmarks.

The Enformer-only RNA arm asks layer 1 to predict transcription straight from
sequence.  This module benchmarks the *pipeline the model was actually built as*:

    sequence --(Enformer, layer 1)--> CTCF / ATAC
            --(delta applied to the EXPERIMENTAL tracks)-->
      seq + perturbed CTCF/ATAC --(layer 3)--> Hi-C + RNA

Only the CTCF/ATAC channels carry the allele signal into layer 3.  Enformer is
run on the maternal and the paternal sequence, and the two predictions are used
to *modulate the experimental bigwig tracks* rather than to replace them -- which
matters, because layer 3 was trained on experimental tracks (with at most one
Enformer-predicted tile mixed in per step under ``enformer_mix_mode='window'``),
so wholesale replacement would be out of distribution while a multiplicative
delta keeps the input where the model expects it.

Two ways to turn the two Enformer predictions into two allele track sets:

``split`` (default)
    Symmetric redistribution, the same formula as ``--allele-peak-split``:
    ``track_allele = 2 * E * P_allele / (P_mat + P_pat)``.  The bulk experimental
    peak is divided between the alleles in the predicted ratio, so
    ``track_mat / track_pat == P_mat / P_pat`` exactly while their mean stays at
    the experimental value.  Nothing is invented and nothing is lost.  Note the
    absolute value cap of the original script is OFF by default here: these
    tracks peak well above it, and capping the value would clip both alleles of a
    strong peak to the same ceiling -- erasing the allele ratio exactly where the
    signal is strongest.
``delta``
    Literal ALT-vs-REF perturbation: the paternal (reference) allele keeps the
    experimental track untouched and the maternal track is multiplied by the
    Enformer fold change ``P_mat / P_pat``.  Asymmetric -- the reference side is
    the bulk track, which already contains *both* alleles' signal -- but it is
    the direct analogue of the ``enformer_seq`` perturbation mode.

Spaces (easy to get wrong, all verified against the trainers):

* Enformer heads emit **linear** signal (trained as ``MSE(log1p(head), log1p
  target)``), so ratios of head outputs are already fold changes.
* Layer-3 **input** tracks are ``ln(1+x)`` (``norm='log'`` bigwigs), so the
  perturbation is applied in linear space and converted back.
* Layer-3 **1D outputs** are ``ln(1+x)`` too (fit against the same normalised
  targets), so they must be expm1'd before an allele ratio is formed.
* Layer-3 sequence input is 5-channel **ATCGN** one-hot; the Enformer tiler
  reorders to ACGT itself.
"""
import argparse
import math

import numpy as np
import torch
import torch.nn.functional as F

import cshark.model.corigami_models as corigami_models
from cshark.inference.utils.enformer_utils import (
    ENFORMER_CONTEXT_LENGTH, ENFORMER_TARGET_LEN, ENFORMER_TRIM,
)

#: Layer-3 input window, fixed by the architecture (``TrainModule.window``).
HIER_WINDOW = 2_097_152
#: Uniform base distribution used to pad sequence at chromosome edges.
_PAD_VALUE = 0.25


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------
def layer3_output_features(hp):
    """The 1D tracks layer 3 actually predicts, which is not ``output_features``.

    ``--layer3-exclude-layer4-targets`` hands the layer-4 coverage targets
    (RNA, mNET, ...) to layer 4 alone and drops them from layer 3's 1D head,
    while ``output_features`` in the hparams still lists every target the RUN
    had.  Building layer 3 from that list gives a head with too many channels
    (12 instead of 8 on the mESC checkpoints) and indexes its output by the
    wrong column, so this is the list every caller should use.
    """
    feats = list(hp.output_features)
    if getattr(hp, 'layer3_exclude_layer4_targets', False):
        excluded = set(getattr(hp, 'layer4_coverage_features', None) or [])
        feats = [f for f in feats if f not in excluded]
    return feats


def load_layer3_model(checkpoint_path, device):
    """Rebuild the layer-3 (Hi-C + 1D) model from a hierarchical checkpoint.

    Returns ``(model, hparams)``; ``hparams.layer3_output_features`` is set to
    the tracks layer 3 really predicts (see :func:`layer3_output_features`), so
    callers index its 1D output by that rather than by ``output_features``.

    The weights live under ``model.*`` in the checkpoint; the Enformer under
    ``enformer.*`` is loaded separately by ``load_enformer_from_checkpoint`` so
    the two layers can come from different checkpoints if wanted.
    """
    ck = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    hp = argparse.Namespace(**ck['hyper_parameters'])
    if getattr(hp, 'output_features', None) is None:
        raise ValueError(f'{checkpoint_path} has no output_features; layer 3 does '
                         'not predict 1D tracks in this checkpoint.')
    state = {k[len('model.'):]: v for k, v in ck['state_dict'].items()
             if k.startswith('model.')}
    out_feats = layer3_output_features(hp)
    # The weights are the authority: an hparam combination nobody anticipated
    # would otherwise fail deep inside load_state_dict with a shape mismatch
    # instead of saying which tracks are missing.
    head = state.get('decoder_1d.conv_end.weight')
    if head is not None and head.shape[0] != len(out_feats):
        raise ValueError(
            f'{checkpoint_path}: layer 3\'s 1D head has {head.shape[0]} channels '
            f'but {len(out_feats)} track names were derived ({out_feats}).\n'
            f'  output_features = {list(hp.output_features)}\n'
            f'  layer4_coverage_features = '
            f'{list(getattr(hp, "layer4_coverage_features", []) or [])}\n'
            f'  layer3_exclude_layer4_targets = '
            f'{getattr(hp, "layer3_exclude_layer4_targets", False)}\n'
            'Cannot tell which channel is which track; refusing to guess.')
    hp.layer3_output_features = out_feats
    cond = getattr(hp, 'conditioning_vec', None)
    ModelClass = getattr(corigami_models, hp.model_type)
    model = ModelClass(
        num_genomic_features=len(hp.input_features),
        num_target_tracks=len(out_feats),
        conditioning_vec_size=len(cond[0].split(',')) if cond is not None else None,
        mid_hidden=hp.model_latent_dim,
        predict_hic=hp.predict_hic,
        diploid=getattr(hp, 'dataset_assembly2', None) is not None,
        predict_1d=True,
        target_mat_size=hp.mat_size,
        target_1d_length=hp.target_1d_size,
        recon_1d=hp.recon_1d,
        seq_filter_size=hp.seq_filter_size,
        activation_1d=None,
    )
    model.load_state_dict(state, strict=True)
    model.eval().to(device)
    dropped = [f for f in hp.output_features if f not in out_feats]
    print(f'[hier] Layer 3: inputs {list(hp.input_features)} -> outputs '
          f'{out_feats} (window {HIER_WINDOW:,} bp, '
          f'{hp.target_1d_size} 1D samples = '
          f'{HIER_WINDOW // hp.target_1d_size} bp each, '
          f'predict_hic={hp.predict_hic})')
    if dropped:
        print(f'[hier] Layer 3 does NOT predict {dropped} in this checkpoint '
              f'(layer3_exclude_layer4_targets): layer 4 owns those targets, so '
              f'any layer-3 read-out of them is skipped rather than reported.')
    return model, hp


def load_layer4_model(checkpoint_path, device):
    """Rebuild the layer-4 expression model from a hierarchical checkpoint.

    Layer 4's weights live under ``expression_model.*``, alongside layer 3's
    ``model.*`` and layer 1's ``enformer.*``, so one checkpoint carries the whole
    chain.  The coverage scaling means are a persistent buffer, which matters:
    layer 4 emits *model-space* coverage and those constants are the only way
    back to signal.

    Returns ``(model, hparams)``.
    """
    from cshark.model.layer4_models import HierarchicalExpressionModel

    ck = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    hp = argparse.Namespace(**ck['hyper_parameters'])
    state = {k[len('expression_model.'):]: v for k, v in ck['state_dict'].items()
             if k.startswith('expression_model.')}
    if not state:
        raise ValueError(
            f'{checkpoint_path} has no expression_model.* weights; it is a '
            'layer-3 checkpoint. Evaluate it with the layer-3 route instead.')
    cov_feats = list(getattr(hp, 'layer4_coverage_features',
                             ['rna_plus', 'rna_minus']))
    in_feats = list(getattr(hp, 'layer4_input_features', None)
                    or hp.input_features)
    # Rebuild the input mask from the hparams.  The buffer is not persisted (so
    # that checkpoints predating it still load), which means forgetting this
    # would hand layer 4 a channel it was never trained on -- silently, and
    # looking exactly like a modelling result.
    mask_names = list(getattr(hp, 'layer4_mask_features', None) or [])
    mask_idx = [in_feats.index(f) for f in mask_names if f in in_feats]
    missing = [f for f in mask_names if f not in in_feats]
    if missing:
        raise ValueError(
            f'{checkpoint_path}: layer4_mask_features {missing} are not among '
            f'layer 4\'s inputs {in_feats}; cannot reproduce the training-time '
            'masking.')
    model = HierarchicalExpressionModel(
        num_input_tracks=len(in_feats),
        num_coverage_tracks=len(cov_feats),
        window=HIER_WINDOW,
        mat_size=hp.mat_size,
        target_1d_size=hp.target_1d_size,
        latent_dim=hp.layer4_latent_dim,
        num_blocks=hp.layer4_num_blocks,
        seq_filter_size=hp.seq_filter_size,
        hic_mode=hp.layer4_hic_mode,
        transformer_layers=hp.layer4_transformer_layers,
        propagate_rounds=hp.layer4_propagate_rounds,
        splice_crop=hp.layer4_splice_crop,
        num_splice_usage_tracks=hp.layer4_splice_usage_tracks,
        junction_head=hp.layer4_junction_head,
        num_junction_tissues=hp.layer4_junction_tissues,
        gene_expression_head=hp.layer4_gene_head,
        mask_track_idx=mask_idx,
        mask_prob=float(getattr(hp, 'layer4_mask_prob', 1.0)),
    )
    model.load_state_dict(state, strict=True)
    model.eval().to(device)
    print(f'[layer4] inputs {in_feats} + contacts ({hp.layer4_hic_mode}) -> '
          f'coverage {cov_feats} at {HIER_WINDOW // hp.target_1d_size} bp'
          + (', gene head on' if hp.layer4_gene_head else '')
          + (f', splice crop {hp.layer4_splice_crop:,} bp'
             if hp.layer4_splice_crop else ''))
    print(f'[layer4] coverage track means '
          f'{dict(zip(cov_feats, [round(float(v), 4) for v in model.coverage_track_means]))}'
          f'; squashing={bool(getattr(hp, "layer4_squash_rna", True))}')
    if mask_names:
        always = float(getattr(hp, 'layer4_mask_prob', 1.0)) >= 1.0
        print(f'[layer4] Inputs masked from layer 4: {mask_names} '
              + ('(always -- reproduced here, as at training time)' if always else
                 f'(training-only regularisation at p='
                 f'{getattr(hp, "layer4_mask_prob", 1.0)}; NOT masked at inference)'))
    return model, hp


# ---------------------------------------------------------------------------
# Window planning
# ---------------------------------------------------------------------------
def plan_gene_windows(genes, window=HIER_WINDOW, margin_frac=0.1, verbose=True):
    """Cover every gene's scoring interval with as few layer-3 windows as possible.

    Genes are swept in genomic order; each new window is placed so the first
    uncovered gene sits just inside the left margin, and every following gene
    whose scoring interval fits in the *safe band* (the window minus a
    ``margin_frac`` margin at each end) joins the same window.  Sharing windows
    matters: a window costs ~18 Enformer forward passes per allele, and genes
    cluster, so one window often serves several.

    The margin keeps genes away from the window edges, where the model has
    one-sided context.  Returns ``[(chrom, win_start, [gene_index, ...]), ...]``.
    """
    margin = int(window * margin_frac)
    plans = []
    for chrom, grp in genes.groupby('chr', sort=False):
        grp = grp.sort_values('score_start')
        idx = list(grp.index.values)
        starts = grp['score_start'].values.astype(np.int64)
        ends = grp['score_end'].values.astype(np.int64)
        i = 0
        while i < len(idx):
            # Anchor the window so this gene starts just inside the left margin.
            win_start = int(starts[i]) - margin
            win_start = max(0, win_start)
            safe_lo, safe_hi = win_start + margin, win_start + window - margin
            members = []
            while i < len(idx) and starts[i] >= safe_lo and ends[i] <= safe_hi:
                members.append(idx[i])
                i += 1
            if not members:
                # Gene longer than the safe band: centre the window on it and
                # accept the reduced flank rather than dropping the gene.
                centre = (int(starts[i]) + int(ends[i])) // 2
                win_start = max(0, centre - window // 2)
                members = [idx[i]]
                i += 1
            plans.append((chrom, int(win_start), members))
    if verbose:
        n_genes = sum(len(m) for _, _, m in plans)
        print(f'[hier] {n_genes} genes -> {len(plans)} layer-3 windows '
              f'({n_genes / max(1, len(plans)):.1f} genes/window; '
              f'~{math.ceil(window / ENFORMER_TARGET_LEN)} Enformer tiles per '
              f'window per allele)')
    return plans


# ---------------------------------------------------------------------------
# Enformer tiling, cached on a global grid
# ---------------------------------------------------------------------------
class EnformerTileCache:
    """Enformer predictions for one chromosome/allele on a global tile grid.

    Tiles are anchored at multiples of ``ENFORMER_TARGET_LEN`` in *chromosome*
    coordinates rather than relative to each window, so overlapping layer-3
    windows reuse the same tiles instead of recomputing them -- with a 10%
    margin the windows overlap heavily, and this is what keeps the Enformer cost
    proportional to the sequence covered rather than to the number of windows.

    Predictions are kept at the native 128-bp bin resolution (896 per tile) and
    upsampled to bp only when a window is assembled, which is both the memory
    win and exactly what the trainer does (``_tile_to_full``).
    """

    def __init__(self, wrapper, seq_feature, n_tracks, device, batch_size=4,
                 use_amp=False):
        self.wrapper = wrapper
        self.seq = seq_feature
        self.n_tracks = n_tracks
        self.device = device
        self.batch_size = batch_size
        self.use_amp = use_amp
        self.chrlen = len(seq_feature)
        self.tiles = {}          # tile index -> (896, K) linear prediction
        self.n_forward = 0

    def _load_ctx(self, tile_start):
        """ACGT one-hot Enformer input window for the tile whose OUTPUT starts
        at ``tile_start`` (edge-padded with the uniform base distribution)."""
        ctx_start = tile_start - ENFORMER_TRIM
        ctx_end = ctx_start + ENFORMER_CONTEXT_LENGTH
        s, e = max(0, ctx_start), min(self.chrlen, ctx_end)
        core = self.seq.get(s, e).astype(np.float32)[:, :4][:, [0, 2, 3, 1]]
        left, right = s - ctx_start, ctx_end - e
        if left > 0 or right > 0:
            pad = np.full((1, 4), _PAD_VALUE, dtype=np.float32)
            core = np.concatenate([np.tile(pad, (left, 1)), core,
                                   np.tile(pad, (right, 1))], axis=0)
        return core

    def ensure(self, tile_indices):
        """Predict any of ``tile_indices`` not already cached."""
        todo = sorted(t for t in set(tile_indices) if t not in self.tiles)
        for i0 in range(0, len(todo), self.batch_size):
            batch = todo[i0:i0 + self.batch_size]
            x = torch.from_numpy(np.stack([
                self._load_ctx(t * ENFORMER_TARGET_LEN) for t in batch])).to(self.device)
            with torch.no_grad():
                if self.use_amp and self.device.type == 'cuda':
                    with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                        out = self.wrapper(x)
                else:
                    out = self.wrapper(x)
            out = out.float().cpu().numpy()          # (B, 896, K), LINEAR space
            for j, t in enumerate(batch):
                self.tiles[t] = out[j].copy()
            self.n_forward += len(batch)

    def window_bp(self, win_start, window, track_idx):
        """Assemble a ``(window, len(track_idx))`` bp-resolution LINEAR prediction.

        Each tile is upsampled 896 -> ENFORMER_TARGET_LEN with the same linear
        interpolation the trainer uses, then the window is sliced out.
        """
        t0 = win_start // ENFORMER_TARGET_LEN
        t1 = (win_start + window - 1) // ENFORMER_TARGET_LEN
        self.ensure(range(t0, t1 + 1))
        chunks = []
        for t in range(t0, t1 + 1):
            tile = torch.from_numpy(self.tiles[t][:, track_idx]).unsqueeze(0)
            up = F.interpolate(tile.permute(0, 2, 1), size=ENFORMER_TARGET_LEN,
                               mode='linear', align_corners=True)
            chunks.append(up.permute(0, 2, 1)[0].numpy())
        full = np.concatenate(chunks, axis=0)
        off = win_start - t0 * ENFORMER_TARGET_LEN
        return full[off:off + window]

    def tiles_for(self, win_start, window):
        t0 = win_start // ENFORMER_TARGET_LEN
        t1 = (win_start + window - 1) // ENFORMER_TARGET_LEN
        return range(t0, t1 + 1)


# ---------------------------------------------------------------------------
# Turning two Enformer predictions into two allele input tracks
# ---------------------------------------------------------------------------
def allele_track_pair(exp_track_log1p, pred_mat, pred_pat, mode='split',
                      cap=10.0, value_cap=0.0, span_mask=None,
                      track_is_log1p=True):
    """Build the maternal / paternal versions of one experimental input track.

    ``exp_track_log1p`` is the bulk experimental track as layer 3 consumes it
    (``ln(1+x)`` when ``track_is_log1p``); ``pred_mat``/``pred_pat`` are the
    Enformer predictions for the two alleles in LINEAR space, aligned bp-for-bp.
    ``span_mask`` (optional boolean array) restricts the perturbation to a
    region -- outside it both alleles keep the untouched experimental track.

    ``cap`` bounds the fold change in ``delta`` mode.  ``split`` mode needs no
    such bound -- its multiplier ``2 * frac`` already lives in [0, 2] -- and
    deliberately does NOT take an absolute value cap by default: capping the
    *value* would clip both alleles of a strong peak to the same ceiling and so
    erase the allele ratio exactly where the signal is strongest (a 50-unit CTCF
    peak split 60/40 would come out 10 vs 10 under a cap of 10).  Pass
    ``value_cap > 0`` only if the tracks are on a scale where that ceiling is
    genuinely above the peaks.

    Returns ``(mat, pat)`` in the same space as the input, plus writes nothing.
    """
    exp_lin = np.expm1(exp_track_log1p) if track_is_log1p else exp_track_log1p.copy()
    exp_lin = np.clip(exp_lin, 0, None)
    tot = pred_mat + pred_pat

    if mode == 'split':
        # 50/50 wherever the model predicts nothing, so a peak with no predicted
        # allele preference survives intact instead of collapsing to zero.
        with np.errstate(invalid='ignore', divide='ignore'):
            frac_mat = np.where(tot > 0, pred_mat / np.where(tot > 0, tot, 1.0), 0.5)
        frac_mat = np.clip(np.nan_to_num(frac_mat, nan=0.5), 0.0, 1.0)
        mat_lin = 2.0 * exp_lin * frac_mat
        pat_lin = 2.0 * exp_lin * (1.0 - frac_mat)
    elif mode == 'delta':
        fc = np.divide(pred_mat, np.clip(pred_pat, 1e-6, None))
        fc = np.clip(np.nan_to_num(fc, nan=1.0, posinf=cap), 1.0 / cap, cap)
        mat_lin = exp_lin * fc
        pat_lin = exp_lin
    else:
        raise ValueError(f"Unknown allele track mode '{mode}' (split|delta).")

    if value_cap and value_cap > 0:
        mat_lin = np.minimum(mat_lin, value_cap)
        pat_lin = np.minimum(pat_lin, value_cap)
    if span_mask is not None:
        mat_lin = np.where(span_mask, mat_lin, exp_lin)
        pat_lin = np.where(span_mask, pat_lin, exp_lin)
    if track_is_log1p:
        return np.log1p(np.clip(mat_lin, 0, None)), np.log1p(np.clip(pat_lin, 0, None))
    return np.clip(mat_lin, 0, None), np.clip(pat_lin, 0, None)


# ---------------------------------------------------------------------------
# Layer-3 forward
# ---------------------------------------------------------------------------
def layer3_predict_outputs(model, seq_onehot, feature_tracks, device,
                           use_amp=False):
    """Layer 3's full output dict for one window: ``{'1d': ..., 'hic': ...}``.

    Layer 4 consumes the contact map, so unlike the layer-3-only route this
    cannot skip the Hi-C head.
    """
    feats = np.stack(feature_tracks, axis=1) if feature_tracks else None
    x = seq_onehot if feats is None else np.concatenate([seq_onehot, feats], axis=1)
    x = torch.from_numpy(np.ascontiguousarray(x, dtype=np.float32)).unsqueeze(0).to(device)
    with torch.no_grad():
        if use_amp and device.type == 'cuda':
            with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                out = model(x)
        else:
            out = model(x)
    return {k: (v.float() if torch.is_tensor(v) else v) for k, v in out.items()}


def layer4_predict(model, seq_onehot, feature_tracks, contact_map, device,
                   use_amp=False, gene_bins=None, gene_strand=None):
    """Run layer 4 on one window.

    ``contact_map`` is layer 3's predicted map (or an experimental one) as a
    torch tensor ``(1, N, N)``; ``gene_bins``/``gene_strand`` drive the gene-level
    head and may be omitted.  ``coverage`` comes back in MODEL space -- the caller
    unscales it with ``model.unscale_coverage``.  The 1 bp splice branch is
    skipped (``need_splice=False``): it costs the bulk of layer 4's compute and
    nothing in this benchmark reads it.
    """
    feats = np.stack(feature_tracks, axis=1) if feature_tracks else None
    x = seq_onehot if feats is None else np.concatenate([seq_onehot, feats], axis=1)
    seq_t = torch.from_numpy(np.ascontiguousarray(
        x[:, :5], dtype=np.float32)).unsqueeze(0).to(device)
    tracks_t = (torch.from_numpy(np.ascontiguousarray(
        x[:, 5:], dtype=np.float32)).unsqueeze(0).to(device)
        if x.shape[1] > 5 else None)
    gb = gs = gm = None
    if gene_bins is not None and model.gene_head is not None:
        gb = torch.as_tensor(gene_bins, dtype=torch.long, device=device).unsqueeze(0)
        gs = torch.as_tensor(gene_strand, dtype=torch.float32, device=device).unsqueeze(0)
        gm = torch.ones(gb.shape[:2], dtype=torch.bool, device=device)
    with torch.no_grad():
        if use_amp and device.type == 'cuda':
            with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                out = model(seq_t, tracks_t, contact_map, gene_bins=gb,
                            gene_strand=gs, gene_mask=gm, need_splice=False)
        else:
            out = model(seq_t, tracks_t, contact_map, gene_bins=gb,
                        gene_strand=gs, gene_mask=gm, need_splice=False)
    return {k: (v.float() if torch.is_tensor(v) else v) for k, v in out.items()}


def layer3_predict_1d(model, seq_onehot, feature_tracks, device, use_amp=False):
    """Run layer 3 on one window and return its 1D output, ``(target_1d, K)``.

    ``seq_onehot`` is ``(window, 5)`` ATCGN; ``feature_tracks`` is a list of
    ``(window,)`` arrays in the checkpoint's ``input_features`` order, in the
    same ``ln(1+x)`` space the trainer fed.  Output is left in the model's own
    space (log1p when ``bigwig_log_transform``) -- the caller converts.
    """
    feats = np.stack(feature_tracks, axis=1) if feature_tracks else None
    x = seq_onehot if feats is None else np.concatenate([seq_onehot, feats], axis=1)
    x = torch.from_numpy(np.ascontiguousarray(x, dtype=np.float32)).unsqueeze(0).to(device)
    with torch.no_grad():
        if use_amp and device.type == 'cuda':
            with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                out = model(x)
        else:
            out = model(x)
    pred = out.get('1d')
    if pred is None:
        raise RuntimeError('Layer 3 returned no 1D output; the checkpoint was '
                           'built with predict_1d=False.')
    return pred.float().cpu().numpy()[0]              # (target_1d_size, K)
