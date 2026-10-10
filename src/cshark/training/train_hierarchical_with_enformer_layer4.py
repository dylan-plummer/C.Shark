"""End-to-end training for the four-layer hierarchy, through to transcription.

    layer 1  Enformer          sequence            -> CTCF / ATAC / ...
    layer 2  RAD21 predictor   seq + tracks        -> RAD21            (optional)
    layer 3  Hi-C model        seq + tracks        -> Hi-C (+ 1D recon)
    layer 4  Expression model  seq + tracks + Hi-C -> RNA / CAGE, splicing, genes

Layers 1-3 are trained exactly as they are today: this module subclasses the
existing :class:`cshark.training.train_hierarchical_with_enformer.TrainModule`
rather than reimplementing it, so the Enformer tiling, the curriculum mixing and
the Hi-C objective stay bit-for-bit the same and a layer-4 run stays comparable
with the runs that came before it.  What is added is layer 4 and its losses.

Chaining
--------
The same ``mix_prob`` curriculum that feeds layer 1's predicted tracks into
layers 2/3 also decides whether layer 4 sees layer 3's *predicted* contact map or
the experimental one.  When it sees the predicted map the transcription loss
back-propagates through the contact-map head into layer 3, and through the
Enformer-predicted tracks into layer 1 -- one loss, one graph, all four layers.
Early epochs (``--pretrain-epochs``) run at ``mix_prob = 0`` so each layer first
learns on ground-truth inputs; feeding a randomly-initialised upstream layer
forward from step 0 mostly teaches the downstream layer to ignore it.

Objectives
----------
Coverage uses the AlphaGenome multinomial+Poisson objective rather than MSE, and
the splice head is a weighted 5-way cross-entropy; see
``cshark.training.losses_alphagenome`` for why each choice matters.  Per-track
means for the target scaling are estimated from the training set at startup
(``--layer4-track-mean-samples``) unless given explicitly.

Example
-------
python -m cshark.training.train_hierarchical_with_enformer_layer4 \
    --celltypes 129_b6 --assembly mm10 --data-root /path/cshark_data/data \
    --input-features ctcf atac --target-features ctcf atac rna_plus rna_minus \
    --layer4-coverage-features rna_plus rna_minus \
    --gtf /path/mm10_genes.gtf \
    --layer4-hic-mode bias --batch-size 1 --accumulate-grad-batches 8
"""
import argparse
import os
import random

import torch
import torch.nn.functional as F
import lightning as pl
from lightning.pytorch import callbacks
from lightning.pytorch.loggers.wandb import WandbLogger

from cshark.data.layer4_dataset import Layer4Dataset, build_splice_annotation
from cshark.data.splice_features import BACKGROUND
from cshark.model.layer4_models import HierarchicalExpressionModel
from cshark.training import losses_alphagenome as ag
from cshark.training.train_hierarchical_with_enformer import (
    TrainModule as HierarchicalTrainModule, ENFORMER_TARGET_LEN, finalize_args,
    safe_corr,
)

#: Batches averaged when the coverage scale has to be estimated from data.
_TRACK_MEAN_BATCHES = 8


def _pearson(a, b):
    """Pearson r over two flattened tensors, 0 when either is constant.

    Used only as the kernel of :func:`_spearman`; metric logging goes through
    ``safe_corr``, which skips undefined windows instead of zero-filling.
    """
    a, b = a.float().flatten(), b.float().flatten()
    a = a - a.mean()
    b = b - b.mean()
    denom = a.norm() * b.norm()
    if denom <= 0 or not torch.isfinite(denom):
        return torch.zeros((), device=a.device)
    return (a @ b) / denom


def _spearman(a, b, max_points=200_000):
    """Spearman rho = Pearson on ranks.

    RNA coverage spans orders of magnitude, so Pearson on raw values is
    dominated by whichever gene happens to be most expressed in the window;
    the rank correlation is what says whether the *profile* is right.
    Subsampled above ``max_points`` because argsort over a full 2 Mb window per
    validation batch is not worth the time.
    """
    a, b = a.float().flatten(), b.float().flatten()
    if a.numel() > max_points:
        idx = torch.linspace(0, a.numel() - 1, max_points,
                             device=a.device).long()
        a, b = a[idx], b[idx]
    ra = a.argsort().argsort().float()
    rb = b.argsort().argsort().float()
    return _pearson(ra, rb)


# ---------------------------------------------------------------------------
# Training module
# ---------------------------------------------------------------------------
class Layer4TrainModule(HierarchicalTrainModule):
    """Layers 1-3 exactly as before, plus layer 4 and its objectives."""

    def __init__(self, args):
        super().__init__(args)
        hp = self.hparams
        # Which of layer 3's input tracks layer 4 also consumes.  Defaults to all
        # of them: layer 4 is the read-out, so withholding an available track
        # from it only makes sense as an ablation.
        self.layer4_input_features = list(
            getattr(hp, 'layer4_input_features', None) or hp.input_features)
        self.layer4_input_idx = [hp.input_features.index(f)
                                 for f in self.layer4_input_features]
        # Masked channels are positions within LAYER 4's input list, not within
        # input_features: layers 1-3 keep every track, which is the point --
        # h3k36me3 is a legitimate epigenome target there and only becomes a
        # shortcut when the thing being predicted from it is transcription.
        mask_names = list(getattr(hp, 'layer4_mask_features', None) or [])
        unknown = [f for f in mask_names if f not in self.layer4_input_features]
        if unknown:
            raise SystemExit(
                f'--layer4-mask-features {unknown} are not among layer 4\'s input '
                f'tracks {self.layer4_input_features}. Masking a name that is not '
                f'there would silently mask nothing and the run would look like a '
                f'successful ablation.')
        self.layer4_mask_idx = [self.layer4_input_features.index(f)
                                for f in mask_names]
        if mask_names:
            how = ('always (training and inference)'
                   if hp.layer4_mask_prob >= 1.0
                   else f'with p={hp.layer4_mask_prob} during training only')
            print(f'[layer4] Masking {mask_names} from layer 4 {how}. '
                  f'Layers 1-3 still see them.')
        # Coverage targets, as columns of the stacked target tensor.
        self.coverage_features = list(hp.layer4_coverage_features)
        missing = [f for f in self.coverage_features if f not in hp.output_features]
        if missing:
            raise ValueError(
                f'--layer4-coverage-features {missing} are not in --target-features '
                f'{list(hp.output_features)}; the dataset would never load them.')
        # Indices into the dataset's full target tensor (which still carries every
        # --target-features track); unaffected by layer 3's narrower head.
        self.coverage_indices = [list(hp.output_features).index(f)
                                 for f in self.coverage_features]
        feats3, idx3 = self.layer3_1d_targets()
        if len(idx3) != len(hp.output_features):
            print(f'[layer4] layer-3 1D head sized to {feats3} '
                  f'({len(idx3)} of {len(hp.output_features)} target features); '
                  f'layer 4 owns {self.coverage_features}. Pass '
                  f'--no-layer3-exclude-layer4-targets to put transcription back '
                  f'into layer 3 as well.')
        # Which coverage track is "sense" for each gene orientation.
        self.sense_strand_index = {}
        for i, f in enumerate(self.coverage_features):
            if f.endswith('_plus') or f.endswith('_fw') or f.endswith('forward'):
                self.sense_strand_index.setdefault('+', i)
            elif f.endswith('_minus') or f.endswith('_rev') or f.endswith('reverse'):
                self.sense_strand_index.setdefault('-', i)
        self.sense_strand_index.setdefault('+', 0)
        self.sense_strand_index.setdefault('-', min(1, len(self.coverage_features) - 1))

        self.expression_model = HierarchicalExpressionModel(
            num_input_tracks=len(self.layer4_input_features),
            num_coverage_tracks=len(self.coverage_features),
            window=self.window,
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
            grad_checkpoint=hp.layer4_grad_checkpoint,
            mask_track_idx=self.layer4_mask_idx,
            mask_prob=hp.layer4_mask_prob,
        )
        self._track_means_ready = False
        self._track_mean_acc = []
        #: Last validation correlation per upstream track, driving the mix gate.
        self._upstream_corr = {}
        if getattr(hp, 'layer4_track_means', None):
            means = torch.tensor([float(x) for x in hp.layer4_track_means],
                                 dtype=torch.float32)
            if len(means) != len(self.coverage_features):
                raise ValueError('--layer4-track-means must give one value per '
                                 '--layer4-coverage-features')
            self.expression_model.coverage_track_means.copy_(means)
            self._track_means_ready = True
        self.splice_annotation = None      # built lazily in get_dataset

    def layer3_1d_targets(self):
        """Layer 3's OWN 1D targets: everything except what layer 4 predicts.

        THE HIERARCHY.  ``--target-features`` has to name every 1D track the
        *dataset* loads, because both layers read from it -- so layer 3's head was
        also being fit to rna/mnet, putting two layers on the same task and
        pulling one latent toward Hi-C, eight chromatin tracks and transcription
        at once.  Layer 3's head is now sized and supervised on its own tracks
        only.

        Computed from ``self.hparams`` rather than stored on the instance because
        the parent's ``__init__`` calls ``get_model`` (which needs this) before a
        subclass ``__init__`` body has run.
        """
        out = list(self.hparams.output_features or [])
        if not getattr(self.hparams, 'layer3_exclude_layer4_targets', True):
            return out, list(range(len(out)))
        cov = set(getattr(self.hparams, 'layer4_coverage_features', None) or [])
        idx = [i for i, f in enumerate(out) if f not in cov]
        if not idx:                     # every target belongs to layer 4
            return out, list(range(len(out)))
        return [out[i] for i in idx], idx

    # -- curriculum -----------------------------------------------------
    @staticmethod
    def _blend(ground_truth, predicted, alpha):
        """Convex blend of a ground-truth and a predicted track.

        The original hand-off REPLACED a track outright the moment ``mix_prob``
        went above zero at the end of the pretrain phase.  With
        ``--enformer-mix-mode full`` that swaps an entire 2 Mb window for a
        prediction that, at epoch 20, correlates ~0.1-0.2 with the truth: layer 3
        is handed an input unlike anything it has ever seen, and the Hi-C loss on
        those steps produces large gradients pointing away from the solution that
        was working.  That is the cliff -- good Hi-C for 20 epochs, then instantly
        poor.

        A convex blend makes the same curriculum continuous.  At alpha=0 the
        input is exactly the experimental track; alpha then ramps, so layer 3
        meets its upstream layer's errors gradually and the gradient stays
        informative instead of destructive.
        """
        if alpha <= 0:
            return ground_truth
        if alpha >= 1:
            return predicted
        return (1.0 - alpha) * ground_truth + alpha * predicted

    def current_mix_alpha(self):
        """How much of the predicted track to blend in (0-1)."""
        if self.hparams.mix_strategy == 'swap':
            return 1.0                     # old behaviour, for comparison
        return float(self.current_mix_prob())

    def _mix_gate_open(self, track):
        """Is the upstream layer good enough for this track to be fed forward?

        Feeding a prediction downstream before it carries any signal teaches the
        downstream layer to ignore that channel -- and, worse, damages weights
        that already worked on the real input.  The gate holds a track back until
        its own validation correlation clears ``--mix-gate-corr``, so the
        curriculum is driven by measured quality rather than by epoch number
        alone.  Unmeasured tracks (before the first validation) stay closed.
        """
        thresh = float(self.hparams.mix_gate_corr)
        if thresh <= 0:
            return True
        corr = self._upstream_corr.get(track)
        return corr is not None and corr >= thresh

    def on_train_epoch_start(self):
        """Snapshot the last validation correlations to drive the mix gate."""
        metrics = getattr(self.trainer, 'callback_metrics', {}) or {}
        for track in list(self.enformer_tracks) + (['rad21'] if self.use_rad21 else []):
            key = ('val_corr_1d_first_layer_rad21' if track == 'rad21'
                   else f'val_enformer_corr_1d_{track}')
            if key in metrics:
                try:
                    self._upstream_corr[track] = float(metrics[key])
                except (TypeError, ValueError):
                    pass
        alpha = self.current_mix_alpha() if self.current_mix_prob() > 0 else 0.0
        open_tracks = [t for t in self._upstream_corr if self._mix_gate_open(t)]
        self.log('mix_alpha', alpha, on_epoch=True, logger=True, sync_dist=False)
        self.log('mix_gate_open_tracks', float(len(open_tracks)), on_epoch=True,
                 logger=True, sync_dist=False)
        if self.current_mix_prob() > 0:
            print(f'[layer4] epoch {self.current_epoch}: mix_prob='
                  f'{self.current_mix_prob():.3f} alpha={alpha:.3f}; gate open for '
                  f'{open_tracks or "nothing yet"} '
                  f'(corr {({k: round(v, 3) for k, v in self._upstream_corr.items()})})')

    # -- data -----------------------------------------------------------
    def get_dataset(self, args, mode, celltype):
        """Layer-3's dataset, wrapped with the layer-4 targets."""
        base = super().get_dataset(args, mode, celltype)
        if not self._track_means_ready:
            self.set_track_means_from_data(args)
        if self.splice_annotation is None:
            self.splice_annotation = build_splice_annotation(
                args.gtf, cache_path=args.gtf_cache)
        return Layer4Dataset(
            base, self.splice_annotation,
            window=self.window,
            splice_crop=args.layer4_splice_crop,
            bp_per_bin=self.expression_model.bp_per_bin,
            target_1d_size=args.target_1d_size,
            coverage_indices=self.coverage_indices,
            sense_strand_index=self.sense_strand_index,
            max_genes=args.layer4_max_genes,
            max_splice_sites=(args.layer4_max_junction_sites
                              if args.layer4_junction_head else 0),
        )

    def proc_batch(self, batch):
        """Unpack the layer-4 dict batch into the tensors the PARENT expects.

        The return signature must match the parent's exactly -- it is
        ``(inputs, mat, target_1d_tracks, condition_vec, celltype_idx)`` -- because
        ``super().validation_step()`` unpacks it to compute every layer-1/2/3
        correlation.  Any mismatch here breaks that delegation, so the layer-4
        extras are read straight off the batch dict instead of being appended.

        ``celltype_idx`` is present only when layer 1 has celltype-split heads
        (``--enformer-split-heads-by-celltype``); Layer4Dataset copies it from the
        underlying GenomeDataset.
        """
        seq = batch['seq'].float()
        feats = batch['features'].float()
        inputs = torch.cat([seq, feats], dim=2) if feats.numel() else seq
        mat = batch['mat'].float()
        target_1d = batch['target_1d'].float()
        celltype_idx = batch.get('celltype_idx')
        if celltype_idx is not None:
            celltype_idx = torch.as_tensor(celltype_idx).reshape(-1).long().to(inputs.device)
        return inputs, mat, target_1d, None, celltype_idx

    # -- track-mean estimation ------------------------------------------
    def _track_mean_from_bigwig(self, path):
        """Genome-wide mean of a bigwig, straight from its header.

        ``sumData / nBasesCovered`` is exact and costs one open, which is far
        better than sampling: the scaling constant divides every target, so
        estimating it from a batch that happens to land on a quiet window
        inflates every subsequent loss by orders of magnitude (observed: a
        coverage loss of 19 becoming 1917 because window 0 of chr10 is nearly
        empty).
        """
        import pyBigWig
        bw = pyBigWig.open(path)
        try:
            h = bw.header()
            return float(h['sumData']) / max(1.0, float(h['nBasesCovered']))
        finally:
            bw.close()

    def validate_feature_files(self, args):
        """Fail fast when a bigwig a layer-4 target needs is not there.

        ``GenomicFeature`` treats a missing file as ``track_present=False`` and
        returns a constant -1, which the coverage loss clamps to 0.  Training
        then runs happily for days teaching layer 4 to predict no transcription
        at all -- no crash, no NaN, nothing on the dashboard that looks wrong.
        A missing *coverage* target is therefore fatal; other missing targets are
        only reported, since the parent's arms may legitimately not have them.
        """
        missing_cov, missing_other = [], []
        for celltype in args.dataset_celltypes:
            root = (f'{args.dataset_data_root}/{args.dataset_assembly}/{celltype}'
                    f'/genomic_features')
            for f in list(args.output_features or []) + list(args.input_features or []):
                if not os.path.exists(os.path.join(root, f'{f}.bw')):
                    (missing_cov if f in self.coverage_features
                     else missing_other).append(f'{celltype}/{f}.bw')
        if missing_other:
            print(f'[layer4] WARNING: missing feature bigwigs {sorted(set(missing_other))} '
                  f'-- GenomicFeature returns a constant -1 for these, so any loss '
                  f'on them is meaningless.')
        if missing_cov:
            raise SystemExit(
                f'[layer4] Missing coverage-target bigwigs: {sorted(set(missing_cov))}\n'
                f'         under {args.dataset_data_root}/{args.dataset_assembly}/'
                f'<celltype>/genomic_features/\n'
                f'         Layer 4 is trained against these; without them the target '
                f'is a constant and the model would learn to predict zero '
                f'transcription everywhere without ever erroring. Provide the '
                f'bigwigs, or point --layer4-coverage-features at tracks that exist.')

    def set_track_means_from_data(self, args):
        """Fill the coverage scaling means from the training bigwigs."""
        self.validate_feature_files(args)
        celltype = args.dataset_celltypes[0]
        root = (f'{args.dataset_data_root}/{args.dataset_assembly}/{celltype}'
                f'/genomic_features')
        means = []
        for f in self.coverage_features:
            path = os.path.join(root, f'{f}.bw')
            try:
                means.append(max(self._track_mean_from_bigwig(path), 1e-3))
            except Exception as exc:
                print(f'[layer4] could not read {path} ({exc}); using 1.0')
                means.append(1.0)
        t = torch.tensor(means, dtype=torch.float32)
        self.expression_model.coverage_track_means.copy_(
            t.to(self.expression_model.coverage_track_means.device))
        self._track_means_ready = True
        print(f'[layer4] coverage track means (genome-wide, from bigwig headers): '
              f'{dict(zip(self.coverage_features, [round(m, 4) for m in means]))}')

    def _estimate_track_means(self, target_1d):
        """Fallback when the bigwigs could not be read: accumulate over batches.

        Averaged across ``_TRACK_MEAN_BATCHES`` batches rather than taken from
        one, so a single quiet window cannot set the scale for the whole run.
        """
        with torch.no_grad():
            cov = target_1d[:, :, self.coverage_indices]
            lin = torch.expm1(torch.clamp(cov, min=0.0, max=30.0))
            self._track_mean_acc.append(lin.mean(dim=(0, 1)).cpu())
        if len(self._track_mean_acc) < _TRACK_MEAN_BATCHES:
            return                          # keep the current (1.0) scale for now
        means = torch.stack(self._track_mean_acc).mean(0).clamp(min=1e-3)
        self.expression_model.coverage_track_means.copy_(means.to(
            self.expression_model.coverage_track_means.device))
        self._track_means_ready = True
        print(f'[layer4] coverage track means (sampled over '
              f'{_TRACK_MEAN_BATCHES} batches): '
              f'{dict(zip(self.coverage_features, [round(float(m), 4) for m in means]))}')

    # -- layer 4 --------------------------------------------------------
    def layer4_step(self, inputs, contact_map, batch, target_1d, prefix='train'):
        """Run layer 4 and return ``(total_loss, {name: value})``."""
        hp = self.hparams
        seq = inputs[:, :, :5]
        tracks = inputs[:, :, [5 + i for i in self.layer4_input_idx]]

        donor = batch.get('donor_pos') if hp.layer4_junction_head else None
        acceptor = batch.get('acceptor_pos') if hp.layer4_junction_head else None
        out = self.expression_model(
            seq, tracks, contact_map,
            gene_bins=batch.get('gene_bins') if hp.layer4_gene_head else None,
            gene_strand=batch.get('gene_strand'),
            gene_mask=batch.get('gene_mask'),
            donor_pos=donor, acceptor_pos=acceptor)

        logs = {}
        # --- coverage: multinomial + Poisson in the scaled space ---------
        cov_target = target_1d[:, :, self.coverage_indices]
        # Targets are stored as ln(1+x); the objective is defined on counts.
        cov_target = torch.expm1(torch.clamp(cov_target, min=0.0, max=30.0))
        # resolution=1: our 1D targets are bin MEANS, not summed counts, so the
        # per-track mean is already the right normaliser (see scale_targets).
        scaled_target = ag.scale_targets(
            cov_target, self.expression_model.coverage_track_means,
            resolution=1, apply_squashing=hp.layer4_squash_rna)
        mask = torch.ones((cov_target.shape[0], 1, cov_target.shape[2]),
                          dtype=torch.bool, device=cov_target.device)
        cov = ag.multinomial_poisson_loss(
            scaled_target, out['coverage'], mask,
            multinomial_resolution=hp.layer4_multinomial_resolution,
            positional_weight=hp.layer4_positional_weight)
        loss = hp.loss_weight_coverage * cov['loss']
        logs['coverage'] = cov['loss'].detach()
        logs['coverage_count'] = cov['loss_count']
        logs['coverage_positional'] = cov['loss_positional']

        # --- splice site classification ----------------------------------
        if 'splice_class_logits' in out and hp.loss_weight_splice > 0:
            l_splice = ag.splice_class_loss(
                out['splice_class_logits'], batch['splice_class'],
                mask=batch.get('splice_mask'), background_index=BACKGROUND,
                positive_weight=(hp.layer4_splice_positive_weight
                                 if hp.layer4_splice_positive_weight > 0 else None))
            loss = loss + hp.loss_weight_splice * l_splice
            logs['splice'] = l_splice.detach()
            # The splice loss on its own cannot tell a working head from one that
            # has learnt "background everywhere", so score the ranking directly.
            # Validation only: PR-AUC leaves the GPU and is not worth it per step.
            if prefix == 'val':
                names = {0: 'donor_plus', 1: 'acceptor_plus',
                         2: 'donor_minus', 3: 'acceptor_minus'}
                topk = ag.splice_topk_accuracy(out['splice_class_logits'],
                                               batch['splice_class'],
                                               background_index=BACKGROUND)
                prauc = ag.splice_pr_auc(out['splice_class_logits'],
                                         batch['splice_class'],
                                         background_index=BACKGROUND)
                for c, v in topk.items():
                    logs[f'splice_topk_{names.get(c, c)}'] = torch.tensor(v)
                if topk:
                    logs['splice_topk'] = torch.tensor(
                        sum(topk.values()) / len(topk))
                for c, v in prauc.items():
                    logs[f'splice_prauc_{names.get(c, c)}'] = torch.tensor(v)
                if prauc:
                    logs['splice_prauc'] = torch.tensor(
                        sum(prauc.values()) / len(prauc))

        # --- splice usage (only where a site is annotated) ----------------
        if 'splice_usage_logits' in out and 'splice_usage' in batch:
            site_mask = (batch['splice_class'] != BACKGROUND).unsqueeze(-1)
            l_usage = ag.splice_usage_loss(
                out['splice_usage_logits'], batch['splice_usage'].float(), site_mask)
            loss = loss + hp.loss_weight_splice_usage * l_usage
            logs['splice_usage'] = l_usage.detach()

        # --- junctions ----------------------------------------------------
        if 'junction_counts' in out and 'junction_counts' in batch:
            jmask = batch.get('junction_mask')
            if jmask is None:
                jmask = torch.ones_like(batch['junction_counts'], dtype=torch.bool)
            l_junc = ag.junction_loss(out['junction_counts'],
                                      batch['junction_counts'].float(), jmask)
            loss = loss + hp.loss_weight_junction * l_junc
            logs['junction'] = l_junc.detach()

        # --- gene-level expression ---------------------------------------
        if 'gene_expression' in out:
            l_gene = ag.gene_expression_loss(
                out['gene_expression'][..., 0], batch['gene_target'].float(),
                batch['gene_mask'])
            loss = loss + hp.loss_weight_gene * l_gene
            logs['gene'] = l_gene.detach()

        for k, v in logs.items():
            is_metric = k.startswith('splice_topk') or k.startswith('splice_prauc')
            name = f'{prefix}_{k}' if is_metric else f'{prefix}_loss4_{k}'
            self.log(name, v.to(loss.device) if torch.is_tensor(v) else v,
                     on_step=(prefix == 'train' and not is_metric),
                     on_epoch=True, prog_bar=(k in ('coverage', 'splice_topk')),
                     logger=True, sync_dist=True)
        return loss, logs

    # -- full step ------------------------------------------------------
    def _hierarchy_step(self, batch, prefix):
        hp = self.hparams
        inputs, mat, target_1d, condition_vec, celltype_idx = self.proc_batch(batch)
        if not self._track_means_ready:
            self._estimate_track_means(target_1d)
        L = inputs.shape[1]
        abs_rad21 = 5 + self.rad21_idx if self.use_rad21 else None
        head_group = self.head_group_for(celltype_idx)
        mix_prob = self.current_mix_prob() if prefix == 'train' else 0.0
        total = 0.0

        # ---- layer 1: Enformer ------------------------------------------
        grad_tile_idx = random.randrange(len(self.enformer_tile_starts))
        # Only training needs the graph.  Requesting it elsewhere retains a full
        # Enformer forward's activations that no backward pass ever frees, which
        # is what made the first validation epoch OOM.
        want_grad = (prefix == 'train')
        use_enformer_mix = random.random() < mix_prob
        need_full = use_enformer_mix and hp.enformer_mix_mode == 'full'
        if need_full:
            enf_full_log1p, grad_out, grad_start = self.enformer_assemble_full(
                inputs, grad_tile_idx=grad_tile_idx, head_group=head_group)
        else:
            grad_start = self.enformer_tile_starts[grad_tile_idx]
            grad_out = self._enformer_forward_tile(inputs, grad_start,
                                                  use_grad=want_grad,
                                                  head_group=head_group)
            enf_full_log1p = None
        loss_enformer = self.enformer_window_loss(inputs, grad_out, grad_start)
        total = total + hp.training_loss_weight_enformer * loss_enformer
        self.log(f'{prefix}_loss_enformer', loss_enformer, on_epoch=True,
                 prog_bar=True, logger=True, sync_dist=True)

        if use_enformer_mix:
            inputs = inputs.clone()
            replaced = [i for i in range(len(self.enformer_input_idx))
                        if random.random() < hp.enformer_track_mix_prob]
            if not replaced:
                replaced = [random.randrange(len(self.enformer_input_idx))]
            # Only hand a track forward once layer 1 can actually predict it.
            replaced = [i for i in replaced
                        if self._mix_gate_open(self.enformer_tracks[i])]
            alpha = self.current_mix_alpha()
            if hp.enformer_mix_mode == 'window':
                end = min(grad_start + ENFORMER_TARGET_LEN, L)
                grad_up = self._tile_to_full(grad_out, L)[:, :end - grad_start, :]
                for i in replaced:
                    ch = 5 + self.enformer_input_idx[i]
                    inputs[:, grad_start:end, ch] = self._blend(
                        inputs[:, grad_start:end, ch], grad_up[:, :, i], alpha)
            else:
                for i in replaced:
                    ch = 5 + self.enformer_input_idx[i]
                    inputs[:, :, ch] = self._blend(
                        inputs[:, :, ch], enf_full_log1p[:, :, i], alpha)

        # ---- layer 2: RAD21 ---------------------------------------------
        if self.use_rad21:
            inputs_wo = torch.cat([inputs[:, :, :abs_rad21],
                                   inputs[:, :, abs_rad21 + 1:]], dim=2)
            rad21_pred = self.input_pred_model(
                inputs_wo, conditioning_vec=condition_vec).get('1d')[:, :, 0]
            gt = inputs[:, :, abs_rad21].detach()
            gt_ds = F.interpolate(gt.unsqueeze(1), size=rad21_pred.shape[1],
                                  mode='linear', align_corners=True).squeeze(1).float()
            loss_rad21 = F.mse_loss(rad21_pred, gt_ds)
            total = total + hp.training_loss_weight_rad21 * loss_rad21
            self.log(f'{prefix}_loss_rad21', loss_rad21, on_epoch=True,
                     logger=True, sync_dist=True)
            if random.random() < mix_prob and self._mix_gate_open('rad21'):
                inputs = inputs.clone()
                rad21_up = F.interpolate(
                    rad21_pred.unsqueeze(1), size=L, mode='linear',
                    align_corners=True).squeeze(1).float()
                inputs[:, :, abs_rad21] = self._blend(
                    inputs[:, :, abs_rad21], rad21_up, self.current_mix_alpha())

        # ---- layer 3: Hi-C ----------------------------------------------
        outputs = self(inputs, conditioning_vec=condition_vec)
        pred_hic = outputs.get('hic')
        if hp.predict_hic:
            loss_hic = self.criterion(pred_hic, mat)
            total = total + hp.training_loss_weight_hic * loss_hic
            self.log(f'{prefix}_loss_hic', loss_hic, on_epoch=True, prog_bar=True,
                     logger=True, sync_dist=True)
        if outputs.get('1d') is not None and hp.training_loss_weight_1d > 0:
            pred_1d = outputs['1d']
            _, idx = self.layer3_1d_targets()
            loss_1d = F.mse_loss(pred_1d, target_1d[:, :, idx])
            total = total + hp.training_loss_weight_1d * loss_1d
            self.log(f'{prefix}_loss_1d', loss_1d, on_epoch=True, logger=True,
                     sync_dist=True)

        # ---- layer 4: transcription -------------------------------------
        # Curriculum: hand layer 4 the *predicted* map with probability
        # mix_prob, otherwise the experimental one.  Only the predicted branch
        # carries gradient into layer 3's Hi-C head, which is the whole point of
        # chaining rather than training layer 4 on ground-truth contacts.
        if pred_hic is not None and random.random() < mix_prob:
            contact = pred_hic
        else:
            contact = mat if hp.predict_hic else None
        if contact is not None and hp.layer4_detach_hic:
            contact = contact.detach()
        loss4, _ = self.layer4_step(inputs, contact, batch, target_1d, prefix=prefix)
        total = total + hp.training_loss_weight_layer4 * loss4

        self.log(f'{prefix}_loss', total, on_epoch=True, prog_bar=True,
                 logger=True, sync_dist=True)
        return total

    def on_after_backward(self):
        """Log the gradient norm reaching each layer.

        This is the diagnostic a chained model actually needs: the point of
        training all four layers on one graph is that the transcription loss
        reaches layer 1, and nothing else on the dashboard tells you whether it
        does.  A layer whose norm sits at zero is detached; one whose norm dwarfs
        the others is about to destabilise everything below it (which is what
        --gradient-clip-val is for).  Watched-model histograms cannot answer this
        -- wandb files them per parameter, and per layer is the question.

        Computed every ``--log-every-n-steps`` steps: walking the Enformer's
        parameters is not free.
        """
        every = max(1, int(self.hparams.trainer_log_every_n_steps))
        if self.global_step % every:
            return
        layers = {'layer1_enformer': self.enformer,
                  'layer3_hic': self.model,
                  'layer4_expression': self.expression_model}
        if self.use_rad21 and self.input_pred_model is not None:
            layers['layer2_rad21'] = self.input_pred_model
        for name, module in layers.items():
            total, n = 0.0, 0
            for prm in module.parameters():
                if prm.grad is not None:
                    total += float(prm.grad.detach().norm(2).item()) ** 2
                    n += 1
            if n:
                self.log(f'grad_norm/{name}', total ** 0.5, on_step=True,
                         on_epoch=False, logger=True, sync_dist=False)

    def training_step(self, batch, batch_idx):
        return self._hierarchy_step(batch, 'train')

    def validation_step(self, batch, batch_idx):
        """Every layer-1/2/3 validation metric from the parent, plus layer 4.

        Delegating to ``super()`` rather than reimplementing is what keeps a
        layer-4 run comparable with the layer-3 runs: it logs the same
        ``val_enformer_corr_1d_*``, ``val_corr_1d_first_layer_rad21``,
        ``val_hic_corr``, ``val_corr_1d_*`` and ``val_hic_corr_chained`` keys, so
        the two sit on the same W&B axes.  Layer 4 is then scored on top.

        Nothing here returns a graph-attached tensor: the parent's ``val_loss``
        is detached before use, and layer 4 runs under ``no_grad``.
        """
        base_loss = super().validation_step(batch, batch_idx)
        base_loss = (base_loss.detach() if torch.is_tensor(base_loss)
                     else torch.tensor(float(base_loss or 0.0), device=self.device))

        inputs, mat, target_1d, condition_vec, celltype_idx = self.proc_batch(batch)
        head_group = self.head_group_for(celltype_idx)
        with torch.no_grad():
            # Ground-truth upstream inputs, so this is layer 4's own skill rather
            # than a measure of how bad the layers above it currently are; the
            # chained variant below is the end-to-end number.
            contact = mat if self.hparams.predict_hic else None
            loss4, _ = self.layer4_step(inputs, contact, batch, target_1d,
                                        prefix='val')
            self.layer4_val_metrics(inputs, contact, batch, target_1d)

            if self.hparams.layer4_val_chained and self.hparams.predict_hic:
                # The full pipeline: Enformer tracks -> (rad21) -> predicted
                # Hi-C -> layer 4.  This is the number the inference pipeline
                # actually delivers, and it can be far below the ground-truth
                # one; reporting only the latter would flatter the model.
                chained = self._chained_inputs(inputs, condition_vec,
                                               head_group=head_group)
                chained_hic = self(chained, conditioning_vec=condition_vec).get('hic')
                self.layer4_val_metrics(chained, chained_hic, batch, target_1d,
                                        suffix='_chained')

        total = base_loss + self.hparams.training_loss_weight_layer4 * loss4.detach()
        self.log('val_loss4', loss4.detach(), on_step=False, on_epoch=True,
                 logger=True, sync_dist=True)
        self.log('val_loss_all', total, on_step=False, on_epoch=True,
                 prog_bar=True, logger=True, sync_dist=True)
        return None

    def _chained_inputs(self, inputs, condition_vec, head_group=None):
        """Replace the Enformer-predicted tracks (and RAD21) in ``inputs``.

        Mirrors the parent's ``val_hic_corr_chained`` construction so the layer-4
        chained metrics measure the same pipeline the parent's Hi-C one does.
        ``head_group`` routes each sample through its own celltype's layer-1 head.
        """
        L = inputs.shape[1]
        enf = torch.log1p(torch.clamp(
            self.enformer_predict_1d(inputs, head_group=head_group), min=0))
        chained = inputs.clone()
        for i, in_idx in enumerate(self.enformer_input_idx):
            chained[:, :, 5 + in_idx] = enf[:, :, i]
        if self.use_rad21:
            abs_rad21 = 5 + self.rad21_idx
            no_rad21 = torch.cat([chained[:, :, :abs_rad21],
                                  chained[:, :, abs_rad21 + 1:]], dim=2)
            pred = self.input_pred_model(
                no_rad21, conditioning_vec=condition_vec).get('1d')[:, :, 0]
            chained[:, :, abs_rad21] = F.interpolate(
                pred.unsqueeze(1), size=L, mode='linear',
                align_corners=True).squeeze(1).float()
        return chained

    def layer4_val_metrics(self, inputs, contact_map, batch, target_1d, suffix=''):
        """Correlations for layer 4's heads, in EXPERIMENTAL space.

        Coverage is unscaled out of model space before correlating, so the number
        is comparable with the layer-3 ``val_corr_1d_*`` keys and with any
        external RNA benchmark.  Spearman is reported alongside Pearson because
        RNA coverage spans orders of magnitude and a single highly expressed gene
        otherwise dictates the Pearson.
        """
        seq = inputs[:, :, :5]
        tracks = inputs[:, :, [5 + i for i in self.layer4_input_idx]]
        out = self.expression_model(
            seq, tracks, contact_map,
            gene_bins=batch.get('gene_bins') if self.hparams.layer4_gene_head else None,
            gene_strand=batch.get('gene_strand'), gene_mask=batch.get('gene_mask'),
            need_splice=False)
        cov = self.expression_model.unscale_coverage(
            out['coverage'], apply_squashing=self.hparams.layer4_squash_rna)
        target = torch.expm1(torch.clamp(
            target_1d[:, :, self.coverage_indices], min=0.0, max=30.0))
        for i, feature in enumerate(self.coverage_features):
            p, t = cov[..., i].flatten(), target[..., i].flatten()
            # Skip, not zero-fill, when a window has no variance: averaging a
            # placeholder into the epoch aggregate is what makes a metric lie.
            r = safe_corr(p, t)
            if r is not None:
                self.log(f'val_corr_coverage_{feature}{suffix}', r,
                         on_step=False, on_epoch=True, prog_bar=(suffix == ''),
                         logger=True, sync_dist=True)
            rho = _spearman(p, t)
            if torch.isfinite(rho):
                self.log(f'val_spearman_coverage_{feature}{suffix}', rho,
                         on_step=False, on_epoch=True, logger=True, sync_dist=True)
        if 'gene_expression' in out and 'gene_target' in batch:
            m = batch['gene_mask'].bool()
            if m.any():
                p = out['gene_expression'][..., 0][m]
                t = batch['gene_target'].float()[m]
                # The quantity downstream analysis actually uses, so it belongs
                # on the dashboard next to the track correlations.
                r = safe_corr(p, t)
                if r is not None:
                    self.log(f'val_gene_corr{suffix}', r, on_step=False,
                             on_epoch=True, prog_bar=(suffix == ''), logger=True,
                             sync_dist=True)
                rho = _spearman(p, t)
                if torch.isfinite(rho):
                    self.log(f'val_gene_spearman{suffix}', rho, on_step=False,
                             on_epoch=True, logger=True, sync_dist=True)
        return out

    def configure_optimizers(self):
        """Per-layer learning rates: the Enformer trunk is fine-tuned gently, the
        newly initialised layer 4 is not."""
        hp = self.hparams
        groups = [
            {'params': self.model.parameters(), 'lr': hp.optimizer_lr},
            {'params': self.enformer.parameters(), 'lr': hp.enformer_lr},
            {'params': self.expression_model.parameters(), 'lr': hp.layer4_lr},
        ]
        if self.use_rad21 and self.input_pred_model is not None:
            groups.append({'params': self.input_pred_model.parameters(),
                           'lr': hp.rad21_lr})
        optimizer = torch.optim.AdamW(groups, lr=hp.optimizer_lr,
                                      weight_decay=hp.layer4_weight_decay)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=10)
        return {'optimizer': optimizer,
                'lr_scheduler': {'scheduler': scheduler, 'monitor': 'val_loss'}}


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def add_layer4_args(parser):
    g = parser.add_argument_group(
        'layer 4 (transcription)',
        'Sequence + epigenome + predicted Hi-C -> RNA/CAGE coverage, splicing and '
        'gene-level expression. See cshark.model.layer4_models.')
    g.add_argument('--layer4-coverage-features', nargs='+',
                   default=['rna_plus', 'rna_minus'],
                   help='Coverage tracks layer 4 predicts, in head order. Must be a '
                        'subset of --target-features (default: rna_plus rna_minus). '
                        'Add cage_plus/cage_minus when CAGE bigwigs exist.')
    g.add_argument('--layer4-mask-features', nargs='*', default=None,
                   metavar='TRACK',
                   help='Input tracks to hide from LAYER 4 only; layers 1-3 keep '
                        'them. Use for channels that are downstream read-outs of '
                        'what layer 4 predicts rather than causes of it -- '
                        'h3k36me3 above all, which elongating Pol II deposits '
                        'co-transcriptionally, so predicting RNA from it is close '
                        'to predicting RNA from RNA. Measured on '
                        'mESC_layer4_RNAseq_mNET_256: ablating h3k36me3 moves '
                        'predicted RNA 10.5x more than ablating ctcf, and only '
                        '13%% of layer 4\'s input sensitivity sits in the channels '
                        'a CTCF knockout actually changes.')
    g.add_argument('--layer4-mask-prob', type=float, default=1.0,
                   help='1.0 (default) masks always, in training AND inference, '
                        'so the channel is genuinely out of layer 4\'s reach. A '
                        'value in (0,1) masks only during training, as '
                        'dropout-style regularisation, leaving the channel '
                        'available at inference.')
    g.add_argument('--layer4-input-features', nargs='+', default=None,
                   help='Subset of --input-features layer 4 consumes '
                        '(default: all of them).')
    g.add_argument('--gtf', required=True,
                   help='GENCODE GTF for splice-site and gene targets.')
    g.add_argument('--gtf-cache', default=None,
                   help='Where to cache the parsed GTF (default: next to the GTF).')
    g.add_argument('--layer4-hic-mode', choices=['bias', 'propagate', 'both', 'none'],
                   default='bias',
                   help="How the contact map conditions layer 4. 'bias' (default) "
                        'adds it to the attention logits; \'propagate\' does explicit '
                        "contact-weighted message passing; 'none' is the ablation "
                        'that measures what the 3D input is worth.')
    g.add_argument('--layer4-detach-hic', dest='layer4_detach_hic',
                   action='store_true', default=True,
                   help="Stop the transcription loss reshaping layer 3's Hi-C "
                        'head. ON BY DEFAULT: Hi-C has its own supervision from '
                        'measured contact maps, and letting an RNA loss pull on '
                        'that head means a layer-4 gradient can degrade the '
                        'contact prediction to make transcription easier to fit. '
                        'Layer 4 still receives the predicted map and still '
                        'trains on it; only the reverse gradient is cut.')
    g.add_argument('--no-layer4-detach-hic', dest='layer4_detach_hic',
                   action='store_false',
                   help='Let the transcription loss back-propagate into the Hi-C '
                        'head (fully end-to-end, and the ablation for whether '
                        'that coupling helps or hurts).')
    g.add_argument('--mix-strategy', choices=['blend', 'swap'], default='blend',
                   help="How a predicted upstream track enters the next layer. "
                        "'blend' (default) uses a convex mix, alpha ramping from "
                        "0, so the curriculum is continuous. 'swap' replaces the "
                        'track outright the moment mixing starts -- the original '
                        'behaviour, and the reason Hi-C collapsed at the end of '
                        'the pretrain phase.')
    g.add_argument('--mix-gate-corr', type=float, default=0.3,
                   help='Hold a predicted track back until its own validation '
                        'correlation reaches this (default 0.3; 0 disables the '
                        'gate). Feeding a prediction forward before it carries '
                        'signal teaches the layer below to ignore the channel and '
                        'damages weights that already worked.')
    g.add_argument('--no-layer3-exclude-layer4-targets',
                   dest='layer3_exclude_layer4_targets', action='store_false',
                   help="Let layer 3's 1D head also be trained on layer 4's "
                        'coverage targets (the old behaviour, which had both '
                        'layers predicting transcription).')
    g.add_argument('--layer3-exclude-layer4-targets',
                   dest='layer3_exclude_layer4_targets', action='store_true',
                   default=True, help=argparse.SUPPRESS)
    g.add_argument('--layer4-latent-dim', type=int, default=512)
    g.add_argument('--layer4-num-blocks', type=int, default=11,
                   help='Encoder depth; the downsample factor is 2**(n+1), so 11 '
                        'gives 4,096 bp bins and 512 of them - matching a 512x512 '
                        'contact map one-to-one.')
    g.add_argument('--layer4-transformer-layers', type=int, default=8)
    g.add_argument('--layer4-propagate-rounds', type=int, default=1)
    g.add_argument('--layer4-splice-crop', type=int, default=131072,
                   help='Central bp the 1 bp splice heads cover (0 disables them). '
                        'The whole window at 1 bp does not fit in memory; the crop '
                        'is centred, so windows still tile the genome across epochs.')
    g.add_argument('--layer4-splice-positive-weight', type=float, default=0.0,
                   help='Weight of the donor/acceptor classes against background '
                        '(0 = set it automatically from the batch class balance, '
                        'the default, because ~99.99%% of bases are background and '
                        'an unweighted softmax just predicts "never a site").')
    g.add_argument('--layer4-splice-usage-tracks', type=int, default=0,
                   help='Number of splice-usage tracks (0 = head disabled). Needs '
                        'junction-derived PSI targets in the batch.')
    g.add_argument('--layer4-junction-head', action='store_true',
                   help='Enable the donor x acceptor junction head (needs junction '
                        'count targets).')
    g.add_argument('--layer4-junction-tissues', type=int, default=1)
    g.add_argument('--layer4-max-junction-sites', type=int, default=64)
    g.add_argument('--layer4-gene-head', dest='layer4_gene_head',
                   action='store_true', default=True,
                   help='Per-gene expression head (on by default).')
    g.add_argument('--no-layer4-gene-head', dest='layer4_gene_head',
                   action='store_false')
    g.add_argument('--layer4-max-genes', type=int, default=64)
    g.add_argument('--layer4-multinomial-resolution', type=int, default=128,
                   help='Segment length (in 64 bp output samples) for the '
                        'multinomial term (default 128 = 8,192 bp).')
    g.add_argument('--layer4-positional-weight', type=float, default=5.0,
                   help="Weight of the profile term against the count term "
                        "(AlphaGenome's production value is 5.0).")
    g.add_argument('--layer4-squash-rna', dest='layer4_squash_rna',
                   action='store_true', default=True,
                   help='Power-law (^0.75) compression of coverage targets, as '
                        'AlphaGenome applies to RNA-seq. On by default.')
    g.add_argument('--no-layer4-squash-rna', dest='layer4_squash_rna',
                   action='store_false')
    g.add_argument('--layer4-track-means', nargs='+', default=None,
                   help='Per-coverage-track means for target scaling (default: '
                        'estimated from the first training batch).')
    g.add_argument('--wandb-project', default='c.shark',
                   help='W&B project (default c.shark, matching the layer-1/2/3 '
                        'trainer so runs land side by side). Needs --use-wandb.')
    g.add_argument('--wandb-run-name', default=None,
                   help='W&B run name (default: let W&B generate one).')
    g.add_argument('--wandb-log-freq', type=int, default=500,
                   help='Steps between gradient/parameter histograms for layers 3 '
                        'and 4 (default 500).')
    g.add_argument('--log-every-n-steps', dest='trainer_log_every_n_steps',
                   type=int, default=50,
                   help='Trainer logging interval (default 50).')
    g.add_argument('--layer4-viz-loci', nargs='*', default=['auto'],
                   help="Loci to visualise each validation epoch, as "
                        "'chr:start' (the 2 Mb window start). 'auto' (default) "
                        'picks the validation windows with the most annotated '
                        'transcription, which are the ones where a wrong RNA '
                        'prediction is actually visible. Pass with no values to '
                        'disable the figures.')
    g.add_argument('--layer4-viz-every-n-epochs', type=int, default=1,
                   help='Draw the per-layer figures every N validation epochs '
                        '(default 1).')
    g.add_argument('--layer4-val-chained', dest='layer4_val_chained',
                   action='store_true', default=True,
                   help='Also score layer 4 on the FULLY CHAINED path (Enformer '
                        'tracks -> rad21 -> predicted Hi-C -> layer 4), logged '
                        'with a _chained suffix. On by default: that is the number '
                        'the inference pipeline delivers, and it can sit far below '
                        'the ground-truth-input one.')
    g.add_argument('--no-layer4-val-chained', dest='layer4_val_chained',
                   action='store_false',
                   help='Skip the chained layer-4 validation metrics (saves one '
                        'layer-3 and one layer-4 forward per validation batch).')
    g.add_argument('--layer4-grad-checkpoint', action='store_true',
                   help='Recompute layer-4 activations in the backward pass '
                        'instead of storing them. Costs ~30%% more time and frees '
                        'several GB -- a 2 Mb window through four layers uses '
                        '~22 GB at batch 1 without it, so this is what a batch '
                        'larger than 1 needs on a 24 GB card.')
    g.add_argument('--accumulate-grad-batches', type=int, default=1,
                   help='Gradient accumulation. A 2 Mb window through four layers '
                        'rarely fits more than batch 1-2 per GPU, so this is how '
                        'an effective batch size is reached.')
    g.add_argument('--gradient-clip-val', type=float, default=1.0,
                   help='Gradient-norm clip (default 1.0). The freshly initialised '
                        'layer-4 heads produce large gradients for the first few '
                        'hundred steps; without clipping they destabilise the '
                        'pretrained layers below them.')
    g.add_argument('--layer4-lr', type=float, default=2e-4)
    g.add_argument('--layer4-weight-decay', type=float, default=0.01)
    g.add_argument('--loss-weight-layer4', dest='training_loss_weight_layer4',
                   type=float, default=1.0)
    g.add_argument('--loss-weight-coverage', dest='loss_weight_coverage',
                   type=float, default=1.0)
    g.add_argument('--loss-weight-splice', dest='loss_weight_splice',
                   type=float, default=1.0)
    g.add_argument('--loss-weight-splice-usage', dest='loss_weight_splice_usage',
                   type=float, default=1.0)
    g.add_argument('--loss-weight-junction', dest='loss_weight_junction',
                   type=float, default=1.0)
    g.add_argument('--loss-weight-gene', dest='loss_weight_gene',
                   type=float, default=1.0)
    return parser


def build_parser():
    """The layer-1/2/3 CLI plus the layer-4 group, as one parser."""
    from cshark.training import train_hierarchical_with_enformer as base
    return base.init_parser(return_parser=True, extend=add_layer4_args)


def main():
    parser = build_parser()
    # finalize_args applies the same derived defaults (notably enformer_tracks)
    # that init_parser applies when it drives parsing itself.
    args = finalize_args(parser.parse_args())
    pl.seed_everything(args.run_seed, workers=True)
    module = Layer4TrainModule(args)

    # Select on val_loss_all (layers 1-3 + layer 4), not the parent's val_loss:
    # for a layer-4 run, choosing checkpoints on a loss that ignores layer 4
    # would optimise the wrong thing.
    trainer_callbacks = [
        callbacks.ModelCheckpoint(monitor='val_loss_all',
                                  save_top_k=args.trainer_save_top_n,
                                  mode='min', dirpath=args.run_save_path,
                                  filename='layer4-{epoch:02d}-{val_loss_all:.4f}'),
        callbacks.EarlyStopping(monitor='val_loss_all',
                                patience=args.trainer_patience, mode='min'),
        callbacks.LearningRateMonitor(logging_interval='epoch'),
    ]
    if args.layer4_viz_loci:
        from cshark.training.layer4_viz import Layer4VizCallback
        trainer_callbacks.append(Layer4VizCallback(
            loci=args.layer4_viz_loci, out_dir=os.path.join(args.run_save_path, 'viz'),
            every_n_epochs=args.layer4_viz_every_n_epochs))
    # Weights & Biases, matching the layer-1/2/3 trainer's project so a layer-4
    # run sits next to the runs it is meant to be compared with.  Every self.log
    # call in the module -- the layer-4 losses and the splice metrics included --
    # reaches wandb through this logger; without one attached they are computed
    # and thrown away.
    logger = None
    if args.use_wandb:
        logger = WandbLogger(project=args.wandb_project, name=args.wandb_run_name)
        logger.log_hyperparams(vars(args))
        # Watch layer 3 and layer 4 (not the Enformer: watching 250M frozen-ish
        # parameters floods the run with histograms for no benefit).
        logger.watch(module.model, log='gradients', log_freq=args.wandb_log_freq)
        logger.watch(module.expression_model, log='gradients',
                     log_freq=args.wandb_log_freq)

    trainer = pl.Trainer(
        max_epochs=args.trainer_max_epochs,
        accelerator='gpu' if torch.cuda.is_available() else 'cpu',
        devices=args.trainer_num_gpu,
        accumulate_grad_batches=args.accumulate_grad_batches,
        gradient_clip_val=args.gradient_clip_val,
        logger=logger if logger is not None else True,
        log_every_n_steps=args.trainer_log_every_n_steps,
        callbacks=trainer_callbacks,
    )
    trainer.fit(module,
                train_dataloaders=module.get_dataloader(args, 'train'),
                val_dataloaders=module.get_dataloader(args, 'val'))


if __name__ == '__main__':
    main()
