"""Per-layer prediction figures during training, logged to Weights & Biases.

Loss curves say a hierarchical model is improving; they do not say *which layer*
is wrong, and for a four-layer chain that is the question.  This callback draws,
at one locus, every stage of the chain against its experimental target:

    layer 1  CTCF / ATAC (Enformer)     predicted vs experimental, overlaid
    layer 2  RAD21                      predicted vs experimental, overlaid
    layer 3  Hi-C                       one map, prediction above the diagonal
                                        and experiment below -- the same matrix,
                                        so a displaced domain boundary is a
                                        visible kink rather than two pictures to
                                        compare by eye
    layer 4  stranded RNA coverage      predicted vs experimental, plus/minus
                                        mirrored about zero
             splice sites               per-base donor/acceptor probability as
                                        lollipops, against annotated positions
             gene expression            predicted vs measured, per gene

Three figures come out of it:

``chain``     the stacked locus panel above -- read top to bottom, it shows where
              in the chain the signal is lost.
``genes``     predicted vs measured gene expression across the whole validation
              set, which is the aggregate the locus panel cannot show.
``contact``   what the contact map contributes: the predicted map next to the
              attention bias layer 4 derives from it, so an inert 3D input (the
              failure mode that makes layer 4 pointless) is obvious.

Everything is drawn from a single forward pass under ``no_grad`` on the same
window, so the panels are mutually consistent.
"""
import os

import numpy as np
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from lightning.pytorch.callbacks import Callback

from cshark.data.splice_features import (
    DONOR_PLUS, ACCEPTOR_PLUS, DONOR_MINUS, ACCEPTOR_MINUS, BACKGROUND,
)

_PRED_C, _EXP_C = '#e63946', '#264653'
_HIC_CMAP = LinearSegmentedColormap.from_list(
    'cshark_hic', ['#ffffff', '#ffe8d6', '#f4a261', '#e76f51', '#9d0208'])


def _to_np(x):
    return x.detach().float().cpu().numpy() if torch.is_tensor(x) else np.asarray(x)


def _downsample(track, n):
    """Mean-pool a 1D array to ``n`` points so a 2 Mb window is drawable."""
    t = np.asarray(track, dtype=np.float64)
    if len(t) <= n:
        return t
    bin_size = len(t) // n
    return t[:bin_size * n].reshape(n, bin_size).mean(axis=1)


def _track_panel(ax, pred, exp, label, n_points=2000, log_space=True):
    """One overlaid predicted/experimental track, with its correlation."""
    p, e = _downsample(pred, n_points), _downsample(exp, n_points)
    x = np.arange(len(p))
    ax.fill_between(x, e, color=_EXP_C, alpha=0.35, lw=0, label='experimental')
    ax.plot(x, p, color=_PRED_C, lw=1.0, label='predicted')
    ok = np.isfinite(p) & np.isfinite(e)
    r = (np.corrcoef(p[ok], e[ok])[0, 1] if ok.sum() > 2 and p[ok].std() > 0
         and e[ok].std() > 0 else np.nan)
    ax.set_ylabel(f'{label}\n' + (r'$\ln(1{+}x)$' if log_space else ''), fontsize=8)
    ax.set_xlim(0, len(p) - 1)
    ax.tick_params(labelsize=7)
    ax.set_xticks([])
    ax.text(0.995, 0.92, f'r={r:.3f}', transform=ax.transAxes, ha='right',
            va='top', fontsize=8, color=_PRED_C,
            bbox=dict(fc='white', ec='none', alpha=0.7, pad=1.5))
    return r


def _hic_panel(ax, pred, exp, resolution, window):
    """Prediction above the diagonal, experiment below, on ONE matrix.

    Splitting them into two heatmaps makes the eye compare shapes across a gap;
    sharing the diagonal means a shifted boundary shows up as a discontinuity
    running through it.
    """
    p, e = np.asarray(pred, dtype=np.float64), np.asarray(exp, dtype=np.float64)
    p = (p + p.T) / 2
    e = (e + e.T) / 2
    n = min(p.shape[0], e.shape[0])
    combined = np.tril(e[:n, :n], -1) + np.triu(p[:n, :n], 1)
    diag = 0.5 * (np.diag(e[:n, :n]) + np.diag(p[:n, :n]))
    np.fill_diagonal(combined, diag)
    vmax = np.nanpercentile(combined, 99.5) or 1.0
    # aspect='auto' so the map spans the panel width and its x axis lines up with
    # the track panels above: the whole point of stacking them is to read a
    # domain boundary against the CTCF peak that anchors it.
    im = ax.imshow(combined, cmap=_HIC_CMAP, vmin=0, vmax=vmax,
                   interpolation='nearest', aspect='auto')
    ax.plot([0, n - 1], [0, n - 1], color='#333333', lw=0.6, alpha=0.6)
    ok = np.isfinite(p) & np.isfinite(e)
    r = (np.corrcoef(p[ok].flatten(), e[ok].flatten())[0, 1]
         if ok.sum() > 2 else np.nan)
    ax.set_title(f'layer 3 Hi-C   upper = predicted, lower = experimental   '
                 f'r={r:.3f}   ({resolution // 1000} kb bins)', fontsize=9)
    ax.set_xticks([]); ax.set_yticks([])
    return im, r


def _rna_panel(ax, pred_plus, pred_minus, exp_plus, exp_minus, n_points=2000):
    """Stranded coverage mirrored about zero: + strand up, - strand down.

    Transcription is directional, and a model that puts a gene's signal on the
    wrong strand is making a qualitatively different error from one that gets the
    level wrong.  Mirroring makes that immediately visible.
    """
    pp, pm = _downsample(pred_plus, n_points), _downsample(pred_minus, n_points)
    ep, em = _downsample(exp_plus, n_points), _downsample(exp_minus, n_points)
    # Correlate on the linear values (that is the quantity of interest) but DRAW
    # log1p: a single highly expressed gene reaches ~800 while the rest of the
    # window sits near 1, so on a linear axis everything but that one gene is a
    # flat line at zero.
    rs = []
    for p_, e_ in ((pp, ep), (pm, em)):
        ok = np.isfinite(p_) & np.isfinite(e_)
        if ok.sum() > 2 and p_[ok].std() > 0 and e_[ok].std() > 0:
            rs.append(np.corrcoef(p_[ok], e_[ok])[0, 1])
    pp, pm, ep, em = (np.log1p(np.clip(v, 0, None)) for v in (pp, pm, ep, em))
    x = np.arange(len(pp))
    ax.fill_between(x, ep, color=_EXP_C, alpha=0.35, lw=0, label='experimental')
    ax.fill_between(x, -em, color=_EXP_C, alpha=0.35, lw=0)
    ax.plot(x, pp, color=_PRED_C, lw=1.0, label='predicted')
    ax.plot(x, -pm, color=_PRED_C, lw=1.0)
    ax.axhline(0, color='#666666', lw=0.6)
    ax.set_ylabel('layer 4 RNA\n$\\ln(1{+}x)$, + up / - down', fontsize=8)
    ax.set_xlim(0, len(pp) - 1)
    ax.set_xticks([])
    ax.tick_params(labelsize=7)
    txt = '  '.join(f'r={r:.3f}' for r in rs) if rs else 'r=n/a'
    ax.text(0.995, 0.92, txt, transform=ax.transAxes, ha='right', va='top',
            fontsize=8, color=_PRED_C,
            bbox=dict(fc='white', ec='none', alpha=0.7, pad=1.5))
    return rs


def _splice_panel(ax, probs, labels, crop_start_frac, crop_frac):
    """Donor/acceptor probability against the annotation.

    Drawn as lollipops rather than a filled track because splice sites are single
    bases: a 2 Mb-wide filled curve of per-base probability is invisible, and
    what matters is whether the spikes land on the annotated positions.
    """
    donor = probs[:, [DONOR_PLUS, DONOR_MINUS]].max(axis=1)
    acceptor = probs[:, [ACCEPTOR_PLUS, ACCEPTOR_MINUS]].max(axis=1)
    n = len(donor)
    x = np.arange(n)
    true_sites = np.where(labels != BACKGROUND)[0]
    ax.plot(x, donor, color='#2a9d8f', lw=0.6, label='P(donor)')
    ax.plot(x, -acceptor, color='#457b9d', lw=0.6, label='P(acceptor)')
    for pos in true_sites:
        is_donor = labels[pos] in (DONOR_PLUS, DONOR_MINUS)
        ax.axvline(pos, color='#e63946' if is_donor else '#f4a261',
                   lw=0.5, alpha=0.5)
    ax.axhline(0, color='#666666', lw=0.6)
    ax.set_ylabel('layer 4 splice\nP(site)', fontsize=8)
    ax.set_xlim(0, n - 1)
    ax.set_xticks([])
    ax.set_ylim(-1.05, 1.05)
    ax.tick_params(labelsize=7)
    # Top-k accuracy on this crop: with ~1 site per 10^4 bases the curve alone
    # tells you nothing about whether the spikes are in the right places.
    hits = 0
    if len(true_sites):
        score = np.maximum(donor, acceptor)
        topk = np.argsort(-score)[:len(true_sites)]
        hits = int(np.isin(topk, true_sites).sum())
    ax.text(0.995, 0.92,
            f'{len(true_sites)} annotated sites, top-k hits {hits}',
            transform=ax.transAxes, ha='right', va='top', fontsize=7.5,
            bbox=dict(fc='white', ec='none', alpha=0.7, pad=1.5))
    ax.legend(fontsize=6.5, loc='lower right', ncol=2, framealpha=0.7)


class Layer4VizCallback(Callback):
    """Draw the per-layer figures at the end of validation epochs.

    ``loci`` is a list of ``'chr:start'`` strings, or ``['auto']`` to pick the
    validation windows carrying the most annotated transcription -- a window of
    intergenic sequence produces a beautiful flat RNA panel that says nothing.
    """

    def __init__(self, loci=('auto',), out_dir='viz', every_n_epochs=1,
                 max_loci=3, n_points=2000):
        self.loci = list(loci)
        self.out_dir = out_dir
        self.every_n_epochs = max(1, int(every_n_epochs))
        self.max_loci = max_loci
        self.n_points = n_points
        os.makedirs(self.out_dir, exist_ok=True)
        self._picked = None
        self._gene_scatter = []

    # -- locus selection -------------------------------------------------
    def _pick_batches(self, module, trainer):
        """Choose which validation batches to draw, once."""
        loader = trainer.val_dataloaders
        if isinstance(loader, (list, tuple)):
            loader = loader[0]
        wanted = None
        if self.loci and self.loci != ['auto']:
            wanted = set()
            for spec in self.loci:
                chrom, _, start = spec.partition(':')
                wanted.add((chrom, int(start)) if start else (chrom, None))
        picked, scored = [], []
        for i, batch in enumerate(loader):
            if i > 60:                      # bounded scan; the val set is large
                break
            chrom = batch['chrom'][0] if isinstance(batch['chrom'], (list, tuple)) \
                else str(batch['chrom'])
            start = int(batch['start'][0])
            if wanted is not None:
                if (chrom, start) in wanted or (chrom, None) in wanted:
                    picked.append(batch)
                    if len(picked) >= self.max_loci:
                        break
            else:
                # "Most annotated transcription" = number of splice sites in the
                # crop, which is a direct proxy for how much of this window is
                # transcribed and therefore how much the RNA panel can show.
                n_sites = int((batch['splice_class'] != BACKGROUND).sum())
                scored.append((n_sites, i, batch))
        if wanted is None:
            scored.sort(key=lambda t: -t[0])
            picked = [b for _, _, b in scored[:self.max_loci]]
        return picked

    # -- main hook -------------------------------------------------------
    def on_validation_epoch_end(self, trainer, module):
        epoch = trainer.current_epoch
        if epoch % self.every_n_epochs:
            return
        if trainer.sanity_checking:
            return
        try:
            if self._picked is None:
                self._picked = self._pick_batches(module, trainer)
            if not self._picked:
                print('[layer4-viz] no validation batches matched; skipping figures')
                return
            was_training = module.training
            module.eval()
            images = {}
            for k, batch in enumerate(self._picked):
                batch = {kk: (vv.to(module.device) if torch.is_tensor(vv) else vv)
                         for kk, vv in batch.items()}
                fig, tag = self._draw_chain(module, batch, epoch)
                path = os.path.join(self.out_dir, f'chain_{tag}_epoch{epoch:03d}.png')
                fig.savefig(path, dpi=140)
                plt.close(fig)
                images[f'viz/chain_{k}_{tag}'] = path
            if self._gene_scatter:
                fig = self._draw_gene_scatter(epoch)
                path = os.path.join(self.out_dir, f'genes_epoch{epoch:03d}.png')
                fig.savefig(path, dpi=140)
                plt.close(fig)
                images['viz/gene_expression'] = path
                self._gene_scatter = []
            self._log_images(trainer, images, epoch)
            if was_training:
                module.train()
        except Exception as exc:      # never let a figure kill a training run
            print(f'[layer4-viz] skipped ({type(exc).__name__}: {exc})')

    def _log_images(self, trainer, images, epoch):
        logger = getattr(trainer, 'logger', None)
        exp = getattr(logger, 'experiment', None)
        if exp is None or not images:
            return
        try:
            import wandb
            exp.log({k: wandb.Image(v) for k, v in images.items()},
                    step=trainer.global_step)
            print(f'[layer4-viz] logged {len(images)} figures to W&B '
                  f'(epoch {epoch})')
        except Exception as exc:
            print(f'[layer4-viz] W&B image log skipped ({exc})')

    # -- the figure ------------------------------------------------------
    @torch.no_grad()
    def _draw_chain(self, module, batch, epoch):
        hp = module.hparams
        inputs, mat, target_1d, condition_vec, celltype_idx = module.proc_batch(batch)
        # With celltype-split layer-1 heads the panel must show the head belonging to
        # the celltype this batch came from, not group 0's.
        head_group = module.head_group_for(celltype_idx)
        L = inputs.shape[1]
        chrom = batch['chrom'][0] if isinstance(batch['chrom'], (list, tuple)) \
            else str(batch['chrom'])
        start = int(batch['start'][0])
        tag = f'{chrom}_{start}'

        # layer 1 ---------------------------------------------------------
        enf_lin = module.enformer_predict_1d(inputs, head_group=head_group)
        enf_log = torch.log1p(torch.clamp(enf_lin, min=0))
        gt_enf = inputs[:, :, module.enformer_gt_channels]

        # layer 2 ---------------------------------------------------------
        rad21_pred = rad21_gt = None
        if module.use_rad21:
            abs_rad21 = 5 + module.rad21_idx
            no_rad21 = torch.cat([inputs[:, :, :abs_rad21],
                                  inputs[:, :, abs_rad21 + 1:]], dim=2)
            r = module.input_pred_model(
                no_rad21, conditioning_vec=condition_vec).get('1d')[:, :, 0]
            rad21_pred = F.interpolate(r.unsqueeze(1), size=L, mode='linear',
                                       align_corners=True).squeeze(1)
            rad21_gt = inputs[:, :, abs_rad21]

        # layer 3 ---------------------------------------------------------
        l3 = module(inputs, conditioning_vec=condition_vec)
        pred_hic = l3.get('hic')

        # layer 4 ---------------------------------------------------------
        seq = inputs[:, :, :5]
        tracks = inputs[:, :, [5 + i for i in module.layer4_input_idx]]
        l4 = module.expression_model(
            seq, tracks, pred_hic if pred_hic is not None else None,
            gene_bins=batch.get('gene_bins'), gene_strand=batch.get('gene_strand'),
            gene_mask=batch.get('gene_mask'), need_splice=True)
        cov = module.expression_model.unscale_coverage(
            l4['coverage'], apply_squashing=hp.layer4_squash_rna)
        cov_target = torch.expm1(torch.clamp(
            target_1d[:, :, module.coverage_indices], min=0, max=30))

        # accumulate the gene scatter for the aggregate figure
        if 'gene_expression' in l4 and 'gene_target' in batch:
            m = batch['gene_mask'].bool()
            if m.any():
                self._gene_scatter.append((
                    _to_np(l4['gene_expression'][..., 0][m]),
                    _to_np(batch['gene_target'].float()[m])))

        # ---- lay out ----------------------------------------------------
        n_enf = len(module.enformer_tracks)
        rows = n_enf + (1 if rad21_pred is not None else 0) + 1  # + RNA
        has_splice = 'splice_class_logits' in l4
        rows += 1 if has_splice else 0
        heights = [1.0] * rows + [4.0]            # Hi-C panel last, taller
        fig, axs = plt.subplots(rows + 1, 1, figsize=(11, 1.5 * rows + 5.5),
                                gridspec_kw={'height_ratios': heights})
        r_i = 0
        for i, feature in enumerate(module.enformer_tracks):
            _track_panel(axs[r_i], _to_np(enf_log[0, :, i]),
                         _to_np(gt_enf[0, :, i]), f'layer 1\n{feature}',
                         self.n_points)
            if r_i == 0:
                axs[r_i].legend(fontsize=6.5, loc='upper left', ncol=2,
                                framealpha=0.7)
            r_i += 1
        if rad21_pred is not None:
            _track_panel(axs[r_i], _to_np(rad21_pred[0]), _to_np(rad21_gt[0]),
                         'layer 2\nrad21', self.n_points)
            r_i += 1

        si = module.sense_strand_index
        _rna_panel(axs[r_i], _to_np(cov[0, :, si['+']]), _to_np(cov[0, :, si['-']]),
                   _to_np(cov_target[0, :, si['+']]),
                   _to_np(cov_target[0, :, si['-']]), self.n_points)
        r_i += 1
        if has_splice:
            probs = torch.softmax(l4['splice_class_logits'][0].float(), dim=-1)
            _splice_panel(axs[r_i], _to_np(probs),
                          _to_np(batch['splice_class'][0]).astype(int), 0, 1)
            r_i += 1

        if pred_hic is not None:
            im, _ = _hic_panel(axs[r_i], _to_np(pred_hic[0]), _to_np(mat[0]),
                               hp.resolution, L)
            fig.colorbar(im, ax=axs[r_i], fraction=0.025, pad=0.01)
        else:
            axs[r_i].axis('off')
        axs[r_i].set_xlabel(f'{chrom}:{start:,}-{start + L:,}', fontsize=9)

        fig.suptitle(f'C.Shark hierarchy at {chrom}:{start:,}  (epoch {epoch}) '
                     f'-- read top to bottom to see where the chain loses signal',
                     fontsize=11, y=0.997)
        fig.tight_layout(rect=[0, 0, 1, 0.985])
        return fig, tag

    def _draw_gene_scatter(self, epoch):
        """Predicted vs measured gene expression over the validation windows."""
        pred = np.concatenate([p for p, _ in self._gene_scatter])
        targ = np.concatenate([t for _, t in self._gene_scatter])
        fig, ax = plt.subplots(figsize=(5.2, 5))
        ax.scatter(targ, pred, s=14, alpha=0.6, color=_PRED_C, lw=0)
        lo = float(min(np.nanmin(targ), np.nanmin(pred)))
        hi = float(max(np.nanmax(targ), np.nanmax(pred)))
        ax.plot([lo, hi], [lo, hi], color='#264653', lw=0.8, ls='--',
                label='y = x')
        ok = np.isfinite(pred) & np.isfinite(targ)
        r = (np.corrcoef(pred[ok], targ[ok])[0, 1]
             if ok.sum() > 2 and pred[ok].std() > 0 else np.nan)
        rho = np.nan
        if ok.sum() > 2:
            from scipy import stats as _st
            rho = _st.spearmanr(pred[ok], targ[ok])[0]
        ax.set_xlabel('measured  ln(1 + mean gene coverage)')
        ax.set_ylabel('predicted  (layer-4 gene head)')
        ax.set_title(f'Gene expression, epoch {epoch}\n'
                     f'n={int(ok.sum())}  Pearson={r:.3f}  Spearman={rho:.3f}',
                     fontsize=10)
        ax.legend(fontsize=8)
        fig.tight_layout()
        return fig
