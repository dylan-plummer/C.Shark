"""Figures for the celltype-specific hierarchy evaluation.

The celltype benchmark differs from the allele one in a way that drives every
plot here: **the input sequence is identical across celltypes.**  In the
allele-specific setting the two arms differ by SNPs and layer 1 can in principle
read the difference off the sequence.  Here it cannot -- alpha, beta and delta
cells share a genome, so every celltype-specific prediction layer 1 makes comes
from its own head, and every celltype-specific prediction the full model makes
comes from the ATAC channel that head produces.

That makes the *dynamic range* of the predictions a first-class result rather
than a footnote: a model whose per-celltype heads have collapsed onto each other
predicts log2 ratio ~0 everywhere, which scores an unremarkable-looking
correlation of ~0 while actually meaning "the split did nothing".  Every figure
here therefore shows the predicted effect-size distribution next to the measured
one, not just their correlation.
"""

import os

import numpy as np
from scipy import stats

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve

#: Stable colours per celltype so a celltype reads the same in every figure.
CELLTYPE_COLOURS = ['#264653', '#e76f51', '#2a9d8f', '#e9c46a', '#8e7dbe',
                    '#f4a261', '#457b9d', '#a8dadc']
_C_BG = '#c7ccd1'
_C_EXP = '#e76f51'
_C_PRED = '#457b9d'
_C_SHARED = '#2a9d8f'


def celltype_colour(i):
    return CELLTYPE_COLOURS[i % len(CELLTYPE_COLOURS)]


def _finite_pair(x, y):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    ok = np.isfinite(x) & np.isfinite(y)
    return x[ok], y[ok]


def _robust_lim(vals, pct=99.0, pad=1.08, floor=0.5):
    v = np.abs(np.asarray(vals, dtype=float))
    v = v[np.isfinite(v)]
    if not len(v):
        return floor
    return max(floor, float(np.percentile(v, pct)) * pad)


# ---------------------------------------------------------------------------
# Layer 1: celltype-specific peaks
# ---------------------------------------------------------------------------
def plot_pair_peaks(df, pair, metrics, min_abs_log2, out_dir, prefix='layer1_atac'):
    """Five-panel figure for one celltype pair's differential ATAC.

    Panels 1-4 mirror the allele-specific peak figure (scatter, direction ROC,
    predicted bias split by measured direction, accuracy vs effect size) so the
    two benchmarks can be read side by side.  Panel 5 is specific to this
    setting: the measured and predicted effect-size distributions on the same
    axis, which is the only way to see head collapse -- a model with no
    celltype-specific signal still produces panels 1-4 that look merely weak.
    """
    a, b = pair
    exp = df['exp_log2ratio'].values
    pred = df['pred_log2ratio'].values
    labels = df['exp_a_biased'].values
    correct = df['direction_correct'].values.astype(bool)
    m_all = metrics['all']

    fig = plt.figure(figsize=(16, 10))
    gs = fig.add_gridspec(2, 3)

    # (1) Effect-size concordance -------------------------------------------
    ax = fig.add_subplot(gs[0, 0])
    ax.axhline(0, color='grey', lw=0.8, ls='--')
    ax.axvline(0, color='grey', lw=0.8, ls='--')
    ax.scatter(exp[correct], pred[correct], s=10, alpha=0.45, lw=0,
               color=_C_SHARED, label='direction correct', rasterized=True)
    ax.scatter(exp[~correct], pred[~correct], s=10, alpha=0.45, lw=0,
               color=_C_EXP, label='direction wrong', rasterized=True)
    xl, yl = _robust_lim(exp), _robust_lim(pred)
    ax.set_xlim(-xl, xl)
    ax.set_ylim(-yl, yl)
    ax.set_xlabel(f'Measured log2({a} / {b})')
    ax.set_ylabel(f'Predicted log2({a} / {b})')
    ax.set_title(f'n={m_all["n"]}   Pearson={m_all["pearson_r"]:.3f}  '
                 f'Spearman={m_all["spearman_r"]:.3f}', fontsize=10)
    ax.legend(fontsize=8, loc='upper left')

    # (2) Direction ROC ------------------------------------------------------
    ax = fig.add_subplot(gs[0, 1])
    if len(np.unique(labels)) >= 2:
        fpr, tpr, _ = roc_curve(labels, pred)
        ax.plot(fpr, tpr, color='#264653', lw=2, label=f'AUC={m_all["roc_auc"]:.3f}')
        strong = metrics.get(f'strong_|log2|>={min_abs_log2}', {})
        sub = df[df['exp_log2ratio'].abs() >= min_abs_log2]
        if len(sub) > 10 and len(np.unique(sub['exp_a_biased'])) >= 2:
            f2, t2, _ = roc_curve(sub['exp_a_biased'].values,
                                  sub['pred_log2ratio'].values)
            ax.plot(f2, t2, color='#e76f51', lw=1.6, ls='--',
                    label=f'|log2|>={min_abs_log2:g}: AUC={strong.get("roc_auc", float("nan")):.3f}'
                          f' (n={strong.get("n", 0)})')
    ax.plot([0, 1], [0, 1], color='grey', ls='--', lw=0.8)
    ax.set_xlabel('False positive rate')
    ax.set_ylabel('True positive rate')
    ax.set_title(f'Direction ROC ({a}-biased peaks)', fontsize=10)
    ax.legend(fontsize=8, loc='lower right')

    # (3) Predicted bias split by measured direction ------------------------
    ax = fig.add_subplot(gs[0, 2])
    groups = [pred[labels == 1], pred[labels == 0]]
    names = [f'measured {a}-biased', f'measured {b}-biased']
    keep = [(g, n) for g, n in zip(groups, names) if len(g)]
    if keep:
        parts = ax.violinplot([g for g, _ in keep], showmeans=True, showextrema=False)
        for pc, colour in zip(parts['bodies'], (_C_SHARED, _C_EXP)):
            pc.set_facecolor(colour)
            pc.set_alpha(0.55)
        ax.set_xticks(range(1, len(keep) + 1))
        ax.set_xticklabels([n for _, n in keep], fontsize=8)
    ax.axhline(0, color='grey', lw=0.8, ls='--')
    ax.set_ylabel(f'Predicted log2({a} / {b})')
    ax.set_title('Predicted bias by measured direction', fontsize=10)

    # (4) Direction accuracy vs measured effect size ------------------------
    ax = fig.add_subplot(gs[1, 0])
    hi = max(1.0, float(np.nanpercentile(np.abs(exp), 95))) if len(exp) else 1.0
    thresholds = np.linspace(0, hi, 25)
    accs, ns = [], []
    for t in thresholds:
        sel = np.abs(exp) >= t
        accs.append(float(correct[sel].mean()) if sel.sum() else np.nan)
        ns.append(int(sel.sum()))
    ax.plot(thresholds, accs, color='#e9c46a', lw=2)
    ax.axhline(0.5, color='grey', ls='--', lw=0.8, label='chance')
    ax.axvline(min_abs_log2, color='#e76f51', ls=':', lw=1,
               label=f'strong cut ({min_abs_log2:g})')
    ax.set_xlabel('|measured log2 ratio| >= threshold')
    ax.set_ylabel('Direction accuracy')
    ax.set_ylim(0, 1.02)
    ax.set_title('Accuracy vs effect size', fontsize=10)
    ax.legend(fontsize=8, loc='lower right')
    ax2 = ax.twinx()
    ax2.plot(thresholds, ns, color='grey', lw=0.9, alpha=0.6)
    ax2.set_ylabel('n peaks', fontsize=8, color='grey')
    ax2.tick_params(labelsize=7, colors='grey')

    # (5) THE dynamic-range panel -------------------------------------------
    ax = fig.add_subplot(gs[1, 1:])
    bins = np.linspace(-max(xl, yl), max(xl, yl), 80)
    ax.hist(exp, bins=bins, color=_C_EXP, alpha=0.55, label='measured', density=True)
    ax.hist(pred, bins=bins, color=_C_PRED, alpha=0.55, label='predicted', density=True)
    ax.axvline(0, color='grey', lw=0.8, ls='--')
    me = float(np.nanmedian(np.abs(exp))) if len(exp) else float('nan')
    mp = float(np.nanmedian(np.abs(pred))) if len(pred) else float('nan')
    ax.set_xlabel(f'log2({a} / {b})')
    ax.set_ylabel('Density')
    ax.set_title('Effect-size RANGE, not just correlation: a model whose per-celltype '
                 'heads collapsed\npredicts ~0 everywhere and still scores r~0 -- '
                 'that is a different failure from being wrong.', fontsize=9)
    ax.legend(fontsize=9, loc='upper right')
    ax.text(0.02, 0.97,
            f'median |log2| measured : {me:.3f}\n'
            f'median |log2| predicted: {mp:.3f}\n'
            f'compression           : {(mp / me if me else float("nan")):.3f}x',
            transform=ax.transAxes, va='top', ha='left', fontsize=9,
            family='monospace',
            bbox=dict(fc='white', ec='#adb5bd', alpha=0.9))

    fig.suptitle(f'Layer 1 (fine-tuned Enformer): celltype-specific ATAC   '
                 f'{a} vs {b}', fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    path = os.path.join(out_dir, f'{prefix}_{a}_vs_{b}.png')
    fig.savefig(path, dpi=180)
    plt.close(fig)
    print(f'[viz] Wrote {path}')
    return path


def plot_layer1_summary(pair_metrics, per_celltype, out_dir,
                        fname='layer1_summary.png'):
    """Cross-pair summary: differential skill per pair + absolute skill per celltype."""
    pairs = list(pair_metrics.keys())
    keys = ['pearson_r', 'spearman_r', 'roc_auc', 'direction_accuracy']
    labels = ['Pearson r', 'Spearman r', 'ROC-AUC', 'Dir. accuracy']

    fig, axs = plt.subplots(1, 3, figsize=(17, 5))

    ax = axs[0]
    x = np.arange(len(keys))
    width = 0.8 / max(1, len(pairs))
    for i, p in enumerate(pairs):
        vals = [pair_metrics[p]['all'].get(k, np.nan) for k in keys]
        ax.bar(x + i * width, vals, width, label=p, color=celltype_colour(i))
    ax.axhline(0.5, color='grey', ls='--', lw=0.8)
    ax.axhline(0.0, color='black', lw=0.8)
    ax.set_xticks(x + width * (len(pairs) - 1) / 2)
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel('Score')
    ax.set_title('Differential ATAC skill per celltype pair\n(all peaks)', fontsize=10)
    ax.legend(fontsize=8)

    ax = axs[1]
    cts = list(per_celltype.keys())
    xs = np.arange(len(cts))
    ax.bar(xs - 0.2, [per_celltype[c]['pearson_r_log'] for c in cts], 0.4,
           label='Pearson (log1p)', color='#264653')
    ax.bar(xs + 0.2, [per_celltype[c]['spearman_r'] for c in cts], 0.4,
           label='Spearman', color='#2a9d8f')
    ax.set_xticks(xs)
    ax.set_xticklabels(cts, fontsize=9, rotation=15)
    ax.set_ylim(0, 1.02)
    ax.set_ylabel('Correlation with measured ATAC')
    ax.set_title('Absolute ATAC skill per celltype\n(predicted vs measured at peaks)',
                 fontsize=10)
    ax.legend(fontsize=8)

    ax = axs[2]
    med_exp = [pair_metrics[p]['all']['median_abs_exp_log2'] for p in pairs]
    med_pred = [pair_metrics[p]['all']['median_abs_pred_log2'] for p in pairs]
    xs = np.arange(len(pairs))
    ax.bar(xs - 0.2, med_exp, 0.4, label='measured', color=_C_EXP)
    ax.bar(xs + 0.2, med_pred, 0.4, label='predicted', color=_C_PRED)
    ax.set_xticks(xs)
    ax.set_xticklabels(pairs, fontsize=8, rotation=20, ha='right')
    ax.set_ylabel('median |log2 ratio| at peaks')
    ax.set_title('How much celltype-specific signal the heads\nactually produce',
                 fontsize=10)
    ax.legend(fontsize=8)

    fig.suptitle('Layer 1 celltype-specific ATAC: summary', fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    path = os.path.join(out_dir, fname)
    fig.savefig(path, dpi=190)
    plt.close(fig)
    print(f'[viz] Wrote {path}')
    return path


def plot_peak_loci(records, celltypes, out_dir, flank, agg_name='mean',
                   fname='layer1_top_differential_loci.png'):
    """Predicted per-celltype ATAC profiles around the most differential peaks.

    One row per peak: every celltype's predicted layer-1 profile over
    ``centre +/- flank``, the peak body shaded, and a right-hand panel comparing
    the measured and predicted aggregate per celltype -- the measured bars are
    what the prediction is trying to reproduce, so they belong in the same frame.
    """
    n = len(records)
    if not n:
        return None
    fig_h = 2.6 * n + 0.6
    fig, axs = plt.subplots(n, 2, figsize=(15, fig_h), squeeze=False,
                            gridspec_kw={'width_ratios': [3.2, 1]})
    for r, (ax, ax_bar) in zip(records, axs):
        xs = r['profile_bp'] / 1e3
        for i, ct in enumerate(celltypes):
            ax.plot(xs, r['profiles'][ct], color=celltype_colour(i), lw=1.2,
                    label=ct)
        ax.axvspan(r['start'] / 1e3, r['end'] / 1e3, color='grey', alpha=0.15,
                   lw=0, label='peak')
        ax.set_xlim(xs[0], xs[-1])
        ax.set_ylabel('pred ATAC')
        ax.set_title(f"{r['chrom']}:{r['start']:,}-{r['end']:,}   "
                     f"{r['pair_label']}   measured log2={r['exp_log2ratio']:+.2f}   "
                     f"predicted log2={r['pred_log2ratio']:+.2f}   "
                     f"({'agree' if r['agree'] else 'DISAGREE'})", fontsize=9)
        ax.legend(fontsize=7, loc='upper right', framealpha=0.85, ncol=2)

        xs_b = np.arange(len(celltypes))
        exp_v = np.array([r['exp'][c] for c in celltypes], dtype=float)
        pred_v = np.array([r['pred'][c] for c in celltypes], dtype=float)
        # Separate axes: measured and predicted ATAC are not on a common scale
        # (the head fits signal, it does not fit the bigwig's library depth), so
        # a shared axis would make the comparison about scale, not about shape.
        ax_bar.bar(xs_b - 0.2, exp_v / max(exp_v.max(), 1e-9), 0.4,
                   color='#adb5bd', label='measured')
        ax_bar.bar(xs_b + 0.2, pred_v / max(pred_v.max(), 1e-9), 0.4,
                   color='#264653', label='predicted')
        ax_bar.set_xticks(xs_b)
        ax_bar.set_xticklabels([c.replace('_total', '') for c in celltypes],
                               fontsize=7, rotation=20)
        ax_bar.set_ylabel('normalised', fontsize=7)
        ax_bar.set_title(f'{agg_name} over peak\n(each scaled to its own max)',
                         fontsize=7.5)
        ax_bar.tick_params(labelsize=7)
        ax_bar.legend(fontsize=6.5)

    axs[-1, 0].set_xlabel('Position (kb)')
    fig.suptitle(f'Layer 1: most differential ATAC peaks ({n} shown)',
                 fontsize=13, y=0.997)
    fig.tight_layout(rect=[0, 0, 1, 1 - 0.42 / fig_h])
    path = os.path.join(out_dir, fname)
    fig.savefig(path, dpi=170)
    plt.close(fig)
    print(f'[viz] Wrote {path}')
    return path


# ---------------------------------------------------------------------------
# Full model: transcription
# ---------------------------------------------------------------------------
def plot_rna_performance(df, celltypes, modes, out_dir,
                         fname='rna_absolute_performance.png'):
    """Predicted vs measured per-gene sense coverage, per celltype and input mode.

    Log axes: RNA spans four orders of magnitude and a linear scatter is one blob
    at the origin.  Read this before any differential result -- a model that
    cannot predict transcription at all cannot be expected to predict which
    celltype transcribes more.
    """
    n_rows = len(modes)
    n_cols = len(celltypes)
    fig, axs = plt.subplots(n_rows, n_cols, figsize=(4.4 * n_cols, 4.3 * n_rows),
                            squeeze=False)
    for r, mode in enumerate(modes):
        for c, ct in enumerate(celltypes):
            ax = axs[r][c]
            x = df[f'exp_rna_{ct}'].values
            y = df[f'pred_rna_{mode}_{ct}'].values
            x, y = _finite_pair(x, y)
            ok = (x > 0) & (y > 0)
            ax.scatter(x[ok], y[ok], s=6, alpha=0.35, lw=0, color=celltype_colour(c),
                       rasterized=True)
            if ok.sum() >= 4:
                rho = stats.spearmanr(x[ok], y[ok])[0]
                rp = stats.pearsonr(np.log1p(x[ok]), np.log1p(y[ok]))[0]
                lo = max(min(x[ok].min(), y[ok].min()), 1e-4)
                hi = max(x[ok].max(), y[ok].max())
                ax.plot([lo, hi], [lo, hi], color='black', lw=0.7, alpha=0.4)
                ax.set_xscale('log'); ax.set_yscale('log')
                ax.set_title(f'{ct}  [{mode} ATAC in]\n'
                             f'Spearman={rho:.3f}  Pearson(log1p)={rp:.3f}  '
                             f'n={int(ok.sum())}', fontsize=9)
            ax.set_xlabel('Measured sense coverage')
            ax.set_ylabel('Predicted sense coverage')
    fig.suptitle('Full model: absolute transcription skill per celltype', fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    path = os.path.join(out_dir, fname)
    fig.savefig(path, dpi=180)
    plt.close(fig)
    print(f'[viz] Wrote {path}')
    return path


def plot_rna_chain(chain, out_dir, fname='rna_celltype_chain.png'):
    """Where celltype specificity survives -- or dies -- along the hierarchy.

    Left: the median |log2 celltype ratio| at each stage, from the ATAC that goes
    into layer 3 through to the RNA that comes out, with the measured
    transcriptional difference as a reference line.  A stage where the magnitude
    collapses is the answer to "why is the RNA result weak" -- no read-out
    downstream of a dead stage can show anything.

    Right: whether each stage carries the difference as *information* (Spearman
    against the measured RNA ratio), so a stage with magnitude but no
    information is distinguishable from one with neither.
    """
    pairs = list(chain.keys())
    if not pairs:
        return None
    stages = chain[pairs[0]]['stages']
    stage_names = [s['name'] for s in stages]

    fig, axs = plt.subplots(1, 2, figsize=(15, 5.6))

    ax = axs[0]
    for i, p in enumerate(pairs):
        vals = [s['median_abs_log2'] for s in chain[p]['stages']]
        ax.plot(range(len(vals)), vals, 'o-', color=celltype_colour(i), lw=1.8,
                ms=6, label=p)
        ax.axhline(chain[p]['measured_rna_median_abs_log2'],
                   color=celltype_colour(i), ls=':', lw=1.2, alpha=0.8)
    ax.set_yscale('log')
    ax.set_xticks(range(len(stage_names)))
    ax.set_xticklabels(stage_names, rotation=20, ha='right', fontsize=8.5)
    ax.set_ylabel('median |log2 celltype ratio|')
    ax.set_title('Effect magnitude along the chain\n'
                 '(dotted = the measured RNA difference for that pair)', fontsize=10)
    ax.legend(fontsize=8)
    ax.grid(alpha=0.25, which='both', axis='y')

    ax = axs[1]
    x = np.arange(len(stage_names))
    width = 0.8 / max(1, len(pairs))
    for i, p in enumerate(pairs):
        vals = [s['spearman_vs_measured_rna'] for s in chain[p]['stages']]
        ax.bar(x + i * width, vals, width, color=celltype_colour(i), label=p)
    ax.axhline(0, color='black', lw=0.8)
    ax.set_xticks(x + width * (len(pairs) - 1) / 2)
    ax.set_xticklabels(stage_names, rotation=20, ha='right', fontsize=8.5)
    ax.set_ylabel('Spearman vs MEASURED RNA log2 ratio')
    ax.set_title('Information about the measured difference at each stage',
                 fontsize=10)
    ax.legend(fontsize=8)
    ax.grid(alpha=0.25, axis='y')

    fig.suptitle('Celltype specificity through the hierarchy', fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    path = os.path.join(out_dir, fname)
    fig.savefig(path, dpi=190)
    plt.close(fig)
    print(f'[viz] Wrote {path}')
    return path


def _volcano_panel(ax, df, log2col, pcol, exp_idx, pred_idx, title, xlabel,
                   label_idx, gene_col='gene'):
    x = df[log2col].values.astype(float)
    p = df[pcol].values.astype(float)
    y = -np.log10(np.clip(p, 1e-300, 1.0))
    pos = {int(k): i for i, k in enumerate(df.index)}
    shared = sorted(set(map(int, exp_idx)) & set(map(int, pred_idx)))
    ax.scatter(x, y, s=6, color=_C_BG, alpha=0.5, lw=0, rasterized=True)
    for idx, colour, lbl in (
            (sorted(set(map(int, exp_idx)) - set(shared)), _C_EXP, 'top measured only'),
            (sorted(set(map(int, pred_idx)) - set(shared)), _C_PRED, 'top predicted only'),
            (shared, _C_SHARED, 'top in both')):
        if idx:
            q = [pos[i] for i in idx]
            ax.scatter(x[q], y[q], s=24, color=colour, alpha=0.9, lw=0.3,
                       edgecolor='white', label=f'{lbl} ({len(idx)})')
    for i in label_idx:
        if i in pos:
            q = pos[i]
            ax.annotate(str(df[gene_col].values[q]), (x[q], y[q]), fontsize=6.5,
                        xytext=(3, 3), textcoords='offset points')
    lim = _robust_lim(x, 99.5, floor=1.0)
    ax.set_xlim(-lim, lim)
    ax.axvline(0, color='grey', lw=0.6, ls='--')
    ax.set_xlabel(xlabel)
    ax.set_ylabel(f'-log10({pcol})')
    ax.set_title(title, fontsize=9.5)
    ax.legend(fontsize=7, loc='upper left')


def plot_rna_volcanoes(df, exp_idx, pred_idx, stats_dict, pair, mode, out_dir,
                       n_labels=12, fname=None):
    """Measured and predicted differential-expression volcanoes for one pair.

    THE TWO Y AXES ARE NOT THE SAME STATISTIC, and are labelled as such.  There
    is one bigwig per celltype here, so the measured side has no replicates and
    its p-value is an empirical null over the effect-size distribution: a
    RANKING, not calibrated significance.  The predicted side is a paired test
    over the gene's bins, which are autocorrelated and so anti-conservative.
    Because the nulls differ, the highlighted sets are chosen by EFFECT SIZE, and
    the overlap below is counted separately for up and down -- a gene the data
    calls up and the model calls down is not agreement.
    """
    a, b = pair
    shared = sorted(set(map(int, exp_idx)) & set(map(int, pred_idx)))
    label_idx = list(shared[:n_labels])
    if len(label_idx) < n_labels:
        extra = [i for i in list(exp_idx) + list(pred_idx) if i not in shared]
        label_idx += extra[:n_labels - len(label_idx)]

    fig, axs = plt.subplots(2, 2, figsize=(14, 12))
    _volcano_panel(axs[0, 0], df, 'exp_log2ratio', 'exp_pval', exp_idx, pred_idx,
                   f'MEASURED differential expression  (n={len(df)})\n'
                   f'empirical null -- a ranking, not calibrated significance',
                   f'Measured log2({a} / {b})', label_idx)
    _volcano_panel(axs[0, 1], df, 'pred_log2ratio', 'pred_pval', exp_idx, pred_idx,
                   f'PREDICTED differential expression  (n={len(df)})\n'
                   f'paired test over gene bins, {mode} ATAC input',
                   f'Predicted log2({a} / {b})', label_idx)

    ax = axs[1, 0]
    x, y = df['exp_log2ratio'].values, df['pred_log2ratio'].values
    pos = {int(k): i for i, k in enumerate(df.index)}
    ax.scatter(x, y, s=7, color=_C_BG, alpha=0.5, lw=0, rasterized=True)
    for idx, colour in ((sorted(set(map(int, exp_idx)) - set(shared)), _C_EXP),
                        (sorted(set(map(int, pred_idx)) - set(shared)), _C_PRED),
                        (shared, _C_SHARED)):
        if idx:
            q = [pos[i] for i in idx]
            ax.scatter(x[q], y[q], s=26, color=colour, alpha=0.9, lw=0.3,
                       edgecolor='white')
    xf, yf = _finite_pair(x, y)
    r = stats.pearsonr(xf, yf)[0] if len(xf) >= 3 else float('nan')
    rho = stats.spearmanr(xf, yf)[0] if len(xf) >= 3 else float('nan')
    ax.set_xlim(-_robust_lim(x, 99.5, floor=1.0), _robust_lim(x, 99.5, floor=1.0))
    ax.set_ylim(-_robust_lim(y, 99.5, floor=1.0), _robust_lim(y, 99.5, floor=1.0))
    ax.axhline(0, color='grey', lw=0.6, ls='--')
    ax.axvline(0, color='grey', lw=0.6, ls='--')
    ax.set_xlabel(f'Measured log2({a} / {b})')
    ax.set_ylabel(f'Predicted log2({a} / {b})')
    ax.set_title(f'Effect-size concordance   Pearson={r:.3f}  Spearman={rho:.3f}',
                 fontsize=10)

    ax = axs[1, 1]
    cats, ns_exp, ns_sh, ns_pred, exps = [], [], [], [], []
    for key in ('up', 'down', 'either'):
        st = stats_dict.get(key)
        if not st:
            continue
        cats.append(key)
        ns_exp.append(st['n_experimental'] - st['n_shared'])
        ns_sh.append(st['n_shared'])
        ns_pred.append(st['n_predicted'] - st['n_shared'])
        exps.append(st['n_shared_expected_by_chance'])
    xs = np.arange(len(cats))
    ax.bar(xs - 0.25, ns_exp, 0.25, color=_C_EXP, label='measured only')
    ax.bar(xs, ns_sh, 0.25, color=_C_SHARED, label='shared')
    ax.bar(xs + 0.25, ns_pred, 0.25, color=_C_PRED, label='predicted only')
    ax.plot(xs, exps, 'k_', ms=18, mew=2, label='shared expected by chance')
    ax.set_xticks(xs)
    ax.set_xticklabels(cats, fontsize=9)
    ax.set_ylabel('Genes')
    ax.set_title('Top differential gene overlap, by direction', fontsize=10)
    ax.legend(fontsize=7.5)
    txt = '\n'.join(
        f"{k:>6}: {stats_dict[k]['n_shared']:>3} shared, "
        f"{stats_dict[k]['fold_enrichment']:.2f}x, p={stats_dict[k]['hypergeometric_p']:.2g}"
        for k in cats)
    ax.text(0.02, 0.98, txt, transform=ax.transAxes, va='top', ha='left',
            fontsize=8, family='monospace',
            bbox=dict(fc='white', ec='#adb5bd', alpha=0.9))

    fig.suptitle(f'Celltype-specific transcription: {a} vs {b}   '
                 f'[{mode} ATAC into layer 3]', fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    path = os.path.join(out_dir, fname or f'rna_de_{a}_vs_{b}_{mode}.png')
    fig.savefig(path, dpi=180)
    plt.close(fig)
    print(f'[viz] Wrote {path}')
    return path


def plot_mode_comparison(summary, out_dir, fname='rna_input_mode_comparison.png'):
    """Experimental ATAC vs Enformer-predicted ATAC as the layer-3 input.

    The gap between the two bars is the cost of replacing measured chromatin with
    layer 1's prediction -- i.e. how much of the end-to-end result is carried by
    the model rather than handed to it.
    """
    pairs = sorted({p for (p, _m) in summary})
    modes = sorted({m for (_p, m) in summary})
    metrics = [('spearman', 'Spearman (effect size)'),
               ('direction_accuracy', 'Direction accuracy'),
               ('roc_auc', 'ROC-AUC')]
    fig, axs = plt.subplots(1, len(metrics), figsize=(5.4 * len(metrics), 5))
    for ax, (key, label) in zip(np.atleast_1d(axs), metrics):
        x = np.arange(len(pairs))
        width = 0.8 / max(1, len(modes))
        for i, m in enumerate(modes):
            vals = [summary.get((p, m), {}).get(key, np.nan) for p in pairs]
            ax.bar(x + i * width, vals, width,
                   color=(_C_EXP if m == 'experimental' else _C_PRED),
                   label=f'{m} ATAC in')
        if key != 'spearman':
            ax.axhline(0.5, color='grey', ls='--', lw=0.8, label='chance')
        ax.axhline(0, color='black', lw=0.8)
        ax.set_xticks(x + width * (len(modes) - 1) / 2)
        ax.set_xticklabels(pairs, fontsize=8, rotation=20, ha='right')
        ax.set_ylabel(label)
        ax.set_title(label, fontsize=10)
        ax.legend(fontsize=8)
    fig.suptitle('Celltype-specific transcription: measured chromatin vs '
                 'layer-1 predicted chromatin as the layer-3 input', fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    path = os.path.join(out_dir, fname)
    fig.savefig(path, dpi=190)
    plt.close(fig)
    print(f'[viz] Wrote {path}')
    return path
