"""Figures for the allele-specific transcription benchmark.

The volcanoes answer "does the model rank the same genes as the experiment".
These answer the two questions that come next, and that the volcanoes cannot:

``chain attenuation``
    How large is the allelic signal at each stage of the chain -- the perturbed
    CTCF/ATAC going into layer 3, the tracks coming out of it, the contact map,
    the RNA read-out -- against the size of the measured effect.  Every negative
    result in this benchmark so far has come down to that profile, and it was
    only ever printed as text.

``per-locus allele deltas``
    At one gene, the ``log2(maternal / paternal)`` *track* at each stage.  This is
    deliberately not two overlaid allele tracks: the predicted allelic difference
    is on the order of 1-2%, so two overlaid curves are visually identical and
    say nothing.  Plotting the ratio itself puts the allele signal on the y axis
    where it can be seen, and stacking the stages shows whether it survives the
    hand-off from chromatin to transcription.

``read-out comparison``
    Experimental effect against each available read-out (layer-3 RNA, layer-4
    coverage, layer-4 gene head) on identical windows, so "did layer 4 help" is a
    picture rather than three numbers in a JSON file.
"""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy import stats

_MAT_C, _PAT_C, _DELTA_C = '#e63946', '#264653', '#6a4c93'
_STAGE_C = '#2a9d8f'


def _downsample(arr, n):
    a = np.asarray(arr, dtype=np.float64)
    if len(a) <= n:
        return a
    b = len(a) // n
    return a[:b * n].reshape(n, b).mean(axis=1)


# ---------------------------------------------------------------------------
# 1. Chain attenuation
# ---------------------------------------------------------------------------
def plot_chain_attenuation(metrics, out_path, experimental_median=None):
    """Median |log2 allele ratio| at each stage, and each stage's skill.

    Left: the magnitude profile along the chain, with the measured effect as a
    reference line.  A chain that starts an order of magnitude below the
    measurement cannot be rescued by any downstream layer, and that is visible
    here at a glance.  Right: how well each stage's own allele ratio correlates
    with the measured gene bias, so a stage that carries magnitude but no
    information is distinguishable from one that carries neither.
    """
    stages, values = [], []
    peaks = metrics.get('perturbation_at_peaks') or {}
    for f, v in peaks.items():
        stages.append(f'{f}\n(injected,\nat peaks)')
        values.append(v['median_abs_log2ratio'])
    chan = metrics.get('channel_response') or {}
    for f, st in chan.items():
        for key, lbl in (('into_layer3', 'into L3'), ('out_of_layer3', 'out of L3')):
            if key in st:
                stages.append(f'{f}\n({lbl},\nat gene)')
                values.append(st[key]['median_abs_log2ratio'])
    cmp = metrics.get('readout_comparison') or {}
    _READOUT_LABEL = {'layer3_rna': 'layer 3\nRNA', 'layer4_rna': 'layer 4\nRNA',
                      'layer4_gene': 'layer 4\ngene'}
    for name in ('layer3_rna', 'layer4_rna', 'layer4_gene'):
        if name in cmp:
            stages.append(_READOUT_LABEL[name])
            values.append(cmp[name]['median_abs_log2ratio'])

    if not stages:
        return None
    fig, axs = plt.subplots(1, 2, figsize=(13, 4.6),
                            gridspec_kw={'width_ratios': [1.5, 1]})
    ax = axs[0]
    x = np.arange(len(stages))
    ax.bar(x, values, color=_STAGE_C, width=0.62)
    ax.plot(x, values, color=_DELTA_C, lw=1.2, marker='o', ms=4)
    if experimental_median:
        ax.axhline(experimental_median, color=_MAT_C, ls='--', lw=1.3,
                   label=f'measured effect ({experimental_median:.3f})')
        ax.legend(fontsize=8, loc='upper left')
    ax.set_xticks(x)
    ax.set_xticklabels(stages, fontsize=7)
    ax.set_ylabel('median |log2(maternal / paternal)|')
    ax.set_yscale('log')
    ax.set_title('Allelic signal along the chain\n'
                 '(log scale: the gap to the measured effect is the whole story)',
                 fontsize=10)
    ax.grid(axis='y', alpha=0.25, which='both')

    ax = axs[1]
    names, rs, aucs = [], [], []
    for name in ('layer3_rna', 'layer4_rna', 'layer4_gene'):
        if name in cmp:
            names.append(name.replace('_rna', ' RNA').replace('_gene', ' gene'))
            rs.append(cmp[name]['spearman_r'])
            aucs.append(cmp[name]['roc_auc'] - 0.5)
    if names:
        xx = np.arange(len(names))
        ax.bar(xx - 0.18, rs, 0.36, label='Spearman vs measured', color=_STAGE_C)
        ax.bar(xx + 0.18, aucs, 0.36, label='ROC-AUC - 0.5', color=_DELTA_C)
        ax.axhline(0, color='#333333', lw=0.8)
        ax.set_xticks(xx)
        ax.set_xticklabels(names, fontsize=8)
        # Symmetric about zero so the SIGN is legible: a bar below the line is
        # anti-correlation, not a small positive result.
        span = max(0.02, float(np.nanmax(np.abs(rs + aucs))) * 1.3)
        ax.set_ylim(-span, span)
        ax.set_ylabel('skill (0 = chance)')
        ax.set_title('Does any read-out carry allele information?', fontsize=10)
        ax.legend(fontsize=8)
        ax.grid(axis='y', alpha=0.25)
    else:
        ax.axis('off')
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    print(f'[hier-viz] Wrote {out_path}')
    return out_path


# ---------------------------------------------------------------------------
# 2. Read-out comparison
# ---------------------------------------------------------------------------
def plot_readout_comparison(df, out_path, cols=None):
    """Measured allele bias against each read-out, on identical genes."""
    cols = cols or [('layer3_rna_log2ratio', 'layer 3 RNA'),
                    ('layer4_rna_log2ratio', 'layer 4 coverage'),
                    ('layer4_gene_log2ratio', 'layer 4 gene head')]
    cols = [(c, l) for c, l in cols if c in df and np.isfinite(df[c]).any()]
    if not cols:
        return None
    exp = df['exp_log2ratio'].values
    fig, axs = plt.subplots(1, len(cols), figsize=(4.6 * len(cols), 4.5),
                            squeeze=False)
    for ax, (col, label) in zip(axs[0], cols):
        v = df[col].values.astype(float)
        ok = np.isfinite(v) & np.isfinite(exp)
        ax.scatter(exp[ok], v[ok], s=8, alpha=0.35, color=_STAGE_C, lw=0,
                   rasterized=True)
        ax.axhline(0, color='#888888', lw=0.7, ls='--')
        ax.axvline(0, color='#888888', lw=0.7, ls='--')
        r = rho = np.nan
        if ok.sum() > 3:
            r = stats.pearsonr(exp[ok], v[ok])[0]
            rho = stats.spearmanr(exp[ok], v[ok])[0]
        dir_acc = (np.mean(np.sign(v[ok]) == np.sign(exp[ok])) if ok.sum() else np.nan)
        # Note the axes are NOT shared: the predicted range is ~10x narrower than
        # the measured one, and forcing a common scale would collapse every
        # prediction onto the zero line and hide the structure being judged.
        ax.set_xlabel('measured log2(129 / b6)')
        ax.set_ylabel(f'predicted -- {label}')
        ax.set_title(f'{label}\nn={int(ok.sum())}  Pearson={r:.3f}  '
                     f'Spearman={rho:.3f}\ndirection acc={dir_acc:.3f}',
                     fontsize=9)
    fig.suptitle('Read-out comparison: identical windows, identical perturbation',
                 fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    print(f'[hier-viz] Wrote {out_path}')
    return out_path


# ---------------------------------------------------------------------------
# 3. Per-locus allele deltas
# ---------------------------------------------------------------------------
def plot_allele_locus(rec, out_path, n_points=1500):
    """Stacked ``log2(maternal / paternal)`` tracks at one gene.

    ``rec`` is one record captured during the evaluation pass; see
    ``capture_locus_record``.  Each panel is a *ratio*, not a pair of tracks:
    at a 1-2% allelic difference two overlaid tracks are indistinguishable, and
    the ratio is the quantity the benchmark scores.
    """
    stages = [(k, v) for k, v in rec['stages'].items() if v is not None]
    if not stages:
        return None
    n = len(stages)
    fig, axs = plt.subplots(n, 1, figsize=(11, 1.55 * n + 1.4), squeeze=False,
                            sharex=True)
    axs = axs[:, 0]
    span_lo, span_hi = rec['span']
    gene_lo, gene_hi = rec['gene_span']
    for ax, (label, track) in zip(axs, stages):
        y = _downsample(track, n_points)
        x = np.linspace(span_lo, span_hi, len(y)) / 1e6
        ax.axhline(0, color='#888888', lw=0.7)
        ax.fill_between(x, y, 0, where=y >= 0, color=_MAT_C, alpha=0.55, lw=0)
        ax.fill_between(x, y, 0, where=y < 0, color=_PAT_C, alpha=0.55, lw=0)
        ax.plot(x, y, color='#333333', lw=0.5)
        ax.axvspan(gene_lo / 1e6, gene_hi / 1e6, color='#999999', alpha=0.13, lw=0)
        # The scored interval can be 10 kb inside a 500 kb view, so the shaded
        # band alone is easy to miss; mark its centre.
        ax.axvline((gene_lo + gene_hi) / 2e6, color='#555555', lw=0.7, ls=':')
        lim = max(1e-4, float(np.nanpercentile(np.abs(y), 99.5)) * 1.25)
        ax.set_ylim(-lim, lim)
        ax.set_ylabel(f'{label}\nlog2 M/P', fontsize=7.5)
        ax.tick_params(labelsize=7)
        ax.text(0.005, 0.9, f'median |log2|={np.median(np.abs(y)):.4f}',
                transform=ax.transAxes, fontsize=7, va='top',
                bbox=dict(fc='white', ec='none', alpha=0.7, pad=1.2))
    # The measured value is a single per-gene number, so it belongs as a
    # reference line on the read-out panel rather than as a track.  It is usually
    # an order of magnitude outside the predicted range, in which case an
    # axhline lands off the axis and silently disappears -- so say so explicitly
    # with an arrow at the edge.  That gap IS the result.
    exp_v = rec['exp_log2ratio']
    lo_y, hi_y = axs[-1].get_ylim()
    if np.isfinite(exp_v) and lo_y <= exp_v <= hi_y:
        axs[-1].axhline(exp_v, color=_MAT_C, ls='--', lw=1.2,
                        label=f'measured = {exp_v:+.3f}')
        axs[-1].legend(fontsize=7, loc='lower right', framealpha=0.8)
    elif np.isfinite(exp_v):
        edge = hi_y if exp_v > hi_y else lo_y
        axs[-1].annotate(
            f'measured = {exp_v:+.3f}  ({abs(exp_v) / max(1e-9, hi_y):.0f}x '
            f'outside this axis)',
            xy=(0.5, edge), xycoords=('axes fraction', 'data'),
            xytext=(0.5, 0.80 if exp_v > hi_y else 0.20),
            textcoords='axes fraction', ha='center', fontsize=8, color=_MAT_C,
            arrowprops=dict(arrowstyle='-|>', color=_MAT_C, lw=1.2),
            bbox=dict(fc='white', ec=_MAT_C, alpha=0.9, pad=2))
    axs[-1].set_xlabel(f"{rec['chrom']} position (Mb)   "
                       f"grey band = scored interval", fontsize=9)
    fig.suptitle(
        f"{rec['gene']}  {rec['chrom']}:{gene_lo:,}-{gene_hi:,}   "
        f"measured log2={rec['exp_log2ratio']:+.3f}   "
        f"predicted log2={rec['pred_log2ratio']:+.3f}   "
        f"{rec['n_snps']} SNPs in the scored interval\n"
        f"red = maternal (129) higher, blue = paternal (b6) higher",
        fontsize=10, y=0.995)
    fig.tight_layout(rect=[0, 0, 1, 1 - 0.5 / (1.55 * n + 1.4)])
    fig.savefig(out_path, dpi=170)
    plt.close(fig)
    print(f'[hier-viz] Wrote {out_path}')
    return out_path


def log2_ratio(mat, pat, eps=1e-3, is_log1p=False):
    """``log2(mat / pat)`` for two aligned arrays, tolerating zeros.

    ``is_log1p`` inputs are converted to linear first: a ratio of ``ln(1+x)``
    values is not a fold change (it shrinks toward 1 as the level rises), which
    would make the same allelic difference look smaller at strong peaks.
    """
    a = np.asarray(mat, dtype=np.float64)
    b = np.asarray(pat, dtype=np.float64)
    if is_log1p:
        a = np.expm1(np.clip(a, 0, 60))
        b = np.expm1(np.clip(b, 0, 60))
    return np.log2((np.clip(a, 0, None) + eps) / (np.clip(b, 0, None) + eps))


def select_viz_genes(genes, n, min_abs_log2=0.0):
    """Which genes to draw: the largest MEASURED allele effects.

    Chosen from the experimental table alone, before any model runs, so the
    windows can be captured during the single evaluation pass.  Picking by
    measured effect (not predicted) is the point: these are the genes where the
    model had something real to find.
    """
    if n <= 0 or not len(genes):
        return []
    ratio = np.abs(np.log2(np.clip(genes['ratio'].values, 1e-3, None)))
    ok = np.where(ratio >= min_abs_log2)[0]
    order = ok[np.argsort(-ratio[ok])]
    return [int(genes.index[i]) for i in order[:n]]
