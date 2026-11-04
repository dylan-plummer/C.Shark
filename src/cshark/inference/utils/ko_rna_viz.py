"""Figures for the CTCF-knockout transcription benchmark.

Three questions, three figures:

``WT performance``
    Does layer 4 predict the RNA-seq it was trained on?  Bin-level (64 bp) and
    gene-level, on the held-out chromosomes.  Both axes are log because RNA
    coverage spans four orders of magnitude and a linear scatter is a single
    blob at the origin with a handful of outliers.

``KO response``
    Predicted log2(KO / WT) per gene against the measured log2(KO / WT).  The
    axes are deliberately NOT shared: as with every perturbation this model
    makes, the predicted range is far narrower than the measured one, and
    forcing a common scale would collapse the prediction onto the zero line and
    hide whatever structure is there to judge.

``chain response``
    Median |log2(KO / WT)| at every stage between the knocked-out CTCF track and
    the RNA read-out, with the measured transcriptional effect as a reference
    line.  A chain that has already attenuated to nothing by the contact map
    cannot produce a transcriptional response downstream, and that is the first
    thing to check when the KO scatter looks flat.

``per-locus``
    One gene, everything at once: the predicted contact map before and after the
    knockout, the tracks that drove the change, and the RNA read-out against the
    experiment.  This is the figure that shows *mechanism* rather than
    correlation.
"""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import gridspec
from scipy import stats

_WT_C, _KO_C, _DELTA_C = '#264653', '#e63946', '#6a4c93'
_EXP_C = '#457b9d'
LN2 = float(np.log(2.0))


def _downsample(arr, n):
    a = np.asarray(arr, dtype=np.float64)
    if len(a) <= n:
        return a
    b = len(a) // n
    return a[:b * n].reshape(n, b).mean(axis=1)


def _finite_pair(x, y):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    ok = np.isfinite(x) & np.isfinite(y)
    return x[ok], y[ok]


# ---------------------------------------------------------------------------
# 1. WT performance
# ---------------------------------------------------------------------------
def plot_wt_performance(df, bin_samples, cov_features, out_path,
                        bin_metrics=None):
    """Predicted vs experimental WT RNA: bin level, then gene level.

    ``bin_samples`` is ``{feature: (pred, exp)}`` of subsampled 64 bp bins -- the
    scatter is a subsample, but the Pearson quoted comes from ``bin_metrics``,
    which was accumulated over EVERY bin, so the figure and the JSON agree.
    ``df`` carries the per-gene means.
    """
    bin_metrics = bin_metrics or {}
    panels = [f for f in cov_features if f in bin_samples
              and len(bin_samples[f][0]) > 3]
    n = len(panels) + 2
    fig, axs = plt.subplots(1, n, figsize=(4.5 * n, 4.3), squeeze=False)
    axs = axs[0]

    for ax, feat in zip(axs, panels):
        p, e = _finite_pair(*bin_samples[feat])
        ax.scatter(e + 0.1, p + 0.1, s=3, alpha=0.15, color=_WT_C, lw=0,
                   rasterized=True)
        lim = [0.1, max(np.nanmax(e), np.nanmax(p), 1.0) * 1.3]
        ax.plot(lim, lim, color='#888888', lw=0.8, ls='--')
        ax.set_xscale('log'); ax.set_yscale('log')
        ax.set_xlim(lim); ax.set_ylim(lim)
        m = bin_metrics.get(feat, {})
        r = m.get('pearson_r', stats.pearsonr(e, p)[0] if len(e) > 3 else np.nan)
        rl = m.get('pearson_r_log1p',
                   stats.pearsonr(np.log1p(e), np.log1p(p))[0]
                   if len(e) > 3 else np.nan)
        rho = m.get('spearman_r_subsample',
                    stats.spearmanr(e, p)[0] if len(e) > 3 else np.nan)
        n_all = m.get('n_bins', len(e))
        ax.set_xlabel(f'experimental {feat} (+0.1)')
        ax.set_ylabel(f'predicted {feat} (+0.1)')
        ax.set_title(f'{feat}, 64 bp bins\nn={n_all:,}  Pearson={r:.3f}  '
                     f'Pearson(log1p)={rl:.3f}\nSpearman={rho:.3f} '
                     f'({len(e):,} bins shown)', fontsize=9)
        ax.grid(alpha=0.2, which='both')

    # Gene level: sense-strand mean coverage over the gene body.
    ax = axs[len(panels)]
    p, e = _finite_pair(df['pred_wt_cov'].values, df['exp_wt_cov'].values)
    ax.scatter(e + 0.1, p + 0.1, s=7, alpha=0.35, color=_WT_C, lw=0, rasterized=True)
    if len(e) > 3:
        lim = [0.1, max(np.nanmax(e), np.nanmax(p), 1.0) * 1.3]
        ax.plot(lim, lim, color='#888888', lw=0.8, ls='--')
        ax.set_xlim(lim); ax.set_ylim(lim)
        r = stats.pearsonr(np.log1p(e), np.log1p(p))[0]
        rho = stats.spearmanr(e, p)[0]
        ax.set_title(f'per gene, sense coverage\nn={len(e):,}  '
                     f'Pearson(log1p)={r:.3f}  Spearman={rho:.3f}', fontsize=9)
    ax.set_xscale('log'); ax.set_yscale('log')
    ax.set_xlabel('experimental mean coverage (+0.1)')
    ax.set_ylabel('predicted mean coverage (+0.1)')
    ax.grid(alpha=0.2, which='both')

    # Gene head: its own target space, log1p(mean density).
    ax = axs[len(panels) + 1]
    p, e = _finite_pair(df['pred_wt_gene'].values, df['exp_gene_target'].values)
    if len(e) > 3:
        ax.scatter(e, p, s=7, alpha=0.35, color=_DELTA_C, lw=0, rasterized=True)
        lo = min(np.nanmin(e), np.nanmin(p)); hi = max(np.nanmax(e), np.nanmax(p))
        ax.plot([lo, hi], [lo, hi], color='#888888', lw=0.8, ls='--')
        r = stats.pearsonr(e, p)[0]
        rho = stats.spearmanr(e, p)[0]
        ax.set_title(f'gene head, log1p(mean density)\nn={len(e):,}  '
                     f'Pearson={r:.3f}  Spearman={rho:.3f}', fontsize=9)
    else:
        ax.set_title('gene head: no data', fontsize=9)
    ax.set_xlabel('experimental gene target')
    ax.set_ylabel('predicted gene head')
    ax.grid(alpha=0.2)

    fig.suptitle('WT RNA-seq: predicted vs experimental', fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    fig.savefig(out_path, dpi=180)
    plt.close(fig)
    print(f'[ko-viz] Wrote {out_path}')
    return out_path


# ---------------------------------------------------------------------------
# 2. KO response
# ---------------------------------------------------------------------------
def plot_ko_response(df, metrics, out_path, cols=None):
    """Measured log2(KO / WT) against each read-out, plus the effect-size
    distributions the correlations are computed over."""
    cols = cols or [('pred_log2fc', 'layer 4 coverage'),
                    ('gene_log2fc', 'layer 4 gene head'),
                    ('layer3_log2fc', 'layer 3 RNA')]
    cols = [(c, l) for c, l in cols if c in df and np.isfinite(df[c]).any()]
    n = len(cols) + 1
    fig, axs = plt.subplots(1, n, figsize=(4.6 * n, 4.4), squeeze=False)
    axs = axs[0]
    exp = df['exp_log2fc'].values.astype(float)

    for ax, (col, label) in zip(axs, cols):
        v = df[col].values.astype(float)
        ok = np.isfinite(v) & np.isfinite(exp)
        ax.scatter(exp[ok], v[ok], s=9, alpha=0.35, color=_KO_C, lw=0,
                   rasterized=True)
        ax.axhline(0, color='#888888', lw=0.7, ls='--')
        ax.axvline(0, color='#888888', lw=0.7, ls='--')
        r = rho = dir_acc = np.nan
        if ok.sum() > 3:
            r = stats.pearsonr(exp[ok], v[ok])[0]
            rho = stats.spearmanr(exp[ok], v[ok])[0]
            dir_acc = float(np.mean(np.sign(v[ok]) == np.sign(exp[ok])))
        ax.set_xlabel('measured log2(KO / WT)')
        ax.set_ylabel(f'predicted log2(KO / WT) -- {label}')
        ax.set_title(f'{label}\nn={int(ok.sum())}  Pearson={r:.3f}  '
                     f'Spearman={rho:.3f}\ndirection acc={dir_acc:.3f}',
                     fontsize=9)
        ax.grid(alpha=0.2)

    # Effect-size distributions: the predicted response is usually an order of
    # magnitude smaller than the measured one, which no scatter shows plainly.
    ax = axs[-1]
    series = [('measured', exp, _EXP_C)]
    if 'exp_null_log2fc' in df:
        # The replicate null belongs next to the measured effect, not in a
        # footnote: if the two boxes overlap, the measurement is noise and no
        # correlation against it can be large.
        series.append(('replicate null', df['exp_null_log2fc'].values.astype(float),
                       _EXP_C))
    series += [(l, df[c].values.astype(float), _KO_C) for c, l in cols]
    parts, labels = [], []
    for lbl, v, _c in series:
        v = v[np.isfinite(v)]
        if len(v) > 3:
            parts.append(np.abs(v))
            labels.append(lbl)
    if parts:
        ax.boxplot(parts, labels=labels, showfliers=False, widths=0.6)
        ax.set_yscale('log')
        ax.set_ylabel('|log2(KO / WT)|')
        ax.set_title('Effect size: measured vs predicted\n'
                     '(log scale -- the gap is the attenuation)', fontsize=9)
        ax.tick_params(axis='x', labelsize=8, rotation=20)
        ax.grid(axis='y', alpha=0.25, which='both')

    sub = metrics.get('ko_headline', {})
    thr = sub.get('effect_threshold', 0.0)
    n_strong = sub.get('n_genes_strong', 0)
    title = (f"CTCF knockout transcriptional response  "
             f"({sub.get('n_genes', 0)} expressed genes; direction accuracy "
             f"{sub.get('direction_accuracy', float('nan')):.3f}")
    # Only claim the "responding subset" number when there is a subset: at 3 h
    # of degradation there often is not, and 'nan' in a title reads as a bug.
    title += (f"; {sub.get('direction_accuracy_strong', float('nan')):.3f} on "
              f"the {n_strong} genes with |measured log2FC| >= {thr:g})"
              if n_strong > 3 else
              f"; no gene passed |measured log2FC| >= {thr:g})")
    fig.suptitle(title, fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    fig.savefig(out_path, dpi=180)
    plt.close(fig)
    print(f'[ko-viz] Wrote {out_path}')
    return out_path


# ---------------------------------------------------------------------------
# 3. Chain response
# ---------------------------------------------------------------------------
def plot_chain_response(metrics, out_path, measured_median=None):
    """Median |log2(KO / WT)| at each stage of the chain, and each stage's skill."""
    chain = metrics.get('chain_response') or {}
    order = ['input_ctcf_at_peaks', 'input_ctcf_at_gene', 'input_rad21_at_gene',
             'pred_rad21_at_gene', 'pred_hic_at_gene', 'layer3_rna', 'layer4_rna',
             'layer4_gene']
    label = {'input_ctcf_at_peaks': 'CTCF in\n(at peaks)',
             'input_ctcf_at_gene': 'CTCF in\n(at gene)',
             'input_rad21_at_gene': 'RAD21 in\n(at gene)',
             'pred_rad21_at_gene': 'RAD21 out\n(layer 3)',
             'pred_hic_at_gene': 'contacts\n(layer 3)',
             'layer3_rna': 'RNA\n(layer 3)',
             'layer4_rna': 'RNA\n(layer 4)',
             'layer4_gene': 'gene head\n(layer 4)'}
    stages = [k for k in order if k in chain and np.isfinite(
        chain[k].get('median_abs_log2fc', np.nan))]
    if not stages:
        return None
    values = [chain[k]['median_abs_log2fc'] for k in stages]

    fig, axs = plt.subplots(1, 2, figsize=(13, 4.6),
                            gridspec_kw={'width_ratios': [1.6, 1]})
    ax = axs[0]
    x = np.arange(len(stages))
    ax.bar(x, values, color='#2a9d8f', width=0.62)
    ax.plot(x, values, color=_DELTA_C, lw=1.2, marker='o', ms=4)
    if measured_median:
        ax.axhline(measured_median, color=_KO_C, ls='--', lw=1.3,
                   label=f'measured RNA effect ({measured_median:.3f})')
        ax.legend(fontsize=8, loc='upper right')
    ax.set_xticks(x)
    ax.set_xticklabels([label[k] for k in stages], fontsize=7)
    ax.set_ylabel('median |log2(KO / WT)|')
    ax.set_yscale('log')
    ax.set_title('Knockout signal along the chain\n'
                 '(log scale: where the perturbation dies is the whole story)',
                 fontsize=10)
    ax.grid(axis='y', alpha=0.25, which='both')

    ax = axs[1]
    cmp = metrics.get('readout_comparison') or {}
    names = [k for k in ('layer4_rna', 'layer4_gene', 'layer3_rna') if k in cmp]
    if names:
        rs = [cmp[k].get('spearman_r', np.nan) for k in names]
        aucs = [cmp[k].get('roc_auc', np.nan) - 0.5 for k in names]
        xx = np.arange(len(names))
        ax.bar(xx - 0.18, rs, 0.36, label='Spearman vs measured', color='#2a9d8f')
        ax.bar(xx + 0.18, aucs, 0.36, label='ROC-AUC - 0.5', color=_DELTA_C)
        ax.axhline(0, color='#333333', lw=0.8)
        ax.set_xticks(xx)
        ax.set_xticklabels([n.replace('_', ' ') for n in names], fontsize=8)
        span = max(0.02, float(np.nanmax(np.abs(rs + aucs))) * 1.3)
        ax.set_ylim(-span, span)
        ax.set_ylabel('skill (0 = chance)')
        ax.set_title('Does any read-out carry the KO response?', fontsize=10)
        ax.legend(fontsize=8)
        ax.grid(axis='y', alpha=0.25)
    else:
        ax.axis('off')
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)
    print(f'[ko-viz] Wrote {out_path}')
    return out_path


# ---------------------------------------------------------------------------
# 4. Per-locus mechanism
# ---------------------------------------------------------------------------
def _crop_map(mat, lo_frac, hi_frac):
    n = mat.shape[0]
    lo = int(np.floor(lo_frac * n))
    hi = int(np.ceil(hi_frac * n))
    lo = max(0, min(n - 2, lo))
    hi = max(lo + 2, min(n, hi))
    return mat[lo:hi, lo:hi]


def plot_ko_locus(rec, out_path, n_points=1400):
    """Everything the chain did at one gene, WT against KO.

    Top row: the predicted contact map before the knockout, after it, and the
    difference.  Below: the input tracks that changed (CTCF, RAD21), layer 3's
    RAD21 read-out, and the RNA read-out against the experiment.  All panels
    share one x axis over ``gene +/- flank``, and the maps are cropped to the
    same interval so a change in contacts can be read against the change in
    transcription at the same coordinate.
    """
    lo, hi = rec['view']
    ws, window = rec['window_start'], rec['window']
    lo_f, hi_f = (lo - ws) / window, (hi - ws) / window
    xlim = (lo / 1e6, hi / 1e6)

    tracks = rec['tracks']          # {name: (wt, ko)} over the FULL window
    primary = rec.get('primary_assay', 'rna')
    panel_defs = [
        ('CTCF input\n(log1p)', 'ctcf_in', False),
        ('RAD21 input\n(log1p)', 'rad21_in', False),
        ('RAD21 layer 3\n(linear)', 'rad21_out', False),
        (f'{primary} measured\n(sense, linear)', 'rna_exp', True),
        (f'{primary} layer 4\n(sense, linear)', 'rna_pred', True),
    ]
    # Any further assay layer 4 predicts (mNET-seq, CAGE, ...) is stacked below
    # the primary one, on the same x axis, so a nascent response can be read
    # against the steady-state one at the same coordinate.
    panel_defs += [(label, key, True)
                   for key, label in rec.get('extra_panels', [])]
    panel_defs = [p for p in panel_defs if p[1] in tracks]
    n_tr = len(panel_defs) + 1      # + the predicted log2FC panel

    fig = plt.figure(figsize=(13.5, 4.6 + 1.35 * n_tr))
    gs = gridspec.GridSpec(1 + n_tr, 3, figure=fig,
                           height_ratios=[3.1] + [1.0] * n_tr,
                           hspace=0.28, wspace=0.22)

    # --- contact maps -----------------------------------------------------
    wt_map = _crop_map(rec['hic_wt'], lo_f, hi_f)
    ko_map = _crop_map(rec['hic_ko'], lo_f, hi_f)
    vmax = float(np.nanpercentile(np.concatenate([wt_map.ravel(), ko_map.ravel()]), 99.5))
    vmax = max(vmax, 1e-6)
    ext = [lo / 1e6, hi / 1e6, hi / 1e6, lo / 1e6]
    for j, (m, title) in enumerate([(wt_map, 'predicted Hi-C, WT'),
                                    (ko_map, 'predicted Hi-C, CTCF KO')]):
        ax = fig.add_subplot(gs[0, j])
        im = ax.imshow(m, cmap='Reds', vmin=0, vmax=vmax, extent=ext,
                       interpolation='none')
        ax.set_title(title, fontsize=9)
        ax.tick_params(labelsize=7)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03).ax.tick_params(labelsize=6)
    d = (ko_map - wt_map) / LN2      # model space is ln(1+x): /ln2 -> log2 units
    dlim = max(1e-6, float(np.nanpercentile(np.abs(d), 99.5)))
    ax = fig.add_subplot(gs[0, 2])
    im = ax.imshow(d, cmap='bwr', vmin=-dlim, vmax=dlim, extent=ext,
                   interpolation='none')
    ax.set_title(f'delta log2 contacts (KO - WT)\nmedian |delta|='
                 f'{np.median(np.abs(d)):.4f}', fontsize=9)
    ax.tick_params(labelsize=7)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03).ax.tick_params(labelsize=6)

    # --- tracks -----------------------------------------------------------
    def _view(arr):
        a = np.asarray(arr, dtype=np.float64)
        i0 = int(round((lo - ws) / window * len(a)))
        i1 = int(round((hi - ws) / window * len(a)))
        i0, i1 = max(0, i0), min(len(a), max(i0 + 2, i1))
        return _downsample(a[i0:i1], n_points)

    gene_lo, gene_hi = rec['gene_span']
    axes = []
    for row, (label, key, logscale) in enumerate(panel_defs, start=1):
        ax = fig.add_subplot(gs[row, :])
        wt, ko = tracks[key]
        ywt, yko = _view(wt), _view(ko)
        x = np.linspace(lo, hi, len(ywt)) / 1e6
        ax.fill_between(x, ywt, 0, color=_WT_C, alpha=0.35, lw=0, label='WT')
        ax.plot(x, ywt, color=_WT_C, lw=0.7)
        ax.plot(x, yko, color=_KO_C, lw=0.8, label='CTCF KO')
        if logscale:
            # linthresh has to come from the data: a track whose whole range is
            # below a hard-coded threshold renders as one filled block with
            # scientific-notation ticks, which is what a near-flat head looked
            # like before. Below ~10x dynamic range a linear axis is clearer.
            ymax = float(np.nanmax(np.concatenate([ywt, yko])) or 0.0)
            if ymax > 10:
                ax.set_yscale('symlog', linthresh=max(ymax / 1e3, 1e-3))
        ax.set_ylabel(label, fontsize=7.5)
        ax.tick_params(labelsize=7)
        ax.axvspan(gene_lo / 1e6, gene_hi / 1e6, color='#999999', alpha=0.13, lw=0)
        ax.set_xlim(*xlim)
        ax.tick_params(labelbottom=False)     # one x axis, on the bottom panel
        if row == 1:
            ax.legend(fontsize=7, loc='upper right', framealpha=0.85, ncol=2)
        axes.append(ax)

    # --- predicted log2 fold change ---------------------------------------
    ax = fig.add_subplot(gs[1 + len(panel_defs), :])
    wt, ko = tracks['rna_pred']
    ywt, yko = _view(wt), _view(ko)
    pc = rec.get('pseudocount', 1.0)
    y = np.log2((np.clip(yko, 0, None) + pc) / (np.clip(ywt, 0, None) + pc))
    x = np.linspace(lo, hi, len(y)) / 1e6
    ax.axhline(0, color='#888888', lw=0.7)
    ax.fill_between(x, y, 0, where=y >= 0, color=_KO_C, alpha=0.5, lw=0)
    ax.fill_between(x, y, 0, where=y < 0, color=_WT_C, alpha=0.5, lw=0)
    ax.plot(x, y, color='#333333', lw=0.5)
    ax.axvspan(gene_lo / 1e6, gene_hi / 1e6, color='#999999', alpha=0.13, lw=0)
    lim = max(1e-4, float(np.nanpercentile(np.abs(y), 99.5)) * 1.3)
    ax.set_ylim(-lim, lim)
    ax.set_xlim(*xlim)
    ax.set_ylabel('predicted RNA\nlog2(KO / WT)', fontsize=7.5)
    ax.tick_params(labelsize=7)
    ax.set_xlabel(f"{rec['chrom']} position (Mb)   grey band = scored gene body",
                  fontsize=9)

    exp_fc, pred_fc = rec['exp_log2fc'], rec['pred_log2fc']
    gene_fc = rec.get('gene_log2fc', np.nan)
    notes = []
    if not rec.get('scored', True):
        notes.append('training chromosome -- not held out')
    floor = rec.get('min_wt_coverage')
    if floor is not None and rec['exp_wt_cov'] < floor:
        # Worth saying loudly: a fold change on a gene with no reads is noise
        # over noise, however clean the picture looks.
        notes.append(f"NOT EXPRESSED in this WT RNA-seq (mean sense coverage "
                     f"{rec['exp_wt_cov']:.3f} < {floor:g}): its measured fold "
                     f"change is noise, and it is excluded from the metrics")
    fig.suptitle(
        f"{rec['gene']}  {rec['chrom']}:{gene_lo:,}-{gene_hi:,} ({rec['strand']})   "
        f"measured log2(KO/WT)={exp_fc:+.3f}   layer-4 coverage={pred_fc:+.3f}   "
        f"gene head={gene_fc:+.3f}\n"
        f"WT coverage: measured {rec['exp_wt_cov']:.2f} vs predicted "
        f"{rec['pred_wt_cov']:.2f}    window {rec['chrom']}:{ws:,}-{ws + window:,}"
        + (('\n[' + '; '.join(notes) + ']') if notes else ''),
        fontsize=10, y=0.995)
    fig.savefig(out_path, dpi=170, bbox_inches='tight')
    plt.close(fig)
    print(f'[ko-viz] Wrote {out_path}')
    return out_path


# ---------------------------------------------------------------------------
# 5. Volcanoes: the most up- and down-regulated genes, measured vs predicted
# ---------------------------------------------------------------------------
# The volcano panel itself is shared with the allele-specific benchmark rather
# than reimplemented: same axis capping, same three-colour overlap code, so the
# two benchmarks' figures read identically.
from cshark.inference.utils.allele_rna import (            # noqa: E402
    _volcano_panel, _robust_cap, _C_EXP, _C_PRED, _C_SHARED, _C_BG,
)


def plot_ko_volcanoes(df, exp_sets, pred_sets, overlap, out_path,
                      fdr_thresh=0.1, min_abs_log2=0.0, n_labels=12,
                      rank='effect', exp_null_mode='replicate',
                      title='CTCF knockout: measured vs predicted transcription'):
    """Measured and predicted volcanoes with a shared highlight set.

    ``exp_sets`` / ``pred_sets`` are ``(up_index, down_index)`` pairs; ``overlap``
    is ``{'up': ..., 'down': ..., 'either': ...}`` from ``hypergeom_overlap``.

    Both volcanoes carry the SAME three colours -- top measured only, top
    predicted only, top in BOTH -- so the overlap is readable off either panel,
    while direction stays where a volcano puts it: left is down, right is up.
    The y axes are NOT the same statistic on the two sides and are labelled as
    such; see ``add_volcano_statistics`` for why they cannot be.
    """
    exp_up, exp_down = exp_sets
    pred_up, pred_down = pred_sets
    exp_idx = list(exp_up) + list(exp_down)
    pred_idx = list(pred_up) + list(pred_down)
    shared = sorted(set(map(int, exp_idx)) & set(map(int, pred_idx)))
    label_idx = list(shared[:n_labels])
    if len(label_idx) < n_labels:
        extra = [i for i in exp_idx if int(i) not in set(shared)]
        label_idx += extra[:n_labels - len(label_idx)]

    fig, axs = plt.subplots(2, 2, figsize=(14, 12))
    gate = (f'top {len(exp_up)} up / {len(exp_down)} down by effect size'
            if rank == 'effect'
            else f'FDR<={fdr_thresh:g} and |log2|>={min_abs_log2:g}')
    # Name the null honestly: the two modes answer different questions, and
    # 'empirical' is the fallback that does NOT use the replicates.
    exp_null_text = {
        'replicate': 'replicate noise model, expression-stratified '
                     '(did it move beyond measurement noise?)',
        'empirical': 'robust z-score against the other genes, NO replicate null '
                     '(is it unusual among genes?)',
    }.get(exp_null_mode, str(exp_null_mode))
    _volcano_panel(axs[0, 0], df, 'exp_log2fc', 'exp_pval', 'exp_fdr',
                   exp_idx, pred_idx,
                   f'MEASURED transcriptional response  (n={len(df)})\n'
                   f'{exp_null_text}',
                   'measured log2(KO / WT)', fdr_thresh, min_abs_log2, label_idx)
    _volcano_panel(axs[0, 1], df, 'pred_log2fc', 'pred_pval', 'pred_fdr',
                   exp_idx, pred_idx,
                   f'PREDICTED transcriptional response  (n={len(df)})\n'
                   f'paired-bin null on the layer-4 coverage head -- 64 bp bins '
                   f'are\nautocorrelated, so read the y axis as a RANKING, not '
                   f'a calibrated FDR',
                   'predicted log2(KO / WT)', fdr_thresh, min_abs_log2, label_idx)

    # --- effect-size concordance, same colour code ------------------------
    ax = axs[1, 0]
    x = df['exp_log2fc'].values.astype(float)
    y = df['pred_log2fc'].values.astype(float)
    pos = {int(k): i for i, k in enumerate(df.index)}
    ax.scatter(x, y, s=7, color=_C_BG, alpha=0.5, lw=0, rasterized=True)
    for idx, colour in ((sorted(set(map(int, exp_idx)) - set(shared)), _C_EXP),
                        (sorted(set(map(int, pred_idx)) - set(shared)), _C_PRED),
                        (shared, _C_SHARED)):
        if idx:
            p = [pos[i] for i in idx if i in pos]
            ax.scatter(x[p], y[p], s=26, color=colour, alpha=0.9, lw=0.3,
                       edgecolor='white')
    ok = np.isfinite(x) & np.isfinite(y)
    r = stats.pearsonr(x[ok], y[ok])[0] if ok.sum() >= 3 else float('nan')
    rho = stats.spearmanr(x[ok], y[ok])[0] if ok.sum() >= 3 else float('nan')
    for cap, setter in ((_robust_cap(np.abs(x), 99.5, floor=0.1), ax.set_xlim),
                        (_robust_cap(np.abs(y), 99.5, floor=0.1), ax.set_ylim)):
        if cap is not None:
            setter(-cap, cap)
    ax.axhline(0, color='grey', lw=0.6, ls='--')
    ax.axvline(0, color='grey', lw=0.6, ls='--')
    ax.set_xlabel('measured log2(KO / WT)')
    ax.set_ylabel('predicted log2(KO / WT)')
    ax.set_title(f'Effect-size concordance   Pearson={r:.3f}  Spearman={rho:.3f}',
                 fontsize=10)

    # --- overlap, up and down separately ----------------------------------
    # Direction-matched overlap is the question worth asking: a gene the
    # experiment calls up and the model calls down is not agreement, and a single
    # |log2|-ranked set would count it as such.
    ax = axs[1, 1]
    groups = [(k, overlap[k]) for k in ('up', 'down', 'either') if k in overlap]
    width = 0.26
    xs = np.arange(len(groups))
    for j, (part, colour, lbl) in enumerate((
            ('exp_only', _C_EXP, 'measured only'),
            ('shared', _C_SHARED, 'shared'),
            ('pred_only', _C_PRED, 'predicted only'))):
        vals = []
        for _k, st in groups:
            if part == 'shared':
                vals.append(st['n_shared'])
            elif part == 'exp_only':
                vals.append(st['n_experimental'] - st['n_shared'])
            else:
                vals.append(st['n_predicted'] - st['n_shared'])
        ax.bar(xs + (j - 1) * width, vals, width, color=colour, label=lbl)
    for i, (_k, st) in enumerate(groups):
        exp_chance = st['n_shared_expected_by_chance']
        ax.plot([xs[i] - width * 1.6, xs[i] + width * 1.6],
                [exp_chance, exp_chance], color='black', ls='--', lw=1,
                label='expected shared by chance' if i == 0 else None)
    ax.set_xticks(xs)
    ax.set_xticklabels([k.upper() for k, _ in groups], fontsize=9)
    ax.set_ylabel('Genes')
    ax.set_title('Top-gene overlap, by direction', fontsize=10)
    # Headroom for the stats box and a legend below the bars, so neither lands
    # on top of the data.
    ax.set_ylim(0, max(1.0, ax.get_ylim()[1]) * 1.75)
    ax.legend(fontsize=8, loc='upper right', framealpha=0.9)
    txt = 'shared / measured, vs chance\n' + '\n'.join(
        f"{k:<7}{st['n_shared']:>4}/{st['n_experimental']:<4} "
        f"{st['fold_enrichment']:>5.2f}x p={st['hypergeometric_p']:.2g}"
        for k, st in groups)
    ax.text(0.02, 0.98, txt, transform=ax.transAxes, va='top', ha='left',
            fontsize=8, family='monospace',
            bbox=dict(fc='white', ec='#adb5bd', alpha=0.9))

    fig.suptitle(f'{title}   (highlight sets: {gate})', fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    print(f'[ko-viz] Wrote {out_path}')
    return out_path
