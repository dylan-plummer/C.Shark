"""Allele-specific *transcription* support for the Enformer AS benchmark.

The fine-tuned hierarchical Enformer layer now carries stranded RNA heads
(``rna_plus`` / ``rna_minus``) alongside ``ctcf`` / ``atac``.  This module holds
everything that is specific to scoring those heads against experimentally
quantified allele-specific transcription: reading the per-gene ratio table,
turning both the experimental and the predicted allele bias into a volcano
(effect size + significance), and comparing which genes each side calls.

Experimental input (``*.ratio.res``)
------------------------------------
Tab-separated, one row per gene, columns by position::

    V1  V2     V3   V4    V5           V6      m         p         ratio
    chr start  end  gene  total_count  strand  maternal  paternal  m/p

``m``/``p`` are *not* raw counts.  They are built as

    m = pseudocount + norm_factor * raw_maternal_reads
    p = pseudocount + raw_paternal_reads

with a shared additive pseudocount (8 in the 129xB6 files) and a library-size
factor applied to the maternal strain only, so that ``ratio == 1`` means "no
allele bias".  Both are recovered exactly by :func:`read_rna_ratios`
(``pseudocount`` = the minimum observed value, ``norm_factor`` = the smallest
positive increment above it, because raw counts are integers), which matters
because:

* the *effect size* has to come from the normalised, shrunk ``ratio`` -- that is
  the quantity the experiment reports, and the pseudocount is what keeps genes
  with a handful of reads off the axis limits;
* the *p-value* has to come from the raw counts -- a binomial test needs
  integers, and the pseudocount would otherwise fake away most of the evidence.
  The null proportion is ``1 / (1 + norm_factor)`` rather than 0.5, since under
  no allele bias the *normalised* counts (not the raw ones) are equal.

Genes whose two values are both at the pseudocount floor carry zero
allele-informative reads; they are not "unbiased", they are unmeasured, and
:func:`filter_rna_genes` drops them.

Predicted significance
----------------------
The model is deterministic, so there is no sampling noise to test against.  Two
complementary nulls are offered (``pred_pvalue_mode``):

``bins`` (default)
    Paired Wilcoxon signed-rank test over the predicted bins spanning the gene
    body: is the maternal-minus-paternal difference *consistently* signed along
    the transcript, rather than driven by one bin?  The bins are first averaged
    into at most ``max_points`` equal segments, because otherwise the test's
    sample size is the gene length and the volcano ranks genes by how long they
    are.  Needs at least ``min_bins`` bins; shorter genes fall back to
    ``empirical``.
``empirical``
    Robust z-score of the gene's predicted log2 ratio against the genome-wide
    distribution of predicted log2 ratios (median / MAD).  Asks how unusual this
    gene's allelic difference is compared with the model's baseline spread.

Both are BH-corrected.  "Most differentially expressed" is then defined
identically on the two sides -- pass the FDR cut, pass a minimum |log2 ratio|,
rank by |log2 ratio| (or by p-value; see ``rank_by``) -- so the top-gene overlap
between the panels is a fair comparison rather than an artefact of two different
ranking rules.
"""
import numpy as np
import pandas as pd
from scipy import stats

#: Gene strand -> Enformer head carrying that strand's transcription.
RNA_STRAND_TRACKS = {'+': 'rna_plus', '-': 'rna_minus'}
#: Track names this module needs on the checkpoint.
RNA_TRACKS = ('rna_plus', 'rna_minus')
#: Chromosomes excluded from RNA scoring by default.  chrX allele bias in an F1
#: is dominated by X-inactivation (an epigenetic, parent-of-origin effect no
#: sequence-only model can see -- same argument as the imprinting filter); chrY
#: and chrM are not diploid, so an allele ratio there is meaningless.
RNA_DEFAULT_EXCLUDE_CHROMS = ('chrX', 'chrY', 'chrM')


# ---------------------------------------------------------------------------
# Input
# ---------------------------------------------------------------------------
def read_rna_ratios(path, pseudocount=None, norm_factor=None, verbose=True):
    """Read a ``*.ratio.res`` allele-specific transcription table.

    Returns a DataFrame with ``chr/start/end/gene/total_count/strand``, the file's
    ``maternal``/``paternal``/``ratio`` columns, and the recovered raw read counts
    ``raw_maternal``/``raw_paternal`` (plus ``n_allele_reads`` = their sum).
    ``pseudocount``/``norm_factor`` override the auto-detection described in the
    module docstring.
    """
    df = pd.read_csv(path, sep='\t')
    if df.shape[1] < 9:
        raise ValueError(f'{path}: expected >=9 columns (chr start end gene total '
                         f'strand m p ratio), found {df.shape[1]}: {list(df.columns)}')
    df = df.iloc[:, :9]
    df.columns = ['chr', 'start', 'end', 'gene', 'total_count', 'strand',
                  'maternal', 'paternal', 'ratio']
    df['chr'] = df['chr'].astype(str)
    df['start'] = df['start'].astype(int)
    df['end'] = df['end'].astype(int)
    df['strand'] = df['strand'].astype(str)
    for c in ('total_count', 'maternal', 'paternal', 'ratio'):
        df[c] = pd.to_numeric(df[c], errors='coerce')
    df = df.dropna(subset=['maternal', 'paternal', 'ratio']).reset_index(drop=True)

    pseudo, factor = _infer_count_scaling(df, pseudocount, norm_factor)
    df['raw_maternal'] = np.rint((df['maternal'].values - pseudo) / factor).astype(int)
    df['raw_paternal'] = np.rint(df['paternal'].values - pseudo).astype(int)
    df['raw_maternal'] = df['raw_maternal'].clip(lower=0)
    df['raw_paternal'] = df['raw_paternal'].clip(lower=0)
    df['n_allele_reads'] = df['raw_maternal'] + df['raw_paternal']
    df.attrs['pseudocount'] = float(pseudo)
    df.attrs['norm_factor'] = float(factor)
    if verbose:
        n_floor = int((df['n_allele_reads'] == 0).sum())
        print(f'[rna] {path}: {len(df)} genes; pseudocount={pseudo:g}, '
              f'maternal norm factor={factor:.5f}; {n_floor} genes '
              f'({100 * n_floor / max(1, len(df)):.1f}%) sit at the pseudocount '
              f'floor (no allele-informative reads).')
    return df


def _infer_count_scaling(df, pseudocount, norm_factor):
    """Recover the additive pseudocount and the maternal library-size factor."""
    m = df['maternal'].values.astype(float)
    p = df['paternal'].values.astype(float)
    pseudo = float(min(m.min(), p.min())) if pseudocount is None else float(pseudocount)
    if norm_factor is not None:
        return pseudo, float(norm_factor)
    # Raw counts are integers, so the smallest positive excess over the
    # pseudocount is exactly one read's worth of the maternal scale factor.
    excess = m - pseudo
    excess = excess[excess > 1e-9]
    factor = float(np.min(excess)) if len(excess) else 1.0
    # Guard against a pathological detection (e.g. a table already in raw counts
    # where the smallest excess is a single read on a huge library).
    if not (0.5 <= factor <= 2.0):
        print(f'[rna] Warning: inferred maternal norm factor {factor:.4f} is '
              f'implausible; using 1.0. Pass --rna-norm-factor to override.')
        factor = 1.0
    return pseudo, factor


def filter_rna_genes(df, min_allele_reads=10, min_total_count=0,
                     exclude_chroms=RNA_DEFAULT_EXCLUDE_CHROMS, verbose=True):
    """Keep genes with enough allele-informative signal to be scored."""
    n0 = len(df)
    keep = df['n_allele_reads'].values >= int(min_allele_reads)
    n_reads = int((~keep).sum())
    keep &= df['total_count'].fillna(0).values >= float(min_total_count)
    n_total = int((~keep).sum()) - n_reads
    keep &= df['strand'].isin(RNA_STRAND_TRACKS).values
    if exclude_chroms:
        keep &= ~df['chr'].isin(set(exclude_chroms)).values
    out = df[keep].reset_index(drop=True)
    if verbose:
        print(f'[rna] Gene filter: {n0} -> {len(out)} '
              f'(dropped {n_reads} with <{min_allele_reads} allele reads, '
              f'{max(0, n_total)} below total-count cut, '
              f'{n0 - n_reads - max(0, n_total) - len(out)} on excluded '
              f'chromosomes / unknown strand: {",".join(exclude_chroms or [])}).')
    return out


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------
def benjamini_hochberg(pvals):
    """BH-FDR adjusted p-values; NaNs pass through as NaN."""
    p = np.asarray(pvals, dtype=float)
    out = np.full(p.shape, np.nan)
    ok = np.isfinite(p)
    if not ok.any():
        return out
    q = p[ok]
    n = len(q)
    order = np.argsort(q)
    ranked = q[order] * n / np.arange(1, n + 1)
    ranked = np.minimum.accumulate(ranked[::-1])[::-1]
    adj = np.empty(n)
    adj[order] = np.clip(ranked, 0, 1)
    out[ok] = adj
    return out


def experimental_pvalues(df):
    """Two-sided binomial test per gene on the recovered raw allele counts.

    The null proportion is ``1/(1+norm_factor)``: equal *normalised* counts, i.e.
    the point where the file's ``ratio`` is 1.
    """
    factor = float(df.attrs.get('norm_factor', 1.0))
    null_p = 1.0 / (1.0 + factor)
    k = df['raw_maternal'].values.astype(int)
    n = df['n_allele_reads'].values.astype(int)
    pvals = np.ones(len(df))
    for i in range(len(df)):
        if n[i] <= 0:
            pvals[i] = np.nan
        else:
            pvals[i] = stats.binomtest(int(k[i]), int(n[i]), null_p,
                                       alternative='two-sided').pvalue
    return pvals, null_p


def empirical_pvalues(log2ratio):
    """Robust two-sided z-test of each value against the median/MAD of the set."""
    x = np.asarray(log2ratio, dtype=float)
    ok = np.isfinite(x)
    out = np.full(x.shape, np.nan)
    if ok.sum() < 3:
        return out, np.nan, np.nan
    centre = float(np.median(x[ok]))
    mad = float(np.median(np.abs(x[ok] - centre)))
    scale = 1.4826 * mad
    if scale <= 0:
        scale = float(np.std(x[ok])) or np.nan
    if not np.isfinite(scale) or scale <= 0:
        return out, centre, np.nan
    out[ok] = 2.0 * stats.norm.sf(np.abs(x[ok] - centre) / scale)
    return out, centre, scale


def paired_bin_pvalue(mat_bins, pat_bins, min_bins=8, max_points=100):
    """Paired Wilcoxon signed-rank p over a gene's predicted bins (NaN if too few).

    ``max_points`` averages the bins into at most that many equal-width segments
    of the gene body first.  Without it the test's sample size *is* the gene
    length, so a megabase gene clears any significance cut on a negligible
    allelic difference while a short gene with a large one does not -- the
    predicted volcano would then rank genes by length.
    """
    m = np.asarray(mat_bins, dtype=float)
    p = np.asarray(pat_bins, dtype=float)
    d = m - p
    d = d[np.isfinite(d)]
    if len(d) < int(min_bins) or not np.any(d != 0):
        return np.nan
    if max_points and len(d) > int(max_points):
        edges = np.linspace(0, len(d), int(max_points) + 1).astype(int)
        d = np.array([d[a:b].mean() for a, b in zip(edges[:-1], edges[1:])
                      if b > a])
        if not np.any(d != 0):
            return np.nan
    try:
        return float(stats.wilcoxon(d, alternative='two-sided',
                                    zero_method='wilcox').pvalue)
    except ValueError:
        return np.nan


def add_volcano_statistics(df, pred_pvalue_mode='bins', min_bins=8):
    """Attach p-values / FDRs for both the experimental and predicted volcano.

    Expects ``exp_log2ratio``, ``pred_log2ratio`` and (for ``bins``) a
    ``pred_pval_bins`` column filled in during prediction.  Adds
    ``exp_pval/exp_fdr``, ``pred_pval_empirical``, ``pred_pval``/``pred_fdr``.
    """
    out = df.copy()
    out['exp_pval'], null_p = experimental_pvalues(out)
    out['exp_fdr'] = benjamini_hochberg(out['exp_pval'].values)

    emp, centre, scale = empirical_pvalues(out['pred_log2ratio'].values)
    out['pred_pval_empirical'] = emp
    if 'pred_pval_bins' not in out:
        out['pred_pval_bins'] = np.nan
    if pred_pvalue_mode == 'empirical':
        pred_p = emp.copy()
    else:
        pred_p = out['pred_pval_bins'].values.astype(float).copy()
        gap = ~np.isfinite(pred_p)
        if gap.any():
            pred_p[gap] = emp[gap]
            print(f'[rna] {int(gap.sum())} genes had <{min_bins} predicted bins; '
                  f'used the empirical null for those.')
    out['pred_pval'] = pred_p
    out['pred_fdr'] = benjamini_hochberg(pred_p)
    out.attrs.update(df.attrs)
    out.attrs['exp_null_proportion'] = null_p
    out.attrs['pred_pvalue_mode'] = pred_pvalue_mode
    out.attrs['pred_empirical_null'] = {'centre': centre, 'scale': scale}
    return out


def select_top(df, log2col, fdrcol, pcol, top_n, fdr_thresh, min_abs_log2,
               rank_by='abs_log2'):
    """Indices of the "most differentially expressed" genes on one side.

    Genes must pass ``fdrcol <= fdr_thresh`` and ``|log2| >= min_abs_log2``; the
    survivors are ranked by |log2 ratio| (``rank_by='abs_log2'``) or by p-value,
    and the first ``top_n`` are returned.  The same rule is applied to both sides
    so the overlap is comparable.
    """
    sig = (df[fdrcol].values <= fdr_thresh) & \
          (np.abs(df[log2col].values) >= min_abs_log2)
    sub = df[sig]
    if not len(sub):
        return sub.index[:0]
    if rank_by == 'pvalue':
        order = sub.assign(_a=-np.abs(sub[log2col].values)).sort_values(
            [pcol, '_a'], kind='mergesort').index
    else:
        order = sub.assign(_a=-np.abs(sub[log2col].values)).sort_values(
            ['_a', pcol], kind='mergesort').index
    return order[:int(top_n)]


def overlap_stats(df, exp_idx, pred_idx):
    """Hypergeometric overlap between the two top-gene sets."""
    n_total = int(len(df))
    a, b = set(map(int, exp_idx)), set(map(int, pred_idx))
    shared = a & b
    n_shared = len(shared)
    expected = len(a) * len(b) / n_total if n_total else float('nan')
    p = (float(stats.hypergeom.sf(n_shared - 1, n_total, len(a), len(b)))
         if n_shared and len(a) and len(b) else float('nan'))
    union = len(a | b)
    same_sign = int(np.sum(np.sign(df.loc[sorted(shared), 'exp_log2ratio'].values) ==
                           np.sign(df.loc[sorted(shared), 'pred_log2ratio'].values))) \
        if n_shared else 0
    return {
        'n_genes_tested': n_total,
        'n_top_experimental': len(a),
        'n_top_predicted': len(b),
        'n_shared': n_shared,
        'n_shared_expected_by_chance': float(expected),
        'fold_enrichment': float(n_shared / expected) if expected else float('nan'),
        'jaccard': float(n_shared / union) if union else float('nan'),
        'hypergeometric_p': p,
        'n_shared_same_direction': same_sign,
        'frac_shared_same_direction': (float(same_sign / n_shared)
                                       if n_shared else float('nan')),
    }


def snp_dose_report(df, verbose=True):
    """Summarise predicted allele effects against the variant dose per gene.

    Three things worth knowing come out of this: (1) genes whose input sequence
    is identical between the two genomes MUST have a predicted log2 ratio of
    exactly 0 -- anything else is a bug, not biology; (2) if the predicted effect
    size grows with SNP count, the model is at least responding to the
    substituted variants; (3) if direction accuracy stays at chance in every SNP
    stratum, the response is variant *dose* rather than variant *consequence*.
    """
    if 'n_snps_input' not in df or (df['n_snps_input'] < 0).all():
        return None
    d = df[df['n_snps_input'] >= 0]
    out = {'n_genes': int(len(d))}
    zero = d[d['n_snps_input'] == 0]
    out['n_identical_input'] = int(len(zero))
    if len(zero):
        out['max_abs_pred_log2_identical_input'] = float(
            np.abs(zero['pred_log2ratio']).max())
    ok = np.isfinite(d['pred_log2ratio'].values)
    out['spearman_abs_pred_log2_vs_snps'] = float(stats.spearmanr(
        d['n_snps_input'].values[ok], np.abs(d['pred_log2ratio'].values[ok]))[0])
    out['spearman_abs_exp_log2_vs_snps'] = float(stats.spearmanr(
        d['n_snps_input'].values[ok], np.abs(d['exp_log2ratio'].values[ok]))[0])
    strata = []
    for lo, hi in ((0, 1), (1, 50), (50, 200), (200, 600), (600, np.inf)):
        s = d[(d['n_snps_input'] >= lo) & (d['n_snps_input'] < hi)]
        if not len(s):
            continue
        strata.append({
            'snps': f'{lo}-{"inf" if hi == np.inf else hi}',
            'n': int(len(s)),
            'median_abs_pred_log2': float(np.median(np.abs(s['pred_log2ratio']))),
            'median_abs_exp_log2': float(np.median(np.abs(s['exp_log2ratio']))),
            'direction_accuracy': float(s['direction_correct'].mean()),
        })
    out['strata'] = strata
    if verbose:
        print(f"[rna] SNP-dose control ({out['n_genes']} genes, "
              f"{out['n_identical_input']} with identical input sequence"
              + (f", max |pred log2| there = "
                 f"{out['max_abs_pred_log2_identical_input']:.2e}"
                 if len(zero) else '') + '):')
        print(f"      {'SNPs in input':>14} {'n':>6} {'|pred log2|':>12} "
              f"{'|exp log2|':>11} {'dir acc':>8}")
        for s in strata:
            print(f"      {s['snps']:>14} {s['n']:>6} "
                  f"{s['median_abs_pred_log2']:>12.3f} "
                  f"{s['median_abs_exp_log2']:>11.3f} "
                  f"{s['direction_accuracy']:>8.3f}")
        print(f"      |pred log2| vs SNP count: Spearman="
              f"{out['spearman_abs_pred_log2_vs_snps']:+.3f}   "
              f"(experimental: {out['spearman_abs_exp_log2_vs_snps']:+.3f})")
    return out


# ---------------------------------------------------------------------------
# Volcano figure
# ---------------------------------------------------------------------------
_C_BG = '#c9ccd1'
_C_EXP = '#2a9d8f'
_C_PRED = '#457b9d'
_C_SHARED = '#e63946'


def _sig_line(pvals, fdrs, thresh):
    """The -log10(p) height at which BH-FDR crosses ``thresh`` (NaN if never)."""
    ok = np.isfinite(pvals) & np.isfinite(fdrs)
    sel = ok & (fdrs <= thresh)
    if not sel.any():
        return np.nan
    return -np.log10(max(float(np.max(pvals[sel])), 1e-300))


def _robust_cap(vals, pct=99.0, pad=1.15, floor=5.0):
    """Axis cap that keeps the bulk of the data readable despite a few extremes."""
    v = np.asarray(vals, dtype=float)
    v = v[np.isfinite(v)]
    if not len(v):
        return None
    cap = float(np.percentile(v, pct)) * pad
    if not np.isfinite(cap) or cap <= 0:
        return None
    cap = max(cap, floor)
    return cap if v.max() > cap else None


def _volcano_panel(ax, df, log2col, pcol, fdrcol, exp_idx, pred_idx, title,
                   xlabel, fdr_thresh, min_abs_log2, label_idx):
    x = df[log2col].values
    y = -np.log10(np.clip(df[pcol].values, 1e-300, 1.0))
    # A handful of genes with astronomically small p-values (or huge ratios)
    # would otherwise flatten every other point onto the axis, so both axes are
    # capped at a robust quantile and the offenders are drawn as triangles/
    # arrows at the edge rather than dropped.
    y_cap = _robust_cap(y, pct=99.0)
    x_cap = _robust_cap(np.abs(x), pct=99.5, floor=1.0)
    y_over = y > y_cap if y_cap is not None else np.zeros(len(y), bool)
    x_over = np.abs(x) > x_cap if x_cap is not None else np.zeros(len(x), bool)
    if y_cap is not None:
        y = np.minimum(y, y_cap)
    if x_cap is not None:
        x = np.clip(x, -x_cap, x_cap)
    capped = y_over | x_over
    exp_only = sorted(set(map(int, exp_idx)) - set(map(int, pred_idx)))
    pred_only = sorted(set(map(int, pred_idx)) - set(map(int, exp_idx)))
    shared = sorted(set(map(int, exp_idx)) & set(map(int, pred_idx)))
    pos = {int(k): i for i, k in enumerate(df.index)}

    ax.scatter(x[~capped], y[~capped], s=7, color=_C_BG, alpha=0.55, lw=0,
               label='all genes', rasterized=True)
    if capped.any():
        ax.scatter(x[capped], y[capped], s=14, color=_C_BG, alpha=0.8, lw=0,
                   marker='^', label=f'off-scale (n={int(capped.sum())})')
    for idx, colour, lbl in ((exp_only, _C_EXP, f'top experimental only (n={len(exp_only)})'),
                             (pred_only, _C_PRED, f'top predicted only (n={len(pred_only)})'),
                             (shared, _C_SHARED, f'top in BOTH (n={len(shared)})')):
        if not idx:
            continue
        p = np.array([pos[i] for i in idx])
        for sel, mk, size in ((~capped[p], 'o', 26), (capped[p], '^', 34)):
            if sel.any():
                ax.scatter(x[p[sel]], y[p[sel]], s=size, color=colour, alpha=0.9,
                           lw=0.3, edgecolor='white', marker=mk,
                           label=lbl if mk == 'o' else None)

    hline = _sig_line(df[pcol].values, df[fdrcol].values, fdr_thresh)
    if np.isfinite(hline):
        ax.axhline(hline, color='grey', ls='--', lw=0.8)
        ax.text(ax.get_xlim()[1], hline, f' FDR {fdr_thresh:g}', fontsize=7,
                color='grey', va='bottom', ha='right')
    if min_abs_log2 > 0:
        for v in (-min_abs_log2, min_abs_log2):
            ax.axvline(v, color='grey', ls=':', lw=0.8)
    ax.axvline(0, color='black', lw=0.5, alpha=0.4)

    # Alternate the label offset so neighbouring gene names do not collide.
    for n, i in enumerate(label_idx):
        if i not in pos:
            continue
        k = pos[i]
        dx, dy = ((4, 4), (4, -8), (-4, 6), (-4, -9))[n % 4]
        ax.annotate(str(df['gene'].values[k]), (x[k], y[k]), fontsize=6.5,
                    xytext=(dx, dy), textcoords='offset points', color='#22223b',
                    ha='left' if dx > 0 else 'right')
    ax.set_xlabel(xlabel + (' (capped)' if x_cap is not None else ''))
    ax.set_ylabel(f'-log10({pcol.replace("_", " ")})'
                  + (' (capped)' if y_cap is not None else ''))
    ax.set_title(title, fontsize=10)
    ax.legend(fontsize=7, loc='upper left', framealpha=0.85)


def plot_rna_volcanoes(df, exp_idx, pred_idx, stats_dict, out_path,
                       fdr_thresh=0.05, min_abs_log2=1.0, n_labels=12,
                       pred_pvalue_mode='bins',
                       source_label='the Enformer RNA heads',
                       title='Allele-specific transcription: experimental vs predicted'):
    """Experimental + predicted volcanoes with a shared highlight set, plus the
    concordance scatter and the overlap summary.

    Both volcanoes are coloured by the SAME three categories -- top experimental
    only, top predicted only, top in both -- so the overlap is readable off
    either panel.
    """
    import matplotlib.pyplot as plt

    shared = sorted(set(map(int, exp_idx)) & set(map(int, pred_idx)))
    label_idx = list(shared[:n_labels])
    if len(label_idx) < n_labels:
        extra = [i for i in list(exp_idx) + list(pred_idx) if i not in shared]
        label_idx += extra[:n_labels - len(label_idx)]

    fig, axs = plt.subplots(2, 2, figsize=(14, 12))
    _volcano_panel(axs[0, 0], df, 'exp_log2ratio', 'exp_pval', 'exp_fdr',
                   exp_idx, pred_idx,
                   f'EXPERIMENTAL allele-specific transcription  (n={len(df)})\n'
                   f'binomial test on allele-informative reads',
                   'Experimental log2(129 / b6)', fdr_thresh, min_abs_log2, label_idx)
    _volcano_panel(axs[0, 1], df, 'pred_log2ratio', 'pred_pval', 'pred_fdr',
                   exp_idx, pred_idx,
                   f'PREDICTED allele-specific transcription  (n={len(df)})\n'
                   f'{pred_pvalue_mode} null on {source_label}',
                   'Predicted log2(129 / b6)', fdr_thresh, min_abs_log2, label_idx)

    # (3) Concordance of the two effect sizes, same colour code -------------
    ax = axs[1, 0]
    x, y = df['exp_log2ratio'].values, df['pred_log2ratio'].values
    pos = {int(k): i for i, k in enumerate(df.index)}
    ax.scatter(x, y, s=7, color=_C_BG, alpha=0.5, lw=0, rasterized=True)
    for idx, colour in ((sorted(set(map(int, exp_idx)) - set(shared)), _C_EXP),
                        (sorted(set(map(int, pred_idx)) - set(shared)), _C_PRED),
                        (shared, _C_SHARED)):
        if idx:
            p = [pos[i] for i in idx]
            ax.scatter(x[p], y[p], s=26, color=colour, alpha=0.9, lw=0.3,
                       edgecolor='white')
    ok = np.isfinite(x) & np.isfinite(y)
    r = stats.pearsonr(x[ok], y[ok])[0] if ok.sum() >= 3 else float('nan')
    rho = stats.spearmanr(x[ok], y[ok])[0] if ok.sum() >= 3 else float('nan')
    for cap, setter in ((_robust_cap(np.abs(x), 99.5, floor=1.0), ax.set_xlim),
                        (_robust_cap(np.abs(y), 99.5, floor=1.0), ax.set_ylim)):
        if cap is not None:
            setter(-cap, cap)
    ax.axhline(0, color='grey', lw=0.6, ls='--'); ax.axvline(0, color='grey', lw=0.6, ls='--')
    ax.set_xlabel('Experimental log2(129 / b6)')
    ax.set_ylabel('Predicted log2(129 / b6)')
    ax.set_title(f'Effect-size concordance   Pearson={r:.3f}  Spearman={rho:.3f}',
                 fontsize=10)

    # (4) Overlap summary ---------------------------------------------------
    ax = axs[1, 1]
    n_exp = stats_dict['n_top_experimental']
    n_pred = stats_dict['n_top_predicted']
    n_sh = stats_dict['n_shared']
    ax.bar([0, 1, 2], [n_exp - n_sh, n_sh, n_pred - n_sh],
           color=[_C_EXP, _C_SHARED, _C_PRED], width=0.6)
    ax.axhline(stats_dict['n_shared_expected_by_chance'], color='black', ls='--',
               lw=1, label=f"expected shared by chance "
                           f"({stats_dict['n_shared_expected_by_chance']:.1f})")
    ax.set_xticks([0, 1, 2])
    ax.set_xticklabels(['experimental\nonly', 'shared', 'predicted\nonly'], fontsize=9)
    ax.set_ylabel('Genes')
    ax.set_title('Top allele-specific gene overlap', fontsize=10)
    ax.legend(fontsize=8, loc='upper right')
    txt = (f"fold enrichment: {stats_dict['fold_enrichment']:.2f}x\n"
           f"hypergeometric p: {stats_dict['hypergeometric_p']:.3g}\n"
           f"Jaccard: {stats_dict['jaccard']:.3f}\n"
           f"shared w/ same direction: {stats_dict['n_shared_same_direction']}"
           f"/{n_sh}")
    ax.text(0.02, 0.98, txt, transform=ax.transAxes, va='top', ha='left',
            fontsize=8.5, family='monospace',
            bbox=dict(fc='white', ec='#adb5bd', alpha=0.9))

    fig.suptitle(f'{title}  (FDR<={fdr_thresh:g}, |log2|>={min_abs_log2:g})',
                 fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    print(f'[rna] Wrote {out_path}')
