"""Benchmark a layer-2 RAD21 predictor on WT and on CTCF depletion.

One checkpoint per run (a ``--layer2-only`` checkpoint, with or without
``--layer2-no-seq``, or any hierarchical checkpoint that carries
``input_pred_model.*``).  For every 2 Mb window of the evaluated chromosomes
the model is run three times; every input other than CTCF stays WT:

  * wt    -- the measured WT tracks.
  * ko    -- simulated depletion: CTCF knocked out of its own input track
             (``track_ko``, mode ``knockout`` by default: called peaks replaced
             by local background, the operator perturb.py and
             evaluate_ctcf_ko_rna.py use).
  * koexp -- measured depletion: the CTCF ChIP after degron depletion, put on
             the WT input's scale (``MeasuredCtcfKO``).  Separates errors of
             the knockout simulation from errors of the model.

Windows overlap by half and only the central half of each prediction is kept.
Predictions are scored against the experimental WT RAD21 and the experimental
RAD21 after CTCF depletion, at RAD21 peaks called on either experiment:

  fig_benchmark[_measured_ctcf]  pred vs exp WT peak strength | pred vs exp KO
                                 peak strength | ROC for lost (and gained) peaks
  fig_log2fc[_measured_ctcf]     predicted vs experimental KO/WT log2FC per peak
  loci/      track views of example peaks (hit / missed / false call, from the
             simulated-KO arm; both KO arms drawn), listed in loci.tsv
  genes/     track views of the --genes loci, listed in genes.tsv

Baselines, so the numbers can be read:
  * no-change -- the WT prediction scored against the KO experiment.  RAD21 is
    mostly unchanged, so KO R^2 alone says little; it has to beat this.
  * CTCF removed -- each peak scored by how much CTCF signal its KO input lost
    there.  The model's ROC has to beat this to show it learnt more than
    "RAD21 goes where CTCF went".

Normalisation.  The *_norm_10 bigwigs are each divided by their own
per-chromosome 99.95th percentile (normalize_bigwig.py).  Two consequences:

  * RAD21 KO target: with the strongest (CTCF-site) peaks gone after depletion
    the KO divisor is smaller and the KO track sits ~4-5x above the WT one.
    Fold changes are centred on each chromosome's median over peaks (for
    experiment and prediction alike) before thresholding and correlating, and
    KO peak strengths are put on the WT scale with the same factor
    (``exp_ko_wt_scale``) before they are compared with predictions.
  * CTCF KO input: the same scheme on the depleted CTCF ChIP divides by what is
    left after depletion and inflates its background ~8x, which the model
    would read as CTCF everywhere.  ``MeasuredCtcfKO`` instead starts from the
    raw tracks: the KO library is divided by the WT track's per-chromosome
    divisor (exactly what the model's WT CTCF input is) after matching the two
    libraries -- by their background (default; depletion removes peaks, not
    background) or by total coverage (``--ctcf-ko-scaling library``).
"""
import os
import json
import argparse

import numpy as np
import pandas as pd
import pyBigWig
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import roc_curve, roc_auc_score, average_precision_score
from tqdm import tqdm

from cshark.data.data_feature import SequenceFeature, bigwig_values
from cshark.inference.utils.hierarchical_utils import load_hierarchical_rad21_predictor
from cshark.perturb.operators.track_ops import track_ko

WINDOW = 2097152
DATA = '/mnt/jinstore/JinLab03/xxl1432/HiCorr_Deeploop/11.C.shark_checkpoints/cshark_data/data/mm10'
RAW_CTCF = '/home/xww/WWW/mESC_CTCF'
# Chromosomes GenomeDataset holds out of training (val chr10, test chr15, chrX).
HELD_OUT = ('chr10', 'chr15', 'chrX')
# KO arm -> (figure-file suffix, description).
ARMS = {'ko': ('', 'simulated CTCF KO'), 'koexp': ('_measured_ctcf', 'measured CTCF-depleted input')}
COLORS = {'wt': '#3a4fbf', 'ko': '#d1495b', 'koexp': '#7b2cbf', 'ctcf_wt': '#555555'}


def build_parser():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--checkpoint', required=True)
    p.add_argument('--wt-dir', default=f'{DATA}/Hsieh_CTCF_WT_5kb_norm_10',
                   help='WT celltype dir; its genomic_features/ supplies the model inputs and WT RAD21.')
    p.add_argument('--ko-dir', default=f'{DATA}/Hsieh_CTCF_KO_5kb_norm_10')
    p.add_argument('--ko-rad21', default='exp_rad21_after_ctcf_ko_hsieh_0_10.bw',
                   help='Experimental RAD21 after CTCF depletion, under <ko-dir>/genomic_features/.')
    p.add_argument('--seq-dir', default=None, help='Default: <wt-dir>/../dna_sequence.')
    p.add_argument('--chroms', nargs='+', default=['chr10', 'chr15'],
                   help='Default: the trainer\'s val (chr10) and test (chr15) chromosomes, never trained on.')
    p.add_argument('--out-dir', default=None,
                   help='Default: benchmarks/rad21_ctcf_ko/<checkpoint name>.')
    p.add_argument('--label', default=None, help='Name used in figure titles. Default: checkpoint name.')
    p.add_argument('--ctcf-ko-mode', default='knockout')
    p.add_argument('--ctcf-ko-peak-height', type=float, default=0.5,
                   help='Peak threshold of the knockout, in log1p units (default 0.5, as in the '
                        'RAD21 perturbation scripts and evaluate_ctcf_ko_rna.py).')
    p.add_argument('--peak-quantile', type=float, default=0.99,
                   help='RAD21 peaks: summits above this quantile of the non-zero bins of a '
                        'chromosome, called on the WT and on the KO experiment and merged.')
    p.add_argument('--peak-min-dist-bp', type=int, default=1000)
    p.add_argument('--peak-half-width-bp', type=int, default=512,
                   help='Peak strength = mean signal within this distance of the summit.')
    p.add_argument('--lfc-threshold', type=float, default=1.0,
                   help='|centred log2 KO/WT| at which an experimental peak counts as lost/gained.')
    p.add_argument('--pseudocount', type=float, default=0.1,
                   help='Added to peak strengths before the log2 fold change (tracks are 0-10 scaled).')
    p.add_argument('--batch-size', type=int, default=2)
    p.add_argument('--device', default=None)

    g = p.add_argument_group('measured CTCF-depleted input (the koexp arm)')
    g.add_argument('--wt-ctcf-raw', default=f'{RAW_CTCF}/WT.mESC.CTCF.hsieh.bw',
                   help='Raw WT CTCF the model\'s WT ctcf.bw was normalised from (checked per chromosome).')
    g.add_argument('--ko-ctcf-raw', default=f'{RAW_CTCF}/KO.mESC.CTCF.hsieh.bw',
                   help='Raw CTCF ChIP after depletion, same pipeline as --wt-ctcf-raw.')
    g.add_argument('--ctcf-ko-scaling', choices=['background', 'library'], default='background',
                   help='How the KO library is matched to WT before the WT divisor: background = '
                        'trimmed mean of the non-peak signal per chromosome (default); library = '
                        'total coverage (leaves the KO background raised by the reads that left the peaks).')
    g.add_argument('--no-measured-ctcf', dest='measured_ctcf', action='store_false',
                   help='Skip the koexp arm.')

    g = p.add_argument_group('locus gallery')
    g.add_argument('--loci-per-class', type=int, default=6,
                   help='Loci drawn per category (lost/gained x hit/missed/false). Hits are a random '
                        'sample; for errors, half are the largest |pred - exp log2FC| and half random. '
                        '0 skips the gallery.')
    g.add_argument('--locus-flank-bp', type=int, default=10000, help='Plotted on each side of the summit.')
    g.add_argument('--context-tracks', nargs='+', default=['atac', 'h3k27ac', 'h3k4me3'],
                   help='WT tracks drawn under the RAD21/CTCF rows.')
    g.add_argument('--seed', type=int, default=0)

    g = p.add_argument_group('gene loci')
    g.add_argument('--genes', nargs='+', default=None,
                   help='Gene names to draw (e.g. Cdiptos Pcdhga2 Notch1 Mesp1), on any chromosome; '
                        'each is predicted in its own 2 Mb window.')
    g.add_argument('--genes-only', action='store_true',
                   help='Only draw --genes: skip the genome-wide benchmark.')
    g.add_argument('--gene-flank-bp', type=int, default=50000, help='Plotted on each side of the gene body.')
    g.add_argument('--gtf', default='cshark_data/data/mm10/mm10_genes.gtf',
                   help='GENCODE GTF (default: the repo\'s mm10 GENCODE vM25 link, relative to the repo root).')
    return p


# ---------------------------------------------------------------------------
# Measured CTCF after depletion, on the model's WT input scale
# ---------------------------------------------------------------------------
def _background(v):
    """Trimmed mean of the non-zero signal below its 90th percentile."""
    nz = v[v > 0]
    return float(nz[nz <= np.quantile(nz, 0.9)].mean())


class MeasuredCtcfKO:
    """Raw KO CTCF -> the scale of the model's WT CTCF input.

    WT input = clip(raw_WT / q, 0, 1) * 10, q = the per-chromosome 99.95th
    percentile of the non-zero raw WT signal (normalize_bigwig.py).  The KO
    input is clip(raw_KO * s / q, 0, 1) * 10 with the SAME q and a library
    factor s, so CTCF that is gone after depletion reads as gone.
    """

    def __init__(self, wt_raw, ko_raw, wt_norm, scaling):
        self.wt_raw, self.ko_raw, self.wt_norm, self.scaling = wt_raw, ko_raw, wt_norm, scaling
        self.stats = {}
        if scaling == 'library':
            with pyBigWig.open(wt_raw) as a, pyBigWig.open(ko_raw) as b:
                self.library = a.header()['sumData'] / b.header()['sumData']

    def factors(self, chrom):
        if chrom not in self.stats:
            with pyBigWig.open(self.wt_raw) as a, pyBigWig.open(self.ko_raw) as b:
                n = min(a.chroms(chrom), b.chroms(chrom))
                w = np.nan_to_num(bigwig_values(a, chrom, 0, n), nan=0.0)
                k = np.nan_to_num(bigwig_values(b, chrom, 0, n), nan=0.0)
            q = float(np.quantile(w[w != 0], 0.9995))
            s = _background(w) / _background(k) if self.scaling == 'background' else self.library
            # The WT divisor must reproduce the model's WT CTCF input.
            mid = max(0, n // 2 - WINDOW // 2)
            with pyBigWig.open(self.wt_norm) as c:
                ref = np.nan_to_num(bigwig_values(c, chrom, mid, mid + WINDOW), nan=0.0)
            err = float(np.abs(np.clip(w[mid:mid + WINDOW] / q, 0, 1) * 10 - ref).max())
            strong = w >= 0.5 * q                      # WT input >= 5: strong CTCF sites
            st = {'wt_divisor': q, 'ko_scale': s, 'scaling': self.scaling, 'wt_reconstruction_max_err': err,
                  'ctcf_left_at_strong_wt_sites': float(k[strong].mean() * s / w[strong].mean()),
                  'ko_over_wt_background': _background(k) * s / _background(w)}
            self.stats[chrom] = st
            print(f'[ctcf-ko] {chrom}: WT divisor {q:.1f}, KO x{s:.3f} ({self.scaling}); CTCF left at strong '
                  f'WT sites {st["ctcf_left_at_strong_wt_sites"]:.1%}, KO/WT background '
                  f'{st["ko_over_wt_background"]:.2f}; WT reconstruction max err {err:.2g}')
            if err > 1e-3:
                print(f'[ctcf-ko] WARNING: {self.wt_raw} does not reproduce {self.wt_norm} on {chrom} '
                      f'(max err {err:.3g}); the measured-KO input is on the wrong scale.')
            del w, k
        return self.stats[chrom]['wt_divisor'], self.stats[chrom]['ko_scale']

    def window(self, chrom, start, end):
        q, s = self.factors(chrom)
        with pyBigWig.open(self.ko_raw) as b:
            v = np.nan_to_num(bigwig_values(b, chrom, start, end), nan=0.0)
        return (np.clip(v * s / q, 0, 1) * 10).astype(np.float32)


# ---------------------------------------------------------------------------
# Prediction
# ---------------------------------------------------------------------------
class Predictor:
    """Runs the model on the wt / ko / koexp inputs of 2 Mb windows."""

    def __init__(self, model, track_paths, ctcf_i, log_tracks, n_out, seq_dir, args, device, measured=None):
        self.model, self.track_paths, self.ctcf_i = model, track_paths, ctcf_i
        self.log_tracks, self.n_out, self.seq_dir = log_tracks, n_out, seq_dir
        self.args, self.device, self.measured = args, device, measured
        self.bin_bp = WINDOW // n_out

    def _forward(self, xs):
        x = torch.from_numpy(np.stack(xs)).to(self.device)
        with torch.no_grad():
            y = self.model(x)['1d'][:, :, 0].float()
        return np.clip(torch.expm1(y).cpu().numpy(), 0, None)

    def _inputs(self, chrom, seq, bws, ws):
        onehot = seq.get(ws, ws + WINDOW).astype(np.float32)
        tracks = [np.nan_to_num(bigwig_values(bw, chrom, ws, ws + WINDOW), nan=0.0, posinf=0.0, neginf=0.0)
                  .astype(np.float32) for bw in bws]
        to_in = np.log1p if self.log_tracks else (lambda t: t)
        tracks = [to_in(t) for t in tracks]
        c = self.ctcf_i
        arms = {'wt': tracks, 'ko': list(tracks)}
        arms['ko'][c] = track_ko(0, WINDOW, tracks[c].astype(np.float64).copy(), window=WINDOW,
                                 ko_mode=self.args.ctcf_ko_mode,
                                 peak_height=self.args.ctcf_ko_peak_height).astype(np.float32)
        if self.measured is not None:
            arms['koexp'] = list(tracks)
            arms['koexp'][c] = to_in(self.measured.window(chrom, ws, ws + WINDOW))
        lin = np.expm1 if self.log_tracks else (lambda t: t)
        ctcf = {a: lin(t[c]).reshape(self.n_out, self.bin_bp).mean(1) for a, t in arms.items()}
        return {a: np.concatenate([onehot, np.stack(t, 1)], 1) for a, t in arms.items()}, ctcf

    def run(self, chrom, starts, desc=None):
        """Yield ``(window start, {pred_<arm>, ctcf_<arm>: (n_out,) linear})`` per window."""
        seq = SequenceFeature(path=os.path.join(self.seq_dir, f'{chrom}.fa.gz'))
        bws = [pyBigWig.open(p) for p in self.track_paths]
        try:
            batch = []
            for i, ws in enumerate(tqdm(starts, desc=desc or chrom, disable=len(starts) < 3)):
                batch.append((ws,) + self._inputs(chrom, seq, bws, ws))
                if len(batch) == self.args.batch_size or i == len(starts) - 1:
                    preds = {a: self._forward([b[1][a] for b in batch]) for a in batch[0][1]}
                    for j, (w, _, ctcf) in enumerate(batch):
                        yield w, {**{f'pred_{a}': preds[a][j] for a in preds},
                                  **{f'ctcf_{a}': v for a, v in ctcf.items()}}
                    batch = []
        finally:
            for bw in bws:
                bw.close()

    def chrom(self, chrom, chrom_len):
        """Per-bin tracks over ``chrom`` from the central half of half-overlapping
        windows.  Bins no kept centre covers (the chromosome tail) are NaN."""
        n_bins = chrom_len // self.bin_bp
        starts = list(range(0, chrom_len - WINDOW + 1, WINDOW // 2))
        q, out = self.n_out // 4, {}
        for ws, d in self.run(chrom, starts):
            k = starts.index(ws)
            lo = 0 if k == 0 else q
            hi = self.n_out if k == len(starts) - 1 else self.n_out - q
            b0 = ws // self.bin_bp
            for key, v in d.items():
                out.setdefault(key, np.full(n_bins, np.nan, dtype=np.float32))[b0 + lo:b0 + hi] = v[lo:hi]
        return out

    def region(self, chrom, chrom_len, start, end):
        """Per-bin tracks over [start, end) from one window centred on it -> (first bin, tracks)."""
        ws = int(np.clip((start + end) // 2 - WINDOW // 2, 0, chrom_len - WINDOW))
        ws -= ws % self.bin_bp
        (_, d), = list(self.run(chrom, [ws]))
        lo, hi = (max(start, ws) - ws) // self.bin_bp, (min(end, ws + WINDOW) - ws) // self.bin_bp
        return ws // self.bin_bp + lo, {k: v[lo:hi] for k, v in d.items()}


def binned_bigwig(path, chrom, n_bins, bin_bp, start_bin=0):
    """Mean linear signal of ``path`` in ``n_bins`` bins of ``bin_bp`` from ``start_bin``."""
    out = np.empty(n_bins, dtype=np.float32)
    step = (WINDOW // bin_bp) * bin_bp
    s0 = start_bin * bin_bp
    with pyBigWig.open(path) as bw:
        for s in range(0, n_bins * bin_bp, step):
            e = min(s + step, n_bins * bin_bp)
            v = np.nan_to_num(bigwig_values(bw, chrom, s0 + s, s0 + e), nan=0.0)
            out[s // bin_bp:e // bin_bp] = v.reshape(-1, bin_bp).mean(1)
    return out


# ---------------------------------------------------------------------------
# Peaks
# ---------------------------------------------------------------------------
def call_summits(signal, quantile, min_dist):
    nz = signal[signal > 0]
    if nz.size == 0:
        return np.array([], dtype=int)
    idx, _ = find_peaks(signal, height=np.quantile(nz, quantile), distance=min_dist)
    return idx


def merge_summits(a, b, strength, min_dist):
    """Union of two summit sets; within ``min_dist`` keep the stronger summit."""
    allp = np.unique(np.concatenate([a, b]))
    keep = []
    for i in allp:
        if keep and i - keep[-1] < min_dist:
            if strength[i] > strength[keep[-1]]:
                keep[-1] = i
        else:
            keep.append(i)
    return np.array(keep, dtype=int)


def peak_strength(signal, summits, hw):
    c = np.concatenate([[0.0], np.cumsum(np.nan_to_num(signal, nan=0.0), dtype=np.float64)])
    lo = np.clip(summits - hw, 0, len(signal))
    hi = np.clip(summits + hw + 1, 0, len(signal))
    return (c[hi] - c[lo]) / (hi - lo)


def chrom_summits(exp_wt, exp_ko, covered, args, bin_bp):
    min_dist = max(1, args.peak_min_dist_bp // bin_bp)
    hw = args.peak_half_width_bp // bin_bp
    summits = merge_summits(call_summits(np.where(covered, exp_wt, 0), args.peak_quantile, min_dist),
                            call_summits(np.where(covered, exp_ko, 0), args.peak_quantile, min_dist),
                            np.maximum(exp_wt, exp_ko), min_dist)
    summits = summits[(summits >= hw) & (summits < len(exp_wt) - hw)]
    return summits[covered[summits - hw] & covered[summits + hw]]


def ko_rad21_scale(exp_wt, exp_ko, summits, hw, pc):
    """Median KO/WT ratio at peaks: the per-chromosome scaling offset of the KO target."""
    lfc = np.log2((peak_strength(exp_ko, summits, hw) + pc) / (peak_strength(exp_wt, summits, hw) + pc))
    return float(2 ** np.median(lfc))


# ---------------------------------------------------------------------------
# Metrics and figures
# ---------------------------------------------------------------------------
def r2(x, y):
    return float(pearsonr(x, y)[0] ** 2)


def corr_block(pred, exp):
    lp, le = np.log1p(pred), np.log1p(exp)
    return {'r2': r2(pred, exp), 'pearson': float(pearsonr(pred, exp)[0]),
            'spearman': float(spearmanr(pred, exp)[0]), 'pearson_log1p': float(pearsonr(lp, le)[0])}


def roc_block(labels, score):
    if labels.sum() == 0 or labels.sum() == len(labels):
        return None
    return {'auc': float(roc_auc_score(labels, score)), 'auprc': float(average_precision_score(labels, score)),
            'n_pos': int(labels.sum()), 'frac_pos': float(labels.mean())}


def despine(ax):
    for s in ('top', 'right'):
        ax.spines[s].set_visible(False)


def scatter_panel(ax, exp, pred, title_x, title_y, r2_val):
    lim = float(np.quantile(np.concatenate([exp, pred]), 0.999)) * 1.05
    ax.scatter(exp, pred, s=3, color='#3a4fbf', alpha=0.6, linewidths=0, rasterized=True)
    ax.plot([0, lim], [0, lim], color='gray', lw=1)
    ax.set_xlim(0, lim)
    ax.set_ylim(0, lim)
    ax.set_aspect('equal')
    ax.set_xlabel(title_x)
    ax.set_ylabel(title_y)
    ax.text(0.15, 0.85, f'R$^2$ = {r2_val:.3f}', transform=ax.transAxes, fontsize=12)
    despine(ax)


def plot_benchmark(df, m, arm, label, path):
    a = m['arms'][arm]
    fig = plt.figure(figsize=(13, 4.6))
    gs = fig.add_gridspec(1, 3, width_ratios=[1, 1, 1.15], wspace=0.45)
    ax1, ax2, ax3 = (fig.add_subplot(gs[0, i]) for i in range(3))
    scatter_panel(ax1, df.exp_wt.values, df.pred_wt.values, 'Experimental Control\nRAD21 peak strength',
                  'Predicted Control\nRAD21 peak strength', m['wt_peaks']['r2'])
    scatter_panel(ax2, df.exp_ko_wt_scale.values, df[f'pred_{arm}'].values,
                  '$\\Delta$CTCF-IAA experimental\nRAD21 peak strength (control scale)',
                  '$\\Delta$CTCF-IAA predicted\nRAD21 peak strength', a['ko_peaks']['r2'])
    fig.text(0.33, 0.97, 'Comparison between pred RAD21 peak strength and exp RAD21 peak strength',
             ha='center', fontsize=12)
    fig.text(0.33, 0.92, f'({label}; $\\Delta$CTCF input: {ARMS[arm][1]})', ha='center', fontsize=11)

    ax3.plot([0, 1], [0, 1], ls='--', color='gray')
    curves = [('lost', f'pred_score_lost_{arm}', '#f5a623', '-', 'Model, lost'),
              ('lost', f'ctcf_removed_{arm}', '#9b9b9b', ':', 'CTCF removed, lost'),
              ('gained', f'pred_score_gained_{arm}', '#2a9d8f', '-', 'Model, gained')]
    for cls, col, color, ls, name in curves:
        y = (df['exp_class'] == cls).values
        if y.sum() < 10:
            continue
        fpr, tpr, _ = roc_curve(y, df[col].values)
        auc = roc_auc_score(y, df[col].values)
        ax3.plot(fpr, tpr, color=color, ls=ls, lw=2, label=f'{name} (n={int(y.sum())})\nAUC {auc:.3f}')
    ax3.legend(loc='lower right', frameon=False, fontsize=8.5)
    ax3.set_xlim(0, 1)
    ax3.set_ylim(0, 1.02)
    ax3.set_xlabel('False positive rate')
    ax3.set_ylabel('True positive rate')
    ax3.set_title('Comparison with\nexperimental lost/gained RAD21 peaks', fontsize=11)
    despine(ax3)
    fig.savefig(path + '.png', dpi=300, bbox_inches='tight')
    fig.savefig(path + '.pdf', bbox_inches='tight')
    plt.close(fig)


def plot_log2fc(df, m, arm, label, path):
    fig, ax = plt.subplots(figsize=(5.2, 5))
    colors = {'lost': '#d1495b', 'gained': '#2a9d8f', 'unchanged': '#b0b0b0'}
    col = f'pred_lfc_{arm}'
    for cls in ('unchanged', 'lost', 'gained'):
        d = df[df.exp_class == cls]
        ax.scatter(d.exp_lfc, d[col], s=4, color=colors[cls], alpha=0.6, linewidths=0,
                   label=f'{cls} (n={len(d)})', rasterized=True)
    lim = float(np.quantile(np.abs(np.concatenate([df.exp_lfc, df[col]])), 0.999)) * 1.05
    ax.plot([-lim, lim], [-lim, lim], color='gray', lw=1)
    ax.axhline(0, color='k', lw=0.5)
    ax.axvline(0, color='k', lw=0.5)
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_xlabel('Experimental log$_2$ ($\\Delta$CTCF / control) RAD21')
    ax.set_ylabel('Predicted log$_2$ ($\\Delta$CTCF / control) RAD21')
    d = m['arms'][arm]['log2fc']
    ax.set_title(f'{label}; {ARMS[arm][1]}\nPearson {d["pearson"]:.3f}, Spearman {d["spearman"]:.3f}', fontsize=10)
    ax.legend(frameon=False, fontsize=8, loc='upper left', markerscale=3)
    despine(ax)
    fig.savefig(path + '.png', dpi=300, bbox_inches='tight')
    fig.savefig(path + '.pdf', bbox_inches='tight')
    plt.close(fig)


# ---------------------------------------------------------------------------
# Track views (locus gallery and gene loci)
# ---------------------------------------------------------------------------
def predicted_classes(df, arm):
    """Rank-matched calls: the model calls as many peaks lost (gained) as the
    experiment does -- the k most negative (positive) predicted log2FC -- so
    the hit/missed/false split does not hinge on a predicted-scale threshold."""
    out = np.full(len(df), 'unchanged', dtype=object)
    for cls in ('lost', 'gained'):
        k = int((df.exp_class == cls).sum())
        out[np.argsort(-df[f'pred_score_{cls}_{arm}'].values, kind='stable')[:k]] = cls
    return out


CATEGORIES = {
    'lost_hit': 'Lost in experiment, predicted lost',
    'lost_missed': 'Lost in experiment, NOT predicted lost',
    'lost_false': 'Predicted lost, NOT lost in experiment',
    'gained_hit': 'Gained in experiment, predicted gained',
    'gained_missed': 'Gained in experiment, NOT predicted gained',
    'gained_false': 'Predicted gained, NOT gained in experiment',
}


def pick_loci(df, n, seed):
    """{category: [(row index, how picked)]}, from the simulated-KO arm.  Hits:
    random.  Errors: the largest |pred - exp| log2FC (systematic mistakes
    surface there, but so do noisy low-signal peaks) plus a random sample
    (what a typical error is)."""
    rng = np.random.default_rng(seed)
    err = (df.pred_lfc_ko - df.exp_lfc).abs().values
    picks = {}
    for cls in ('lost', 'gained'):
        exp_pos = (df.exp_class == cls).values
        pred_pos = (df.pred_class_ko == cls).values
        for kind, mask in (('hit', exp_pos & pred_pos), ('missed', exp_pos & ~pred_pos),
                           ('false', ~exp_pos & pred_pos)):
            idx = np.flatnonzero(mask)
            chosen = []
            if kind != 'hit':
                worst = idx[np.argsort(-err[idx])][:n // 2]
                chosen = [(i, 'largest error') for i in worst]
                idx = np.setdiff1d(idx, worst)
            rest = rng.choice(idx, min(n - len(chosen), len(idx)), replace=False) if len(idx) else []
            picks[f'{cls}_{kind}'] = chosen + [(int(i), 'random') for i in rest]
    return picks


def context_tracks(args):
    out = {}
    for t in args.context_tracks:
        p = os.path.join(args.wt_dir, 'genomic_features', f'{t}.bw')
        if os.path.isfile(p):
            out[t] = ('H3K' + t[3:] if t.startswith('h3k') else t.upper(), p)
    return out


def track_series(s, lo, hi):
    """The RAD21 / CTCF rows for bins [lo, hi) of per-bin tracks ``s``."""
    ko_arms = [a for a in ARMS if f'pred_{a}' in s]
    return {
        'rad21_exp': [(s['exp_wt'][lo:hi], COLORS['wt'], 'control'),
                      (s['exp_ko'][lo:hi], COLORS['ko'], '$\\Delta$CTCF')],
        'rad21_pred': [(s['pred_wt'][lo:hi], COLORS['wt'], 'control')]
                      + [(s[f'pred_{a}'][lo:hi], COLORS[a], ARMS[a][1]) for a in ko_arms],
        'ctcf_in': [(s['ctcf_wt'][lo:hi], COLORS['ctcf_wt'], 'control')]
                   + [(s[f'ctcf_{a}'][lo:hi], COLORS[a], ARMS[a][1]) for a in ko_arms],
    }


TRACK_ROWS = [('rad21_exp', 'RAD21\nexperiment'), ('rad21_pred', 'RAD21\npredicted'),
              ('ctcf_in', 'CTCF\nmodel input')]


def draw_tracks(axes, x, series, row_labels, legend):
    for ax, (key, row_label) in zip(axes, row_labels):
        for k, (y, c, name) in enumerate(series[key]):
            if k == 0:
                ax.fill_between(x, 0, y, color=c, alpha=0.35, lw=0, label=name, step='mid')
            else:
                ax.step(x, y, color=c, lw=1.0, where='mid', label=name)
        ax.set_ylim(bottom=0)
        ax.tick_params(labelsize=7)
        despine(ax)
        if row_label is not None:
            ax.set_ylabel(row_label, fontsize=8, rotation=0, ha='right', va='center')
        if legend and key in ('rad21_exp', 'rad21_pred', 'ctcf_in'):
            ax.legend(fontsize=6.5, frameon=False, loc='upper left')


def plot_locus_category(rows, store, cat, label, args, extra, bin_bp, path):
    """One column per locus; rows: RAD21 exp, RAD21 pred, CTCF input, context tracks."""
    flank = args.locus_flank_bp // bin_bp
    hw = args.peak_half_width_bp // bin_bp
    row_labels = TRACK_ROWS + [(k, lbl) for k, (lbl, _) in extra.items()]
    fig, axes = plt.subplots(len(row_labels), len(rows), figsize=(3.0 * len(rows) + 1, 1.25 * len(row_labels) + 1.3),
                             squeeze=False, sharex='col')
    for j, (_, r, how) in enumerate(rows):
        s = store[r.chrom]
        b = r.summit // bin_bp
        lo, hi = max(0, b - flank), min(len(s['exp_wt']), b + flank + 1)
        x = (np.arange(lo, hi) * bin_bp + bin_bp // 2 - r.summit) / 1000
        series = track_series(s, lo, hi)
        for k, (_, p) in extra.items():
            series[k] = [(binned_bigwig(p, r.chrom, hi - lo, bin_bp, lo), COLORS['ctcf_wt'], None)]
        draw_tracks(axes[:, j], x, series, [(k, lbl if j == 0 else None) for k, lbl in row_labels], legend=j == 0)
        for ax in axes[:, j]:
            ax.axvspan(-hw * bin_bp / 1000, hw * bin_bp / 1000, color='#f5a623', alpha=0.15, lw=0)
        axes[0, j].set_title(f'{r.chrom}:{r.summit:,}  [{how}]\nlog2FC exp {r.exp_lfc:+.2f}, pred {r.pred_lfc_ko:+.2f}'
                             + (f' (measured input {r.pred_lfc_koexp:+.2f})' if 'pred_lfc_koexp' in r else '')
                             + f'\npeak exp {r.exp_wt:.2f}$\\to${r.exp_ko_wt_scale:.2f}, '
                             f'pred {r.pred_wt:.2f}$\\to${r.pred_ko:.2f}', fontsize=7.5)
        axes[-1, j].set_xlabel('kb from summit', fontsize=8)
    fig.suptitle(f'{CATEGORIES[cat]} (simulated KO)  ({label})\nexperimental $\\Delta$CTCF RAD21 on control scale; '
                 f'shaded = peak window; log2FC centred per chromosome', fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(path + '.png', dpi=200)
    fig.savefig(path + '.pdf')
    plt.close(fig)


def plot_loci(df, store, label, args, out_dir, bin_bp):
    loci_dir = os.path.join(out_dir, 'loci')
    os.makedirs(loci_dir, exist_ok=True)
    extra = context_tracks(args)
    listing = []
    for cat, picks in pick_loci(df, args.loci_per_class, args.seed).items():
        if not picks:
            print(f'[rad21-ko] loci: no {cat} peaks.')
            continue
        rows = [(i, df.loc[i], how) for i, how in picks]
        plot_locus_category(rows, store, cat, label, args, extra, bin_bp, os.path.join(loci_dir, cat))
        for i, r, how in rows:
            listing.append({'category': cat, 'picked': how, 'locus': f'{r.chrom}:{r.summit - args.locus_flank_bp}-'
                            f'{r.summit + args.locus_flank_bp}', **r.to_dict()})
    pd.DataFrame(listing).to_csv(os.path.join(loci_dir, 'loci.tsv'), sep='\t', index=False)


def load_gene_table(gtf):
    """{chrom: [(start, end, strand, name)]} from the GTF (parse cached next to it)."""
    from cshark.data.splice_features import SpliceSiteAnnotation
    return SpliceSiteAnnotation(gtf, verbose=False).genes


def find_genes(table, names):
    by_name = {}
    for chrom, genes in table.items():
        for s, e, strand, name in genes:
            key = name.lower()
            if key not in by_name or e - s > by_name[key][2] - by_name[key][1]:
                by_name[key] = (chrom, int(s), int(e), strand, name)
    found = [by_name[n.lower()] for n in names if n.lower() in by_name]
    missing = [n for n in names if n.lower() not in by_name]
    if missing:
        print(f'[rad21-ko] WARNING: genes not in the GTF: {missing}')
    return found


MAX_GENE_LANES = 10


def gene_lanes(table, chrom, g0, g1, target):
    """Gene bodies in [g0, g1) packed into lanes so neither bodies nor labels
    overlap -> ([(start, end, strand, name, lane)], n_lanes, n_dropped).  The
    target goes first so it always gets the top lane; genes that would need
    more than MAX_GENE_LANES lanes (gene clusters such as Pcdh) are dropped."""
    genes = sorted((g for g in table.get(chrom, []) if g[1] > g0 and g[0] < g1),
                   key=lambda g: (g[3] != target, g[0]))
    label_bp = 0.0075 * (g1 - g0)                 # ~ width of one label character
    lanes, placed, dropped = [], [], 0             # lanes: occupied (start, end) spans per lane
    for s, e, strand, name in genes:
        span = (max(s, g0), max(min(e, g1), max(s, g0) + len(name) * label_bp))
        lane = next((i for i, occ in enumerate(lanes)
                     if all(span[1] < a or span[0] > b for a, b in occ)), len(lanes))
        if lane >= MAX_GENE_LANES:
            dropped += 1
            continue
        if lane == len(lanes):
            lanes.append([])
        lanes[lane].append(span)
        placed.append((s, e, strand, name, lane))
    return placed, max(1, len(lanes)), dropped


def draw_genes(ax, layout, g0, g1, target):
    """Gene bodies from ``gene_lanes`` on ``ax`` (kb axis), the target in red."""
    placed, n_lanes, dropped = layout
    for s, e, strand, name, lane in placed:
        c = '#d1495b' if name == target else '#444444'
        y = -lane
        ax.plot([max(s, g0) / 1000, min(e, g1) / 1000], [y, y], color=c, lw=3 if name == target else 1.5,
                solid_capstyle='butt')
        tip = e if strand == '+' else s
        if g0 <= tip < g1:
            ax.annotate('', xy=(tip / 1000 + (1 if strand == '+' else -1) * 0.004 * (g1 - g0) / 1000, y),
                        xytext=(tip / 1000, y), arrowprops=dict(arrowstyle='-|>', color=c, lw=1))
        ax.text(max(s, g0) / 1000, y + 0.3, name, fontsize=6.5, color=c,
                fontweight='bold' if name == target else 'normal')
    if dropped:
        ax.text(1.0, -n_lanes + 0.5, f'+{dropped} more genes not drawn', transform=ax.get_yaxis_transform(),
                ha='right', fontsize=6.5, color='#777777')
    ax.set_ylim(-n_lanes + 0.3, 1.0)
    ax.set_yticks([])
    ax.set_ylabel('Genes', fontsize=8, rotation=0, ha='right', va='center')
    for s in ('top', 'right', 'left'):
        ax.spines[s].set_visible(False)


def plot_genes(predictor, genes, table, chrom_sizes, scale_cache, args, label, out_dir, wt_rad21, ko_rad21):
    """One figure per gene: its body +/- --gene-flank-bp, predicted in one window centred on it."""
    gene_dir = os.path.join(out_dir, 'genes')
    os.makedirs(gene_dir, exist_ok=True)
    bin_bp = predictor.bin_bp
    hw = args.peak_half_width_bp // bin_bp
    extra = context_tracks(args)
    listing = []
    for chrom, gs, ge, strand, name in genes:
        clen = chrom_sizes[chrom]
        g0 = max(0, gs - args.gene_flank_bp)
        g1 = min(clen, ge + args.gene_flank_bp)
        if g1 - g0 > WINDOW // 2:            # keep the region inside one window's central half
            mid = (gs + ge) // 2
            g0, g1 = max(0, mid - WINDOW // 4), min(clen, mid + WINDOW // 4)
            print(f'[rad21-ko] {name}: region capped at {(g1 - g0) // 1000} kb around the gene centre.')
        b0, s = predictor.region(chrom, clen, g0, g1)
        n = len(s['pred_wt'])
        if chrom not in scale_cache:         # the KO target's per-chromosome offset
            nb = clen // bin_bp
            wt, ko = binned_bigwig(wt_rad21, chrom, nb, bin_bp), binned_bigwig(ko_rad21, chrom, nb, bin_bp)
            summits = chrom_summits(wt, ko, np.ones(nb, bool), args, bin_bp)
            scale_cache[chrom] = ko_rad21_scale(wt, ko, summits, hw, args.pseudocount)
        s['exp_wt'] = binned_bigwig(wt_rad21, chrom, n, bin_bp, b0)
        s['exp_ko'] = binned_bigwig(ko_rad21, chrom, n, bin_bp, b0) / scale_cache[chrom]
        x = (np.arange(b0, b0 + n) * bin_bp + bin_bp // 2) / 1000
        row_labels = TRACK_ROWS + [(k, lbl) for k, (lbl, _) in extra.items()]
        layout = gene_lanes(table, chrom, b0 * bin_bp, (b0 + n) * bin_bp, name)
        gene_h = 0.25 + 0.22 * layout[1]
        fig, axes = plt.subplots(len(row_labels) + 1, 1, figsize=(12, 1.15 * len(row_labels) + gene_h + 1.2),
                                 sharex=True, gridspec_kw={'height_ratios': [1] * len(row_labels) + [gene_h / 1.15]})
        series = track_series(s, 0, n)
        for k, (_, p) in extra.items():
            series[k] = [(binned_bigwig(p, chrom, n, bin_bp, b0), COLORS['ctcf_wt'], None)]
        draw_tracks(axes[:-1], x, series, row_labels, legend=True)
        for ax in axes[:-1]:
            ax.axvspan(gs / 1000, ge / 1000, color='#f5a623', alpha=0.12, lw=0)
        draw_genes(axes[-1], layout, b0 * bin_bp, (b0 + n) * bin_bp, name)
        axes[-1].ticklabel_format(axis='x', useOffset=False, style='plain')
        axes[-1].set_xlabel(f'{chrom} (kb)', fontsize=9)
        held = chrom in HELD_OUT
        fig.suptitle(f'{name}  {chrom}:{gs:,}-{ge:,} ({strand})  [{label}]\n'
                     + ('held-out chromosome' if held else 'NOTE: the model was trained on this chromosome')
                     + '; experimental $\\Delta$CTCF RAD21 on control scale; shaded = gene body', fontsize=10)
        fig.tight_layout(rect=(0, 0, 1, 0.95))
        fig.savefig(os.path.join(gene_dir, f'{name}.png'), dpi=200)
        fig.savefig(os.path.join(gene_dir, f'{name}.pdf'))
        plt.close(fig)
        row = {'gene': name, 'chrom': chrom, 'start': gs, 'end': ge, 'strand': strand,
               'plotted': f'{chrom}:{b0 * bin_bp}-{(b0 + n) * bin_bp}', 'held_out_chrom': held,
               'ko_rad21_scale': scale_cache[chrom]}
        body = slice((gs - b0 * bin_bp) // bin_bp, (ge - b0 * bin_bp) // bin_bp + 1)
        for k in ('exp_wt', 'exp_ko', *(f'pred_{a}' for a in ('wt', *ARMS) if f'pred_{a}' in s)):
            row[f'{k}_gene_body_mean'] = float(np.mean(s[k][body]))
        listing.append(row)
        print(f'[rad21-ko] gene {name} ({chrom}:{gs:,}-{ge:,}) -> {gene_dir}/{name}.png')
    pd.DataFrame(listing).to_csv(os.path.join(gene_dir, 'genes.tsv'), sep='\t', index=False)


# ---------------------------------------------------------------------------
def main():
    args = build_parser().parse_args()
    if args.genes_only and not args.genes:
        raise SystemExit('--genes-only needs --genes.')
    name = os.path.splitext(os.path.basename(args.checkpoint))[0]
    label = args.label or name
    out_dir = args.out_dir or os.path.join('benchmarks', 'rad21_ctcf_ko', name)
    os.makedirs(out_dir, exist_ok=True)
    seq_dir = args.seq_dir or os.path.join(os.path.dirname(args.wt_dir.rstrip('/')), 'dna_sequence')
    device = torch.device(args.device or ('cuda' if torch.cuda.is_available() else 'cpu'))

    model, all_tracks, rad21_idx, _ = load_hierarchical_rad21_predictor(args.checkpoint, device=device)
    model.eval()
    hp = torch.load(args.checkpoint, map_location='cpu', weights_only=False)['hyper_parameters']
    in_feats = [f for f in hp['input_features'] if f != 'rad21']
    log_tracks = hp.get('bigwig_log_transform', True)
    n_out = hp.get('target_1d_size', 8192)
    bin_bp = WINDOW // n_out
    feat_dir = os.path.join(args.wt_dir, 'genomic_features')
    track_paths = [os.path.join(feat_dir, f'{f}.bw') for f in in_feats]
    wt_rad21 = os.path.join(feat_dir, 'rad21.bw')
    ko_rad21 = os.path.join(args.ko_dir, 'genomic_features', args.ko_rad21)
    measured = None
    if args.measured_ctcf:
        measured = MeasuredCtcfKO(args.wt_ctcf_raw, args.ko_ctcf_raw, os.path.join(feat_dir, 'ctcf.bw'),
                                  args.ctcf_ko_scaling)
    arms = [a for a in ARMS if a == 'ko' or measured is not None]
    predictor = Predictor(model, track_paths, in_feats.index('ctcf'), log_tracks, n_out, seq_dir, args, device,
                          measured)
    print(f'[rad21-ko] {name}: inputs {in_feats}, sequence '
          f'{"ignored" if hp.get("layer2_no_seq", False) else "used"}, {bin_bp} bp bins; KO inputs: '
          f'simulated ({args.ctcf_ko_mode!r}, peak height {args.ctcf_ko_peak_height})'
          + (f', measured ({args.ctcf_ko_scaling} scaling)' if measured else '') + '.')
    with pyBigWig.open(wt_rad21) as bw:
        chrom_sizes = bw.chroms()
    scale_cache = {}

    if not args.genes_only:
        benchmark(predictor, arms, args, label, out_dir, chrom_sizes, scale_cache, wt_rad21, ko_rad21,
                  in_feats, hp, measured)
    if args.genes:
        table = load_gene_table(args.gtf)
        plot_genes(predictor, find_genes(table, args.genes), table, chrom_sizes, scale_cache, args, label,
                   out_dir, wt_rad21, ko_rad21)


def benchmark(predictor, arms, args, label, out_dir, chrom_sizes, scale_cache, wt_rad21, ko_rad21,
              in_feats, hp, measured):
    bin_bp = predictor.bin_bp
    hw = args.peak_half_width_bp // bin_bp
    pc = args.pseudocount
    rows, store = [], {}
    bin_stats = {k: ([], []) for k in ('wt', *arms)}
    for chrom in args.chroms:
        pred = predictor.chrom(chrom, chrom_sizes[chrom])
        n_bins = len(pred['pred_wt'])
        exp_wt = binned_bigwig(wt_rad21, chrom, n_bins, bin_bp)
        exp_ko = binned_bigwig(ko_rad21, chrom, n_bins, bin_bp)
        covered = np.isfinite(pred['pred_wt'])
        for a in ('wt', *arms):
            bin_stats[a][0].append((exp_wt if a == 'wt' else exp_ko)[covered])
            bin_stats[a][1].append(pred[f'pred_{a}'][covered])

        summits = chrom_summits(exp_wt, exp_ko, covered, args, bin_bp)
        d = pd.DataFrame({'chrom': chrom, 'summit': summits * bin_bp + bin_bp // 2})
        d['exp_wt'] = peak_strength(exp_wt, summits, hw)
        d['exp_ko'] = peak_strength(exp_ko, summits, hw)
        for k, v in pred.items():
            d[k] = peak_strength(v, summits, hw)
        d['exp_lfc_raw'] = np.log2((d.exp_ko + pc) / (d.exp_wt + pc))
        # Per-chromosome scaling offset of the KO target (see module docstring).
        scale_cache[chrom] = float(2 ** d.exp_lfc_raw.median())
        d['exp_lfc'] = d.exp_lfc_raw - d.exp_lfc_raw.median()
        d['exp_ko_wt_scale'] = d.exp_ko / scale_cache[chrom]
        for a in arms:
            raw = np.log2((d[f'pred_{a}'] + pc) / (d.pred_wt + pc))
            d[f'pred_lfc_raw_{a}'] = raw
            d[f'pred_lfc_{a}'] = raw - raw.median()
        if args.loci_per_class > 0:      # per-bin tracks for the locus gallery
            store[chrom] = dict(pred, exp_wt=exp_wt, exp_ko=exp_ko / scale_cache[chrom])
        print(f'[rad21-ko] {chrom}: {len(d)} peaks; median raw log2 KO/WT exp {d.exp_lfc_raw.median():+.2f}, '
              + ', '.join(f'pred ({a}) {d[f"pred_lfc_raw_{a}"].median():+.2f}' for a in arms))
        rows.append(d)

    df = pd.concat(rows, ignore_index=True)
    t = args.lfc_threshold
    df['exp_class'] = np.where(df.exp_lfc <= -t, 'lost', np.where(df.exp_lfc >= t, 'gained', 'unchanged'))
    for a in arms:
        df[f'pred_score_lost_{a}'] = -df[f'pred_lfc_{a}']
        df[f'pred_score_gained_{a}'] = df[f'pred_lfc_{a}']
        df[f'ctcf_removed_{a}'] = df.ctcf_wt - df[f'ctcf_{a}']
        df[f'pred_class_{a}'] = predicted_classes(df, a)

    def bins(k):
        e, p = (np.concatenate(x) for x in bin_stats[k])
        return {'pearson_log1p': float(pearsonr(np.log1p(p), np.log1p(e))[0]),
                'spearman': float(spearmanr(p[::7], e[::7])[0]), 'n_bins': int(len(e))}

    lost = (df.exp_class == 'lost').values
    gained = (df.exp_class == 'gained').values
    m = {
        'checkpoint': args.checkpoint, 'label': label, 'input_features': in_feats,
        'uses_sequence': not hp.get('layer2_no_seq', False), 'chroms': args.chroms, 'bin_bp': bin_bp,
        'n_peaks': int(len(df)), 'settings': {k: v for k, v in vars(args).items() if k != 'checkpoint'},
        'ko_rad21_scale': {c: scale_cache[c] for c in args.chroms},
        'measured_ctcf_ko_normalisation': measured.stats if measured else None,
        'wt_bins': bins('wt'),
        'wt_peaks': corr_block(df.pred_wt.values, df.exp_wt.values),
        # The model predicting "nothing changes" -- what KO R^2 has to beat.
        'ko_peaks_no_change_baseline': corr_block(df.pred_wt.values, df.exp_ko_wt_scale.values),
        'arms': {},
    }
    for a in arms:
        m['arms'][a] = {
            'description': ARMS[a][1],
            'ko_bins': bins(a),
            'ko_peaks': corr_block(df[f'pred_{a}'].values, df.exp_ko_wt_scale.values),
            'ko_peaks_unscaled': corr_block(df[f'pred_{a}'].values, df.exp_ko.values),
            'log2fc': {'pearson': float(pearsonr(df[f'pred_lfc_{a}'], df.exp_lfc)[0]),
                       'spearman': float(spearmanr(df[f'pred_lfc_{a}'], df.exp_lfc)[0]),
                       'r2': r2(df[f'pred_lfc_{a}'], df.exp_lfc)},
            'lost_roc': roc_block(lost, df[f'pred_score_lost_{a}'].values),
            'lost_roc_ctcf_removed_baseline': roc_block(lost, df[f'ctcf_removed_{a}'].values),
            'gained_roc': roc_block(gained, df[f'pred_score_gained_{a}'].values),
        }
    df.to_csv(os.path.join(out_dir, 'peaks.tsv'), sep='\t', index=False)
    with open(os.path.join(out_dir, 'metrics.json'), 'w') as f:
        json.dump(m, f, indent=2)
    for a in arms:
        plot_benchmark(df, m, a, label, os.path.join(out_dir, f'fig_benchmark{ARMS[a][0]}'))
        plot_log2fc(df, m, a, label, os.path.join(out_dir, f'fig_log2fc{ARMS[a][0]}'))
    if args.loci_per_class > 0:
        plot_loci(df, store, label, args, out_dir, bin_bp)

    fmt = lambda b: 'n/a' if b is None else f'AUC {b["auc"]:.3f}, AUPRC {b["auprc"]:.3f} (n={b["n_pos"]})'
    print(f'\n[rad21-ko] {label}: {len(df)} peaks on {args.chroms}')
    print(f'  WT peaks   R2 {m["wt_peaks"]["r2"]:.3f}  Spearman {m["wt_peaks"]["spearman"]:.3f}  '
          f'| bins Pearson(log1p) {m["wt_bins"]["pearson_log1p"]:.3f}')
    print(f'  KO no-change baseline R2 {m["ko_peaks_no_change_baseline"]["r2"]:.3f}')
    for a in arms:
        x = m['arms'][a]
        print(f'  [{ARMS[a][1]}]')
        print(f'    KO peaks   R2 {x["ko_peaks"]["r2"]:.3f}  Spearman {x["ko_peaks"]["spearman"]:.3f}')
        print(f'    log2FC     Pearson {x["log2fc"]["pearson"]:.3f}  Spearman {x["log2fc"]["spearman"]:.3f}')
        print(f'    lost ROC   model {fmt(x["lost_roc"])} | CTCF-removed baseline '
              f'{fmt(x["lost_roc_ctcf_removed_baseline"])}')
        print(f'    gained ROC model {fmt(x["gained_roc"])}')
    print(f'  -> {out_dir}')


if __name__ == '__main__':
    main()
