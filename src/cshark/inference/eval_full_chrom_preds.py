#!/usr/bin/env python
"""
Evaluate full-chromosome predictions from perturb.py against a ground-truth
reference contact map.

Prediction TSV (from perturb.py full-chrom mode):
    chrom1  start1  end1  a1  chrom2  start2  end2  a2  WT  KO

Ground-truth reference TSV / BED-like:
    chrom1  start1  end1  chrom2  start2  end2  count

The script:
  1. Loads both files.
  2. Aligns the reference to the prediction resolution by binning
     (floor-dividing genomic coordinates by the prediction resolution).
  3. Merges on (chrom1, bin1, chrom2, bin2) pairs.
  4. Reports global metrics (Pearson R, Spearman rho, MSE, MAE, etc.).
  5. Reports per-1 Mb-region Pearson R values.
  6. Optionally writes outputs to TSV files.
"""

from __future__ import annotations

import argparse
import sys
import os
import warnings

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr
from tqdm import tqdm


# ──────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────

def _bin_coord(coord: int, resolution: int) -> int:
    """Floor-align a genomic coordinate to the nearest bin start."""
    return (coord // resolution) * resolution


def load_predictions(path: str) -> pd.DataFrame:
    """Load the prediction TSV produced by perturb.py full-chrom mode."""
    df = pd.read_csv(path, sep='\t')
    if 'pred' in df.columns:
        # some perturb.py versions had a single 'pred' column instead of 'WT'/'KO'
        df.rename(columns={'pred': 'WT'}, inplace=True)
    required = {'chrom1', 'start1', 'end1', 'chrom2', 'start2', 'end2', 'WT'}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Prediction file is missing columns: {missing}")
    return df


def load_reference(path: str, chrom: str | None = None,
                   chunksize: int = 500_000, has_header: bool = True) -> pd.DataFrame:
    """Load the ground-truth reference file, optionally filtering to a single
    chromosome during reading so the full file never lives in memory.

    Accepts either:
      - A header-ful TSV with columns chrom1 start1 end1 chrom2 start2 end2 count
      - A headerless 7-column BED-like file (same column order)

    Parameters
    ----------
    path : str
        Path to the reference file.
    chrom : str or None
        If provided, only rows where *both* chrom1 and chrom2 equal this value
        are kept.  Reading is done in chunks so only matching rows accumulate
        in memory.
    chunksize : int
        Number of rows per chunk when streaming (default 500 000).
    """
    col_names = ['chrom1', 'start1', 'end1',
                 'chrom2', 'start2', 'end2', 'count']


    read_kw: dict = dict(sep='\t')
    if has_header:
        read_kw['header'] = 0          # use the file's own header row
    else:
        read_kw['header'] = None
        read_kw['names'] = col_names

    n_rows = sum(1 for _ in open(path)) - (1 if has_header else 0)
    print(f'  Reference file has {n_rows:,} rows ({"with" if has_header else "no"} header).')
    n_chunks = (n_rows + chunksize - 1) // chunksize

    # Stream in chunks, keeping only the target chromosome
    if chrom is not None:
        chunks: list[pd.DataFrame] = []
        for chunk in tqdm(pd.read_csv(path, chunksize=chunksize, **read_kw), desc='Loading reference', unit='chunk', total=n_chunks):
            filtered = chunk[(chunk['chrom1'] == chrom) &
                            (chunk['chrom2'] == chrom)]
            if len(filtered) > 0:
                chunks.append(filtered)
        if chunks:
            df = pd.concat(chunks, ignore_index=True)
        else:
            df = pd.DataFrame(columns=col_names)
    else:
        df = pd.read_csv(path, **read_kw)

    required = {'chrom1', 'start1', 'end1', 'chrom2', 'start2', 'end2', 'count'}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Reference file is missing columns: {missing}")
    return df


def detect_resolution(df: pd.DataFrame) -> int:
    """Infer the resolution from a prediction dataframe by looking at
    the most common (end1 - start1) value."""
    diffs = (df['end1'] - df['start1']).value_counts()
    return int(diffs.index[0])


def align_reference_to_resolution(ref: pd.DataFrame, resolution: int, val_col: str) -> pd.DataFrame:
    """Bin reference coordinates to the prediction resolution.

    For each reference pair, compute bin-start = floor(start / res) * res.
    If the reference has a finer resolution, multiple entries may map to
    the same bin pair – we aggregate by mean.
    """
    ref = ref.copy()
    ref['bin_start1'] = ref['start1'].apply(lambda x: _bin_coord(x, resolution))
    ref['bin_end1'] = ref['bin_start1'] + resolution
    ref['bin_start2'] = ref['start2'].apply(lambda x: _bin_coord(x, resolution))
    ref['bin_end2'] = ref['bin_start2'] + resolution

    # Aggregate duplicate bin pairs (finer-resolution reference)
    agg = (ref.groupby(['chrom1', 'bin_start1', 'bin_end1',
                        'chrom2', 'bin_start2', 'bin_end2'])
              .agg(count=(val_col, 'mean'))
              .reset_index())
    return agg


def align_prediction_to_resolution(pred: pd.DataFrame, resolution: int,
                                   value_col: str = 'WT') -> pd.DataFrame:
    """Re-bin prediction coordinates to a (coarser) target resolution.

    Each prediction pair is floor-binned to *resolution* and duplicate
    bin-pairs are aggregated by mean over WT (and KO if present).
    """
    pred = pred.copy()
    pred['start1'] = pred['start1'].apply(lambda x: _bin_coord(x, resolution))
    pred['end1'] = pred['start1'] + resolution
    pred['start2'] = pred['start2'].apply(lambda x: _bin_coord(x, resolution))
    pred['end2'] = pred['start2'] + resolution

    # Determine which value columns to aggregate
    agg_dict: dict[str, tuple[str, str]] = {value_col: (value_col, 'mean')}
    if 'KO' in pred.columns and value_col != 'KO':
        agg_dict['KO'] = ('KO', 'mean')

    group_cols = ['chrom1', 'start1', 'end1', 'chrom2', 'start2', 'end2']
    agg = pred.groupby(group_cols).agg(**agg_dict).reset_index()
    return agg


def merge_pred_ref(pred: pd.DataFrame, ref_aligned: pd.DataFrame,
                   value_col: str = 'WT') -> pd.DataFrame:
    """Merge predictions with aligned reference on binned coordinates."""
    merged = pred.merge(
        ref_aligned,
        left_on=['chrom1', 'start1', 'end1', 'chrom2', 'start2', 'end2'],
        right_on=['chrom1', 'bin_start1', 'bin_end1', 'chrom2', 'bin_start2', 'bin_end2'],
        how='inner',
        suffixes=('', '_ref'),
    )
    return merged


def compute_metrics(pred_vals: np.ndarray, ref_vals: np.ndarray,
                    label: str = '') -> dict:
    """Compute a suite of comparison metrics between two arrays."""
    mask = np.isfinite(pred_vals) & np.isfinite(ref_vals)
    p = pred_vals[mask]
    r = ref_vals[mask]
    n = len(p)
    metrics = {'label': label, 'n_pairs': n}
    if n < 3:
        metrics.update({
            'pearson_r': np.nan, 'pearson_p': np.nan,
            'spearman_rho': np.nan, 'spearman_p': np.nan,
            'mse': np.nan, 'mae': np.nan,
            'mean_pred': np.nan, 'mean_ref': np.nan,
        })
        return metrics

    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        pr, pp = pearsonr(p, r)
        sr, sp = spearmanr(p, r)

    mse = float(np.mean((p - r) ** 2))
    mae = float(np.mean(np.abs(p - r)))
    metrics.update({
        'pearson_r': pr, 'pearson_p': pp,
        'spearman_rho': sr, 'spearman_p': sp,
        'mse': mse, 'mae': mae,
        'mean_pred': float(np.mean(p)),
        'mean_ref': float(np.mean(r)),
    })
    return metrics


def compute_distance_stratified_metrics(merged: pd.DataFrame,
                                        value_col: str = 'WT',
                                        resolution: int = 8192,
                                        n_bins: int = 10) -> pd.DataFrame:
    """Compute Pearson R stratified by genomic distance between loci."""
    merged = merged.copy()
    merged['distance'] = np.abs(merged['start2'] - merged['start1'])
    max_dist = merged['distance'].max()
    if max_dist == 0:
        return pd.DataFrame()

    bin_edges = np.linspace(0, max_dist + 1, n_bins + 1)
    merged['dist_bin'] = pd.cut(merged['distance'], bins=bin_edges, right=False,
                                labels=[f'{int(bin_edges[i])}-{int(bin_edges[i+1])}'
                                        for i in range(n_bins)])
    rows = []
    for dist_label, grp in merged.groupby('dist_bin', observed=True):
        m = compute_metrics(grp[value_col].values, grp['count'].values,
                            label=str(dist_label))
        m['distance_bin'] = str(dist_label)
        rows.append(m)
    return pd.DataFrame(rows)


# ──────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description='Evaluate full-chromosome Hi-C predictions against a '
                    'ground-truth reference contact map.')

    parser.add_argument('--pred', dest='pred_path', required=True,
                        help='Path to the prediction .tsv from perturb.py '
                             'full-chromosome mode.')
    parser.add_argument('--ref', dest='ref_path', required=True,
                        help='Path to the ground-truth reference file '
                             '(chrom1 start1 end1 chrom2 start2 end2 count).')
    parser.add_argument('--value-col', dest='value_col', default='WT',
                        choices=['WT', 'KO'],
                        help='Which prediction column to evaluate '
                             '(default: WT).')
    parser.add_argument('--resolution', dest='resolution', type=int,
                        default=None,
                        help='Target resolution (bp) for comparison. Both '
                             'predictions and reference are binned to this '
                             'resolution before evaluation. If not set, the '
                             'native prediction resolution is used.')
    parser.add_argument('--region-size', dest='region_size', type=int,
                        default=1_000_000,
                        help='Size (bp) of non-overlapping regions for '
                             'per-region metrics (default: 1000000).')
    parser.add_argument('--log-transform', dest='log_transform',
                        action='store_true',
                        help='Apply log1p transform to both prediction and '
                             'reference values before computing metrics.')
    parser.add_argument('--min-count', dest='min_count', type=float,
                        default=0.0,
                        help='Minimum reference count to include a pair '
                             '(filters very low / zero entries).')
    parser.add_argument('--max-distance', dest='max_distance', type=int,
                        default=None,
                        help='Maximum genomic distance (bp) between loci to '
                             'include. Useful for filtering very long-range '
                             'pairs that are mostly noise.')
    parser.add_argument('--distance-bins', dest='distance_bins', type=int,
                        default=10,
                        help='Number of distance bins for stratified metrics '
                             '(default: 10).')
    parser.add_argument('--out', dest='out_prefix', default=None,
                        help='Output prefix for results files. If not given, '
                             'results are printed to stdout only.')

    args = parser.parse_args(args=None if sys.argv[1:] else ['--help'])

    # ── Load data ─────────────────────────────────────────────────────
    print('Loading predictions …')
    pred = load_predictions(args.pred_path)
    print(f'  {len(pred):,} prediction pairs loaded.')

    # Identify the target chromosome from the predictions (cis-only: single chrom)
    pred_chroms = pred['chrom1'].unique()
    target_chrom = pred_chroms[0] if len(pred_chroms) == 1 else None
    if target_chrom is not None:
        print(f'  Target chromosome: {target_chrom}')
    else:
        print(f'  Multiple chromosomes in predictions: {list(pred_chroms)}')

    print('Loading reference (streaming, chromosome-filtered) …')
    ref = load_reference(args.ref_path, chrom=target_chrom)
    print(f'  {len(ref):,} reference pairs loaded.')

    # ── Detect / set target resolution ─────────────────────────────────
    pred_res = detect_resolution(pred)
    print(f'  Detected prediction resolution: {pred_res:,} bp')
    target_res = args.resolution if args.resolution is not None else pred_res
    if target_res < pred_res:
        print(f'  ⚠  Requested resolution ({target_res:,} bp) is finer than the '
              f'prediction resolution ({pred_res:,} bp). Using prediction '
              f'resolution instead.')
        target_res = pred_res
    if target_res != pred_res:
        print(f'  Coarsening predictions from {pred_res:,} bp → {target_res:,} bp …')
        pred = align_prediction_to_resolution(pred, target_res,
                                              value_col=args.value_col)
        print(f'  {len(pred):,} prediction pairs after coarsening.')

    # ── Align reference to target resolution ──────────────────────────
    print(f'Aligning reference to target resolution ({target_res:,} bp) …')
    ref_aligned = align_reference_to_resolution(ref, target_res, val_col='count')
    print(f'  {len(ref_aligned):,} reference bin-pairs after alignment.')

    # ── Merge ─────────────────────────────────────────────────────────
    merged = merge_pred_ref(pred, ref_aligned, value_col=args.value_col)
    print(f'  {len(merged):,} pairs after inner merge.')

    if len(merged) == 0:
        print('\n⚠  No overlapping pairs found. Check that chromosome names, '
              'coordinates, and resolutions are compatible.')
        sys.exit(1)

    # ── Optional filters ──────────────────────────────────────────────
    if args.min_count > 0:
        before = len(merged)
        merged = merged[merged['count'] >= args.min_count].reset_index(drop=True)
        print(f'  Filtered by min_count={args.min_count}: '
              f'{before:,} → {len(merged):,} pairs.')

    if args.max_distance is not None:
        merged['_dist'] = np.abs(merged['start2'] - merged['start1'])
        before = len(merged)
        merged = merged[merged['_dist'] <= args.max_distance].reset_index(drop=True)
        merged.drop(columns=['_dist'], inplace=True)
        print(f'  Filtered by max_distance={args.max_distance:,}: '
              f'{before:,} → {len(merged):,} pairs.')

    # ── Optional log transform ────────────────────────────────────────
    pred_vals = merged[args.value_col].values.astype(float)
    ref_vals = merged['count'].values.astype(float)
    if args.log_transform:
        pred_vals = np.log1p(pred_vals)
        ref_vals = np.log1p(ref_vals)

    # ── Global metrics ────────────────────────────────────────────────
    print('\n' + '=' * 70)
    print('GLOBAL CHROMOSOME-WIDE METRICS')
    print('=' * 70)
    global_metrics = compute_metrics(pred_vals, ref_vals, label='global')
    for k, v in global_metrics.items():
        if k == 'label':
            continue
        if isinstance(v, float):
            print(f'  {k:20s} = {v:.6f}')
        else:
            print(f'  {k:20s} = {v}')

    # ── Per-region metrics ────────────────────────────────────────────
    region_size = args.region_size
    print(f'\n{"=" * 70}')
    print(f'PER-{region_size // 1_000_000}Mb REGION METRICS  '
          f'(Pearson R for each non-overlapping region)')
    print('=' * 70)

    # Assign each pair to a region based on the midpoint of its two loci
    merged['mid'] = (merged['start1'] + merged['start2']) / 2.0
    merged['region_start'] = (merged['mid'] // region_size).astype(int) * region_size
    merged['region_end'] = merged['region_start'] + region_size

    # Also compute per-region using anchor1 position for within-region contacts
    merged['region_start_a1'] = (merged['start1'] // region_size).astype(int) * region_size

    region_rows = []
    for (chrom, rstart), grp in merged.groupby(
            [merged['chrom1'], merged['region_start']]):
        p = grp[args.value_col].values.astype(float)
        r = grp['count'].values.astype(float)
        if args.log_transform:
            p = np.log1p(p)
            r = np.log1p(r)
        m = compute_metrics(p, r,
                            label=f'{chrom}:{int(rstart)}-{int(rstart + region_size)}')
        m['chrom'] = chrom
        m['region_start'] = int(rstart)
        m['region_end'] = int(rstart + region_size)
        region_rows.append(m)

    region_df = pd.DataFrame(region_rows)
    region_df.sort_values(['chrom', 'region_start'], inplace=True)

    # Print summary table
    print(f'\n  {"Region":<35s} {"N pairs":>10s} {"Pearson R":>12s} '
          f'{"Spearman ρ":>12s} {"MSE":>12s}')
    print('  ' + '-' * 85)
    for _, row in region_df.iterrows():
        rstr = f'{row["chrom"]}:{int(row["region_start"]):,}-{int(row["region_end"]):,}'
        print(f'  {rstr:<35s} {int(row["n_pairs"]):>10,} '
              f'{row["pearson_r"]:>12.4f} '
              f'{row["spearman_rho"]:>12.4f} '
              f'{row["mse"]:>12.6f}')

    # Summary statistics across regions
    valid_r = region_df['pearson_r'].dropna()
    if len(valid_r) > 0:
        print(f'\n  Regions evaluated:   {len(valid_r)}')
        print(f'  Mean Pearson R:      {valid_r.mean():.4f}')
        print(f'  Median Pearson R:    {valid_r.median():.4f}')
        print(f'  Std Pearson R:       {valid_r.std():.4f}')
        print(f'  Min Pearson R:       {valid_r.min():.4f}  '
              f'({region_df.loc[valid_r.idxmin(), "label"]})')
        print(f'  Max Pearson R:       {valid_r.max():.4f}  '
              f'({region_df.loc[valid_r.idxmax(), "label"]})')

    # ── Distance-stratified metrics ───────────────────────────────────
    print(f'\n{"=" * 70}')
    print('DISTANCE-STRATIFIED METRICS')
    print('=' * 70)

    merged_for_dist = merged.copy()
    if args.log_transform:
        merged_for_dist[args.value_col] = np.log1p(
            merged_for_dist[args.value_col].astype(float))
        merged_for_dist['count'] = np.log1p(
            merged_for_dist['count'].astype(float))

    dist_df = compute_distance_stratified_metrics(
        merged_for_dist, value_col=args.value_col,
        resolution=target_res, n_bins=args.distance_bins)

    if len(dist_df) > 0:
        print(f'\n  {"Distance bin (bp)":<35s} {"N pairs":>10s} '
              f'{"Pearson R":>12s} {"Spearman ρ":>12s}')
        print('  ' + '-' * 73)
        for _, row in dist_df.iterrows():
            print(f'  {row["distance_bin"]:<35s} '
                  f'{int(row["n_pairs"]):>10,} '
                  f'{row["pearson_r"]:>12.4f} '
                  f'{row["spearman_rho"]:>12.4f}')

    # ── Save outputs ──────────────────────────────────────────────────
    if args.out_prefix is not None:
        os.makedirs(os.path.dirname(args.out_prefix) or '.', exist_ok=True)

        # Global metrics
        global_out = f'{args.out_prefix}_global_metrics.tsv'
        pd.DataFrame([global_metrics]).to_csv(global_out, sep='\t', index=False)
        print(f'\n  Wrote global metrics   → {global_out}')

        # Per-region metrics
        region_out = f'{args.out_prefix}_region_metrics.tsv'
        region_df.to_csv(region_out, sep='\t', index=False)
        print(f'  Wrote per-region metrics → {region_out}')

        # Distance-stratified metrics
        if len(dist_df) > 0:
            dist_out = f'{args.out_prefix}_distance_metrics.tsv'
            dist_df.to_csv(dist_out, sep='\t', index=False)
            print(f'  Wrote distance metrics   → {dist_out}')

        # Merged data (for downstream analysis)
        merged_out = f'{args.out_prefix}_merged.tsv'
        out_cols = ['chrom1', 'start1', 'end1', 'chrom2', 'start2', 'end2',
                    args.value_col, 'count']
        if 'KO' in merged.columns and args.value_col != 'KO':
            out_cols.insert(-1, 'KO')
        merged[out_cols].to_csv(merged_out, sep='\t', index=False)
        print(f'  Wrote merged pairs       → {merged_out}')

        # plot histogram of region r values
        try:
            import matplotlib.pyplot as plt
            plt.figure(figsize=(6, 4))
            plt.hist(region_df['pearson_r'].dropna(), bins=20, edgecolor='k')
            plt.xlabel('Pearson R across regions')
            plt.ylabel('Number of regions')
            plt.title('Distribution of per-region Pearson R')
            hist_out = f'{args.out_prefix}_region_r_histogram.png'
            plt.savefig(hist_out, dpi=300)
            plt.close()
            print(f'  Wrote region R histogram → {hist_out}')
        except ImportError:
            print('  matplotlib not available, skipping histogram plot.')

    print('\nDone.')


if __name__ == '__main__':
    main()
