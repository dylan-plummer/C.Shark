"""Full-chromosome haplotype runner (function #2 at chromosome scale).

Chromosome-wide counterpart of ``scopes/allele_split.py:run_allele_haplotype``:
slide 2Mb windows across the chromosome and, for every window, split the WT bulk
tracks into maternal/paternal alleles using the PROVIDED whole-genome prediction
bigwigs (``--maternal-pred`` / ``--paternal-pred``), redistribute RAD21 via the
hierarchical predictor per allele, then predict Hi-C twice -- once per allele,
each with its own haplotype sequence (``--maternal-seq`` / ``--paternal-seq``).

Why the predictions are PROVIDED rather than computed here: the backbone output
depends only on the sequence, not on the sliding-window layout, so running
Enformer/AlphaGenome inside the loop both duplicates work (windows overlap 50%)
and is prohibitively slow -- measured 85.2 s per window for the two sequences,
i.e. ~70 h genome-wide for the backbone alone. Precomputing it once per haplotype
turns that into a bigwig read.

Deliberately a separate module from ``scopes/full_chrom.py``: that one is the
verified CTCF-KO production path and is left untouched. The sliding-window setup
and pixel-aggregation shape below mirror it; the redistribution steps reuse
``redistribute_from_provided_preds`` and ``_redistribute_rad21_alleles`` verbatim
from the single-locus path, so both scales run the same math.
"""
import os

import numpy as np
import pandas as pd
import cooler
import seaborn as sns
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from tqdm import tqdm

from cshark.data.data_feature import GenomicFeature, SequenceFeature
import cshark.inference.utils.inference_utils as infer
from cshark.inference.utils.inference_utils import write_tmp_cooler
from cshark.inference.utils.hierarchical_utils import (
    load_hierarchical_rad21_predictor, predict_rad21,
)
from cshark.perturb.config import WINDOW
from cshark.perturb.models.base import CSharkModel
from cshark.perturb.models.enformer import redistribute_from_provided_preds
from cshark.perturb.scopes.allele_split import _redistribute_rad21_alleles

LABEL_A, LABEL_B = 'maternal', 'paternal'


def _bin_mean(arr, n_bins):
    """Mean-pool a bp-resolution 1D array down to ``n_bins`` (edge-padded)."""
    arr = np.asarray(arr, dtype=float).ravel()
    if len(arr) == n_bins:
        return arr
    bin_size = max(1, len(arr) // n_bins)
    usable = bin_size * n_bins
    if usable > len(arr):
        arr = np.pad(arr, (0, usable - len(arr)), mode='edge')
    return arr[:usable].reshape(n_bins, bin_size).mean(axis=1)


def _track_by_name(name, ctcf, atac, others, input_track_names):
    """Return the in-memory array for ``name`` out of a (ctcf, atac, others) set."""
    if name == 'ctcf':
        return ctcf
    if name == 'atac':
        return atac
    offset = sum(1 for t in ('ctcf', 'atac') if t in input_track_names)
    other_names = input_track_names[offset:]
    if others is not None and name in other_names:
        return others[other_names.index(name)]
    return None


def run_full_chrom_haplotype(cfg):
    args = cfg
    window = WINDOW
    res = cfg.resolution
    image_scale = cfg.mat_size
    cap = cfg.enformer_delta_cap
    chr_name = args.chr_name

    # --- guards -----------------------------------------------------------
    if not cfg.maternal_seq_path or not cfg.paternal_seq_path:
        raise SystemExit('[allele-haplotype] requires --maternal-seq and --paternal-seq.')
    shared_pred = sorted(set(cfg.maternal_pred or {}) & set(cfg.paternal_pred or {}))
    if not shared_pred:
        raise SystemExit(
            '[allele-haplotype] no track has BOTH --maternal-pred and --paternal-pred, so the two '
            'alleles would be identical apart from their sequences. Supply matching '
            'track=bigwig pairs, e.g. --maternal-pred ctcf=<mat>.bw --paternal-pred ctcf=<pat>.bw'
        )

    if args.ctcf_path is not None:
        chr_length = GenomicFeature(args.ctcf_path, 'bw').length(chr_name)
    else:
        chr_length = len(SequenceFeature(path=os.path.join(args.seq_path, f'{chr_name}.fa.gz')))
    if args.out_file is None:
        args.out_file = os.path.join(args.output_path,
                                     f'{args.outname}_{args.celltype}_{chr_name}_full_chr_haplotype.tsv')
    os.makedirs(args.output_path, exist_ok=True)
    os.makedirs('tmp', exist_ok=True)
    print(f'Chromosome length: {chr_length}')

    # --- window layout (same as run_full_chrom) ---------------------------
    region = args.region
    region_start = region_end = None
    step_size = int(window / args.n_overlap_preds)
    if region is not None:
        if ':' in region:
            chr_name, region = region.split(':')
        region_start, region_end = (int(v) for v in region.split('-'))
        starts = np.arange(region_start - step_size, region_end + step_size, step_size)
    else:
        starts = np.arange(0, chr_length - window, step_size)
    ends = starts + window

    # --- input tracks ------------------------------------------------------
    input_track_names, input_track_paths = [], []
    if args.ctcf_path is not None:
        input_track_names.append('ctcf'); input_track_paths.append(args.ctcf_path)
    if args.atac_path is not None:
        input_track_names.append('atac'); input_track_paths.append(args.atac_path)
    if cfg.other_feats is not None:
        for feat in cfg.other_feats:
            input_track_names.append(os.path.basename(feat).split('.')[0])
            input_track_paths.append(feat)

    # --- hierarchical RAD21 predictor -------------------------------------
    hierarchical_rad21_model = None
    fill_rad21 = False
    rad21_insert_other_pos = None
    if cfg.hierarchical_model_path is not None:
        hierarchical_rad21_model, _, _, _ = load_hierarchical_rad21_predictor(cfg.hierarchical_model_path)
        if 'rad21' not in input_track_names:
            # Predict + insert rad21 each window. Done inline (not via
            # prepare_rad21_input) because that helper short-circuits once
            # 'rad21' is in input_track_names and would then stop inserting the
            # array into the per-window other_regions.
            from cshark.inference.utils.model_utils import get_all_track_names
            main_all_tracks, _, _ = get_all_track_names(cfg.model_path)
            if 'rad21' in main_all_tracks:
                other_main = [t for t in main_all_tracks if t not in ('ctcf', 'atac')]
                rad21_insert_other_pos = other_main.index('rad21')
                fill_rad21 = True
                insert_global = 2 + rad21_insert_other_pos
                input_track_names.insert(insert_global, 'rad21')
                input_track_paths.insert(insert_global, 'tmp/rad21_hierarchical_wt_pred.bw')
                print(f'[hierarchical] Will predict RAD21 each window and insert at '
                      f'other-position {rad21_insert_other_pos}.')
            else:
                print('[hierarchical] Warning: rad21 not in main model tracks. Disabling hierarchical.')
                hierarchical_rad21_model = None
    print(f'[allele-haplotype] Tracks with both preds (allele-specific): {shared_pred}')
    no_pred = [t for t in input_track_names if t not in shared_pred and t != 'rad21']
    if no_pred:
        print(f'[allele-haplotype] No preds for {no_pred} -- these keep the WT bulk track in '
              f'BOTH alleles (identical between maternal and paternal).')

    # --- accumulators ------------------------------------------------------
    results = {'a1': [], 'a2': [], LABEL_A: [], LABEL_B: []}
    bins = []
    res_1d_bp = max(1, cfg.resolution_1d)
    n_bins_1d = max(1, window // res_1d_bp)
    tracks_1d = {'chrom': [], 'start': [], 'end': []}
    track_1d_cols = []
    for t in input_track_names:
        for suffix in ('experimental', LABEL_A, LABEL_B):
            col = f'{t}_{suffix}'
            track_1d_cols.append(col)
            tracks_1d[col] = []

    print(f'Predicting {chr_name} in {len(starts)} windows (two alleles each).')

    # Open each haplotype's per-chromosome fasta once, not once per window.
    hap_seq_a = SequenceFeature(path=os.path.join(cfg.maternal_seq_path, f'{chr_name}.fa.gz'))
    hap_seq_b = SequenceFeature(path=os.path.join(cfg.paternal_seq_path, f'{chr_name}.fa.gz'))

    model = None
    for start, end in tqdm(zip(starts, ends), desc='Predicting', total=len(starts)):
        # WT bulk tracks E + reference sequence (baseline / channel count).
        seq_ref, ctcf_region, atac_region, other_regions = infer.load_region(
            chr_name, start, args.seq_path, args.ctcf_path, args.atac_path, cfg.other_feats,
            seq2_path=None, window=window, bigwig_log=cfg.bigwig_log_transform)

        if fill_rad21 and rad21_insert_other_pos is not None:
            from cshark.inference.utils.inference_utils import preprocess_default as _pp
            rad21_linear = predict_rad21(
                hierarchical_rad21_model,
                _pp(seq_ref, ctcf_region, atac_region, other_regions), rad21_idx=None)
            ref_len = len(ctcf_region) if ctcf_region is not None else (
                len(atac_region) if atac_region is not None else
                (len(other_regions[0]) if other_regions else len(rad21_linear)))
            if len(rad21_linear) != ref_len:
                rad21_linear = np.interp(np.linspace(0, 1, ref_len),
                                         np.linspace(0, 1, len(rad21_linear)), rad21_linear)
            rad21_log1p = np.log1p(rad21_linear)
            if other_regions is None:
                other_regions = [rad21_log1p]
            else:
                other_regions.insert(rad21_insert_other_pos, rad21_log1p)

        if model is None:
            num_genomic_features = 2 if other_regions is None else 2 + len(other_regions)
            if atac_region is None:
                num_genomic_features -= 1
            if ctcf_region is None:
                num_genomic_features -= 1
            model = CSharkModel(cfg, num_genomic_features=num_genomic_features, diploid=False)

        # Each allele's haploid sequence. Read straight off the hoisted fasta
        # readers -- going through load_region here would re-read all the input
        # bigwigs once per allele per window just to discard them.
        seq_a = hap_seq_a.get(start, start + window)
        seq_b = hap_seq_b.get(start, start + window)

        # Split the bulk tracks by the provided maternal/paternal prediction ratio,
        # then split RAD21 by the hierarchical model's per-allele prediction.
        set_a, set_b, _redistributed = redistribute_from_provided_preds(
            ctcf_region, atac_region, other_regions, input_track_names,
            cfg.maternal_pred, cfg.paternal_pred, chr_name, start, window,
            cap=cap, track_is_log1p=cfg.bigwig_log_transform)
        set_a, set_b = _redistribute_rad21_alleles(
            set_a, set_b, seq_a=seq_a, seq_b=seq_b, other_regions_wt=other_regions,
            input_track_names=input_track_names,
            hierarchical_rad21_model=hierarchical_rad21_model,
            cap=cap, bigwig_log_transform=cfg.bigwig_log_transform)

        # One Hi-C prediction per allele, each with its own haplotype sequence.
        pred_a = model.predict_arrays(seq_a, *set_a, input_track_names[2:])['hic']
        pred_b = model.predict_arrays(seq_b, *set_b, input_track_names[2:])['hic']

        write_tmp_cooler(pred_a, chr_name, start, out_file='tmp/tmp_hap_a.cool', res=res)
        write_tmp_cooler(pred_b, chr_name, start, out_file='tmp/tmp_hap_b.cool', res=res)
        cool_a = cooler.Cooler('tmp/tmp_hap_a.cool')
        cool_b = cooler.Cooler('tmp/tmp_hap_b.cool')
        pix_a = cool_a.pixels()[:].rename(columns={'count': LABEL_A})
        pix_b = cool_b.pixels()[:].rename(columns={'count': LABEL_B})
        pixels = pix_a.merge(pix_b, how='outer')
        results['a1'].extend(pixels['bin1_id'].tolist())
        results['a2'].extend(pixels['bin2_id'].tolist())
        results[LABEL_A].extend(pixels[LABEL_A].tolist())
        results[LABEL_B].extend(pixels[LABEL_B].tolist())
        bins.append(cool_a.bins()[:])

        # Per-track 1D values, binned to --resolution-1d. Emitted in LINEAR space so
        # the conservation identity (maternal + paternal == 2 x experimental, where
        # uncapped) holds against the input bigwigs.
        bin_range = np.int32(np.linspace(start, start + window - res_1d_bp, n_bins_1d))
        tracks_1d['chrom'].extend([chr_name] * n_bins_1d)
        tracks_1d['start'].extend(bin_range.tolist())
        tracks_1d['end'].extend((bin_range + res_1d_bp).tolist())
        for t in input_track_names:
            exp_v = _track_by_name(t, ctcf_region, atac_region, other_regions, input_track_names)
            a_v = _track_by_name(t, *set_a, input_track_names)
            b_v = _track_by_name(t, *set_b, input_track_names)
            for suffix, v in ((f'{t}_experimental', exp_v), (f'{t}_{LABEL_A}', a_v),
                              (f'{t}_{LABEL_B}', b_v)):
                if v is None:
                    tracks_1d[suffix].extend([float('nan')] * n_bins_1d)
                    continue
                # expm1 BEFORE pooling: the redistribution identity
                # (maternal + paternal == 2 * experimental) is linear, and
                # expm1(mean(log1p(x))) != mean(x), so averaging in log1p space
                # would break it.
                lin = np.expm1(np.clip(v, 0, None)) if cfg.bigwig_log_transform else v
                tracks_1d[suffix].extend(_bin_mean(lin, n_bins_1d).tolist())

    # --- aggregate Hi-C ----------------------------------------------------
    try:
        os.makedirs(os.path.dirname(os.path.abspath(args.out_file)), exist_ok=True)
    except Exception as e:
        print(f'Error creating directory: {e}')
    if not args.out_file.endswith('.tsv'):
        args.out_file += '.tsv'

    res_df = pd.DataFrame(results).groupby(['a1', 'a2']).mean().reset_index()
    res_df['a1'] = 'A_' + res_df['a1'].astype(str)
    res_df['a2'] = 'A_' + res_df['a2'].astype(str)
    bins_df = pd.concat(bins, ignore_index=True).drop_duplicates().reset_index(drop=True)
    bins_df['bin_id'] = 'A_' + bins_df.index.astype(str)
    chr_map = bins_df.set_index('bin_id')['chrom'].to_dict()
    start_map = bins_df.set_index('bin_id')['start'].to_dict()
    end_map = bins_df.set_index('bin_id')['end'].to_dict()
    res_df['chrom1'] = res_df['a1'].map(chr_map)
    res_df['chrom2'] = res_df['a2'].map(chr_map)
    res_df['start1'] = res_df['a1'].map(start_map)
    res_df['start2'] = res_df['a2'].map(start_map)
    res_df['end1'] = res_df['a1'].map(end_map)
    res_df['end2'] = res_df['a2'].map(end_map)
    res_df = res_df[['chrom1', 'start1', 'end1', 'a1', 'chrom2', 'start2', 'end2', 'a2',
                     LABEL_A, LABEL_B]]
    if region is not None:
        res_df = res_df[(res_df['start1'] >= region_start) & (res_df['end1'] <= region_end) &
                        (res_df['start2'] >= region_start) & (res_df['end2'] <= region_end)]
        bins_df = bins_df[(bins_df['start'] >= region_start) & (bins_df['end'] <= region_end)]
    res_df.to_csv(args.out_file, sep='\t', header=True, index=False)
    bins_df.to_csv(args.out_file.replace('.tsv', '_bins.tsv'), sep='\t', header=False, index=False)
    print(f'[allele-haplotype] Saved pixels to {args.out_file}')

    # --- coolers: one per allele + paternal-minus-maternal difference ------
    bins_cooler_df = bins_df[['chrom', 'start', 'end']].copy()
    bins_cooler_df.reset_index(inplace=True)
    diff_df = res_df[['a1', 'a2']].copy()
    diff_df['count'] = res_df[LABEL_B].values - res_df[LABEL_A].values
    for label, src in ((LABEL_A, res_df[['a1', 'a2', LABEL_A]].rename(columns={LABEL_A: 'count'})),
                       (LABEL_B, res_df[['a1', 'a2', LABEL_B]].rename(columns={LABEL_B: 'count'})),
                       ('diff', diff_df)):
        df = src.copy()
        df['bin1_id'] = df['a1'].map(lambda x: int(x.replace('A_', '')))
        df['bin2_id'] = df['a2'].map(lambda x: int(x.replace('A_', '')))
        cooler.create_cooler(args.out_file.replace('.tsv', f'_{label}.cool'), bins_cooler_df,
                             df[['bin1_id', 'bin2_id', 'count']], ordered=True,
                             dtypes={'count': 'float32'})
    print(f'[allele-haplotype] Wrote coolers _{LABEL_A}.cool / _{LABEL_B}.cool / _diff.cool '
          f'(diff = {LABEL_B} - {LABEL_A}).')

    # --- aggregate 1D tracks ----------------------------------------------
    if len(tracks_1d['chrom']) > 0:
        tr_df = pd.DataFrame(tracks_1d).groupby(['chrom', 'start', 'end']).mean().reset_index()
        if region is not None:
            tr_df = tr_df[(tr_df['start'] >= region_start) & (tr_df['end'] <= region_end)]
        tr_df = tr_df.sort_values('start').reset_index(drop=True)
        tr_df = tr_df[['chrom', 'start', 'end'] + track_1d_cols]
        bed_path = args.out_file.replace('.tsv', '_haplotype_tracks.bed')
        tr_df.round(4).to_csv(bed_path, sep='\t', header=True, index=False)
        print(f'[allele-haplotype] Saved per-track table to {bed_path}')

        import pyBigWig
        ref_bw = next((p for p in input_track_paths if os.path.exists(p)), None)
        if ref_bw is not None:
            _b = pyBigWig.open(ref_bw)
            header_1d = list(_b.chroms().items())
            _b.close()
        else:
            header_1d = [(chr_name, int(chr_length))]
        chroms_1d = tr_df['chrom'].tolist()
        starts_1d = tr_df['start'].astype(int).tolist()
        ends_1d = tr_df['end'].astype(int).tolist()
        for col in track_1d_cols:
            out_bw = pyBigWig.open(args.out_file.replace('.tsv', f'_{col}.bw'), 'w')
            out_bw.addHeader(header_1d)
            out_bw.addEntries(chroms_1d, starts_1d, ends=ends_1d,
                              values=tr_df[col].astype(float).tolist())
            out_bw.close()
        print(f'[allele-haplotype] Wrote {len(track_1d_cols)} per-track bigwigs '
              f'({len(input_track_names)} tracks x experimental/{LABEL_A}/{LABEL_B}).')

    fig, ax = plt.subplots(figsize=(10, 10))
    sns.scatterplot(data=res_df, x=LABEL_A, y=LABEL_B, ax=ax, s=4, alpha=0.3)
    ax.set_title(f'{LABEL_A} vs {LABEL_B}')
    plt.savefig(os.path.join(args.output_path,
                             f'{args.outname}{args.celltype}_{args.chr_name}_haplotype_scatter.png'),
                dpi=300)
    plt.close(fig)
    print('[allele-haplotype] Done.')
