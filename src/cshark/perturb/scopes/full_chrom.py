"""Full-chromosome perturbation runner.

Faithful transcription of the full-chrom branch of the original ``main()``
(perturb.py lines 262-651): slide 2Mb windows across the chromosome (or
``--region``), predict WT + KO per window, average overlapping pixels, and write
the aggregated TSV / cooler / hierarchical-RAD21 bed+bigwig / scatter outputs.
No enformer, no pyGenomeTracks.

The body is byte-identical to the original (only dedented and wrapped): we set
``args = cfg`` because the block reads everything via ``args.X`` and PerturbConfig
exposes all those fields. ``deletion_with_padding`` resolves to the new package's
verbatim operator, and predictions go through a single cached CSharkModel
(loaded once, reused across all sliding windows -- fixes the old per-window reload).
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
from skimage.transform import resize

from cshark.data.data_feature import GenomicFeature, HiCFeature, SequenceFeature
import cshark.inference.utils.inference_utils as infer
from cshark.inference.utils.inference_utils import write_tmp_cooler, oe_normalize_cooler
from cshark.inference.utils.hierarchical_utils import (
    load_hierarchical_rad21_predictor, hierarchical_rad21_update,
)
from cshark.perturb.config import WINDOW
from cshark.perturb.operators import deletion_with_padding
from cshark.perturb.models.base import CSharkModel
from cshark.perturb.seq_source import load_alt_fasta_region, align_alt_to_wt


def run_full_chrom(cfg):
    args = cfg
    other_feats = cfg.other_feats
    window = WINDOW
    res = cfg.resolution
    image_scale = cfg.mat_size
    chr_name = args.chr_name
    if args.ctcf_path is not None:
        bw = GenomicFeature(args.ctcf_path, 'bw')
        chr_length = bw.length(chr_name)
    else:
        seq_file = os.path.join(args.seq_path, f'{chr_name}.fa.gz')
        seq_feature = SequenceFeature(path=seq_file)
        chr_length = len(seq_feature)
    if args.out_file is None:
        args.out_file = os.path.join(args.output_path, f'{args.outname}_{args.celltype}_{chr_name}_full_chr.tsv')
    print(f'Chromosome length: {chr_length}')

    seq_path = args.seq_path
    ctcf_path = args.ctcf_path
    atac_path = args.atac_path
    model_path = args.model_path
    mid_hidden = args.mid_hidden
    ko_data = args.ko_data
    ko_mode = args.ko_mode
    region = args.region

    step_size = int(window / args.n_overlap_preds)
    if region is not None:
        if ':' in region:
            chr_name, region = region.split(':')
            start, end = region.split('-')
            region_start = int(start)
            region_end = int(end)
        else:
            start, end = region.split('-')
            region_start = int(start)
            region_end = int(end)
        starts = np.arange(region_start - step_size, region_end + step_size, step_size)
    else:
        starts = np.arange(0, chr_length - window, step_size)
    ends = starts + window
    results = {'a1': [], 'a2': [], 'WT': [], 'KO': []}
    if args.oe_norm:
        results['exp_WT'] = []
    bins = []

    input_track_names = []
    input_track_paths = []
    if ctcf_path is not None:
        input_track_names.append('ctcf')
        input_track_paths.append(ctcf_path)
    if atac_path is not None:
        input_track_names.append('atac')
        input_track_paths.append(atac_path)
    if other_feats is not None:
        for other_feat in other_feats:
            input_track_names.append(os.path.basename(other_feat).split('.')[0])
            input_track_paths.append(other_feat)

    ko_channels = []
    channel_offset = sum(1 for t in ['ctcf', 'atac'] if t in input_track_names)
    for ko in ko_data:
        if ko in input_track_names:
            ko_channels.append(input_track_names.index(ko))
        elif ko != 'seq':
            print(f'Warning: {ko} not found in input track names. Skipping KO for {ko}.')

    diploid = args.seq2_path is not None

    # Load hierarchical RAD21 predictor if requested
    hierarchical_rad21_model = None
    hierarchical_rad21_idx = None
    rad21_other_idx_hier = None
    use_hierarchical = args.hierarchical_model_path is not None
    fill_rad21 = False          # whether to predict+insert rad21 each window
    rad21_insert_other_pos = None
    if use_hierarchical:
        hierarchical_rad21_model, hier_all_tracks, hierarchical_rad21_idx, _ = \
            load_hierarchical_rad21_predictor(args.hierarchical_model_path)
        other_track_names_hier = input_track_names[channel_offset:]
        if 'rad21' not in other_track_names_hier:
            # Predict and insert rad21 from the model rather than disabling
            from cshark.inference.utils.model_utils import get_all_track_names as _gat
            main_all_tracks, _, _ = _gat(model_path)
            if 'rad21' in main_all_tracks:
                other_main = [t for t in main_all_tracks if t not in ('ctcf', 'atac')]
                rad21_insert_other_pos = other_main.index('rad21')
                fill_rad21 = True
                insert_global = 2 + rad21_insert_other_pos
                input_track_names.insert(insert_global, 'rad21')
                input_track_paths.insert(insert_global, 'tmp/rad21_hierarchical_wt_pred.bw')
                rad21_other_idx_hier = rad21_insert_other_pos
                print(f'[hierarchical] Will predict RAD21 each window and insert at '
                      f'other-position {rad21_insert_other_pos}.')
            else:
                print('[hierarchical] Warning: rad21 not in main model tracks. Disabling hierarchical.')
                use_hierarchical = False
                hierarchical_rad21_model = None
        else:
            rad21_other_idx_hier = other_track_names_hier.index('rad21')

    # --- --alt-fasta whole-window ALT sequence (seq / enformer_seq / alphagenome_seq) ---
    # Full-chrom counterpart of the single-locus path: replace every window's
    # sequence with the alternate genome and run through the backbone selected by
    # --ko-mode. The heavy Enformer/AlphaGenome model is loaded ONCE here (not
    # per window) and called per window via the low-level *_seq_knockout helpers.
    alt_fasta = getattr(cfg, 'alt_fasta', None)
    alt_seq_active = alt_fasta is not None
    _modes = set(ko_mode or [])
    enformer_seq_mode = alt_seq_active and 'enformer_seq' in _modes
    alphagenome_seq_mode = alt_seq_active and 'alphagenome_seq' in _modes
    enf_target_tracks = cfg.enformer_tracks if cfg.enformer_tracks is not None else ['ctcf', 'atac', 'rad21']
    _species = 'mouse' if 'mm10' in (cfg.assembly or '') else 'human'
    enformer_seq_knockout = alphagenome_seq_knockout = None
    enformer_model = enformer_track_names = enf_device = None
    ag_model = ag_track_names = ag_device = ag_org_idx = ag_resolvers = None
    if alt_seq_active and not (enformer_seq_mode or alphagenome_seq_mode) and 'seq' not in _modes:
        print('[alt-fasta] WARNING: --ko-mode has none of seq/enformer_seq/alphagenome_seq; '
              'applying the alt sequence as a plain main-model (seq) substitution.')
    if enformer_seq_mode:
        from cshark.inference.utils.enformer_utils import (
            load_enformer_from_checkpoint, load_enformer_pretrained, enformer_seq_knockout,
        )
        print('[alt-fasta] Loading Enformer once for full-chromosome sequence perturbation...')
        if cfg.enformer_model_path is not None:
            enformer_model, enformer_track_names, enf_device = load_enformer_from_checkpoint(
                cfg.enformer_model_path, enformer_tracks=enf_target_tracks)
        else:
            enformer_model, enformer_track_names, enf_device = load_enformer_pretrained(
                target_tracks=enf_target_tracks, species=_species, celltype=cfg.celltype)
    if alphagenome_seq_mode:
        from cshark.inference.utils.alphagenome_utils import (
            load_alphagenome, alphagenome_seq_knockout,
        )
        print('[alt-fasta] Loading AlphaGenome once for full-chromosome sequence perturbation...')
        ag_model, ag_track_names, ag_device, ag_org_idx, ag_resolvers = load_alphagenome(
            cfg.alphagenome_model_path, target_tracks=enf_target_tracks, species=_species,
            celltype=cfg.celltype, metadata_path=cfg.alphagenome_metadata_path)

    results_hierarchical = {'chrom': [], 'start': [], 'end': []}
    if use_hierarchical:
        for col in ['rad21_WT_pred', 'rad21_KO_pred', 'rad21_delta', 'rad21_fc',
                    'rad21_perturbed', 'rad21_experimental']:
            results_hierarchical[col] = []

    # Enformer/AlphaGenome predicted + perturbed track accumulation (alt-fasta).
    # Mirrors the RAD21 block: for every track the backbone predicts we store its
    # prediction on the WT and ALT sequence (+delta); for tracks that are also
    # model inputs we additionally store the experimental (original) and perturbed
    # (delta-applied, feeds Hi-C) values. All binned to resolution_1d and averaged
    # over overlapping windows before being written as bigwigs.
    enf_seq_backbone_active = enformer_seq_mode or alphagenome_seq_mode
    enf_tool = 'alphagenome' if alphagenome_seq_mode else 'enformer'

    # --- opt-in allele peak redistribution (--allele-peak-split) ------------
    # Default (off) keeps the direct-apply behaviour: the backbone fold-change is
    # multiplied onto the bulk track and one perturbed Hi-C is produced. With the
    # flag on, each bulk track is instead SPLIT into two allele tracks
    # (2*E*frac, capped) and each allele is predicted separately, so the bulk is
    # treated as the sum of the two alleles rather than as something to scale.
    redistribute_alleles = alt_seq_active and cfg.allele_peak_split and enf_seq_backbone_active
    if cfg.allele_peak_split and not redistribute_alleles:
        print('[allele-peak-split] WARNING: needs --alt-fasta together with --ko-mode '
              'enformer_seq or alphagenome_seq; falling back to direct apply-fc.')
    redistribute_enformer_alleles = _redistribute_rad21_alleles = None
    if redistribute_alleles:
        # Same helpers the single-locus allele paths use, so both scales run the
        # same redistribution math.
        from cshark.perturb.models.enformer import redistribute_enformer_alleles
        from cshark.perturb.scopes.allele_split import _redistribute_rad21_alleles
        print(f'[allele-peak-split] Peak redistribution via {enf_tool}: every window is split '
              f'into WT/ALT allele track sets (2*E*frac, CAP={cfg.enformer_delta_cap}) and '
              f'predicted separately -> two Hi-C sets, no direct fold-change apply.')
    lbl_wt = 'wt_after_redistribution' if redistribute_alleles else 'WT'
    lbl_alt = 'alt_after_redistribution' if redistribute_alleles else 'KO'
    if redistribute_alleles and args.oe_norm:
        raise SystemExit(
            '[allele-peak-split] --oe-norm is not supported with peak redistribution: it '
            'normalises against an experimental WT Hi-C, but both outputs here are alleles '
            'of that same sample rather than a WT/KO pair. Drop one of the two flags.')
    enf_res_1d_bp = max(1, cfg.resolution_1d)
    n_bins_enf = max(1, window // enf_res_1d_bp)
    # Enformer output track names (same order the backbone was loaded with).
    enf_backbone_track_names = list(ag_track_names if alphagenome_seq_mode else (enformer_track_names or enf_target_tracks))
    # Tracks that are also model inputs -> they get perturbed and have an experimental copy.
    enf_input_tracks = [t for t in enf_backbone_track_names if t in input_track_names]
    # In redistribution mode every model input gets allele-specific values, including
    # rad21 -- which the backbone does not predict (it comes from the hierarchical
    # model), so it would otherwise never be recorded.
    enf_value_tracks = list(input_track_names) if redistribute_alleles else enf_input_tracks
    results_enformer = {'chrom': [], 'start': [], 'end': []}
    enf_out_cols = []          # ordered list of value columns, set on first window
    if enf_seq_backbone_active:
        for t in enf_backbone_track_names:
            enf_out_cols += [f'{t}_pred_WT', f'{t}_pred_ALT', f'{t}_pred_delta']
        for t in enf_value_tracks:
            enf_out_cols += [f'{t}_experimental']
            if redistribute_alleles:
                enf_out_cols += [f'{t}_wt_after_redistribution',
                                 f'{t}_alt_after_redistribution']
            else:
                enf_out_cols += [f'{t}_perturbed']
        for col in enf_out_cols:
            results_enformer[col] = []

    def _enf_bin(arr):
        """Mean-pool a bp-resolution 1D array to n_bins_enf bins (edge-padded)."""
        arr = np.asarray(arr, dtype=float).ravel()
        if len(arr) == n_bins_enf:
            return arr
        bin_size = max(1, len(arr) // n_bins_enf)
        usable = bin_size * n_bins_enf
        if usable > len(arr):
            arr = np.pad(arr, (0, usable - len(arr)), mode='edge')
        return arr[:usable].reshape(n_bins_enf, bin_size).mean(axis=1)

    def _enf_track_by_name(name, ctcf, atac, others):
        """Return the in-memory input-track array for a backbone track name."""
        if name == 'ctcf':
            return ctcf
        if name == 'atac':
            return atac
        other_offset = sum(1 for t in ['ctcf', 'atac'] if t in input_track_names)
        other_names = input_track_names[other_offset:]
        if others is not None and name in other_names:
            return others[other_names.index(name)]
        return None

    track_names = []
    results_1d = {'chrom': [], 'start': [], 'end': []}
    bins_1d = []

    model = None  # CSharkModel: loaded once on first window, reused for all WT/KO preds
    for start, end in tqdm(zip(starts, ends), desc='Predicting', total=len(starts)):
        seq_region, ctcf_region, atac_region, other_regions = infer.load_region(chr_name,
                start, seq_path, ctcf_path, atac_path, other_feats, window=window)

        # When rad21 was absent from input bigwigs, predict it and insert so
        # the main model receives the correct number of input channels.
        if fill_rad21 and rad21_insert_other_pos is not None:
            from cshark.inference.utils.hierarchical_utils import predict_rad21
            from cshark.inference.utils.inference_utils import preprocess_default as _pp
            _inputs = _pp(seq_region, ctcf_region, atac_region, other_regions)
            rad21_linear = predict_rad21(hierarchical_rad21_model, _inputs, rad21_idx=None)
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

        num_genomic_features = 2 if other_regions is None else 2 + len(other_regions)
        if atac_region is None:
            num_genomic_features -= 1
        if ctcf_region is None:
            num_genomic_features -= 1

        if model is None:
            model = CSharkModel(cfg, num_genomic_features=num_genomic_features, diploid=diploid)
        if redistribute_alleles:
            # Both predictions are per-allele and happen after the split below, so
            # there is no bulk baseline pass here -- still two forwards per window.
            pred_before_output = pred_before = None
        else:
            pred_before_output = model.predict_arrays(seq_region, ctcf_region, atac_region,
                                                      other_regions, input_track_names[2:])
            pred_before = pred_before_output['hic']

        # Save WT copies for hierarchical delta and/or the alt-fasta backbone delta.
        # seq_region_wt is needed by both; the per-track WT copies only by hierarchical.
        if use_hierarchical or alt_seq_active:
            seq_region_wt = seq_region.copy()
        if use_hierarchical:
            ctcf_region_wt = ctcf_region.copy() if ctcf_region is not None else None
            atac_region_wt = atac_region.copy() if atac_region is not None else None
            other_regions_wt = [r.copy() for r in other_regions] if other_regions is not None else None
            experimental_rad21 = other_regions[rad21_other_idx_hier].copy()

        enf_res = None
        wt_set = alt_set = None
        if alt_seq_active:
            # Replace this window's sequence with the alternate genome, then (for
            # enformer_seq/alphagenome_seq) adjust the input tracks by the backbone
            # WT-vs-ALT 1D delta. Plain 'seq' just feeds the alt sequence to the model.
            n_alleles = max(1, seq_region.shape[1] // 5)
            alt_region = load_alt_fasta_region(alt_fasta, chr_name, start, window, n_alleles)
            seq_region = align_alt_to_wt(alt_region, seq_region)
            # Snapshot the experimental (pre-perturbation) input tracks so we can
            # emit them alongside the perturbed versions.
            enf_exp_window = {}
            if enf_seq_backbone_active:
                for t in enf_value_tracks:
                    v = _enf_track_by_name(t, ctcf_region, atac_region, other_regions)
                    enf_exp_window[t] = v.copy() if v is not None else None
            # In redistribution mode the *_seq_knockout call is only used for its
            # wt_pred/alt_pred; its in-place delta-applied tracks are discarded, so
            # it gets COPIES -- otherwise the redistribution would use delta-applied
            # values as the bulk E (the bug fixed in 3a4700e on the single-locus path).
            _ct = ctcf_region.copy() if (redistribute_alleles and ctcf_region is not None) else ctcf_region
            _at = atac_region.copy() if (redistribute_alleles and atac_region is not None) else atac_region
            _ot = ([r.copy() for r in other_regions]
                   if (redistribute_alleles and other_regions is not None) else other_regions)
            if enformer_seq_mode:
                _ct, _at, _ot, enf_res = enformer_seq_knockout(
                    seq_region_wt, _ct, _at, _ot,
                    input_track_names, enformer_model, enformer_track_names,
                    perturb_track_names=enf_target_tracks, alt_seq_region=seq_region,
                    window=window, delta_mode=cfg.enformer_delta_mode, cap=cfg.enformer_delta_cap,
                    track_is_log1p=cfg.bigwig_log_transform, device=enf_device)
            elif alphagenome_seq_mode:
                _ct, _at, _ot, enf_res = alphagenome_seq_knockout(
                    seq_region_wt, _ct, _at, _ot,
                    input_track_names, ag_model, ag_track_names, ag_resolvers, ag_org_idx,
                    perturb_track_names=enf_target_tracks, alt_seq_region=seq_region,
                    window=window, delta_mode=cfg.enformer_delta_mode, cap=cfg.enformer_delta_cap,
                    track_is_log1p=cfg.bigwig_log_transform, device=ag_device)
            if redistribute_alleles:
                # Split the pristine bulk tracks by the backbone's WT-vs-ALT ratio.
                wt_set, alt_set = redistribute_enformer_alleles(
                    ctcf_region, atac_region, other_regions, input_track_names, enf_res,
                    cap=cfg.enformer_delta_cap, track_is_log1p=cfg.bigwig_log_transform)
            else:
                ctcf_region, atac_region, other_regions = _ct, _at, _ot

            # Accumulate backbone predicted + perturbed tracks for this window.
            if enf_res is not None:
                bnames = enf_res.get('enformer_track_names', enf_backbone_track_names)
                wt_pred = np.asarray(enf_res['wt_pred'])   # (bp, n_tracks) linear
                alt_pred = np.asarray(enf_res['alt_pred'])
                res_e = int(window / n_bins_enf)
                bin_range_e = np.int32(np.linspace(start, start + window - res_e, n_bins_enf))
                results_enformer['chrom'].extend([chr_name] * n_bins_enf)
                results_enformer['start'].extend(bin_range_e.tolist())
                results_enformer['end'].extend((bin_range_e + res_e).tolist())
                for t in enf_backbone_track_names:
                    ti = bnames.index(t) if t in bnames else None
                    wt_b = _enf_bin(wt_pred[:, ti]) if ti is not None else np.zeros(n_bins_enf)
                    alt_b = _enf_bin(alt_pred[:, ti]) if ti is not None else np.zeros(n_bins_enf)
                    results_enformer[f'{t}_pred_WT'].extend(wt_b.tolist())
                    results_enformer[f'{t}_pred_ALT'].extend(alt_b.tolist())
                    results_enformer[f'{t}_pred_delta'].extend((alt_b - wt_b).tolist())
                # The per-track experimental / perturbed (or per-allele) values are
                # recorded further down, AFTER the RAD21 step, so they reflect the
                # final arrays that actually feed the model.
        else:
            seq_region, ctcf_region, atac_region, other_regions = deletion_with_padding(
                chr_name, start, start, window, seq_region, ctcf_region,
                atac_region, other_regions, ko_data=ko_data, ko_channels=ko_channels,
                channel_offset=channel_offset, ko_mode=ko_mode,
                peak_height=args.peak_height)

        if redistribute_alleles:
            # RAD21 is split per allele from the hierarchical model's own per-allele
            # predictions (same helper the single-locus allele paths use), so the
            # WT-vs-KO hierarchical_rad21_update does not apply here. The resulting
            # per-allele RAD21 lands in the {track}_*_after_redistribution columns.
            if hierarchical_rad21_model is not None and 'rad21' in input_track_names:
                wt_set, alt_set = _redistribute_rad21_alleles(
                    wt_set, alt_set, seq_a=seq_region_wt, seq_b=seq_region,
                    other_regions_wt=other_regions, input_track_names=input_track_names,
                    hierarchical_rad21_model=hierarchical_rad21_model,
                    cap=cfg.enformer_delta_cap,
                    bigwig_log_transform=cfg.bigwig_log_transform)
            hierarchical_results_window = None
        elif use_hierarchical and other_regions is not None:
            # Use the tensor position of rad21 (not the checkpoint's internal idx)
            rad21_tensor_idx = input_track_names.index('rad21')
            other_regions, hierarchical_results_window = hierarchical_rad21_update(
                hierarchical_rad21_model, rad21_tensor_idx,
                seq_region_wt, ctcf_region_wt, atac_region_wt, other_regions_wt,
                seq_region, ctcf_region, atac_region, other_regions,
                experimental_rad21,
                input_track_names,
                delta_mode=args.hierarchical_delta_mode,
                cap=args.hierarchical_delta_cap,
                window=window,
            )
        else:
            hierarchical_results_window = None

        # Per-track values that actually feed the model, recorded after the RAD21
        # step so rad21 itself is included. Emitted in LINEAR space.
        if enf_seq_backbone_active and enf_res is not None:
            def _lin_bin(v):
                """Track array (log1p) -> linear, then mean-pooled to n_bins_enf.

                expm1 BEFORE pooling: these are signal means, and
                expm1(mean(log1p(x))) != mean(x). Pooling first biases peaky tracks
                low and would break the redistribution identity
                wt + alt == 2 * experimental.
                """
                if v is None:
                    return np.zeros(n_bins_enf)
                lin = np.expm1(np.clip(v, 0, None)) if cfg.bigwig_log_transform else v
                return _enf_bin(lin)

            for t in enf_value_tracks:
                results_enformer[f'{t}_experimental'].extend(
                    _lin_bin(enf_exp_window.get(t)).tolist())
                if redistribute_alleles:
                    results_enformer[f'{t}_wt_after_redistribution'].extend(
                        _lin_bin(_enf_track_by_name(t, *wt_set)).tolist())
                    results_enformer[f'{t}_alt_after_redistribution'].extend(
                        _lin_bin(_enf_track_by_name(t, *alt_set)).tolist())
                else:
                    results_enformer[f'{t}_perturbed'].extend(
                        _lin_bin(_enf_track_by_name(t, ctcf_region, atac_region,
                                                    other_regions)).tolist())

        if redistribute_alleles:
            # Two per-allele predictions replace the (bulk baseline, perturbed) pair:
            # each allele uses its own sequence and its own redistributed tracks.
            pred_before_output = model.predict_arrays(seq_region_wt, *wt_set,
                                                     input_track_names[2:])
            pred_output = model.predict_arrays(seq_region, *alt_set,
                                               input_track_names[2:])
            pred_before = pred_before_output['hic']
        else:
            pred_output = model.predict_arrays(seq_region, ctcf_region, atac_region,
                                               other_regions, input_track_names[2:])
        pred = pred_output['hic']

        # Collect the model's 1D reconstruction-head predictions for every track
        # (ctcf/atac/rad21/h3k27ac/... -- whatever the checkpoint reconstructs).
        # pred_*_1d is (n_bins, n_tracks) in linear space; track order matches
        # model.track_names_1d. WT = before KO, KO = after the perturbation.
        pred_before_1d = pred_before_output['1d']
        pred_ko_1d = pred_output['1d']
        if pred_before_1d is not None and pred_ko_1d is not None:
            n_tracks_1d = min(pred_before_1d.shape[1], pred_ko_1d.shape[1])
            if not track_names:
                track_names = list((model.track_names_1d or [])[:n_tracks_1d])
                while len(track_names) < n_tracks_1d:
                    track_names.append(f'track{len(track_names)}')
                for t in track_names:
                    for suffix in ('WT_pred', 'KO_pred', 'delta'):
                        results_1d.setdefault(f'{t}_{suffix}', [])
            n_bins_1d = pred_before_1d.shape[0]
            res_1d_bp = int(window / n_bins_1d)
            bin_range_1d = np.int32(np.linspace(start, start + window - res_1d_bp, n_bins_1d))
            results_1d['chrom'].extend([chr_name] * n_bins_1d)
            results_1d['start'].extend(bin_range_1d.tolist())
            results_1d['end'].extend((bin_range_1d + res_1d_bp).tolist())
            for i, t in enumerate(track_names):
                wt_vals = np.asarray(pred_before_1d[:, i], dtype=float)
                ko_vals = np.asarray(pred_ko_1d[:, i], dtype=float)
                results_1d[f'{t}_WT_pred'].extend(wt_vals.tolist())
                results_1d[f'{t}_KO_pred'].extend(ko_vals.tolist())
                results_1d[f'{t}_delta'].extend((ko_vals - wt_vals).tolist())

        write_tmp_cooler(pred, chr_name, start, res=res)
        write_tmp_cooler(pred_before, chr_name, start, out_file='tmp/tmp_before.cool', res=res)
        pred_cooler = cooler.Cooler('tmp/tmp.cool')
        pred_before_cooler = cooler.Cooler('tmp/tmp_before.cool')
        wt_pixels = pred_before_cooler.pixels()[:].rename(columns={'count': 'WT'})
        ko_pixels = pred_cooler.pixels()[:].rename(columns={'count': 'KO'})

        if args.oe_norm:
            ctcf_filename = os.path.basename(ctcf_path).split('.')[0]
            hic_path = ctcf_path.replace('genomic_features', 'hic_matrix').replace(f'/{ctcf_filename}.bw', '') + f'/{chr_name}.npz'
            hic = HiCFeature(path=hic_path)
            gt_res = 10000 if res == 8192 else (5000 if res == 4096 else res)
            mat = hic.get(start, window=int(window), res=gt_res)
            mat = resize(mat, (int(image_scale), int(image_scale)), anti_aliasing=True, preserve_range=True)
            mat += 0.01
            write_tmp_cooler(mat, chr_name, start, window=(int(window * 2)), out_file='tmp/tmp_true.cool', res=res)
            true_pixels = cooler.Cooler('tmp/tmp_true.cool').pixels()[:].rename(columns={'count': 'exp_WT'})
            pixels = wt_pixels.merge(ko_pixels, how='outer').merge(true_pixels, how='outer')
            results['exp_WT'].extend(pixels['exp_WT'].tolist())
        else:
            pixels = wt_pixels.merge(ko_pixels, how='outer')
        results['a1'].extend(pixels['bin1_id'].tolist())
        results['a2'].extend(pixels['bin2_id'].tolist())
        results['WT'].extend(pixels['WT'].tolist())
        results['KO'].extend(pixels['KO'].tolist())
        bins.append(pred_before_cooler.bins()[:])

        # Collect hierarchical results
        if use_hierarchical and hierarchical_results_window is not None:
            wt_pred_h = hierarchical_results_window['wt_pred']
            ko_pred_h = hierarchical_results_window['ko_pred']
            delta_h = hierarchical_results_window['delta']
            fc_h = hierarchical_results_window['fold_change']
            perturbed_h = hierarchical_results_window['perturbed_rad21']
            n_bins_h = len(wt_pred_h)
            res_h = int(window / n_bins_h)
            bin_range_h = np.int32(np.linspace(start, start + window - res_h, n_bins_h))
            results_hierarchical['chrom'].extend([chr_name] * n_bins_h)
            results_hierarchical['start'].extend(bin_range_h.tolist())
            results_hierarchical['end'].extend((bin_range_h + res_h).tolist())
            results_hierarchical['rad21_WT_pred'].extend(wt_pred_h.tolist())
            results_hierarchical['rad21_KO_pred'].extend(ko_pred_h.tolist())
            results_hierarchical['rad21_delta'].extend(delta_h.tolist())
            results_hierarchical['rad21_fc'].extend(fc_h.tolist())
            # perturbed_h is log1p (or None in prediction-only mode)
            if perturbed_h is not None:
                if len(perturbed_h) != n_bins_h:
                    perturbed_h = np.interp(np.linspace(0, 1, n_bins_h),
                                            np.linspace(0, 1, len(perturbed_h)), perturbed_h)
                results_hierarchical['rad21_perturbed'].extend(perturbed_h.tolist())
            else:
                results_hierarchical['rad21_perturbed'].extend([float('nan')] * n_bins_h)
            # Experimental RAD21 (only available when rad21 bigwig provided)
            if experimental_rad21 is not None:
                exp_rad21_h = experimental_rad21.copy()
                if len(exp_rad21_h) != n_bins_h:
                    exp_rad21_h = np.interp(np.linspace(0, 1, n_bins_h),
                                            np.linspace(0, 1, len(exp_rad21_h)), exp_rad21_h)
                results_hierarchical['rad21_experimental'].extend(exp_rad21_h.tolist())
            else:
                results_hierarchical['rad21_experimental'].extend([float('nan')] * n_bins_h)

    try:
        os.makedirs(os.path.dirname(args.out_file), exist_ok=True)
    except Exception as e:
        print(f"Error creating directory: {e}")
    if not args.out_file.endswith('.tsv'):
        args.out_file += '.tsv'

    res_df = pd.DataFrame(results).groupby(['a1', 'a2']).mean().reset_index()
    res_df['a1'] = 'A_' + res_df['a1'].astype(str)
    res_df['a2'] = 'A_' + res_df['a2'].astype(str)
    print(res_df)
    bins_df = pd.concat(bins, ignore_index=True).drop_duplicates().reset_index(drop=True)
    bins_df['bin_id'] = 'A_' + bins_df.index.astype(str)
    print(bins_df)
    chr_map = bins_df.set_index('bin_id')['chrom'].to_dict()
    start_map = bins_df.set_index('bin_id')['start'].to_dict()
    end_map = bins_df.set_index('bin_id')['end'].to_dict()
    res_df['chrom1'] = res_df['a1'].map(chr_map)
    res_df['chrom2'] = res_df['a2'].map(chr_map)
    res_df['start1'] = res_df['a1'].map(start_map)
    res_df['start2'] = res_df['a2'].map(start_map)
    res_df['end1'] = res_df['a1'].map(end_map)
    res_df['end2'] = res_df['a2'].map(end_map)
    res_df = res_df[['chrom1', 'start1', 'end1', 'a1', 'chrom2', 'start2', 'end2', 'a2', 'WT', 'KO'] +
                    (['exp_WT'] if args.oe_norm else [])]
    if redistribute_alleles:
        res_df = res_df.rename(columns={'WT': lbl_wt, 'KO': lbl_alt})
    if region is not None:
        res_df = res_df[(res_df['start1'] >= region_start) & (res_df['end1'] <= region_end) &
                        (res_df['start2'] >= region_start) & (res_df['end2'] <= region_end)]
        bins_df = bins_df[(bins_df['start'] >= region_start) & (bins_df['end'] <= region_end)]
    res_df.to_csv(args.out_file, sep='\t', header=True, index=False)
    bins_df.to_csv(args.out_file.replace('.tsv', '_bins.tsv'), sep='\t', header=False, index=False)

    # Cooler outputs
    bins_cooler_df = bins_df[['chrom', 'start', 'end']].copy()
    bins_cooler_df.reset_index(inplace=True)
    # 'A_<i>' labels index the UNFILTERED bin table. With --region, bins_df has been
    # subset, so those labels no longer match the 0..N-1 row positions cooler expects
    # -- remap through the surviving bins (and drop pixels whose bins were filtered
    # out). Without --region this map is the identity, so the cooler is unchanged.
    bin_pos = {b: i for i, b in enumerate(bins_df['bin_id'])}
    wt_cooler_df = res_df[['a1', 'a2', lbl_wt]].rename(columns={lbl_wt: 'count'})
    ko_cooler_df = res_df[['a1', 'a2', lbl_alt]].rename(columns={lbl_alt: 'count'})
    _cooler_dfs = []
    for df in [wt_cooler_df, ko_cooler_df]:
        df['bin1_id'] = df['a1'].map(bin_pos)
        df['bin2_id'] = df['a2'].map(bin_pos)
        df = df.dropna(subset=['bin1_id', 'bin2_id'])
        df['bin1_id'] = df['bin1_id'].astype(int)
        df['bin2_id'] = df['bin2_id'].astype(int)
        _cooler_dfs.append(df[['bin1_id', 'bin2_id', 'count']])
    wt_cooler_df, ko_cooler_df = _cooler_dfs
    cooler.create_cooler(args.out_file.replace('.tsv', f'_{lbl_wt}.cool'), bins_cooler_df, wt_cooler_df, ordered=True, dtypes={'count': 'float32'})
    cooler.create_cooler(args.out_file.replace('.tsv', f'_{lbl_alt}.cool'), bins_cooler_df, ko_cooler_df, ordered=True, dtypes={'count': 'float32'})

    if args.oe_norm:
        exp_wt_cooler_df = res_df[['a1', 'a2', 'exp_WT']].rename(columns={'exp_WT': 'count'})
        exp_wt_cooler_df['bin1_id'] = exp_wt_cooler_df['a1'].map(bin_pos)
        exp_wt_cooler_df['bin2_id'] = exp_wt_cooler_df['a2'].map(bin_pos)
        exp_wt_cooler_df = exp_wt_cooler_df.dropna(subset=['bin1_id', 'bin2_id'])
        exp_wt_cooler_df['bin1_id'] = exp_wt_cooler_df['bin1_id'].astype(int)
        exp_wt_cooler_df['bin2_id'] = exp_wt_cooler_df['bin2_id'].astype(int)
        exp_wt_cooler_df = exp_wt_cooler_df[['bin1_id', 'bin2_id', 'count']]
        cooler.create_cooler(args.out_file.replace('.tsv', '_exp_WT.cool'), bins_cooler_df, exp_wt_cooler_df, ordered=True, dtypes={'count': 'float32'})

        dist_norm_res_WT_df = oe_normalize_cooler(cooler.Cooler(args.out_file.replace('.tsv', '_WT.cool')))
        dist_norm_res_KO_df = oe_normalize_cooler(cooler.Cooler(args.out_file.replace('.tsv', '_KO.cool')))
        dist_norm_res_exp_WT_df = oe_normalize_cooler(cooler.Cooler(args.out_file.replace('.tsv', '_exp_WT.cool')))
        dist_norm_res_df = dist_norm_res_WT_df.merge(dist_norm_res_KO_df, on=['bin1_id', 'bin2_id'], suffixes=('_WT', '_KO'))
        dist_norm_res_df = dist_norm_res_df.merge(dist_norm_res_exp_WT_df, on=['bin1_id', 'bin2_id'])
        dist_norm_res_df = dist_norm_res_df.rename(columns={'count': 'exp_WT', 'count_WT': 'WT', 'count_KO': 'KO'})
        dist_norm_res_df['a1'] = res_df['a1']
        dist_norm_res_df['a2'] = res_df['a2']
        for col in ['WT', 'KO', 'exp_WT']:
            dist_norm_res_df[col] = dist_norm_res_df[col].round(3)
        dist_norm_res_df['chrom1'] = chr_name
        dist_norm_res_df['chrom2'] = chr_name
        dist_norm_res_df['start1'] = dist_norm_res_df['a1'].map(start_map)
        dist_norm_res_df['end1'] = dist_norm_res_df['a1'].map(end_map)
        dist_norm_res_df['start2'] = dist_norm_res_df['a2'].map(start_map)
        dist_norm_res_df['end2'] = dist_norm_res_df['a2'].map(end_map)
        dist_norm_res_df.dropna(inplace=True)
        dist_norm_res_df = dist_norm_res_df[['chrom1', 'start1', 'end1', 'a1', 'chrom2', 'start2', 'end2', 'a2', 'WT', 'KO', 'exp_WT']]
        for col in ['start1', 'end1', 'start2', 'end2']:
            dist_norm_res_df[col] = dist_norm_res_df[col].astype(int)
        dist_norm_res_df = dist_norm_res_df[(dist_norm_res_df['WT'] >= 1e-4) | (dist_norm_res_df['KO'] >= 1e-4) | (dist_norm_res_df['exp_WT'] >= 1e-4)].reset_index(drop=True)
        dist_norm_res_df.to_csv(args.out_file.replace('.tsv', '_oe_norm.tsv'), sep='\t', header=True, index=False)
        for suffix, norm_df in [('_WT_oe_norm', dist_norm_res_WT_df), ('_KO_oe_norm', dist_norm_res_KO_df), ('_exp_WT_oe_norm', dist_norm_res_exp_WT_df)]:
            cooler.create_cooler(args.out_file.replace('.tsv', f'{suffix}.cool'), bins_cooler_df, norm_df[['bin1_id', 'bin2_id', 'count']], ordered=True, dtypes={'count': 'float32'})

    # Save hierarchical RAD21 predictions
    if use_hierarchical and len(results_hierarchical['chrom']) > 0:
        res_hier_df = pd.DataFrame(results_hierarchical).groupby(['chrom', 'start', 'end']).mean().reset_index()
        hier_col_names = ['rad21_WT_pred', 'rad21_KO_pred', 'rad21_delta', 'rad21_fc',
                          'rad21_perturbed', 'rad21_experimental']
        res_hier_df = res_hier_df[['chrom', 'start', 'end'] + hier_col_names]
        if region is not None:
            res_hier_df = res_hier_df[(res_hier_df['start'] >= region_start) & (res_hier_df['end'] <= region_end)]
        hier_out_path = args.out_file.replace('.tsv', '_hierarchical_rad21.bed')
        for col in hier_col_names:
            res_hier_df[col] = res_hier_df[col].round(4)
        res_hier_df.to_csv(hier_out_path, sep='\t', header=True, index=False)
        print(f'[hierarchical] Saved to {hier_out_path}')

        import pyBigWig
        rad21_bw_base = input_track_paths[input_track_names.index('rad21')]
        _bw_ref = pyBigWig.open(rad21_bw_base)
        header_list = list(_bw_ref.chroms().items())
        _bw_ref.close()
        hier_sorted = res_hier_df.sort_values('start').reset_index(drop=True)
        for bw_out_path, col_name in [
            (args.out_file.replace('.tsv', '_hierarchical_rad21_WT.bw'), 'rad21_WT_pred'),
            (args.out_file.replace('.tsv', '_hierarchical_rad21_KO.bw'), 'rad21_KO_pred'),
            # perturbed stored as log1p in results; convert to linear for bigwig
            (args.out_file.replace('.tsv', '_hierarchical_rad21_perturbed_linear.bw'), 'rad21_perturbed'),
            (args.out_file.replace('.tsv', '_hierarchical_rad21_experimental.bw'), 'rad21_experimental'),
            (args.out_file.replace('.tsv', '_hierarchical_rad21_delta.bw'), 'rad21_delta'),
        ]:
            out_bw = pyBigWig.open(bw_out_path, 'w')
            out_bw.addHeader(header_list)
            values = hier_sorted[col_name].astype(float).tolist()
            if col_name == 'rad21_perturbed':
                # stored as log1p — convert to linear for visualization
                values = np.expm1(np.clip(values, 0, None)).tolist()
            elif col_name == 'rad21_experimental':
                # stored as log1p — convert to linear
                values = np.expm1(np.clip(values, 0, None)).tolist()
            out_bw.addEntries(hier_sorted['chrom'].tolist(),
                              hier_sorted['start'].astype(int).tolist(),
                              ends=hier_sorted['end'].astype(int).tolist(),
                              values=values)
            out_bw.close()
            print(f'[hierarchical] Wrote {bw_out_path}')

        fig, axs = plt.subplots(1, 3, figsize=(15, 5))
        sns.scatterplot(data=res_hier_df, x='rad21_WT_pred', y='rad21_KO_pred', alpha=0.5, ax=axs[0])
        axs[0].set_title('Hierarchical RAD21 WT vs KO')
        axs[0].set_aspect('equal', adjustable='box')
        axs[1].hist(res_hier_df['rad21_delta'], bins=50, alpha=0.7)
        axs[1].set_title('RAD21 Delta Distribution')
        axs[2].hist(res_hier_df['rad21_fc'], bins=50, alpha=0.7)
        axs[2].set_title('RAD21 Fold Change Distribution')
        plt.tight_layout()
        plt.savefig(os.path.join(args.output_path, f'{args.outname}{args.celltype}_{args.chr_name}_hierarchical_rad21.png'), dpi=300)
        plt.close(fig)

    # Save the 1D reconstruction-head predictions for every track as bigwigs.
    # This writes WT (unperturbed), KO (perturbed) and delta bigwigs per track,
    # so the full-chromosome run emits all predicted 1D tracks, not just RAD21.
    if track_names and len(results_1d['chrom']) > 0:
        res_1d_df = pd.DataFrame(results_1d).groupby(['chrom', 'start', 'end']).mean().reset_index()
        if region is not None:
            res_1d_df = res_1d_df[(res_1d_df['start'] >= region_start) & (res_1d_df['end'] <= region_end)]
        res_1d_df = res_1d_df.sort_values('start').reset_index(drop=True)
        bed_1d_path = args.out_file.replace('.tsv', '_pred_1d_tracks.bed')
        res_1d_df.round(4).to_csv(bed_1d_path, sep='\t', header=True, index=False)
        print(f'[1d] Saved predicted 1D tracks table ({len(track_names)} tracks) to {bed_1d_path}')

        import pyBigWig
        # Chrom-size header: reuse a real input bigwig if available, else a
        # single-chrom header derived from the chromosome length.
        ref_bw_path = next((p for p in (ctcf_path, atac_path) if p), None)
        if ref_bw_path is not None:
            _bw_ref = pyBigWig.open(ref_bw_path)
            header_1d = list(_bw_ref.chroms().items())
            _bw_ref.close()
        else:
            header_1d = [(chr_name, int(chr_length))]
        chroms_1d = res_1d_df['chrom'].tolist()
        starts_1d = res_1d_df['start'].astype(int).tolist()
        ends_1d = res_1d_df['end'].astype(int).tolist()
        for t in track_names:
            # In redistribution mode the two passes are the two alleles, not WT vs KO.
            for suffix, col in ((lbl_wt, f'{t}_WT_pred'), (lbl_alt, f'{t}_KO_pred'),
                                ('delta', f'{t}_delta')):
                bw_out_path = args.out_file.replace('.tsv', f'_pred_1d_{t}_{suffix}.bw')
                out_bw = pyBigWig.open(bw_out_path, 'w')
                out_bw.addHeader(header_1d)
                out_bw.addEntries(chroms_1d, starts_1d, ends=ends_1d,
                                  values=res_1d_df[col].astype(float).tolist())
                out_bw.close()
        print(f'[1d] Wrote {len(track_names)} tracks x 3 ({lbl_wt}/{lbl_alt}/delta) '
              f'predicted-track bigwigs.')

    # Save Enformer/AlphaGenome predicted + perturbed tracks as bigwigs (alt-fasta).
    # Per backbone track: prediction on the WT and ALT sequence (+delta); for tracks
    # that are also model inputs, the experimental (original) and perturbed (feeds
    # Hi-C) values too -- the direct analogue of the RAD21 bigwig set.
    if enf_seq_backbone_active and enf_out_cols and len(results_enformer['chrom']) > 0:
        res_enf_df = pd.DataFrame(results_enformer).groupby(['chrom', 'start', 'end']).mean().reset_index()
        if region is not None:
            res_enf_df = res_enf_df[(res_enf_df['start'] >= region_start) & (res_enf_df['end'] <= region_end)]
        res_enf_df = res_enf_df.sort_values('start').reset_index(drop=True)
        bed_enf_path = args.out_file.replace('.tsv', f'_{enf_tool}_pred_tracks.bed')
        res_enf_df.round(4).to_csv(bed_enf_path, sep='\t', header=True, index=False)
        print(f'[{enf_tool}] Saved predicted/perturbed tracks table to {bed_enf_path}')

        import pyBigWig
        ref_bw_path = next((p for p in (ctcf_path, atac_path) if p), None)
        if ref_bw_path is not None:
            _bw_ref = pyBigWig.open(ref_bw_path)
            header_enf = list(_bw_ref.chroms().items())
            _bw_ref.close()
        else:
            header_enf = [(chr_name, int(chr_length))]
        chroms_e = res_enf_df['chrom'].tolist()
        starts_e = res_enf_df['start'].astype(int).tolist()
        ends_e = res_enf_df['end'].astype(int).tolist()
        for col in enf_out_cols:
            bw_out_path = args.out_file.replace('.tsv', f'_{enf_tool}_{col}.bw')
            out_bw = pyBigWig.open(bw_out_path, 'w')
            out_bw.addHeader(header_enf)
            out_bw.addEntries(chroms_e, starts_e, ends=ends_e,
                              values=res_enf_df[col].astype(float).tolist())
            out_bw.close()
        print(f'[{enf_tool}] Wrote {len(enf_out_cols)} predicted/perturbed track bigwigs '
              f'({len(enf_backbone_track_names)} backbone tracks).')

    fig, ax = plt.subplots(figsize=(10, 10))
    sns.scatterplot(data=res_df, x=lbl_wt, y=lbl_alt, ax=ax)
    ax.set_title(f'{lbl_wt} vs {lbl_alt}')
    plt.savefig(os.path.join(args.output_path, f'{args.outname}{args.celltype}_{args.chr_name}_scatter.png'), dpi=300)
    plt.close(fig)
