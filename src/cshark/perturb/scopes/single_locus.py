"""Single-locus perturbation runner.

Faithful transcription of the original ``single_deletion`` (perturb.py lines
653-1628). The function BODY below is byte-identical to the original except:
  * the parameters are unpacked from ``cfg`` (a PerturbConfig) instead of a long
    argument list, and ``args.resolution_1d`` / ``args.whitespace`` are read
    from ``cfg``;
  * ``deletion_with_padding`` / ``seq_perturb`` / ``reverse_complement`` resolve
    to the new package's (identically-named, verbatim) implementations.

This is intentionally a minimal first refactor step (file split + operators/
config from the new package) so that its output matches the old script
byte-for-byte. Further extraction (arcs/plots helpers, CSharkModel, enformer/
hierarchical helpers) is layered in afterwards with re-verification.
"""
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
from importlib.resources import files
from skimage.transform import resize

from cshark.data.data_feature import GenomicFeature, HiCFeature, SequenceFeature
import cshark.inference.utils.inference_utils as infer
from cshark.inference.utils.inference_utils import (
    write_tmp_cooler, write_tmp_chipseq_ko, knockout_peaks, get_axis_range_from_bigwig,
    chunk_shuffle, write_tmp_pred_bigwig, oe_normalize_cooler,
)
from cshark.inference.utils import plot_utils, model_utils
from cshark.inference.utils.enformer_utils import (
    load_enformer_pretrained, load_enformer_from_checkpoint, enformer_seq_knockout,
    write_tmp_enformer_ko_bigwig, write_tmp_enformer_delta_bigwig,
)
from cshark.inference.utils.hierarchical_utils import (
    load_hierarchical_rad21_predictor, hierarchical_rad21_update,
    write_tmp_hierarchical_rad21_bigwig, write_tmp_hierarchical_delta_bigwig,
)
from cshark.inference.tracks_files import get_tracks

from cshark.perturb.config import WINDOW
from cshark.perturb.dna import en_dict, reverse_complement
from cshark.perturb.operators import deletion_with_padding, seq_perturb
from cshark.perturb.output.arcs import write_arcs, write_regions
from cshark.perturb.output.plots import plot_prediction_matrix
from cshark.perturb.models.base import CSharkModel

# module-level plotting constants (verbatim from the original perturb.py)
font_size = 15
plot_width = 17
track_label_fraction = 0.13


def run_single_locus(cfg):
    # --- unpack cfg into the local names the original single_deletion used ---
    output_path = cfg.output_path
    outname = cfg.outname
    celltype = cfg.celltype
    chr_name = cfg.chr_name
    start = cfg.start
    deletion_starts = cfg.deletion_start
    deletion_widths = cfg.deletion_width
    alt_bp = cfg.alt_bp
    model_path = cfg.model_path
    seq_path = cfg.seq_path
    ctcf_path = cfg.ctcf_path
    atac_path = cfg.atac_path
    other_feats = cfg.other_feats
    seq2_path = cfg.seq2_path
    assembly = cfg.assembly
    ko_data = cfg.ko_data
    ko_mode = cfg.ko_mode
    region = cfg.region
    mid_hidden = cfg.mid_hidden
    seq_filter_size = cfg.seq_filter_size
    recon_1d = cfg.recon_1d
    bigwig_log_transform = cfg.bigwig_log_transform
    plot_bigwigs = cfg.plot_bigwigs
    plot_pred_bigwigs = cfg.plot_pred_bigwigs
    plot_pred_log2fc = False
    min_val_true = cfg.min_val_true
    max_val_true = cfg.max_val_true
    min_val_pred = cfg.min_val_pred
    max_val_pred = cfg.max_val_pred
    plot_diff = cfg.plot_diff
    min_val_diff = cfg.min_val_diff
    max_val_diff = cfg.max_val_diff
    plot_bigwig_q = cfg.plot_bigwig_q
    peak_height = cfg.peak_height
    ctcf_motif_p = cfg.ctcf_motif_p
    undo_log = cfg.hic_log_transform
    no_plots = cfg.no_plots
    silent = cfg.silent
    enformer_model_path = cfg.enformer_model_path
    enformer_delta_mode = cfg.enformer_delta_mode
    enformer_delta_cap = cfg.enformer_delta_cap
    enformer_tracks = cfg.enformer_tracks
    hierarchical_model_path = cfg.hierarchical_model_path
    hierarchical_delta_mode = cfg.hierarchical_delta_mode
    hierarchical_delta_cap = cfg.hierarchical_delta_cap
    # module globals that main() set from args
    window = WINDOW
    res = cfg.resolution
    image_scale = cfg.mat_size
    resolution_1d = cfg.resolution_1d
    whitespace = cfg.whitespace
    # --- verbatim body of the original single_deletion follows ---
    os.makedirs(output_path, exist_ok=True)
    if not outname.endswith('_') and outname != '':
        outname += '_'
    if plot_bigwigs is None:
        plot_bigwigs = []
    if plot_pred_bigwigs is None:
        plot_pred_bigwigs = []
    diploid = seq2_path is not None
    ko_data_types = ko_data
    if isinstance(peak_height, float) and deletion_starts is not None:
        peak_height = [peak_height] * len(deletion_starts)
    if isinstance(peak_height, list):
        if len(peak_height) != len(deletion_starts):
            peak_height = [peak_height[0]] * len(deletion_starts)
    tmp_ko_data = []
    for ko_data_type in ko_data:
        if ko_data_type not in tmp_ko_data:
            tmp_ko_data.append(ko_data_type)
    ko_data = tmp_ko_data

    seq_region, ctcf_region, atac_region, other_regions = infer.load_region(chr_name,
            start, seq_path, ctcf_path, atac_path, other_feats, seq2_path=seq2_path,
            window=window, bigwig_log=bigwig_log_transform)

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

    # Load hierarchical RAD21 predictor early so it can fill a missing rad21 track
    hierarchical_rad21_model = None
    hierarchical_rad21_idx = None
    if hierarchical_model_path is not None:
        hierarchical_rad21_model, hier_all_tracks, hierarchical_rad21_idx, _ = \
            load_hierarchical_rad21_predictor(hierarchical_model_path)

    # When rad21 is absent from input bigwigs, predict it from the provided tracks
    # and insert it at the position the main model expects, so the main model
    # receives the correct number of input channels.
    if hierarchical_rad21_model is not None and 'rad21' not in input_track_names:
        from cshark.inference.utils.hierarchical_utils import predict_rad21
        from cshark.inference.utils.model_utils import get_all_track_names
        from cshark.inference.utils.inference_utils import preprocess_default

        main_all_tracks, _, _ = get_all_track_names(model_path)
        if 'rad21' in main_all_tracks:
            # Determine where rad21 sits in the "other" slots of the main model
            other_main_names = [t for t in main_all_tracks if t not in ('ctcf', 'atac')]
            rad21_other_pos = other_main_names.index('rad21')

            # Predict rad21 from the current WT inputs (no channel removal needed:
            # our input tensor already excludes rad21)
            wt_inputs_np = preprocess_default(seq_region, ctcf_region, atac_region, other_regions)
            rad21_linear = predict_rad21(hierarchical_rad21_model, wt_inputs_np,
                                         rad21_idx=None)  # inputs already exclude rad21

            # Resample from model output resolution to bigwig track resolution
            ref_track = ctcf_region if ctcf_region is not None else (
                atac_region if atac_region is not None else
                (other_regions[0] if other_regions else None))
            track_len = len(ref_track) if ref_track is not None else len(rad21_linear)
            if len(rad21_linear) != track_len:
                rad21_linear = np.interp(
                    np.linspace(0, 1, track_len),
                    np.linspace(0, 1, len(rad21_linear)),
                    rad21_linear,
                )
            rad21_log1p = np.log1p(rad21_linear)

            if other_regions is None:
                other_regions = [rad21_log1p]
            else:
                other_regions.insert(rad21_other_pos, rad21_log1p)

            # Use the WT-pred bigwig path as the "experimental" track for plotting.
            # It will be written later by write_tmp_hierarchical_rad21_bigwig.
            rad21_pred_bw = os.path.abspath('tmp/rad21_hierarchical_wt_pred.bw')
            insert_at_global = 2 + rad21_other_pos  # position in full track list
            input_track_names.insert(insert_at_global, 'rad21')
            input_track_paths.insert(insert_at_global, rad21_pred_bw)

            print(f'[hierarchical] Predicted RAD21 from {len(input_track_names) - 1} input tracks, '
                  f'inserted at other-position {rad21_other_pos}.')

    num_genomic_features = 2 if other_regions is None else 2 + len(other_regions)
    if atac_region is None:
        num_genomic_features -= 1
    if ctcf_region is None:
        num_genomic_features -= 1

    # Load the main model ONCE (replaces per-call load_default inside infer.prediction).
    model = CSharkModel(cfg, num_genomic_features=num_genomic_features, diploid=diploid)
    # Baseline prediction (WT)
    pred_before_output = model.predict_arrays(seq_region, ctcf_region, atac_region,
                                              other_regions, input_track_names[2:])
    pred_before = pred_before_output['hic']
    if not no_plots:
        plot_prediction_matrix(pred_before, os.path.join(output_path, f'{outname}{celltype}_{chr_name}_{start}_pred_before.png'), 'Prediction before perturbation')
    pred_before_1d = pred_before_output['1d']

    # Save WT copies for hierarchical delta computation
    seq_region_wt = seq_region.copy()
    ctcf_region_wt = ctcf_region.copy() if ctcf_region is not None else None
    atac_region_wt = atac_region.copy() if atac_region is not None else None
    other_regions_wt = [r.copy() for r in other_regions] if other_regions is not None else None

    if len(input_track_paths) == 0:
        print('No input tracks found. Using plot_bigwigs only.')
        genome_data_path = os.path.dirname(seq_path)
        ctcf_path = os.path.join(genome_data_path, celltype, 'genomic_features', 'ctcf.bw')
        input_track_paths.append(ctcf_path)
        input_track_names.append('ctcf')

    plot_track_names = []
    plot_track_paths = []
    for plot_track in plot_bigwigs:
        if plot_track not in input_track_names:
            plot_track_names.append(plot_track)
            plot_track_paths.append(input_track_paths[0].replace(input_track_names[0], plot_track))
    for plot_track in plot_pred_bigwigs:
        if plot_track not in plot_track_names:
            plot_track_names.append(plot_track)
            plot_track_paths.append(input_track_paths[0].replace(input_track_names[0], plot_track))

    ko_channels = []
    channel_offset = sum(1 for t in ['ctcf', 'atac'] if t in input_track_names)
    for ko in ko_data:
        if ko in input_track_names:
            ko_channels.append(input_track_names.index(ko))
        elif ko != 'seq':
            print(f'Warning: {ko} not found in input track names. Skipping KO for {ko}.')

    # Resolve --alt list
    alt_bp_list = alt_bp if alt_bp is not None else []
    _alt_keywords = {'reverse', 'shuffle', 'random'}
    if deletion_starts is not None and deletion_widths is None:
        deletion_widths = []
        for i, ds in enumerate(deletion_starts):
            if i < len(alt_bp_list) and alt_bp_list[i].lower() not in _alt_keywords:
                deletion_widths.append(max(1, len(alt_bp_list[i])))
            else:
                deletion_widths.append(1)

    def _resolve_alt_string(raw_alt, rel_start, rel_end, current_seq_region, label):
        idx_to_base = {0: 'a', 1: 't', 2: 'c', 3: 'g', 4: 'n'}
        raw_alt_lower = raw_alt.lower()
        if raw_alt_lower == 'reverse':
            rc = reverse_complement(current_seq_region[rel_start:rel_end, :])
            alt_string = ''.join(idx_to_base[row.argmax()] for row in rc)
            print(f'[{label}] Using reverse-complement of {chr_name}:{start + rel_start}-{start + rel_end}')
            return alt_string
        if raw_alt_lower == 'shuffle':
            sub = current_seq_region[rel_start:rel_end, :].copy()
            idxs = np.arange(sub.shape[0])
            np.random.shuffle(idxs)
            sub = sub[idxs, :]
            alt_string = ''.join(idx_to_base[row.argmax()] for row in sub)
            print(f'[{label}] Using shuffled bases of {chr_name}:{start + rel_start}-{start + rel_end}')
            return alt_string
        if raw_alt_lower == 'random':
            bases = 'acgt'
            alt_string = ''.join(np.random.choice(list(bases)) for _ in range(rel_end - rel_start))
            print(f'[{label}] Using random bases for {chr_name}:{start + rel_start}-{start + rel_end}')
            return alt_string
        return raw_alt

    # Apply perturbations
    enformer_seq_active = False
    hierarchical_active = hierarchical_rad21_model is not None
    pending_track_perturbations = []

    if deletion_starts is not None and deletion_widths is not None:
        for ko_idx, (deletion_start, deletion_width, ko_data_type, knockout_mode, ko_height) in enumerate(
                zip(deletion_starts, deletion_widths, ko_data_types, ko_mode, peak_height)):
            raw_alt = alt_bp_list[ko_idx] if ko_idx < len(alt_bp_list) else 'n' * deletion_width

            if knockout_mode == 'enformer_seq':
                deletion_start -= 1
                enformer_seq_active = True
                rel_start = deletion_start - start
                rel_end = rel_start + deletion_width
                alt_string = _resolve_alt_string(raw_alt, rel_start, rel_end, seq_region, 'enformer_seq')
                print(f'[enformer_seq] Queued {len(alt_string)} base(s) at '
                      f'{chr_name}:{deletion_start} (rel {rel_start}): {alt_string.upper()}')
                for bp_offset, base in enumerate(alt_string):
                    abs_pos = rel_start + bp_offset
                    if 0 <= abs_pos < seq_region.shape[0]:
                        if seq2_path is not None:
                            seq1 = seq_region[:, :seq_region.shape[1] // 2]
                            seq2 = seq_region[:, seq_region.shape[1] // 2:]
                            seq1 = seq_perturb(abs_pos, base, seq1)
                            seq2 = seq_perturb(abs_pos, base, seq2)
                            seq_region = np.concatenate((seq1, seq2), axis=1)
                        else:
                            seq_region = seq_perturb(abs_pos, base, seq_region)
                continue

            if knockout_mode == 'seq':
                deletion_start -= 1
                rel_start = deletion_start - start
                rel_end = rel_start + deletion_width
                alt_string = _resolve_alt_string(raw_alt, rel_start, rel_end, seq_region, 'seq')
                print(f'[seq] Substituting {len(alt_string)} base(s) at '
                      f'{chr_name}:{deletion_start} (rel {rel_start}): {alt_string.upper()}')
                for bp_offset, base in enumerate(alt_string):
                    abs_pos = rel_start + bp_offset
                    if 0 <= abs_pos < seq_region.shape[0]:
                        if seq2_path is not None:
                            seq1 = seq_region[:, :seq_region.shape[1] // 2]
                            seq2 = seq_region[:, seq_region.shape[1] // 2:]
                            seq1 = seq_perturb(abs_pos, base, seq1)
                            seq2 = seq_perturb(abs_pos, base, seq2)
                            seq_region = np.concatenate((seq1, seq2), axis=1)
                        else:
                            seq_region = seq_perturb(abs_pos, base, seq_region)
                continue

            if ko_data_type in input_track_names:
                ko_channel = input_track_names.index(ko_data_type)
            else:
                ko_channel = -1
            left_del_pad = None
            right_del_pad = None
            if knockout_mode in ('del', 'deletion', 'delete'):
                left_pad_bp = deletion_width // 2
                right_pad_bp = deletion_width - left_pad_bp
                left_pad_seq, left_pad_ctcf, left_pad_atac, left_pad_other = infer.load_region(chr_name,
                    start - left_pad_bp, seq_path, ctcf_path, atac_path, other_feats,
                    seq2_path=seq2_path, window=left_pad_bp, bigwig_log=bigwig_log_transform)
                left_del_pad = (left_pad_seq, left_pad_ctcf, left_pad_atac, left_pad_other)
                right_pad_seq, right_pad_ctcf, right_pad_atac, right_pad_other = infer.load_region(chr_name,
                    start + window + right_pad_bp, seq_path, ctcf_path, atac_path, other_feats,
                    seq2_path=seq2_path, window=right_pad_bp, bigwig_log=bigwig_log_transform)
                right_del_pad = (right_pad_seq, right_pad_ctcf, right_pad_atac, right_pad_other)
            pending_track_perturbations.append((
                deletion_start, deletion_width, ko_data_type, ko_channel,
                channel_offset, knockout_mode, ko_height, left_del_pad, right_del_pad,
            ))

    if enformer_seq_active:
        print('[enformer_seq] Loading Enformer model for cumulative sequence perturbation...')
        enf_target_tracks = enformer_tracks if enformer_tracks is not None else ['ctcf', 'atac', 'rad21']
        enf_species = 'mouse' if 'mm10' in (assembly or '') else 'human'
        if enformer_model_path is not None:
            enformer_model, enformer_track_names, enf_device = load_enformer_from_checkpoint(
                enformer_model_path, enformer_tracks=enf_target_tracks)
        else:
            enformer_model, enformer_track_names, enf_device = load_enformer_pretrained(
                target_tracks=enf_target_tracks, species=enf_species, celltype=celltype)

        ctcf_region, atac_region, other_regions, enformer_results = enformer_seq_knockout(
            seq_region_wt, ctcf_region, atac_region, other_regions,
            input_track_names, enformer_model, enformer_track_names,
            perturb_track_names=enf_target_tracks, alt_seq_region=seq_region,
            window=window, delta_mode=enformer_delta_mode, cap=enformer_delta_cap,
            track_is_log1p=bigwig_log_transform, device=enf_device,
        )
        print(f'[enformer_seq] Delta applied: mode={enformer_delta_mode}, cap={enformer_delta_cap}.')

        perturbed_track_names = set(enformer_results.get('perturbed_track_names', []))
        enformer_perturbed_track_names = {t.lower() for t in perturbed_track_names}
        for enf_idx, enf_name in enumerate(enformer_results['enformer_track_names']):
            if enf_name in perturbed_track_names and enf_name in input_track_names:
                track_path = input_track_paths[input_track_names.index(enf_name)]
                write_tmp_enformer_ko_bigwig(
                    track_path, enformer_results['fold_change'], enformer_results['delta'],
                    enformer_results['fold_change_log1p'], enformer_results['log1p_delta'],
                    enf_idx, enf_name, chr_name, start, window=window,
                    delta_mode=enformer_delta_mode, cap=enformer_delta_cap,
                    track_is_log1p=bigwig_log_transform,
                )
                write_tmp_enformer_delta_bigwig(
                    track_path, enformer_results['fold_change'], enformer_results['delta'],
                    enformer_results['fold_change_log1p'], enformer_results['log1p_delta'],
                    enf_idx, enf_name, chr_name, start, window=window,
                    delta_mode='additive', track_is_log1p=bigwig_log_transform,
                )
    else:
        enformer_perturbed_track_names = set()

    for deletion_start, deletion_width, ko_data_type, ko_channel, channel_offset, knockout_mode, ko_height, left_del_pad, right_del_pad in pending_track_perturbations:
        seq_region, ctcf_region, atac_region, other_regions = deletion_with_padding(
                chr_name, start, deletion_start, deletion_width,
                seq_region, ctcf_region, atac_region, other_regions,
                ko_data=[ko_data_type], ko_channels=[ko_channel],
                channel_offset=channel_offset,
                ko_mode=[knockout_mode], peak_height=ko_height,
                left_del_pad=left_del_pad, right_del_pad=right_del_pad)

    # Hierarchical RAD21 update: predict delta and apply to experimental track.
    # rad21 is always in input_track_names here (inserted during track loading if absent).
    if hierarchical_rad21_model is not None and 'rad21' in input_track_names:
        other_offset = sum(1 for t in ['ctcf', 'atac'] if t in input_track_names)
        other_track_names = input_track_names[other_offset:]
        rad21_other_idx = other_track_names.index('rad21')
        experimental_rad21_log1p = other_regions_wt[rad21_other_idx].copy()

        # rad21_idx is rad21's position among ALL non-seq input channels
        # (ctcf=0, atac=1, then others in order) — this is what predict_rad21 uses
        # to remove the rad21 channel from the preprocess_default tensor.
        rad21_tensor_idx = input_track_names.index('rad21')

        # Use an existing bigwig for the chromosome header (the path stored in
        # input_track_paths for rad21 may be a not-yet-written tmp/ path when
        # rad21 was predicted/inserted rather than provided directly).
        rad21_bw_path = input_track_paths[input_track_names.index('rad21')]
        if not os.path.exists(rad21_bw_path):
            # Fall back to first available real bigwig for header
            rad21_bw_path = next((p for p in input_track_paths if os.path.exists(p)), rad21_bw_path)

        other_regions, hierarchical_results = hierarchical_rad21_update(
            hierarchical_rad21_model, rad21_tensor_idx,
            seq_region_wt, ctcf_region_wt, atac_region_wt, other_regions_wt,
            seq_region, ctcf_region, atac_region, other_regions,
            experimental_rad21_log1p,
            input_track_names,
            delta_mode=hierarchical_delta_mode,
            cap=hierarchical_delta_cap,
            window=window,
        )

        write_tmp_hierarchical_rad21_bigwig(
            rad21_bw_path,
            hierarchical_results['wt_pred'],          # linear
            hierarchical_results['ko_pred'],           # linear
            hierarchical_results['perturbed_rad21'],   # log1p or None
            chr_name, start, window=window,
        )
        write_tmp_hierarchical_delta_bigwig(
            rad21_bw_path,
            hierarchical_results['delta'],
            hierarchical_results['fold_change'],
            chr_name, start, window=window,
        )

    # Rewrite Enformer KO plotting tracks from final in-memory inputs
    if enformer_perturbed_track_names:
        other_offset = sum(1 for t in ['ctcf', 'atac'] if t in input_track_names)
        other_track_names = input_track_names[other_offset:]
        for track_name in sorted(enformer_perturbed_track_names):
            if track_name not in input_track_names:
                continue
            track_path = input_track_paths[input_track_names.index(track_name)]
            if track_name == 'ctcf':
                track_values = ctcf_region
            elif track_name == 'atac':
                track_values = atac_region
            elif other_regions is not None and track_name in other_track_names:
                track_values = other_regions[other_track_names.index(track_name)]
            else:
                track_values = None
            if track_values is not None:
                # Track values are in log1p space; convert to linear for bigwig
                if bigwig_log_transform:
                    track_values = np.expm1(track_values)
                write_tmp_pred_bigwig(
                    track_path, track_values, track_name, chr_name, start,
                    suffix='enformer_ko', window=window,
                )

    # KO prediction
    pred_output = model.predict_arrays(seq_region, ctcf_region, atac_region,
                                       other_regions, input_track_names[2:])
    pred = pred_output['hic']

    if not no_plots:
        plot_prediction_matrix(pred, os.path.join(output_path, f'{outname}{celltype}_{chr_name}_{start}_pred.png'), 'Prediction after perturbation')
    pred_1d = pred_output['1d']

    if 'del' in ko_mode or 'deletion' in ko_mode or 'delete' in ko_mode and whitespace:
        deletion_start = deletion_starts[0]
        left_pad_px = deletion_widths[0] // 2 // res
        right_pad_px = (deletion_widths[0] - deletion_widths[0] // 2) // res
        del_start_px = (deletion_start - start) // res
        pred = np.concatenate((
            pred[:, :del_start_px],
            np.zeros((pred.shape[0], deletion_widths[0] // res)),
            pred[:, del_start_px:]
        ), axis=1)
        pred = pred[:, left_pad_px:pred.shape[1]-right_pad_px]
        if pred_1d is not None:
            pred_1d = np.concatenate((
                pred_1d[:del_start_px],
                np.zeros((deletion_widths[0] // res, pred_1d.shape[1])),
                pred_1d[del_start_px:]
            ), axis=0)
            pred_1d = pred_1d[left_pad_px:pred_1d.shape[0]-right_pad_px]

    # Write 1D track prediction bigwigs (pred_1d is already in linear space from prediction())
    track_names = model_utils.get_1d_track_names(model_path)
    if track_names is None:
        track_names = []
    for track_idx, track_name in enumerate(track_names):
        try:
            ctcf_pred_before = pred_before_1d[:, track_idx]
            ctcf_pred = pred_1d[:, track_idx]
        except (IndexError, TypeError):
            break
        if track_name not in plot_track_names and track_name not in input_track_names:
            continue
        ctcf_log2fc = np.log2((ctcf_pred + 1e-5) / (ctcf_pred_before + 1e-5))
        log2fc_norm = ctcf_log2fc * ctcf_pred_before
        ymax = max(np.max(ctcf_pred_before), np.max(ctcf_pred))

        if not no_plots and track_name in plot_track_names:
            fig, axs = plt.subplots(3, 1, figsize=(10, 5))
            axs[0].plot(ctcf_pred_before, label='Before', color='blue')
            axs[0].fill_between(range(len(ctcf_pred_before)), ctcf_pred_before, 0, color='blue', alpha=0.2)
            axs[0].set_ylim(0, ymax)
            axs[1].plot(ctcf_pred, label='After', color='orange')
            axs[1].fill_between(range(len(ctcf_pred)), ctcf_pred, 0, color='orange', alpha=0.2)
            axs[1].set_ylim(0, ymax)
            axs[2].plot(log2fc_norm, label='Log2FC', color='green')
            axs[2].fill_between(range(len(log2fc_norm)), log2fc_norm, 0, color='green', alpha=0.2)
            axs[0].set_title(f'{track_name.upper()} Before')
            axs[1].set_title(f'{track_name.upper()} After')
            axs[2].set_title(f'{track_name.upper()} Log2FC * original signal')
            plt.tight_layout()
            plt.savefig(os.path.join(output_path, f'{outname}{celltype}_{chr_name}_{start}_{track_name}_log2fc.png'), dpi=300)
            plt.close(fig)
        if track_name in plot_pred_bigwigs:
            # pred_1d values are in linear space (expm1 applied in prediction())
            write_tmp_pred_bigwig(input_track_paths[0], ctcf_pred_before, track_name, chr_name, start, suffix='pred_WT')
            write_tmp_pred_bigwig(input_track_paths[0], ctcf_pred - ctcf_pred_before, track_name, chr_name, start, suffix='pred_diff')
            if plot_pred_log2fc:
                ctcf_log2fc_clipped = np.clip(ctcf_log2fc, -1, 1)
                write_tmp_pred_bigwig(input_track_paths[0], ctcf_log2fc_clipped, track_name, chr_name, start, suffix='pred_KO')
            else:
                write_tmp_pred_bigwig(input_track_paths[0], ctcf_pred, track_name, chr_name, start, suffix='pred_KO')

    plot_ground_truth = False
    if not no_plots:
        try:
            ctcf_filename = os.path.basename(ctcf_path).split('.')[0]
            hic_path = ctcf_path.replace('genomic_features', 'hic_matrix').replace(f'/{ctcf_filename}.bw', '') + f'/{chr_name}.npz'
            hic = HiCFeature(path=hic_path)
            gt_res = 10000 if res == 8192 else (5000 if res == 4096 else res)
            mat = hic.get(start, window=int(window), res=gt_res)
            mat = resize(mat, (int(image_scale), int(image_scale)), anti_aliasing=True)
            mat += 0.01
            plot = plot_utils.MatrixPlot(output_path, mat, 'ground_truth', celltype, chr_name, start)
            plot.plot(vmin=1.0, vmax=2.5)
            plot_ground_truth = True
        except Exception as e:
            print(e)
            print('No ground truth found')
            mat = np.zeros_like(pred)

    write_tmp_cooler(pred, chr_name, start, res=res)
    write_tmp_cooler(pred_before, chr_name, start, out_file='tmp/tmp_before.cool', res=res)
    if plot_ground_truth:
        write_tmp_cooler(mat, chr_name, start, window=(int(window * 2)), out_file='tmp/tmp_true.cool', res=res)

    diff = pred - pred_before
    write_tmp_cooler(diff, chr_name, start, out_file='tmp/tmp_diff.cool', res=res)
    if deletion_starts is not None and deletion_widths is not None:
        one_perturb_already_done = {}
        print(len(deletion_starts), len(deletion_widths), len(ko_data_types), len(ko_mode), len(peak_height))
        for deletion_start, deletion_width, ko_data_type, knockout_mode, ko_height in zip(deletion_starts, deletion_widths, ko_data_types, ko_mode, peak_height):
            print(f'Writing KO for {ko_data_type} with mode {knockout_mode} at {deletion_start}-{deletion_start + deletion_width}')
            if ko_data_type in input_track_names:
                ko_path = input_track_paths[input_track_names.index(ko_data_type)]
                if ko_data_type in one_perturb_already_done:
                    ko_path = f'tmp/{ko_data_type}_ko.bw'
                write_tmp_chipseq_ko(ko_path, ko_data_type, chr_name, start, deletion_start, deletion_width, ko_mode=knockout_mode, peak_height=ko_height)
                one_perturb_already_done[ko_data_type] = True
            elif ko_data_type != 'seq':
                print(f'Warning: {ko_data_type} not found in input track names. Skipping KO for {ko_data_type}.')

    write_regions(deletion_starts, deletion_widths, chr_name, 'tmp/regions.bed')

    region_start = int(region.split(':')[1].split('-')[0]) if region is not None else start
    region_end = int(region.split(':')[1].split('-')[1]) if region is not None else start + window

    write_arcs(pred_before, chr_name, start, res, region_start, region_end, 'tmp/arcs.bed', quantile=0.99)
    if plot_diff:
        write_arcs(diff, chr_name, start, res, region_start, region_end, 'tmp/arcs_diff.bed', two_sided=True)
    write_arcs(pred, chr_name, start, res, region_start, region_end, 'tmp/arcs_ko.bed', quantile=0.99)
    if plot_ground_truth:
        write_arcs(mat, chr_name, start, res, region_start, region_end, 'tmp/arcs_true.bed', quantile=0.99)

    if '/mm10/' in ctcf_path:
        assembly = 'mm10'
    elif '/hg38/' in ctcf_path:
        assembly = 'hg38'
    assembly_idx = ctcf_path.index(f'/{assembly}/')
    data_root = ctcf_path[:assembly_idx]
    tracks = get_tracks(data_root, celltype, assembly)
    lines = tracks.split('\n')
    lines = [line + '\n' for line in lines]
    colors = ['red', 'blue', 'green', 'orange', 'purple', 'pink', 'brown', 'gray', 'cyan', 'magenta',
              'yellow', 'red', 'blue', 'green', 'orange', 'purple', 'pink']
    ko_track_names = {track_name.lower() for track_name in ko_data}

    def resolve_ko_display_track(track_name, track_path):
        canonical = track_name.lower()
        display_path = track_path
        if hierarchical_active and canonical == 'rad21':
            # Don't redirect to perturbed.bw here — the hierarchical track blocks
            # below write WT pred / KO pred / Perturbed (model input) explicitly.
            # Keeping display_path == track_path suppresses the confusing "rad21 KO"
            # entry that would otherwise duplicate the perturbed track.
            pass
        elif canonical in ko_track_names and os.path.exists(f'tmp/{track_name}_ko.bw'):
            display_path = f'tmp/{track_name}_ko.bw'
        return canonical, display_path

    with open('tmp/tmp_tracks.ini', 'w') as f:
        for line in lines:
            if 'arcs.bed' in line:
                line = line.replace('arcs.bed', 'arcs_ko.bed')

            if '[Genes]' in line:
                for track_i, (track_name, track_path) in enumerate(
                        zip(input_track_names + plot_track_names, input_track_paths + plot_track_paths)):
                    track_name = os.path.basename(track_path).split('.')[0]
                    canonical_track_name, display_track_path = resolve_ko_display_track(track_name, track_path)
                    track_max = None
                    is_model_input = track_name in input_track_names
                    final_track_path = display_track_path if os.path.exists(display_track_path) else None
                    wt_track_path = track_path if os.path.exists(track_path) else None
                    if is_model_input and wt_track_path is not None:
                        f.write(f'[{track_name} WT]\n')
                        f.write(f'file = {wt_track_path}\n')
                        f.write('height = 2\n')
                        f.write(f'color = {colors[track_i]}\n')
                        f.write(f'title = {track_name} WT\n')
                        f.write('min_value = 0\n')
                        wt_track_max = get_axis_range_from_bigwig(wt_track_path, chr_name, start, q=plot_bigwig_q)
                        if wt_track_max is not None:
                            f.write(f'max_value = {wt_track_max}\n')
                        f.write('number_of_bins = 512\n\n')
                        track_max = wt_track_max
                    if final_track_path is not None and (not is_model_input or final_track_path != wt_track_path):
                        final_title = f'{track_name} KO' if is_model_input else track_name
                        f.write(f'[{track_name}]\n')
                        f.write(f'file = {final_track_path}\n')
                        f.write('height = 2\n')
                        f.write(f'color = {colors[track_i]}\n')
                        f.write(f'title = {final_title}\n')
                        f.write('min_value = 0\n')
                        final_track_max = get_axis_range_from_bigwig(final_track_path, chr_name, start, q=plot_bigwig_q)
                        if final_track_max is not None:
                            f.write(f'max_value = {final_track_max}\n')
                        f.write('number_of_bins = 512\n\n')
                        track_max = final_track_max
                    elif track_max is None and final_track_path is not None:
                        track_max = get_axis_range_from_bigwig(final_track_path, chr_name, start, q=plot_bigwig_q)
                    if final_track_path is not None or wt_track_path is not None:
                        if canonical_track_name == 'ctcf' and ctcf_motif_p is not None:
                            f.write('[CTCF motif]\n')
                            motif_file = str(files("cshark").joinpath(f"static/ctcf_motifs_jaspar.bed"))
                            motifs = pd.read_csv(motif_file, sep='\t', names=['chrom', 'start', 'end', 'strand', 'q'])
                            motifs = motifs.loc[motifs['q'] > ctcf_motif_p].reset_index(drop=True)
                            motifs.to_csv('tmp/ctcf_motif.bed', sep='\t', header=False, index=False)
                            f.write('file = tmp/ctcf_motif.bed\n')
                            f.write('file_type = bed\n')
                            f.write('fontsize = 10\n')
                            f.write('display = interleaved\n')
                        if (enformer_seq_active and canonical_track_name in enformer_perturbed_track_names and
                                os.path.exists(f'tmp/{track_name}_enformer_ko.bw')):
                            f.write(f'[{track_name} Enformer KO]\n')
                            f.write(f'file = tmp/{track_name}_enformer_ko.bw\n')
                            f.write('height = 2\n')
                            f.write(f'color = {colors[track_i]}\n')
                            f.write(f'title = {track_name} Enformer KO\n')
                            f.write('min_value = 0\n')
                            if track_max is not None:
                                f.write(f'max_value = {track_max}\n')
                            f.write('number_of_bins = 512\n\n')
                        if hierarchical_active and canonical_track_name == 'rad21':
                            if os.path.exists('tmp/rad21_hierarchical_wt_pred.bw'):
                                f.write('[RAD21 Hier. WT pred]\n')
                                f.write('file = tmp/rad21_hierarchical_wt_pred.bw\n')
                                f.write('height = 2\n')
                                f.write('color = darkgreen\n')
                                f.write('title = RAD21 Hier. WT pred\n')
                                f.write('min_value = 0\n')
                                if track_max is not None:
                                    f.write(f'max_value = {track_max}\n')
                                f.write('number_of_bins = 512\n\n')
                            if os.path.exists('tmp/rad21_hierarchical_ko_pred.bw'):
                                f.write('[RAD21 Hier. KO pred]\n')
                                f.write('file = tmp/rad21_hierarchical_ko_pred.bw\n')
                                f.write('height = 2\n')
                                f.write('color = darkorange\n')
                                f.write('title = RAD21 Hier. KO pred\n')
                                f.write('min_value = 0\n')
                                if track_max is not None:
                                    f.write(f'max_value = {track_max}\n')
                                f.write('number_of_bins = 512\n\n')
                            if os.path.exists('tmp/rad21_hierarchical_perturbed.bw'):
                                f.write('[RAD21 Perturbed (model input)]\n')
                                f.write('file = tmp/rad21_hierarchical_perturbed.bw\n')
                                f.write('height = 2\n')
                                f.write('color = crimson\n')
                                f.write('title = RAD21 Perturbed (model input)\n')
                                f.write('min_value = 0\n')
                                if track_max is not None:
                                    f.write(f'max_value = {track_max}\n')
                                f.write('number_of_bins = 512\n\n')
                    if track_name in plot_pred_bigwigs:
                        pred_ko_file = f'tmp/{track_name}_pred_KO.bw'
                        if (not plot_pred_log2fc and hierarchical_active and
                                track_name.lower() == 'rad21' and
                                os.path.exists('tmp/rad21_hierarchical_ko_pred.bw')):
                            pred_ko_file = 'tmp/rad21_hierarchical_ko_pred.bw'
                        if os.path.exists(pred_ko_file):
                            f.write(f'[{track_name} pred]\n')
                            f.write(f'file = {pred_ko_file}\n')
                            f.write('height = 2\n')
                            if plot_pred_log2fc:
                                f.write(f'title = {track_name} log2FC\n')
                                f.write('color = red\n')
                                f.write('negative_color = blue\n')
                                f.write('min_value = -1\n')
                                f.write('max_value = 1\n')
                            else:
                                f.write(f'title = {track_name} pred KO\n')
                                f.write(f'color = {colors[track_i]}\n')
                                f.write('min_value = 0\n')
                                if track_max is not None:
                                    f.write(f'max_value = {track_max}\n')
                            f.write('number_of_bins = 512\n\n')

                f.write('[KO pred]\n')
                f.write('file = tmp/tmp.cool\n')
                f.write(f'min_value = {min_val_pred}\n')
                if max_val_pred is not None:
                    f.write(f'max_value = {max_val_pred}\n')
                f.write('colormap =  [ (1.0, 1.0, 1.0), (1.0, 0.92, 0.92),(1.0, 0.8, 0.8),(1.0, 0.6, 0.6), (1.0, 0.4, 0.4),(1.0, 0.294, 0.294)]\n')
                f.write('file_type = hic_matrix_square\n\n')

            f.write(line)

        f.write('\n')
        f.write('[deletion]')
        f.write('# bed file with regions to highlight\n')
        f.write('file = tmp/regions.bed\n')
        f.write('alpha = 0.25\n')
        f.write('type = vhighlight\n')

    if plot_ground_truth:
        lines = tracks.split('\n')
        lines = [line + '\n' for line in lines]
        with open('tmp/tmp_tracks_true.ini', 'w') as f:
            for line in lines:
                if 'arcs.bed' in line:
                    line = line.replace('arcs.bed', 'arcs_true.bed')
                if '[Genes]' in line:
                    for track_i, (track_name, track_path) in enumerate(
                            zip(input_track_names + plot_track_names, input_track_paths + plot_track_paths)):
                        track_name = os.path.basename(track_path).split('.')[0]
                        try:
                            track_max = get_axis_range_from_bigwig(track_path, chr_name, start, q=plot_bigwig_q)
                        except Exception as e:
                            print(f'Error getting axis range for {track_path}: {e}')
                            continue
                        f.write(f'[{track_name}]\n')
                        f.write(f'file = {track_path}\n')
                        f.write('height = 2\n')
                        f.write(f'color = {colors[track_i]}\n')
                        f.write(f'title = {track_name}\n')
                        f.write('min_value = 0\n')
                        if track_max is not None:
                            f.write(f'max_value = {track_max}\n')
                        f.write('number_of_bins = 512\n\n')
                    f.write('[deeploop]\n')
                    f.write('file = tmp/tmp_true.cool\n')
                    f.write(f'min_value = {min_val_true}\n')
                    if max_val_true is not None:
                        f.write(f'max_value = {max_val_true}\n')
                    f.write('colormap =  [ (1.0, 1.0, 1.0), (1.0, 0.92, 0.92),(1.0, 0.8, 0.8),(1.0, 0.6, 0.6), (1.0, 0.4, 0.4),(1.0, 0.294, 0.294)]\n')
                    f.write('file_type = hic_matrix_square\n\n')
                f.write(line)

    lines = tracks.split('\n')
    lines = [line + '\n' for line in lines]
    with open('tmp/tmp_tracks_pred.ini', 'w') as f:
        for line in lines:
            if '[Genes]' in line:
                for track_i, (track_name, track_path) in enumerate(
                        zip(input_track_names + plot_track_names, input_track_paths + plot_track_paths)):
                    track_name = os.path.basename(track_path).split('.')[0]
                    canonical_track_name = track_name.lower()
                    if os.path.exists(track_path):
                        f.write(f'[{track_name}]\n')
                        f.write(f'file = {track_path}\n')
                        f.write('height = 2\n')
                        f.write(f'color = {colors[track_i]}\n')
                        f.write(f'title = {track_name}\n')
                        f.write('min_value = 0\n')
                        track_max = get_axis_range_from_bigwig(track_path, chr_name, start, q=plot_bigwig_q)
                        if track_max is not None:
                            f.write(f'max_value = {track_max}\n')
                        f.write('number_of_bins = 512\n\n')
                    else:
                        track_max = None
                    if track_name in plot_pred_bigwigs:
                        pred_wt_file = f'tmp/{track_name}_pred_WT.bw'
                        if (hierarchical_active and track_name.lower() == 'rad21' and
                                os.path.exists('tmp/rad21_hierarchical_wt_pred.bw')):
                            pred_wt_file = 'tmp/rad21_hierarchical_wt_pred.bw'
                        f.write(f'[{track_name} pred]\n')
                        f.write(f'file = {pred_wt_file}\n')
                        f.write('height = 2\n')
                        f.write(f'color = {colors[track_i]}\n')
                        f.write(f'title = {track_name} pred\n')
                        f.write('min_value = 0\n')
                        if track_max is not None:
                            f.write(f'max_value = {track_max}\n')
                        f.write('number_of_bins = 512\n\n')
                    if hierarchical_active and canonical_track_name == 'rad21':
                        if os.path.exists('tmp/rad21_hierarchical_wt_pred.bw'):
                            f.write('[RAD21 Hier. WT pred]\n')
                            f.write('file = tmp/rad21_hierarchical_wt_pred.bw\n')
                            f.write('height = 2\n')
                            f.write('color = darkgreen\n')
                            f.write('title = RAD21 Hier. WT pred\n')
                            f.write('min_value = 0\n')
                            if track_max is not None:
                                f.write(f'max_value = {track_max}\n')
                            f.write('number_of_bins = 512\n\n')
                f.write('[WT pred]\n')
                f.write('file = tmp/tmp_before.cool\n')
                f.write(f'min_value = {min_val_pred}\n')
                if max_val_pred is not None:
                    f.write(f'max_value = {max_val_pred}\n')
                f.write('colormap =  [ (1.0, 1.0, 1.0), (1.0, 0.92, 0.92),(1.0, 0.8, 0.8),(1.0, 0.6, 0.6), (1.0, 0.4, 0.4),(1.0, 0.294, 0.294)]\n')
                f.write('file_type = hic_matrix_square\n\n')
            f.write(line)

    if plot_diff:
        lines = tracks.split('\n')
        lines = [line + '\n' for line in lines]
        with open('tmp/tmp_tracks_diff.ini', 'w') as f:
            for line in lines:
                if 'arcs.bed' in line:
                    line = line.replace('arcs.bed', 'arcs_diff.bed')
                    f.write(line)
                    f.write('line_width = 1\n')
                    f.write('color = bwr\n')
                    f.write('alpha = 0.5\n')
                    f.write('height = 3\n')
                    f.write('file_type = links\n')
                    f.write('links_type = arcs\n')
                    f.write('orientation = inverted\n')
                    break

                if '[Genes]' in line:
                    for track_i, (track_name, track_path) in enumerate(
                            zip(input_track_names + plot_track_names, input_track_paths + plot_track_paths)):
                        track_name = os.path.basename(track_path).split('.')[0]
                        canonical_track_name = track_name.lower()
                        track_max = None
                        if os.path.exists(track_path):
                            f.write(f'[{track_name}]\n')
                            f.write(f'file = {track_path}\n')
                            f.write('height = 2\n')
                            f.write(f'color = {colors[track_i]}\n')
                            f.write(f'title = {track_name}\n')
                            f.write('min_value = 0\n')
                            track_max = get_axis_range_from_bigwig(track_path, chr_name, start, q=plot_bigwig_q)
                            if track_max is not None:
                                f.write(f'max_value = {track_max}\n')
                            f.write('number_of_bins = 512\n\n')
                            if track_name in ko_data:
                                f.write(f'[{track_name} KO]\n')
                                f.write(f'file = tmp/{track_name}_ko.bw\n')
                                f.write('height = 2\n')
                                f.write(f'color = {colors[track_i]}\n')
                                f.write(f'title = {track_name} KO\n')
                                f.write('min_value = 0\n')
                                if track_max is not None:
                                    f.write(f'max_value = {track_max}\n')
                                f.write('number_of_bins = 512\n\n')
                            if (enformer_seq_active and canonical_track_name in enformer_perturbed_track_names and
                                    os.path.exists(f'tmp/{track_name}_enformer_delta.bw')):
                                enformer_ko_file = f'tmp/{track_name}_enformer_ko.bw'
                                if os.path.exists(enformer_ko_file):
                                    f.write(f'[{track_name} Enformer KO]\n')
                                    f.write(f'file = {enformer_ko_file}\n')
                                    f.write('height = 2\n')
                                    f.write(f'color = {colors[track_i]}\n')
                                    f.write(f'title = {track_name} Enformer KO\n')
                                    f.write('min_value = 0\n')
                                    if track_max is not None:
                                        f.write(f'max_value = {track_max}\n')
                                    f.write('number_of_bins = 512\n\n')
                                f.write(f'[{track_name} Enformer Delta]\n')
                                f.write(f'file = tmp/{track_name}_enformer_delta.bw\n')
                                f.write('height = 2\n')
                                f.write('color = red\n')
                                f.write('negative_color = blue\n')
                                f.write(f'title = {track_name} Enformer delta\n')
                                f.write('min_value = -0.5\n')
                                f.write('max_value = 0.5\n')
                                f.write('number_of_bins = 512\n\n')
                            if hierarchical_active and canonical_track_name == 'rad21':
                                if os.path.exists('tmp/rad21_hierarchical_delta.bw'):
                                    f.write('[RAD21 Hier. Delta]\n')
                                    f.write('file = tmp/rad21_hierarchical_delta.bw\n')
                                    f.write('height = 2\n')
                                    f.write('color = red\n')
                                    f.write('negative_color = blue\n')
                                    f.write('title = RAD21 Hier. Delta\n')
                                    f.write('min_value = -0.5\n')
                                    f.write('max_value = 0.5\n')
                                    f.write('number_of_bins = 512\n\n')
                                if os.path.exists('tmp/rad21_hierarchical_perturbed.bw'):
                                    f.write('[RAD21 Perturbed (model input)]\n')
                                    f.write('file = tmp/rad21_hierarchical_perturbed.bw\n')
                                    f.write('height = 2\n')
                                    f.write('color = crimson\n')
                                    f.write('title = RAD21 Perturbed (model input)\n')
                                    f.write('min_value = 0\n')
                                    if track_max is not None:
                                        f.write(f'max_value = {track_max}\n')
                                    f.write('number_of_bins = 512\n\n')
                        if track_name in plot_pred_bigwigs:
                            pred_diff_file = f'tmp/{track_name}_pred_diff.bw'
                            if (plot_pred_log2fc and os.path.exists(f'tmp/{track_name}_pred_KO.bw')):
                                pred_diff_file = f'tmp/{track_name}_pred_KO.bw'
                            if (not plot_pred_log2fc and hierarchical_active and
                                    track_name.lower() == 'rad21' and
                                    os.path.exists('tmp/rad21_hierarchical_delta.bw')):
                                pred_diff_file = 'tmp/rad21_hierarchical_delta.bw'
                            if os.path.exists(pred_diff_file):
                                f.write(f'[{track_name} pred diff]\n')
                                f.write(f'file = {pred_diff_file}\n')
                                f.write('height = 2\n')
                                f.write('color = red\n')
                                f.write('negative_color = blue\n')
                                if plot_pred_log2fc:
                                    f.write(f'title = {track_name} pred log2FC\n')
                                else:
                                    f.write(f'title = {track_name} pred delta\n')
                                f.write('min_value = -1\n')
                                f.write('max_value = 1\n')
                                f.write('number_of_bins = 512\n\n')

                    f.write('[Diff]\n')
                    f.write('file = tmp/tmp_diff.cool\n')
                    f.write(f'min_value = {min_val_diff}\n')
                    if max_val_diff is not None:
                        f.write(f'max_value = {max_val_diff}\n')
                    f.write('colormap = bwr\n')
                    f.write('file_type = hic_matrix_square\n\n')
                f.write(line)

    if not no_plots:
        try:
            region = region if region is not None else f"{chr_name}:{start}-{start + window}"

            if plot_diff:
                tracks_cmd = f"pyGenomeTracks --tracks tmp/tmp_tracks_diff.ini -o {os.path.join(output_path, f'{outname}{celltype}_{chr_name}_{start}_ctcf_ko_tracks_diff.png')} --region {region} --fontSize {font_size} --plotWidth {plot_width} --trackLabelFraction {track_label_fraction}"
                if silent:
                    tracks_cmd += ' > /dev/null 2>&1'
                os.system(tracks_cmd)
            if plot_ground_truth:
                tracks_cmd = f"pyGenomeTracks --tracks tmp/tmp_tracks_true.ini -o {os.path.join(output_path, f'{outname}{celltype}_{chr_name}_{start}_ctcf_true_tracks.png')} --region {region} --fontSize {font_size} --plotWidth {plot_width} --trackLabelFraction {track_label_fraction}"
                if silent:
                    tracks_cmd += ' > /dev/null 2>&1'
                os.system(tracks_cmd)
            tracks_cmd = f"pyGenomeTracks --tracks tmp/tmp_tracks_pred.ini -o {os.path.join(output_path, f'{outname}{celltype}_{chr_name}_{start}_ctcf_pred_tracks.png')} --region {region} --fontSize {font_size} --plotWidth {plot_width} --trackLabelFraction {track_label_fraction}"
            if silent:
                tracks_cmd += ' > /dev/null 2>&1'
            os.system(tracks_cmd)
            if deletion_starts is not None and deletion_widths is not None:
                tracks_cmd = f"pyGenomeTracks --tracks tmp/tmp_tracks.ini -o {os.path.join(output_path, f'{outname}{celltype}_{chr_name}_{start}_ctcf_ko_tracks.png')} --region {region} --fontSize {font_size} --plotWidth {plot_width} --trackLabelFraction {track_label_fraction}"
                if silent:
                    tracks_cmd += ' > /dev/null 2>&1'
                os.system(tracks_cmd)
        except Exception as e:
            print(e)

        try:
            os.rename('tmp/ctcf_motif.bed', 'tmp/ctcf_motifs_detected.bed')
        except Exception:
            pass
