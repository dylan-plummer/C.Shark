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
from cshark.perturb.output.tracks_ini import build_track_inis
from cshark.perturb.models.hierarchical import prepare_rad21_input, apply_rad21_update
from cshark.perturb.models.enformer import apply_enformer_seq_ko, rewrite_enformer_ko_tracks

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
    other_regions = prepare_rad21_input(
        atac_region=atac_region, ctcf_region=ctcf_region, hierarchical_rad21_model=hierarchical_rad21_model, input_track_names=input_track_names, input_track_paths=input_track_paths, model_path=model_path, other_regions=other_regions, seq_region=seq_region)

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

    ctcf_region, atac_region, other_regions, enformer_perturbed_track_names = apply_enformer_seq_ko(
        assembly=assembly, atac_region=atac_region, bigwig_log_transform=bigwig_log_transform, celltype=celltype, chr_name=chr_name, ctcf_region=ctcf_region, enformer_delta_cap=enformer_delta_cap, enformer_delta_mode=enformer_delta_mode, enformer_model_path=enformer_model_path, enformer_seq_active=enformer_seq_active, enformer_tracks=enformer_tracks, input_track_names=input_track_names, input_track_paths=input_track_paths, other_regions=other_regions, seq_region=seq_region, seq_region_wt=seq_region_wt, start=start, window=window)

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
    other_regions = apply_rad21_update(
        atac_region=atac_region, atac_region_wt=atac_region_wt, chr_name=chr_name, ctcf_region=ctcf_region, ctcf_region_wt=ctcf_region_wt, hierarchical_delta_cap=hierarchical_delta_cap, hierarchical_delta_mode=hierarchical_delta_mode, hierarchical_rad21_model=hierarchical_rad21_model, input_track_names=input_track_names, input_track_paths=input_track_paths, other_regions=other_regions, other_regions_wt=other_regions_wt, seq_region=seq_region, seq_region_wt=seq_region_wt, start=start, window=window)

    rewrite_enformer_ko_tracks(
        atac_region=atac_region, bigwig_log_transform=bigwig_log_transform, chr_name=chr_name, ctcf_region=ctcf_region, enformer_perturbed_track_names=enformer_perturbed_track_names, input_track_names=input_track_names, input_track_paths=input_track_paths, other_regions=other_regions, start=start, window=window)

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

    # pyGenomeTracks .ini files (extracted; verbatim logic in output/tracks_ini.py)
    build_track_inis(
        assembly=assembly, celltype=celltype, chr_name=chr_name, ctcf_motif_p=ctcf_motif_p,
        ctcf_path=ctcf_path, enformer_perturbed_track_names=enformer_perturbed_track_names, enformer_seq_active=enformer_seq_active, hierarchical_active=hierarchical_active,
        input_track_names=input_track_names, input_track_paths=input_track_paths, ko_data=ko_data, max_val_diff=max_val_diff,
        max_val_pred=max_val_pred, max_val_true=max_val_true, min_val_diff=min_val_diff, min_val_pred=min_val_pred,
        min_val_true=min_val_true, plot_bigwig_q=plot_bigwig_q, plot_diff=plot_diff, plot_ground_truth=plot_ground_truth,
        plot_pred_bigwigs=plot_pred_bigwigs, plot_pred_log2fc=plot_pred_log2fc, plot_track_names=plot_track_names, plot_track_paths=plot_track_paths,
        start=start,
    )

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
