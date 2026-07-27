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
from cshark.perturb.output.plots import plot_prediction_matrix, plot_pred_1d_tracks
from cshark.perturb.models.base import CSharkModel
from cshark.perturb.output.tracks_ini import build_track_inis, run_pygenometracks
from cshark.perturb.models.hierarchical import prepare_rad21_input, apply_rad21_update
from cshark.perturb.models.enformer import apply_enformer_seq_ko, rewrite_enformer_ko_tracks
from cshark.perturb.models.alphagenome import apply_alphagenome_seq_ko
from cshark.perturb.operators.planning import plan_perturbations
from cshark.perturb.seq_source import load_alt_fasta_region, align_alt_to_wt

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
    alt_fasta = cfg.alt_fasta
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
    alphagenome_model_path = cfg.alphagenome_model_path
    alphagenome_metadata_path = cfg.alphagenome_metadata_path
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

    seq_region, deletion_widths, pending_track_perturbations, enformer_seq_active, alphagenome_seq_active, hierarchical_active = plan_perturbations(
        alt_bp=alt_bp, atac_path=atac_path, bigwig_log_transform=bigwig_log_transform, channel_offset=channel_offset, chr_name=chr_name, ctcf_path=ctcf_path, deletion_starts=deletion_starts, deletion_widths=deletion_widths, hierarchical_rad21_model=hierarchical_rad21_model, input_track_names=input_track_names, ko_data_types=ko_data_types, ko_mode=ko_mode, other_feats=other_feats, peak_height=peak_height, seq2_path=seq2_path, seq_path=seq_path, seq_region=seq_region, start=start, window=window)

    # --- whole-window ALT sequence from --alt-fasta -------------------------
    # Treat every base in the alternate fasta directory as the ALT sequence for
    # the seq / enformer_seq / alphagenome_seq ko-modes: replace the entire
    # window's sequence with the alternate genome (the WT copy saved above is the
    # delta baseline) and activate the backbone selected by --ko-mode. This
    # supersedes any --alt/--ko-start sequence edits for this run.
    if alt_fasta is not None:
        n_alleles = max(1, seq_region.shape[1] // 5)
        alt_region = load_alt_fasta_region(alt_fasta, chr_name, start, window, n_alleles)
        seq_region = align_alt_to_wt(alt_region, seq_region)
        _modes = set(ko_mode or [])
        if 'enformer_seq' in _modes:
            enformer_seq_active = True
        if 'alphagenome_seq' in _modes:
            alphagenome_seq_active = True
        if not (_modes & {'seq', 'enformer_seq', 'alphagenome_seq'}):
            print('[alt-fasta] WARNING: --ko-mode has none of seq/enformer_seq/'
                  'alphagenome_seq; applying the alt sequence as a plain main-model '
                  '(seq) substitution.')
        print(f'[alt-fasta] Loaded whole-window ALT sequence from {alt_fasta} for '
              f'{chr_name}:{start}-{start + window} (alleles={n_alleles}); '
              f'enformer_seq_active={enformer_seq_active}, '
              f'alphagenome_seq_active={alphagenome_seq_active}.')

    # --- opt-in allele-specific peak redistribution (--allele-peak-split) ---
    # Only meaningful with an enformer_seq perturbation (it needs the WT-vs-ALT
    # Enformer predictions). The existing path below is left completely untouched;
    # when the flag is off (or there is no enformer_seq) we never enter this branch.
    if cfg.allele_peak_split and enformer_seq_active:
        from cshark.perturb.scopes.allele_split import run_allele_peak_split
        return run_allele_peak_split(
            cfg, model=model, seq_region=seq_region, seq_region_wt=seq_region_wt,
            ctcf_region=ctcf_region, atac_region=atac_region, other_regions=other_regions,
            other_regions_wt=other_regions_wt, input_track_names=input_track_names,
            input_track_paths=input_track_paths, pred_before=pred_before,
            pred_before_1d=pred_before_1d, plot_track_names=plot_track_names,
            plot_track_paths=plot_track_paths, hierarchical_rad21_model=hierarchical_rad21_model,
            deletion_starts=deletion_starts, deletion_widths=deletion_widths,
            res=res, image_scale=image_scale, window=window)
    if cfg.allele_peak_split and not enformer_seq_active:
        print('[allele-peak-split] WARNING: --allele-peak-split needs an enformer_seq '
              'perturbation (--ko seq --ko-mode enformer_seq); using the standard path.')

    ctcf_region, atac_region, other_regions, enformer_perturbed_track_names = apply_enformer_seq_ko(
        assembly=assembly, atac_region=atac_region, bigwig_log_transform=bigwig_log_transform, celltype=celltype, chr_name=chr_name, ctcf_region=ctcf_region, enformer_delta_cap=enformer_delta_cap, enformer_delta_mode=enformer_delta_mode, enformer_model_path=enformer_model_path, enformer_seq_active=enformer_seq_active, enformer_tracks=enformer_tracks, input_track_names=input_track_names, input_track_paths=input_track_paths, other_regions=other_regions, seq_region=seq_region, seq_region_wt=seq_region_wt, start=start, window=window)

    # AlphaGenome sequence-perturbation KO (drop-in counterpart of enformer_seq).
    # Writes the same tmp/{track}_enformer_ko.bw / _enformer_delta.bw plotting
    # bigwigs, so we fold its perturbed-track set into the shared name set below.
    ctcf_region, atac_region, other_regions, alphagenome_perturbed_track_names = apply_alphagenome_seq_ko(
        assembly=assembly, atac_region=atac_region, bigwig_log_transform=bigwig_log_transform, celltype=celltype, chr_name=chr_name, ctcf_region=ctcf_region, enformer_delta_cap=enformer_delta_cap, enformer_delta_mode=enformer_delta_mode, enformer_tracks=enformer_tracks, alphagenome_model_path=alphagenome_model_path, alphagenome_metadata_path=alphagenome_metadata_path, alphagenome_seq_active=alphagenome_seq_active, input_track_names=input_track_names, input_track_paths=input_track_paths, other_regions=other_regions, seq_region=seq_region, seq_region_wt=seq_region_wt, start=start, window=window)

    # Unified flags/track-set for the (backbone-agnostic) plotting pipeline.
    seq_model_active = enformer_seq_active or alphagenome_seq_active
    enformer_perturbed_track_names = set(enformer_perturbed_track_names) | set(alphagenome_perturbed_track_names)
    # Which backbone's KO/delta bigwigs to (re)write and plot: tmp/{track}_{ko_tool}_ko.bw etc.
    ko_tool = 'alphagenome' if alphagenome_seq_active else 'enformer'

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
        atac_region=atac_region, bigwig_log_transform=bigwig_log_transform, chr_name=chr_name, ctcf_region=ctcf_region, enformer_perturbed_track_names=enformer_perturbed_track_names, input_track_names=input_track_names, input_track_paths=input_track_paths, other_regions=other_regions, start=start, window=window, tool=ko_tool)

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
    plot_pred_1d_tracks(
        celltype=celltype, chr_name=chr_name, input_track_names=input_track_names, input_track_paths=input_track_paths, model_path=model_path, no_plots=no_plots, outname=outname, output_path=output_path, plot_pred_bigwigs=plot_pred_bigwigs, plot_pred_log2fc=plot_pred_log2fc, plot_track_names=plot_track_names, pred_1d=pred_1d, pred_before_1d=pred_before_1d, start=start)

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
        ctcf_path=ctcf_path, enformer_perturbed_track_names=enformer_perturbed_track_names, enformer_seq_active=seq_model_active, hierarchical_active=hierarchical_active,
        input_track_names=input_track_names, input_track_paths=input_track_paths, ko_data=ko_data, max_val_diff=max_val_diff,
        max_val_pred=max_val_pred, max_val_true=max_val_true, min_val_diff=min_val_diff, min_val_pred=min_val_pred,
        min_val_true=min_val_true, plot_bigwig_q=plot_bigwig_q, plot_diff=plot_diff, plot_ground_truth=plot_ground_truth,
        plot_pred_bigwigs=plot_pred_bigwigs, plot_pred_log2fc=plot_pred_log2fc, plot_track_names=plot_track_names, plot_track_paths=plot_track_paths,
        start=start, ko_tool=ko_tool,
    )

    run_pygenometracks(
        region=region, celltype=celltype, chr_name=chr_name, deletion_starts=deletion_starts, deletion_widths=deletion_widths, font_size=font_size, no_plots=no_plots, outname=outname, output_path=output_path, plot_diff=plot_diff, plot_ground_truth=plot_ground_truth, plot_width=plot_width, silent=silent, start=start, track_label_fraction=track_label_fraction, window=window, fig_kind=('snp' if seq_model_active else None))
