import os
#os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
import numpy as np
import pandas as pd
import sys
import cooler
import seaborn as sns
import matplotlib.pyplot as plt
from tqdm import tqdm
from importlib.resources import files
from skimage.transform import resize

from cshark.data.data_feature import GenomicFeature, HiCFeature, SequenceFeature
import cshark.inference.utils.inference_utils as infer
from cshark.inference.utils.inference_utils import write_tmp_cooler, write_tmp_chipseq_ko, knockout_peaks, get_axis_range_from_bigwig, chunk_shuffle, write_tmp_pred_bigwig, oe_normalize_cooler
from cshark.inference.utils import plot_utils, model_utils
from cshark.inference.utils.enformer_utils import (
    load_enformer_pretrained,
    load_enformer_from_checkpoint,
    enformer_seq_knockout,
    write_tmp_enformer_ko_bigwig,
    write_tmp_enformer_delta_bigwig,
)
from cshark.inference.utils.hierarchical_utils import (
    load_hierarchical_rad21_predictor,
    hierarchical_rad21_update,
    write_tmp_hierarchical_rad21_bigwig,
    write_tmp_hierarchical_delta_bigwig,
)
from cshark.inference.tracks_files import get_tracks

import argparse

window = 2097152
res = 8192
image_scale = 256
en_dict = {'a' : 0, 't' : 1, 'c' : 2, 'g' : 3, 'n' : 4}
font_size = 15
plot_width = 17
track_label_fraction = 0.13

def reverse_complement(seq):
    seq = np.flip(seq, 0)
    seq_comp = np.concatenate([seq[:, 1:2],
                                seq[:, 0:1],
                                seq[:, 3:4],
                                seq[:, 2:3],
                                seq[:, 4:5]], axis = 1)
    return seq_comp

# https://sumit-ghosh.com/posts/parsing-dictionary-key-value-pairs-kwargs-argparse-python/
class ParseKwargs(argparse.Action):
    def __call__(self, parser, namespace, values, option_string=None):
        setattr(namespace, self.dest, dict())
        for value in values:
            key, value = value.split('=')
            getattr(namespace, self.dest)[key] = value

def main():
    global window, res, image_scale
    parser = argparse.ArgumentParser(description='C.Origami Editing Module.')

    parser.add_argument('--out', dest='output_path',
                        default='outputs',
                        help='output path for storing results (default: %(default)s)')

    parser.add_argument('--celltype', dest='celltype',
                        help='Sample cell type for prediction, used for output separation', required=True)
    parser.add_argument('--assembly', dest='assembly', default='hg19',
                        help='Genome assembly version (hg19, hg38, mm10)')
    parser.add_argument('--outname', dest='outname', default='',
                        help='Output prefix for saving plots and predictions')
    parser.add_argument('--chr', dest='chr_name',
                        help='Chromosome for prediction', required=True)
    parser.add_argument('--start', dest='start', type=int,
                        help='Starting point for prediction', required=False)
    parser.add_argument('--model', dest='model_path',
                        help='Path to the model checkpoint', required=True)
    parser.add_argument('--resolution', dest='resolution', type=int, default=8192,
                      help='Resolution (bp) of output Hi-C matrix')
    parser.add_argument('--resolution-1d', dest='resolution_1d', type=int, default=256,
                      help='Resolution (bp) of output 1D tracks')
    parser.add_argument('--matrix-size', dest='mat_size', type=int, default=256,
                      help='Size of output Hi-C matrix')
    parser.add_argument('--latent_size', dest='mid_hidden', type=int, default=256,
                        help='', required=False)
    parser.add_argument('--seq-filter-size', dest='seq_filter_size', type=int, default=3,
                        help='Size of the 1D conv filter for sequence data', required=False)
    parser.add_argument('--no-recon', dest='recon_1d',
                        action='store_false',
                        help='Whether to reconstruct 1D tracks from full features or from sequence only')
    parser.add_argument('--no-hic-log-transform', dest='hic_log_transform',
                        action='store_false',
                        help='Whether to apply log transformation to Hi-C matrices')
    parser.add_argument('--no-bigwig-log-transform', dest='bigwig_log_transform',
                        action='store_false',
                        help='Whether to apply log transformation to bigwig tracks')
    parser.add_argument('--oe-norm', dest='oe_norm',
                        action='store_true',
                        help='Whether to apply observed/expected normalization to Hi-C matrices')

    parser.add_argument('--out-file', dest='out_file',
                        help='Path to the output file for full chromosome prediction', required=False)
    parser.add_argument('--seq', dest='seq_path',
                        help='Path to the folder where the sequence .fa.gz files are stored', required=True)
    parser.add_argument('--seq2', dest='seq2_path',
                        help='Path to the second allele sequence .fa.gz files', required=False)
    parser.add_argument('--bigwigs', nargs='*', help='Paths to the bigwig files for genomic features', required=False,
                        action=ParseKwargs)
    parser.add_argument('--plot-bigwigs', dest='plot_bigwigs', nargs='+',
                        help='Names of bigwig tracks to plot (not inputs)', required=False)
    parser.add_argument('--plot-pred-bigwigs', dest='plot_pred_bigwigs', nargs='+',
                        help='Names of predicted bigwig tracks to plot', required=False)
    parser.add_argument('--ko', dest='ko_data', type=str, nargs='+', default=[],
                        help='name of data modalities to knockout', required=False)
    parser.add_argument('--ko-mode', dest='ko_mode', type=str, nargs='+', default=['zero'],
                        help='how we simulate the knockout of 1d peaks (zero, mean, knockout, shuffle, knockout_shuffle, reverse, reverse_motif)', required=False)
    parser.add_argument('--ko-start', dest='deletion_start', nargs='+', type=int,
                        help='Starting points for deletion.', required=False)
    parser.add_argument('--ko-width', dest='deletion_width', nargs='+', type=int,
                        help='Width for deletion.', required=False)
    parser.add_argument('--peak-height', dest='peak_height', nargs='+', type=float,
                        help='Peak height threshold for knockout.', default=2.0)
    parser.add_argument('--alt', dest='alt_bp', type=str, nargs='+',
                        help='Alt base(s) for seq / enformer_seq ko-modes.',
                        required=False)
    parser.add_argument('--padding', dest='end_padding_type',
                        default='zero',
                        help='Padding type, either zero or follow (default: %(default)s)')
    parser.add_argument('--hide-line', dest='hide_deletion_line',
                        action = 'store_true',
                        help='Remove the line showing deletion site')
    parser.add_argument('--whitespace', dest='whitespace', action = 'store_true',
                        help='Add whitespace around the deletion site for better visualization')
    parser.add_argument('--region', '--locus', dest='region',
                        help='specific region to visualize, otherwise full 2Mb window', required=False)

    # Screening params
    parser.add_argument('--screen-start', dest='screen_start', type=int,
                        help='Starting point for screening.', required=False)
    parser.add_argument('--screen-end', dest='screen_end', type=int,
                        help='Ending point for screening.', required=False)
    parser.add_argument('--step-size', dest='step_size', type=int, default=1000,
                        help='step size of perturbations in screening.', required=False)
    parser.add_argument('--n-top-sites', dest='n_top_sites', type=int, default=5,
                        help='number of most impactful sites to visualize after screening', required=False)
    parser.add_argument('--plot-diff', dest='plot_diff',
                        action = 'store_true',
                        help='plot the difference heatmap instead of comparisons')
    parser.add_argument('--silent', dest='silent',
                        action = 'store_true',
                        help='do not print out pyGenomeTracks logs')
    parser.add_argument('--load-screen', dest='load_screen',
                        action = 'store_true',
                        help='load the screen results from a saved bedgraph')

    parser.add_argument('--n-overlap-pred', dest='n_overlap_preds', type=int, default=2,
                        help='Number of predictions for each pixel (controls step size of sliding window)', required=False)

    # Plotting params
    parser.add_argument('--min-val-true', dest='min_val_true', type=float, default=0.5,
                        help='min value for color scale of ground truth data', required=False)
    parser.add_argument('--max-val-true', dest='max_val_true', type=float, default=None,
                        help='max value for color scale of ground truth data', required=False)
    parser.add_argument('--min-val-pred', dest='min_val_pred', type=float, default=0.1,
                        help='min value for color scale of prediction data', required=False)
    parser.add_argument('--max-val-pred', dest='max_val_pred', type=float, default=None,
                        help='max value for color scale of prediction data', required=False)
    parser.add_argument('--min-val-diff', dest='min_val_diff', type=float, default=-0.5,
                        help='min value for color scale of diff matrix', required=False)
    parser.add_argument('--max-val-diff', dest='max_val_diff', type=float, default=0.5,
                        help='max value for color scale of diff matrix', required=False)
    parser.add_argument('--track-diff-abs-max', dest='track_diff_abs_max', type=float, default=0.5,
                        help='absolute max value for 1D diff and delta track color scales (default: %(default)s)', required=False)
    parser.add_argument('--plot-bigwig-q', dest='plot_bigwig_q', type=float, default=0.995,
                        help='Quantile cutoff for bigwig plot max values (default: %(default)s)', required=False)
    parser.add_argument('--ctcf-motif-p', dest='ctcf_motif_p', type=int, default=None,
                        help='max p-value (transformed) to display CTCF motif', required=False)
    parser.add_argument('--no-plots', dest='no_plots',
                        action = 'store_true',
                        help='do not generate plots')

    # Enformer-based perturbation params
    parser.add_argument('--enformer-model', dest='enformer_model_path', type=str, default=None,
                        help='Path to a fine-tuned Enformer checkpoint (.ckpt).')
    parser.add_argument('--enformer-delta-mode', dest='enformer_delta_mode', type=str, default='multiplicative',
                        help='How to apply Enformer-predicted delta: multiplicative, additive, or prediction')
    parser.add_argument('--enformer-delta-cap', dest='enformer_delta_cap', type=float, default=10.0,
                        help='Cap on fold-change values when using enformer_seq mode (default: 10.0)')
    parser.add_argument('--enformer-tracks', dest='enformer_tracks', type=str, nargs='+',
                        default=['ctcf', 'atac'],
                        help='Target track names for Enformer delta predictions')

    # Hierarchical RAD21 predictor params
    parser.add_argument('--hierarchical-model', dest='hierarchical_model_path', type=str, default=None,
                        help='Path to the hierarchical RAD21 predictor checkpoint (.ckpt).')
    parser.add_argument('--hierarchical-delta-mode', dest='hierarchical_delta_mode', type=str,
                        default='multiplicative',
                        help='How to apply the hierarchical RAD21 delta: multiplicative, additive, or prediction (default: multiplicative)')
    parser.add_argument('--hierarchical-delta-cap', dest='hierarchical_delta_cap', type=float,
                        default=None,
                        help='Cap on fold-change values in hierarchical multiplicative mode')

    args = parser.parse_args(args=None if sys.argv[1:] else ['--help'])

    os.makedirs('tmp', exist_ok=True)

    bigwigs = args.bigwigs
    if bigwigs is None:
        bigwigs = {}
    args.ctcf_path = None
    args.atac_path = None
    if 'ctcf' in bigwigs:
        args.ctcf_path = bigwigs['ctcf']
    if 'atac' in bigwigs:
        args.atac_path = bigwigs['atac']
    other_feats = []
    for key in bigwigs:
        if key not in ['ctcf', 'atac']:
            other_feats.append(bigwigs[key])
    if other_feats == []:
        other_feats = None
    if type(args.ko_data) == str:
        args.ko_data = [args.ko_data]
    if type(args.ko_mode) == str:
        args.ko_mode = [args.ko_mode]

    image_scale = args.mat_size
    res = args.resolution

    if args.start is not None:
        single_deletion(args.output_path, args.outname, args.celltype, args.chr_name, args.start,
                args.deletion_start, args.deletion_width,
                args.alt_bp,
                args.model_path,
                args.seq_path, args.ctcf_path, args.atac_path, other_feats,
                args,
                seq2_path=args.seq2_path, assembly=args.assembly,
                ko_data=args.ko_data, ko_mode=args.ko_mode,
                region=args.region,
                mid_hidden=args.mid_hidden,
                seq_filter_size=args.seq_filter_size,
                bigwig_log_transform=args.bigwig_log_transform,
                recon_1d=args.recon_1d,
                plot_bigwigs=args.plot_bigwigs, plot_pred_bigwigs=args.plot_pred_bigwigs,
                min_val_true=args.min_val_true, max_val_true=args.max_val_true,
                min_val_pred=args.min_val_pred, max_val_pred=args.max_val_pred, plot_diff=args.plot_diff,
                min_val_diff=args.min_val_diff, max_val_diff=args.max_val_diff,
                track_diff_abs_max=args.track_diff_abs_max,
                plot_bigwig_q=args.plot_bigwig_q,
                peak_height=args.peak_height,
                ctcf_motif_p=args.ctcf_motif_p,
                undo_log=args.hic_log_transform,
                no_plots=args.no_plots,
                silent=args.silent,
                enformer_model_path=args.enformer_model_path,
                enformer_delta_mode=args.enformer_delta_mode,
                enformer_delta_cap=args.enformer_delta_cap,
                enformer_tracks=args.enformer_tracks,
                hierarchical_model_path=args.hierarchical_model_path,
                hierarchical_delta_mode=args.hierarchical_delta_mode,
                hierarchical_delta_cap=args.hierarchical_delta_cap)
    else:
        # Full chromosome prediction
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

        results_hierarchical = {'chrom': [], 'start': [], 'end': []}
        if use_hierarchical:
            for col in ['rad21_WT_pred', 'rad21_KO_pred', 'rad21_delta', 'rad21_fc',
                        'rad21_perturbed', 'rad21_experimental']:
                results_hierarchical[col] = []

        track_names = []
        results_1d = {'chrom': [], 'start': [], 'end': []}
        bins_1d = []

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

            pred_before_output = infer.prediction(seq_region, ctcf_region, atac_region, model_path,
                                                  other_regions,
                                                  num_genomic_features=num_genomic_features,
                                                  mat_size=image_scale, diploid=diploid,
                                                  target_1d_length=int(window / args.resolution_1d),
                                                  mid_hidden=mid_hidden,
                                                  seq_filter_size=args.seq_filter_size,
                                                  recon_1d=args.recon_1d,
                                                  undo_log=args.hic_log_transform,
                                                  other_feat_names=input_track_names[2:])
            pred_before = pred_before_output['hic']

            # Save WT copies for hierarchical delta
            if use_hierarchical:
                seq_region_wt = seq_region.copy()
                ctcf_region_wt = ctcf_region.copy() if ctcf_region is not None else None
                atac_region_wt = atac_region.copy() if atac_region is not None else None
                other_regions_wt = [r.copy() for r in other_regions] if other_regions is not None else None
                experimental_rad21 = other_regions[rad21_other_idx_hier].copy()

            seq_region, ctcf_region, atac_region, other_regions = deletion_with_padding(
                chr_name, start, start, window, seq_region, ctcf_region,
                atac_region, other_regions, ko_data=ko_data, ko_channels=ko_channels,
                channel_offset=channel_offset, ko_mode=ko_mode,
                peak_height=args.peak_height)

            if use_hierarchical and other_regions is not None:
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

            pred_output = infer.prediction(seq_region, ctcf_region, atac_region, model_path,
                                           other_regions,
                                           num_genomic_features=num_genomic_features,
                                           mat_size=image_scale, diploid=diploid,
                                           target_1d_length=int(window / args.resolution_1d),
                                           mid_hidden=mid_hidden,
                                           seq_filter_size=args.seq_filter_size,
                                           recon_1d=args.recon_1d,
                                           undo_log=args.hic_log_transform,
                                           other_feat_names=input_track_names[2:])
            pred = pred_output['hic']

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
        if region is not None:
            res_df = res_df[(res_df['start1'] >= region_start) & (res_df['end1'] <= region_end) &
                            (res_df['start2'] >= region_start) & (res_df['end2'] <= region_end)]
            bins_df = bins_df[(bins_df['start'] >= region_start) & (bins_df['end'] <= region_end)]
        res_df.to_csv(args.out_file, sep='\t', header=True, index=False)
        bins_df.to_csv(args.out_file.replace('.tsv', '_bins.tsv'), sep='\t', header=False, index=False)

        # Cooler outputs
        bins_cooler_df = bins_df[['chrom', 'start', 'end']].copy()
        bins_cooler_df.reset_index(inplace=True)
        wt_cooler_df = res_df[['a1', 'a2', 'WT']].rename(columns={'WT': 'count'})
        ko_cooler_df = res_df[['a1', 'a2', 'KO']].rename(columns={'KO': 'count'})
        for df in [wt_cooler_df, ko_cooler_df]:
            df['bin1_id'] = df['a1'].map(lambda x: int(x.replace('A_', '')))
            df['bin2_id'] = df['a2'].map(lambda x: int(x.replace('A_', '')))
        wt_cooler_df = wt_cooler_df[['bin1_id', 'bin2_id', 'count']]
        ko_cooler_df = ko_cooler_df[['bin1_id', 'bin2_id', 'count']]
        cooler.create_cooler(args.out_file.replace('.tsv', '_WT.cool'), bins_cooler_df, wt_cooler_df, ordered=True, dtypes={'count': 'float32'})
        cooler.create_cooler(args.out_file.replace('.tsv', '_KO.cool'), bins_cooler_df, ko_cooler_df, ordered=True, dtypes={'count': 'float32'})

        if args.oe_norm:
            exp_wt_cooler_df = res_df[['a1', 'a2', 'exp_WT']].rename(columns={'exp_WT': 'count'})
            exp_wt_cooler_df['bin1_id'] = exp_wt_cooler_df['a1'].map(lambda x: int(x.replace('A_', '')))
            exp_wt_cooler_df['bin2_id'] = exp_wt_cooler_df['a2'].map(lambda x: int(x.replace('A_', '')))
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

        fig, ax = plt.subplots(figsize=(10, 10))
        sns.scatterplot(data=res_df, x='WT', y='KO', ax=ax)
        ax.set_title('WT vs KO')
        plt.savefig(os.path.join(args.output_path, f'{args.outname}{args.celltype}_{args.chr_name}_scatter.png'), dpi=300)
        plt.close(fig)


def single_deletion(output_path, outname, celltype, chr_name, start, deletion_starts, deletion_widths,
                    alt_bp,
                    model_path, seq_path, ctcf_path, atac_path, other_feats,
                    args,
                    seq2_path=None, assembly='hg19',
                    ko_data=['ctcf'], ko_mode=['zero'], region=None, mid_hidden=256, seq_filter_size=3,
                    recon_1d=True, bigwig_log_transform=True,
                    plot_bigwigs=[], plot_pred_bigwigs=[], plot_pred_log2fc=False,
                    min_val_true=1.0, max_val_true=None, min_val_pred=0.1, max_val_pred=None,
                    plot_diff=False, min_val_diff=-0.5, max_val_diff=0.5,
                    track_diff_abs_max=0.5,
                    plot_bigwig_q=0.995, peak_height=2.0, ctcf_motif_p=500,
                    undo_log=True, no_plots=False, silent=False,
                    enformer_model_path=None, enformer_delta_mode='multiplicative',
                    enformer_delta_cap=10.0, enformer_tracks=None,
                    hierarchical_model_path=None, hierarchical_delta_mode='multiplicative',
                    hierarchical_delta_cap=None):
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

    # Baseline prediction (WT)
    pred_before_output = infer.prediction(seq_region, ctcf_region, atac_region, model_path,
                                          other_regions,
                                          num_genomic_features=num_genomic_features,
                                          mat_size=image_scale, diploid=diploid,
                                          mid_hidden=mid_hidden, seq_filter_size=seq_filter_size,
                                          target_1d_length=int(window / args.resolution_1d),
                                          recon_1d=recon_1d, undo_log=undo_log,
                                          bigwig_log=bigwig_log_transform,
                                          other_feat_names=input_track_names[2:])
    pred_before = pred_before_output['hic']
    if not no_plots:
        plt.imshow(pred_before, cmap='Reds')
        plt.colorbar()
        plt.title('Prediction before perturbation')
        plt.savefig(os.path.join(output_path, f'{outname}{celltype}_{chr_name}_{start}_pred_before.png'), dpi=300)
        plt.close()
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
    pred_output = infer.prediction(seq_region, ctcf_region, atac_region, model_path,
                                   other_regions,
                                   num_genomic_features=num_genomic_features,
                                   mat_size=image_scale,
                                   target_1d_length=int(window / args.resolution_1d),
                                   diploid=diploid, mid_hidden=mid_hidden,
                                   seq_filter_size=seq_filter_size,
                                   recon_1d=recon_1d, undo_log=undo_log,
                                   bigwig_log=bigwig_log_transform,
                                   other_feat_names=input_track_names[2:])
    pred = pred_output['hic']

    if not no_plots:
        plt.imshow(pred, cmap='Reds')
        plt.colorbar()
        plt.title('Prediction after perturbation')
        plt.savefig(os.path.join(output_path, f'{outname}{celltype}_{chr_name}_{start}_pred.png'), dpi=300)
        plt.close()
    pred_1d = pred_output['1d']

    if 'del' in ko_mode or 'deletion' in ko_mode or 'delete' in ko_mode and args.whitespace:
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

    with open('tmp/regions.bed', 'w') as f:
        if deletion_starts is not None and deletion_widths is not None:
            for deletion_start, deletion_width in zip(deletion_starts, deletion_widths):
                f.write(f'{chr_name}\t{deletion_start}\t{deletion_start + deletion_width}\n')

    baseline_cutoff = np.quantile(pred_before, 0.99)
    cutoff = np.quantile(pred, 0.99)
    if plot_diff:
        diff_cutoff_gain = np.quantile(diff[diff > 0], 0.99) if np.sum(diff > 0) > 0 else 0.0
        diff_cutoff_loss = np.quantile(diff[diff < 0], 0.01) if np.sum(diff < 0) > 0 else 0.0
    if plot_ground_truth:
        ground_truth_cutoff = np.quantile(mat, 0.99)
    region_start = int(region.split(':')[1].split('-')[0]) if region is not None else start
    region_end = int(region.split(':')[1].split('-')[1]) if region is not None else start + window

    with open('tmp/arcs.bed', 'w') as f:
        for i in range(pred_before.shape[0]):
            for j in range(pred_before.shape[1]):
                pixel_start_i = i * res + start
                pixel_end_i = i * res + start + res
                pixel_start_j = j * res + start
                pixel_end_j = j * res + start + res
                if (pred_before[i, j] > baseline_cutoff and
                        pixel_start_i > region_start and pixel_end_i < region_end and
                        pixel_start_j > region_start and pixel_end_j < region_end):
                    f.write(f'{chr_name}\t{pixel_start_i}\t{pixel_end_i}\t{chr_name}\t{pixel_start_j}\t{pixel_end_j}\t{pred_before[i, j]}\n')
    if plot_diff:
        with open('tmp/arcs_diff.bed', 'w') as f:
            for i in range(diff.shape[0]):
                for j in range(diff.shape[1]):
                    pixel_start_i = i * res + start
                    pixel_end_i = i * res + start + res
                    pixel_start_j = j * res + start
                    pixel_end_j = j * res + start + res
                    if ((diff[i, j] > diff_cutoff_gain or diff[i, j] < diff_cutoff_loss) and
                            pixel_start_i > region_start and pixel_end_i < region_end and
                            pixel_start_j > region_start and pixel_end_j < region_end):
                        f.write(f'{chr_name}\t{pixel_start_i}\t{pixel_end_i}\t{chr_name}\t{pixel_start_j}\t{pixel_end_j}\t{diff[i, j]}\n')

    with open('tmp/arcs_ko.bed', 'w') as f:
        for i in range(pred.shape[0]):
            for j in range(pred.shape[1]):
                pixel_start_i = i * res + start
                pixel_end_i = i * res + start + res
                pixel_start_j = j * res + start
                pixel_end_j = j * res + start + res
                if (pred[i, j] > cutoff and
                        pixel_start_i > region_start and pixel_end_i < region_end and
                        pixel_start_j > region_start and pixel_end_j < region_end):
                    f.write(f'{chr_name}\t{pixel_start_i}\t{pixel_end_i}\t{chr_name}\t{pixel_start_j}\t{pixel_end_j}\t{pred[i, j]}\n')
    if plot_ground_truth:
        with open('tmp/arcs_true.bed', 'w') as f:
            for i in range(mat.shape[0]):
                for j in range(mat.shape[1]):
                    pixel_start_i = i * res + start
                    pixel_end_i = i * res + start + res
                    pixel_start_j = j * res + start
                    pixel_end_j = j * res + start + res
                    if (mat[i, j] > ground_truth_cutoff and
                            pixel_start_i > region_start and pixel_end_i < region_end and
                            pixel_start_j > region_start and pixel_end_j < region_end):
                        f.write(f'{chr_name}\t{pixel_start_i}\t{pixel_end_i}\t{chr_name}\t{pixel_start_j}\t{pixel_end_j}\t{mat[i, j]}\n')

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
                                f.write(f'min_value = {-track_diff_abs_max}\n')
                                f.write(f'max_value = {track_diff_abs_max}\n')
                                f.write('number_of_bins = 512\n\n')
                            if hierarchical_active and canonical_track_name == 'rad21':
                                if os.path.exists('tmp/rad21_hierarchical_delta.bw'):
                                    f.write('[RAD21 Hier. Delta]\n')
                                    f.write('file = tmp/rad21_hierarchical_delta.bw\n')
                                    f.write('height = 2\n')
                                    f.write('color = red\n')
                                    f.write('negative_color = blue\n')
                                    f.write('title = RAD21 Hier. Delta\n')
                                    f.write(f'min_value = {-track_diff_abs_max}\n')
                                    f.write(f'max_value = {track_diff_abs_max}\n')
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
                                f.write(f'min_value = {-track_diff_abs_max}\n')
                                f.write(f'max_value = {track_diff_abs_max}\n')
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


def deletion_with_padding(chr_name, start, deletion_start, deletion_width, seq_region, ctcf_region, atac_region,
                          other_regions=None, ko_data=['ctcf'], ko_channels=[0], channel_offset=0, ko_mode=['zero'],
                          peak_height=2.0, left_del_pad=None, right_del_pad=None):
    ''' Delete all signals at a specified location with corresponding padding at the end '''
    for track_name, knockout_mode, channel_idx in zip(ko_data, ko_mode, ko_channels):
        if track_name == 'ctcf':
            ctcf_region = track_ko(deletion_start - start,
                deletion_start - start + deletion_width,
                ctcf_region, ko_mode=knockout_mode, peak_height=peak_height)
        elif track_name == 'atac':
            atac_region = track_ko(deletion_start - start,
                deletion_start - start + deletion_width,
                atac_region, ko_mode=knockout_mode, peak_height=peak_height)
        elif track_name == 'seq':
            if knockout_mode == 'knockout' or knockout_mode == 'zero':
                seq_region[deletion_start - start:deletion_start - start + deletion_width, :] = 0
                seq_region[deletion_start - start:deletion_start - start + deletion_width, 4] = 1
            elif knockout_mode in ('del', 'deletion', 'delete'):
                left_seq_pad, left_ctcf_pad, left_atac_pad, left_other_pads = left_del_pad
                right_seq_pad, right_ctcf_pad, right_atac_pad, right_other_pads = right_del_pad
                print(left_seq_pad.shape, seq_region.shape, seq_region[:deletion_start - start, :].shape)
                seq_region = np.concatenate((left_seq_pad, seq_region[:deletion_start - start, :],
                                            seq_region[deletion_start - start + deletion_width:, :], right_seq_pad), axis=0)
                ctcf_region = np.concatenate((left_ctcf_pad, ctcf_region[:deletion_start - start],
                                            ctcf_region[deletion_start - start + deletion_width:], right_ctcf_pad), axis=0)
                atac_region = np.concatenate((left_atac_pad, atac_region[:deletion_start - start],
                                            atac_region[deletion_start - start + deletion_width:], right_atac_pad), axis=0)
                if other_regions is not None:
                    for i in range(len(other_regions)):
                        other_regions[i] = np.concatenate((left_other_pads[i], other_regions[i][:deletion_start - start],
                                                    other_regions[i][deletion_start - start + deletion_width:], right_other_pads[i]), axis=0)
            elif knockout_mode == 'shuffle':
                idxs = np.arange(seq_region[deletion_start - start:deletion_start - start + deletion_width, :].shape[0])
                np.random.shuffle(idxs)
                seq_region[deletion_start - start:deletion_start - start + deletion_width, :] = seq_region[deletion_start - start:deletion_start - start + deletion_width, :][idxs, :]
            elif knockout_mode == 'random':
                rand_bases = np.random.choice(4, size=(deletion_width,))
                rand_seq = np.zeros((deletion_width, 5), dtype=np.float32)
                for i in range(4):
                    rand_seq[:, i] = (rand_bases == i).astype(np.float32)
                if deletion_start - start >= 0 and deletion_start - start + deletion_width <= seq_region.shape[0]:
                    seq_region[deletion_start - start:deletion_start - start + deletion_width, :] = rand_seq
            elif knockout_mode == 'reverse':
                seq_region[deletion_start - start:deletion_start - start + deletion_width, :] = reverse_complement(seq_region[deletion_start - start:deletion_start - start + deletion_width, :])
            elif knockout_mode == 'reverse_motif':
                from pyjaspar import jaspardb
                jdb_obj = jaspardb(release='JASPAR2024')
                motifs = jdb_obj.fetch_motifs(
                        collection=['CORE'], tf_name='CTCF',
                        tax_group=['Vertebrates'], species=['9606'], all_versions=False)
                motif = motifs[0]
                matrix_dict = motif.counts.normalize()
                matrix = []
                for base in ['A', 'T', 'C', 'G', 'N']:
                    if base in matrix_dict:
                        matrix.append(list(matrix_dict[base]))
                    else:
                        matrix.append([0] * len(matrix_dict['A']))
                matrix = np.array(matrix).T
                seq_scan = seq_region[deletion_start - start:deletion_start - start + deletion_width, :]
                corrs = []
                is_reverse = []
                for i in range(seq_scan.shape[0]):
                    try:
                        corr = np.dot(seq_scan[i: i + matrix.shape[0], :].flatten(), matrix.flatten()) / matrix.shape[0]
                        rc_seq = reverse_complement(seq_scan[i: i + matrix.shape[0], :])
                        corr_reverse = np.dot(rc_seq.flatten(), matrix.flatten()) / matrix.shape[0]
                        is_reverse.append(corr_reverse > corr)
                        corr = max(corr, corr_reverse)
                        corrs.append(corr)
                    except Exception:
                        break
                corrs = np.array(corrs)
                max_idx = np.argmax(corrs)
                ref_bases = []
                for i in range(deletion_start - start + max_idx, deletion_start - start + max_idx + matrix.shape[0]):
                    if i < 0 or i >= len(seq_region):
                        ref_bases.append('N')
                    else:
                        ref_bases.append(list(en_dict.keys())[list(en_dict.values()).index(seq_region[i].argmax())])
                ref_bases = ''.join(ref_bases).upper()
                top_n = np.sum(corrs > 0.65)
                if top_n > 0:
                    max_idxs = np.argsort(corrs)[-top_n:]
                    forward_motif_xs = []
                    forward_motif_ys = []
                    reverse_motif_xs = []
                    reverse_motif_ys = []
                    for i in max_idxs:
                        ref_bases = []
                        for j in range(deletion_start - start + i, deletion_start - start + i + matrix.shape[0]):
                            if j < 0 or j >= len(seq_region):
                                ref_bases.append('N')
                            else:
                                ref_bases.append(list(en_dict.keys())[list(en_dict.values()).index(seq_region[j].argmax())])
                        ref_bases = ''.join(ref_bases).upper()
                        print(f'{chr_name}:{deletion_start + i} - {ref_bases} {"<" if is_reverse[i] else ">"} (corr: {corrs[i]:.3f})')
                        if is_reverse[i]:
                            reverse_motif_xs.append(i)
                            reverse_motif_ys.append(corrs[i])
                        else:
                            forward_motif_xs.append(i)
                            forward_motif_ys.append(corrs[i])
                        motif_seq = seq_region[deletion_start - start + i: deletion_start - start + i + matrix.shape[0], :]
                        rc_motif_seq = reverse_complement(motif_seq)
                        seq_region[deletion_start - start + i: deletion_start - start + i + matrix.shape[0], :] = rc_motif_seq

                    fig = plt.figure(figsize=(15, 4))
                    plt.plot(corrs)
                    plt.scatter(forward_motif_xs, forward_motif_ys, color='blue', marker='>', label='Forward motif')
                    plt.scatter(reverse_motif_xs, reverse_motif_ys, color='red', marker='<', label='Reverse motif')
                    plt.savefig('tmp/ctcf_corr.png')
                    plt.close()

                    with open('tmp/ctcf_motif.bed', 'w') as f:
                        for i in max_idxs:
                            f.write(f'{chr_name}\t{deletion_start + i}\t{deletion_start + i + matrix.shape[0]}\t{"<" if is_reverse[i] else ">"}\t{corrs[i]:.3f}\n')
                else:
                    print('No motifs found with correlation > 0.65')

        elif other_regions is not None:
            original = other_regions[channel_idx - channel_offset].copy()
            other_regions[channel_idx - channel_offset] = track_ko(deletion_start - start,
                deletion_start - start + deletion_width,
                other_regions[channel_idx - channel_offset], ko_mode=knockout_mode, peak_height=peak_height)
            if np.array_equal(original, other_regions[channel_idx - channel_offset]):
                print(f'Warning: {track_name} KO did not change the signal. Check the KO mode.')
    return seq_region, ctcf_region, atac_region, other_regions


def track_ko(start, end, track, window=2097152, ko_mode='zero', peak_height=2.0):
    if ko_mode == 'zero':
        track[start:end] = 0
    elif ko_mode == 'mean':
        mean = np.mean(np.concatenate([track[:start], track[end:]]))
        track[start:end] = mean
    elif ko_mode == 'knockout':
        track[start:end] = knockout_peaks(track[start:end], threshold=peak_height)
    elif 'increase' in ko_mode:
        increase_factor = float(ko_mode.split('_')[1]) if '_' in ko_mode else 2.0
        track[start:end] = knockout_peaks(track[start:end], threshold=peak_height, increase_factor=increase_factor)
    elif 'cluster' in ko_mode:
        cluster_ratio = float(ko_mode.split('_')[1]) if '_' in ko_mode else 0.05
        cluster_indices = np.random.choice(np.arange(start, end), size=int((end - start) * cluster_ratio), replace=False)
        for idx in cluster_indices:
            track[idx] = np.random.uniform(1, 5)
    elif ko_mode == 'shuffle':
        track[start:end] = chunk_shuffle(track[start:end])
    elif ko_mode == 'knockout_shuffle':
        track[start:end] = knockout_peaks(track[start:end], threshold=peak_height)
        track[start:end] = chunk_shuffle(track[start:end])
    elif ko_mode in ('reverse', 'reverse_motif'):
        track[start:end] = track[start:end][::-1]
    else:
        raise ValueError(f'ko_mode must be one of: zero, mean, knockout, increase, cluster, shuffle, knockout_shuffle, reverse, reverse_motif')
    return track[:window]


def seq_perturb(start, alt, seq, window=2097152):
    """Simulate DNA sequence variants."""
    new_entry = np.zeros(5)
    alt_idx = en_dict[alt.lower()]
    new_entry[alt_idx] = 1
    ref_entry = seq[start, :]
    ref = ref_entry.argmax()
    ref_base = list(en_dict.keys())[list(en_dict.values()).index(ref)]
    print(f'Pos: {start}, Alt: {alt}, Ref: {ref_base.upper()}')
    if ref == alt_idx:
        print('No change')
    ref_bases = []
    for i in range(start - 10, start + 10):
        if i == start:
            ref_bases.append('*')
        if i < 0 or i >= len(seq):
            ref_bases.append('N')
        else:
            ref_bases.append(list(en_dict.keys())[list(en_dict.values()).index(seq[i].argmax())])
        if i == start:
            ref_bases.append('*')
    print(''.join(ref_bases).upper())
    seq[start, :] = new_entry
    return seq[:window]


if __name__ == '__main__':
    main()
