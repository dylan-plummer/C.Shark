import os
import numpy as np
import pandas as pd
import sys
import cooler 
import seaborn as sns
import matplotlib.pyplot as plt
from tqdm import tqdm
from skimage.transform import resize

from cshark.data.data_feature import GenomicFeature, HiCFeature
import cshark.inference.utils.inference_utils as infer
from cshark.inference.utils.inference_utils import write_tmp_cooler, write_tmp_chipseq_ko, knockout_peaks, get_axis_range_from_bigwig
from cshark.inference.utils import plot_utils, model_utils
from cshark.inference.tracks_files import get_tracks

import argparse

window = 2097152
res = 8192
image_scale = 256
en_dict = {'a' : 0, 't' : 1, 'c' : 2, 'g' : 3, 'n' : 4}
font_size = 15
plot_width = 17
track_label_fraction = 0.13

# https://sumit-ghosh.com/posts/parsing-dictionary-key-value-pairs-kwargs-argparse-python/
class ParseKwargs(argparse.Action):
    def __call__(self, parser, namespace, values, option_string=None):
        setattr(namespace, self.dest, dict())
        for value in values:
            key, value = value.split('=')
            getattr(namespace, self.dest)[key] = value

def main():
    parser = argparse.ArgumentParser(description='C.Origami Editing Module.')
    
    # Output location
    parser.add_argument('--out', dest='output_path', 
                        default='outputs',
                        help='output path for storing results (default: %(default)s)')

    # Location related params
    parser.add_argument('--celltype', dest='celltype', 
                        help='Sample cell type for prediction, used for output separation', required=True)
    parser.add_argument('--outname', dest='outname', default='',
                                help='Output prefix for saving plots and predictions')
    parser.add_argument('--chr', dest='chr_name', 
                        help='Chromosome for prediction', required=True)
    parser.add_argument('--start', dest='start', type=int,
                        help='Starting point for prediction (width is 2097152 bp which is the input window size)', required=False)
    parser.add_argument('--model', dest='model_path', 
                        help='Path to the model checkpoint', required=True)
    parser.add_argument('--latent_size', dest='mid_hidden', type=int, default=256,
                                help='', required=False)
    parser.add_argument('--out-file', dest='out_file', 
                        help='Path to the output file if doing full chromosome prediction', required=False)
    parser.add_argument('--seq', dest='seq_path', 
                        help='Path to the folder where the sequence .fa.gz files are stored', required=True)
    parser.add_argument('--bigwigs', nargs='*', help='Paths to the bigwig files for genomic features', required=True,
                        action=ParseKwargs)
    parser.add_argument('--ko', dest='ko_data', type=str, nargs='+', default=[],
                        help='name of data modalities to knockout', required=False)
    parser.add_argument('--ko-mode', dest='ko_mode', type=str, default='zero',
                        help='min value for color scale of grount truth data', required=False)

    # Deletion related params
    parser.add_argument('--ko-start', dest='deletion_start', nargs='+', type=int,
                        help='Starting points for deletion.', required=False)
    parser.add_argument('--ko-width', dest='deletion_width', nargs='+', type=int,
                        help='Width for deletion.', required=False)
    parser.add_argument('--var-pos', dest='var_pos', type=int, nargs='+',
                        help='Variant position', required=False)
    parser.add_argument('--alt', dest='alt_bp', type=str, nargs='+',
                        help='Variant alt base', required=False)
    parser.add_argument('--padding', dest='end_padding_type', 
                        default='zero',
                        help='Padding type, either zero or follow. Using zero: the missing region at the end will be padded with zero for ctcf and atac seq, while sequence will be padded with N (unknown necleotide). Using follow: the end will be padded with features in the following region (default: %(default)s)')
    parser.add_argument('--hide-line', dest='hide_deletion_line', 
                        action = 'store_true',
                        help='Remove the line showing deletion site (default: %(default)s)')
    parser.add_argument('--region', dest='region', 
                                help='specific region to visualize, otherwise full 2Mb window', required=False)
    
    # Screening related params
    parser.add_argument('--screen-start', dest='screen_start', type=int,
                        help='Starting point for screening.', required=False)
    parser.add_argument('--screen-end', dest='screen_end', type=int,
                        help='Ending point for screening.', required=False)
    parser.add_argument('--perturb-width', dest='perturb_width', type=int, default=1000,
                        help='Width of perturbation used for screening.', required=False)
    parser.add_argument('--step-size', dest='step_size', type=int, default=1000,
                        help='step size of perturbations in screening.', required=False)
    parser.add_argument('--n-top-sites', dest='n_top_sites', type=int, default=5,
                        help='number of most impactful sites to visualize after screening', required=False)
    parser.add_argument('--plot-diff', dest='plot_diff', 
                        action = 'store_true',
                        help='plot the difference heatmap instead of comparisons')
    parser.add_argument('--load-screen', dest='load_screen', 
                        action = 'store_true',
                        help='load the screen results from a saved bedgraph')

    parser.add_argument('--n-overlap-pred', dest='n_overlap_preds', type=int, default=2,
                        help='Number of predictions for each pixel (controls step size of sliding window)', required=False)
    
    # plotting related params
    parser.add_argument('--min-val-true', dest='min_val_true', type=float, default=0.5,
                        help='min value for color scale of grount truth data', required=False)
    parser.add_argument('--max-val-true', dest='max_val_true', type=float, default=None,
                        help='max value for color scale of grount truth data', required=False)
    parser.add_argument('--min-val-pred', dest='min_val_pred', type=float, default=0.1,
                        help='min value for color scale of prediction data', required=False)
    parser.add_argument('--max-val-pred', dest='max_val_pred', type=float, default=None,
                        help='max value for color scale of prediction data', required=False)
    parser.add_argument('--min-val-diff', dest='min_val_diff', type=float, default=-0.5,
                        help='min value for color scale of diff matrix', required=False)
    parser.add_argument('--max-val-diff', dest='max_val_diff', type=float, default=0.5,
                        help='max value for color scale of diff matrix', required=False)

    args = parser.parse_args(args=None if sys.argv[1:] else ['--help'])

    os.makedirs('tmp', exist_ok=True)

    bigwigs = args.bigwigs
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

    # ensure the user has provided either --del-start and --del-width or --screen-start, --screen-end, --perturb-width, --step-size
    if args.screen_start is not None and args.screen_end is not None:
        screening(args.output_path, args.outname, args.celltype, args.chr_name,
                  args.screen_start, args.screen_end, args.perturb_width, args.step_size, 
                    args.model_path,
                    args.seq_path, args.ctcf_path, args.atac_path, other_feats, 
                    ko_data=args.ko_data, ko_mode=args.ko_mode,
                    region = args.region, n_top_sites=args.n_top_sites, plot_diff=args.plot_diff,
                    min_val=args.min_val_pred, max_val=args.max_val_pred, 
                    min_val_diff=args.min_val_diff, max_val_diff=args.max_val_diff,
                    load_screen=args.load_screen)
    elif args.start is not None:
            single_deletion(args.output_path, args.outname, args.celltype, args.chr_name, args.start, 
                    args.deletion_start, args.deletion_width, 
                    args.var_pos, args.alt_bp, 
                    args.model_path,
                    args.seq_path, args.ctcf_path, args.atac_path, other_feats, 
                    ko_data=args.ko_data, ko_mode=args.ko_mode,
                    show_deletion_line = not args.hide_deletion_line,
                    end_padding_type = args.end_padding_type, 
                    region = args.region,
                    mid_hidden=args.mid_hidden, min_val_true=args.min_val_true, max_val_true=args.max_val_true,
                    min_val_pred=args.min_val_pred, max_val_pred=args.max_val_pred, plot_diff=args.plot_diff,
                    min_val_diff=args.min_val_diff, max_val_diff=args.max_val_diff)
    else:  # full chromosome prediction
        # use the step-size arg to do predictions for the whole chromosome
        # load one of the bigwigs to get the chromosome length
        bw = GenomicFeature(args.ctcf_path, 'bw')
        chr_name = args.chr_name
        seq_path = args.seq_path
        ctcf_path = args.ctcf_path
        atac_path = args.atac_path
        model_path = args.model_path
        mid_hidden = args.mid_hidden
        ko_data = args.ko_data
        ko_mode = args.ko_mode
        chr_length = bw.length(chr_name)
        print(f'Chromosome length: {chr_length}')
        step_size = int(window / args.n_overlap_preds)
        starts = np.arange(0, chr_length - window, step_size)
        ends = starts + window
        results = {'a1': [], 'a2': [], 'WT': [], 'KO': []}
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
        # get indices of the input tracks for KO
        ko_channels = []
        for ko in ko_data:
            if ko in input_track_names:
                ko_channels.append(input_track_names.index(ko))
            else:
                print(f'Warning: {ko} not found in input track names. Skipping KO for {ko}.')
        # get track_names from ctcf_path, atac_path, other_feats
        track_names = model_utils.get_1d_track_names(model_path)
        print(track_names)
        results_1d = {'chrom': [], 'start': [], 'end': []}
        for track_name in track_names:
            results_1d[f'{track_name}_WT'] = []
            results_1d[f'{track_name}_KO'] = []
        bins_1d = []
        for start, end in tqdm(zip(starts, ends), desc='Predicting', total=len(starts)):
            #print(f'Start: {start}, End: {end}')
            seq_region, ctcf_region, atac_region, other_regions = infer.load_region(chr_name, 
                    start, seq_path, ctcf_path, atac_path, other_feats, window = window)
            num_genomic_features = 2 if other_regions is None else 2 + len(other_regions)
            if atac_region is None:
                num_genomic_features -= 1
            pred_before_output = infer.prediction(seq_region, ctcf_region, atac_region, model_path, other_regions, num_genomic_features=num_genomic_features, mid_hidden=mid_hidden)
            pred_before = pred_before_output['hic']
            pred_before_1d = pred_before_output['1d']
            seq_region, ctcf_region, atac_region, other_regions = deletion_with_padding(start, 
                start, window, seq_region, ctcf_region, 
                atac_region, other_regions, ko_data=ko_data, ko_channels=ko_channels, ko_mode=ko_mode)
            pred_output = infer.prediction(seq_region, ctcf_region, atac_region, model_path, other_regions, num_genomic_features=num_genomic_features, mid_hidden=mid_hidden)
            pred = pred_output['hic']
            pred_1d = pred_output['1d']
            write_tmp_cooler(pred, chr_name, start)
            write_tmp_cooler(pred_before, chr_name, start, out_file='tmp/tmp_before.cool')
            # load coolers to populate res dict
            pred_cooler = cooler.Cooler('tmp/tmp.cool')
            pred_before_cooler = cooler.Cooler('tmp/tmp_before.cool')
            wt_pixels = pred_before_cooler.pixels()[:]
            ko_pixels = pred_cooler.pixels()[:]
            wt_pixels = wt_pixels.rename(columns={'count': 'WT'})
            ko_pixels = ko_pixels.rename(columns={'count': 'KO'})
            # merge the two cooler files with WT and KO keys
            pixels = wt_pixels.merge(ko_pixels, how='outer')
            results['a1'].extend(pixels['bin1_id'].tolist())
            results['a2'].extend(pixels['bin2_id'].tolist())
            results['WT'].extend(pixels['WT'].tolist())
            results['KO'].extend(pixels['KO'].tolist())
            bins.append(pred_before_cooler.bins()[:])

            if pred_1d is not None:
                res_ratio = int(pred_1d.shape[0] / pred.shape[0])
                res_1d = int(res / res_ratio)
                bin_range = np.int32(np.linspace(start, start + window - res_1d, pred_1d.shape[0]))
                results_1d['chrom'].extend([chr_name] * pred_1d.shape[0])
                results_1d['start'].extend(bin_range.tolist())
                results_1d['end'].extend((bin_range + res_1d).tolist())
                for track_idx, track_name in enumerate(track_names):
                    results_1d[f'{track_name}_WT'].extend(pred_before_1d[:, track_idx].tolist())
                    results_1d[f'{track_name}_KO'].extend(pred_1d[:, track_idx].tolist())
                bins_1d.append(pred_before_cooler.bins()[:])

        # convert the res dict to a dataframe
        res_df = pd.DataFrame(results).groupby(['a1', 'a2']).mean().reset_index()
        res_df['a1'] = 'A_' + res_df['a1'].astype(str)
        res_df['a2'] = 'A_' + res_df['a2'].astype(str)
        print(res_df)
        # convert the bins list to a dataframe
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
        res_df = res_df[['chrom1', 'start1', 'end1', 'a1', 'chrom2', 'start2', 'end2', 'a2', 'WT', 'KO']]
        # make sure outfile directory exists
        os.makedirs(os.path.dirname(args.out_file), exist_ok=True)
        # make sure outfile ends with .tsv
        if not args.out_file.endswith('.tsv'):
            args.out_file += '.tsv'
        # output the dataframe to a bed file
        res_df.to_csv(args.out_file, sep='\t', header=True, index=False)
        bins_df.to_csv(args.out_file.replace('.tsv', '_bins.tsv'), sep='\t', header=False, index=False)
        if pred_1d is not None:
            # convert the res_1d dict to a dataframe
            res_1d_df = pd.DataFrame(results_1d).groupby(['chrom', 'start', 'end']).mean().reset_index()
            print(res_1d_df)
            track_col_names = []
            for track_name in track_names:
                track_col_names.append(f'{track_name}_WT')
                track_col_names.append(f'{track_name}_KO')
            res_1d_df = res_1d_df[['chrom', 'start', 'end'] + track_col_names]
            res_1d_df.to_csv(args.out_file.replace('.tsv', '_1d.tsv'), sep='\t', header=True, index=False)

            # plot the 1d WT and KO overlaid
            fig, axs = plt.subplots(len(track_names), 1, figsize=(5, 5 * len(track_names)))
            for track_idx, track_name in enumerate(track_names):
                sns.scatterplot(data=res_1d_df, x=f'{track_name}_WT', y=f'{track_name}_KO', alpha=0.5, ax=axs[track_idx])
                axs[track_idx].set_xlabel(f'{track_name} WT')
                axs[track_idx].set_ylabel(f'{track_name} KO')
                axs[track_idx].set_title(f'{track_name} WT vs KO')
                # make the axes equal
                axs[track_idx].set_aspect('equal', adjustable='box')

            plt.savefig(os.path.join(args.output_path, f'{args.outname}{args.celltype}_{args.chr_name}_1d_scatter.png'), dpi=300)
            plt.close(fig)
        
        # plot a simple scatter plot of the WT vs KO
        fig, ax = plt.subplots(figsize=(10, 10))
        sns.scatterplot(data=res_df, x='WT', y='KO', ax=ax)
        ax.set_xlabel('WT')
        ax.set_ylabel('KO')
        ax.set_title('WT vs KO')
        
        plt.savefig(os.path.join(args.output_path, f'{args.outname}{args.celltype}_{args.chr_name}_scatter.png'), dpi=300)
        plt.close(fig)

   

def single_deletion(output_path, outname, celltype, chr_name, start, deletion_starts, deletion_widths, 
                    var_pos, alt_bp,
                    model_path, seq_path, ctcf_path, atac_path, other_feats, 
                    ko_data=['ctcf'], ko_mode='zero', show_deletion_line = True, end_padding_type = 'zero', region=None, mid_hidden=256,
                    min_val_true=1.0, max_val_true=None, min_val_pred=0.1, max_val_pred=None, plot_diff=False,
                    min_val_diff=-0.5, max_val_diff=0.5,
                    ctcf_log2=False):
    if not outname.endswith('_') and outname != '':
                outname += '_'
    seq_region, ctcf_region, atac_region, other_regions = infer.load_region(chr_name, 
            start, seq_path, ctcf_path, atac_path, other_feats, window = window, ctcf_log2=ctcf_log2)
    num_genomic_features = 2 if other_regions is None else 2 + len(other_regions)
    if atac_region is None:
            num_genomic_features -= 1
    # do baseline prediction for comparison
    pred_before_output = infer.prediction(seq_region, ctcf_region, atac_region, model_path, other_regions, num_genomic_features=num_genomic_features, mid_hidden=mid_hidden)
    pred_before = pred_before_output['hic']
    pred_before_1d = pred_before_output['1d']

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
    # get indices of the input tracks for KO
    ko_channels = []
    for ko in ko_data:
        if ko in input_track_names:
            ko_channels.append(input_track_names.index(ko))
        else:
            print(f'Warning: {ko} not found in input track names. Skipping KO for {ko}.')

    # Delete inputs
    if deletion_starts is not None and deletion_widths is not None:
        for deletion_start, deletion_width in zip(deletion_starts, deletion_widths):
            seq_region, ctcf_region, atac_region, other_regions = deletion_with_padding(start, 
                    deletion_start, deletion_width, seq_region, ctcf_region, 
                    atac_region, other_regions, ko_data=ko_data, ko_channels=ko_channels, ko_mode=ko_mode)
    
    # perturb sequence if var_pos is not None
    if var_pos is not None and alt_bp is not None:
        for pos, alt in zip(var_pos, alt_bp):
            print(f'Variant pos: {pos}, alt base: {alt}')
            seq_region = seq_perturb(pos - start - 1, alt, seq_region)

    # Prediction
    pred_output = infer.prediction(seq_region, ctcf_region, atac_region, model_path, other_regions, num_genomic_features=num_genomic_features, mid_hidden=mid_hidden)
    pred = pred_output['hic']
    pred_1d = pred_output['1d']

    # get track_names from ctcf_path, atac_path, other_feats
    track_names = model_utils.get_1d_track_names(model_path)
    if track_names is None:
        track_names = []

    for track_idx, track_name in enumerate(track_names):
        ctcf_pred_before = pred_before_1d[:, track_idx]
        ctcf_pred = pred_1d[:, track_idx]
        ctcf_log2fc = np.log2((ctcf_pred + 1e-5) / (ctcf_pred_before + 1e-5))

        fig, axs = plt.subplots(3, 1, figsize=(10, 5))
        axs[0].plot(ctcf_pred_before, label='Before', color='blue')
        # fill to zero
        axs[0].fill_between(range(len(ctcf_pred_before)), ctcf_pred_before, 0, color='blue', alpha=0.2)
        axs[1].plot(ctcf_pred, label='After', color='orange')
        # fill to zero
        axs[1].fill_between(range(len(ctcf_pred)), ctcf_pred, 0, color='orange', alpha=0.2)
        axs[2].plot(ctcf_log2fc, label='Log2FC', color='green')
        # fill to zero
        axs[2].fill_between(range(len(ctcf_log2fc)), ctcf_log2fc, 0, color='green', alpha=0.2)
        axs[0].set_title(f'{track_name.upper()} Before')
        axs[1].set_title(f'{track_name.upper()} After')
        axs[2].set_title(f'{track_name.upper()} Log2FC')
        plt.tight_layout()
        plt.savefig(os.path.join(output_path, f'{outname}{celltype}_{chr_name}_{start}_{track_name}_log2fc.png'), dpi=300)
        plt.close(fig)

    plot_ground_truth = False
    try:    
            hic_path = ctcf_path.replace('genomic_features', 'hic_matrix').replace('/ctcf.bw', '') + f'/{chr_name}.npz'
            hic = HiCFeature(path = hic_path)
            mat = hic.get(start, window=int(window * 1.2))
            mat = resize(mat, (int(image_scale * 1.2), int(image_scale * 1.2)), anti_aliasing=True)
            mat += 0.01
            plot = plot_utils.MatrixPlot(output_path, mat, 'ground_truth', celltype, 
                                    chr_name, start)
            plot.plot(vmin=1.0, vmax=2.5)
            plot_ground_truth = True
    except Exception as e:
            print(e)
            print('No ground truth found')
            mat = np.zeros_like(pred)

    write_tmp_cooler(pred, chr_name, start)
    write_tmp_cooler(pred_before, chr_name, start, out_file='tmp/tmp_before.cool')
    if plot_ground_truth:
        write_tmp_cooler(mat, chr_name, start, window=int(window * 1.3), out_file='tmp/tmp_true.cool')

    diff = pred - pred_before
    write_tmp_cooler(diff, chr_name, start, out_file='tmp/tmp_diff.cool')
    if deletion_starts is not None and deletion_widths is not None:
        for ko in ko_data:
            if ko in input_track_names:
                ko_path = input_track_paths[input_track_names.index(ko)]
                write_tmp_chipseq_ko(ko_path, ko, chr_name, start, deletion_start, deletion_width, ko_mode='knockout')
            else:
                print(f'Warning: {ko} not found in input track names. Skipping KO for {ko}.')
        

    # open tracks.ini file and add two vertical lines for the deletion, write in a new tmp tracks file
    # must first create bed file for deletion
    with open('tmp/regions.bed', 'w') as f:
        if deletion_starts is not None and deletion_widths is not None:
            for deletion_start, deletion_width in zip(deletion_starts, deletion_widths):
                f.write(f'{chr_name}\t{deletion_start}\t{deletion_start + deletion_width}\n')
    

    # write a links file (chr1 start1 end1 chr2 start2 end2 score) for each pixel in the baseline and the deletion
    baseline_cutoff = np.quantile(pred_before, 0.99)
    cutoff = np.quantile(pred, 0.99)  
    if plot_diff:
        diff_cutoff_gain = np.quantile(diff[diff > 0], 0.99)
        diff_cutoff_loss = np.quantile(diff[diff < 0], 0.01)
    if plot_ground_truth:
         ground_truth_cutoff = np.quantile(mat, 0.99)
    region_start = int(region.split(':')[1].split('-')[0]) if region is not None else start
    region_end = int(region.split(':')[1].split('-')[1]) if region is not None else start + window
    # write pred arcs
    with open('tmp/arcs.bed', 'w') as f:
        for i in range(pred_before.shape[0]):
            for j in range(pred_before.shape[1]):
                pixel_start_i = i * res + start
                pixel_end_i = i * res + start + res
                pixel_start_j = j * res + start
                pixel_end_j = j * res + start + res
                if pred_before[i, j] > baseline_cutoff and pixel_start_i > region_start and pixel_end_i < region_end and pixel_start_j > region_start and pixel_end_j < region_end:
                    f.write(f'{chr_name}\t{pixel_start_i}\t{pixel_end_i}\t{chr_name}\t{pixel_start_j}\t{pixel_end_j}\t{pred_before[i, j]}\n')
    if plot_diff:
        # write diff arcs
        with open('tmp/arcs_diff.bed', 'w') as f:
            for i in range(diff.shape[0]):
                for j in range(diff.shape[1]):
                    pixel_start_i = i * res + start
                    pixel_end_i = i * res + start + res
                    pixel_start_j = j * res + start
                    pixel_end_j = j * res + start + res
                    if (diff[i, j] > diff_cutoff_gain or diff[i, j] < diff_cutoff_loss) and pixel_start_i > region_start and pixel_end_i < region_end and pixel_start_j > region_start and pixel_end_j < region_end:
                        f.write(f'{chr_name}\t{pixel_start_i}\t{pixel_end_i}\t{chr_name}\t{pixel_start_j}\t{pixel_end_j}\t{diff[i, j]}\n')
        
    # write pred KO arcs
    with open('tmp/arcs_ko.bed', 'w') as f:
         for i in range(pred.shape[0]):
            for j in range(pred.shape[1]):
                pixel_start_i = i * res + start
                pixel_end_i = i * res + start + res
                pixel_start_j = j * res + start
                pixel_end_j = j * res + start + res
                if pred[i, j] > cutoff and pixel_start_i > region_start and pixel_end_i < region_end and pixel_start_j > region_start and pixel_end_j < region_end:
                    f.write(f'{chr_name}\t{pixel_start_i}\t{pixel_end_i}\t{chr_name}\t{pixel_start_j}\t{pixel_end_j}\t{pred[i, j]}\n')
    if plot_ground_truth:
        # write ground truhth arcs
        with open('tmp/arcs_true.bed', 'w') as f:
                for i in range(mat.shape[0]):
                    for j in range(mat.shape[1]):
                        pixel_start_i = i * res + start
                        pixel_end_i = i * res + start + res
                        pixel_start_j = j * res + start
                        pixel_end_j = j * res + start + res
                        if mat[i, j] > ground_truth_cutoff and pixel_start_i > region_start and pixel_end_i < region_end and pixel_start_j > region_start and pixel_end_j < region_end:
                            f.write(f'{chr_name}\t{pixel_start_i}\t{pixel_end_i}\t{chr_name}\t{pixel_start_j}\t{pixel_end_j}\t{mat[i, j]}\n')

    assembly = 'hg19'
    if '/mm10/' in ctcf_path:
        assembly = 'mm10'
    elif '/hg38/' in ctcf_path:
        assembly = 'hg38'
    # data root is path before /<assembly>
    # e.g. cshark_data/data/mm10/ -> cshark_data/data
    assembly_idx = ctcf_path.index(f'/{assembly}/')
    data_root = ctcf_path[:assembly_idx]
    tracks = get_tracks(data_root, celltype, assembly)
    lines = tracks.split('\n')
    lines = [line + '\n' for line in lines]
    colors = ['red', 'blue', 'green', 'orange', 'purple', 'pink', 'brown', 'gray']
    with open('tmp/tmp_tracks.ini', 'w') as f:
        for line in lines:
            if 'arcs.bed' in line:
                line = line.replace('arcs.bed', 'arcs_ko.bed')
            
            if '[Genes]' in line:
                if 'ctcf' in ko_data:
                    f.write('[CTCF KO]\n')
                    f.write('file = tmp/ctcf_ko.bw\n')
                    f.write('height = 2\n')
                    f.write('color = #ff0000\n')
                    f.write('title = CTCF KO\n')
                    f.write('min_value = 0\n')
                    ctcf_ko_max = get_axis_range_from_bigwig(ctcf_path, chr_name, start)
                    f.write(f'max_value = {ctcf_ko_max}\n')
                    f.write('number_of_bins = 512\n\n')

                # write additional tracks for each input track
                for track_i, (track_name, track_path) in enumerate(zip(input_track_names, input_track_paths)):
                    if track_name == 'ctcf':
                        continue
                    f.write(f'[{track_name}]\n')
                    f.write(f'file = {track_path}\n')
                    f.write('height = 2\n')
                    f.write(f'color = {colors[track_i]}\n')
                    f.write(f'title = {track_name}\n')
                    f.write('min_value = 0\n')
                    f.write('number_of_bins = 512\n\n')
                    if track_name in ko_data:
                        f.write(f'[{track_name} KO]\n')
                        f.write(f'file = tmp/{track_name}_ko.bw\n')
                        f.write('height = 2\n')
                        f.write(f'color = {colors[track_i]}\n')
                        f.write(f'title = {track_name} KO\n')
                        f.write('min_value = 0\n')
                        track_ko_max = get_axis_range_from_bigwig(track_path, chr_name, start)
                        f.write(f'max_value = {track_ko_max}\n')
                        f.write('number_of_bins = 512\n\n')

                f.write('[KO pred]\n')
                f.write('file = tmp/tmp.cool\n')
                f.write(f'min_value = {min_val_pred}\n')
                if max_val_pred is not None:
                    f.write(f'max_value = {max_val_pred}\n')
                f.write('colormap = Reds\n')
                f.write('file_type = hic_matrix_square\n\n')
                

            f.write(line)
             
        # now add the deletion line
        f.write('\n')
        f.write('[deletion]')
        f.write('# bed file with regions to highlight\n')
        f.write('file = tmp/regions.bed\n')
        f.write('# type:\n')
        f.write('type = vhighlight\n')
    
    if plot_ground_truth:
        # write a tracks_true.ini file for the tmp_true.cool
        lines = tracks.split('\n')
        lines = [line + '\n' for line in lines]
        with open('tmp/tmp_tracks_true.ini', 'w') as f:
                for line in lines:
                    if 'arcs.bed' in line:
                        line = line.replace('arcs.bed', 'arcs_true.bed')
                    if '[Genes]' in line:
                        # write additional tracks for each input track
                        for track_i, (track_name, track_path) in enumerate(zip(input_track_names, input_track_paths)):
                            if track_name == 'ctcf':
                                continue
                            f.write(f'[{track_name}]\n')
                            f.write(f'file = {track_path}\n')
                            f.write('height = 2\n')
                            f.write(f'color = {colors[track_i]}\n')
                            f.write(f'title = {track_name}\n')
                            f.write('min_value = 0\n')
                            f.write('number_of_bins = 512\n\n')
                        # add the ground truth hic matrix
                        f.write('[deeploop]\n')
                        f.write('file = tmp/tmp_true.cool\n')
                        f.write(f'min_value = {min_val_true}\n')
                        if max_val_true is not None:
                            f.write(f'max_value = {max_val_true}\n')
                        f.write('colormap = Reds\n')
                        f.write('file_type = hic_matrix_square\n\n')
                    f.write(line)

    # write a tracks_pred file for the tmp_before.cool
    lines = tracks.split('\n')
    lines = [line + '\n' for line in lines]
    with open('tmp/tmp_tracks_pred.ini', 'w') as f:
            for line in lines:
                if '[Genes]' in line:
                    # write additional tracks for each input track
                    for track_i, (track_name, track_path) in enumerate(zip(input_track_names, input_track_paths)):
                        if track_name == 'ctcf':
                            continue
                        f.write(f'[{track_name}]\n')
                        f.write(f'file = {track_path}\n')
                        f.write('height = 2\n')
                        f.write(f'color = {colors[track_i]}\n')
                        f.write(f'title = {track_name}\n')
                        f.write('min_value = 0\n')
                        f.write('number_of_bins = 512\n\n')
                    # add the ground truth hic matrix
                    f.write('[WT pred]\n')
                    f.write('file = tmp/tmp_before.cool\n')
                    f.write(f'min_value = {min_val_pred}\n')
                    if max_val_pred is not None:
                        f.write(f'max_value = {max_val_pred}\n')
                    f.write('colormap = Reds\n')
                    f.write('file_type = hic_matrix_square\n\n')
                f.write(line)
    
    if plot_diff:
        # write a tracks_diff.ini file for the tmp_diff.cool
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
                        break  # arcs always at bottom

                    if '[Genes]' in line:
                        if 'ctcf' in ko_data:
                            f.write('[CTCF KO]\n')
                            f.write('file = tmp/ctcf_ko.bw\n')
                            f.write('height = 2\n')
                            f.write('color = #ff0000\n')
                            f.write('title = CTCF KO\n')
                            f.write('min_value = 0\n')
                            ctcf_ko_max = get_axis_range_from_bigwig(ctcf_path, chr_name, start)
                            f.write(f'max_value = {ctcf_ko_max}\n')
                            f.write('number_of_bins = 512\n\n')
                        for track_i, (track_name, track_path) in enumerate(zip(input_track_names, input_track_paths)):
                            if track_name == 'ctcf':
                                continue
                            f.write(f'[{track_name}]\n')
                            f.write(f'file = {track_path}\n')
                            f.write('height = 2\n')
                            f.write(f'color = {colors[track_i]}\n')
                            f.write(f'title = {track_name}\n')
                            f.write('min_value = 0\n')
                            f.write('number_of_bins = 512\n\n')
                            if track_name in ko_data:
                                f.write(f'[{track_name} KO]\n')
                                f.write(f'file = tmp/{track_name}_ko.bw\n')
                                f.write('height = 2\n')
                                f.write(f'color = {colors[track_i]}\n')
                                f.write(f'title = {track_name} KO\n')
                                f.write('min_value = 0\n')
                                track_ko_max = get_axis_range_from_bigwig(track_path, chr_name, start)
                                f.write(f'max_value = {track_ko_max}\n')
                                f.write('number_of_bins = 512\n\n')
                        # add the ground truth hic matrix
                        f.write('[Diff]\n')
                        f.write('file = tmp/tmp_diff.cool\n')
                        f.write(f'min_value = {min_val_diff}\n')
                        if max_val_diff is not None:
                            f.write(f'max_value = {max_val_diff}\n')
                        f.write('colormap = bwr\n')
                        f.write('file_type = hic_matrix_square\n\n')
                    f.write(line)
        

    try:
        region = region if region is not None else f"{chr_name}:{start}-{start + window}"
        
        if plot_diff:
            tracks_cmd = f"pyGenomeTracks --tracks tmp/tmp_tracks_diff.ini -o {os.path.join(output_path, f'{outname}{celltype}_{chr_name}_{start}_ctcf_ko_tracks_diff.png')} --region {region} --fontSize {font_size} --plotWidth {plot_width} --trackLabelFraction {track_label_fraction} > /dev/null 2>&1"
            os.system(tracks_cmd)
        if plot_ground_truth:
            tracks_cmd = f"pyGenomeTracks --tracks tmp/tmp_tracks_true.ini -o {os.path.join(output_path, f'{outname}{celltype}_{chr_name}_{start}_ctcf_true_tracks.png')} --region {region} --fontSize {font_size} --plotWidth {plot_width} --trackLabelFraction {track_label_fraction} > /dev/null 2>&1"
            os.system(tracks_cmd)
        tracks_cmd = f"pyGenomeTracks --tracks tmp/tmp_tracks_pred.ini -o {os.path.join(output_path, f'{outname}{celltype}_{chr_name}_{start}_ctcf_pred_tracks.png')} --region {region} --fontSize {font_size} --plotWidth {plot_width} --trackLabelFraction {track_label_fraction} > /dev/null 2>&1"
        os.system(tracks_cmd)
        if deletion_starts is not None and deletion_widths is not None:
            tracks_cmd = f"pyGenomeTracks --tracks tmp/tmp_tracks.ini -o {os.path.join(output_path, f'{outname}{celltype}_{chr_name}_{start}_ctcf_ko_tracks.png')} --region {region} --fontSize {font_size} --plotWidth {plot_width} --trackLabelFraction {track_label_fraction} > /dev/null 2>&1"
            os.system(tracks_cmd)

        
    except Exception as e:  # probably no tracks.ini file
         print(e)


def screening(output_path, outname, celltype, chr_name, screen_start, screen_end, perturb_width, step_size, model_path, seq_path, ctcf_path, atac_path, other_paths, 
              ko_data=['ctcf'], ko_mode='zero', region=None, n_top_sites=5, plot_diff=False,
              min_val=0.1, max_val=None, 
              save_pred = False, save_deletion = True, save_diff = True, save_impact_score = True, save_bedgraph = True, plot_impact_score = True, plot_frames = False,
              load_screen=False):
    if not outname.endswith('_') and outname != '':
                outname += '_'
    # Store data and model in memory
    seq, ctcf, atac = infer.load_data_default(chr_name, seq_path, ctcf_path, atac_path)
    if other_paths is not None:
        other_feats = []
        for feat_path in other_paths:
            print(f'Loading {feat_path}')
            other_feats.append(GenomicFeature(path = feat_path, norm = 'log'))
    num_genomic_features = 2 if other_feats is None else 2 + len(other_feats)
    if atac is None:
            num_genomic_features -= 1
    print(f'Number of genomic features: {num_genomic_features}')
    model = model_utils.load_default(model_path, num_genomic_features=num_genomic_features)
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
    # get indices of the input tracks for KO
    ko_channels = []
    for ko in ko_data:
        if ko in input_track_names:
            ko_channels.append(input_track_names.index(ko))
        else:
            print(f'Warning: {ko} not found in input track names. Skipping KO for {ko}.')
    # Generate pertubation windows
    # Windows are centered. Thus, both sides have enough margins
    windows = [w * step_size + screen_start for w in range(int((screen_end - screen_start) / step_size))]
    from tqdm import tqdm
    preds = np.empty((0, 256, 256))
    preds_deletion = np.empty((0, 256, 256))
    diff_maps = np.empty((0, 256, 256))
    perturb_starts = []
    perturb_ends = []
    if load_screen:
         # make sure that the bedgraph exists
        if not os.path.exists(os.path.join(output_path, f"{celltype}/screening/bedgraph/{chr_name}_screen_{screen_start}_{screen_end}_width_{perturb_width}_step_{step_size}_impact_score.bedgraph")):
            raise FileNotFoundError(f"Bedgraph file {os.path.join(output_path, f'{celltype}/screening/bedgraph/{chr_name}_screen_{screen_start}_{screen_end}_width_{perturb_width}_step_{step_size}_impact_score.bedgraph')} does not exist.")
    else:
        print('Screening...')
        for w_start in tqdm(windows):
            pred_start = int(w_start + perturb_width / 2 - 2097152 / 2)
            pred, pred_deletion, diff_map = predict_difference(chr_name, pred_start, int(w_start), perturb_width, model, seq, ctcf, atac, other_feats=other_feats, 
                                                               ko_data=ko_data, ko_channels=ko_channels, ko_mode=ko_mode)
            preds = np.append(preds, np.expand_dims(pred, 0), axis = 0)
            preds_deletion = np.append(preds_deletion, np.expand_dims(pred_deletion, 0), axis = 0)
            diff_maps = np.append(diff_maps, np.expand_dims(diff_map, 0), axis = 0)
            perturb_starts.append(w_start)
            perturb_ends.append(w_start + perturb_width)
        impact_scores = np.abs(diff_maps.mean(axis = (1, 2)))
        plot = plot_utils.MatrixPlotScreen(output_path, perturb_starts, perturb_ends, impact_scores, diff_maps, preds, preds_deletion, 'screening', celltype, chr_name, screen_start, screen_end, perturb_width, step_size, plot_impact_score)
        figure = plot.plot()
        plot.save_data(figure, save_pred, save_deletion, save_diff, save_impact_score, save_bedgraph)

    # load the locations of the top n impact scores
    impact_scores = np.load(f'{os.path.join(output_path, f"{celltype}/screening/npy/{chr_name}_screen_{screen_start}_{screen_end}_width_{perturb_width}_step_{step_size}_impact_score.npy")}')
    # load the locations of the top n impact scores
    top_n = np.argsort(impact_scores)[-n_top_sites:]
    top_n_starts = []
    top_n_ends = []
    for i in top_n:
        top_n_starts.append(windows[i])
        top_n_ends.append(windows[i] + perturb_width)

    for i, (w_start, w_end) in enumerate(zip(top_n_starts, top_n_ends)):
        print(f'Window start: {w_start}, Window end: {w_end}')
        pred_start = int(w_start + perturb_width / 2 - 2097152 / 2)
        pred, pred_deletion, diff_map = predict_difference(chr_name, pred_start, int(w_start), perturb_width, model, seq, ctcf, atac, other_feats=other_feats, 
                                                           ko_data=ko_data, ko_channels=ko_channels, ko_mode=ko_mode)
        write_tmp_cooler(pred, chr_name, pred_start, out_file=f'tmp/tmp.cool')
        write_tmp_cooler(pred_deletion, chr_name, pred_start, out_file='tmp/tmp_deletion.cool')
        write_tmp_cooler(diff_map, chr_name, pred_start, out_file='tmp/tmp_diff.cool')

        # must first create bed file for deletion
        with open('tmp/regions.bed', 'w') as f:
            f.write(f'{chr_name}\t{w_start}\t{w_end}\n')
        
        # with open('tracks_screen.ini', 'r') as f:
        #     lines = f.readlines()
        assembly = 'hg19'
        if '/mm10/' in ctcf_path:
            assembly = 'mm10'
        elif '/hg38/' in ctcf_path:
            assembly = 'hg38'
        # data root is path before /<assembly>
        # e.g. cshark_data/data/mm10/ -> cshark_data/data
        assembly_idx = ctcf_path.index(f'/{assembly}/')
        data_root = ctcf_path[:assembly_idx]
        tracks = get_tracks(data_root, celltype, assembly)
        tracks_screen = tracks
        lines = tracks_screen.split('\n')
        lines = [line + '\n' for line in lines]
        with open('tmp/tmp_tracks.ini', 'w') as f:
            for line in lines:
                if '[ctcf]' in line:
                        # first write the bedgraph output track
                        f.write('[screen score]\n')
                        f.write('height = 2\n')
                        f.write('title = screen score\n')
                        # e.g chr11_screen_9733614_10791870_width_10000_step_10000_impact_score.bedgraph
                        f.write(f'file = {os.path.join(output_path, f"{celltype}/screening/bedgraph/{chr_name}_screen_{screen_start}_{screen_end}_width_{perturb_width}_step_{step_size}_impact_score.bedgraph")}\n')
                        f.write('file_type = bedgraph\n\n')
                if '[Genes]' in line:
                    if plot_diff:
                        f.write('[Diff pred]\n')
                        f.write('file = tmp/tmp_diff.cool\n')
                        f.write(f'min_value = {min_val}\n')
                        f.write(f'max_value = {max_val}\n')
                        f.write('colormap = bwr\n')
                        f.write('file_type = hic_matrix_square\n\n')
                    else:
                        f.write('[WT pred]\n')
                        f.write('file = tmp/tmp.cool\n')
                        f.write('title = WT pred\n')
                        f.write(f'min_value = {min_val}\n')
                        if max_val is not None:
                            f.write(f'max_value = {max_val}\n')
                        f.write('colormap = Reds\n')
                        f.write('file_type = hic_matrix_square\n\n')
                        f.write('[KO pred]\n')
                        f.write('file = tmp/tmp_deletion.cool\n')
                        f.write('title = KO pred\n')
                        f.write(f'min_value = {min_val}\n')
                        if max_val is not None:
                            f.write(f'max_value = {max_val}\n')
                        f.write('colormap = Reds\n')
                        f.write('file_type = hic_matrix_square\n\n')
                f.write(line)
            # now add the deletion line
            f.write('\n')
            f.write('[deletion]\n')
            f.write('# bed file with regions to highlight\n')
            f.write('file = tmp/regions.bed\n')
            f.write('# type:\n')
            f.write('type = vhighlight\n')
                
              
                

        try:
            if region is not None:
                tracks_cmd = f"pyGenomeTracks --tracks tmp/tmp_tracks.ini -o {os.path.join(output_path, f'{outname}{celltype}_{chr_name}_{pred_start}_ctcf_screen_tracks.png')} --region {region} --fontSize {font_size} --plotWidth {plot_width} --trackLabelFraction {track_label_fraction} > /dev/null 2>&1"
            else:
                tracks_cmd = f"pyGenomeTracks --tracks tmp/tmp_tracks.ini -o {os.path.join(output_path, f'{outname}{celltype}_{chr_name}_{pred_start}_ctcf_screen_tracks.png')} --region {chr_name}:{screen_start}-{screen_start + window} --fontSize {font_size} --plotWidth {plot_width} --trackLabelFraction {track_label_fraction} > /dev/null 2>&1"
            os.system(tracks_cmd)
        except Exception as e:
            print(e)

    # delete tmp files
    # for file in os.listdir('tmp'):
    #     try:
    #         os.remove(os.path.join('tmp', file))
    #     except Exception as e:
    #         print(e)
    #         pass
    # # delete tmp folder
    # try:
    #     os.rmdir('tmp')
    # except Exception as e:
    #     print(e)
    #     pass
    

def predict_difference(chr_name, start, deletion_start, deletion_width, model, seq, ctcf, atac, other_feats=None, 
                       ko_data=['ctcf'], ko_channels=[0], ko_mode='zero'):
    # Define window which accomodates deletion
    end = start + 2097152
    seq_region, ctcf_region, atac_region = infer.get_data_at_interval(chr_name, start, end, seq, ctcf, atac)
    other_regions = None
    if other_feats is not None:
        other_regions = []
        for feat in other_feats:
            other_regions.append(feat.get(chr_name, start, end))
    # Unmodified inputs
    if other_regions is None:
        inputs = infer.preprocess_default(seq_region, ctcf_region, atac_region)
    else:
        inputs = infer.preprocess_default(seq_region, ctcf_region, atac_region, other_regions)
    pred = model(inputs)[0].detach().cpu().numpy() # Prediction
    # Inputs with deletion
    inputs_deletion = preprocess_deletion(chr_name, start, deletion_start, 
            deletion_width, seq_region, ctcf_region, atac_region, other_regions=other_regions, 
            ko_data=ko_data, ko_channels=ko_channels, ko_mode=ko_mode)
    pred_deletion = model(inputs_deletion)[0].detach().cpu().numpy() # Prediction
    # Compare inputs:
    diff_map = pred_deletion - pred
    return pred, pred_deletion, diff_map


def preprocess_deletion(chr_name, start, deletion_start, deletion_width, seq_region, ctcf_region, atac_region, 
                        other_regions=None, ko_data=['ctcf'], ko_channels=[0], ko_mode='zero'):
    # Delete inputs
    seq_region, ctcf_region, atac_region, other_regions = deletion_with_padding(start, 
            deletion_start, deletion_width, seq_region, ctcf_region, 
            atac_region, other_regions=other_regions, ko_data=ko_data, ko_channels=ko_channels, ko_mode=ko_mode)
    # Process inputs
    if other_regions is None:
        inputs = infer.preprocess_default(seq_region, ctcf_region, atac_region)
    else:
        inputs = infer.preprocess_default(seq_region, ctcf_region, atac_region, other_regions)
    return inputs

def deletion_with_padding(start, deletion_start, deletion_width, seq_region, ctcf_region, atac_region, 
                          other_regions=None, ko_data=['ctcf'], ko_channels=[0], ko_mode='zero'):
    ''' Delete all signals at a specfied location with corresponding padding at the end '''
    channel_offset = 0
    if 'ctcf' in ko_data:
        channel_offset += 1
    if 'atac' in ko_data:
        channel_offset += 1
    for track_name, channel_idx in zip(ko_data, ko_channels):
        print(f'Knocking out {track_name} at {deletion_start} with width {deletion_width}')
        if track_name == 'ctcf':
            ctcf_region = track_ko(deletion_start - start, 
                deletion_start - start + deletion_width, 
                ctcf_region, ko_mode=ko_mode)
        elif track_name == 'atac':
            atac_region = track_ko(deletion_start - start, 
                deletion_start - start + deletion_width, 
                atac_region, ko_mode=ko_mode)
        elif other_regions is not None:
            original = other_regions[channel_idx - channel_offset].copy()
            other_regions[channel_idx - channel_offset] = track_ko(deletion_start - start,
                deletion_start - start + deletion_width, 
                other_regions[channel_idx - channel_offset], ko_mode=ko_mode)
            if np.array_equal(original, other_regions[channel_idx - channel_offset]):
                print(f'Warning: {track_name} KO did not change the signal. Check the KO mode.')
    return seq_region, ctcf_region, atac_region, other_regions


def track_ko(start, end, track, window = 2097152, ko_mode='zero'):
    if ko_mode == 'zero':
        track[start:end] = 0
    elif ko_mode == 'mean':
        mean = np.mean(track[:start] + track[end:])
        track[start:end] = mean
    elif ko_mode == 'knockout':
        track[start:end] = knockout_peaks(track[start:end])
    else:
        raise ValueError('ko_mode must be either zero or mean')
    return track[:window]

def seq_perturb(start, alt, seq, window = 2097152):
    """
    Simulate DNA sequence variants
    """
    # replace sequence based on en_dict
    new_entry = np.zeros(5)
    alt_idx = en_dict[alt.lower()]
    new_entry[alt_idx] = 1
    ref_entry = seq[start, :]
    ref = ref_entry.argmax()
    ref_base = list(en_dict.keys())[list(en_dict.values()).index(ref)]
    print(f'Pos: {start}, Alt: {alt}, Ref: {ref_base.upper()}')
    if ref == alt_idx:
        print('No change')
    # display surrounding +/- 10bp
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
    ref_bases = ''.join(ref_bases).upper()
    print(f'{ref_bases}')
    seq[start, :] = new_entry
    return seq[:window]
    

if __name__ == '__main__':
    main()

