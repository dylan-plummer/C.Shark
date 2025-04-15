import os
import numpy as np
import pandas as pd
import sys
import torch
import cooler 
import seaborn as sns
import matplotlib.pyplot as plt
from tqdm import tqdm
from skimage.transform import resize
from scipy.sparse import coo_matrix, csr_matrix
from scipy.stats import pearsonr, spearmanr
from contextlib import suppress

from cshark.data.data_feature import GenomicFeature, HiCFeature
import cshark.inference.utils.inference_utils as infer
from cshark.inference.utils import plot_utils, model_utils
from cshark.inference.tracks_files import tracks, tracks_screen

import argparse

window = 2097152
res = 8192
image_scale = 256

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
                                help='', required=True)
    parser.add_argument('--out-file', dest='out_file', 
                        help='Path to the output file if doing full chromosome prediction', required=False)
    parser.add_argument('--seq', dest='seq_path', 
                        help='Path to the folder where the sequence .fa.gz files are stored', required=True)
    parser.add_argument('--ctcf', dest='ctcf_path', 
                        help='Path to the folder where the CTCF ChIP-seq .bw files are stored', required=True)
    parser.add_argument('--atac', dest='atac_path', 
                        help='Path to the folder where the ATAC-seq .bw files are stored', required=False)
    parser.add_argument('--h3k27ac', dest='h3k27ac', 
                                help='Path to the folder where the h3k27ac .bw files are stored', required=False)
    parser.add_argument('--h3k4me3', dest='h3k4me3', 
                            help='Path to the folder where the h3k4me3 .bw files are stored', required=False)
    parser.add_argument('--h3k36me3', dest='h3k36me3', 
                            help='Path to the folder where the h3k36me3 .bw files are stored', required=False)
    parser.add_argument('--h3k4me1', dest='h3k4me1', 
                            help='Path to the folder where the h3k4me1 .bw files are stored', required=False)
    parser.add_argument('--h3k27me3', dest='h3k27me3', 
                            help='Path to the folder where the h3k27me3 .bw files are stored', required=False)
    
    parser.add_argument('--ko-mode', dest='ko_mode', type=str, default='zero',
                        help='min value for color scale of grount truth data', required=False)
    parser.add_argument('--compare-cooler', dest='compare_cooler', type=str, default=None,
                        help='path to a cooler file to compare each output to', required=False)
    parser.add_argument('--compare-name', dest='compare_name', type=str, default='KO',
                        help='name of comparison name (for plotting)', required=False)

    # Deletion related params
    parser.add_argument('--del-start', dest='deletion_start', nargs='+', type=int,
                        help='Starting points for deletion.', required=False)
    parser.add_argument('--del-width', dest='deletion_width', nargs='+', type=int,
                        help='Width for deletion.', required=False)
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
    
    # plotting related params
    parser.add_argument('--min-val-true', dest='min_val_true', type=float, default=0.5,
                        help='min value for color scale of grount truth data', required=False)
    parser.add_argument('--max-val-true', dest='max_val_true', type=float, default=None,
                        help='max value for color scale of grount truth data', required=False)
    parser.add_argument('--min-val-pred', dest='min_val_pred', type=float, default=0.1,
                        help='min value for color scale of prediction data', required=False)
    parser.add_argument('--max-val-pred', dest='max_val_pred', type=float, default=None,
                        help='max value for color scale of prediction data', required=False)

    args = parser.parse_args(args=None if sys.argv[1:] else ['--help'])

    os.makedirs('tmp', exist_ok=True)

    other_feats = None
    if args.h3k27ac is not None and args.h3k4me3 is not None and args.h3k36me3 is not None and args.h3k4me1 is not None and args.h3k27me3 is not None:
                other_feats = [args.h3k27ac, args.h3k4me3, args.h3k36me3, args.h3k4me1, args.h3k27me3]
    elif args.h3k27ac is not None and args.h3k4me3 is not None and args.h3k36me3 is not None and args.h3k4me1 is not None:
            other_feats = [args.h3k27ac, args.h3k4me3, args.h3k36me3, args.h3k4me1]
    elif args.h3k27ac is not None and args.h3k4me3 is not None and args.h3k36me3 is not None:
            other_feats = [args.h3k27ac, args.h3k4me3, args.h3k36me3]
    elif args.h3k27ac is not None and args.h3k4me3 is not None:
            other_feats = [args.h3k27ac, args.h3k4me3]
    elif args.h3k27ac is not None:
            other_feats = [args.h3k27ac]
    elif args.h3k4me3 is not None:
            other_feats = [args.h3k4me3]

    # ensure the user has provided either --del-start and --del-width or --screen-start, --screen-end, --perturb-width, --step-size
    if args.screen_start is not None and args.screen_end is not None:
        screening(args.output_path, args.outname, args.celltype, args.chr_name, args.screen_start, 
            args.screen_end, args.perturb_width, args.step_size, 
            args.model_path,
            args.seq_path, args.ctcf_path, args.atac_path, other_feats, ko_mode=args.ko_mode,
            region = args.region, n_top_sites=args.n_top_sites, plot_diff=args.plot_diff,
            min_val=args.min_val_pred, max_val=args.max_val_pred, load_screen=args.load_screen)
    elif args.deletion_start is not None and args.deletion_width is not None:
            single_deletion(args.output_path, args.outname, args.celltype, args.chr_name, args.start, 
                    args.deletion_start, args.deletion_width, 
                    args.model_path,
                    args.seq_path, args.ctcf_path, args.atac_path, other_feats, ko_mode=args.ko_mode,
                    show_deletion_line = not args.hide_deletion_line,
                    end_padding_type = args.end_padding_type, 
                    region = args.region,
                    mid_hidden=args.mid_hidden, min_val_true=args.min_val_true, max_val_true=args.max_val_true,
                    min_val_pred=args.min_val_pred, max_val_pred=args.max_val_pred, plot_diff=args.plot_diff,
                    compare_cooler=args.compare_cooler, compare_name=args.compare_name)
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
        ko_mode = args.ko_mode
        chr_length = bw.length(chr_name)
        print(f'Chromosome length: {chr_length}')
        step_size = int(window / 2)
        starts = np.arange(0, chr_length - window, step_size)
        ends = starts + window
        res = {'a1': [], 'a2': [], 'WT': [], 'KO': []}
        bins = []
        for start, end in tqdm(zip(starts, ends), desc='Predicting', total=len(starts)):
            #print(f'Start: {start}, End: {end}')
            seq_region, ctcf_region, atac_region, other_regions = infer.load_region(chr_name, 
                    start, seq_path, ctcf_path, atac_path, other_feats, window = window)
            num_genomic_features = 2 if other_regions is None else 2 + len(other_regions)
            if atac_region is None:
                num_genomic_features -= 1
            pred_before = infer.prediction(seq_region, ctcf_region, atac_region, model_path, other_regions, num_genomic_features=num_genomic_features, mid_hidden=mid_hidden)
            seq_region, ctcf_region, atac_region = deletion_with_padding(start, 
                start, window, seq_region, ctcf_region, 
                atac_region, 'zero', ko_mode=ko_mode)
            pred = infer.prediction(seq_region, ctcf_region, atac_region, model_path, other_regions, num_genomic_features=num_genomic_features, mid_hidden=mid_hidden)
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
            res['a1'].extend(pixels['bin1_id'].tolist())
            res['a2'].extend(pixels['bin2_id'].tolist())
            res['WT'].extend(pixels['WT'].tolist())
            res['KO'].extend(pixels['KO'].tolist())
            bins.append(pred_before_cooler.bins()[:])

        # convert the res dict to a dataframe
        res_df = pd.DataFrame(res).groupby(['a1', 'a2']).mean().reset_index()
        # undo the log transformation
        res_df['WT'] = np.exp(res_df['WT']) - 1  
        res_df['KO'] = np.exp(res_df['KO']) - 1
        res_df['a1'] = 'A_' + res_df['a1'].astype(str)
        res_df['a2'] = 'A_' + res_df['a2'].astype(str)
        print(res_df)
        # convert the bins list to a dataframe
        bins_df = pd.concat(bins, ignore_index=True).drop_duplicates().reset_index(drop=True)
        bins_df['bin_id'] = 'A_' + bins_df.index.astype(str)
        print(bins_df) 
        # make sure outfile directory exists
        os.makedirs(os.path.dirname(args.out_file), exist_ok=True)
        # make sure outfile ends with .tsv
        if not args.out_file.endswith('.tsv'):
            args.out_file += '.tsv'
        # output the dataframe to a bed file
        res_df.to_csv(args.out_file, sep='\t', header=True, index=False)
        bins_df.to_csv(args.out_file.replace('.tsv', '_bins.tsv'), sep='\t', header=False, index=False)


    
    

   

def single_deletion(output_path, outname, celltype, chr_name, start, deletion_starts, deletion_widths, model_path, seq_path, ctcf_path, atac_path, other_feats, 
                    ko_mode='zero', show_deletion_line = True, end_padding_type = 'zero', region=None, mid_hidden=256,
                    min_val_true=1.0, max_val_true=None, min_val_pred=0.1, max_val_pred=None, plot_diff=False,
                    compare_cooler=None, compare_name='KO'):
    if not outname.endswith('_') and outname != '':
                outname += '_'
    seq_region, ctcf_region, atac_region, other_regions = infer.load_region(chr_name, 
            start, seq_path, ctcf_path, atac_path, other_feats, window = window)
    num_genomic_features = 2 if other_regions is None else 2 + len(other_regions)
    if atac_region is None:
            num_genomic_features -= 1
    # do baseline prediction for comparison
    pred_before = infer.prediction(seq_region, ctcf_region, atac_region, model_path, other_regions, num_genomic_features=num_genomic_features, mid_hidden=mid_hidden)
    # Initialize plotting class
    plot = plot_utils.MatrixPlotDeletion(output_path, pred_before, 'baseline',
            celltype, chr_name, start, deletion_starts[0], deletion_widths[0], 
            padding_type = end_padding_type,
            show_deletion_line = show_deletion_line)
    plot.plot()

    # Delete inputs
    for deletion_start, deletion_width in zip(deletion_starts, deletion_widths):
        print(f'Deletion start: {deletion_start}, deletion width: {deletion_width}')
        seq_region, ctcf_region, atac_region = deletion_with_padding(start, 
                deletion_start, deletion_width, seq_region, ctcf_region, 
                atac_region, end_padding_type, ko_mode=ko_mode)
    
    # Prediction
    pred = infer.prediction(seq_region, ctcf_region, atac_region, model_path, other_regions, num_genomic_features=num_genomic_features, mid_hidden=mid_hidden)
    # Initialize plotting class
    plot = plot_utils.MatrixPlotDeletion(output_path, pred, 'deletion', 
            celltype, chr_name, start, deletion_start, deletion_width, 
            padding_type = end_padding_type,
            show_deletion_line = show_deletion_line)
    plot.plot()

    plot_ground_truth = False
    try:    
            hic_path = ctcf_path.replace('genomic_features', 'hic_matrix').replace('/ctcf.bw', '') + f'/{chr_name}.npz'
            print(hic_path)
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
    if compare_cooler is not None:
        cooler_comp = cooler.Cooler(compare_cooler)
        comp_mat = cooler_comp.matrix(balance=False).fetch(f'{chr_name}:{start}-{int(start + window)}')
        comp_mat = resize(comp_mat, (int(image_scale), int(image_scale)), anti_aliasing=True)
        write_tmp_cooler(comp_mat, chr_name, start, window=int(window), out_file='tmp/tmp_compare.cool')

        fig, axs = plt.subplots(1, 2, figsize=(10, 5))
        axs[0].imshow(mat[:image_scale, :image_scale], cmap='Reds')
        axs[0].set_title('WT')
        axs[1].imshow(comp_mat[:image_scale, :image_scale], cmap='Reds')
        axs[1].set_title(f'{compare_name} KO')
        plt.savefig(os.path.join(output_path, f'{outname}{celltype}_{chr_name}_{start}_ctcf_{compare_name}_KO.png'), dpi=300)
        plt.close(fig)

    
    if compare_cooler is not None and plot_ground_truth:
        fig, axs = plt.subplots(1, 4, figsize=(12, 4))

        before_auc_wt, tpr, fpr = loop_recovery_roc(mat[:image_scale, :image_scale], pred_before[:image_scale, :image_scale])
        # plot ROC curve 
        axs[0].plot(fpr, tpr, label='vs WT', color='blue')
        axs[0].set_xlabel('False Positive Rate')
        axs[0].set_ylabel('True Positive Rate')
        # now compare to the KO data (should be lower AUC)
        before_auc_ko, tpr, fpr = loop_recovery_roc(comp_mat[:image_scale, :image_scale], pred_before[:image_scale, :image_scale])
        axs[0].plot(fpr, tpr, label=f'vs {compare_name}', color='orange', linestyle='--')
        axs[0].legend()
        axs[0].set_title(f'WT (predicted vs experimental)\nAUC: WT:{before_auc_wt:.2f} vs KO:{before_auc_ko:.2f}')
        axs[0].set_xlim(0, 1)
        axs[0].set_ylim(0, 1)
    
    
        ko_auc_ko, tpr, fpr = loop_recovery_roc(comp_mat[:image_scale, :image_scale], pred[:image_scale, :image_scale])
        # plot ROC curve
        axs[1].plot(fpr, tpr, label=f'vs {compare_name}', color='orange')
        axs[1].set_xlabel('False Positive Rate')
        axs[1].set_ylabel('True Positive Rate')
        # now compare to the WT data (should be lower AUC)
        ko_auc_wt, tpr, fpr = loop_recovery_roc(mat[:image_scale, :image_scale], pred[:image_scale, :image_scale])
        axs[1].plot(fpr, tpr, label='vs WT', color='blue', linestyle='--')
        axs[1].legend()
        axs[1].set_title(f'KO (predicted vs experimental)\nAUC: KO:{ko_auc_ko:.2f} vs WT:{ko_auc_wt:.2f}')
        axs[1].set_xlim(0, 1)
        axs[1].set_ylim(0, 1)

    
        ground_truth_auc_wt, tpr_wt, fpr_wt = loop_recovery_roc(mat[:image_scale, :image_scale], comp_mat[:image_scale, :image_scale])
        ground_truth_auc_ko, tpr_ko, fpr_ko = loop_recovery_roc(comp_mat[:image_scale, :image_scale], mat[:image_scale, :image_scale])
        ground_truth_auc = (ground_truth_auc_wt + ground_truth_auc_ko) / 2
        tpr = (tpr_wt + tpr_ko) / 2
        fpr = (fpr_wt + fpr_ko) / 2
        # plot ROC curve
        axs[2].plot(fpr, tpr, label='Ground Truth', color='green')
        axs[2].plot(fpr_wt, tpr_wt, label='WT', color='blue', linestyle='--')
        axs[2].plot(fpr_ko, tpr_ko, label=f'{compare_name}', color='orange', linestyle='--')
        axs[2].set_title(f'(experimental vs experimental)\nAUC: {ground_truth_auc:.2f}')
        axs[2].set_xlabel('False Positive Rate')
        axs[2].set_ylabel('True Positive Rate')
        axs[2].set_xlim(0, 1)
        axs[2].set_ylim(0, 1)
    
        pred_auc_wt, tpr_wt, fpr_wt = loop_recovery_roc(pred[:image_scale, :image_scale], pred_before[:image_scale, :image_scale])
        pred_auc_ko, tpr_ko, fpr_ko = loop_recovery_roc(pred_before[:image_scale, :image_scale], pred[:image_scale, :image_scale])
        pred_auc = (pred_auc_wt + pred_auc_ko) / 2
        tpr = (tpr_wt + tpr_ko) / 2
        fpr = (fpr_wt + fpr_ko) / 2

        # plot ROC curve
        axs[3].plot(fpr, tpr, label='KO', color='red')
        axs[3].plot(fpr_wt, tpr_wt, label='WT', color='blue', linestyle='--')
        axs[3].plot(fpr_ko, tpr_ko, label=f'{compare_name}', color='orange', linestyle='--')
        axs[3].set_title(f'(predicted vs predicted)' + f'\nAUC: {pred_auc:.2f}')
        axs[3].set_xlabel('False Positive Rate')
        axs[3].set_ylabel('True Positive Rate')
        axs[3].set_xlim(0, 1)
        axs[3].set_ylim(0, 1)

        plt.tight_layout()
        plt.savefig(os.path.join(output_path, f'{outname}{celltype}_{chr_name}_{start}_ctcf_roc.png'), dpi=300)
        plt.close(fig)


    # measure correlation between all of the pixels in the ground truth and the prediction
    # ground_truth_pixels = mat[:image_scale, :image_scale].flatten()
    # pred_ko_pixels = pred[:image_scale, :image_scale].flatten()
    # pred_before_pixels = pred_before[:image_scale, :image_scale].flatten()

    # # measure pairwise correlations and plot a bar plot
    # # pearsonr
    # pearsonr_ko = pearsonr(ground_truth_pixels, pred_ko_pixels)[0]
    # pearsonr_before = pearsonr(ground_truth_pixels, pred_before_pixels)[0]
    # pearson_both = pearsonr(pred_before_pixels, pred_ko_pixels)[0]
    # print(f'Pearsonr KO: {pearsonr_ko}, Pearsonr Before: {pearsonr_before}, Pearsonr Both: {pearson_both}')
    # # spearmanr
    # spearmanr_ko = spearmanr(ground_truth_pixels, pred_ko_pixels)[0]
    # spearmanr_before = spearmanr(ground_truth_pixels, pred_before_pixels)[0]
    # spearman_both = spearmanr(pred_before_pixels, pred_ko_pixels)[0]
    # print(f'Spearmanr KO: {spearmanr_ko}, Spearmanr Before: {spearmanr_before}, Spearmanr Both: {spearman_both}')
    # # plot a bar plot
    # fig, ax = plt.subplots(figsize=(10, 5))
    # sns.barplot(x=['Pearsonr KO', 'Pearsonr Before', 'predictions'], y=[pearsonr_ko, pearsonr_before, pearson_both], ax=ax)
    # ax.set_title('Pearsonr Correlation')
    # ax.set_ylabel('Pearsonr Correlation')
    # plt.savefig(os.path.join(output_path, f'{outname}{celltype}_{chr_name}_{start}_ctcf_pearsonr.png'), dpi=300)
    # plt.close(fig)
    # fig, ax = plt.subplots(figsize=(10, 5))
    # sns.barplot(x=['Spearmanr KO', 'Spearmanr Before', 'predictions'], y=[spearmanr_ko, spearmanr_before, spearman_both], ax=ax)
    # ax.set_title('Spearmanr Correlation')
    # ax.set_ylabel('Spearmanr Correlation')
    # plt.savefig(os.path.join(output_path, f'{outname}{celltype}_{chr_name}_{start}_ctcf_spearmanr.png'), dpi=300)
    # plt.close(fig)


    diff = pred - pred_before
    write_tmp_cooler(diff, chr_name, start, out_file='tmp/tmp_diff.cool')

    # open tracks.ini file and add two vertical lines for the deletion, write in a new tmp tracks file
    # must first create bed file for deletion
    with open('tmp/regions.bed', 'w') as f:
        for deletion_start, deletion_width in zip(deletion_starts, deletion_widths):
            f.write(f'{chr_name}\t{deletion_start}\t{deletion_start + deletion_width}\n')
    

    # write a links file (chr1 start1 end1 chr2 start2 end2 score) for each pixel in the baseline and the deletion
    baseline_cutoff = np.quantile(pred_before, 0.99)
    cutoff = np.quantile(pred, 0.99)    
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


    lines = tracks.split('\n')
    lines = [line + '\n' for line in lines]
    with open('tmp/tmp_tracks.ini', 'w') as f:
        for line in lines:
            if 'arcs.bed' in line:
                line = line.replace('arcs.bed', 'arcs_ko.bed')
            
            if '[Genes]' in line:
                 # add the ground truth hic matrix
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
                    if '[Genes]' in line:
                        # add the ground truth hic matrix
                        f.write('[Diff]\n')
                        f.write('file = tmp/tmp_diff.cool\n')
                        f.write(f'min_value = {min_val_pred}\n')
                        if max_val_pred is not None:
                            f.write(f'max_value = {max_val_pred}\n')
                        f.write('colormap = bwr\n')
                        f.write('file_type = hic_matrix_square\n\n')
                    f.write(line)
        

    try:
        region = region if region is not None else f"{chr_name}:{start}-{start + window}"
        
        if plot_diff:
            tracks_cmd = f"pyGenomeTracks --tracks tmp/tmp_tracks_diff.ini -o {os.path.join(output_path, f'{outname}{celltype}_{chr_name}_{start}_ctcf_ko_tracks_diff.png')} --region {region} --fontSize 15 --plotWidth 17 --trackLabelFraction 0.13 > /dev/null 2>&1"
            os.system(tracks_cmd)
        else: 
            if plot_ground_truth:
                tracks_cmd = f"pyGenomeTracks --tracks tmp/tmp_tracks_true.ini -o {os.path.join(output_path, f'{outname}{celltype}_{chr_name}_{start}_ctcf_true_tracks.png')} --region {region} --fontSize 15 --plotWidth 17 --trackLabelFraction 0.13 > /dev/null 2>&1"
                os.system(tracks_cmd)
            tracks_cmd = f"pyGenomeTracks --tracks tmp/tmp_tracks_pred.ini -o {os.path.join(output_path, f'{outname}{celltype}_{chr_name}_{start}_ctcf_pred_tracks.png')} --region {region} --fontSize 15 --plotWidth 17 --trackLabelFraction 0.13 > /dev/null 2>&1"
            os.system(tracks_cmd)
            tracks_cmd = f"pyGenomeTracks --tracks tmp/tmp_tracks.ini -o {os.path.join(output_path, f'{outname}{celltype}_{chr_name}_{start}_ctcf_ko_tracks.png')} --region {region} --fontSize 15 --plotWidth 17 --trackLabelFraction 0.13 > /dev/null 2>&1"
            os.system(tracks_cmd)

        
    except Exception as e:  # probably no tracks.ini file
         print(e)


def screening(output_path, outname, celltype, chr_name, screen_start, screen_end, perturb_width, step_size, model_path, seq_path, ctcf_path, atac_path, other_paths, 
              ko_mode='zero', region=None, n_top_sites=5, plot_diff=False,
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
            pred, pred_deletion, diff_map = predict_difference(chr_name, pred_start, int(w_start), perturb_width, model, seq, ctcf, atac, other_feats=other_feats, ko_mode=ko_mode)
            #if plot_frames:
            #    plot_combination(output_path, celltype, chr_name, pred_start, w_start, perturb_width, pred, pred_deletion, diff_map, 'screening')
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
        pred, pred_deletion, diff_map = predict_difference(chr_name, pred_start, int(w_start), perturb_width, model, seq, ctcf, atac, other_feats=other_feats, ko_mode=ko_mode)
        write_tmp_cooler(pred, chr_name, pred_start, out_file=f'tmp/tmp.cool')
        write_tmp_cooler(pred_deletion, chr_name, pred_start, out_file='tmp/tmp_deletion.cool')
        write_tmp_cooler(diff_map, chr_name, pred_start, out_file='tmp/tmp_diff.cool')

        # must first create bed file for deletion
        with open('tmp/regions.bed', 'w') as f:
            f.write(f'{chr_name}\t{w_start}\t{w_end}\n')
        
        # with open('tracks_screen.ini', 'r') as f:
        #     lines = f.readlines()
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
                tracks_cmd = f"pyGenomeTracks --tracks tmp/tmp_tracks.ini -o {os.path.join(output_path, f'{outname}{celltype}_{chr_name}_{pred_start}_ctcf_screen_tracks.png')} --region {region} --fontSize 15 --plotWidth 17 --trackLabelFraction 0.13 > /dev/null 2>&1"
            else:
                tracks_cmd = f"pyGenomeTracks --tracks tmp/tmp_tracks.ini -o {os.path.join(output_path, f'{outname}{celltype}_{chr_name}_{pred_start}_ctcf_screen_tracks.png')} --region {chr_name}:{screen_start}-{screen_start + window} --fontSize 15 --plotWidth 17 --trackLabelFraction 0.13 > /dev/null 2>&1"
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
    

def predict_difference(chr_name, start, deletion_start, deletion_width, model, seq, ctcf, atac, other_feats=None, ko_mode='zero'):
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
            deletion_width, seq_region, ctcf_region, atac_region, other_regions=other_regions, ko_mode=ko_mode) # Get data
    pred_deletion = model(inputs_deletion)[0].detach().cpu().numpy() # Prediction
    # Compare inputs:
    diff_map = pred_deletion - pred
    return pred, pred_deletion, diff_map


def preprocess_deletion(chr_name, start, deletion_start, deletion_width, seq_region, ctcf_region, atac_region, other_regions=None, ko_mode='zero'):
    # Delete inputs
    seq_region, ctcf_region, atac_region = deletion_with_padding(start, 
            deletion_start, deletion_width, seq_region, ctcf_region, 
            atac_region, other_regions=other_regions, ko_mode=ko_mode)
    # Process inputs
    if other_regions is None:
        inputs = infer.preprocess_default(seq_region, ctcf_region, atac_region)
    else:
        inputs = infer.preprocess_default(seq_region, ctcf_region, atac_region, other_regions)
    return inputs

def deletion_with_padding(start, deletion_start, deletion_width, seq_region, ctcf_region, atac_region, other_regions=None, ko_mode='zero'):
    ''' Delete all signals at a specfied location with corresponding padding at the end '''
    # CTCF zeroing
    seq_region, ctcf_region, atac_region = ctcf_ko(deletion_start - start, 
            deletion_start - start + deletion_width, 
            seq_region, ctcf_region, atac_region, ko_mode=ko_mode)
    return seq_region, ctcf_region, atac_region

def write_tmp_cooler(pred, chr_name, start, res=8192, window=2097152, out_file='tmp/tmp.cool'):
    bins = pd.DataFrame()
    #bin_range = np.linspace(start, start + window - res, pred.shape[0])
    bin_range = np.arange(0, start + window + res, res)
    bins['start'] = bin_range
    bins['start'] = bins['start'].astype(int)
    bins['end'] = bins['start'] + res
    bins['end'] = bins['end'].astype(int)
    bins['chrom'] = chr_name
    # offset start bin 
    start_offset = int(start / res)

    pixels = pd.DataFrame()
    sparse_mat = coo_matrix(np.triu(pred), dtype=np.float32)
    pixels['bin1_id'] = sparse_mat.row + start_offset
    pixels['bin2_id'] = sparse_mat.col + start_offset
    pixels['count'] = sparse_mat.data 

    pixels.to_csv(out_file + '.csv')

    cooler.create_cooler(out_file, bins, pixels, dtypes={'count': np.float32})


def knockout_peaks(signal_array, threshold=2.0, min_peak_width=5, padding_factor=1.0):
    """
    Simulates knockout of peaks in a signal array by replacing peak regions with background values.
    
    Args:
        signal_array (numpy.ndarray): 1D array containing signal values.
        threshold (float): Minimum signal value to be considered part of a peak.
        min_peak_width (int): Minimum width for a region to be called a peak.
        padding_factor (float): Fraction of peak width to use for background calculation.
            
    Returns:
        numpy.ndarray: Copy of input array with peaks knocked out (replaced with background).
    """
    # Create a copy of the input array to modify
    result = np.copy(signal_array)
    array_length = len(signal_array)
    
    # Find regions above threshold
    above_threshold = signal_array >= threshold
    
    # Track peak regions
    in_peak = False
    peak_start = None
    peaks = []  # Will store (start, end) tuples
    
    # Find peaks
    for i in range(array_length):
        if above_threshold[i]:
            if not in_peak:
                # Start of a new peak
                peak_start = i
                in_peak = True
        else:
            if in_peak:
                # End of current peak
                peak_end = i
                if peak_end - peak_start >= min_peak_width:
                    peaks.append((peak_start, peak_end))
                in_peak = False
    
    # Handle case where array ends during a peak
    if in_peak and array_length - peak_start >= min_peak_width:
        peaks.append((peak_start, array_length))
    
    # Process each peak
    for peak_start, peak_end in peaks:
        peak_width = peak_end - peak_start
        
        # Calculate padding for background, but don't exceed array bounds
        padding = min(int(peak_width * padding_factor), 5)
        
        # Calculate regions before and after peak for background
        pre_start = max(0, peak_start - padding)
        pre_end = peak_start
        
        post_start = peak_end
        post_end = min(array_length, peak_end + padding)
        
        # Calculate mean of surrounding regions as background
        pre_values = signal_array[pre_start:pre_end]
        post_values = signal_array[post_start:post_end]
        
        # Handle empty regions
        pre_mean = np.mean(pre_values) if len(pre_values) > 0 else 0.0
        post_mean = np.mean(post_values) if len(post_values) > 0 else 0.0
        
        # Calculate background value as average of pre and post regions
        background_val = (pre_mean + post_mean) / 2.0
        
        # Replace peak with background value
        result[peak_start:peak_end] = background_val
    
    return result


def ctcf_ko(start, end, seq, ctcf, atac, window = 2097152, ko_mode='zero'):
    # import matplotlib.pyplot as plt
    # fig, ax = plt.subplots(2, 1, figsize=(10, 5))
    # ax[0].plot(ctcf[start:end])
    # ax[0].set_title('CTCF before knockout')
    # vmax = np.max(ctcf[start:end])
    # ax[0].set_ylim(0, vmax)
    if ko_mode == 'zero':
        ctcf[start:end] = 0
    elif ko_mode == 'mean':
        mean = np.mean(ctcf[:start] + ctcf[end:])
        ctcf[start:end] = mean
    elif ko_mode == 'knockout':
        ctcf[start:end] = knockout_peaks(ctcf[start:end])
    else:
        raise ValueError('ko_mode must be either zero or mean')
    # ax[1].plot(ctcf[start:end])
    # ax[1].set_title('CTCF after knockout')
    # ax[1].set_ylim(0, vmax)
    # plt.tight_layout()
    # plt.savefig('tmp/ctcf_ko.png')
    # plt.close()
    if atac is None:
        return seq[:window], ctcf[:window], atac
    else:
        return seq[:window], ctcf[:window], atac[:window]
    
def loop_recovery_roc(mat_true, mat_pred, true_cutoff=0.5, n_points=100):
    mat_true_sparse = (mat_true - np.min(mat_true)) / (np.max(mat_true) - np.min(mat_true))
    mat_pred_sparse = (mat_pred - np.min(mat_pred)) / (np.max(mat_pred) - np.min(mat_pred))
    mat_true_sparse = coo_matrix(mat_true_sparse + 0.001)
    mat_pred_sparse = coo_matrix(mat_pred_sparse + 0.001)
    pixels_true = pd.DataFrame()
    pixels_true['a1'] = mat_true_sparse.row
    pixels_true['a2'] = mat_true_sparse.col
    pixels_true['v'] = mat_true_sparse.data
    pixels_pred = pd.DataFrame()
    pixels_pred['a1'] = mat_pred_sparse.row
    pixels_pred['a2'] = mat_pred_sparse.col
    pixels_pred['v'] = mat_pred_sparse.data

    #true_cutoff = np.quantile(pixels_true['v'], 0.98)

    significant_pixels = pixels_true[pixels_true['v'] > true_cutoff]
    significant_pixels = significant_pixels[['a1', 'a2']]
    insignificant_pixels = pixels_true[pixels_true['v'] <= true_cutoff]
    insignificant_pixels = insignificant_pixels[['a1', 'a2']]
    tprs = []
    fprs = []
    for pred_cutoff in reversed(np.linspace(0, 1, n_points)):
        #pred_cutoff = np.quantile(pixels_pred['v'], q)
        # get the number of true positives
        pred_significant_pixels = pixels_pred[pixels_pred['v'] > pred_cutoff]
        pred_significant_pixels = pred_significant_pixels[['a1', 'a2']]
        pred_insignificant_pixels = pixels_pred[pixels_pred['v'] <= pred_cutoff]
        pred_insignificant_pixels = pred_insignificant_pixels[['a1', 'a2']]
        true_positives = len(pd.merge(significant_pixels, pred_significant_pixels, on=['a1', 'a2']))
        false_positives = len(pd.merge(insignificant_pixels, pred_significant_pixels, on=['a1', 'a2']))
        true_negatives = len(pd.merge(insignificant_pixels, pred_insignificant_pixels, on=['a1', 'a2']))
        false_negatives = len(pd.merge(significant_pixels, pred_insignificant_pixels, on=['a1', 'a2']))
        tpr = true_positives / (true_positives + false_negatives)
        fpr = false_positives / (true_negatives + false_positives)
        tprs.append(tpr)
        fprs.append(fpr)
    # get the area under the curve
    auc = np.trapz(tprs, fprs)
    print(f'Area under the curve: {auc}')
    return auc, np.array(tprs), np.array(fprs)
    

if __name__ == '__main__':
    main()

