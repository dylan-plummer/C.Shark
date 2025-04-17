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
from cshark.inference.tracks_files import get_tracks

import argparse

window = 2097152
res = 8192
image_scale = 256
en_dict = {'a' : 0, 't' : 1, 'c' : 2, 'g' : 3, 'n' : 4}

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
    parser.add_argument('--var-pos', dest='var_pos', type=int, nargs='+',
                        help='Variant position', required=True)
    parser.add_argument('--alt', dest='alt_bp', type=str, nargs='+',
                        help='Variant alt base', required=True)
    parser.add_argument('--padding', dest='end_padding_type', 
                        default='zero',
                        help='Padding type, either zero or follow. Using zero: the missing region at the end will be padded with zero for ctcf and atac seq, while sequence will be padded with N (unknown necleotide). Using follow: the end will be padded with features in the following region (default: %(default)s)')
    parser.add_argument('--hide-line', dest='hide_deletion_line', 
                        action = 'store_true',
                        help='Remove the line showing deletion site (default: %(default)s)')
    parser.add_argument('--region', dest='region', 
                                help='specific region to visualize, otherwise full 2Mb window', required=False)
    
    parser.add_argument('--plot-diff', dest='plot_diff', 
                        action = 'store_true',
                        help='plot the difference heatmap instead of comparisons')
    
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

    if args.var_pos is not None and args.alt_bp is not None:
            if len(args.var_pos) != len(args.alt_bp):
                raise ValueError('Variant positions and alt bases must be the same length')
            single_deletion(args.output_path, args.outname, args.celltype, args.chr_name, args.start, 
                    args.var_pos, args.alt_bp, 
                    args.model_path,
                    args.seq_path, args.ctcf_path, args.atac_path, other_feats, ko_mode=args.ko_mode,
                    show_deletion_line = not args.hide_deletion_line,
                    end_padding_type = args.end_padding_type, 
                    region = args.region,
                    mid_hidden=args.mid_hidden, min_val_true=args.min_val_true, max_val_true=args.max_val_true,
                    min_val_pred=args.min_val_pred, max_val_pred=args.max_val_pred, plot_diff=args.plot_diff,
                    compare_cooler=args.compare_cooler, compare_name=args.compare_name)
    else:
        print('No deletion start or width provided, skipping deletion prediction')


    
    

   

def single_deletion(output_path, outname, celltype, chr_name, start, var_pos, alt_bp, model_path, seq_path, ctcf_path, atac_path, other_feats, 
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

    # Delete inputs
    for deletion_start, deletion_width in zip(var_pos, alt_bp):
        print(f'Variant pos: {deletion_start}, alt base: {deletion_width}')
        seq_region, ctcf_region, atac_region = deletion_with_padding(start, 
                deletion_start, deletion_width, seq_region, ctcf_region, 
                atac_region, end_padding_type, ko_mode=ko_mode)
    
    # Prediction
    pred = infer.prediction(seq_region, ctcf_region, atac_region, model_path, other_regions, num_genomic_features=num_genomic_features, mid_hidden=mid_hidden)

    plot_ground_truth = False
    try:    
            hic_path = ctcf_path.replace('genomic_features', 'hic_matrix').replace('/ctcf.bw', '') + f'/{chr_name}.npz'
            print(hic_path)
            hic = HiCFeature(path = hic_path)
            mat = hic.get(start, window=int(window * 1.2))
            mat = resize(mat, (int(image_scale * 1.2), int(image_scale * 1.2)), anti_aliasing=True)
            mat += 0.01
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

    # open tracks.ini file and add two vertical lines for the deletion, write in a new tmp tracks file
    # must first create bed file for deletion
    with open('tmp/regions.bed', 'w') as f:
        for deletion_start, deletion_width in zip(var_pos, [10000] * len(var_pos)):
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

def deletion_with_padding(start, var_pos, alt_bp, seq_region, ctcf_region, atac_region, other_regions=None, ko_mode='zero'):
    ''' Delete all signals at a specfied location with corresponding padding at the end '''
    seq_region, ctcf_region, atac_region = seq_perturb(var_pos - start - 1, 
            alt_bp, 
            seq_region, ctcf_region, atac_region)
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


def seq_perturb(start, alt, seq, ctcf, atac, window = 2097152):
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
    if atac is None:
        return seq[:window], ctcf[:window], atac
    else:
        return seq[:window], ctcf[:window], atac[:window]
    

if __name__ == '__main__':
    main()

