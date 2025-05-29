import os
import re
import sys
import argparse
import cooler
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from tqdm import tqdm
from scipy.stats import pearsonr, spearmanr
from cshark.preprocessing.cooler_remap import coarsen_to_uniform_bins_vectorized

def sorted_nicely(l):
    convert = lambda text: int(text) if text.isdigit() else text
    alphanum_key = lambda key: [convert(c) for c in re.split('([0-9]+)', key)]
    return sorted(l, key=alphanum_key)


def loop_roc(pred_c, true_c, q_cutoff=0.995, chrom_only=None):
    genome_tprs = []
    genome_fprs = []

    for chrom in pred_c.chromnames:
        if chrom_only and chrom != chrom_only:
            continue
        chromsize = pred_c.chromsizes[chrom]
        locus = f'{chrom}:1-{chromsize}'
        pred_pixels = pred_c.pixels().fetch(locus)
        true_pixels = true_c.pixels().fetch(locus)

        # merge and compute correlation
        merged_pixels = pd.merge(
            pred_pixels, true_pixels, 
            on=['bin1_id', 'bin2_id'], 
            suffixes=('_pred', '_true')
        )
        if merged_pixels.empty:
            print(f'No pixels found for chromosome {chrom}. Skipping...')
            continue
        # Calculate correlation
        corr = pearsonr(merged_pixels['count_pred'], merged_pixels['count_true'])[0]
        print(f'Chromosome: {chrom}, Pearson correlation: {corr:.3f}')
        corr_spearman = spearmanr(merged_pixels['count_pred'], merged_pixels['count_true'])[0]
        print(f'Chromosome: {chrom}, Spearman correlation: {corr_spearman:.3f}')

        loop_cutoff = np.quantile(true_pixels['count'], q_cutoff) 
        print(f'Chromosome: {chrom}, Loop cutoff: {loop_cutoff:.3f}')

        # Split true pixels once outside the loop
        true_significant = true_pixels[true_pixels['count'] > loop_cutoff]
        true_insignificant = true_pixels[true_pixels['count'] <= loop_cutoff]
        
        # Get unique bin pairs from true data for faster lookups
        true_sig_pairs = set(zip(true_significant['bin1_id'], true_significant['bin2_id']))
        true_insig_pairs = set(zip(true_insignificant['bin1_id'], true_insignificant['bin2_id']))
        
        # Pre-calculate quantiles outside the loop
        quantiles = np.quantile(pred_pixels['count'], np.arange(0.01, 1.0, 0.05))

        tprs = [1]
        fprs = [1]
        
        for i, q_value in enumerate(tqdm(np.arange(0.01, 1.0, 0.05))):
            q_cutoff = quantiles[i]
            
            # Split predicted pixels
            pred_significant_mask = pred_pixels['count'] > q_cutoff
            pred_significant = pred_pixels[pred_significant_mask]
            
            # Create bin pair sets for prediction data
            pred_sig_pairs = set(zip(pred_significant['bin1_id'], pred_significant['bin2_id']))
            
            # Calculate confusion matrix using set operations instead of merges
            tp = len(true_sig_pairs.intersection(pred_sig_pairs))
            fp = len(true_insig_pairs.intersection(pred_sig_pairs))
            fn = len(true_sig_pairs) - tp
            tn = len(true_insig_pairs) - fp
            
            tpr = tp / (tp + fn) if (tp + fn) > 0 else 0
            fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
            
            tprs.append(tpr)
            fprs.append(fpr)
        tprs.append(0)
        fprs.append(0)
        genome_tprs.append(tprs)
        genome_fprs.append(fprs)

    # compute auc
    tprs = np.mean(genome_tprs, axis=0)
    fprs = np.mean(genome_fprs, axis=0)
    
    auc = -np.trapezoid(tprs, fprs)
    print(f'AUC: {auc:.3f}')

    # Plot ROC curve
    plt.figure(figsize=(3, 3))
    plt.plot(fprs, tprs)
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.grid()
    plt.show()

    return auc, tprs, fprs


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Loop ROC")
    parser.add_argument(
        "--cooler",
        type=str,
        required=True,
        help="Path to the cooler file",
    )
    parser.add_argument(
        "--pred-col",
        dest="pred_col",
        type=str,
        default='WT',
        help="Column name for the predicted values",
    )
    parser.add_argument(
        "--pred",
        type=str,
        required=True,
        help="Path to the predicted WT/KO files",
    )
    parser.add_argument(
        "--out",
        type=str,
        default="outputs",
        help="Path to the output directory",
    )
    parser.add_argument(
        "--chrom",
        type=str,
        default=None,
        help="If provided, only this chromosome will be used for the analysis",
    )

    os.makedirs('tmp', exist_ok=True)
    
    args = parser.parse_args()

    bins = []
    pixels_wt_dfs = []
    last_bin = 0
    for file in tqdm(sorted_nicely(os.listdir(args.pred))):
        if file.endswith('.tsv') and 'bins' not in file:
            if args.chrom and args.chrom not in file:
                continue
            df = pd.read_csv(os.path.join(args.pred, file), sep='\t')
            #   chrom1  start1  end1   a1 chrom2  start2   end2   a2        WT        KO
            # 0   chr1       0  4096  A_0   chr1       0   4096  A_0  0.063954  0.085381
            # 1   chr1       0  4096  A_0   chr1    4096   8192  A_1  0.066772  0.092672
            bins_file = file.replace('.tsv', '_bins.tsv')
            bins_file = os.path.join(args.pred, bins_file)
            bins_df = pd.read_csv(bins_file, sep='\t', header=None, names=['chrom', 'start', 'end', 'anchor'])
            pixels_wt = df[['a1', 'a2', args.pred_col]]
            pixels_wt.loc[:, 'a1'] = pixels_wt['a1'].str.replace('A_', '').astype(int) + last_bin
            pixels_wt.loc[:, 'a2'] = pixels_wt['a2'].str.replace('A_', '').astype(int) + last_bin
            # rename columns
            pixels_wt.columns = ['bin1_id', 'bin2_id', 'count']
            pixels_wt_dfs.append(pixels_wt)
            bins_df = bins_df[['chrom', 'start', 'end']]
            bins.append(bins_df)
            last_bin += len(bins_df)

    pixels_wt_dfs = pd.concat(pixels_wt_dfs).fillna(0)
    bins = pd.concat(bins, ignore_index=True)

    # Create the cooler object
    out_path = os.path.join(args.out, f'pred_{args.pred_col}.cool')
    cooler.create_cooler(out_path, bins, pixels_wt_dfs, dtypes={'count': np.float32})

    c = cooler.Cooler(args.cooler)
    bins = c.bins()[:]
    bins.to_csv('tmp/bins.tsv', index=False, sep='\t', header=False)

    coarsen_to_uniform_bins_vectorized(f'{args.out}/pred_{args.pred_col}.cool', f'{args.out}/pred_{args.pred_col}_5kb.cool', 'tmp/bins.tsv', 'mean')
    
    pred_c = cooler.Cooler(f'{args.out}/pred_{args.pred_col}_5kb.cool')
    true_c = cooler.Cooler(args.cooler)

    auc, tprs, fprs = loop_roc(pred_c, true_c, chrom_only=args.chrom)

    plt.figure(figsize=(4, 4))
    plt.plot(fprs, tprs)
    plt.plot([0, 1], [0, 1], 'k--', label='Random')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('AUC: {:.2f}'.format(auc))
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.grid()
    plt.legend()
    plt.savefig(f'{args.out}/{args.pred_col}_roc_curve.png', dpi=300)
    plt.close()

    np.savetxt(f'{args.out}/{args.pred_col}_tprs.txt', tprs)
    np.savetxt(f'{args.out}/{args.pred_col}_fprs.txt', fprs)