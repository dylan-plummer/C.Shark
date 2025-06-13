import os
import re
import argparse
import cooler
import numpy as np
import pandas as pd

from tqdm import tqdm
from cshark.preprocessing.cooler_remap import coarsen_to_uniform_bins_vectorized

def sorted_nicely(l):
    convert = lambda text: int(text) if text.isdigit() else text
    alphanum_key = lambda key: [convert(c) for c in re.split('([0-9]+)', key)]
    return sorted(l, key=alphanum_key)


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
    parser.add_argument(
        "--outname",
        type=str,
        required=True,
        help="Name of the output file",
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
    out_path = f'tmp/pred_{args.pred_col}.cool'
    cooler.create_cooler(out_path, bins, pixels_wt_dfs, dtypes={'count': np.float32})

    c = cooler.Cooler(args.cooler)
    bins = c.bins()[:]
    bins.to_csv('tmp/bins.tsv', index=False, sep='\t', header=False)

    coarsen_to_uniform_bins_vectorized(out_path, args.outname, 'tmp/bins.tsv', 'mean')
    