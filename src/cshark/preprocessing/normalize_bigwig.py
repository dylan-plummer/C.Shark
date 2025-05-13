import os
import sys
import pyBigWig
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm

def normalize_bigwig(bw, out_bw, q=0.9995):
    # Open the bigWig file
    bw = pyBigWig.open(bw, 'r')
    # Create a new bigWig file for output
    out_bw = pyBigWig.open(out_bw, 'w')
    # Add the chromosomes to the new bigWig file
    header = bw.chroms().items()
    header_list =list(header)
    out_bw.addHeader(header_list)
    # Get the chromosome names
    chroms = bw.chroms()
    for chrom in tqdm(chroms):
        # Get the data for the chromosome
        data = bw.values(chrom, 0, bw.chroms()[chrom])
        
        # Rank-normalize the data
        data = np.nan_to_num(data)
        normalized_data = data / np.quantile(data[data != 0], q=q)
        normalized_data = np.clip(normalized_data, 0, 1)
        normalized_data = np.nan_to_num(normalized_data)
        # Write the normalized data to the new bigWig file
        out_bw.addEntries(chrom, 0, ends=bw.chroms()[chrom], values=normalized_data, span=1, step=1)
    # Close the bigWig files
    bw.close()
    out_bw.close()
        

if __name__ == "__main__":
    in_bw = sys.argv[1]
    out_bw = sys.argv[2]
    if not os.path.exists(in_bw):
        print(f"Input bigWig file {in_bw} does not exist.")
        sys.exit(1)

    normalize_bigwig(in_bw, out_bw)
    print(f"Normalized bigWig file saved to {out_bw}.")