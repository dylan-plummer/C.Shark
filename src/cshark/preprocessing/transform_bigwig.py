from math import log
import os
import sys
from cshark import data
import pyBigWig
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
from scipy import stats

def quantile_normalize(query_vector, reference_vector):
    # Sort both vectors once
    query_sorted_idx = np.argsort(query_vector)
    ref_sorted = np.sort(reference_vector)
    
    # If lengths differ, interpolate reference to match query length
    if len(ref_sorted) != len(query_vector):
        ref_quantiles = np.interp(
            np.linspace(0, 1, len(query_vector)),
            np.linspace(0, 1, len(ref_sorted)),
            ref_sorted
        )
    else:
        ref_quantiles = ref_sorted
    
    # Create output array and fill in sorted order
    result = np.empty_like(query_vector)
    result[query_sorted_idx] = ref_quantiles
    
    return result

def normalize_bigwig(bw_ref, bw_query, out_bw):
    # Open the bigWig file
    bw_ref = pyBigWig.open(bw_ref, 'r')
    bw_query = pyBigWig.open(bw_query, 'r')
    # Create a new bigWig file for output
    out_bw = pyBigWig.open(out_bw, 'w')
    # Add the chromosomes to the new bigWig file
    header = bw_ref.chroms().items()
    header_list =list(header)
    out_bw.addHeader(header_list)
    # Get the chromosome names
    chroms = bw_query.chroms()
    for chrom in tqdm(chroms):
        # Get the data for the chromosome
        data_query = bw_query.values(chrom, 0, bw_query.chroms()[chrom])
        data_query = np.nan_to_num(data_query)
        try:
            data_ref = bw_ref.values(chrom, 0, bw_ref.chroms()[chrom])
        except KeyError:
            print(bw_ref.chroms().keys())
            ref_chrom_default = bw_ref.chroms().keys().__iter__().__next__()  # Fallback to the first chromosome if the current one is not found
            data_ref = bw_ref.values(ref_chrom_default, 0, bw_ref.chroms()[ref_chrom_default], numpy=True)
            # make the reference data the same length as the query data
            if len(data_ref) < len(data_query): # extend with duplicated values
                while len(data_ref) < len(data_query):
                    data_ref = np.concatenate((data_ref, data_ref))
                data_ref = data_ref[:len(data_query)]
            elif len(data_ref) > len(data_query): # truncate the reference data
                data_ref = data_ref[:len(data_query)]
        data_ref = np.nan_to_num(data_ref)

        normalized_data = quantile_normalize(data_query, data_ref)
        print(f"Normalizing {chrom} with {len(data_query)} query values and {len(data_ref)} reference values.")
        print(len(normalized_data), len(data_query), len(data_ref))
        # Write the normalized data to the new bigWig file
        out_bw.addEntries(chrom, 0, ends=bw_query.chroms()[chrom], values=normalized_data, span=1, step=1)

        # plot a histogram before and after normalization
        # fig, ax = plt.subplots(1, 3, figsize=(12, 6))
        # ax[0].hist(data_query, bins=100, color='blue', alpha=0.5, log=True)
        # ax[0].set_title(f"{chrom} - Before Normalization")
        # ax[0].set_xlabel("Value")
        # ax[0].set_ylabel("Frequency")
        # ax[1].hist(normalized_data, bins=100, color='green', alpha=0.5, log=True)
        # ax[1].set_title(f"{chrom} - After Normalization")
        # ax[1].set_xlabel("Value")
        # ax[1].set_ylabel("Frequency")
        # ax[2].hist(data_ref, bins=100, color='red', alpha=0.5, log=True)
        # ax[2].set_title(f"{chrom} - Reference Data")
        # ax[2].set_xlabel("Value")
        # ax[2].set_ylabel("Frequency")
        # plt.tight_layout()
        # plt.savefig(f"{chrom}_normalization_histogram.png")
        # plt.close(fig)
    # Close the bigWig files
    bw_ref.close()
    bw_query.close()
    out_bw.close()
        

if __name__ == "__main__":
    ref_bw = sys.argv[1]
    query_bw = sys.argv[2]
    out_bw = sys.argv[3]
    if not os.path.exists(ref_bw):
        print(f"Input bigWig file {ref_bw} does not exist.")
        sys.exit(1)

    normalize_bigwig(ref_bw, query_bw, out_bw)
    print(f"Normalized bigWig file saved to {out_bw}.")
