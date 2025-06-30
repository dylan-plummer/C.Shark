import os
import sys
import cooler 
import numpy as np
import pandas as pd 

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

if __name__ == "__main__":
    ref_cooler = sys.argv[1]
    query_directory = sys.argv[2]
    out_directory = sys.argv[3]
    os.makedirs(out_directory, exist_ok=True)
    # Load the reference cooler file
    ref_cooler = cooler.Cooler(ref_cooler)

    for chrom_file in os.listdir(query_directory):
        if chrom_file.endswith('.tsv') and 'bins' not in chrom_file:
            chrom_name = chrom_file.split('.')[0]
            print(f'Processing {chrom_name}')
            query_data = pd.read_csv(
                os.path.join(query_directory, chrom_file),
                sep='\t')
            print(query_data.head())
            chrom = chrom_name.split('_')[0]
            reference_data = ref_cooler.pixels().fetch(chrom)['count'].values
            # reference_data = ref_cooler.pixels().fetch(chrom_name)['count'].values  
            
            # Normalize each row in the query data
            normalized_wt = quantile_normalize(
                query_data['WT'].values, 
                reference_data
            )
            normalized_ko = quantile_normalize(
                query_data['KO'].values, 
                reference_data
            )
            
            out_df = query_data.copy()
            out_df['WT'] = normalized_wt
            out_df['KO'] = normalized_ko
            out_df.to_csv(
                os.path.join(out_directory, chrom_file),
                sep='\t',
                index=False
            )
    

