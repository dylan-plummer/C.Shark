import os
import sys
import cooler
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

def coarsen_to_uniform_bins_vectorized(input_cool_path, output_cool_path, bin_file, agg_type='mean'):
    """
    Coarsens a cooler file with non-uniform bins to a new cooler file with uniform bins,
    using vectorized operations and chunked processing for maximum efficiency.

    Parameters:
        input_cool_path (str): Path to the input cooler file with non-uniform bins.
        output_cool_path (str): Path to save the output cooler file with uniform bins.
        uniform_binsize (int): The desired uniform bin size in base pairs for the output cooler.
        n_workers (int): Number of worker processes to use for parallel processing.
    """

    c = cooler.Cooler(input_cool_path)
    chromsizes = c.chromsizes
    original_bins = c.bins()[:] # Load original bins once   
    original_bins['mid'] = (original_bins['start'] + original_bins['end']) // 2 # Calculate midpoints

    # Create new uniform bins
    # check if is directory
    if os.path.isdir(bin_file):
        new_bins = []
        for file in os.listdir(bin_file):
            bins_df = pd.read_csv(os.path.join(bin_file, file), sep='\t', header=None, names=['chrom', 'start', 'end'], usecols=[0, 1, 2])
            new_bins.append(bins_df)
        new_bins = pd.concat(new_bins, ignore_index=True)
    else:
        new_bins = pd.read_csv(bin_file, sep='\t', header=None, names=['chrom', 'start', 'end'], usecols=[0, 1, 2])
    new_bins['mid'] = (new_bins['start'] + new_bins['end']) // 2 # Calculate midpoints

    # Prepare to collect pixel chunks for the new cooler
    new_pixels_chunks = []

    # Process chromosomes in chunks
    for chrom in tqdm(chromsizes.index, desc="Processing chromosomes"):
        pixels = c.pixels().fetch(chrom) # Fetch pixels for the chromosome

        if not pixels.empty: # Process only if there are pixels
            # Fetch bins for the current chromosome for vectorized operations
            chr_bins = original_bins[original_bins['chrom'] == chrom]
            
            new_chr_bins = new_bins[new_bins['chrom'] == chrom]
            min_bin_id = new_chr_bins.index.min() # Minimum bin ID for the chromosome
            print(f"Processing chromosome {chrom} with {len(chr_bins)} original bins and {len(new_chr_bins)} new bins")

            # Vectorized bin mapping: Find indices of new bins for bin1_ids and bin2_ids
            new_bin1_ids_chunk = np.searchsorted(new_chr_bins['mid'].values, chr_bins.loc[pixels['bin1_id']]['mid'].values, side='right')
            new_bin2_ids_chunk = np.searchsorted(new_chr_bins['mid'].values, chr_bins.loc[pixels['bin2_id']]['mid'].values, side='right')

            # Create DataFrame for the new pixels chunk (within this chromosome)
            chunk_result = pd.DataFrame({
                'bin1_id': new_bin1_ids_chunk + min_bin_id, # Add min_bin_id to get the global bin ID
                'bin2_id': new_bin2_ids_chunk + min_bin_id,
                'count': pixels['count'].values # Keep counts from original pixels
            })
            new_pixels_chunks.append(chunk_result) # Append chunk to list


    # Concatenate all chromosome pixel chunks
    if new_pixels_chunks:
        new_pixels_df = pd.concat(new_pixels_chunks, ignore_index=True)
    else: # Handle case with no pixels
        new_pixels_df = pd.DataFrame({'bin1_id': [], 'bin2_id': [], 'count': []})


    # Aggregate pixels by bin pairs in the new binning (efficient groupby)
    if agg_type == 'mean':
        aggregated_pixels = new_pixels_df.groupby(['bin1_id', 'bin2_id'], observed=True)['count'].mean().reset_index()
    elif agg_type == 'sum':
        aggregated_pixels = new_pixels_df.groupby(['bin1_id', 'bin2_id'], observed=True)['count'].sum().reset_index()
    elif agg_type == 'max':
        aggregated_pixels = new_pixels_df.groupby(['bin1_id', 'bin2_id'], observed=True)['count'].max().reset_index()
    elif agg_type == 'min':
        aggregated_pixels = new_pixels_df.groupby(['bin1_id', 'bin2_id'], observed=True)['count'].min().reset_index()
    new_bins = new_bins[['chrom', 'start', 'end']]
    aggregated_pixels['bin1_id'] += 1
    aggregated_pixels['bin2_id'] += 1
    # Create the new cooler file
    cooler.create_cooler(output_cool_path, new_bins, aggregated_pixels, 
                         dtypes={'count': float})

    print(f"Cooler file created at: {output_cool_path}")

    # load cooler and plot some heatmaps
    c = cooler.Cooler(output_cool_path)
    chrom = 'chr2'
    mat_size = 2000000  #bp
    start = 162000000
    end = start + mat_size
    mat = c.matrix(balance=False).fetch(f'{chrom}:{start}-{end}')
    plt.imshow(mat, cmap='Reds')
    plt.colorbar()  
    plt.savefig('chr2_162Mb.png')
    plt.close()


if __name__ == "__main__":
    input_cooler_file = sys.argv[1] # Replace with your input cooler file path
    output_cooler_file = sys.argv[2] # Replace with your desired output cooler file path
    bins = sys.argv[3]  # 10kb
    agg_type = 'mean'  # Default aggregation type
    if len(sys.argv) > 4:
        # Check if the user provided an aggregation type
        if sys.argv[4] in ['mean', 'sum', 'max', 'min']:
            agg_type = sys.argv[4]
        else:
            print("Invalid aggregation type. Defaulting to 'mean'.")
            agg_type = 'mean'

    coarsen_to_uniform_bins_vectorized(input_cooler_file, output_cooler_file, bins, agg_type),