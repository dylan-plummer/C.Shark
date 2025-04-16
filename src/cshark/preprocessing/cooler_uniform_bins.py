import sys
import cooler
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

def coarsen_to_uniform_bins_vectorized(input_cool_path, output_cool_path, uniform_binsize):
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
    new_bins = cooler.util.binnify(chromsizes, uniform_binsize)
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
            print(new_chr_bins)

            # Vectorized bin mapping: Find indices of new bins for bin1_ids and bin2_ids
            new_bin1_ids_chunk = np.searchsorted(new_chr_bins['mid'].values, chr_bins.loc[pixels['bin1_id'] - 1]['mid'].values, side='right')
            new_bin2_ids_chunk = np.searchsorted(new_chr_bins['mid'].values, chr_bins.loc[pixels['bin2_id'] - 1]['mid'].values, side='right')

            # Create DataFrame for the new pixels chunk (within this chromosome)
            chunk_result = pd.DataFrame({
                'bin1_id': new_bin1_ids_chunk + min_bin_id, # Add min_bin_id to get the global bin ID
                'bin2_id': new_bin2_ids_chunk + min_bin_id,
                'count': pixels['count'].values # Keep counts from original pixels
            })
            print(chunk_result)
            new_pixels_chunks.append(chunk_result) # Append chunk to list


    # Concatenate all chromosome pixel chunks
    if new_pixels_chunks:
        new_pixels_df = pd.concat(new_pixels_chunks, ignore_index=True)
    else: # Handle case with no pixels
        new_pixels_df = pd.DataFrame({'bin1_id': [], 'bin2_id': [], 'count': []})


    # Aggregate pixels by bin pairs in the new binning (efficient groupby)
    aggregated_pixels = new_pixels_df.groupby(['bin1_id', 'bin2_id'], observed=True)['count'].mean().reset_index()

    # Create the new cooler file
    cooler.create_cooler(output_cool_path, new_bins, aggregated_pixels, 
                         dtypes={'count': float})

    print(f"Cooler file with uniform bins (size {uniform_binsize}bp) created at: {output_cool_path}")

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

    # zoomify cooler to create mcool file
    cooler.zoomify_cooler(output_cool_path, output_cool_path.replace('.cool', '.mcool'), 
                   resolutions=[uniform_binsize, uniform_binsize * 2, uniform_binsize * 5],
                   chunksize=1000000)


if __name__ == "__main__":
    input_cooler_file = sys.argv[1] # Replace with your input cooler file path
    output_cooler_file = sys.argv[2] # Replace with your desired output cooler file path
    uniform_binsize_bp = int(sys.argv[3])  # 10kb

    coarsen_to_uniform_bins_vectorized(input_cooler_file, output_cooler_file, uniform_binsize_bp)