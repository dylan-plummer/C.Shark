from ast import In
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
import sys
import argparse
import numpy as np
import pandas as pd
import torch
import pyBigWig
import logomaker
import matplotlib.pyplot as plt
from skimage.transform import resize
from tqdm import tqdm
from pyjaspar import jaspardb
from scipy.signal import find_peaks
from scipy.ndimage import gaussian_filter1d

# Import necessary functions from the C.Shark codebase
import cshark.inference.utils.inference_utils as infer
from cshark.inference.utils.model_utils import load_default, get_all_track_names
from cshark.inference.utils.inference_utils import knockout_peaks
import multiprocessing as mp
from functools import partial

# Define constants from the codebase
WINDOW_SIZE = 2097152

class MemeMotif:
    """A simple class to hold motif data parsed from a MEME file."""
    def __init__(self, name, matrix):
        self.name = name
        # The matrix from MEME is already a probability matrix (ACGT order)
        self._matrix = matrix
        self.length = matrix.shape[0]

    @property
    def counts(self):
        """
        This property provides a compatible interface for the original code's
        call to `motif.counts.normalize()`.
        """
        return self

    def normalize(self):
        """
        Returns the probability matrix in the dictionary format expected by the
        worker functions. The MEME format provides probabilities for A, C, G, T.
        """
        # The script's internal representation is A, T, C, G, N.
        # This dictionary will be used to construct the PWM in the correct order.
        return {
            'A': self._matrix[:, 0],
            'C': self._matrix[:, 1],
            'G': self._matrix[:, 2],
            'T': self._matrix[:, 3]
        }


def parse_meme_file(filepath):
    """Parses a MEME file and returns a list of MemeMotif objects."""
    motifs = []
    if not os.path.exists(filepath):
        print(f"Error: MEME file not found at {filepath}")
        sys.exit(1)
        
    with open(filepath, 'r') as f:
        lines = f.readlines()

    current_motif_name = None
    matrix_lines = []
    is_parsing_matrix = False

    for line in lines:
        line = line.strip()
        if not line:
            continue

        if line.startswith("MOTIF"):
            # If we finished parsing a previous motif, save it
            if current_motif_name and matrix_lines:
                # MEME format is ACGT
                matrix = np.array([list(map(float, row.split())) for row in matrix_lines])
                motifs.append(MemeMotif(current_motif_name, matrix))

            # Start a new motif
            parts = line.split()
            current_motif_name = parts[1] if len(parts) > 1 else "Unknown"
            matrix_lines = []
            is_parsing_matrix = False

        elif line.startswith("letter-probability matrix"):
            is_parsing_matrix = True

        elif is_parsing_matrix and line and (line[0].isdigit() or line.startswith(" 0.")):
            matrix_lines.append(line)

    # Add the very last motif in the file
    if current_motif_name and matrix_lines:
        matrix = np.array([list(map(float, row.split())) for row in matrix_lines])
        motifs.append(MemeMotif(current_motif_name, matrix))
    
    print(f"Successfully parsed {len(motifs)} motifs from {filepath}")
    return motifs

def reverse_complement(seq_array):
    """Generate the reverse complement of a one-hot encoded DNA sequence array."""
    # Assuming seq_array shape is (length, 5) for A,T,C,G,N
    complement_map = {0: 1, 1: 0, 2: 3, 3: 2, 4: 4}  # A<->T, C<->G, N->N
    rev_comp = seq_array[::-1].copy()
    for i in range(rev_comp.shape[1]):
        rev_comp[:, i] = seq_array[::-1, complement_map[i]]
    return rev_comp

class ParseKwargs(argparse.Action):
    def __call__(self, parser, namespace, values, option_string=None):
        setattr(namespace, self.dest, dict())
        for value in values:
            key, value = value.split('=')
            getattr(namespace, self.dest)[key] = value

from numpy.lib.stride_tricks import as_strided

def _process_motif_vectorized(motif, gradient_pwms, seq_start, score_threshold=5.0):
    """Worker function to scan for a single motif using vectorized numpy operations."""
    try:
        matrix_dict = motif.counts.normalize()
    except Exception:
        return []

    bases = ['A', 'T', 'C', 'G', 'N']
    matrix = np.array([matrix_dict.get(b, [0]*motif.length) for b in bases]).T # Shape: (MotifLength, 5)
    
    len_pwm = matrix.shape[0]
    seq_len = gradient_pwms.shape[0]
    
    if len_pwm > seq_len:
        return []

    # Flatten the PWMs for dot product
    pwm_flat = matrix.flatten()
    pwm_rc_flat = reverse_complement(matrix).flatten()

    # --- The Magic: Vectorization with stride_tricks ---
    # Create a view of all sliding windows of the gradient array
    n_windows = seq_len - len_pwm + 1
    # Itemsize is the size of one element (e.g., 4 bytes for float32)
    itemsize = gradient_pwms.itemsize
    # Create a 2D view of shape (n_windows, len_pwm, 5)
    window_view = as_strided(gradient_pwms,
                             shape=(n_windows, len_pwm, 5),
                             strides=(5 * itemsize, 5 * itemsize, itemsize))
    
    # Reshape for matrix multiplication
    # Each row is now a flattened window
    all_windows_flat = window_view.reshape(n_windows, -1)

    # Perform all dot products in a single matrix-vector multiplication
    scores_fw = all_windows_flat @ pwm_flat / len_pwm
    scores_rc = all_windows_flat @ pwm_rc_flat / len_pwm
    # ----------------------------------------------------

    # Find positions where either score exceeds the threshold
    #passing_indices = np.where((scores_fw > score_threshold) | (scores_rc > score_threshold))[0]
    # Find positions where either score exceeds the threshold (in absolute terms)
    passing_indices = np.where((np.abs(scores_fw) > score_threshold) | (np.abs(scores_rc) > score_threshold))[0]

    if len(passing_indices) == 0:
        return []

    # Efficiently create the results
    rows = [{
        'motif': motif.name,
        'pos': int(pos),
        'score_fw': float(scores_fw[pos]),
        'score_rc': float(scores_rc[pos])
    } for pos in passing_indices]

    return rows

def save_scores_as_bigwig(scores, original_bw_path, chr_name, start, out_path):
    """Saves attribution scores as a BigWig file."""
    try:
        with pyBigWig.open(original_bw_path) as bw_in:
            header = bw_in.header()['cl'] if bw_in.header().get('cl') is not None else list(bw_in.chroms().items())
        
        with pyBigWig.open(out_path, "w") as bw_out:
            bw_out.addHeader(header)
            
            # Ensure scores is a 1D numpy array
            scores_np = scores.detach().cpu().numpy().flatten()

            # Create entries for the BigWig file
            starts = np.arange(start, start + len(scores_np), dtype=np.int64)
            ends = starts + 1
            values = scores_np.astype(np.float64)

            bw_out.addEntries([chr_name] * len(starts), starts, ends=ends, values=values)
        print(f"Successfully saved attribution scores to {out_path}")

    except Exception as e:
        print(f"Error saving BigWig file: {e}")
        # Fallback to saving as a bedGraph if BigWig fails
        print("Saving as bedGraph as a fallback...")
        bedgraph_path = os.path.splitext(out_path)[0] + ".bedgraph"
        df = pd.DataFrame({
            'chrom': chr_name,
            'start': starts,
            'end': ends,
            'score': values
        })
        df.to_csv(bedgraph_path, sep='\t', header=False, index=False)
        print(f"Successfully saved attribution scores to {bedgraph_path}")


def main():
    parser = argparse.ArgumentParser(description='C.Shark Gradient Ascent in Input Space.')
    # Add arguments similar to perturb.py for consistency
    parser.add_argument('--model', dest='model_path', required=True, help='Path to the model checkpoint.')
    parser.add_argument('--chr', dest='chr_name', required=True, help='Chromosome for prediction.')
    parser.add_argument('--start', dest='start', type=int, required=True, help=f'Starting point for the {WINDOW_SIZE}bp input window.')
    parser.add_argument('--seq', dest='seq_path', required=True, help='Path to the folder with sequence .fa.gz files.')
    parser.add_argument('--out-dir', dest='out_dir', required=True, help='Output directory')
    parser.add_argument('--meme-file', dest='meme_file', required=False, help='Path to the HOCOMOCO MEME file for motif scanning.')
    parser.add_argument('--tf', dest='tf', required=False, help='Name of the transcription factor for motif scanning.')
    parser.add_argument('--viz-bp', dest='viz_bp', type=int, default=50, help='Base pair range(+/-) for visualization.')
    # add KO mode (shuffle, zero, N)
    parser.add_argument('--ko-mode', dest='ko_mode', choices=['shuffle', 'zero', 'N', 'n'], help='Knockout mode for input features.')
    
    # Optional arguments for other epigenetic features
    parser.add_argument('--bigwigs', nargs='*', help='Paths to the bigwig files for genomic features, specified as key=value pairs (e.g., ctcf=path/to/ctcf.bw).', 
                        action=ParseKwargs)
    parser.add_argument('--snp-file', dest='snp_file', required=False, help='Path to the SNP file in BED format.')

    # Argument for the target locus
    parser.add_argument('--target-locus', dest='target_locus', required=True, 
                        help='The region in the Hi-C map to maximize, format: chr1:start1-end1_chr1:start2-end2')
    parser.add_argument('--plot-locus', dest='plot_locus', required=True, 
                        help='The region in the Hi-C map to plot, format: chr1:start1-end1')

    parser.add_argument('--resolution', dest='resolution', type=int, default=8192, help='Resolution of the Hi-C map used by the model.')
    parser.add_argument('--matrix-size', dest='mat_size', type=int, default=256, help='Matrix size used by the model.')
    parser.add_argument('--latent_size', dest='mid_hidden', type=int, default=256, help='Latent size of the model.')
    parser.add_argument('--ctcf-ko', dest='ctcf_ko', action='store_true', help='Whether to knockout CTCF peaks in the input.')
    parser.add_argument('--no-abs', dest='abs', action='store_false', help='Whether to use absolute values for saliency scores.')
    parser.add_argument('--n-loci', dest='n_motifs', type=int, default=10, help='Number of top saliency peaks to visualize.')
    parser.add_argument('--target-1d-length', dest='target_1d_length', type=int, default=8192, help='Length of the 1D targets used by the model.')
    parser.add_argument('--vmin', dest='vmin', type=float, default=0, help='Minimum value for Hi-C plotting.')
    parser.add_argument('--vmax', dest='vmax', type=float, default=None, help='Maximum value for Hi-C plotting.')


    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    # --- 1. Load Data and Model ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Dynamically get track names from the model checkpoint
    all_tracks, _, input_tracks = get_all_track_names(args.model_path)
    
    # Prepare bigwig paths based on model's expected inputs
    bigwig_paths = []
    if args.bigwigs:
        for track_name in input_tracks:
            if track_name in args.bigwigs:
                bigwig_paths.append(args.bigwigs[track_name])

    print(f"Loading data for region {args.chr_name}:{args.start}...")
    seq_region, ctcf_region, atac_region, other_regions = infer.load_region(
        args.chr_name, args.start, args.seq_path, 
        args.bigwigs.get('ctcf'), args.bigwigs.get('atac'), other_paths=bigwig_paths[2:] if len(bigwig_paths) > 2 else None,
        window=WINDOW_SIZE
    )

    if args.ctcf_ko and ctcf_region is not None:
        print("Knocking out CTCF peaks in the input region...")
        ctcf_region = knockout_peaks(ctcf_region, threshold=0.5)
    
    print("Loading model...")
    try:
        model = load_default(
            args.model_path, 
            num_genomic_features=len(input_tracks),
            mat_size=args.mat_size,
            mid_hidden=args.mid_hidden,
            seq_filter_size=15,
            target_1d_length=args.target_1d_length,
            recon_1d=True
        ).to(device)
    except Exception as e:
        model = load_default(
            args.model_path, 
            num_genomic_features=len(input_tracks),
            mat_size=args.mat_size,
            mid_hidden=args.mid_hidden,
            seq_filter_size=15,
            target_1d_length=args.target_1d_length,
            recon_1d=False
        ).to(device)

    model.eval() # Set model to evaluation mode

    # --- 2. Prepare Input Tensor ---
    print("Preprocessing input data...")
    inputs = infer.preprocess_default(seq_region, ctcf_region, atac_region, other_regions)
    
    # IMPORTANT: Enable gradient calculation for the input tensor
    inputs.requires_grad = True

    # --- 3. Forward Pass and Score Calculation ---
    print("Performing forward pass...")
    # Zero out any existing gradients
    model.zero_grad()
    if inputs.grad is not None:
        inputs.grad.zero_()

    try:
        outputs = model(inputs)
    except Exception as e:
        input_dict = {'seq': inputs[..., :5], 'ctcf': inputs[..., 5:6], 'atac': inputs[..., 6:7]}
        outputs = model(input_dict, predict_tracks=all_tracks + ['hic'])
    try:
        pred_hic = (outputs.get('hic') + outputs.get('hic').transpose(1, 2)) / 2
        pred_hic = torch.expm1(pred_hic)
    except AttributeError as e:  # corigami base model
        pred_hic = (outputs + outputs.transpose(1, 2)) / 2
        pred_hic = torch.expm1(pred_hic)

    # Parse the target locus and convert to pixel coordinates
    try:
        locus1_str, locus2_str = args.target_locus.split('_')
        _, region1 = locus1_str.split(':')
        _, region2 = locus2_str.split(':')
        start1, end1 = map(int, region1.split('-'))
        start2, end2 = map(int, region2.split('-'))
        
        # Convert genomic coordinates to pixel coordinates relative to the window
        p_start1 = (start1 - args.start) // args.resolution
        p_end1 = (end1 - args.start) // args.resolution
        p_start2 = (start2 - args.start) // args.resolution
        p_end2 = (end2 - args.start) // args.resolution

        # Ensure coordinates are within the matrix bounds
        p_start1, p_end1 = max(0, p_start1), min(args.mat_size, p_end1)
        p_start2, p_end2 = max(0, p_start2), min(args.mat_size, p_end2)

        if p_start1 >= p_end1 or p_start2 >= p_end2:
            raise ValueError("Invalid locus coordinates after mapping to pixels.")

    except ValueError as e:
        print(f"Error parsing --target-locus. Please use format 'chr:start-end_chr:start-end'. Details: {e}")
        sys.exit(1)

    # parse the plot locus if provided, otherwise use the full matrix
    try:
        plot_locus_str = args.plot_locus
        _, plot_region = plot_locus_str.split(':')
        plot_start, plot_end = map(int, plot_region.split('-'))
        # Convert genomic coordinates to pixel coordinates relative to the window
        plot_p_start = (plot_start - args.start) // args.resolution
        plot_p_end = (plot_end - args.start) // args.resolution
        # Ensure coordinates are within the matrix bounds
        plot_p_start, plot_p_end = max(0, plot_p_start), min(args.mat_size, plot_p_end)

        if plot_p_start >= plot_p_end:
            raise ValueError("Invalid plot locus coordinates after mapping to pixels.")

    except ValueError as e:
        print(f"Error parsing --plot-locus. Please use format 'chr:start-end'. Details: {e}")
        sys.exit(1)

    print(f"Maximizing score for pixel region: rows[{p_start1}:{p_end1}], cols[{p_start2}:{p_end2}]")
    
    # The score is the mean of the predicted values in the target region
    score = pred_hic[0, p_start1:p_end1, p_start2:p_end2].sum()
    print(f"Initial score for target locus: {score.item()}")

    # --- 4. Backward Pass (Gradient Calculation) ---
    print("Performing backward pass to get gradients...")
    score.backward()

    # The gradients are now stored in inputs.grad
    gradients = inputs.grad[0] # Get gradients for the single item in the batch

    # --- 5. Calculate Saliency Scores for Sequence ---
    # We only care about the first 5 channels, which correspond to the one-hot encoded sequence
    seq_gradients = gradients[:, :5]
    seq_gradients = seq_gradients * seq_region[:, :5]  # Mask gradients by the actual sequence
    saliency_scores = (seq_gradients).sum(dim=-1)

    # Take the absolute value as we care about the magnitude of the effect
    saliency_scores_abs = torch.abs(saliency_scores)
    # plot the top N peaks (+/- viz_bp) as sequence logos
    smoothed_scores_abs = gaussian_filter1d(saliency_scores_abs.detach().cpu().numpy(), sigma=5)
    
    # Find top N peaks based on smoothed absolute saliency scores
    peak_indices = find_peaks(smoothed_scores_abs, distance=100, height=np.percentile(smoothed_scores_abs, 90))[0]
    top_peak_indices = peak_indices[np.argsort(smoothed_scores_abs[peak_indices])][::-1][:args.n_motifs]
    print(f"Found {len(top_peak_indices)} saliency peaks to visualize.")

    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(saliency_scores.detach().cpu().numpy(), color='orange', alpha=0.5, label='Raw Saliency')
    ax.plot(smoothed_scores_abs, color='blue', label='Smoothed Absolute Saliency')
    ax.scatter(top_peak_indices, smoothed_scores_abs[top_peak_indices], color='red', zorder=5, label='Top Peaks')
    ax.set_title('Saliency Scores Across Input Sequence')
    ax.set_xlabel('Position in Input Sequence (bp)')
    ax.set_ylabel('Saliency Score')
    ax.legend()
    plt.savefig(f"{args.out_dir}/saliency_scores_plot.png", dpi=300, bbox_inches='tight')
    plt.close(fig)
        
    print(f"Calculated saliency scores. Max score: {saliency_scores.max().item()}, Mean score: {saliency_scores.mean().item()}")
    
    if args.meme_file is None:
        jdb_obj = jaspardb(release='JASPAR2024')
        motifs = jdb_obj.fetch_motifs(
                collection = ['CORE'],
                tax_group = ['Vertebrates'],
                species=['9606'],
                tf_name = args.tf if args.tf is not None else None,
                all_versions = False)
    else:
        # Load motifs from the specified MEME file
        motifs = parse_meme_file(args.meme_file)
        if not motifs:
            print("No motifs were loaded from the MEME file. Exiting.")
            return
        if args.tf is not None:
            # Filter motifs by the specified TF name
            motifs = [m for m in motifs if args.tf.lower() in m.name.lower()]
            if not motifs:
                print(f"No motifs found for TF '{args.tf}' in the provided MEME file. Exiting.")
                return
    
    # scan for known motifs in the saliency scores
    saliency_scores_np = saliency_scores.detach().cpu().numpy()
    gradient_pwms = seq_gradients.detach().cpu().numpy()

    motifs_list = list(motifs)
    worker = partial(_process_motif_vectorized, gradient_pwms=gradient_pwms, seq_start=args.start, score_threshold=0.05)

    # Use a Pool to parallelize motif scanning
    results = []
    with mp.Pool(processes=min(5, len(motifs_list))) as pool:
        for motif_rows in tqdm(pool.imap_unordered(worker, motifs_list), total=len(motifs_list)):
            if motif_rows:
                results.extend(motif_rows)

    motif_df = pd.DataFrame(results)
    if motif_df.empty:
        print("No motifs found matching the saliency profile. Exiting visualization.")
        return
        
    motif_df['score_max'] = motif_df[['score_fw', 'score_rc']].max(axis=1)
    motif_df['score_min'] = motif_df[['score_fw', 'score_rc']].min(axis=1)
    #motif_df.to_csv(f"{args.out_dir}/saliency_motif_correlations.tsv", sep='\t', index=False)
    
    # # --- 6. Save the Output and Visualize ---
    print("Saving attribution scores and generating plots...")
    ref_bw_path = args.bigwigs.get('ctcf')
    if not ref_bw_path:
        print("Warning: A reference BigWig (--bigwigs ctcf=...) is required to save the output as a BigWig.")
    else:
        output_bw_path = f"{args.out_dir}/saliency_scores.bw"
        save_scores_as_bigwig(saliency_scores, ref_bw_path, args.chr_name, args.start, output_bw_path)

    if args.snp_file:
        snps = pd.read_csv(args.snp_file, sep='\t', names=['chrom', 'start', 'name', 'ref', 'alt'])
        snps['start'] += 1 
        print(f"Loaded {len(snps)} SNPs for annotation.")

    peak_summaries = []
    
    # --- New peak-centric plotting loop ---
    for peak_pos in top_peak_indices:
        
        window_radius = args.viz_bp
        # Define the window to plot, centered on the peak
        seq_start_idx = peak_pos - window_radius
        seq_end_idx = peak_pos + window_radius

        # Boundary checks
        if seq_start_idx < 0:
            seq_start_idx = 0
            seq_end_idx = 2 * window_radius
        if seq_end_idx > len(gradient_pwms):
            seq_end_idx = len(gradient_pwms)
            seq_start_idx = seq_end_idx - (2 * window_radius)
        
        # Genomic coordinates for titles and SNP querying
        locus_start_genomic = args.start + seq_start_idx
        locus_end_genomic = args.start + seq_end_idx
        
        # --- Plotting the Saliency Logo ---
        if args.abs:
            seq_slice = np.abs(gradient_pwms[seq_start_idx:seq_end_idx, :])
        else:
            seq_slice = gradient_pwms[seq_start_idx:seq_end_idx, :]
        df_seq = pd.DataFrame(seq_slice, columns=['A', 'T', 'C', 'G', 'N'])
        df_seq = df_seq[['A', 'C', 'G', 'T']]
        logo = logomaker.Logo(df_seq, figsize=(15, 3), color_scheme='classic')
        
        locus_str = f"{args.chr_name}:{locus_start_genomic + 1}-{locus_end_genomic}"
        logo.ax.set_title(f'Saliency at Peak {peak_pos} ({locus_str})')
        
        # Highlight the exact peak position
        peak_plot_pos = peak_pos - seq_start_idx
        logo.ax.axvline(peak_plot_pos, color='black', linestyle='--', alpha=0.8, label=f'Peak Max ({peak_pos})')
        
        # --- Find and Annotate Motifs within this Window ---
        motifs_in_window = motif_df[(motif_df['pos'] >= seq_start_idx) & (motif_df['pos'] < seq_end_idx)].copy()
        
        if not motifs_in_window.empty:
            # Sort by score magnitude to find the most important ones
            motifs_in_window = motifs_in_window.reindex(motifs_in_window['score_max'].abs().sort_values(ascending=False).index)
            top_motifs_in_window = pd.concat([
                motifs_in_window.head(4),
                motifs_in_window.tail(4)
            ]).drop_duplicates(subset=['motif', 'pos']).reset_index(drop=True)
            y_pos_counter = 0
            max_y = logo.ax.get_ylim()[1]
            y_positions = [max_y * 0.95, max_y * 0.8, max_y * 0.65, max_y * 0.5, max_y*0.35]
            color_choices = ['red', 'blue', 'green', 'orange', 'purple']
            for _, motif_row in top_motifs_in_window.iterrows():
                m_name = motif_row['motif']
                m_pos = motif_row['pos']
                
                # Get motif length
                motif_len = 0
                for m in motifs:
                    if m.name == m_name:
                        motif_len = m.length
                        break
                if motif_len == 0: continue

                highlight_start = m_pos - seq_start_idx
                highlight_end = highlight_start + motif_len
                
                # Use alternating colors for clarity and different heights
                color = color_choices[y_pos_counter % len(color_choices)]
                logo.ax.axvspan(highlight_start - 0.5, 
                                highlight_end - 0.5, 
                                ymin=0, ymax=y_positions[y_pos_counter % len(y_positions)]/max_y,
                                color=color, alpha=0.1)
                logo.ax.text(highlight_start, y_positions[y_pos_counter % len(y_positions)], m_name, color=color, fontsize=9)
                y_pos_counter += 1

        if args.snp_file:
            snps_in_region = snps[(snps['chrom'] == args.chr_name) &
                                (snps['start'] >= locus_start_genomic) &
                                (snps['start'] < locus_end_genomic)].copy()

            if not snps_in_region.empty:
                for _, snp_row in snps_in_region.iterrows():
                    snp_plot_pos = snp_row['start'] - locus_start_genomic
                    line_color = 'purple'
                    logo.ax.axvline(snp_plot_pos, color=line_color, linestyle=':', alpha=0.9)
                    snp_label = f"{snp_row['name']} ({snp_row['ref']}>{snp_row['alt']})"
                    logo.ax.text(snp_plot_pos, logo.ax.get_ylim()[1] * 0.95, snp_label, color=line_color, fontsize=8, rotation=90)

        plt.savefig(f"{args.out_dir}/saliency_logo_peak_{peak_pos}.png", dpi=300, bbox_inches='tight')
        plt.close()

        # --- In-Silico Perturbation based on the TOP motif in the window ---
        if motifs_in_window.empty:
            print(f"No motifs found near peak {peak_pos}. Skipping in-silico perturbation.")
            continue
            
        top_motif_in_window = motifs_in_window.iloc[0]
        matched_motif = top_motif_in_window['motif']
        pos = top_motif_in_window['pos']

        motif = next((m for m in motifs if m.name == matched_motif), None)
        if motif is None: continue

        if args.ko_mode == 'zero':
            # generate baseline prediction with zeroing out the motif region
            baseline_seq = seq_region.copy()
            insert_start, insert_end = pos, pos + motif.length
            if insert_end > baseline_seq.shape[0]: continue
            
            baseline_seq[insert_start:insert_end, :5] = 0
            perturb_inputs = infer.preprocess_default(baseline_seq, ctcf_region, atac_region, other_regions).to(device)
        elif args.ko_mode.lower() == 'n':
            # generate baseline prediction with 'N' insertion at the motif region
            baseline_seq = seq_region.copy()
            insert_start, insert_end = pos, pos + motif.length
            if insert_end > baseline_seq.shape[0]: continue
            
            baseline_seq[insert_start:insert_end, :5] = np.array([0, 0, 0, 0, 1])  # 'N' encoding
            perturb_inputs = infer.preprocess_default(baseline_seq, ctcf_region, atac_region, other_regions).to(device)
        else:
            # generate baseline prediction with random sequence insertion at the same position
            random_seq = np.random.rand(motif.length, 5)
            random_seq = random_seq / random_seq.sum(axis=1, keepdims=True)
            baseline_seq = seq_region.copy()
            insert_start, insert_end = pos, pos + motif.length
            if insert_end > baseline_seq.shape[0]: continue
            baseline_seq[insert_start:insert_end, :5] = random_seq
            perturb_inputs = infer.preprocess_default(baseline_seq, ctcf_region, atac_region, other_regions).to(device)

        with torch.no_grad():
            try:
                perturb_outputs = model(perturb_inputs)
                perturb_hic_pred = (perturb_outputs.get('hic') + perturb_outputs.get('hic').transpose(1, 2)) / 2
            except Exception:
                input_dict = {'seq': perturb_inputs[..., :5], 'ctcf': perturb_inputs[..., 5:6], 'atac': perturb_inputs[..., 6:7]}
                perturb_outputs = model(input_dict, predict_tracks=all_tracks + ['hic'])
                perturb_hic_pred = (perturb_outputs.get('hic') + perturb_outputs.get('hic').transpose(1, 2)) / 2

            perturb_hic_pred = torch.expm1(perturb_hic_pred)

        # visualize the difference in the Hi-C map
        diff_hic = pred_hic - perturb_hic_pred
        mean_loop_change = diff_hic[0, p_start1:p_end1, p_start2:p_end2].mean().item()

        summary_data = top_motif_in_window.to_dict()
        summary_data['peak_pos'] = peak_pos
        summary_data['mean_hic_change'] = mean_loop_change

        if abs(mean_loop_change) < 0.001:
            print(f"Mean change for motif {matched_motif} at peak {peak_pos} is negligible ({mean_loop_change:.4f}), skipping visualization.")
            peak_summaries.append(summary_data)
            continue
            
        diff_hic_cropped = diff_hic[:, plot_p_start:plot_p_end, plot_p_start:plot_p_end]
        vmax = np.percentile(np.abs(diff_hic_cropped.detach().cpu().numpy()), 99)
        
        fig, ax = plt.subplots(figsize=(8, 7))
        im = ax.imshow(diff_hic_cropped[0].detach().cpu().numpy(), cmap='bwr', norm=plt.Normalize(vmin=-vmax, vmax=vmax))
        ax.set_title(f'Hi-C Change (Original - Perturb {args.ko_mode}) from Motif {matched_motif} at Peak {peak_pos}')
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="Change in Hi-C score")
        
        insert_pixel = (insert_start // args.resolution) - plot_p_start
        ax.axvline(x=insert_pixel, color='green', linestyle='--', label=f'Motif Pos ({pos})')
        ax.axhline(y=insert_pixel, color='green', linestyle='--')
        
        rect = plt.Rectangle((p_start2 - plot_p_start, p_start1 - plot_p_start), 
                             p_end2 - p_start2, p_end1 - p_start1, 
                             linewidth=2, edgecolor='yellow', facecolor='none', linestyle='--', label='Target Locus')
        ax.add_patch(rect)
        ax.legend()
        plt.savefig(f"{args.out_dir}/hic_change_peak_{peak_pos}_motif_{matched_motif}.png", dpi=300, bbox_inches='tight')
        plt.close(fig)

        # visualize the hic before and after
        fig, axs = plt.subplots(1, 2, figsize=(14, 6))
        if args.vmax is not None:
            vmax = args.vmax
        else:
            vmax_orig = np.max(pred_hic[0, plot_p_start:plot_p_end, plot_p_start:plot_p_end].detach().cpu().numpy())
            vmax_baseline = np.max(perturb_hic_pred[0, plot_p_start:plot_p_end, plot_p_start:plot_p_end].detach().cpu().numpy())
            vmax = max(vmax_orig, vmax_baseline)
        im0 = axs[0].imshow(pred_hic[0, plot_p_start:plot_p_end, plot_p_start:plot_p_end].detach().cpu().numpy(), cmap='Reds', norm=plt.Normalize(vmin=args.vmin, vmax=vmax))
        axs[0].set_title('Original Hi-C Prediction')
        axs[0].set_xticks([])
        axs[0].set_yticks([])
        plt.colorbar(im0, ax=axs[0], fraction=0.046, pad=0.04, label="Hi-C score")
        im1 = axs[1].imshow(perturb_hic_pred[0, plot_p_start:plot_p_end, plot_p_start:plot_p_end].detach().cpu().numpy(), cmap='Reds', norm=plt.Normalize(vmin=0, vmax=vmax))
        axs[1].set_title(f'Perturb {args.ko_mode} Hi-C Prediction')
        axs[1].set_xticks([])
        axs[1].set_yticks([])
        axs[1].axvline(x=insert_pixel, color='green', linestyle='--', label=f'Motif Pos ({pos})')
        axs[1].axhline(y=insert_pixel, color='green', linestyle='--')
        rect = plt.Rectangle((p_start2 - plot_p_start, p_start1 - plot_p_start), 
                             p_end2 - p_start2, p_end1 - p_start1, 
                             linewidth=2, edgecolor='yellow', facecolor='none', linestyle='--', label='Target Locus')
        axs[1].add_patch(rect)
        axs[1].legend()
        plt.colorbar(im1, ax=axs[1], fraction=0.046, pad=0.04, label="Hi-C score")
        plt.savefig(f"{args.out_dir}/hic_comparison_peak_{peak_pos}_motif_{matched_motif}.png", dpi=300, bbox_inches='tight')
        plt.close(fig)

        mean_rad21_change = 0
        try:
            if 'rad21' in all_tracks and outputs.get('1d') is not None:
                try:
                    original_rad21 = np.expm1(outputs.get('1d')['rad21'][0].detach().cpu().numpy().flatten())
                    modified_rad21 = np.expm1(perturb_outputs.get('1d')['rad21'][0].detach().cpu().numpy().flatten())
                except IndexError:
                    rad21_index = all_tracks.index('rad21')
                    original_rad21 = np.expm1(outputs.get('1d')[0, ..., rad21_index].detach().cpu().numpy().flatten())
                    modified_rad21 = np.expm1(perturb_outputs.get('1d')[0, ..., rad21_index].detach().cpu().numpy().flatten())
                rad21_diff = original_rad21 - modified_rad21
                mean_rad21_change = rad21_diff.mean().item()
        except Exception as e:
            print(f"Error calculating Rad21 change: {e}")
            print(outputs.get('1d'))
            pass
            
        summary_data['mean_rad21_change'] = mean_rad21_change
        peak_summaries.append(summary_data)


    # --- Generate Final Summary ---
    if not peak_summaries:
        print("No significant motif effects were found to summarize.")
        return

    summary_df = pd.DataFrame(peak_summaries)
    summary_df = summary_df.sort_values(by='mean_hic_change', ascending=False).reset_index(drop=True)
    summary_df.to_csv(f"{args.out_dir}/peak_effects_summary.tsv", sep='\t', index=False)
    print("\nTop perturbation effects summary:")
    print(summary_df[['peak_pos', 'motif', 'score_max', 'mean_hic_change']].head())


    fig, ax = plt.subplots(figsize=(8,6))
    ax.scatter(summary_df['mean_hic_change'], summary_df['score_max'])
    ax.set_ylabel('Motif Correlation Score (Max)')
    ax.set_xlabel('Mean Hi-C Change (Original - Random)')
    ax.set_title('Motif Saliency Correlation vs. In-Silico Effect')
    ax.axhline(0, color='gray', linestyle='--', alpha=0.7)
    ax.axvline(0, color='gray', linestyle='--', alpha=0.7)
    
    # Label top points
    for i, row in summary_df.head(15).iterrows():
        ax.text(row['mean_hic_change'], row['score_max'], row['motif'], fontsize=8)
    for i, row in summary_df.tail(15).iterrows():
        ax.text(row['mean_hic_change'], row['score_max'], row['motif'], fontsize=8)

    plt.savefig(f"{args.out_dir}/summary_correlation_vs_hic_change.png", dpi=300, bbox_inches='tight')
    plt.close(fig)
    
    if 'mean_rad21_change' in summary_df.columns:
        fig, ax = plt.subplots(figsize=(8,6))
        ax.scatter(summary_df['mean_rad21_change'], summary_df['mean_hic_change'])
        ax.set_xlabel('Mean Rad21 Change')
        ax.set_ylabel('Mean Hi-C Change')
        ax.set_title('In-Silico Rad21 Change vs. Hi-C Change')
        ax.axhline(0, color='gray', linestyle='--', alpha=0.7)
        ax.axvline(0, color='gray', linestyle='--', alpha=0.7)
        
        for i, row in summary_df.head(15).iterrows():
            ax.text(row['mean_rad21_change'], row['mean_hic_change'], row['motif'], fontsize=8)
        for i, row in summary_df.tail(15).iterrows():
            ax.text(row['mean_rad21_change'], row['mean_hic_change'], row['motif'], fontsize=8)
        
        plt.savefig(f"{args.out_dir}/summary_rad21_change_vs_hic_change.png", dpi=300, bbox_inches='tight')
        plt.close(fig)
            

if __name__ == '__main__':
    main()