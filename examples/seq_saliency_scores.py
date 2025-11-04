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
    
    # Optional arguments for other epigenetic features
    parser.add_argument('--bigwigs', nargs='*', help='Paths to the bigwig files for genomic features, specified as key=value pairs (e.g., ctcf=path/to/ctcf.bw).', 
                        action=ParseKwargs)

    # Argument for the target locus
    parser.add_argument('--target-locus', dest='target_locus', required=True, 
                        help='The region in the Hi-C map to maximize, format: chr1:start1-end1_chr1:start2-end2')
    
    parser.add_argument('--resolution', dest='resolution', type=int, default=8192, help='Resolution of the Hi-C map used by the model.')
    parser.add_argument('--matrix-size', dest='mat_size', type=int, default=256, help='Matrix size used by the model.')
    parser.add_argument('--latent_size', dest='mid_hidden', type=int, default=256, help='Latent size of the model.')
    parser.add_argument('--ctcf-ko', dest='ctcf_ko', action='store_true', help='Whether to knockout CTCF peaks in the input.')
    parser.add_argument('--n-motifs', dest='n_motifs', type=int, default=25, help='Number of top motifs to report in either direction')


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
    model = load_default(
        args.model_path, 
        num_genomic_features=len(input_tracks),
        mat_size=args.mat_size,
        mid_hidden=args.mid_hidden,
        seq_filter_size=15,
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
    saliency_scores = (seq_gradients).sum(dim=-1)

    # Take the absolute value as we care about the magnitude of the effect
    saliency_scores = torch.abs(saliency_scores)
    # plot the top 10 peaks (+/- 10bp) as sequence logos
    smoothed_scores = gaussian_filter1d(saliency_scores.detach().cpu().numpy(), sigma=5)
    top_indices = find_peaks(smoothed_scores, distance=100, height=np.percentile(smoothed_scores, 90))[0]
    # filter to top 10
    top_indices = top_indices[np.argsort(smoothed_scores[top_indices])][-10:]

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(smoothed_scores, color='blue')
    ax.plot(saliency_scores.detach().cpu().numpy(), color='orange', alpha=0.5)
    ax.set_title('Saliency Scores Across Input Sequence')
    ax.set_xlabel('Position in Input Sequence (bp)')
    ax.set_ylabel('Saliency Score')
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
    top_motifs = []
    motif_df = {'motif': [], 
                'pos': [], 
                'score_fw': [],
                'score_rc': [],}
    saliency_scores = saliency_scores.detach().cpu().numpy()
    gradient_pwms = seq_gradients.detach().cpu().numpy()
    

    motifs_list = list(motifs)
    worker = partial(_process_motif_vectorized, gradient_pwms=gradient_pwms, seq_start=args.start, score_threshold=0.05)

    # Use a Pool to parallelize motif scanning
    results = []
    with mp.Pool(processes=min( 10 or 1, len(motifs_list) )) as pool:
        for motif_rows in tqdm(pool.imap_unordered(worker, motifs_list), total=len(motifs_list)):
            if motif_rows:
                results.extend(motif_rows)

    # Populate motif_df from aggregated results
    for r in results:
        motif_df['motif'].append(r['motif'])
        motif_df['pos'].append(r['pos'])
        motif_df['score_fw'].append(r['score_fw'])
        motif_df['score_rc'].append(r['score_rc'])
        
    
    motif_df = pd.DataFrame(motif_df)
    motif_df['pos_100bp'] = (motif_df['pos'] // 100) * 100
    # groupby position and take max score as motif at that position
    motif_df = motif_df.groupby(['motif', 'pos_100bp'], as_index=False).max()
    print(motif_df.head())

    motif_df.to_csv(f"{args.out_dir}/saliency_motif_correlations.tsv", sep='\t', index=False)
    # plot histogram of motif scores
    fig, ax = plt.subplots(figsize=(6,4))
    ax.hist(motif_df['score_fw'], bins=50, alpha=0.5, label='Forward Strand')
    ax.hist(motif_df['score_rc'], bins=50, alpha=0.5, label='Reverse Strand')
    ax.set_title('Histogram of Motif Correlation Scores with Saliency')
    ax.set_xlabel('Correlation Score')
    ax.set_ylabel('Frequency')
    ax.legend()
    plt.savefig(f"{args.out_dir}/motif_correlation_histogram.png", dpi=300, bbox_inches='tight')
    plt.close(fig)
        

        
    filter_df = pd.DataFrame(motif_df)
    filter_df['corr_max'] = filter_df[['score_fw', 'score_rc']].max(axis=1)
    filter_df['corr_min'] = filter_df[['score_fw', 'score_rc']].min(axis=1)
    # across each 100bp position, only keep the motif with the highest correlation (positive and negative)
    filter_df_max = filter_df.loc[filter_df.groupby('pos_100bp')['corr_max'].idxmax()].reset_index(drop=True)
    filter_df_min = filter_df.loc[filter_df.groupby('pos_100bp')['corr_min'].idxmin()].reset_index(drop=True)
    filter_df = pd.concat([filter_df_max, filter_df_min]).drop_duplicates().reset_index(drop=True)
    # get top motifs
    top_motifs = filter_df.nlargest(args.n_motifs, 'corr_max')
    top_motifs = pd.concat([top_motifs, filter_df.nsmallest(args.n_motifs, 'corr_min')]).reset_index(drop=True)
    print("Top motifs correlated with saliency scores:")
    print(top_motifs.head(10))
    print(top_motifs.tail(10))
    # --- 6. Save the Output ---
    print("Saving attribution scores...")
    # We need a reference BigWig to copy the header from. We can use the CTCF track.
    ref_bw_path = args.bigwigs.get('ctcf')
    if not ref_bw_path:
        print("Error: A reference BigWig (--bigwigs ctcf=...) is required to save the output.")
        sys.exit(1)

    # plot top 10 positions as sequence logos with the reference sequence logo below
    print(gradient_pwms.shape)
    window_size = 30
    hic_diffs = []
    rad21_diffs = []
    for idx, row in top_motifs.iterrows():
        pos = row['pos_100bp']
        matched_motif = row['motif']
        seq_start = pos - window_size // 2
        seq_end = seq_start + window_size
        seq_slice = gradient_pwms[seq_start:seq_end, :]
        if seq_slice.shape[0] < window_size:
            # pad with Ns
            pad_len = window_size - seq_slice.shape[0]
            pad_array = np.zeros((pad_len, 5))
            pad_array[:, 4] = 1  # N
            if seq_start == 0:
                seq_slice = np.vstack([pad_array, seq_slice])
            else:
                seq_slice = np.vstack([seq_slice, pad_array])
        df_seq = pd.DataFrame(seq_slice, columns=['A', 'T', 'C', 'G', 'N'])
        df_seq = df_seq[['A', 'C', 'G', 'T']]  # logomaker expects this order
        logo = logomaker.Logo(df_seq, color_scheme='classic')
        logo.ax.set_title(f'Motif: {matched_motif} at Pos: {row["pos"]}')
        plt.savefig(f"{args.out_dir}/motif_logo_{matched_motif}_{pos}.png", dpi=300, bbox_inches='tight')
        plt.close()
        # also plot the reference sequence logo
        ref_seq_array = seq_region[seq_start:seq_end, :5]
        df_ref_seq = pd.DataFrame(ref_seq_array, columns=['A', 'T', 'C', 'G', 'N'])
        df_ref_seq = df_ref_seq[['A', 'C', 'G', 'T']]  # logomaker expects this order
        logo = logomaker.Logo(df_ref_seq, color_scheme='classic')
        logo.ax.set_title(f'Reference Sequence at Pos: {pos}')
        plt.savefig(f"{args.out_dir}/reference_logo_{matched_motif}_{pos}.png", dpi=300, bbox_inches='tight')
        plt.close()

        # and finally the motif PWM logo
        if args.meme_file is not None:
            # find motif in parsed MEME motifs
            motif = None
            for m in motifs:
                if m.name == matched_motif:
                    motif = m
                    break
            if motif is None:
                print(f"Motif {matched_motif} not found in MEME file.")
                continue
        else:
            motifs = jdb_obj.fetch_motifs(
                collection = ['CORE'],
                tax_group = ['Vertebrates'],
                species=['9606'],
                all_versions = False)
            motif = None
            for m in motifs:
                if m.name == matched_motif:
                    motif = m
                    break
            if motif is None:
                print(f"Motif {matched_motif} not found in JASPAR database.")
                continue
        try:
            matrix_dict = motif.counts.normalize()
        except Exception:
            print(f"Could not normalize motif counts for {matched_motif}.")
            continue
        bases = ['A', 'T', 'C', 'G', 'N']
        matrix = np.array([matrix_dict.get(b, [0]*motif.length) for b in bases]).T # Shape: (MotifLength, 5)
        df_motif = pd.DataFrame(matrix, columns=['A', 'C', 'G', 'T', 'N'])
        df_motif = df_motif[['A', 'C', 'G', 'T']]  # logomaker expects this order
        fig, ax = plt.subplots(figsize=(10, 4))
        logomaker.Logo(df_motif, ax=ax, color_scheme='classic')
        ax.set_title(f'Motif PWM: {matched_motif}')
        plt.savefig(f"{args.out_dir}/motif_pwm_logo_{matched_motif}_{pos}.png", dpi=300, bbox_inches='tight')
        plt.close(fig)

        # overlay the motif PWM logo on the gradient logo 
        fig, ax = plt.subplots(figsize=(10, 4))
        logo = logomaker.Logo(df_seq, ax=ax, color_scheme='classic')
        ax.set_title(f'Motif: {matched_motif} at Pos: {row["pos"]} with PWM Overlay')
        # overlay motif PWM as a smaller logo at the aligned position above the gradient logo
        motif_logo_ax = fig.add_axes([0.52, 1.2, len(df_motif) / len(df_seq), 0.5])
        logomaker.Logo(df_motif, ax=motif_logo_ax, color_scheme='classic')
        motif_logo_ax.set_xticks([])
        motif_logo_ax.set_yticks([])
        plt.savefig(f"{args.out_dir}/motif_logo_with_pwm_{matched_motif}_{pos}.png", dpi=300, bbox_inches='tight')
        plt.close(fig)

        # insert the motif into the input sequence at the position and see effect on prediction
        # get one-hot encoded motif sequence
        motif_seq = []
        for i in range(motif.length):
            col = []
            for base in ['A', 'T', 'C', 'G', 'N']:
                col.append(matrix_dict.get(base, [0]*motif.length)[i])
            motif_seq.append(col)
        motif_seq = np.array(motif_seq)  # Shape: (motif.length, 5)
        # create a copy of the original sequence
        modified_seq = seq_region.copy()
        # insert motif at the position
        insert_start = pos
        insert_end = insert_start + motif.length
        if insert_end > modified_seq.shape[0]:
            print(f"Cannot insert motif {matched_motif} at position {pos} due to length constraints.")
            continue
        modified_seq[insert_start:insert_end, :5] = motif_seq
        # prepare modified input tensor
        modified_inputs = infer.preprocess_default(modified_seq, ctcf_region, atac_region, other_regions)
        modified_inputs = modified_inputs.to(device)
        # forward pass
        with torch.no_grad():
            try:
                modified_outputs = model(modified_inputs)
            except Exception as e:
                input_dict = {'seq': modified_inputs[..., :5], 'ctcf': modified_inputs[..., 5:6], 'atac': modified_inputs[..., 6:7]}
                modified_outputs = model(input_dict, predict_tracks=all_tracks + ['hic'])
            try:
                modified_pred_hic = (modified_outputs.get('hic') + modified_outputs.get('hic').transpose(1, 2)) / 2
                modified_pred_hic = torch.expm1(modified_pred_hic)
            except AttributeError as e:  # corigami base model
                modified_pred_hic = (modified_outputs + modified_outputs.transpose(1, 2)) / 2
                modified_pred_hic = torch.expm1(modified_pred_hic)
        # compare the score in the target region
        modified_score = modified_pred_hic[0, p_start1:p_end1, p_start2:p_end2].sum()  

        # generate baseline prediction with random sequence insertion at the same position
        random_seq = np.random.rand(motif.length, 5)
        random_seq = random_seq / random_seq.sum(axis=1, keepdims=True)  # normalize to sum to 1
        baseline_seq = seq_region.copy()
        baseline_seq[insert_start:insert_end, :5] = random_seq
        baseline_inputs = infer.preprocess_default(baseline_seq, ctcf_region, atac_region, other_regions)
        baseline_inputs = baseline_inputs.to(device)
        with torch.no_grad():
            try:
                baseline_outputs = model(baseline_inputs)
            except Exception as e:
                input_dict = {'seq': baseline_inputs[..., :5], 'ctcf': baseline_inputs[..., 5:6], 'atac': baseline_inputs[..., 6:7]}
                baseline_outputs = model(input_dict, predict_tracks=all_tracks + ['hic'])
            try:
                baseline_pred_hic = (baseline_outputs.get('hic') + baseline_outputs.get('hic').transpose(1, 2)) / 2
                baseline_pred_hic = torch.expm1(baseline_pred_hic)
            except AttributeError as e:  # corigami base model
                baseline_pred_hic = (baseline_outputs + baseline_outputs.transpose(1, 2)) / 2
                baseline_pred_hic = torch.expm1(baseline_pred_hic)
        baseline_score = baseline_pred_hic[0, p_start1:p_end1, p_start2:p_end2].sum()

        # visualize the difference in the Hi-C map
        diff_hic = modified_pred_hic - baseline_pred_hic
        fig, ax = plt.subplots(figsize=(12, 6))
        im = ax.imshow(diff_hic[0].detach().cpu().numpy(), cmap='bwr', norm=plt.Normalize(vmin=-np.percentile(np.abs(diff_hic.detach().cpu().numpy()), 99), vmax=np.percentile(np.abs(diff_hic.detach().cpu().numpy()), 99)))
        ax.set_title(f'Change in Hi-C Map After Inserting Motif {matched_motif} at Pos {pos}')
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        # plot motif insertion position lines
        ax.axvline(x=insert_start // args.resolution, color='green', linestyle='--', label='Motif Insertion Position')
        ax.axhline(y=insert_start // args.resolution, color='green', linestyle='--')
        ax.legend()
        plt.savefig(f"{args.out_dir}/hic_change_after_inserting_{matched_motif}_{pos}.png", dpi=300, bbox_inches='tight')
        plt.close(fig)

        mean_loop_change = diff_hic[0, p_start1:p_end1, p_start2:p_end2].mean().item()
        print(f"Mean change in target locus after inserting motif {matched_motif} at pos {pos}: {mean_loop_change:.2f}")
        hic_diffs.append(mean_loop_change)

        # also visualize the modified rad21 signal
        if 'rad21' in input_tracks:
            rad21_idx = input_tracks.index('rad21') + 5  # +5 for the sequence channels
            pred_1d = outputs.get('1d')
            modified_pred_1d = modified_outputs.get('1d')
            original_rad21 = pred_1d['rad21'][0].detach().cpu().numpy().flatten()
            modified_rad21 = modified_pred_1d['rad21'][0].detach().cpu().numpy().flatten()
            # undo log1p
            original_rad21 = np.expm1(original_rad21)
            modified_rad21 = np.expm1(modified_rad21)
            # clip negative values to zero
            original_rad21 = np.clip(original_rad21, 0, 10)
            modified_rad21 = np.clip(modified_rad21, 0, 10)
            rad21_diff = modified_rad21 - original_rad21
            fig, axs = plt.subplots(3, 1, figsize=(12, 8), sharex=True)
            axs[0].plot(original_rad21, color='blue')
            axs[0].set_title('Original Predicted Rad21 Signal')
            axs[1].plot(modified_rad21, color='green')
            axs[1].set_title(f'Modified Predicted Rad21 Signal After Inserting {matched_motif}')
            axs[2].plot(rad21_diff, color='red')
            axs[2].set_title('Difference in Rad21 Signal')
            axs[2].set_xlabel('Genomic Position (100bp bins)')
            plt.savefig(f"{args.out_dir}/rad21_change_after_inserting_{matched_motif}_{pos}.png", dpi=300, bbox_inches='tight')
            plt.close(fig)
            mean_rad21_change = rad21_diff.mean().item()
            #print(f"Mean change in Rad21 signal after inserting motif {matched_motif} at pos {pos}: {mean_rad21_change:.3f}")
            rad21_diffs.append(mean_rad21_change)

    # summarize the effects
    summary_df = top_motifs.copy()
    summary_df['mean_hic_change'] = hic_diffs
    if rad21_diffs:
        summary_df['mean_rad21_change'] = rad21_diffs
    summary_df.to_csv(f"{args.out_dir}/motif_insertion_effects_summary.tsv", sep='\t', index=False)
    # sort by diff
    summary_df = summary_df.sort_values(by=['mean_hic_change', 'mean_rad21_change'] if rad21_diffs else ['mean_hic_change'], ascending=False).reset_index(drop=True)

    fig, ax = plt.subplots(figsize=(8,6))
    ax.scatter(summary_df['corr_max'], summary_df['mean_hic_change'])
    ax.set_xlabel('Motif Correlation with Saliency Score')
    ax.set_ylabel('Mean Hi-C Change After Insertion')
    ax.set_title('Motif Correlation vs Hi-C Change After Insertion')
    # label points with motif names
    for i, row in summary_df.iterrows():
        ax.text(row['corr_max'], row['mean_hic_change'], row['motif'])
        if i >= 20:
            break  # only label top 20 for clarity
    plt.savefig(f"{args.out_dir}/motif_correlation_vs_hic_change.png", dpi=300, bbox_inches='tight')
    plt.close(fig)
    
    if rad21_diffs:
        fig, ax = plt.subplots(figsize=(8,6))
        ax.scatter(summary_df['corr_max'], summary_df['mean_rad21_change'])
        ax.set_ylabel('Mean Rad21 Change After Insertion')
        ax.set_title('Motif Correlation vs Rad21 Change After Insertion')
        # label points with motif names
        for i, row in summary_df.iterrows():
            ax.text(row['corr_max'], row['mean_rad21_change'], row['motif'])
            if i >= 20:
                break
        plt.savefig(f"{args.out_dir}/motif_correlation_vs_rad21_change.png", dpi=300, bbox_inches='tight')
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(8,6))
        ax.scatter(summary_df['mean_rad21_change'], summary_df['mean_hic_change'])
        ax.set_xlabel('Mean Rad21 Change After Insertion')
        ax.set_ylabel('Mean Hi-C Change After Insertion')
        ax.set_title('Rad21 Change vs Hi-C Change After Insertion')
        # label points with motif names
        for i, row in summary_df.iterrows():
            ax.text(row['mean_rad21_change'], row['mean_hic_change'], row['motif'])
            if i >= 20:
                break
        plt.savefig(f"{args.out_dir}/rad21_change_vs_hic_change.png", dpi=300, bbox_inches='tight')
        plt.close(fig)
            

    # plot heatmap with top motif positions marked
    fig, ax = plt.subplots(figsize=(12, 6))
    im = ax.imshow(pred_hic[0].detach().cpu().numpy(), cmap='Reds', norm=plt.Normalize(vmin=0, vmax=np.percentile(pred_hic.detach().cpu().numpy(), 99)))
    ax.set_title('Predicted Hi-C Map with Top Motif Positions')
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    for idx, row in top_motifs.iterrows():
        pos = row['pos_100bp'] // args.resolution
        ax.plot([0, args.mat_size], [pos, pos], color='blue', linestyle='--', linewidth=1)
        ax.plot([pos, pos], [0, args.mat_size], color='blue', linestyle='--', linewidth=1)
    plt.savefig(f"{args.out_dir}/predicted_hic_with_motifs.png", dpi=300, bbox_inches='tight')
    plt.close(fig)

if __name__ == '__main__':
    main()