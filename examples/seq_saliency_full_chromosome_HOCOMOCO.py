import os
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
# pyjaspar is no longer needed
# from pyjaspar import jaspardb 
from scipy.signal import find_peaks
from scipy.ndimage import gaussian_filter1d
import multiprocessing as mp
import torch.nn.functional as F
from functools import partial

# Import necessary functions from the C.Shark codebase
# Ensure cshark is installed and accessible in your environment
import cshark.inference.utils.inference_utils as infer
from cshark.inference.utils.model_utils import load_default, get_all_track_names

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
    # This assumes A, T, C, G, N one-hot encoding.
    complement_map = {0: 1, 1: 0, 2: 3, 3: 2, 4: 4}  # A<->T, C<->G, N->N
    
    # Check if the array is writeable, if not, create a copy
    if not seq_array.flags.writeable:
        seq_array = seq_array.copy()

    rev_comp = seq_array[::-1, :]
    # Apply the complement mapping by swapping columns
    rev_comp = rev_comp[:, [complement_map[i] for i in range(rev_comp.shape[1])]]
    return rev_comp

class ParseKwargs(argparse.Action):
    def __call__(self, parser, namespace, values, option_string=None):
        setattr(namespace, self.dest, dict())
        for value in values:
            key, value = value.split('=', 1)
            getattr(namespace, self.dest)[key] = value


# This worker function now expects PyTorch tensors on the correct device
def _process_motif_gpu(motif, chromosome_gradient_pwms_gpu, score_threshold=5.0):
    """Worker function for GPU-based motif scanning using 1D convolution."""
    try:
        # For MemeMotif, this returns a dict of probabilities
        matrix_dict = motif.counts.normalize()
    except Exception:
        return []

    # The internal order for gradients and PWMs is A, T, C, G, N
    bases = ['A', 'T', 'C', 'G', 'N']
    pwm = np.array([matrix_dict.get(b, [0.0]*motif.length) for b in bases]).T
    len_pwm = pwm.shape[0]
    
    if len_pwm > chromosome_gradient_pwms_gpu.shape[2]:
        return []
    
    # Convert PWM to a PyTorch tensor and shape it as a conv1d kernel
    # Kernel shape: (out_channels, in_channels, width) -> (1, 5, MotifLength)
    device = chromosome_gradient_pwms_gpu.device
    kernel_fw = torch.tensor(pwm, dtype=torch.float32).to(device).T.unsqueeze(0)
    
    pwm_rc = reverse_complement(pwm)
    kernel_rc = torch.tensor(pwm_rc, dtype=torch.float32).to(device).T.unsqueeze(0)

    # Perform convolution
    scores_fw_tensor = F.conv1d(chromosome_gradient_pwms_gpu, kernel_fw)
    scores_rc_tensor = F.conv1d(chromosome_gradient_pwms_gpu, kernel_rc)

    # convert to cosine similarity
    scores_fw = scores_fw_tensor.squeeze() / len_pwm
    scores_rc = scores_rc_tensor.squeeze() / len_pwm
    
    # Find passing indices on the GPU
    passing_indices = torch.where((scores_fw.abs() > score_threshold) | (scores_rc.abs() > score_threshold))[0]

    if len(passing_indices) == 0:
        return []

    # Move results back to CPU for standard Python list creation
    scores_fw_cpu = scores_fw.cpu().numpy()
    scores_rc_cpu = scores_rc.cpu().numpy()
    passing_indices_cpu = passing_indices.cpu().numpy()
    
    rows = [{
        'motif': motif.name,
        'pos': int(pos),
        'score_fw': float(scores_fw_cpu[pos]),
        'score_rc': float(scores_rc_cpu[pos])
    } for pos in passing_indices_cpu]
        
    return rows


def _process_motif(motif, chromosome_gradient_pwms, score_threshold=5.0):
    """Worker function to scan for a single motif across the entire chromosome's gradients."""
    rows = []
    try:
        matrix_dict = motif.counts.normalize()
    except Exception:
        return rows

    # The internal order for gradients and PWMs is A, T, C, G, N
    bases = ['A', 'T', 'C', 'G', 'N']
    pwm = np.array([matrix_dict.get(b, [0.0]*motif.length) for b in bases]).T # Shape: (MotifLength, 5)

    pwm_rc = reverse_complement(pwm)
    len_pwm = pwm.shape[0]
    
    seq_len = chromosome_gradient_pwms.shape[0]
    if len_pwm > seq_len:
        return rows

    # Slide the PWM over the chromosome gradients and calculate correlation
    for i in range(seq_len - len_pwm + 1):
        window = chromosome_gradient_pwms[i:i + len_pwm]
        # Calculate scores for forward and reverse complement motifs
        score_fw = abs(np.dot(window.flatten(), pwm.flatten()) / len_pwm)
        score_rc = abs(np.dot(window.flatten(), pwm_rc.flatten()) / len_pwm)
        
        if score_fw > score_threshold or score_rc > score_threshold:
            rows.append({
                'motif': motif.name,
                'pos': int(i),
                'score_fw': float(score_fw),
                'score_rc': float(score_rc)
            })
    return rows

def save_scores_as_bigwig(scores, original_bw_path, chr_name, out_path):
    """Saves chromosome-wide attribution scores as a BigWig file."""
    if not original_bw_path or not os.path.exists(original_bw_path):
        print(f"Error: Reference BigWig file not found at {original_bw_path}")
        try:
            with pyBigWig.open(out_path, "w") as bw_out:
                header = [(chr_name, len(scores))]
                bw_out.addHeader(header)
                starts = np.arange(0, len(scores), dtype=np.int64)
                ends = starts + 1
                bw_out.addEntries([chr_name] * len(starts), starts, ends=ends, values=scores.astype(np.float64))
            print(f"Successfully saved attribution scores to {out_path} with a generated header.")
        except Exception as e_fallback:
            print(f"Fallback BigWig save failed: {e_fallback}")
        return

    try:
        with pyBigWig.open(original_bw_path) as bw_in:
            if chr_name not in bw_in.chroms():
                print(f"Error: Chromosome '{chr_name}' not found in the header of {original_bw_path}.")
                header = [(chr_name, len(scores))]
            else:
                 header = list(bw_in.chroms().items())

        with pyBigWig.open(out_path, "w") as bw_out:
            bw_out.addHeader(header)
            
            starts = np.arange(0, len(scores), dtype=np.int64)
            ends = starts + 1
            values = scores.astype(np.float64)

            bw_out.addEntries([chr_name] * len(starts), starts, ends=ends, values=values)
        print(f"Successfully saved attribution scores to {out_path}")

    except Exception as e:
        print(f"Error saving BigWig file: {e}")
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
    parser = argparse.ArgumentParser(description='C.Shark Gradient Attribution across a full chromosome.')
    parser.add_argument('--model', dest='model_path', required=True, help='Path to the model checkpoint.')
    parser.add_argument('--chr', dest='chr_name', required=True, help='Chromosome for prediction (e.g., chr1).')
    parser.add_argument('--step-size', dest='step_size', type=int, default=WINDOW_SIZE // 2, help='Step size for the sliding window.')
    parser.add_argument('--seq', dest='seq_path', required=True, help='Path to the folder with sequence .fa.gz files.')
    parser.add_argument('--out-file', dest='out_file', required=True, help='Output path for the attribution scores BigWig file.')
    
    # New argument for the MEME file
    parser.add_argument('--meme-file', dest='meme_file', required=True, help='Path to the HOCOMOCO MEME file for motif scanning.')

    parser.add_argument('--bigwigs', nargs='*', help='Paths to the bigwig files for genomic features, specified as key=value pairs (e.g., ctcf=path/to/ctcf.bw).', 
                        action=ParseKwargs, required=True)
    
    parser.add_argument('--resolution', dest='resolution', type=int, default=8192, help='Resolution of the Hi-C map used by the model.')
    parser.add_argument('--matrix-size', dest='mat_size', type=int, default=256, help='Matrix size used by the model.')
    parser.add_argument('--latent_size', dest='mid_hidden', type=int, default=256, help='Latent size of the model.')
    parser.add_argument('--motif-score-threshold', dest='motif_score_threshold', type=float, default=3.0, help='Threshold for motif correlation scores.')
    parser.add_argument('--num-procs', dest='num_procs', type=int, default=10, help='Number of processes for parallel motif scanning.')

    args = parser.parse_args()

    # --- 1. Setup ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    ref_bw_path = args.bigwigs.get('ctcf')
    if not ref_bw_path or not os.path.exists(ref_bw_path):
        print("Error: A reference BigWig for CTCF (--bigwigs ctcf=...) is required to determine chromosome length.")
        sys.exit(1)

    with pyBigWig.open(ref_bw_path) as bw:
        if args.chr_name not in bw.chroms():
            print(f"Error: Chromosome '{args.chr_name}' not found in reference BigWig '{ref_bw_path}'.")
            sys.exit(1)
        chr_length = bw.chroms(args.chr_name)

    print(f"Processing {args.chr_name} with length {chr_length}...")

    chromosome_gradient_pwms = np.zeros((chr_length, 5), dtype=np.float32)
    chromosome_counts = np.zeros(chr_length, dtype=np.int16)

    all_tracks, _, input_tracks = get_all_track_names(args.model_path)
    model = load_default(
        args.model_path, 
        num_genomic_features=len(input_tracks),
        mat_size=args.mat_size,
        mid_hidden=args.mid_hidden,
        seq_filter_size=15,
        recon_1d=True
    ).to(device)
    model.eval()

    # --- 2. Sliding Window Gradient Calculation ---
    window_starts = range(0, chr_length - WINDOW_SIZE + 1, args.step_size)
    
    for start_pos in tqdm(window_starts, desc="Processing chromosome windows"):
        end_pos = start_pos + WINDOW_SIZE
        try:
            bigwig_paths = [args.bigwigs.get(track) for track in input_tracks if args.bigwigs.get(track)]
            seq_region, ctcf_region, atac_region, other_regions = infer.load_region(
                 args.chr_name, start_pos, args.seq_path, args.bigwigs.get('ctcf'), args.bigwigs.get('atac'), other_paths=bigwig_paths[2:] if len(bigwig_paths) > 2 else None,
                window=WINDOW_SIZE
            )
        except Exception as e:
            print(f"Skipping window at {start_pos} due to data loading error: {e}")
            continue

        inputs = infer.preprocess_default(seq_region, ctcf_region, atac_region, other_regions).to(device)
        inputs.requires_grad = True
        
        model.zero_grad()
        try:
            outputs = model(inputs)
        except Exception as e:
            input_dict = {'seq': inputs[..., :5], 'ctcf': inputs[..., 5:6], 'atac': inputs[..., 6:7]}
            outputs = model(input_dict, predict_tracks=all_tracks + ['hic'])
        pred_hic = (outputs.get('hic') + outputs.get('hic').transpose(1, 2)) / 2
        pred_hic = torch.expm1(pred_hic)

        score = pred_hic.sum()
        
        if score.item() > 0:
            score.backward()
            gradients = inputs.grad[0].cpu().numpy()
            seq_gradients = gradients[:, :5]
            chromosome_gradient_pwms[start_pos:end_pos] += seq_gradients
            chromosome_counts[start_pos:end_pos] += 1

    # --- 3. Finalize Scores by Averaging ---
    print("Averaging scores from overlapping windows...")
    valid_indices = chromosome_counts > 0
    chromosome_gradient_pwms[valid_indices] /= chromosome_counts[valid_indices, np.newaxis]
    
    final_saliency_scores = np.abs(chromosome_gradient_pwms).sum(axis=-1)
    
    # --- 4. Save Chromosome-Wide Saliency Scores ---
    print("Saving chromosome-wide attribution scores...")
    save_scores_as_bigwig(final_saliency_scores, ref_bw_path, args.chr_name, args.out_file)

    # --- 5. Chromosome-Wide Motif Scanning ---
    print("Scanning for motifs across the entire chromosome...")
    
    # Load motifs from the specified MEME file
    motifs = parse_meme_file(args.meme_file)
    if not motifs:
        print("No motifs were loaded from the MEME file. Exiting.")
        return

    motifs_list = list(motifs)

    print("Moving gradients to GPU for motif scanning...")
    # Reshape for conv1d: (batch, channels, length) -> (1, 5, Length)
    grad_pwms_tensor_gpu = torch.from_numpy(chromosome_gradient_pwms).T.unsqueeze(0).to(device)
    worker = partial(_process_motif_gpu, 
                     chromosome_gradient_pwms_gpu=grad_pwms_tensor_gpu, 
                     score_threshold=args.motif_score_threshold)
    
    # Using a single-process loop for GPU operations is often more stable
    results = []
    for motif in tqdm(motifs_list, desc="Scanning motifs on GPU"):
        rows = worker(motif)
        if rows: results.extend(rows)

    if not results:
        print("No motifs found above the specified score threshold.")
        return

    motif_df = pd.DataFrame(results)
    print(f"Found {len(motif_df)} motif occurrences above the score threshold.")
    
    motif_df['score_max'] = motif_df[['score_fw', 'score_rc']].abs().max(axis=1)
    
    # To reduce redundancy, group by nearby positions and take the max score
    motif_df['pos_100bp'] = (motif_df['pos'] // 100) * 100
    motif_df = motif_df.loc[motif_df.groupby(['motif', 'pos_100bp'])['score_max'].idxmax()]


    output_motif_path = os.path.splitext(args.out_file)[0] + "_motifs.tsv"
    motif_df.to_csv(output_motif_path, sep='\t', index=False)
    print(f"Saved motif scanning results to {output_motif_path}")

    # Print top motifs by max score
    top_motifs = motif_df.nlargest(40, 'score_max')
    print("\nTop 40 motifs correlated with saliency scores:")
    print(top_motifs[['motif', 'pos', 'score_max']])

    # Print most frequent motifs
    top_motif_counts = motif_df['motif'].value_counts().head(40)
    print("\nTop 40 most frequent motifs:")
    print(top_motif_counts)


if __name__ == '__main__':
    main()