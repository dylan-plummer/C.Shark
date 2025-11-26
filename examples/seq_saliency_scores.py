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
import seaborn as sns
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

def one_hot_to_seq(one_hot_array):
    """Convert a one-hot encoded sequence array to a string sequence."""
    int_to_base = {0: 'A', 1: 'T', 2: 'C', 3: 'G', 4: 'N'}
    seq = ''.join([int_to_base[np.argmax(pos)] for pos in one_hot_array])
    return seq

class MemeMotif:
    """A simple class to hold motif data parsed from a MEME file."""
    def __init__(self, name, matrix):
        self.name = name
        self._matrix = matrix
        self.length = matrix.shape[0]

    @property
    def counts(self):
        return self

    def normalize(self):
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
            if current_motif_name and matrix_lines:
                matrix = np.array([list(map(float, row.split())) for row in matrix_lines])
                motifs.append(MemeMotif(current_motif_name, matrix))
            parts = line.split()
            current_motif_name = parts[1] if len(parts) > 1 else "Unknown"
            matrix_lines = []
            is_parsing_matrix = False
        elif line.startswith("letter-probability matrix"):
            is_parsing_matrix = True
        elif is_parsing_matrix and line and (line[0].isdigit() or line.startswith(" 0.")):
            matrix_lines.append(line)

    if current_motif_name and matrix_lines:
        matrix = np.array([list(map(float, row.split())) for row in matrix_lines])
        motifs.append(MemeMotif(current_motif_name, matrix))
    
    print(f"Successfully parsed {len(motifs)} motifs from {filepath}")
    return motifs

def reverse_complement(seq_array):
    """Generate the reverse complement of a one-hot encoded DNA sequence array."""
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
    matrix = np.array([matrix_dict.get(b, [0]*motif.length) for b in bases]).T
    
    len_pwm = matrix.shape[0]
    seq_len = gradient_pwms.shape[0]
    
    if len_pwm > seq_len:
        return []

    pwm_flat = matrix.flatten()
    pwm_rc_flat = reverse_complement(matrix).flatten()

    n_windows = seq_len - len_pwm + 1
    itemsize = gradient_pwms.itemsize
    window_view = as_strided(gradient_pwms,
                             shape=(n_windows, len_pwm, 5),
                             strides=(5 * itemsize, 5 * itemsize, itemsize))
    
    all_windows_flat = window_view.reshape(n_windows, -1)
    scores_fw = all_windows_flat @ pwm_flat / len_pwm
    scores_rc = all_windows_flat @ pwm_rc_flat / len_pwm

    passing_indices = np.where((np.abs(scores_fw) > score_threshold) | (np.abs(scores_rc) > score_threshold))[0]

    if len(passing_indices) == 0:
        return []

    return [{
        'motif': motif.name,
        'motif_length': len_pwm,
        'pos': int(pos),
        'score_fw': float(scores_fw[pos]),
        'score_rc': float(scores_rc[pos])
    } for pos in passing_indices]

def save_scores_as_bigwig(scores, original_bw_path, chr_name, start, out_path):
    """Saves attribution scores as a BigWig file."""
    try:
        with pyBigWig.open(original_bw_path) as bw_in:
            header = bw_in.header()['cl'] if bw_in.header().get('cl') is not None else list(bw_in.chroms().items())
        
        with pyBigWig.open(out_path, "w") as bw_out:
            bw_out.addHeader(header)
            scores_np = scores.detach().cpu().numpy().flatten()
            starts = np.arange(start, start + len(scores_np), dtype=np.int64)
            ends = starts + 1
            values = scores_np.astype(np.float64)
            bw_out.addEntries([chr_name] * len(starts), starts, ends=ends, values=values)
        print(f"Successfully saved attribution scores to {out_path}")

    except Exception as e:
        print(f"Error saving BigWig file: {e}")
        bedgraph_path = os.path.splitext(out_path)[0] + ".bedgraph"
        df = pd.DataFrame({'chrom': chr_name, 'start': starts, 'end': ends, 'score': values})
        df.to_csv(bedgraph_path, sep='\t', header=False, index=False)
        print(f"Successfully saved attribution scores to {bedgraph_path}")


def run_perturbation(model, base_seq, ctcf_track, atac_track, other_tracks, motif_pos, motif_len, ko_mode, device, all_tracks):
    """Helper function to run a single perturbation and return the predicted Hi-C map."""
    
    perturbed_seq = base_seq.copy()
    insert_start, insert_end = motif_pos, motif_pos + motif_len
    if insert_end > perturbed_seq.shape[0]:
        return None

    if ko_mode == 'zero':
        perturbed_seq[insert_start:insert_end, :5] = 0
    elif ko_mode.lower() == 'n':
        perturbed_seq[insert_start:insert_end, :5] = np.array([0, 0, 0, 0, 1])
    else:  # 'shuffle'
        random_seq = np.random.rand(motif_len, 5)
        random_seq = random_seq / random_seq.sum(axis=1, keepdims=True)
        perturbed_seq[insert_start:insert_end, :5] = random_seq
        
    perturb_inputs = infer.preprocess_default(perturbed_seq, ctcf_track, atac_track, other_tracks).to(device)

    with torch.no_grad():
        try:
            perturb_outputs = model(perturb_inputs)
            perturb_hic_pred = (perturb_outputs.get('hic') + perturb_outputs.get('hic').transpose(1, 2)) / 2
        except Exception:
            input_dict = {'seq': perturb_inputs[..., :5], 'ctcf': perturb_inputs[..., 5:6], 'atac': perturb_inputs[..., 6:7]}
            perturb_outputs = model(input_dict, predict_tracks=all_tracks + ['hic'])
            perturb_hic_pred = (perturb_outputs.get('hic') + perturb_outputs.get('hic').transpose(1, 2)) / 2
        
        perturb_hic_pred = torch.expm1(perturb_hic_pred)

    return perturb_hic_pred, perturb_outputs


def main():
    parser = argparse.ArgumentParser(description='Causal TF discovery using in-silico perturbation.')
    # (Arguments are identical to the original script)
    parser.add_argument('--model', dest='model_path', required=True, help='Path to the model checkpoint.')
    parser.add_argument('--chr', dest='chr_name', required=True, help='Chromosome for prediction.')
    parser.add_argument('--start', dest='start', type=int, required=True, help=f'Starting point for the {WINDOW_SIZE}bp input window.')
    parser.add_argument('--seq', dest='seq_path', required=True, help='Path to the folder with sequence .fa.gz files.')
    parser.add_argument('--out-dir', dest='out_dir', required=True, help='Output directory')
    parser.add_argument('--meme-file', dest='meme_file', required=False, help='Path to the HOCOMOCO MEME file for motif scanning.')
    parser.add_argument('--tf', dest='tf', required=False, help='Name of the transcription factor for motif scanning.')
    parser.add_argument('--viz-bp', dest='viz_bp', type=int, default=50, help='Base pair range(+/-) for visualization.')
    parser.add_argument('--pad-bp', dest='pad_bp', type=int, default=5, help='Base pair range(+/-) for outputting sequences.')
    parser.add_argument('--ko-mode', dest='ko_mode', default='shuffle', choices=['shuffle', 'zero', 'N', 'n'], help='Knockout mode for input features.')
    parser.add_argument('--bigwigs', nargs='*', help='Paths to the bigwig files for genomic features, specified as key=value pairs.', action=ParseKwargs)
    parser.add_argument('--snp-file', dest='snp_file', required=False, help='Path to the SNP file in BED format.')
    parser.add_argument('--target-locus', dest='target_locus', required=True, help='The region in the Hi-C map to maximize, format: chr1:start1-end1_chr1:start2-end2')
    parser.add_argument('--plot-locus', dest='plot_locus', required=True, help='The region in the Hi-C map to plot, format: chr1:start1-end1')
    parser.add_argument('--resolution', dest='resolution', type=int, default=8192, help='Resolution of the Hi-C map used by the model.')
    parser.add_argument('--matrix-size', dest='mat_size', type=int, default=256, help='Matrix size used by the model.')
    parser.add_argument('--latent_size', dest='mid_hidden', type=int, default=256, help='Latent size of the model.')
    parser.add_argument('--ignore-motifs', dest='ignore_motifs', action='store_true', help='Ignore motif scanning and just focus on raw sequence')
    parser.add_argument('--ctcf-ko', dest='ctcf_ko', action='store_true', help='Whether to knockout CTCF peaks in the input.')
    parser.add_argument('--no-abs', dest='abs', action='store_false', help='Whether to use absolute values for saliency scores.')
    parser.add_argument('--n-loci', dest='n_motifs', type=int, default=10, help='Number of top saliency peaks to investigate.')
    parser.add_argument('--n-causal-viz', dest='n_causal_viz', type=int, default=10, help='Number of top causal TFs to visualize.')
    parser.add_argument('--target-1d-length', dest='target_1d_length', type=int, default=8192, help='Length of the 1D targets used by the model.')
    parser.add_argument('--vmin', dest='vmin', type=float, default=0, help='Minimum value for Hi-C plotting.')
    parser.add_argument('--vmax', dest='vmax', type=float, default=None, help='Maximum value for Hi-C plotting.')

    args = parser.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    # --- 1. Load Data and Model ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    all_tracks, _, input_tracks = get_all_track_names(args.model_path)
    bigwig_paths = []
    if args.bigwigs:
        for track_name in input_tracks:
            if track_name in args.bigwigs:
                bigwig_paths.append(args.bigwigs[track_name])

    print(f"Loading data for region {args.chr_name}:{args.start}...")
    other_bigwig_paths = [p for k, p in args.bigwigs.items() if k not in ['ctcf', 'atac']]
    seq_region, ctcf_region, atac_region, other_regions = infer.load_region(
        args.chr_name, args.start, args.seq_path, 
        args.bigwigs.get('ctcf'), args.bigwigs.get('atac'), other_paths=other_bigwig_paths,
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
        ).to(device).eval()
    except Exception as e:
        model = load_default(
            args.model_path, 
            num_genomic_features=len(input_tracks),
            mat_size=args.mat_size,
            mid_hidden=args.mid_hidden,
            seq_filter_size=15,
            target_1d_length=args.target_1d_length,
            recon_1d=False
        ).to(device).eval()

    # --- 2. Get Saliency Map to Identify Candidate Regions ---
    print("Preprocessing input data and enabling gradient tracking...")
    inputs = infer.preprocess_default(seq_region, ctcf_region, atac_region, other_regions)
    inputs.requires_grad = True

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

    # Parse target and plot loci
    locus1_str, locus2_str = args.target_locus.split('_')
    p_start1 = (int(locus1_str.split(':')[1].split('-')[0]) - args.start) // args.resolution
    p_end1 = (int(locus1_str.split(':')[1].split('-')[1]) - args.start) // args.resolution
    p_start2 = (int(locus2_str.split(':')[1].split('-')[0]) - args.start) // args.resolution
    p_end2 = (int(locus2_str.split(':')[1].split('-')[1]) - args.start) // args.resolution
    plot_p_start = (int(args.plot_locus.split(':')[1].split('-')[0]) - args.start) // args.resolution
    plot_p_end = (int(args.plot_locus.split(':')[1].split('-')[1]) - args.start) // args.resolution
    
    score = pred_hic[0, p_start1:p_end1, p_start2:p_end2].sum()
    print(f"Initial score for target locus: {score.item()}")
    
    print("Performing backward pass to get gradients...")
    score.backward()
    gradients = inputs.grad[0]
    seq_gradients = gradients[:, :5] * seq_region[:, :5]
    saliency_scores = (seq_gradients).sum(dim=-1)

    smoothed_scores_abs = gaussian_filter1d(torch.abs(saliency_scores).detach().cpu().numpy(), sigma=5)
    smoothed_scores = gaussian_filter1d(saliency_scores.detach().cpu().numpy(), sigma=5)
    peak_indices = find_peaks(smoothed_scores_abs, distance=100, height=np.percentile(smoothed_scores_abs, 90))[0]
    top_peak_indices = peak_indices[np.argsort(smoothed_scores_abs[peak_indices])][::-1][:args.n_motifs]
    print(f"Found {len(top_peak_indices)} saliency peaks to investigate for causal motifs.")

    # convert to genomic coordinates and save top peaks dataframe with surrounding sequence
    bp_pad = args.pad_bp
    peak_genomic_starts = args.start + top_peak_indices
    peak_scores = smoothed_scores[top_peak_indices]
    ref_seq = seq_region
    peak_seq = []
    ref_bases = []
    for peak_idx in top_peak_indices:
        seq_start = max(0, peak_idx - bp_pad)
        seq_end = min(len(ref_seq), peak_idx + bp_pad + 1)
        seq_slice = ref_seq[seq_start:seq_end, :5]
        seq_str = one_hot_to_seq(seq_slice)
        peak_seq.append(seq_str)
        ref_bases.append(seq_str[bp_pad])  # center base
    peak_df = pd.DataFrame({'chrom': [args.chr_name] * len(peak_genomic_starts),
                            'pos': peak_genomic_starts,
                            'score': peak_scores,
                            'ref': ref_bases,
                            'sequence': peak_seq})
    peak_df.to_csv(f"{args.out_dir}/sequence_peaks.tsv", sep='\t', index=False)

    if not args.ignore_motifs:
        # --- 3. Scan for all potential motifs ---
        if args.meme_file is None:
            jdb_obj = jaspardb(release='JASPAR2024')
            motifs = jdb_obj.fetch_motifs(collection=['CORE'], tax_group=['Vertebrates'], species=['9606'], tf_name=args.tf)
        else:
            motifs = parse_meme_file(args.meme_file)
            if args.tf:
                motifs = [m for m in motifs if args.tf.lower() in m.name.lower()]

        gradient_pwms = seq_gradients.detach().cpu().numpy()
        worker = partial(_process_motif_vectorized, gradient_pwms=gradient_pwms, seq_start=args.start, score_threshold=0.05)
        results = []
        with mp.Pool(processes=min(5, len(motifs))) as pool:
            for motif_rows in tqdm(pool.imap_unordered(worker, motifs), total=len(motifs)):
                if motif_rows: results.extend(motif_rows)

        if not results:
            print("No motifs found matching the saliency profile. Exiting.")
            return
            
        motif_df = pd.DataFrame(results)
        motif_df['score_max'] = motif_df[['score_fw', 'score_rc']].max(axis=1)
        motif_df['score_magnitude'] = motif_df[['score_fw', 'score_rc']].abs().max(axis=1)

        # --- PHASE 1: Systematic In-Silico Perturbation (Discovery) ---
        print("\n--- Phase 1: Starting systematic perturbation of candidate motifs ---")
        all_perturbation_results = []
        MAX_MOTIFS_TO_TEST_PER_PEAK = 5

        for peak_pos in tqdm(top_peak_indices, desc="Investigating Saliency Peaks"):
            window_radius = args.viz_bp * 2 # Use a wider window to find candidate motifs
            seq_start_idx = max(0, peak_pos - window_radius)
            seq_end_idx = min(len(gradient_pwms), peak_pos + window_radius)

            motifs_in_window = motif_df[(motif_df['pos'] + motif_df['motif_length'] > seq_start_idx) & (motif_df['pos'] < seq_end_idx)].copy()
            if motifs_in_window.empty: continue
            
            # Prioritize motifs with higher correlation scores, but test several
            motifs_in_window = motifs_in_window.sort_values('score_magnitude', ascending=False)
            
            for _, motif_to_perturb in motifs_in_window.head(MAX_MOTIFS_TO_TEST_PER_PEAK).iterrows():
                motif_name = motif_to_perturb['motif']
                motif_pos = motif_to_perturb['pos']
                motif_len = motif_to_perturb['motif_length']

                # Run perturbation
                perturbed_hic, perturb_outputs = run_perturbation(model, seq_region, ctcf_region, atac_region, other_regions, motif_pos, motif_len, args.ko_mode, device, all_tracks)
                if perturbed_hic is None: continue

                # Calculate the effect on the target loop
                diff_hic = pred_hic - perturbed_hic
                mean_loop_change = diff_hic[0, p_start1:p_end1, p_start2:p_end2].mean().item()
                
                result_data = motif_to_perturb.to_dict()
                result_data['peak_pos'] = peak_pos
                result_data['mean_hic_change'] = mean_loop_change
                all_perturbation_results.append(result_data)

        if not all_perturbation_results:
            print("No significant motif effects were found after perturbation. Exiting.")
            return
        

        # --- PHASE 2: Causal Ranking ---
        print("\n--- Phase 2: Ranking TFs by causal effect size ---")
        causal_summary_df = pd.DataFrame(all_perturbation_results).drop_duplicates(subset=['motif', 'pos'])
        causal_summary_df['abs_hic_change'] = causal_summary_df['mean_hic_change'].abs()
        # calculate correlation matrix between all motifs
        motif_mean_changes = causal_summary_df.groupby('motif')['mean_hic_change'].mean()
        motif_corr_matrix = np.zeros((len(motif_mean_changes), len(motif_mean_changes)))
        motif_names = motif_mean_changes.index.tolist()
        print(motif_names)
        for i, motif_i in tqdm(enumerate(motif_names), desc="Calculating motif correlation matrix"):
            for j, motif_j in enumerate(motif_names):
                if i <= j:
                    corr = causal_summary_df[causal_summary_df['motif'] == motif_i]['mean_hic_change'].corr(
                        causal_summary_df[causal_summary_df['motif'] == motif_j]['mean_hic_change']
                    )
                    motif_corr_matrix[i, j] = corr
                    motif_corr_matrix[j, i] = corr
        print("Motif correlation matrix:")
        print(motif_corr_matrix)
        sns.clustermap(motif_corr_matrix, xticklabels=motif_names, yticklabels=motif_names, cmap='vlag', center=0)
        plt.savefig(f"{args.out_dir}/motif_correlation_matrix.png", dpi=300)
        plt.close()

        # Find the most impactful instance for each unique motif name
        final_ranking_df = causal_summary_df.loc[causal_summary_df.groupby('motif')['abs_hic_change'].idxmax()]
        final_ranking_df = final_ranking_df.sort_values(by='abs_hic_change', ascending=False).reset_index(drop=True)
        
        final_ranking_df.to_csv(f"{args.out_dir}/causal_effects_summary.tsv", sep='\t', index=False)
        print("Top TFs ranked by causal effect (absolute Hi-C change):")
        print(final_ranking_df[['motif', 'peak_pos', 'pos', 'mean_hic_change', 'score_magnitude']].head())

        print(final_ranking_df[final_ranking_df['motif'] == 'PATZ1.H13CORE.1.P.C'])

        # --- PHASE 3: Visualization of Top Causal TFs ---
        print(f"\n--- Phase 3: Visualizing top {args.n_causal_viz} causal TFs ---")
        if args.snp_file:
            snps = pd.read_csv(args.snp_file, sep='\t', names=['chrom', 'start', 'name', 'ref', 'alt'])
            snps['start'] += 1 

        for _, row in final_ranking_df.head(args.n_causal_viz).iterrows():
            motif_name = row['motif']
            motif_pos = row['pos']
            motif_len = row['motif_length']
            peak_pos = row['peak_pos']
            
            print(f"Visualizing effect of {motif_name} at position {motif_pos} (from peak {peak_pos})...")

            # Regenerate the specific perturbation for plotting
            perturbed_hic, _ = run_perturbation(model, seq_region, ctcf_region, atac_region, other_regions, motif_pos, motif_len, args.ko_mode, device, all_tracks)
            diff_hic = pred_hic - perturbed_hic

            # Plot 1: Hi-C Difference Map
            diff_hic_cropped = diff_hic[:, plot_p_start:plot_p_end, plot_p_start:plot_p_end]
            vmax_diff = np.percentile(np.abs(diff_hic_cropped.detach().cpu().numpy()), 99.5)
            
            fig, ax = plt.subplots(figsize=(8, 7))
            im = ax.imshow(diff_hic_cropped[0].detach().cpu().numpy(), cmap='bwr', norm=plt.Normalize(vmin=-vmax_diff, vmax=vmax_diff))
            ax.set_title(f'Hi-C Change from Perturbing {motif_name} at {motif_pos}\nMean Loop Change: {row["mean_hic_change"]:.4f}')
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="Change in Hi-C score")
            
            insert_pixel = (motif_pos // args.resolution) - plot_p_start
            ax.axvline(x=insert_pixel, color='green', linestyle='--', label=f'Motif Pos ({motif_pos})')
            ax.axhline(y=insert_pixel, color='green', linestyle='--')
            rect = plt.Rectangle((p_start2 - plot_p_start, p_start1 - plot_p_start), p_end2 - p_start2, p_end1 - p_start1, linewidth=2, edgecolor='yellow', facecolor='none', linestyle='--', label='Target Locus')
            ax.add_patch(rect)
            ax.legend()
            plt.savefig(f"{args.out_dir}/hic_change_causal_{motif_name}_pos{motif_pos}.png", dpi=300, bbox_inches='tight')
            plt.close(fig)

            # Plot 2: Saliency Logo for the region
            window_radius = args.viz_bp
            seq_start_idx = max(0, peak_pos - window_radius)
            seq_end_idx = min(len(gradient_pwms), peak_pos + window_radius)
            
            locus_start_genomic = args.start + seq_start_idx
            locus_end_genomic = args.start + seq_end_idx
            locus_str = f"{args.chr_name}:{locus_start_genomic + 1}-{locus_end_genomic}"
            
            seq_slice = np.abs(gradient_pwms[seq_start_idx:seq_end_idx, :]) if args.abs else gradient_pwms[seq_start_idx:seq_end_idx, :]
            df_seq = pd.DataFrame(seq_slice, columns=['A', 'T', 'C', 'G', 'N'])[['A', 'C', 'G', 'T']]
            logo = logomaker.Logo(df_seq, figsize=(15, 3), color_scheme='classic')
            logo.ax.set_title(f'Saliency at Peak {peak_pos} ({locus_str}) -- Causal Motif: {motif_name}')
            
            highlight_start = motif_pos - seq_start_idx
            logo.ax.axvspan(highlight_start - 0.5, highlight_start + motif_len - 0.5, color='yellow', alpha=0.3, zorder=-1)
            logo.ax.text(highlight_start, logo.ax.get_ylim()[1], f'-> {motif_name}', color='red', fontsize=10)

            if args.snp_file:
                snps_in_region = snps[(snps['chrom'] == args.chr_name) & (snps['start'] >= locus_start_genomic) & (snps['start'] < locus_end_genomic)]
                for _, snp_row in snps_in_region.iterrows():
                    snp_plot_pos = snp_row['start'] - locus_start_genomic
                    logo.ax.axvline(snp_plot_pos, color='purple', linestyle=':', alpha=0.9)
                    logo.ax.text(snp_plot_pos, logo.ax.get_ylim()[1] * 0.9, f"{snp_row['name']}", color='purple', fontsize=8, rotation=90)
            
            plt.savefig(f"{args.out_dir}/saliency_logo_causal_{motif_name}_pos{motif_pos}.png", dpi=300, bbox_inches='tight')
            plt.close()


        # --- Final Summary Plots ---
        print("Generating final summary plots...")
        fig, ax = plt.subplots(figsize=(8,6))
        ax.scatter(final_ranking_df['mean_hic_change'], final_ranking_df['score_magnitude'])
        ax.set_ylabel('Motif Saliency Correlation Score')
        ax.set_xlabel('Mean Hi-C Change (Causal Effect)')
        ax.set_title('Saliency Correlation vs. Causal Effect of TFs')
        ax.axhline(0, color='gray', linestyle='--', alpha=0.7)
        ax.axvline(0, color='gray', linestyle='--', alpha=0.7)
        
        # Label top points by causal effect
        for i, row in final_ranking_df.head(15).iterrows():
            ax.text(row['mean_hic_change'], row['score_magnitude'], row['motif'], fontsize=8)
        
        plt.savefig(f"{args.out_dir}/summary_correlation_vs_causal_effect.png", dpi=300, bbox_inches='tight')
        plt.close(fig)

if __name__ == '__main__':
    main()