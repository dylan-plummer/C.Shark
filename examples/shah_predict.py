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
from matplotlib.colors import PowerNorm

# Import necessary functions from the C.Shark codebase
import cshark.inference.utils.inference_utils as infer
from cshark.inference.utils.model_utils import load_default, get_all_track_names

# Define constants from the codebase
WINDOW_SIZE = 2097152

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


def main():
    parser = argparse.ArgumentParser(description='C.Shark Gradient Ascent in Input Space.')
    # Add arguments similar to perturb.py for consistency
    parser.add_argument('--model', dest='model_path', required=True, help='Path to the model checkpoint.')
    parser.add_argument('--chr', dest='chr_name', required=True, help='Chromosome for prediction.')
    parser.add_argument('--start', dest='start', type=int, required=True, help=f'Starting point for the {WINDOW_SIZE}bp input window.')
    parser.add_argument('--seq', dest='seq_path', required=True, help='Path to the folder with sequence .fa.gz files.')
    parser.add_argument('--out-dir', dest='out_dir', required=True, help='Output path for animation frames')
    parser.add_argument('--contexts', dest='contexts', required=True, nargs='+')
    parser.add_argument('--context-names', dest='context_names', required=False, nargs='+')
    parser.add_argument('--frames', dest='frames', type=int, default=30)
    
    # Optional arguments for other epigenetic features
    parser.add_argument('--bigwigs', nargs='*', help='Paths to the bigwig files for genomic features, specified as key=value pairs (e.g., ctcf=path/to/ctcf.bw).', 
                        action=ParseKwargs)

    # Argument for the target locus
    parser.add_argument('--target-locus', dest='target_locus', required=True, 
                        help='The region in the Hi-C map to maximize, format: chr1:start1-end1_chr1:start2-end2')
    
    parser.add_argument('--resolution', dest='resolution', type=int, default=8192, help='Resolution of the Hi-C map used by the model.')
    parser.add_argument('--matrix-size', dest='mat_size', type=int, default=256, help='Matrix size used by the model.')
    parser.add_argument('--latent_size', dest='mid_hidden', type=int, default=256, help='Latent size of the model.')


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
    
    print("Loading model...")
    model = load_default(
        args.model_path, 
        num_genomic_features=len(input_tracks),
        conditioning_vec_size=3,
        mat_size=args.mat_size,
        mid_hidden=args.mid_hidden,
        seq_filter_size=15,
        recon_1d=True
    ).to(device)
    model.eval() # Set model to evaluation mode

    # --- 2. Prepare Input Tensor ---
    inputs = infer.preprocess_default(seq_region, ctcf_region, atac_region, other_regions)

    # --- 3. Forward Pass and Score Calculation ---

    init_context = np.array([float(c) for c in args.contexts[0].split(',')], dtype=np.float32)
    # interpolate context vectors over frames
    print(args.contexts)
    for c in args.contexts[1:]:
        next_context = np.array([float(v) for v in c.split(',')], dtype=np.float32)
        init_context = np.vstack((init_context, next_context))
    context_vectors = np.zeros((args.frames * len(args.contexts), init_context.shape[1]), dtype=np.float32)
    for i in range(init_context.shape[1]):
        context_vectors[:, i] = np.interp(
            np.linspace(0, init_context.shape[0]-1, args.frames * len(args.contexts)),
            np.arange(init_context.shape[0]),
            init_context[:, i]
        )

    # create labels if context names are provided
    titles = [''] * len(context_vectors)
    if args.context_names and len(args.context_names) == len(args.contexts):
        init_context_name = args.context_names[0]
        titles = [init_context_name] * len(context_vectors)
        for idx, name in enumerate(args.context_names[1:]):
            next_context_name = name
            interp_names = np.linspace(0, 1, args.frames)
            for f_idx in range(args.frames):
                alpha = interp_names[f_idx]
                titles[idx * args.frames + f_idx] = f"{init_context_name} -> {next_context_name} ({alpha:.2f})"
            init_context_name = next_context_name
        titles[ (len(args.context_names)-1) * args.frames : ] = [args.context_names[-1]] * args.frames
        

    for frame_idx, context_vec in enumerate(tqdm(context_vectors, desc="Generating frames")):
        context = torch.tensor(context_vec.reshape(1, -1), dtype=torch.float32).to(device)
        outputs = model(inputs, conditioning_vec=context)
        pred_hic = outputs.get('hic')
        pred_hic = (pred_hic + pred_hic.transpose(1,2)) / 2  # Make symmetric
        # undo log1p transformation
        pred_hic = torch.expm1(pred_hic)

        # plot predicted Hi-C map with context settings as sliders
        # pred_hic_np = pred_hic.detach().cpu().numpy()[0, :, :]
        # plt.imshow(pred_hic_np, cmap='Reds', norm=PowerNorm(gamma=0.5, vmin=0.5, vmax=10))
        # plt.colorbar()
        # plt.title(f'Predicted Hi-C Map - Frame {frame_idx+1}')
        # plt.savefig(os.path.join(args.out_dir, f'predicted_hic_map_frame_{frame_idx+1:03d}.png'), dpi=300)
        # plt.close()
        pred_hic_np = pred_hic.detach().cpu().numpy()[0, :, :]
        # Extract target locus coordinates
        locus1, locus2 = args.target_locus.split('_')
        chr1, coords1 = locus1.split(':')
        start1, end1 = map(int, coords1.split('-'))
        chr2, coords2 = locus2.split(':')
        start2, end2 = map(int, coords2.split('-'))
        # Convert genomic coordinates to matrix indices
        bin_size = args.resolution
        matrix_start1 = (start1 - args.start) // bin_size
        matrix_end1 = (end1 - args.start) // bin_size
        matrix_start2 = (start2 - args.start) // bin_size
        matrix_end2 = (end2 - args.start) // bin_size
        target_region = pred_hic_np[matrix_start1:matrix_end1, matrix_start2:matrix_end2]
        fig, axs = plt.subplots(1, 2, figsize=(10, 5))
        im0 = axs[0].imshow(target_region, cmap='afmhot_r', norm=PowerNorm(gamma=0.5, vmin=0.5, vmax=20))
        axs[0].set_title(f'Target Locus Hi-C Interaction - Frame {frame_idx+1}')
        plt.colorbar(im0, ax=axs[0], fraction=0.046, pad=0.04)
        # Show context vector as bar plot
        axs[1].bar(range(len(context_vec)), context_vec)
        axs[1].set_title('Context Vector')
        # separation,lifetime,rate
        axs[1].set_xticks(range(len(context_vec)))
        axs[1].set_xticklabels(['Separation', 'Lifetime', 'Rate'])
        axs[1].set_ylim(0, max(context_vectors.flatten()) * 1.1)
        # label bars with values
        for i, v in enumerate(context_vec):
            axs[1].text(i, v + 0.01, f"{v:.2f}", ha='center', va='bottom')
        fig.suptitle(titles[frame_idx], fontsize=16)
        plt.tight_layout()
        plt.savefig(os.path.join(args.out_dir, f'predicted_hic_map_frame_{frame_idx+1:03d}.png'), dpi=300)
        plt.close()

    # use ffmpeg to compile frames into a video
    video_path = os.path.join(args.out_dir, 'hic_evolution.mp4')
    os.system(f'ffmpeg -y -framerate 10 -i {os.path.join(args.out_dir, "predicted_hic_map_frame_%03d.png")} -c:v libx264 -pix_fmt yuv420p {video_path}')
    print(f'Video saved to {video_path}')

    


if __name__ == '__main__':
    main()