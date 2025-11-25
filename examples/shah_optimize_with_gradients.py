import os
import sys
import argparse
import numpy as np
import pandas as pd
import torch
import pyBigWig
import logomaker
import matplotlib.pyplot as plt
import scipy.sparse as sp
from skimage.transform import resize
from scipy.ndimage import gaussian_filter
from tqdm import tqdm
from matplotlib.colors import PowerNorm

# Import necessary functions from the C.Shark codebase
import cshark.inference.utils.inference_utils as infer
from cshark.data.data_feature import HiCFeature
from cshark.inference.utils.model_utils import load_default, get_all_track_names

# Define constants from the codebase
WINDOW_SIZE = 2097152

class ParseKwargs(argparse.Action):
    def __call__(self, parser, namespace, values, option_string=None):
        setattr(namespace, self.dest, dict())
        for value in values:
            key, value = value.split('=')
            getattr(namespace, self.dest)[key] = value

def main():
    parser = argparse.ArgumentParser(description='C.Shark Bayesian optimization of Context Parameters.')
    parser.add_argument('--model', dest='model_path', required=True, help='Path to the model checkpoint.')
    parser.add_argument('--chr', dest='chr_name', required=True, help='Chromosome for prediction.')
    
    # Modified range arguments
    parser.add_argument('--start', dest='start', type=int, required=True, help=f'Starting coordinate.')
    parser.add_argument('--end', dest='end', type=int, required=True, help=f'Ending coordinate.')
    parser.add_argument('--step-size', dest='step_size', type=int, default=1000000, help='Step size between windows (default: 1Mb).')
    
    parser.add_argument('--seq', dest='seq_path', required=True, help='Path to the folder with sequence .fa.gz files.')
    parser.add_argument('--out-dir', dest='out_dir', required=True, help='Output path for animation frames')
    parser.add_argument('--init-context', dest='init_context', required=True)
    parser.add_argument('--hic', dest='hic_path', required=True, help='Path to the Hi-C file')
    
    parser.add_argument('--bigwigs', nargs='*', help='Paths to the bigwig files for genomic features, specified as key=value pairs.', 
                        action=ParseKwargs)
    
    parser.add_argument('--resolution', dest='resolution', type=int, default=8192, help='Resolution of the Hi-C map used by the model.')
    parser.add_argument('--matrix-size', dest='mat_size', type=int, default=256, help='Matrix size used by the model.')
    parser.add_argument('--latent_size', dest='mid_hidden', type=int, default=256, help='Latent size of the model.')
    parser.add_argument('--n-iter', dest='n_iter', type=int, default=1000, help='Number of iterations for optimization.')
    parser.add_argument('--ignore-diag', dest='ignore_diag', type=int, default=0, help='Number of diagonals to ignore in Hi-C loss calculation.')
    parser.add_argument('--viz', dest='viz', action='store_true', help='Whether to generate visualization plots.')
    
    # New batch size argument
    parser.add_argument('--batch-size', dest='batch_size', type=int, default=6, help='Batch size for optimization.')

    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # --- 1. Setup Model Info ---
    all_tracks, _, input_tracks = get_all_track_names(args.model_path)
    
    bigwig_paths = []
    if args.bigwigs:
        for track_name in input_tracks:
            if track_name in args.bigwigs:
                bigwig_paths.append(args.bigwigs[track_name])

    # --- 2. Load All Data (Pre-fetch Batches) ---
    # We load all windows into memory to avoid repeated I/O during optimization
    print(f"Loading data from {args.start} to {args.end} with step {args.step_size}...")
    
    coordinates = range(args.start, args.end, args.step_size)
    dataset = []
    
    hic_loader = HiCFeature(path=args.hic_path)
    gt_res = args.resolution
    if args.resolution == 8192:
        gt_res = 10000
    if args.resolution == 4096:
        gt_res = 5000

    for curr_start in tqdm(coordinates, desc="Loading Windows"):
        # Load Input Features
        seq_region, ctcf_region, atac_region, other_regions = infer.load_region(
            args.chr_name, curr_start, args.seq_path, 
            args.bigwigs.get('ctcf'), args.bigwigs.get('atac'), 
            other_paths=bigwig_paths[2:] if len(bigwig_paths) > 2 else None,
            window=WINDOW_SIZE
        )
        
        # Preprocess Input (Usually returns [1, C, L])
        inp_tensor = infer.preprocess_default(seq_region, ctcf_region, atac_region, other_regions)
        
        # Load Target Hi-C
        mat = hic_loader.get(curr_start, window=WINDOW_SIZE, res=gt_res)
        mat = resize(mat, (int(args.mat_size), int(args.mat_size)), anti_aliasing=True)
        mat = np.float32(mat)
        
        # Apply Gaussian filter and masking prep
        mat = gaussian_filter(mat, sigma=0.5)
        
        mask = np.ones_like(mat)
        if args.ignore_diag > 0:
            for i in range(-args.ignore_diag, args.ignore_diag + 1):
                np.fill_diagonal(mask[max(0, i):, max(0, -i):], 0)
        
        # Store as tensors
        target_tensor = torch.tensor(mat, dtype=torch.float32)
        mask_tensor = torch.tensor(mask, dtype=torch.float32)
        
        dataset.append({
            'input': inp_tensor, # [1, C, L]
            'target': target_tensor, # [H, W]
            'mask': mask_tensor, # [H, W]
            'start': curr_start
        })

    if len(dataset) == 0:
        print("No windows found in the specified range.")
        sys.exit(1)

    print(f"Loaded {len(dataset)} windows.")

    # --- 3. Initialize Model ---
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
    model.eval() # Eval mode for weights, but we optimize inputs
    for param in model.parameters():
        param.requires_grad = False

    # --- 4. Initialize Optimization Parameters ---
    context_params = [float(x) for x in args.init_context.split(',')]
    context_tensor = torch.tensor(context_params, dtype=torch.float32).unsqueeze(0).to(device) # [1, 3]
    scale_factor = torch.tensor([1.0], dtype=torch.float32).to(device)
    bias_term = torch.tensor([0.0], dtype=torch.float32).to(device)

    context_tensor.requires_grad = True
    scale_factor.requires_grad = True
    bias_term.requires_grad = True

    optimizer = torch.optim.Adam([context_tensor, scale_factor, bias_term], lr=0.05)
    context_bounds = [(0.1, 8.0), (0.1, 8.0), (0.1, 8.0)]

    # --- 5. Optimization Loop ---
    epoch_errors = []
    
    # For visualization, we pick the first window to track consistency
    viz_idx = 0
    viz_target_np = dataset[viz_idx]['target'].numpy()

    print(viz_target_np.min(), viz_target_np.max())
    
    print("Starting optimization...")
    for epoch in range(args.n_iter):
        
        # Shuffle dataset for batch training
        perm_indices = np.random.permutation(len(dataset))
        epoch_loss = 0.0
        num_batches = 0
        
        for i in range(0, len(dataset), args.batch_size):
            batch_indices = perm_indices[i : i + args.batch_size]
            
            # Collate batch
            batch_inputs = []
            batch_targets = []
            batch_masks = []
            
            for idx in batch_indices:
                batch_inputs.append(dataset[idx]['input'])
                batch_targets.append(dataset[idx]['target'])
                batch_masks.append(dataset[idx]['mask'])
            
            # Stack into tensors: Inputs [B, C, L], Targets [B, H, W]
            input_batch_t = torch.cat(batch_inputs, dim=0).to(device)
            target_batch_t = torch.stack(batch_targets, dim=0).to(device)
            mask_batch_t = torch.stack(batch_masks, dim=0).to(device)
            
            current_bs = input_batch_t.size(0)
            
            optimizer.zero_grad()
            
            # Expand context for batch size: [B, 3]
            current_context = context_tensor.expand(current_bs, -1)
            
            outputs = model(input_batch_t, conditioning_vec=current_context)
            
            pred_hic = outputs.get('hic') # [B, H, W] or [B, 1, H, W] depending on model
            if pred_hic.dim() == 4: 
                pred_hic = pred_hic.squeeze(1)
                
            pred_hic = (pred_hic + pred_hic.transpose(1,2)) / 2  # Make symmetric
            pred_hic = torch.expm1(pred_hic)
            # clip all values to 0-5
            # pred_hic = torch.clamp(pred_hic, 0, 5)
            # target_batch_t = torch.clamp(target_batch_t, 0, 5)
            
            # Apply scale and bias
            #pred_hic = pred_hic * scale_factor + bias_term
            
            # Compute Loss
            # Using MAE (L1) as per original script
            #loss = torch.mean(torch.abs(target_batch_t - pred_hic * mask_batch_t))
            # Using MSE loss
            loss = torch.mean(((target_batch_t - pred_hic) ** 2) * mask_batch_t)
            
            loss.backward()
            optimizer.step()
            
            # Clamp
            with torch.no_grad():
                for c_i in range(len(context_bounds)):
                    context_tensor[0, c_i] = torch.clamp(context_tensor[0, c_i], context_bounds[c_i][0], context_bounds[c_i][1])

            epoch_loss += loss.item()
            num_batches += 1

        avg_epoch_loss = epoch_loss / num_batches
        epoch_errors.append(avg_epoch_loss)

        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"Iteration {epoch+1}/{args.n_iter}, Avg Loss: {avg_epoch_loss:.4f}, Context: {context_tensor.detach().cpu().numpy()}")
            
            if args.viz:
                # Generate prediction for the specific visualization window (index 0)
                with torch.no_grad():
                    viz_input = dataset[viz_idx]['input'].to(device) # [1, C, L]
                    viz_out = model(viz_input, conditioning_vec=context_tensor)
                    viz_pred = viz_out.get('hic')
                    if viz_pred.dim() == 4: viz_pred = viz_pred.squeeze(1)
                    viz_pred = (viz_pred + viz_pred.transpose(1,2)) / 2
                    viz_pred = torch.expm1(viz_pred)
                    #viz_pred = viz_pred * scale_factor + bias_term
                    viz_pred_np = viz_pred.squeeze().cpu().numpy()

                fig, axs = plt.subplots(2,2, figsize=(10, 10))
                
                # Plot 1: Target (Fixed window)
                im0 = axs[0][0].imshow(viz_target_np, cmap='Reds')
                axs[0][0].set_title(f'Target Hi-C\n{args.chr_name}:{dataset[viz_idx]["start"]}')
                
                # Plot 2: Prediction
                im1 = axs[0][1].imshow(viz_pred_np, cmap='Reds')
                axs[0][1].set_title(f'Prediction (Iter {epoch+1})')
                fig.colorbar(im1, ax=axs[:2], orientation='vertical', fraction=.1)
                
                # Plot 3: Context Params
                param_names = ['Separation', 'Lifetime', 'Rate']
                axs[1][0].bar(param_names, context_tensor.detach().cpu().numpy().squeeze())
                # annotate values
                for i, v in enumerate(context_tensor.detach().cpu().numpy().squeeze()):
                    axs[1][0].text(i, v + 0.2, f"{v:.2f}", ha='center')
                axs[1][0].set_ylabel('Parameter Value')
                axs[1][0].set_ylim(0, 8)
                axs[1][0].set_title('Context Parameters')
                
                # Plot 4: Loss Curve
                axs[1][1].plot(epoch_errors, marker='o')
                axs[1][1].set_xlabel('Iteration')
                axs[1][1].set_ylabel('Avg MAE Loss')
                axs[1][1].set_title('Loss over Iterations')
                
                plt.suptitle(f'Iteration {epoch+1}\nAvg Batch Loss: {avg_epoch_loss:.4f}')
                plt.savefig(os.path.join(args.out_dir, f'iteration_{(epoch+1) // 10:03d}.png'))
                plt.close()

    # --- 6. Final Evaluation ---
    # Evaluate average MSE across all windows
    total_mse = 0
    count = 0
    print("Running final evaluation...")
    with torch.no_grad():
        for item in dataset:
            inp = item['input'].to(device)
            tgt = item['target'].cpu().numpy()
            
            out = model(inp, conditioning_vec=context_tensor)
            pred = out.get('hic')
            if pred.dim() == 4: pred = pred.squeeze(1)
            pred = (pred + pred.transpose(1,2)) / 2
            pred = torch.expm1(pred)
            pred = pred.squeeze().cpu().numpy()
            
            total_mse += np.mean((tgt - pred) ** 2)
            count += 1
            
    final_avg_mse = total_mse / count
    print(f"Final Average MSE Score across {count} windows: {final_avg_mse}")

    if args.viz:
        # Save final comparison for the visualization window
        with torch.no_grad():
            viz_input = dataset[viz_idx]['input'].to(device)
            viz_out = model(viz_input, conditioning_vec=context_tensor)
            viz_pred = viz_out.get('hic')
            if viz_pred.dim() == 4: viz_pred = viz_pred.squeeze(1)
            viz_pred = (viz_pred + viz_pred.transpose(1,2)) / 2
            viz_pred = torch.expm1(viz_pred)
            viz_pred_np = viz_pred.squeeze().cpu().numpy()

        fig, axs = plt.subplots(1,2, figsize=(10,5))
        im0 = axs[0].imshow(viz_target_np, cmap='Reds')
        axs[0].set_title('Target Hi-C')
        im1 = axs[1].imshow(viz_pred_np, cmap='Reds')
        axs[1].set_title('Final Prediction')
        fig.colorbar(im1, ax=axs, orientation='vertical', fraction=.1)
        plt.suptitle(f'Final Prediction (Viz Window)\nGlobal Avg MSE: {final_avg_mse:.4f}')
        plt.savefig(os.path.join(args.out_dir, 'final_prediction.png'))
        plt.close()

        os.system(f"ffmpeg -y -framerate 10 -i {os.path.join(args.out_dir, 'iteration_%03d.png')} -c:v libx264 -pix_fmt yuv420p {os.path.join(args.out_dir, 'optimization_process.mp4')}")

if __name__ == '__main__':
    main()