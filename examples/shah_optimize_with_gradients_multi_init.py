import os
import sys
import argparse
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from skimage.transform import resize
from scipy.ndimage import gaussian_filter
from tqdm import tqdm

# Import necessary functions from the C.Shark codebase
try:
    import cshark.inference.utils.inference_utils as infer
    from cshark.data.data_feature import HiCFeature
    from cshark.inference.utils.model_utils import load_default, get_all_track_names
except ImportError:
    print("C.Shark library not found. Please ensure it is installed.")
    sys.exit(1)

# --- CONSTANTS ---
WINDOW_SIZE = 2097152
CONTEXT_PARAM_NAMES = ['Separation', 'Lifetime', 'Rate']
# Bounds defined in Script 2, but widened slightly to match Script 1's flexibility
CONTEXT_BOUNDS = [(0.1, 8.0), (0.1, 8.0), (0.1, 8.0)] 

class ParseKwargs(argparse.Action):
    def __call__(self, parser, namespace, values, option_string=None):
        setattr(namespace, self.dest, dict())
        for value in values:
            try:
                key, value = value.split('=', 1)
                getattr(namespace, self.dest)[key] = value
            except ValueError:
                print(f"Skipping malformed argument: {value}")

def get_optimizer(context_tensor, scale, bias, lr):
    return torch.optim.Adam([context_tensor, scale, bias], lr=lr)

def run_optimization_epoch(model, optimizer, dataset, context_tensor, scale, bias, 
                           batch_size, device):
    
    perm_indices = np.random.permutation(len(dataset))
    epoch_loss = 0.0
    num_batches = 0
    
    for i in range(0, len(dataset), batch_size):
        batch_indices = perm_indices[i : i + batch_size]
        
        batch_inputs = []
        batch_targets = []
        batch_masks = []
        
        for idx in batch_indices:
            batch_inputs.append(dataset[idx]['input'])
            batch_targets.append(dataset[idx]['target'])
            batch_masks.append(dataset[idx]['mask'])
        
        input_batch_t = torch.cat(batch_inputs, dim=0).to(device)
        target_batch_t = torch.stack(batch_targets, dim=0).to(device)
        mask_batch_t = torch.stack(batch_masks, dim=0).to(device)
        
        current_bs = input_batch_t.size(0)
        
        optimizer.zero_grad()
        
        # Expand context for batch size: [B, 3]
        current_context = context_tensor.expand(current_bs, -1)
        
        outputs = model(input_batch_t, conditioning_vec=current_context)
        
        pred_hic = outputs.get('hic')
        if pred_hic.dim() == 4: 
            pred_hic = pred_hic.squeeze(1)
            
        pred_hic = (pred_hic + pred_hic.transpose(1,2)) / 2
        pred_hic = torch.expm1(pred_hic)
        
        # Apply learnable scale and bias if desired
        # pred_hic = pred_hic * scale + bias 

        # Loss Calculation
        loss = torch.mean(((target_batch_t - pred_hic) ** 2) * mask_batch_t) # MSE Alternative
        
        loss.backward()
        optimizer.step()
        
        # Clamp context parameters
        with torch.no_grad():
            for c_i in range(len(CONTEXT_BOUNDS)):
                context_tensor[0, c_i] = torch.clamp(
                    context_tensor[0, c_i], 
                    CONTEXT_BOUNDS[c_i][0], 
                    CONTEXT_BOUNDS[c_i][1]
                )

        epoch_loss += loss.item()
        num_batches += 1
        
    return epoch_loss / num_batches

def main():
    parser = argparse.ArgumentParser(description='C.Shark Robust Batch Optimization of Context.')
    
    # Standard Inputs
    parser.add_argument('--model', dest='model_path', required=True, help='Path to model checkpoint.')
    parser.add_argument('--chr', dest='chr_name', required=True, help='Chromosome.')
    parser.add_argument('--start', dest='start', type=int, required=True, help='Start coordinate.')
    parser.add_argument('--end', dest='end', type=int, required=True, help='End coordinate.')
    parser.add_argument('--step-size', dest='step_size', type=int, default=1000000, help='Step size.')
    parser.add_argument('--seq', dest='seq_path', required=True, help='Path to sequence folder.')
    parser.add_argument('--hic', dest='hic_path', required=True, help='Path to Hi-C file.')
    parser.add_argument('--out-dir', dest='out_dir', required=True, help='Output directory.')
    
    # Track Args
    parser.add_argument('--bigwigs', nargs='*', help='Bigwig paths key=value.', action=ParseKwargs, default={})
    
    # Model Config
    parser.add_argument('--resolution', dest='resolution', type=int, default=8192)
    parser.add_argument('--matrix-size', dest='mat_size', type=int, default=256)
    parser.add_argument('--latent_size', dest='mid_hidden', type=int, default=256)
    
    # Optimization Config
    parser.add_argument('--n-inits', type=int, default=5, help='Number of random initializations.')
    parser.add_argument('--n-iter', dest='n_iter', type=int, default=500, help='Iterations per init.')
    parser.add_argument('--batch-size', dest='batch_size', type=int, default=6, help='Batch size.')
    parser.add_argument('--lr', type=float, default=0.05, help='Learning rate.')
    parser.add_argument('--ignore-diag', dest='ignore_diag', type=int, default=0)
    parser.add_argument('--viz', action='store_true', help='Generate video for the best run.')

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

    # --- 2. Load Dataset (Pre-fetch) ---
    print(f"Loading data from {args.start} to {args.end} (step {args.step_size})...")
    coordinates = range(args.start, args.end, args.step_size)
    dataset = []
    
    hic_loader = HiCFeature(path=args.hic_path)
    gt_res = args.resolution
    if args.resolution == 8192: gt_res = 10000
    if args.resolution == 4096: gt_res = 5000

    for curr_start in tqdm(coordinates, desc="Loading Windows"):
        # Load Inputs
        seq_region, ctcf_region, atac_region, other_regions = infer.load_region(
            args.chr_name, curr_start, args.seq_path, 
            args.bigwigs.get('ctcf'), args.bigwigs.get('atac'), 
            other_paths=bigwig_paths[2:] if len(bigwig_paths) > 2 else None,
            window=WINDOW_SIZE
        )
        inp_tensor = infer.preprocess_default(seq_region, ctcf_region, atac_region, other_regions)
        
        # Load Targets
        mat = hic_loader.get(curr_start, window=WINDOW_SIZE, res=gt_res)
        mat = resize(mat, (int(args.mat_size), int(args.mat_size)), anti_aliasing=True)
        mat = np.float32(gaussian_filter(mat, sigma=0.5))
        
        mask = np.ones_like(mat)
        if args.ignore_diag > 0:
            for i in range(-args.ignore_diag, args.ignore_diag + 1):
                np.fill_diagonal(mask[max(0, i):, max(0, -i):], 0)
        
        dataset.append({
            'input': inp_tensor,
            'target': torch.tensor(mat, dtype=torch.float32),
            'mask': torch.tensor(mask, dtype=torch.float32),
            'start': curr_start
        })

    if not dataset:
        print("No data loaded.")
        sys.exit(1)
    
    print(f"Loaded {len(dataset)} windows.")

    # --- 3. Initialize Model ---
    model = load_default(
        args.model_path, 
        num_genomic_features=len(input_tracks),
        conditioning_vec_size=len(CONTEXT_PARAM_NAMES),
        mat_size=args.mat_size,
        mid_hidden=args.mid_hidden,
        seq_filter_size=15,
        recon_1d=True
    ).to(device)
    model.eval()
    for param in model.parameters():
        param.requires_grad = False

    # --- 4. Multi-Init Optimization ---
    init_results = []
    
    for init_idx in range(args.n_inits):
        print(f"\n--- Initialization {init_idx + 1} / {args.n_inits} ---")
        
        # Random Initialization
        init_vals = [np.random.uniform(low, high) for low, high in CONTEXT_BOUNDS]
        context_tensor = torch.tensor(init_vals, dtype=torch.float32).unsqueeze(0).to(device)
        context_tensor.requires_grad = True
        
        # Scale/Bias (Reset per init)
        scale_factor = torch.tensor([1.0], dtype=torch.float32).to(device)
        bias_term = torch.tensor([0.0], dtype=torch.float32).to(device)
        scale_factor.requires_grad = True
        bias_term.requires_grad = True
        
        optimizer = get_optimizer(context_tensor, scale_factor, bias_term, args.lr)
        
        # Optimization Loop
        final_loss = 0.0
        with tqdm(range(args.n_iter), desc=f"Optimizing Init {init_idx+1}", leave=False) as pbar:
            for epoch in pbar:
                avg_loss = run_optimization_epoch(
                    model, optimizer, dataset, context_tensor, scale_factor, bias_term,
                    args.batch_size, device,
                )
                final_loss = avg_loss
                pbar.set_postfix({'loss': f"{avg_loss:.4f}"})
        
        # Store Results
        res_dict = {
            'Init_ID': init_idx,
            'Final_Loss': final_loss,
            'Scale': scale_factor.item(),
            'Bias': bias_term.item()
        }
        final_ctx = context_tensor.detach().cpu().numpy().flatten()
        for i, name in enumerate(CONTEXT_PARAM_NAMES):
            res_dict[f'Initial_{name}'] = init_vals[i]
            res_dict[f'Final_{name}'] = final_ctx[i]
            
        init_results.append(res_dict)
        print(f"Init {init_idx+1} Finished. Loss: {final_loss:.4f}, Params: {final_ctx}")

    # --- 5. Analysis & Best Run Selection ---
    df = pd.DataFrame(init_results)
    df.to_csv(os.path.join(args.out_dir, 'optimization_summary.csv'), index=False)
    
    best_run = df.loc[df['Final_Loss'].idxmin()]
    print(f"\nBest Run: Init {int(best_run['Init_ID'])+1} with Loss {best_run['Final_Loss']:.5f}")
    print(f"Best Params: {best_run[[f'Final_{x}' for x in CONTEXT_PARAM_NAMES]].values}")

    # --- 6. Visualization (Re-run Best) ---
    if args.viz:
        print("\nRe-running best initialization for visualization video...")
        viz_out_dir = os.path.join(args.out_dir, 'best_run_viz')
        os.makedirs(viz_out_dir, exist_ok=True)
        
        # Setup Best Context
        best_init_vals = best_run[[f'Initial_{x}' for x in CONTEXT_PARAM_NAMES]].values.astype(float)
        context_tensor = torch.tensor(best_init_vals, dtype=torch.float32).unsqueeze(0).to(device)
        context_tensor.requires_grad = True
        
        scale_factor = torch.tensor([1.0], dtype=torch.float32).to(device) # Reset to re-learn path
        bias_term = torch.tensor([0.0], dtype=torch.float32).to(device)
        scale_factor.requires_grad = True
        bias_term.requires_grad = True
        
        optimizer = get_optimizer(context_tensor, scale_factor, bias_term, args.lr)
        
        # Track Stats
        losses = []
        viz_idx = 0 # Visualize the first window in the dataset
        viz_target = dataset[viz_idx]['target'].numpy()
        
        for epoch in tqdm(range(args.n_iter), desc="Viz Generation"):
            avg_loss = run_optimization_epoch(
                model, optimizer, dataset, context_tensor, scale_factor, bias_term,
                args.batch_size, device
            )
            losses.append(avg_loss)
            
            if (epoch + 1) % 10 == 0 or epoch == 0:
                # Generate Viz Frame
                with torch.no_grad():
                    viz_inp = dataset[viz_idx]['input'].to(device)
                    out = model(viz_inp, conditioning_vec=context_tensor)
                    pred = out.get('hic')
                    if pred.dim() == 4: pred = pred.squeeze(1)
                    pred = (pred + pred.transpose(1,2)) / 2
                    pred = torch.expm1(pred).squeeze().cpu().numpy()
                
                fig, axs = plt.subplots(2, 2, figsize=(10, 10))
                
                # Target
                axs[0][0].imshow(viz_target, cmap='Reds', vmin=0, vmax=np.max(pred))
                axs[0][0].set_title(f'Target (Win {dataset[viz_idx]["start"]})')
                
                # Pred
                im = axs[0][1].imshow(pred, cmap='Reds', vmin=0, vmax=np.max(pred))
                axs[0][1].set_title(f'Pred (Iter {epoch+1})')
                fig.colorbar(im, ax=axs[0,:], orientation='vertical', fraction=0.05)
                
                # Params
                curr_params = context_tensor.detach().cpu().numpy().flatten()
                axs[1][0].bar(CONTEXT_PARAM_NAMES, curr_params)
                axs[1][0].set_ylim(0, 8)
                for i, v in enumerate(curr_params):
                    axs[1][0].text(i, v + 0.1, f"{v:.2f}", ha='center')
                axs[1][0].set_title('Context Params')
                
                # Loss
                axs[1][1].plot(losses)
                axs[1][1].set_title('Avg Loss (All Windows)')
                
                plt.suptitle(f'Best Init Re-Run: Iter {epoch+1}')
                plt.savefig(os.path.join(viz_out_dir, f'frame_{(epoch+1)//10:03d}.png'))
                plt.close()

        # Compile Video
        video_path = os.path.join(args.out_dir, 'optimization_best_run.mp4')
        os.system(f"ffmpeg -y -framerate 10 -i {os.path.join(viz_out_dir, 'frame_%03d.png')} -c:v libx264 -pix_fmt yuv420p {video_path}")

        # Plot Parallel Coordinates of all runs
        plt.figure(figsize=(10, 6))
        pd.plotting.parallel_coordinates(df, 'Init_ID', cols=[f'Final_{x}' for x in CONTEXT_PARAM_NAMES] + ['Final_Loss'], colormap='viridis')
        plt.title('Optimization Trajectories (All Inits)')
        plt.savefig(os.path.join(args.out_dir, 'all_inits_parallel_coord.png'))
        plt.close()

        # plot boxplot of final params
        plt.figure(figsize=(8, 6))
        final_params = df[[f'Final_{x}' for x in CONTEXT_PARAM_NAMES]]
        final_params.boxplot()
        plt.title('Final Context Parameters Distribution')
        plt.savefig(os.path.join(args.out_dir, 'final_params_boxplot.png'))
        plt.close()

if __name__ == '__main__':
    main()