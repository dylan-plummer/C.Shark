import os
# use CPU
# os.environ["CUDA_VISIBLE_DEVICES"] = ""
import argparse
import numpy as np
import pandas as pd
from requests import get
import torch
import torch.nn.functional as F
import lightning as pl
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
import cooler
import pyBigWig
import networkx as nx
from tqdm import tqdm
from scipy import ndimage
from skimage.transform import resize

import cshark.model.corigami_models as corigami_models
import cshark.inference.utils.inference_utils as infer
from cshark.inference.utils.model_utils import get_all_track_names
from cshark.inference.utils.inference_utils import write_tmp_cooler, knockout_peaks, get_axis_range_from_bigwig, chunk_shuffle

from enformer_pytorch import from_pretrained
from enformer_pytorch.modeling_enformer import poisson_loss

font_size = 15
plot_width = 17
track_label_fraction = 0.13
track_height_1d = 2
ENFORMER_CONTEXT_LENGTH = 196_608
ENFORMER_TARGET_LEN = 896 * 128  # 114,688 bp
ENFORMER_TRIM = (ENFORMER_CONTEXT_LENGTH - ENFORMER_TARGET_LEN) // 2  # 40,960 bp to skip on left/right


def write_full_cooler(pred_pixels, chr_name, start, res=8192, window=2097152, out_file='tmp/tmp.cool'):
    bins = pd.DataFrame()
    #bin_range = np.linspace(start, start + window - res, pred.shape[0])
    bin_range = np.arange(0, start + window + res, res)
    bins['start'] = bin_range
    bins['start'] = bins['start'].astype(int)
    bins['end'] = bins['start'] + res
    bins['end'] = bins['end'].astype(int)
    bins['chrom'] = chr_name
    pred_pixels.to_csv(out_file + '.csv')

    cooler.create_cooler(out_file, bins, pred_pixels, dtypes={'count': np.float32})
    os.remove(out_file + '.csv')

def write_bigwig(df, chrom, out_file, chrom_len, val_col, bin_size=64):
    """Writes a dataframe to a BigWig file."""
    # Ensure sorted
    df = df.sort_values(['start']).reset_index(drop=True)
    # pyBigWig cannot handle overlaps, but our aggregation step ensures unique bins.
    # It also requires headers.
    bw = pyBigWig.open(out_file, "w")
    bw.addHeader([(chrom, chrom_len)])
    
    starts = df['start'].astype(int).tolist()
    ends = df['end'].astype(int).tolist()
    values = df[val_col].astype(float).tolist()

    print(f"Writing BigWig with {len(starts)} entries...")
    
    bw.addEntries([chrom]*len(starts), 
                  starts, 
                  ends=ends, values=values)
    bw.close()
    print(f"Saved BigWig: {out_file}")

def visualize_force_directed_structure(wt_matrix, ko_matrix, title_suffix, out_path, threshold_percentile=98):
    """
    Generates a force-directed graph layout for WT and KO matrices.
    Uses connected component analysis to aggregate pixel clusters into single edges.
    """
    
    def get_aggregated_interactions(matrix, p_thresh):
        """
        Identifies clusters of high-intensity pixels and aggregates them into single interactions.
        Returns a list of tuples: (bin1, bin2, total_weight)
        """
        # Focus on upper triangle, excluding diagonal and immediate neighbors (k=2)
        # to distinguish loops from the polymer backbone.
        m_upper = np.triu(matrix, k=2)
        
        # 1. Thresholding
        data = m_upper[m_upper > 0]
        if len(data) == 0:
            return []
        
        cutoff = np.percentile(data, p_thresh)
        binary_mask = m_upper >= cutoff
        
        # 2. Connected Component Labeling (find blobs)
        # structural element defaults to connectivity 1 (cross) usually sufficient
        labeled_array, num_features = ndimage.label(binary_mask)
        
        if num_features == 0:
            return []
            
        interactions = []
        indices = np.arange(1, num_features + 1)
        
        # 3. Calculate Centroids and Total Intensity
        # center_of_mass returns (row, col) floats
        centroids = ndimage.center_of_mass(m_upper, labeled_array, indices)
        total_weights = ndimage.sum(m_upper, labeled_array, indices)
        
        # Handle case where only 1 feature exists (center_of_mass returns flat tuple)
        if num_features == 1:
            centroids = [centroids]
            total_weights = [total_weights]

        for (r, c), w in zip(centroids, total_weights):
            u, v = int(round(r)), int(round(c))
            if u != v: 
                interactions.append((u, v, w))
                
        return interactions

    def create_graph(matrix, p_thresh):
        G = nx.Graph()
        n_bins = matrix.shape[0]
        
        # 1. Add Nodes
        G.add_nodes_from(range(n_bins))
        
        # 2. Identify Loops via Aggregation
        loops = get_aggregated_interactions(matrix, p_thresh)
        
        # 3. Determine Backbone Weight
        # To maintain structure, the backbone (polymer chain) must be stronger than
        # the loops, but not so strong that loops can't bend it.
        # We set backbone weight slightly higher than the strongest loop found.
        if loops:
            max_loop = max([w for _,_,w in loops])
            backbone_weight = max_loop * 1.5
        else:
            backbone_weight = 1.0 # arbitrary fallback
            
        # 4. Add Backbone Edges
        for i in range(n_bins - 1):
            G.add_edge(i, i + 1, weight=backbone_weight, type='backbone')

        # 5. Add Loop Edges
        for u, v, w in loops:
            G.add_edge(u, v, weight=w, type='loop')
        
        return G

    print(f"Generating force-directed layouts (Aggregating clusters > {threshold_percentile}th percentile)...")
    
    G_wt = create_graph(wt_matrix, threshold_percentile)
    G_ko = create_graph(ko_matrix, threshold_percentile)
    
    # Compute Layout
    # k=None (default 1/sqrt(n)) often works well, but we can tune it.
    pos_wt = nx.kamada_kawai_layout(G_wt, weight='weight')
    pos_ko = nx.kamada_kawai_layout(G_ko, 
                                    pos=pos_wt,  # initialize with WT layout for consistency
                                    weight='weight')

    # Plotting
    fig, axs = plt.subplots(1, 2, figsize=(20, 10))
    cm = plt.cm.get_cmap('Spectral') 
    
    def draw_chromatin(G, pos, ax, title):
        # Separate edges
        backbone_edges = [(u, v) for u, v, d in G.edges(data=True) if d.get('type') == 'backbone']
        loop_edges = [(u, v) for u, v, d in G.edges(data=True) if d.get('type') == 'loop']
        
        # Draw Loops
        if loop_edges:
            weights = [G[u][v]['weight'] for u,v in loop_edges]
            max_w = max(weights) if weights else 1
            # Scale linewidth by weight
            widths = [1.0 + 3.0 * (w / max_w) for w in weights]
            
            lines = [[pos[u], pos[v]] for u, v in loop_edges]
            lc = LineCollection(lines, colors='crimson', linewidths=widths, alpha=0.6)
            ax.add_collection(lc)

        # Draw Backbone
        lines_bb = [[pos[u], pos[v]] for u, v in backbone_edges]
        lc_bb = LineCollection(lines_bb, colors='gray', linewidths=2.0, alpha=0.4)
        ax.add_collection(lc_bb)

        # Draw Nodes
        node_indices = list(G.nodes())
        coords = np.array([pos[n] for n in node_indices])
        sc = ax.scatter(coords[:, 0], coords[:, 1], c=node_indices, cmap=cm, s=75, zorder=10, edgecolor='black', linewidth=0.5)
        # draw text labels for every 10th node
        for i, n in enumerate(node_indices):
            if n % 10 == 0:
                ax.text(pos[n][0], pos[n][1], str(n // 10), fontsize=8, ha='center', va='center', zorder=15)
        ax.set_title(title)
        ax.axis('off')
        return sc

    sc1 = draw_chromatin(G_wt, pos_wt, axs[0], "WT Predicted Structure")
    sc2 = draw_chromatin(G_ko, pos_ko, axs[1], "KO Predicted Structure")
    
    cbar = plt.colorbar(sc1, ax=axs.ravel().tolist(), shrink=0.6, orientation='horizontal', pad=0.05)
    cbar.set_label(f'Genomic Position (Bins) - {title_suffix}')
    
    plt.suptitle(f"Force-Directed 2D Chromatin Projection (Aggregated Loops)\nLocus: {title_suffix}")
    plt.savefig(out_path)
    plt.close()
    print(f"Structure visualization saved to {out_path}")

    # Generate animation of the transition from WT to KO
    print("Generating transition animation (looping WT <-> KO)...")
    
    # Animation parameters
    hold_frames = 15  # Frames to hold at start and end
    transition_frames = 60  # Frames for the actual transition
    cycle_frames = hold_frames * 2 + transition_frames * 2  # Full WT->KO->WT cycle
    num_cycles = 1  # Number of complete cycles
    total_frames = cycle_frames * num_cycles
    fps = 20
    
    def ease_in_out_cubic(t):
        """Smooth easing function for natural-looking transitions."""
        if t < 0.5:
            return 4 * t * t * t
        else:
            return 1 - pow(-2 * t + 2, 3) / 2
    
    fig, ax = plt.subplots(figsize=(12, 11))

    G_merged = nx.compose(G_wt, G_ko)
    
    def update(frame):
        ax.clear()
        
        # Determine position within current cycle
        cycle_frame = frame % cycle_frames
        
        # Determine interpolation progress
        if cycle_frame < hold_frames:
            # Hold at WT state
            alpha = 0.0
            state_text = "Wild Type (WT)"
            progress = 0.0
        elif cycle_frame < hold_frames + transition_frames:
            # Transition from WT to KO
            raw_progress = (cycle_frame - hold_frames) / transition_frames
            alpha = ease_in_out_cubic(raw_progress)
            progress = raw_progress * 100
            state_text = f"WT → KO ({progress:.0f}%)"
        elif cycle_frame < hold_frames + transition_frames + hold_frames:
            # Hold at KO state
            alpha = 1.0
            state_text = "CTCF Knockout (KO)"
            progress = 100.0
        else:
            # Transition from KO back to WT
            raw_progress = (cycle_frame - hold_frames - transition_frames - hold_frames) / transition_frames
            alpha = 1.0 - ease_in_out_cubic(raw_progress)
            progress = 100.0 - raw_progress * 100
            state_text = f"KO → WT ({100 - progress:.0f}%)"
        
        

        # Interpolate positions
        interp_pos = {
            n: (1 - alpha) * np.array(pos_wt[n]) + alpha * np.array(pos_ko[n]) 
            for n in pos_wt
        }
        
        # Draw chromatin structure
        draw_chromatin(G_merged, interp_pos, ax, state_text)
        
        # Add progress bar
        progress_height = 0.02
        progress_y = -0.15
        ax.add_patch(plt.Rectangle((-0.5, progress_y), 1.0, progress_height, 
                                   fill=True, color='lightgray', 
                                   transform=ax.transAxes, zorder=5))
        ax.add_patch(plt.Rectangle((-0.5, progress_y), progress / 100, progress_height,
                                   fill=True, color='crimson',
                                   transform=ax.transAxes, zorder=6))
        
        # Add state labels
        ax.text(-0.48, progress_y - 0.03, 'WT', transform=ax.transAxes,
               fontsize=10, ha='left', weight='bold' if alpha < 0.5 else 'normal')
        ax.text(0.48, progress_y - 0.03, 'KO', transform=ax.transAxes,
               fontsize=10, ha='right', weight='bold' if alpha >= 0.5 else 'normal')
        
        ax.set_title(f"{state_text}\n{title_suffix}", fontsize=14, pad=20)
    
    import matplotlib.animation as animation
    
    ani = animation.FuncAnimation(fig, update, frames=total_frames, repeat=True)
    
    # Save with optimized settings
    gif_path = out_path.replace('.png', '_transition.gif')
    ani.save(gif_path, writer='pillow', fps=fps, dpi=100)
    
    # Also save as MP4 if possible (better quality, smaller file)
    try:
        mp4_path = out_path.replace('.png', '_transition.mp4')
        ani.save(mp4_path, writer='ffmpeg', fps=fps, dpi=150, 
                bitrate=2000, extra_args=['-vcodec', 'libx264'])
        print(f"MP4 animation saved to {mp4_path}")
    except Exception as e:
        print(f"Could not save MP4 (ffmpeg not available): {e}")
    
    plt.close()
    print(f"GIF animation saved to {gif_path}")

class TrainModule(pl.LightningModule):
    
    def __init__(self, args):
        super().__init__()
        self.save_hyperparameters(args)
        self.predict_1d = self.hparams.output_features is not None
        self.model = self.get_model(args)
        self.args = args
        self.criterion = torch.nn.MSELoss()
        self.window = 2097152 # 2Mb window size
        self.hierachy_in = [('ctcf', 'atac'), ('ctcf', 'atac', 'rad21', 'h3k27ac', 'h3k4me3', 'h3k36me3', 'h3k27me3')]
        self.hierarchy_out = [('rad21')]

        model_name =  args.model_type
        ModelClass = getattr(corigami_models, model_name)
        self.input_pred_model = ModelClass(
            num_genomic_features=7, # Input features
            num_target_tracks=1,    # Target 1D tracks
            conditioning_vec_size=len(self.hparams.conditioning_vec[0].split(',')) if self.hparams.conditioning_vec is not None else None,
            mid_hidden=self.hparams.model_latent_dim,
            predict_hic=False,
            diploid=args.dataset_assembly2 is not None,
            predict_1d=True,
            target_mat_size=args.mat_size,
            target_1d_length=args.target_1d_size,
            recon_1d=args.recon_1d,
            seq_filter_size=args.seq_filter_size,
            activation_1d=None
        )
        self.target_map ={'ctcf': 'CTCF:H1-hESC', 
                          'atac': 'DNASE:H1-hESC', 
                          'rad21': 'CHIP:RAD21:H1-hESC',
                          'h3k27ac': 'CHIP:H3K27ac:H1-hESC',
                          'h3k4me3': 'CHIP:H3K4me3:H1-hESC',
                          'h3k9me3': 'CHIP:H3K9me3:H1-hESC',
                          'h3k36me3': 'CHIP:H3K36me3:H1-hESC',
                          'h3k27me3': 'CHIP:H3K27me3:H1-hESC'}
        self.enformer = self.get_hESC_wrapper(target_tracks=['ctcf', 'atac', 'h3k27ac', 'h3k4me3', 'h3k9me3', 'h3k36me3', 'h3k27me3'])

    def enformer_predict_1d(self, inputs):
        """
        Use enformer to predict 1D input tracks using a sliding window approach.
        Maps each Enformer bin (896 total) to its corresponding 128bp region.
        """
        DOWNSAMPLE_FACTOR = 128
        ENFORMER_OUTPUT_BINS = 896
        
        pred_1d_inputs = torch.zeros((inputs.shape[0], inputs.shape[1], len(self.hparams.input_features) - 1), device=inputs.device)
        pred_1d_counts = torch.zeros((inputs.shape[0], inputs.shape[1], len(self.hparams.input_features) - 1), device=inputs.device)
        
        seq_length = inputs.shape[1]
        step_size = ENFORMER_CONTEXT_LENGTH // 2  # 50% overlap
        
        total_poisson_loss = 0.0
        num_windows = 0
        
        for start in range(-ENFORMER_TRIM, seq_length, step_size):
            end = start + ENFORMER_CONTEXT_LENGTH
            
            if start + ENFORMER_TRIM >= seq_length:
                break
            
            # Extract input (handle negative start for left-edge coverage)
            input_start = max(start, 0)
            input_seq = inputs[:, input_start:end, :4]
            
            # Pad left if window extends before sequence start
            if start < 0:
                left_pad = -start
                input_seq = F.pad(input_seq, (0, 0, left_pad, 0), "constant", 0.25)
            
            # Pad right if needed
            if input_seq.shape[1] < ENFORMER_CONTEXT_LENGTH:
                pad_size = ENFORMER_CONTEXT_LENGTH - input_seq.shape[1]
                input_seq = F.pad(input_seq, (0, 0, 0, pad_size), "constant", 0.25)
            
            # Reorder: ATCG -> ACGT
            input_seq = input_seq[:, :, [0, 2, 3, 1]]
            
            # Get predictions (batch, 896, num_tracks)
            outputs = self.enformer(input_seq.float())
            
            # Calculate global output coordinates
            global_out_start = start + ENFORMER_TRIM
            
            # For each of the 896 bins, map to 128bp region
            for bin_idx in range(ENFORMER_OUTPUT_BINS):
                bin_start = global_out_start + bin_idx * DOWNSAMPLE_FACTOR
                bin_end = bin_start + DOWNSAMPLE_FACTOR
                
                # Clip to sequence bounds
                if bin_start >= seq_length:
                    break
                bin_end = min(bin_end, seq_length)
                
                # Broadcast this bin's prediction across its 128bp region
                pred_1d_inputs[:, bin_start:bin_end, :] += outputs[:, bin_idx:bin_idx+1, :].expand(-1, bin_end - bin_start, -1)
                pred_1d_counts[:, bin_start:bin_end, :] += 1.0
        
        # Average overlapping predictions
        pred_1d_inputs = pred_1d_inputs / pred_1d_counts.clamp(min=1.0)
        
        return pred_1d_inputs

    @staticmethod
    def get_target_indices(species: str, target: str) -> np.ndarray:
        """Fetches and returns the numerical indices for a given target description."""
        targets_file = f"https://raw.githubusercontent.com/calico/basenji/master/manuscripts/cross2020/targets_{species}.txt"
        targets_df = pd.read_csv(targets_file, sep='\t')
        target_mask = targets_df['description'].str.contains(target, case=False)
        target_indices = targets_df[target_mask]['index'].values
        if len(target_indices) == 0:
            raise ValueError(f"No tracks found for target '{target}' in species '{species}'.")
        return target_indices
    
    @staticmethod
    def set_module_requires_grad_(module, requires_grad):
        for param in module.parameters():
            param.requires_grad = requires_grad

    def freeze_all_layers_(self,module):
        self.set_module_requires_grad_(module, False)
    
    def freeze_all_but_last_n_layers_(self, enformer, n):
        self.freeze_all_layers_(enformer)

        transformer_blocks = enformer.transformer

        for module in transformer_blocks[-n:]:
            self.set_module_requires_grad_(module, True)

    def get_hESC_wrapper(self, target_tracks=['ctcf', 'atac']):
        # Load the pre-trained model
        enformer = from_pretrained('EleutherAI/enformer-official-rough', 
                           use_tf_gamma=True)
        self.freeze_all_but_last_n_layers_(enformer, n=1)  # Fine-tune last 1 transformer block
        # 1. Get Indices for specific tracks
        hesc_indices = []
        for track in target_tracks:
            target_desc = self.target_map[track]
            indices = self.get_target_indices('human', target_desc)
            hesc_indices.append(indices[0])
        print(f'Initialized with hESC track indices: {hesc_indices} for {target_tracks}')

        # 2. Define the Adapter Class
        class HESCHeadAdapterWrapper(torch.nn.Module):
            def __init__(self, enformer, hesc_indices, species='human', load_pretrained=False):
                super().__init__()
                self.enformer = enformer
                self.hesc_indices = hesc_indices
                self.species = species
                self.load_pretrained = load_pretrained
                
                # Enformer trunk output is dim * 2 (e.g., 1536 * 2 = 3072)
                embedding_dim = enformer.dim * 2
                
                # Create separate linear heads for each track we want to fine-tune
                self.to_tracks = torch.nn.ModuleList([
                    torch.nn.Linear(in_features=embedding_dim, out_features=1) 
                    for _ in hesc_indices
                ])

                # learnable scale and bias for each track
                self.scale = torch.nn.Parameter(torch.ones(len(hesc_indices)))
                self.bias = torch.nn.Parameter(torch.zeros(len(hesc_indices)))
                
                if self.load_pretrained:
                    # Access the original pre-trained human head
                    # _heads['human'] is a Sequential(Linear, Softplus)
                    original_human_linear = enformer._heads[self.species][0]
                    
                    # Copy weights/biases for the specific indices we care about
                    with torch.no_grad():
                        for i, original_idx in enumerate(hesc_indices):
                            # Copy the specific row from the weight matrix
                            # Original shape: [5313, 3072] -> Extract row [3072] -> Unsqueeze to [1, 3072]
                            self.to_tracks[i].weight.data = original_human_linear.weight.data[original_idx].unsqueeze(0).clone()
                            
                            # Copy the specific bias
                            self.to_tracks[i].bias.data = original_human_linear.bias.data[original_idx].unsqueeze(0).clone()

                # Enformer uses Softplus activation for the final output
                self.activation = torch.nn.Softplus()
                

            def forward(self, x):
                # Efficiently get embeddings without computing all 5313 original heads
                # embeddings shape: (batch, seq_len, 3072)
                embeddings = self.enformer(x, return_only_embeddings=True)
                
                track_preds = []
                for track_i, linear_layer in enumerate(self.to_tracks):
                    # Project embeddings: (batch, seq_len, 1)
                    track_output = linear_layer(embeddings)
                    # Apply learnable scale and bias
                    track_output = track_output * self.scale[track_i] + self.bias[track_i]
                    track_preds.append(track_output)
                
                # Concatenate to (batch, seq_len, num_selected_tracks)
                track_preds = torch.cat(track_preds, dim=-1)

                
                # Apply Softplus to ensure positive values (required for Poisson loss)
                return self.activation(track_preds)

        # 3. Instantiate and return
        return HESCHeadAdapterWrapper(enformer, hesc_indices).to(self.device)

        
        

    def get_model(self, args):
        model_name =  args.model_type
        ModelClass = getattr(corigami_models, model_name)
        num_input_features = 0
        num_input_features = len(self.hparams.input_features)

        num_target_tracks = 0
        if self.predict_1d:
            num_target_tracks = len(self.hparams.output_features)
        print(f'Number of input genomic features: {num_input_features}')
        print(f'Number of target 1D tracks: {num_target_tracks}')

        # Instantiate the model
        model = ModelClass(
            num_genomic_features=num_input_features, # Input features
            num_target_tracks=num_target_tracks,    # Target 1D tracks
            conditioning_vec_size=len(self.hparams.conditioning_vec[0].split(',')) if self.hparams.conditioning_vec is not None else None,
            mid_hidden=self.hparams.model_latent_dim,
            predict_hic=self.hparams.predict_hic,
            diploid=args.dataset_assembly2 is not None,
            predict_1d=False,
            target_mat_size=args.mat_size,
            target_1d_length=args.target_1d_size,
            recon_1d=args.recon_1d,
            seq_filter_size=args.seq_filter_size,
            activation_1d=None
        )
        if args.model_path is not None:
            checkpoint = torch.load(args.model_path, map_location='cuda' if torch.cuda.is_available() else 'cpu')
            model_weights = checkpoint['state_dict']
            for key in list(model_weights):
                model_weights[key.replace('model.', '')] = model_weights.pop(key)
            model.load_state_dict(model_weights)
        return model

    def forward(self, x, conditioning_vec=None):
        return self.model(x, conditioning_vec=conditioning_vec)

    def proc_batch(self, batch):
        target_1d_tracks = None
        if self.hparams.conditioning_vec is not None:
            if self.predict_1d:
                seq, features, mat, target_1d_tracks, start, end, chr_name, chr_idx, condition_vec = batch
            else:
                seq, features, mat, start, end, chr_name, chr_idx, condition_vec = batch
        else:
            condition_vec = None
            if self.hparams.predict_hic:
                if self.predict_1d:
                    seq, features, mat, target_1d_tracks, start, end, chr_name, chr_idx = batch
                else:
                    seq, features, mat, start, end, chr_name, chr_idx = batch
            else:
                if self.predict_1d:
                    seq, features, target_1d_tracks, start, end, chr_name, chr_idx = batch
                else:
                    seq, features, start, end, chr_name, chr_idx = batch
        if len(features) > 0:
            features = torch.cat([feat.unsqueeze(2) for feat in features], dim = 2)
            inputs = torch.cat([seq, features], dim = 2)
        else:
            inputs = seq
        inputs = inputs.float() 
        if self.hparams.predict_hic:
            mat = mat.float()
        else:
            mat = None
        if target_1d_tracks is not None:
            target_1d_tracks = torch.stack(target_1d_tracks, dim = 2)
        target_1d_tracks = target_1d_tracks.float() if target_1d_tracks is not None else None
        condition_vec = condition_vec.float() if condition_vec is not None else None
        return inputs, mat, target_1d_tracks, condition_vec
    

    def get_dataloader(self, args, mode):
        datasets = []
        for celltype in args.dataset_celltypes:
            dataset = self.get_dataset(args, mode, celltype)

            if mode == 'train':
                shuffle = True
            else: # validation and test settings
                shuffle = False
            
            batch_size = args.dataloader_batch_size
            num_workers = args.dataloader_num_workers

            if not args.dataloader_ddp_disabled:
                gpus = args.trainer_num_gpu
                batch_size = int(args.dataloader_batch_size / gpus)
                num_workers = int(args.dataloader_num_workers / gpus) 
            
            datasets.append(dataset)
        dataset = torch.utils.data.ConcatDataset(datasets)

        dataloader = torch.utils.data.DataLoader(
            dataset,
            shuffle=shuffle,
            batch_size=batch_size,
            num_workers=num_workers,
            pin_memory=True,
            prefetch_factor=1,
            persistent_workers=True
        )
        return dataloader

# --- Helper Functions ---

def load_ground_truth_matrix(cooler_path, chrom, start, end, target_res, target_size):
    """Loads ground truth Hi-C from a cooler file for a specific window."""
    try:
        c = cooler.Cooler(cooler_path)
        # Select region
        matrix = c.matrix(balance=True).fetch((chrom, start, end), (chrom, start, end))
        # Handle NaNs
        matrix = np.nan_to_num(matrix)
        # Resize if resolution doesn't match the model output (target_size)
        if matrix.shape[0] != target_size:
            matrix = resize(matrix, (target_size, target_size), anti_aliasing=True, preserve_range=True)
        return matrix
    except Exception as e:
        print(f"Warning: Could not load GT from {cooler_path} for {chrom}:{start}-{end}. {e}")
        return np.zeros((target_size, target_size))

def reverse_complement(seq_tensor):
    """Generates the reverse complement of a one-hot encoded DNA sequence tensor."""
    # seq_tensor shape: (batch, seq_len, 4)
    # A <-> T, C <-> G
    rev_comp = seq_tensor.flip(dims=[1])  # Reverse the sequence
    # we have ATCG order, need to swap A<->T and C<->G
    rev_comp = rev_comp[:, :, [1, 0, 3, 2]]  # Swap A<->T and C<->G
    return rev_comp
    
def apply_sequence_ko(inputs, seq_ko_starts, seq_ko_ends, seq_ko_types):
    """Applies sequence knockouts to the input tensor."""
    for ko_start, ko_end, ko_type in zip(seq_ko_starts, seq_ko_ends, seq_ko_types):
        if ko_type == 'zero':
            inputs[:, ko_start:ko_end, :4] = 0.0
        elif ko_type == 'reverse':
            inputs[:, ko_start:ko_end, :4] = reverse_complement(inputs[:, ko_start:ko_end, :4])
        elif ko_type == 'random':
            seq_len = ko_end - ko_start
            # Generate random one-hot sequence
            rand_bases = np.random.choice(4, size=(inputs.shape[0], seq_len))
            rand_seq = np.zeros((inputs.shape[0], seq_len, 4), dtype=np.float32)
            for i in range(4):
                rand_seq[:, :, i] = (rand_bases == i).astype(np.float32)
            rand_seq = torch.tensor(rand_seq, device=inputs.device)
            # if within range, apply
            if ko_end <= inputs.shape[1] and ko_start >= 0:
                inputs[:, ko_start:ko_end, :4] = rand_seq
        # if seq_ko_type is a valid alt seq (only a, t, c, g, n)
        elif all(base in 'atcgn' for base in ko_type.lower()):
            alt_seq_str = ko_type.lower()
            alt_seq_len = ko_end - ko_start
            if len(alt_seq_str) != alt_seq_len:
                raise ValueError(f"Alt sequence length {len(alt_seq_str)} does not match knockout region length {alt_seq_len}.")
            # Create one-hot encoding for alt sequence
            base_to_onehot = {
                'a': [1, 0, 0, 0],
                't': [0, 1, 0, 0],
                'c': [0, 0, 1, 0],
                'g': [0, 0, 0, 1],
                'n': [0.25, 0.25, 0.25, 0.25]
            }
            alt_onehot = torch.tensor([base_to_onehot[base] for base in alt_seq_str], device=inputs.device).unsqueeze(0)
            # get ref seq for debugging
            if ko_end <= inputs.shape[1] and ko_start >= 0:
                ref_seq = inputs[:, ko_start:ko_end, :4].clone()
                ref_seq_str = ''
                for i in range(ref_seq.shape[1]):
                    base_idx = torch.argmax(ref_seq[0, i]).item()
                    base_char = 'ATCG'[base_idx]
                    ref_seq_str += base_char
                # print comparison
                print(f"Applying alt sequence at {ko_start}-{ko_end}: Ref: {ref_seq_str} -> Alt: {alt_seq_str.upper()}")
            inputs[:, ko_start:ko_end, :4] = alt_onehot
        else:
            raise ValueError(f"Unknown seq_ko_type: {ko_type}")
    return inputs

def track_ko(start, end, track, window=2097152, ko_mode='knockout', peak_height=2.0):
    """
    Perform knockout on a 1D signal track within the given [start, end) region.
    
    Supported ko_mode values:
        - 'zero': Set the region to 0.
        - 'mean': Replace region with mean of flanking signal.
        - 'knockout': Replace detected peaks with local background (default).
        - 'increase_<factor>': Multiply peak values by <factor> (default 2.0).
        - 'cluster_<ratio>': Add random peaks at <ratio> fraction of positions.
        - 'shuffle': Chunk-shuffle the region.
        - 'knockout_shuffle': Knockout peaks then shuffle.
        - 'reverse' / 'reverse_motif': Reverse the signal in the region.
    """
    if ko_mode == 'zero':
        track[start:end] = 0
    elif ko_mode == 'mean':
        flanking = np.concatenate([track[:start], track[end:]])
        mean_val = np.mean(flanking) if len(flanking) > 0 else 0.0
        track[start:end] = mean_val
    elif ko_mode == 'knockout':
        track[start:end] = knockout_peaks(track[start:end], threshold=peak_height)
    elif 'increase' in ko_mode:
        increase_factor = 2.0
        if '_' in ko_mode:
            increase_factor = float(ko_mode.split('_')[1])
        track[start:end] = knockout_peaks(track[start:end], threshold=peak_height, increase_factor=increase_factor)
    elif 'cluster' in ko_mode:
        cluster_ratio = 0.05
        if '_' in ko_mode:
            cluster_ratio = float(ko_mode.split('_')[1])
        cluster_indices = np.random.choice(np.arange(start, end), size=int((end - start) * cluster_ratio), replace=False)
        for idx in cluster_indices:
            track[idx] = np.random.uniform(1, 5)
    elif ko_mode == 'shuffle':
        track[start:end] = chunk_shuffle(track[start:end])
    elif ko_mode == 'knockout_shuffle':
        track[start:end] = knockout_peaks(track[start:end], threshold=peak_height)
        track[start:end] = chunk_shuffle(track[start:end])
    elif ko_mode == 'reverse' or ko_mode == 'reverse_motif':
        track[start:end] = track[start:end][::-1]
    else:
        raise ValueError(f"Unknown ko_mode: '{ko_mode}'. Supported: zero, mean, knockout, increase_<f>, cluster_<r>, shuffle, knockout_shuffle, reverse, reverse_motif")
    return track[:window]


def apply_track_ko(seq_region, ctcf_region, atac_region, other_regions, other_track_names,
                   ko_data, ko_mode, window=2097152, peak_height=0.5):
    """
    Apply knockout perturbations to the specified 1D input tracks across the full window.
    
    Args:
        seq_region: Sequence array.
        ctcf_region: CTCF signal array (or None).
        atac_region: ATAC signal array (or None).
        other_regions: List of other genomic feature arrays (or None).
        other_track_names: List of names for tracks in other_regions.
        ko_data: List of track names to knockout (e.g. ['ctcf', 'rad21']).
        ko_mode: List of KO modes, one per ko_data entry (e.g. ['knockout', 'zero']).
        window: Window size in bp.
        peak_height: Peak height threshold for 'knockout' mode.
    
    Returns:
        Tuple of (seq_region, ctcf_region, atac_region, other_regions) with KO applied.
    """
    if len(ko_mode) == 1 and len(ko_data) > 1:
        ko_mode = ko_mode * len(ko_data)  # broadcast single mode to all tracks
    
    for track_name, mode in zip(ko_data, ko_mode):
        region_len = len(ctcf_region) if ctcf_region is not None else (
            len(atac_region) if atac_region is not None else window)
        if track_name == 'ctcf' and ctcf_region is not None:
            ctcf_region = track_ko(0, region_len, ctcf_region.copy(), window=window,
                                   ko_mode=mode, peak_height=peak_height)
        elif track_name == 'atac' and atac_region is not None:
            atac_region = track_ko(0, region_len, atac_region.copy(), window=window,
                                   ko_mode=mode, peak_height=peak_height)
        elif track_name == 'seq':
            if mode == 'knockout' or mode == 'zero':
                seq_region[:, :4] = 0
                if seq_region.shape[1] > 4:
                    seq_region[:, 4] = 1  # set to N
            elif mode == 'shuffle':
                idxs = np.arange(seq_region.shape[0])
                np.random.shuffle(idxs)
                seq_region = seq_region[idxs, :]
            elif mode == 'reverse':
                seq_region = seq_region[::-1].copy()
                seq_region = seq_region[:, [1, 0, 3, 2] + list(range(4, seq_region.shape[1]))]
        elif other_regions is not None and track_name in other_track_names:
            idx = other_track_names.index(track_name)
            other_regions[idx] = track_ko(0, len(other_regions[idx]), other_regions[idx].copy(),
                                          window=window, ko_mode=mode, peak_height=peak_height)
        else:
            print(f"Warning: KO track '{track_name}' not found in input tracks. Skipping.")
    
    return seq_region, ctcf_region, atac_region, other_regions


def main():
    parser = argparse.ArgumentParser(description='Hierarchical C.Origami Full Chromosome Prediction')
    
    # Paths
    parser.add_argument('--model', dest='model_path', required=True, help='Path to .ckpt file')
    parser.add_argument('--out', dest='out_file', required=True, help='Output .tsv file path')
    
    # Genomic Data
    parser.add_argument('--chrom', required=True, help='Chromosome name (e.g., chr7)')
    parser.add_argument('--locus', required=False, help='Locus (e.g., chr7:55000000-60000000)')
    parser.add_argument('--assembly', default='mm10', help='Genome assembly')
    parser.add_argument('--seq', dest='seq_path', required=True, help='Path to dna_sequence folder')
    parser.add_argument('--ctcf', required=True, help='Path to WT CTCF bigwig')
    parser.add_argument('--atac', required=True, help='Path to ATAC bigwig')
    parser.add_argument('--hic-wt', dest='hic_wt_path', required=False, help='Path to WT .cool file (for ground truth)')
    parser.add_argument('--hic-ko', dest='hic_ko_path', required=False, help='Path to KO .cool file (for ground truth)')
    
    # Parameters
    parser.add_argument('--window', type=int, default=2097152, help='Window size (bp)')
    parser.add_argument('--step-size', type=int, default=1048576, help='Step size for sliding window')
    parser.add_argument('--resolution', type=int, default=4096, help='Hi-C output resolution')
    parser.add_argument('--mat-size', type=int, default=512, help='Model output matrix size')
    parser.add_argument('--peak-height', type=float, default=0.5, help='Peak height threshold for predictions')
    parser.add_argument('--log-transform-bw', action='store_true', default=True, help='Log transform input bigwigs')
    parser.add_argument('--undo-log-hic', action='store_true', default=True, help='Undo log transform on Hi-C output')
    parser.add_argument('--vmin', type=float, default=0.0, help='Minimum value for visualization')
    parser.add_argument('--vmax', type=float, default=None, help='Maximum value for visualization')
    parser.add_argument('--viz-threshold', type=float, default=98.0, help='Percentile threshold for graph visualization edges')

    # seq perturb params
    parser.add_argument('--seq-ko-starts', nargs='+', type=int, help='Start positions of sequence knockouts (bp)')
    parser.add_argument('--seq-ko-sizes', nargs='+', type=int, help='Sizes of sequence knockouts (bp)')
    parser.add_argument('--seq-ko-type', nargs='+', help='Type of sequence knockout (or alt seq) [reverse, zero, random, alt]')

    # 1D track knockout params
    parser.add_argument('--ko', dest='ko_data', type=str, nargs='+', default=['ctcf'],
                        help='Name(s) of input track(s) to knockout (e.g. ctcf, rad21, atac, h3k27ac). Default: ctcf')
    parser.add_argument('--ko-mode', dest='ko_mode', type=str, nargs='+', default=['knockout'],
                        help='KO mode(s), one per --ko track. Options: zero, mean, knockout, shuffle, '
                             'knockout_shuffle, reverse, reverse_motif, increase_<factor>, cluster_<ratio>. '
                             'Default: knockout')

    args = parser.parse_args()

    # Normalize KO arguments
    if isinstance(args.ko_data, str):
        args.ko_data = [args.ko_data]
    if isinstance(args.ko_mode, str):
        args.ko_mode = [args.ko_mode]
    # Broadcast single mode to all KO tracks if needed
    if len(args.ko_mode) == 1 and len(args.ko_data) > 1:
        args.ko_mode = args.ko_mode * len(args.ko_data)
    if len(args.ko_mode) != len(args.ko_data):
        parser.error(f"--ko-mode must have 1 entry or the same number as --ko ({len(args.ko_data)}).")

    ko_label = '+'.join(f'{t}({m})' for t, m in zip(args.ko_data, args.ko_mode))
    print(f"Track KO configuration: {ko_label}")

    print(f"Loading model from {args.model_path}...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(args.model_path, map_location=device)
    hparams = argparse.Namespace(**checkpoint['hyper_parameters'])
    module = TrainModule.load_from_checkpoint(args.model_path, args=hparams)
    inner_model = module.input_pred_model
    inner_model.eval()
    inner_model.to(device)
    
    model = module.model
    model.eval()
    model.to(device)

    # save the final model (without inner_model) as a standalone checkpoint for usage with our original scripts
    checkpoint = torch.load(args.model_path, map_location='cpu')
    model_weights = checkpoint['state_dict']
    for key in list(model_weights):
        if 'input_pred' in key:
            # remove from state dict
            model_weights.pop(key)
    # Save standalone model checkpoint
    standalone_ckpt_path = args.model_path.replace('.ckpt', '_standalone.ckpt')
    checkpoint['state_dict'] = model_weights
    torch.save(checkpoint, standalone_ckpt_path)
    print(f"Standalone model checkpoint saved to {standalone_ckpt_path}")

    # Get chromosome length from CTCF bigwig
    bw = pyBigWig.open(args.ctcf)
    chrom_len = bw.chroms(args.chrom)
    if chrom_len is None:
        chrom_len = bw.chroms(args.chrom.replace('chr', ''))
    print(f"Chromosome {args.chrom} length: {chrom_len} bp")
    bw.close()

    # Inspect checkpoint for track names
    all_tracks, _, input_tracks = get_all_track_names(args.model_path)
    rad21_idx = all_tracks.index('rad21')
    other_paths = []
    other_track_names = []
    for track in all_tracks:
        if track not in ['ctcf', 'atac']:
            other_paths.append(args.ctcf.replace('ctcf.bw', f'{track}.bw'))  # Assuming similar naming
            other_track_names.append(track)
    
    if args.locus:
        locus_parts = args.locus.split(':')
        locus_chrom = locus_parts[0]
        locus_coords = locus_parts[1].split('-')
        locus_start = int(locus_coords[0].replace(',', ''))
        locus_end = int(locus_coords[1].replace(',', ''))
        if locus_chrom != args.chrom:
            raise ValueError(f"Locus chromosome {locus_chrom} does not match specified chromosome {args.chrom}.")
        locus_len = locus_end - locus_start
        print(f"Using locus {args.locus} with length {locus_len} bp.")
        starts = np.arange(locus_start - args.window, 
                           locus_end +1, 
                           args.step_size)
        ends = starts + args.window
    else:
        starts = np.arange(0, chrom_len - args.window, args.step_size)
        ends = starts + args.window
    
    print(f"Predicting on {args.chrom} in {len(starts)} windows.")
    print(f"Hierarchical Mode: Predicting WT, then simulating {ko_label} -> RAD21 Pred -> Hi-C Pred.")

    # Storage for results
    # We will accumulate dataframes of pixels
    pixel_dfs = []
    rad21_dfs = []
    enformer_preds_dfs = []
    ko_track_dfs = {t: [] for t in args.ko_data if t != 'seq'}  # per-window KO'd signal for each track
    os.makedirs('tmp', exist_ok=True)

    with torch.no_grad():
        for start, end in tqdm(zip(starts, ends), total=len(starts), desc="Processing Windows"):
            # load_region returns: seq, ctcf, atac, other_feats
            seq_region, ctcf_region, atac_region, other_feats = infer.load_region(
                args.chrom, start, args.seq_path, args.ctcf, args.atac, other_paths, 
                window=args.window, bigwig_log=args.log_transform_bw
            )
            
            # Preprocess (convert to tensor, add batch dim)
            inputs_wt = infer.preprocess_default(seq_region, ctcf_region, atac_region, other_feats).to(device)

            # if sequence KO is specified, apply it here
            if args.seq_ko_starts and args.seq_ko_sizes and args.seq_ko_type:
                local_starts = np.array(args.seq_ko_starts) - start
                print(start, np.array(args.seq_ko_starts), local_starts)
                inputs_wt = apply_sequence_ko(
                    inputs_wt, 
                    local_starts, 
                    local_starts + np.array(args.seq_ko_sizes), 
                    args.seq_ko_type
                )

                # predict new CTCF and ATAC using Enformer wrapper
                pred_1d_inputs = module.enformer_predict_1d(inputs_wt)
                # Replace CTCF and ATAC channels
                inputs_wt[:,:,5] = pred_1d_inputs[:,:,0]  # CTCF
                inputs_wt[:,:,6] = pred_1d_inputs[:,:,1]  # ATAC


            # Remove Rad21 channel to predict it from sequence/CTCF/ATAC
            inputs_wt_without_rad21 = torch.cat([
                inputs_wt[:,:,:5 + rad21_idx],
                inputs_wt[:,:,6 + rad21_idx:]
            ], dim=2)
            pred_wt_inner = inner_model(inputs_wt_without_rad21)['1d']
            pred_wt_rad21_64bp = torch.expm1(pred_wt_inner[:, :, 0]).squeeze()

            output_wt = model(inputs_wt)
            pred_hic_wt = output_wt['hic'].squeeze().cpu().numpy()
            if args.undo_log_hic:
                pred_hic_wt = np.expm1(pred_hic_wt) # assuming log1p was used in training
            pred_hic_wt = (pred_hic_wt + pred_hic_wt.T) / 2.0  # Symmetrize
            pred_hic_wt = np.clip(pred_hic_wt, a_min=0, a_max=None)  # No negative counts

            # Apply 1D track knockout(s) using the user-specified tracks and modes
            ko_seq, ko_ctcf, ko_atac, ko_other = apply_track_ko(
                seq_region.copy(), 
                ctcf_region.copy() if ctcf_region is not None else None,
                atac_region.copy() if atac_region is not None else None,
                [o.copy() for o in other_feats] if other_feats is not None else None,
                other_track_names,
                ko_data=args.ko_data,
                ko_mode=args.ko_mode,
                window=args.window,
                peak_height=args.peak_height
            )
            inputs_ko = infer.preprocess_default(ko_seq, ko_ctcf, ko_atac, ko_other).to(device)
            inputs_ko_without_rad21 = torch.cat([
                inputs_ko[:,:,:5 + rad21_idx],
                inputs_ko[:,:,6 + rad21_idx:]
            ], dim=2)
            pred_ko_inner = inner_model(inputs_ko_without_rad21)['1d']
            pred_ko_rad21_64bp = torch.expm1(pred_ko_inner[:, :, 0]).squeeze()

            # Resample to match input resolution (bins)
            pred_ko_rad21 = F.interpolate(pred_ko_rad21_64bp.unsqueeze(0).unsqueeze(0),
                                          size=inputs_ko.shape[1],
                                          mode='linear', align_corners=True).squeeze()
            
            # --- SAVE RAD21 TRACKS ---
            # Calculate genomic coordinates for the 1D tracks
            seq_len = pred_ko_rad21_64bp.shape[0]
            bin_size = args.window / seq_len
            rel_starts = np.arange(0, seq_len) * bin_size
            abs_starts = start + rel_starts
            abs_ends = abs_starts + bin_size
            
            rad21_df_window = pd.DataFrame({
                'chrom': args.chrom,
                'start': abs_starts.astype(int),
                'end': abs_ends.astype(int),
                'WT_rad21': pred_wt_rad21_64bp.cpu().numpy(),
                'KO_rad21': pred_ko_rad21_64bp.cpu().numpy()
            })
            rad21_dfs.append(rad21_df_window)

            if args.locus is not None:  # save KO'd input tracks for locus mode
                # Build a map of each KO'd track name -> its perturbed signal array
                ko_signals = {}
                for t in args.ko_data:
                    if t == 'ctcf' and ko_ctcf is not None:
                        ko_signals[t] = ko_ctcf
                    elif t == 'atac' and ko_atac is not None:
                        ko_signals[t] = ko_atac
                    elif t in other_track_names and ko_other is not None:
                        ko_signals[t] = ko_other[other_track_names.index(t)]
                for t, signal in ko_signals.items():
                    col_name = f'{t}_KO'
                    ko_region_64bp = F.interpolate(
                        torch.from_numpy(signal).unsqueeze(0).unsqueeze(0),
                        size=pred_ko_rad21_64bp.shape[0],
                        mode='linear', align_corners=True
                    ).squeeze().numpy()
                    ko_df_window = pd.DataFrame({
                        'chrom': args.chrom,
                        'start': abs_starts.astype(int),
                        'end': abs_ends.astype(int),
                        col_name: ko_region_64bp
                    })
                    ko_track_dfs[t].append(ko_df_window)

            # save enformer predictions if seq KO was applied
            if args.seq_ko_starts and args.seq_ko_sizes and args.seq_ko_type:
                enformer_64bp_pred = F.interpolate(pred_1d_inputs.permute(0,2,1),
                                                  size=pred_wt_rad21_64bp.shape[0],
                                                  mode='linear', align_corners=True).squeeze().permute(1,0).cpu().numpy()
                enformer_df_window = pd.DataFrame({
                    'chrom': args.chrom,
                    'start': abs_starts.astype(int),
                    'end': abs_ends.astype(int),
                    'Enformer_CTCF': enformer_64bp_pred[:,0],
                    'Enformer_ATAC': enformer_64bp_pred[:,1]
                })
                enformer_preds_dfs.append(enformer_df_window)
            
            # Replace RAD21 channel (assumed to be after seq, ctcf, atac channels)
            if 'rad21' not in args.ko_data:
                inputs_ko[:,:,5 + rad21_idx] = pred_ko_rad21
            output_ko = model(inputs_ko)
            pred_hic_ko = output_ko['hic'].squeeze().cpu().numpy()

            if args.undo_log_hic:
                pred_hic_ko = np.expm1(pred_hic_ko)
            pred_hic_ko = (pred_hic_ko + pred_hic_ko.T) / 2.0  # Symmetrize
            pred_hic_ko = np.clip(pred_hic_ko, a_min=0, a_max=None)  # No negative counts

            # Load Ground Truths (if provided)
            gt_hic_wt = np.zeros_like(pred_hic_wt)
            gt_hic_ko = np.zeros_like(pred_hic_ko)
            
            if args.hic_wt_path:
                gt_hic_wt = load_ground_truth_matrix(args.hic_wt_path, args.chrom, start, end, args.resolution, args.mat_size)
            
            if args.hic_ko_path:
                gt_hic_ko = load_ground_truth_matrix(args.hic_ko_path, args.chrom, start, end, args.resolution, args.mat_size)

            # Process into Pixels (Using temp files to handle binning logic consistently)
            write_tmp_cooler(pred_hic_wt, args.chrom, start, out_file='tmp/pred_wt.cool', res=args.resolution)
            write_tmp_cooler(pred_hic_ko, args.chrom, start, out_file='tmp/pred_ko.cool', res=args.resolution)
            write_tmp_cooler(gt_hic_wt, args.chrom, start, out_file='tmp/true_wt.cool', res=args.resolution)
            write_tmp_cooler(gt_hic_ko, args.chrom, start, out_file='tmp/true_ko.cool', res=args.resolution)

            # Read pixels back
            c_pred_wt = cooler.Cooler('tmp/pred_wt.cool')
            c_pred_ko = cooler.Cooler('tmp/pred_ko.cool')
            c_true_wt = cooler.Cooler('tmp/true_wt.cool')
            c_true_ko = cooler.Cooler('tmp/true_ko.cool')

            # sometimes randomly save a heatmap for debugging
            if np.random.rand() < 0.2:
                min_99p_val = 0.5
                if np.percentile(pred_hic_wt, 99) < min_99p_val or np.percentile(pred_hic_ko, 99) < min_99p_val:
                    pass
                else:
                    fig, axs = plt.subplots(2,2, figsize=(10,10))
                    im = axs[0,0].imshow(pred_hic_wt, cmap='Reds',
                                    vmin=0, vmax=np.percentile(pred_hic_wt, 99))
                    plt.colorbar(im, ax=axs[0,0])
                    axs[0,0].set_title('Predicted WT Hi-C')
                    im = axs[0,1].imshow(gt_hic_wt, cmap='Reds',
                                    vmin=0, vmax=np.percentile(gt_hic_wt, 99))
                    plt.colorbar(im, ax=axs[0,1])
                    axs[0,1].set_title('True WT Hi-C')
                    im = axs[1,0].imshow(pred_hic_ko, cmap='Reds',
                                    vmin=0, vmax=np.percentile(pred_hic_ko, 99))
                    plt.colorbar(im, ax=axs[1,0])
                    axs[1,0].set_title('Predicted KO Hi-C')
                    im = axs[1,1].imshow(gt_hic_ko, cmap='Reds',
                                    vmin=0, vmax=np.percentile(gt_hic_ko, 99))
                    plt.colorbar(im, ax=axs[1,1])
                    axs[1,1].set_title('True KO Hi-C')
                    plt.suptitle(f'{args.chrom}:{start}-{end}')
                    plt.savefig(f'tmp/heatmap_{args.chrom}_{start}_{end}.png')
                    plt.close()

            pix_p_wt = c_pred_wt.pixels()[:].rename(columns={'count': 'WT_pred'})
            pix_p_ko = c_pred_ko.pixels()[:].rename(columns={'count': 'KO_pred'})
            pix_t_wt = c_true_wt.pixels()[:].rename(columns={'count': 'WT_true'})
            pix_t_ko = c_true_ko.pixels()[:].rename(columns={'count': 'KO_true'})

            # Merge frames on bin_ids
            # Start with WT Pred as base
            merged = pix_p_wt.merge(pix_p_ko, on=['bin1_id', 'bin2_id'], how='outer')
            merged = merged.merge(pix_t_wt, on=['bin1_id', 'bin2_id'], how='outer')
            merged = merged.merge(pix_t_ko, on=['bin1_id', 'bin2_id'], how='outer')

            # Fill NaNs with 0 (sparse matrix assumption)
            merged = merged.fillna(0)
            # drop rows where all values are <1e-2
            merged = merged[(merged[['WT_pred', 'KO_pred', 'WT_true', 'KO_true']].abs() >= 1e-2).any(axis=1)].reset_index(drop=True)
            # round values to 4 decimal places
            merged[['WT_pred', 'KO_pred', 'WT_true', 'KO_true']] = merged[['WT_pred', 'KO_pred', 'WT_true', 'KO_true']].round(4)
            # Store bins metadata for this window to map back to coords later
            bins_data = c_pred_wt.bins()[:]
            # Map IDs to Coords
            merged['start1'] = bins_data.iloc[merged['bin1_id']]['start'].values
            merged['end1']   = bins_data.iloc[merged['bin1_id']]['end'].values
            merged['start2'] = bins_data.iloc[merged['bin2_id']]['start'].values
            merged['end2']   = bins_data.iloc[merged['bin2_id']]['end'].values
            merged['chrom'] = args.chrom

            # Drop local bin IDs
            merged = merged.drop(columns=['bin1_id', 'bin2_id'])
            
            pixel_dfs.append(merged)

    # --- AGGREGATE & SAVE 1D TRACKS (BigWigs) ---
    print("Aggregating 1D tracks...")
    full_rad21_df = pd.concat(rad21_dfs, ignore_index=True)
    # Average overlapping predictions
    final_rad21_df = full_rad21_df.groupby(['chrom', 'start', 'end']).mean().reset_index()
    bw_wt_path = args.out_file.replace('.tsv', '_WT_rad21.bw')
    bw_ko_path = args.out_file.replace('.tsv', '_KO_rad21.bw')
    # Aggregate and save bigwigs for each KO'd input track
    ko_bw_paths = {}  # track_name -> output bigwig path
    if args.locus:
        for t in args.ko_data:
            if t == 'seq' or t not in ko_track_dfs or len(ko_track_dfs[t]) == 0:
                continue
            col_name = f'{t}_KO'
            full_ko_df = pd.concat(ko_track_dfs[t], ignore_index=True)
            final_ko_df = full_ko_df.groupby(['chrom', 'start', 'end']).mean().reset_index()
            bw_path = args.out_file.replace('.tsv', f'_{t.upper()}_KO.bw')
            write_bigwig(final_ko_df, args.chrom, bw_path, chrom_len, col_name)
            ko_bw_paths[t] = bw_path
            print(f"{t.upper()} KO bigwig saved to {bw_path}")
    if args.seq_ko_starts and args.seq_ko_sizes and args.seq_ko_type:
        full_enformer_df = pd.concat(enformer_preds_dfs, ignore_index=True)
        final_enformer_df = full_enformer_df.groupby(['chrom', 'start', 'end']).mean().reset_index()
        bw_enformer_ctcf_path = args.out_file.replace('.tsv', '_Enformer_CTCF.bw')
        bw_enformer_atac_path = args.out_file.replace('.tsv', '_Enformer_ATAC.bw')
        write_bigwig(final_enformer_df[['chrom', 'start', 'end', 'Enformer_CTCF']], args.chrom, bw_enformer_ctcf_path, chrom_len, 'Enformer_CTCF')
        write_bigwig(final_enformer_df[['chrom', 'start', 'end', 'Enformer_ATAC']], args.chrom, bw_enformer_atac_path, chrom_len, 'Enformer_ATAC')
        print(f"Enformer predicted CTCF and ATAC bigwigs saved to {bw_enformer_ctcf_path} and {bw_enformer_atac_path}")
    
    # Get actual chrom length for header from bigwig (already fetched) or max coordinate
    # Re-using chrom_len calculated earlier from CTCF input
    
    skip_pred_rad21 = 'rad21' in args.ko_data
    write_bigwig(final_rad21_df, args.chrom, bw_wt_path, chrom_len, 'WT_rad21')
    if not skip_pred_rad21:
        write_bigwig(final_rad21_df, args.chrom, bw_ko_path, chrom_len, 'KO_rad21')
    else:
        print("Skipping KO predicted RAD21 bigwig (RAD21 is directly KO'd).")

    print("Aggregating Hi-C results...")
    full_df = pd.concat(pixel_dfs, ignore_index=True)
    # Group by coordinates to handle overlapping windows (average predictions)
    final_df = full_df.groupby(['chrom', 'start1', 'end1', 'start2', 'end2']).mean().reset_index()
    output_df = final_df[['start1', 'end1', 'start2', 'end2', 'WT_true', 'WT_pred', 'KO_true', 'KO_pred']]
    output_df.columns = ['start1', 'end1', 'start2', 'end2', 'WT_true', 'WT_pred', 'KO_true', 'KO_pred']
    output_df['anchor1'] = (output_df['start1'] // args.resolution)
    output_df['anchor2'] = (output_df['start2'] // args.resolution)
    wt_pixels = output_df[['anchor1', 'anchor2', 'WT_pred']].copy()
    ko_pixels = output_df[['anchor1', 'anchor2', 'KO_pred']].copy()
    wt_pixels = wt_pixels.rename(columns={'WT_pred': 'count', 'anchor1': 'bin1_id', 'anchor2': 'bin2_id'})
    ko_pixels = ko_pixels.rename(columns={'KO_pred': 'count', 'anchor1': 'bin1_id', 'anchor2': 'bin2_id'})
    final_cols = ['anchor1', 'anchor2', 'start1', 'end1', 'start2', 'end2', 'WT_true', 'WT_pred', 'KO_true', 'KO_pred']
    output_df = output_df[final_cols]
    
    # Save
    print(f"Saving to {args.out_file}...")
    # Make dir
    os.makedirs(os.path.dirname(os.path.abspath(args.out_file)), exist_ok=True)
    
    output_df.to_csv(args.out_file, sep='\t', index=False)

    if args.locus:
        print(f"Prediction completed for locus {args.locus} on {args.chrom}.")
        # save to cooler for heatmap visualization
        write_full_cooler(wt_pixels, args.chrom, locus_start - args.window, res=args.resolution, window=locus_end - locus_start + args.window * 2, 
                          out_file=args.out_file.replace('.tsv', '') + '_wt.cool')
        write_full_cooler(ko_pixels, args.chrom, locus_start - args.window, res=args.resolution, window=locus_end - locus_start + args.window * 2, 
                          out_file=args.out_file.replace('.tsv', '') + '_ko.cool')
        print(f"Temporary coolers for locus saved to {args.out_file.replace('.tsv', '') + '_wt.cool'} and {args.out_file.replace('.tsv', '') + '_ko.cool'}")
        # Optionally, generate heatmaps
        pred_wt_matrix = cooler.Cooler(args.out_file.replace('.tsv', '') + '_wt.cool').matrix(balance=False).fetch(args.locus)
        pred_ko_matrix = cooler.Cooler(args.out_file.replace('.tsv', '') + '_ko.cool').matrix(balance=False).fetch(args.locus)
        fig, axs = plt.subplots(1,3, figsize=(18,6))
        im = axs[0].imshow(pred_wt_matrix, cmap='Reds',
                        vmin=0, vmax=np.percentile(pred_wt_matrix, 99.8))
        plt.colorbar(im, ax=axs[0])
        axs[0].set_title('Predicted WT Hi-C')
        axs[0].set_xticks([])
        axs[0].set_yticks([])
        im = axs[1].imshow(pred_ko_matrix, cmap='Reds',
                        vmin=0, vmax=np.percentile(pred_ko_matrix, 99.8))
        plt.colorbar(im, ax=axs[1])
        axs[1].set_title(f'Predicted {ko_label} Hi-C')
        axs[1].set_xticks([])
        axs[1].set_yticks([])

        # plot diff heatmap
        diff_matrix =  pred_ko_matrix - pred_wt_matrix
        max_mag = np.percentile(np.abs(diff_matrix), 98)
        im = axs[2].imshow(diff_matrix, cmap='bwr',
                        vmin=-max_mag, vmax=max_mag)
        # label colorbar as loss --> gain
        cbar = plt.colorbar(im, ax=axs[2])
        cbar.set_label('loss (blue) --> gain (red)')
        axs[2].set_title('Predicted WT - KO Hi-C')
        axs[2].set_xticks([])
        axs[2].set_yticks([])
        plt.suptitle(f'{args.chrom}:{locus_start}-{locus_end}')
        plt.savefig(args.out_file.replace('.tsv', '') + f'_heatmap_{args.chrom}_{locus_start}_{locus_end}.png')
        plt.savefig(args.out_file.replace('.tsv', '') + f'_heatmap_{args.chrom}_{locus_start}_{locus_end}.pdf')
        plt.close()

        # 2. Force Directed Graph Visualization
        # graph_out_path = args.out_file.replace('.tsv', '') + f'_structure_{args.chrom}_{locus_start}_{locus_end}.png'
        # visualize_force_directed_structure(
        #     pred_wt_matrix, 
        #     pred_ko_matrix, 
        #     title_suffix=f"{args.chrom}:{locus_start}-{locus_end}", 
        #     out_path=graph_out_path,
        #     threshold_percentile=args.viz_threshold
        # )

        # --- Generate pyGenomeTracks .ini files ---
        # Build metadata for all input tracks: (name, original_path, color)
        track_display_names = {
            'ctcf': 'CTCF', 'atac': 'ATAC', 'rad21': 'RAD21',
            'h3k27ac': 'H3K27ac', 'h3k4me3': 'H3K4me3', 'h3k9me3': 'H3K9me3',
            'h3k36me3': 'H3K36me3', 'h3k27me3': 'H3K27me3'
        }
        track_color_map = {
            'ctcf': 'royalblue', 'atac': 'green', 'rad21': 'blue',
            'h3k27ac': 'red', 'h3k4me3': 'purple', 'h3k9me3': 'brown',
            'h3k36me3': 'pink', 'h3k27me3': 'cyan'
        }
        fallback_colors = ['red', 'purple', 'brown', 'pink', 'cyan', 'magenta', 'lime']
        all_input_tracks = []  # (name, path, color)
        all_input_tracks.append(('ctcf', args.ctcf, track_color_map.get('ctcf', 'royalblue')))
        all_input_tracks.append(('atac', args.atac, track_color_map.get('atac', 'green')))
        for i, (t_name, t_path) in enumerate(zip(other_track_names, other_paths)):
            if os.path.exists(t_path):
                all_input_tracks.append((t_name, t_path, track_color_map.get(t_name, fallback_colors[i % len(fallback_colors)])))

        # Compute vmax for each track from the real bigwig signal
        track_vmax = {}
        for t_name, t_path, _ in all_input_tracks:
            try:
                track_vmax[t_name] = get_axis_range_from_bigwig(t_path, args.chrom, locus_start)
            except Exception:
                track_vmax[t_name] = 'auto'

        ko_short_label = '+'.join(args.ko_data)
        tracks_wt_ini = args.out_file.replace('.tsv', '') + f'_WT_tracks.ini'
        tracks_ko_ini = args.out_file.replace('.tsv', '') + f'_KO_{ko_short_label}_tracks.ini'

        # Helper to write a single bigwig track entry
        def _write_track_entry(fh, section_name, file_path, title, color, vmax_val):
            fh.write(f"[{section_name}]\n")
            fh.write(f"file = {file_path}\n")
            fh.write(f"title = {title}\n")
            fh.write(f"height = {track_height_1d}\n")
            fh.write(f"color = {color}\n")
            fh.write(f"min_value = 0\n")
            fh.write(f"max_value = {vmax_val}\n\n")

        def _write_hic_entry(fh, file_path, title, vmin_val, vmax_val):
            fh.write(f"[pred_hic]\n")
            fh.write(f"file = {file_path}\n")
            fh.write(f"title = {title}\n")
            fh.write(f"file_type = hic_matrix_square\n")
            fh.write(f"min_value = {vmin_val}\n")
            fh.write(f"max_value = {vmax_val}\n")
            fh.write("colormap = [ (1.0, 1.0, 1.0), (1.0, 0.92, 0.92),(1.0, 0.8, 0.8),(1.0, 0.6, 0.6), (1.0, 0.4, 0.4),(1.0, 0.294, 0.294)]\n\n")

        # --- WT tracks.ini ---
        with open(tracks_wt_ini, 'w') as f:
            f.write("[spacer]\nheight = 0.1\ncolor = white\n\n")
            # Write all real input tracks
            for t_name, t_path, t_color in all_input_tracks:
                display = track_display_names.get(t_name, t_name)
                _write_track_entry(f, t_name, t_path, display, t_color, track_vmax.get(t_name, 'auto'))
            # Enformer predictions if seq KO was done
            if args.seq_ko_starts and args.seq_ko_sizes and args.seq_ko_type:
                with open(args.out_file.replace('.tsv', '') + f'_ko_regions.bed', 'w') as bed_file:
                    for ko_start, ko_size, ko_type in zip(args.seq_ko_starts, args.seq_ko_sizes, args.seq_ko_type):
                        ko_end = ko_start + ko_size
                        pad = 0
                        if ko_end - ko_start < 10000:
                            pad = (10000 - (ko_end - ko_start)) // 2
                        bed_file.write(f"{args.chrom}\t{ko_start - pad}\t{ko_end + pad}\t{ko_type}\n")
                _write_track_entry(f, 'enformer_ctcf', bw_enformer_ctcf_path, 'Enformer CTCF', 'teal', track_vmax.get('ctcf', 'auto'))
                f.write(f"[ko highlight]\nfile = {args.out_file.replace('.tsv', '') + f'_ko_regions.bed'}\ntype = vhighlight\n\n")
                _write_track_entry(f, 'enformer_atac', bw_enformer_atac_path, 'Enformer ATAC', 'darkgreen', track_vmax.get('atac', 'auto'))
            # Predicted WT RAD21 (always useful in WT view unless RAD21 is being KO'd)
            if not skip_pred_rad21:
                _write_track_entry(f, 'pred_rad21_wt', bw_wt_path, 'Predicted RAD21', 'orange', track_vmax.get('rad21', 'auto'))
            # Hi-C
            hic_vmax = args.vmax if args.vmax is not None else np.percentile(pred_wt_matrix, 99)
            _write_hic_entry(f, args.out_file.replace('.tsv', '') + '_wt.cool', 'Predicted Hi-C', args.vmin, hic_vmax)

        # --- KO tracks.ini ---
        with open(tracks_ko_ini, 'w') as f:
            f.write("[spacer]\nheight = 0.1\ncolor = white\n\n")
            # Write each input track: KO'd version if available, otherwise original
            for t_name, t_path, t_color in all_input_tracks:
                display = track_display_names.get(t_name, t_name)
                if t_name in ko_bw_paths:
                    _write_track_entry(f, f'{t_name}_ko', ko_bw_paths[t_name],
                                       f'{display} (KO)', t_color, track_vmax.get(t_name, 'auto'))
                else:
                    _write_track_entry(f, t_name, t_path, display, t_color, track_vmax.get(t_name, 'auto'))
            # Predicted KO RAD21 (only if RAD21 is not directly KO'd)
            if not skip_pred_rad21:
                _write_track_entry(f, 'pred_rad21_ko', bw_ko_path, 'Predicted RAD21', 'orange', track_vmax.get('rad21', 'auto'))
            # Hi-C
            hic_vmax_ko = args.vmax if args.vmax is not None else np.percentile(pred_ko_matrix, 99)
            _write_hic_entry(f, args.out_file.replace('.tsv', '') + '_ko.cool', 'Predicted Hi-C', args.vmin, hic_vmax_ko)

        print(f"Tracks INI files saved to {tracks_wt_ini} and {tracks_ko_ini}")
        pygenome_tracks_cmd_wt = f"pyGenomeTracks --tracks {tracks_wt_ini} --region {args.chrom}:{locus_start}-{locus_end} --outFileName {args.out_file.replace('.tsv', '') + f'_WT_tracks_{args.chrom}_{locus_start}_{locus_end}.png'} --dpi 300 --fontSize {font_size} --plotWidth {plot_width} --trackLabelFraction {track_label_fraction}"
        pygenome_tracks_cmd_ko = f"pyGenomeTracks --tracks {tracks_ko_ini} --region {args.chrom}:{locus_start}-{locus_end} --outFileName {args.out_file.replace('.tsv', '') + f'_KO_{ko_short_label}_tracks_{args.chrom}_{locus_start}_{locus_end}.png'} --dpi 300 --fontSize {font_size} --plotWidth {plot_width} --trackLabelFraction {track_label_fraction}"
        os.system(pygenome_tracks_cmd_wt)
        os.system(pygenome_tracks_cmd_ko)
        print(f"pyGenomeTracks visualizations saved.")
        
    
    print("Done.")

if __name__ == '__main__':
    main()