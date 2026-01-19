import os
import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import lightning as pl
import matplotlib.pyplot as plt
import cooler
import pyBigWig
from tqdm import tqdm
from skimage.transform import resize

import cshark.model.corigami_models as corigami_models
import cshark.inference.utils.inference_utils as infer
from cshark.inference.utils.model_utils import get_all_track_names
from cshark.inference.utils.inference_utils import write_tmp_cooler, knockout_peaks

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
    parser.add_argument('--step-size', type=int, default=1000000, help='Step size for sliding window')
    parser.add_argument('--resolution', type=int, default=4096, help='Hi-C output resolution')
    parser.add_argument('--mat-size', type=int, default=512, help='Model output matrix size')
    parser.add_argument('--peak-height', type=float, default=0.5, help='Peak height threshold for predictions')
    parser.add_argument('--log-transform-bw', action='store_true', default=True, help='Log transform input bigwigs')
    parser.add_argument('--undo-log-hic', action='store_true', default=True, help='Undo log transform on Hi-C output')

    args = parser.parse_args()
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
    for track in all_tracks:
        if track not in ['ctcf', 'atac']:
            other_paths.append(args.ctcf.replace('ctcf.bw', f'{track}.bw'))  # Assuming similar naming
    
    if args.locus:
        locus_parts = args.locus.split(':')
        locus_chrom = locus_parts[0]
        locus_coords = locus_parts[1].split('-')
        locus_start = int(locus_coords[0].replace(',', ''))
        locus_end = int(locus_coords[1].replace(',', ''))
        if locus_chrom != args.chrom:
            raise ValueError(f"Locus chromosome {locus_chrom} does not match specified chromosome {args.chrom}.")
        chrom_len = locus_end - locus_start
        print(f"Using locus {args.locus} with length {chrom_len} bp.")
        starts = np.arange(locus_start - args.window, 
                           locus_end +1, 
                           args.step_size)
        ends = starts + args.window
    else:
        starts = np.arange(0, chrom_len - args.window, args.step_size)
        ends = starts + args.window
    
    print(f"Predicting on {args.chrom} ({chrom_len} bp) in {len(starts)} windows.")
    print(f"Hierarchical Mode: Predicting WT, then simulating CTCF KO -> RAD21 Pred -> Hi-C Pred.")

    # Storage for results
    # We will accumulate dataframes of pixels
    pixel_dfs = []
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
            output_wt = model(inputs_wt)
            pred_hic_wt = output_wt['hic'].squeeze().cpu().numpy()
            if args.undo_log_hic:
                pred_hic_wt = np.expm1(pred_hic_wt) # assuming log1p was used in training
            pred_hic_wt = (pred_hic_wt + pred_hic_wt.T) / 2.0  # Symmetrize
            pred_hic_wt = np.clip(pred_hic_wt, a_min=0, a_max=None)  # No negative counts

            # Knockout CTCF in input
            ctcf_ko_region = knockout_peaks(ctcf_region.copy(), threshold=args.peak_height)
            # Feed to model (Model infers RAD21 from zeroed CTCF KO)
            inputs_ko = infer.preprocess_default(seq_region, ctcf_ko_region, atac_region, other_feats).to(device)
            inputs_ko_without_rad21 = torch.cat([
                inputs_ko[:,:,:5 + rad21_idx],
                inputs_ko[:,:,6 + rad21_idx:]
            ], dim=2)
            pred_ko_inner = inner_model(inputs_ko_without_rad21)['1d']
            pred_ko_rad21 = torch.expm1(pred_ko_inner[:, :, 0]).squeeze()

            # Resample to match input resolution (bins)
            pred_ko_rad21 = F.interpolate(pred_ko_rad21.unsqueeze(0).unsqueeze(0),
                                          size=inputs_ko.shape[1],
                                          mode='linear', align_corners=True).squeeze()
            
            # Replace RAD21 channel (assumed to be after seq, ctcf, atac channels)
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

    print("Aggregating results...")
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
        axs[1].set_title('Predicted CTCF KO Hi-C')
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
    
    print("Done.")

if __name__ == '__main__':
    main()