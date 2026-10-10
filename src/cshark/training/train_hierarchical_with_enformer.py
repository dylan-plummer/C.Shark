import os
import sys
import math
import random
import wandb
import torch
import torch.nn.functional as F
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import lightning as pl
import lightning.pytorch.callbacks as callbacks
from lightning.pytorch.callbacks import Callback
from lightning.pytorch.loggers.wandb import WandbLogger
from lightning.pytorch.utilities import grad_norm
from skimage.transform import resize

import cshark.model.corigami_models as corigami_models
from cshark.data import genome_dataset
import cshark.inference.utils.inference_utils as infer
from cshark.inference.utils import plot_utils 
import cshark.data.data_feature as data_feature

from enformer_pytorch import from_pretrained
from enformer_pytorch.modeling_enformer import poisson_loss

ENFORMER_CONTEXT_LENGTH = 196_608
ENFORMER_TARGET_LEN = 896 * 128  # 114,688 bp
ENFORMER_TRIM = (ENFORMER_CONTEXT_LENGTH - ENFORMER_TARGET_LEN) // 2  # 40,960 bp to skip on left/right



def safe_corr(a, b):
  """Pearson r over two tensors, or None when it is undefined.

  ``torch.corrcoef`` returns NaN when either input has zero variance, which
  happens for any validation window sitting in a centromere/telomere gap or on a
  track that is flat there.  Lightning then averages that NaN into the epoch
  aggregate and the whole metric reads NaN -- which is why the correlation
  columns have been unusable during training even though the per-window values
  are fine.  Returning None lets the caller skip that window so the epoch mean is
  taken over the windows where the correlation is actually defined.
  """
  a = a.flatten().float()
  b = b.flatten().float()
  if a.numel() < 2 or b.numel() < 2:
      return None
  if not (torch.isfinite(a).all() and torch.isfinite(b).all()):
      a = torch.nan_to_num(a)
      b = torch.nan_to_num(b)
  am, bm = a - a.mean(), b - b.mean()
  denom = am.norm() * bm.norm()
  if denom <= 0 or not torch.isfinite(denom):
      return None
  r = (am @ bm) / denom
  return r if torch.isfinite(r) else None


class VizCallback(Callback):
    def __init__(self, data_root='cshark_data/data', celltypes=['gm12878'], assembly='hg19', assembly2=None,
                 image_scale=256, resolution=10000,
                 out_dir='deeploop_viz'):
        self.out_dir = out_dir
        os.makedirs(self.out_dir, exist_ok=True)
        self.data_root = data_root
        self.celltypes = celltypes
        self.assembly = assembly
        self.assembly2 = assembly2
        self.image_scale = image_scale  # size of each heatmap (fixed by model)
        self.resolution = resolution
        # self.loci = ['chr1:66000000', 'chr2:500000', 'chr3:145500000',
        #              'chr11:1500000', 'chr2:162000000',
        #              'chr10:122700000', 'chr15:59100000', 'chr12:89300000']
        self.loci = ['chr11:31000000', 'chr1:66000000', 'chr1:36000000', 'chr1:38000000']
        self.chr_names = [s.split(':')[0] for s in self.loci]
        self.starts = [int(s.split(':')[1]) for s in self.loci]
        self.seq = f"{self.data_root}/{self.assembly}/dna_sequence"
        self.seq2 = f"{self.data_root}/{assembly2}/dna_sequence" if assembly2 is not None else None
        # https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE167200
        self.ctcf = {celltype: f"{self.data_root}/{self.assembly}/{celltype}/genomic_features/ctcf.bw" for celltype in celltypes}
        #self.atac = {celltype: None for celltype in celltypes}  # for if we are not using ATAC
        self.atac = {celltype: f"{self.data_root}/{self.assembly}/{celltype}/genomic_features/atac.bw" for celltype in celltypes}
        # /mnt/rstor/genetics/JinLab/fxj45/WWW/ssz20/bigwig
        self.h3k27ac = {celltype: f"{self.data_root}/{self.assembly}/{celltype}/genomic_features/h3k27ac.bw" for celltype in celltypes}
        self.h3k4me3 = {celltype: f"{self.data_root}/{self.assembly}/{celltype}/genomic_features/h3k4me3.bw" for celltype in celltypes}
        # https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSM733679
        self.h3k36me3 = {celltype: f"{self.data_root}/{self.assembly}/{celltype}/genomic_features/h3k36me3.bw" for celltype in celltypes}
        # from here: /mnt/rstor/genetics/JinLab/fxj45/WWW/xww/bigwig
        self.h3k4me1 = {celltype: f"{self.data_root}/{self.assembly}/{celltype}/genomic_features/h3k4me1.bw" for celltype in celltypes}
        self.h3k27me3 = {celltype: f"{self.data_root}/{self.assembly}/{celltype}/genomic_features/h3k27me3.bw" for celltype in celltypes}
        self.rad21 = {celltype: f"{self.data_root}/{self.assembly}/{celltype}/genomic_features/rad21.bw" for celltype in celltypes}
        self.h3k9me3 = {celltype: f"{self.data_root}/{self.assembly}/{celltype}/genomic_features/h3k9me3.bw" for celltype in celltypes}

    def on_train_start(self, trainer, pl_module):
        print("Saving ground truth loci for reference")
        for celltype in self.celltypes:
            for chr_name, start in zip(self.chr_names, self.starts):
                locus = f"{chr_name}:{start}"
                if pl_module.hparams.predict_hic:
                    hic = data_feature.HiCFeature(path = f'{self.data_root}/{self.assembly}/{celltype}/hic_matrix/{chr_name}.npz')
                    mat = hic.get(start, res=self.resolution)
                    mat = resize(mat, (self.image_scale, self.image_scale), anti_aliasing=True, preserve_range=True)
                    os.makedirs(os.path.join(self.out_dir, locus), exist_ok=True)
                    plot = plot_utils.MatrixPlot(os.path.join(self.out_dir, locus), mat, 'ground_truth', celltype, 
                                        chr_name, start, res=self.resolution)
                    plot.plot()
                    tmp_plot_path = os.path.join(self.out_dir, locus, celltype, 'ground_truth', 'imgs', f"{chr_name}_{start}.png")
                    new_plot_path = os.path.join(self.out_dir, locus, celltype, f"ground_truth.png")
                    try:
                        os.rename(tmp_plot_path, new_plot_path)
                        if pl_module.hparams.use_wandb:
                            wandb.log({locus + '_experimental_' + celltype: wandb.Image(new_plot_path)})
                    except Exception as e:
                        print(e)

                # plot the ground truth 1D tracks
                if pl_module.hparams.output_features is not None:
                    os.makedirs(os.path.join(self.out_dir, locus, celltype, '1d_tracks'), exist_ok=True)
                    pred_1d_tracks = []
                    for i, feature in enumerate(pl_module.hparams.output_features):
                        bw = data_feature.GenomicFeature(path = f'{self.data_root}/{self.assembly}/{celltype}/genomic_features/{feature}.bw', norm=None)
                        pred_1d = bw.get(chr_name, start, start + pl_module.window)
                        #pred_1d = resize(pred_1d, (pl_module.hparams.target_1d_size,), anti_aliasing=True, preserve_range=True)
                        bin_size = int(len(pred_1d) / pl_module.hparams.target_1d_size)
                        pred_1d = pred_1d.reshape(-1, bin_size).mean(axis=1)
                        pred_1d_tracks.append(pred_1d)
                    # visualize 1D tracks as shaded plots
                    fig, axs = plt.subplots(len(pred_1d_tracks), 1, figsize=(10, len(pred_1d_tracks) * 2))
                    if len(pred_1d_tracks) == 1:
                        axs = [axs]
                    colors = ['blue', 'orange', 'green', 'red', 'purple', 'brown', 'pink', 'gray']
                    for i, pred_1d in enumerate(pred_1d_tracks):
                        track_name = pl_module.hparams.output_features[i]
                        axs[i].plot(pred_1d, color=colors[i % len(colors)])
                        axs[i].fill_between(range(len(pred_1d)), pred_1d, color=colors[i % len(colors)], alpha=0.5)
                        axs[i].set_title(track_name)
                        axs[i].set_xticks([])
                    plt.tight_layout()
                    plt.savefig(os.path.join(self.out_dir, locus, celltype, '1d_tracks', f"{chr_name}_{start}_ground_truth.png"))
                    plt.close()
                    try:
                        if pl_module.hparams.use_wandb:
                            wandb.log({locus + '_experimental_' + celltype + '_1d_tracks': wandb.Image(os.path.join(self.out_dir, locus, celltype, '1d_tracks', f"{chr_name}_{start}_ground_truth.png"))})
                    except Exception as e:
                        print(e)

    def _feature_path(self, celltype, feature):
        return f"{self.data_root}/{self.assembly}/{celltype}/genomic_features/{feature}.bw"


    def on_validation_epoch_end(self, trainer, pl_module):
        print("Evaluating is starting")
        for celltype in self.celltypes:
            for chr_name, start in zip(self.chr_names, self.starts):
                #try:
                locus = f"{chr_name}:{start}"
                # Build the feature channels from --input-features, in exactly that
                # order, and pass them all through ``other_paths``.  The ctcf/atac
                # arguments of load_region/preprocess_default are hardcoded to a
                # (ctcf, atac) channel pair, which mismatches the model whenever the
                # input set is not literally ['ctcf', 'atac'] -- e.g. an atac-only
                # run would be handed 7 channels for a 6-channel model.
                other_paths = [self._feature_path(celltype, f)
                               for f in pl_module.hparams.input_features]
                seq_region, _, _, other_regions = infer.load_region(chr_name,
                    start, self.seq, None, None, other_paths, seq2_path=self.seq2,
                    bigwig_log=pl_module.hparams.bigwig_log_transform)
                inputs = infer.preprocess_default(seq_region, None, None, other_regions)
                pl_module.model.eval()
                print('inputs shape:', inputs.shape)
                if pl_module.hparams.conditioning_vec is not None:
                    condition_vec_str = pl_module.hparams.conditioning_vec[self.celltypes.index(celltype)]
                    condition_vec = torch.tensor([float(x) for x in condition_vec_str.split(',')]).unsqueeze(0).to(inputs.device)
                    outputs = pl_module.model(inputs, condition_vec)
                else:
                    outputs = pl_module.model(inputs)
                if pl_module.hparams.predict_hic:
                    pred = outputs.get('hic')[0].detach().cpu().numpy()
                    print('pred shape:', pred.shape)
                    pred = (pred + pred.T) * 0.5
                    os.makedirs(os.path.join(self.out_dir, locus), exist_ok=True)
                    plot = plot_utils.MatrixPlot(os.path.join(self.out_dir, locus), pred, 'prediction', celltype, 
                                        chr_name, start, res=self.resolution)
                    plot.plot()
                    tmp_plot_path = os.path.join(self.out_dir, locus, celltype, 'prediction', 'imgs', f"{chr_name}_{start}.png")
                    new_plot_path = os.path.join(self.out_dir, locus, celltype, f"{pl_module.current_epoch}.png")
                    try:
                        os.rename(tmp_plot_path, new_plot_path)
                        if pl_module.hparams.use_wandb:
                            wandb.log({locus + '_' + celltype: wandb.Image(new_plot_path)})
                    except Exception as e:
                        print(e)

                # visualize rad21 prediction from inner_model (only when the
                # layer-2 RAD21 predictor exists, i.e. rad21 is an input feature)
                if getattr(pl_module, 'use_rad21', False) and pl_module.input_pred_model is not None:
                    inputs_without_rad21 = torch.cat([inputs[:, :, :5 + pl_module.hparams.input_features.index('rad21')],
                                                        inputs[:, :, 5 + pl_module.hparams.input_features.index('rad21') + 1:]], dim=2)
                    hierarchical_outputs = pl_module.input_pred_model(inputs_without_rad21)
                    pred_1d_inputs = hierarchical_outputs.get('1d')
                    track_pred = pred_1d_inputs[:, :, 0].detach().cpu().numpy().squeeze()
                    if pl_module.hparams.bigwig_log_transform:
                        track_pred = np.exp(track_pred) - 1  # inverse log transformation
                    os.makedirs(os.path.join(self.out_dir, locus, celltype, '1d_tracks'), exist_ok=True)
                    # visualize 1D tracks as shaded plots compared to ground truth
                    fig, axs = plt.subplots(2, 1, figsize=(10, 4))
                    colors = ['blue', 'orange']
                    axs[0].plot(track_pred, color=colors[0])
                    axs[0].fill_between(range(len(track_pred)), track_pred, color=colors[0], alpha=0.5)
                    axs[0].set_title('Predicted Rad21 Input Feature')
                    axs[0].set_xticks([])
                    # get ground truth rad21 from inputs
                    rad21_idx = pl_module.hparams.input_features.index('rad21')
                    gt_track = inputs[:, :, 5 + rad21_idx].clone().detach().cpu().numpy()
                    bin_size = int(len(track_pred) / pl_module.hparams.target_1d_size)
                    gt_track = gt_track.reshape(-1, bin_size).mean(axis=1)
                    if pl_module.hparams.bigwig_log_transform:
                        gt_track = np.exp(gt_track) - 1  # inverse log transformation
                    axs[1].plot(gt_track, color=colors[1])
                    axs[1].fill_between(range(len(gt_track)), gt_track, color=colors[1], alpha=0.5)
                    axs[1].set_title('Ground Truth Rad21 Input Feature')
                    axs[1].set_xticks([])
                    plt.tight_layout()
                    plt.savefig(os.path.join(self.out_dir, locus, celltype, '1d_tracks', f"{chr_name}_{start}_rad21_input_comparison_{pl_module.current_epoch}.png"))
                    plt.close()
                    try:
                        if pl_module.hparams.use_wandb:
                            wandb.log({locus + '_' + celltype + '_1d_tracks': wandb.Image(os.path.join(self.out_dir, locus, celltype, '1d_tracks', f"{chr_name}_{start}_rad21_input_comparison_{pl_module.current_epoch}.png"))})
                    except Exception as e:
                        print(e)


                # Layer 3 now predicts the --target-features 1D tracks alongside
                # Hi-C; visualize them when present.
                if pl_module.hparams.output_features is not None and outputs.get('1d') is not None:
                    pred_1d_tracks = outputs.get('1d')[0].permute(1, 0).detach().cpu().numpy()
                    os.makedirs(os.path.join(self.out_dir, locus, celltype, '1d_tracks'), exist_ok=True)
                    # visualize 1D tracks as shaded plots
                    fig, axs = plt.subplots(len(pred_1d_tracks), 1, figsize=(10, len(pred_1d_tracks) * 2))
                    if len(pred_1d_tracks) == 1:
                        axs = [axs]
                    colors = ['blue', 'orange', 'green', 'red', 'purple', 'brown', 'pink', 'gray']
                    for i, pred_1d in enumerate(pred_1d_tracks):
                        track_name = pl_module.hparams.output_features[i]
                        if pl_module.hparams.bigwig_log_transform:
                            pred_1d = np.exp(pred_1d) - 1  # inverse log transformation
                        axs[i].plot(pred_1d, color=colors[i % len(colors)])
                        axs[i].fill_between(range(len(pred_1d)), pred_1d, color=colors[i % len(colors)], alpha=0.5)
                        axs[i].set_title(track_name)
                        axs[i].set_xticks([])
                    plt.tight_layout()
                    plt.savefig(os.path.join(self.out_dir, locus, celltype, '1d_tracks', f"{chr_name}_{start}_{pl_module.current_epoch}.png"))
                    plt.close()
                    try:
                        if pl_module.hparams.use_wandb:
                            wandb.log({locus + '_' + celltype + '_1d_tracks': wandb.Image(os.path.join(self.out_dir, locus, celltype, '1d_tracks', f"{chr_name}_{start}_{pl_module.current_epoch}.png"))})
                    except Exception as e:
                        print(e)

                # predict tracks using enformer for comparison.  With split heads the
                # visualised celltype must go through its OWN layer-1 head, otherwise
                # every panel would show group 0's prediction.
                viz_head_group = None
                if getattr(pl_module, 'split_enformer_heads', False):
                    ct_i = list(pl_module.hparams.dataset_celltypes).index(celltype)
                    viz_head_group = int(pl_module.celltype_to_head_group[ct_i])
                enformer_pred_tracks = pl_module.enformer_predict_1d(inputs, head_group=viz_head_group)
                enformer_pred_tracks = enformer_pred_tracks[0].permute(1, 0).detach().cpu().numpy()
                os.makedirs(os.path.join(self.out_dir, locus, celltype, '1d_tracks'), exist_ok=True)
                # visualize 1D tracks as shaded plots
                fig, axs = plt.subplots(len(enformer_pred_tracks), 1, figsize=(10, len(enformer_pred_tracks) * 2))
                if len(enformer_pred_tracks) == 1:
                    axs = [axs]
                colors = ['blue', 'orange', 'green', 'red', 'purple', 'brown', 'pink', 'gray']
                for i, pred_1d in enumerate(enformer_pred_tracks):
                    track_name = pl_module.enformer_tracks[i]
                    axs[i].plot(pred_1d, color=colors[i % len(colors)])
                    axs[i].fill_between(range(len(pred_1d)), pred_1d, color=colors[i % len(colors)], alpha=0.5)
                    axs[i].set_title(track_name + ' (Enformer)')
                    axs[i].set_xticks([])
                plt.tight_layout()
                plt.savefig(os.path.join(self.out_dir, locus, celltype, '1d_tracks', f"{chr_name}_{start}_enformer_{pl_module.current_epoch}.png"))
                plt.close()
                try:
                    if pl_module.hparams.use_wandb:
                        wandb.log({locus + '_' + celltype + '_1d_tracks_enformer': wandb.Image(os.path.join(self.out_dir, locus, celltype, '1d_tracks', f"{chr_name}_{start}_enformer_{pl_module.current_epoch}.png"))})
                except Exception as e:
                    print(e)

                # predict Hi-C using enformer predictions of 1D tracks
                if pl_module.hparams.predict_hic:
                    inputs_with_enformer_tracks = inputs.clone()
                    # replace each Enformer-predicted track with its prediction (rad21 untouched)
                    for enf_i, track_name in enumerate(pl_module.enformer_tracks):
                        feat_idx = pl_module.hparams.input_features.index(track_name)
                        inputs_with_enformer_tracks[:, :, 5 + feat_idx] = torch.log1p(
                            torch.tensor(enformer_pred_tracks[enf_i]).to(inputs.device)
                        ).unsqueeze(0)
                    with torch.no_grad():
                        if pl_module.hparams.conditioning_vec is not None:
                            condition_vec_str = pl_module.hparams.conditioning_vec[self.celltypes.index(celltype)]
                            condition_vec = torch.tensor([float(x) for x in condition_vec_str.split(',')]).unsqueeze(0).to(inputs.device)
                            enformer_hic_output = pl_module.model(inputs_with_enformer_tracks, condition_vec)
                        else:
                            enformer_hic_output = pl_module.model(inputs_with_enformer_tracks)
                    enformer_hic_pred = enformer_hic_output.get('hic')[0].detach().cpu().numpy()
                    enformer_hic_pred = (enformer_hic_pred + enformer_hic_pred.T) * 0.5
                    os.makedirs(os.path.join(self.out_dir, locus), exist_ok=True)
                    plot = plot_utils.MatrixPlot(os.path.join(self.out_dir, locus), enformer_hic_pred, 'enformer_prediction', celltype,
                                        chr_name, start, res=self.resolution)
                    plot.plot()
                    tmp_plot_path = os.path.join(self.out_dir, locus, celltype, 'enformer_prediction', 'imgs', f"{chr_name}_{start}.png")
                    new_plot_path = os.path.join(self.out_dir, locus, celltype, f"enformer_{pl_module.current_epoch}.png")
                    try:
                        os.rename(tmp_plot_path, new_plot_path)
                        if pl_module.hparams.use_wandb:
                            wandb.log({locus + '_enformer_' + celltype: wandb.Image(new_plot_path)})
                    except Exception as e:
                        print(e)


def main():
    args = init_parser()
    init_training(args)
    if args.use_wandb:
        wandb.init(project='', entity='',
                config=args.__dict__)
        #wandb.watch(model, log_freq=2000)
        config = wandb.config

def init_parser(return_parser=False, parser=None, extend=None):
  """Build the trainer CLI.

  ``return_parser=True`` returns the configured ``ArgumentParser`` *before*
  parsing, so a downstream trainer (e.g. the layer-4 script) can add its own
  arguments to exactly this parser instead of duplicating every definition.
  ``extend`` is called with the parser just before parsing, for the same reason.
  """
  parser = parser or argparse.ArgumentParser(description='C.Origami Training Module.')

  # Data and Run Directories
  parser.add_argument('--seed', dest='run_seed', default=2077,
                        type=int,
                        help='Random seed for training')
  parser.add_argument('--save_path', dest='run_save_path', default='checkpoints',
                        help='Path to the model checkpoint')

  # Data directories
  parser.add_argument('--data-root', dest='dataset_data_root', default='data',
                        help='Root path of training data', required=True)
  parser.add_argument('--assembly', dest='dataset_assembly', default='hg19',
                        help='Genome assembly for training data')
  parser.add_argument('--assembly2', dest='dataset_assembly2', default=None,
                        help='Genome assembly for other assembly of double stranded training data')
  parser.add_argument('--alt-assemblies', dest='alt_assemblies', default=None, nargs='+',
                        help='Other genome assemblies for multi-assembly training (should match the celltype names)')
  # list of celltypes
  parser.add_argument('--celltypes', dest='dataset_celltypes', default=['alpha', 'beta'], nargs='+',
                        help='Cell types to train on')

  parser.add_argument('--conditions', dest='conditioning_vec', default=None, nargs='+',
                        help='Conditioning vector values for each cell type')

  # Model parameters
  parser.add_argument('--model-type', dest='model_type', default='MultiTaskConvTransModel',
                        help='CNN with Transformer')
  parser.add_argument('--checkpoint', dest='model_path', default=None,
                            help='start from a pretrained checkpoint')

  # Training Parameters
  parser.add_argument('--patience', dest='trainer_patience', default=80,
                        type=int,
                        help='Epoches before early stopping')
  parser.add_argument('--max-epochs', dest='trainer_max_epochs', default=120,
                        type=int,
                        help='Max epochs')
  parser.add_argument('--save-top-n', dest='trainer_save_top_n', default=5,
                        type=int,
                        help='Top n models to save')
  parser.add_argument('--num-gpu', dest='trainer_num_gpu', default=4,
                        type=int,
                        help='Number of GPUs to use')
  parser.add_argument('--use-wandb', dest='use_wandb',
                        action='store_true',
                        help='Track project on wandb')

  # Dataloader Parameters
  parser.add_argument('--batch-size', dest='dataloader_batch_size', default=4, 
                        type=int,
                        help='Batch size')
  parser.add_argument('--ddp-disabled', dest='dataloader_ddp_disabled',
                        action='store_false',
                        help='Using ddp, adjust batch size')
  parser.add_argument('--num-workers', dest='dataloader_num_workers', default=20,
                        type=int,
                        help='Dataloader workers')
  
  # add args for CTCF, ATAC, and other genomic features as either inputs, outputs, or both
  parser.add_argument('--input-features', dest='input_features', nargs='+',
                            help='Input features to use')
  parser.add_argument('--target-features', dest='output_features', nargs='+',
                            default=None,
                            help='Target features to use')
  parser.add_argument('--resolution', dest='resolution', type=int, default=10000,
                      help='Resolution (bp) of output Hi-C matrix')
  parser.add_argument('--matrix-size', dest='mat_size', type=int, default=256,
                      help='Size of output Hi-C matrix')
  parser.add_argument('--target-feature-size', dest='target_1d_size', type=int, default=8192,
                      help='Size of output 1d track')
  parser.add_argument('--latent-dim', dest='model_latent_dim', type=int, default=256,
                      help='Latent dimension size (mid_hidden)')
  parser.add_argument('--seq-filter-size', dest='seq_filter_size', type=int, default=3,
                      help='Size of 1D convolution filter for sequence input (bp)')
  parser.add_argument('--lr', dest='optimizer_lr', type=float, default=2e-4, help='Learning rate')
  parser.add_argument('--loss-weight-hic', dest='training_loss_weight_hic', type=float, default=1.0,
                      help='Weight for Hi-C loss term')
  parser.add_argument('--loss-weight-1d', dest='training_loss_weight_1d', type=float, default=0.1,
                      help='Weight for 1D track loss term')
  parser.add_argument('--no-hic', dest='predict_hic',
                        action='store_false',
                        help='Whether to predict Hi-C matrices or only 1D tracks')
  parser.add_argument('--no-recon', dest='recon_1d',
                        action='store_false',
                        help='Whether to reconstruct 1D tracks from full features or from sequence only')
  parser.add_argument('--no-hic-log-transform', dest='hic_log_transform',
                        action='store_false',
                        help='Whether to apply log transformation to Hi-C matrices')
  parser.add_argument('--no-bigwig-log-transform', dest='bigwig_log_transform',
                        action='store_false',
                        help='Whether to apply log transformation to BigWig tracks')
  parser.add_argument('--masking-prob', dest='training_masking_prob', type=float, default=0.0,
                        help='Probability of masking an input 1D track during training (0.0 to disable). Recommended: 0.15')
  parser.add_argument('--masking-min-chunk', dest='training_masking_min_chunk', type=int, default=256,
                    help='Minimum size (in bp) of a random mask chunk. Default is one output bin size.')
  parser.add_argument('--masking-max-chunk', dest='training_masking_max_chunk', type=int, default=10240,
                    help='Maximum size (in bp) of a random mask chunk. (e.g., 10kb)')

  # --- Hierarchical end-to-end training -------------------------------------
  # Which tracks the Enformer layer (layer 1) predicts from sequence.  Must be a
  # subset of --input-features and must NOT include rad21 (rad21 is layer 2's job).
  # Defaults to every input feature except rad21.
  parser.add_argument('--enformer-tracks', dest='enformer_tracks', nargs='+', default=None,
                      help='Tracks Enformer (layer 1) predicts from sequence. '
                           'Default: all --input-features except rad21.')
  # Layer 1 head sharing across celltypes.  By default a single head per track is
  # trained across everything in --celltypes, which is what you want when the
  # celltypes are strains/replicates of the SAME biological celltype.  When they are
  # genuinely different celltypes, splitting gives each its own head so layer 1 can
  # emit celltype-specific signal (and the downstream layers inherit it).  Only
  # layer 1 is ever split; layers 2-4 stay shared.
  parser.add_argument('--enformer-split-heads-by-celltype', dest='enformer_split_heads',
                      action='store_true',
                      help='Give each --celltypes entry its own set of Enformer (layer 1) '
                           'track heads, routed per sample during training. '
                           'Default OFF: one shared head per track across all celltypes.')
  parser.add_argument('--enformer-head-groups', dest='enformer_head_groups', nargs='+', default=None,
                      help='Explicit head grouping: one label per --celltypes entry. Celltypes '
                           'sharing a label share a head set, so strains of the same celltype '
                           'can pool while distinct celltypes split '
                           '(e.g. --celltypes 129_b6 cast_b6 beta_total '
                           '--enformer-head-groups alpha alpha beta). Implies splitting.')
  parser.add_argument('--enformer-load-pretrained-heads', dest='enformer_load_pretrained_heads',
                      action='store_true',
                      help='Initialise Enformer track heads from the pretrained Basenji head. '
                           'Default OFF: heads are trained from scratch.')
  parser.add_argument('--enformer-finetune-blocks', dest='enformer_finetune_blocks', type=int, default=1,
                      help='Number of trailing Enformer transformer blocks to fine-tune.')
  parser.add_argument('--enformer-mix-mode', dest='enformer_mix_mode', default='window',
                      choices=['full', 'window'],
                      help="How much of a track the Enformer's prediction is mixed into. "
                           "'window' (DEFAULT): only the single 114,688 bp region that "
                           "carries gradient -- the gentler hand-off, and the cheaper one. "
                           "'full': tile the whole 2Mb window and mix whole tracks; this "
                           "hands layer 3 an entirely predicted input, which is what made "
                           "the Hi-C head collapse when the curriculum started.")

  # Curriculum: pretrain each layer on ground-truth inputs, then ramp up the
  # probability of feeding an upstream layer's *prediction* into the next layer.
  parser.add_argument('--pretrain-epochs', dest='pretrain_epochs', type=int, default=20,
                      help='Epochs with mix probability = 0 (each layer learns on ground-truth inputs).')
  parser.add_argument('--ramp-epochs', dest='ramp_epochs', type=int, default=20,
                      help='Epochs over which the mix probability ramps linearly to --max-mix-prob.')
  parser.add_argument('--max-mix-prob', dest='max_mix_prob', type=float, default=0.5,
                      help='Maximum probability of feeding a predicted upstream track into the next layer.')
  parser.add_argument('--enformer-track-mix-prob', dest='enformer_track_mix_prob', type=float, default=0.5,
                      help='When an Enformer-mix step fires, probability each individual predicted track '
                           'replaces its ground-truth input.')

  # Per-layer loss weights and learning rates
  parser.add_argument('--loss-weight-enformer', dest='training_loss_weight_enformer', type=float, default=1.0,
                      help='Weight for the Enformer (layer 1) track-prediction loss.')
  parser.add_argument('--loss-weight-rad21', dest='training_loss_weight_rad21', type=float, default=1.0,
                      help='Weight for the RAD21 (layer 2) prediction loss.')
  parser.add_argument('--enformer-lr', dest='enformer_lr', type=float, default=1e-4,
                      help='Learning rate for the Enformer layer.')
  parser.add_argument('--rad21-lr', dest='rad21_lr', type=float, default=1e-4,
                      help='Learning rate for the RAD21 (layer 2) predictor.')


  # Hooks for downstream trainers: add their arguments to THIS parser (so the
  # whole CLI stays in one place) and, when they drive parsing themselves, hand
  # the parser back before anything is consumed.
  if extend is not None:
      extend(parser)
  if return_parser:
      return parser

  args = parser.parse_args(args=None if sys.argv[1:] else ['--help'])
  return finalize_args(args)


def finalize_args(args):
  """Fill in the derived defaults and validate the track sets.

  Kept separate from parsing so that a downstream trainer which drives
  ``parse_args`` itself (via ``init_parser(return_parser=True)``) applies exactly
  the same defaulting.  ``enformer_tracks`` in particular is derived here, and a
  trainer that skipped this step would construct TrainModule with it set to None.
  """
  if args.input_features is None:
      args.input_features = []
  # Default: Enformer predicts every input feature except rad21
  if getattr(args, 'enformer_tracks', None) is None:
      args.enformer_tracks = [f for f in args.input_features if f != 'rad21']
  # Validate the enformer track set
  if 'rad21' in args.enformer_tracks:
      raise ValueError("--enformer-tracks must not include 'rad21' (rad21 is predicted by layer 2).")
  missing = [t for t in args.enformer_tracks if t not in args.input_features]
  if missing:
      raise ValueError(f"--enformer-tracks {missing} are not in --input-features {args.input_features}.")

  # --- Layer 1 head grouping ------------------------------------------------
  # Resolve --enformer-split-heads-by-celltype / --enformer-head-groups into two
  # derived hparams that the module and any downstream loader can rely on:
  #   enformer_head_group_names : label per head set, in head order
  #   enformer_head_group_index : head-set index for each --celltypes entry
  celltypes = list(getattr(args, 'dataset_celltypes', None) or [])
  groups = getattr(args, 'enformer_head_groups', None)
  if groups is not None:
      if len(groups) != len(celltypes):
          raise ValueError(f"--enformer-head-groups has {len(groups)} labels but there are "
                           f"{len(celltypes)} --celltypes: {celltypes}.")
      args.enformer_split_heads = True
      labels = list(groups)
  elif getattr(args, 'enformer_split_heads', False):
      # One head set per celltype.
      labels = list(celltypes)
  else:
      labels = None

  if labels is None:
      args.enformer_head_group_names = None
      args.enformer_head_group_index = None
  else:
      # Preserve first-appearance order so head 0 belongs to the first celltype.
      names = list(dict.fromkeys(labels))
      args.enformer_head_group_names = names
      args.enformer_head_group_index = [names.index(l) for l in labels]
      if len(names) == 1:
          print(f'[hierarchical] Enformer head split requested but all celltypes map to a '
                f'single group {names!r}; this is equivalent to the shared-head default.')
  return args

def init_training(args):

    # Early_stopping
    early_stop_callback = callbacks.EarlyStopping(monitor='val_loss', 
                                        min_delta=0.00, 
                                        patience=args.trainer_patience,
                                        verbose=False,
                                        mode="min")
    # Checkpoints
    checkpoint_callback = callbacks.ModelCheckpoint(dirpath=f'{args.run_save_path}/models',
                                        save_top_k=args.trainer_save_top_n, 
                                        monitor='val_loss')

    # LR monitor
    lr_monitor = callbacks.LearningRateMonitor(logging_interval='epoch')

    # Logger
    #csv_logger = pl.loggers.CSVLogger(save_dir = f'{args.run_save_path}/csv')
    #all_loggers = csv_logger
    
    # Assign seed
    pl.seed_everything(args.run_seed, workers=True)
    pl_module = TrainModule(args)
    if args.use_wandb:
        wandb_logger = WandbLogger(project='c.shark')
        wandb_logger.watch(pl_module.model)
    pl_trainer = pl.Trainer(strategy='ddp_find_unused_parameters_true',
                            precision='bf16-mixed',
                            accelerator="gpu" if torch.cuda.is_available() else "cpu", devices=args.trainer_num_gpu,
                            gradient_clip_val=1,
                            logger = wandb_logger if args.use_wandb else None,
                            callbacks = [VizCallback(data_root=args.dataset_data_root,
                                                     celltypes=args.dataset_celltypes,  
                                                     assembly=args.dataset_assembly,
                                                     image_scale=args.mat_size,
                                                     resolution=args.resolution,
                                                     assembly2=args.dataset_assembly2,),
                                         early_stop_callback,
                                         checkpoint_callback,
                                         lr_monitor],
                            max_epochs = args.trainer_max_epochs
                            )
    trainloader = pl_module.get_dataloader(args, 'train')
    valloader = pl_module.get_dataloader(args, 'val')
    testloader = pl_module.get_dataloader(args, 'test')

    # for test_batch_i in range(1):
    #     # load a batch and visualize it for debugging
    #     batch = next(iter(trainloader))
    #     inputs, mat, target_1d_tracks, _ = pl_module.proc_batch(batch)
    #     #print('inputs shape:', inputs.shape) # (batch, window, 5 + num_genomic_features)
    #     #print('mat shape:', mat.shape)  # (batch, image_scale, image_scale)
    #     #print('target_1d_tracks shape:', target_1d_tracks.shape if target_1d_tracks is not None else None)

    #     colors = ['blue', 'orange', 'green', 'red', 'purple', 'brown', 'pink', 'gray']
    #     if len(args.input_features) > 0:
    #         # visualize the input genomic features
    #         genomic_features = inputs[:, :, 5:]
    #         genomic_features = genomic_features[0].detach().cpu().numpy() 
    #         #genomic_features = resize(genomic_features, (pl_module.hparams.target_1d_size,), anti_aliasing=True, preserve_range=True)
    #         fig, axs = plt.subplots(genomic_features.shape[1], 1, figsize=(15, 4))  
    #         if genomic_features.shape[1] == 1:
    #             axs = [axs]
            
    #         for i in range(genomic_features.shape[1]):
    #             if args.bigwig_log_transform:
    #                 track = np.exp(genomic_features[:, i]) - 1  # inverse log transformation
    #             else:
    #                 track = genomic_features[:, i]
    #             bin_size = int(len(track) / pl_module.hparams.target_1d_size)
    #             track = track.reshape(-1, bin_size).mean(axis=1)
    #             axs[i].plot(track, color=colors[i % len(colors)])
    #             axs[i].fill_between(range(len(track)), track, color=colors[i % len(colors)], alpha=0.5)
    #         plt.savefig(f'input_genomic_features.png_{test_batch_i}.png')
    #         plt.close()

    #     # visualize the target Hi-C matrix
    #     if args.predict_hic:
    #         mat = mat[0].detach().cpu().numpy()
    #         mat = resize(mat, (pl_module.hparams.target_1d_size, pl_module.hparams.target_1d_size), anti_aliasing=True, preserve_range=True)
    #         plt.imshow(mat, cmap='Reds', interpolation='none')
    #         plt.colorbar()
    #         plt.title('Target Hi-C Matrix')
    #         plt.savefig(f'target_hic_matrix.png_{test_batch_i}.png')
    #         plt.close()

    #     # visualize the target 1D tracks
    #     if target_1d_tracks is not None:
    #         target_1d_tracks = target_1d_tracks[0].detach().cpu().numpy()
    #         #target_1d_tracks = resize(target_1d_tracks, (pl_module.hparams.target_1d_size,), anti_aliasing=True, preserve_range=True)
    #         fig, axs = plt.subplots(target_1d_tracks.shape[1], 1, figsize=(15, 4))
    #         if target_1d_tracks.shape[1] == 1:
    #             axs = [axs]
    #         for i in range(target_1d_tracks.shape[1]):
    #             if args.bigwig_log_transform:
    #                 track = np.exp(target_1d_tracks[:, i]) - 1  # inverse log transformation
    #             else:
    #                 track = target_1d_tracks[:, i]
    #             axs[i].plot(track, color=colors[i % len(colors)])
    #             axs[i].fill_between(range(len(target_1d_tracks)), track, color=colors[i % len(colors)], alpha=0.5)
    #             #axs[i].set_ylim(0, 11)
    #         plt.title('Target 1D Tracks')
    #         plt.savefig(f'target_1d_tracks.png_{test_batch_i}.png')
    #         plt.close()

    pl_trainer.fit(pl_module, train_dataloaders=trainloader, val_dataloaders=valloader)

    

class TrainModule(pl.LightningModule):
    
    def __init__(self, args):
        super().__init__()
        self.save_hyperparameters(args)
        self.predict_1d = self.hparams.output_features is not None
        self.model = self.get_model(args)
        self.args = args
        self.criterion = torch.nn.MSELoss() # Common loss function
        self.window = 2097152 # 2Mb window size

        # --- Hierarchy bookkeeping ------------------------------------------
        # Layer 1 (Enformer) predicts these tracks from sequence.
        self.enformer_tracks = list(self.hparams.enformer_tracks)
        # Layer 2 (the RAD21 predictor) is OPTIONAL. It is only built and trained
        # when rad21 is one of the input features. Without rad21 the hierarchy
        # collapses to Enformer (layer 1) -> Hi-C (layer 3): Enformer fine-tunes on
        # whatever tracks are present and they feed straight into the Hi-C model.
        self.use_rad21 = 'rad21' in self.hparams.input_features
        self.rad21_idx = self.hparams.input_features.index('rad21') if self.use_rad21 else None
        # Absolute channel indices (5 sequence channels precede the 1D tracks).
        self.enformer_gt_channels = [5 + self.hparams.input_features.index(t)
                                     for t in self.enformer_tracks]
        self.enformer_input_idx = [self.hparams.input_features.index(t)
                                   for t in self.enformer_tracks]

        # Precompute the Enformer tiling grid over the 2Mb window.  Each tile
        # covers ENFORMER_TARGET_LEN output bp; the last tile is clipped so it
        # stays in-bounds (slight overlap with the penultimate tile).
        num_tiles = math.ceil(self.window / ENFORMER_TARGET_LEN)
        max_start = self.window - ENFORMER_TARGET_LEN
        self.enformer_tile_starts = [min(k * ENFORMER_TARGET_LEN, max_start)
                                     for k in range(num_tiles)]

        model_name =  args.model_type
        ModelClass = getattr(corigami_models, model_name)
        if self.use_rad21:
            self.input_pred_model = ModelClass(
                num_genomic_features=len(self.hparams.input_features) - 1, # Input features minus rad21
                num_target_tracks=1,    # Target 1D tracks
                conditioning_vec_size=len(self.hparams.conditioning_vec[0].split(',')) if self.hparams.conditioning_vec is not None else None,
                mid_hidden=self.hparams.model_latent_dim,
                predict_hic=False,
                diploid=args.dataset_assembly2 is not None,
                predict_1d=True,  # layer 2 always predicts the rad21 track; matches the
                                  # inference RAD21 loader (load_hierarchical_rad21_predictor)
                target_mat_size=args.mat_size,
                target_1d_length=args.target_1d_size,
                recon_1d=args.recon_1d,
                seq_filter_size=args.seq_filter_size,
                activation_1d=None
            )
        else:
            self.input_pred_model = None
            print('[hierarchical] No rad21 input feature: skipping the layer-2 RAD21 '
                  'predictor; training Enformer (layer 1) -> Hi-C (layer 3) directly.')
        self.species = 'human' if 'hg' in args.dataset_assembly else 'mouse'
        # ctcf atac rad21 h3k27ac h3k4me3 h3k9me3 h3k36me3 h3k27me3
        if self.species == 'human':
            self.target_map ={
                'ctcf': 'CTCF:H1-hESC', 
                'atac': 'DNASE:H1-hESC', 
                'rad21': 'CHIP:RAD21:H1-hESC',
                'h3k27ac': 'CHIP:H3K27ac:H1-hESC',
                'h3k4me3': 'CHIP:H3K4me3:H1-hESC',
                'h3k9me3': 'CHIP:H3K9me3:H1-hESC',
                'h3k36me3': 'CHIP:H3K36me3:H1-hESC',
                'h3k27me3': 'CHIP:H3K27me3:H1-hESC'
            }
        else:
            self.target_map = {
                'ctcf': 'CHIP:CTCF:C57BL/6 ES-Bruce4',
                'atac': 'DNASE:129 ES-CJ7',
                'rad21': 'CHIP:RAD21:DBA/2 MEL cell line',
                'h3k27ac': 'CHIP:H3K27ac:C57BL/6 ES-Bruce4',
                'h3k4me3': 'CHIP:H3K4me3:C57BL/6 ES-Bruce4',
                'h3k9me3': 'CHIP:H3K9me3:C57BL/6 ES-Bruce4',
                'h3k36me3': 'CHIP:H3K36me3:C57BL/6 ES-Bruce4',
                'h3k27me3': 'CHIP:H3K27me3:C57BL/6 ES-Bruce4'
            }
        # --- Layer 1 head grouping -----------------------------------------
        # ``enformer_head_group_index[c]`` is the head set used by celltype c of
        # --celltypes; None means one shared head set (the default).  Only layer 1
        # splits -- layers 2/3/4 stay shared, so celltype-specific tracks coming out
        # of layer 1 are what makes their outputs celltype-specific.
        group_index = getattr(self.hparams, 'enformer_head_group_index', None)
        self.split_enformer_heads = bool(group_index) and len(set(group_index)) > 1
        if self.split_enformer_heads:
            self.enformer_head_group_names = list(self.hparams.enformer_head_group_names)
            self.num_enformer_head_groups = len(self.enformer_head_group_names)
            # Lookup table celltype index -> head group index, kept as a buffer so it
            # follows the module across devices.
            self.register_buffer('celltype_to_head_group',
                                 torch.tensor(group_index, dtype=torch.long),
                                 persistent=False)
            print(f'[hierarchical] Layer-1 Enformer heads split into '
                  f'{self.num_enformer_head_groups} groups '
                  f'{self.enformer_head_group_names} over celltypes '
                  f'{list(self.hparams.dataset_celltypes)}.')
        else:
            self.enformer_head_group_names = None
            self.num_enformer_head_groups = 1
            self.celltype_to_head_group = None

        # Layer 1: Enformer predicts the (non-rad21) input tracks from sequence.
        # Heads are trained from scratch by default (load_pretrained=False).
        self.enformer = self.get_hESC_wrapper(
            target_tracks=self.enformer_tracks,
            load_pretrained=self.hparams.enformer_load_pretrained_heads,
            finetune_blocks=self.hparams.enformer_finetune_blocks,
            num_head_groups=self.num_enformer_head_groups,
        )

    @staticmethod
    def get_target_indices(species: str, target: str) -> np.ndarray:
        """Fetches and returns the numerical indices for a given target description."""
        targets_file = f"https://raw.githubusercontent.com/calico/basenji/master/manuscripts/cross2020/targets_{species}.txt"
        targets_df = pd.read_csv(targets_file, sep='\t')
        # reset index to be 0-based
        targets_df['index'] = targets_df.index
        target_mask = targets_df['description'].str.contains(target, case=False)
        print(targets_df[target_mask])
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

    def get_hESC_wrapper(self, target_tracks=['ctcf', 'atac'], load_pretrained=False,
                         finetune_blocks=1, num_head_groups=1):
        # Load the pre-trained model
        enformer = from_pretrained('EleutherAI/enformer-official-rough',
                           use_tf_gamma=True)
        self.freeze_all_but_last_n_layers_(enformer, n=finetune_blocks)  # Fine-tune trailing blocks
        # 1. Get Indices for specific tracks
        hesc_indices = []
        for track in target_tracks:
            try:
                target_desc = self.target_map[track]
                indices = self.get_target_indices(self.species, target_desc)
                hesc_indices.append(indices[0])
            except KeyError:  # track no found, raise error if using petrained, fine if not
                if load_pretrained:
                    raise ValueError(f"Track '{track}' not found in target map for species '{self.species}'.")
                else:
                    print(f"Warning: Track '{track}' not found in target map for species '{self.species}'. Skipping.")
                    target_desc = self.target_map['ctcf']
                    indices = self.get_target_indices(self.species, target_desc)
                    hesc_indices.append(indices[0])
                
        print(f'Initialized with hESC track indices: {hesc_indices} for {target_tracks}')

        # 2. Define the Adapter Class
        class HESCHeadAdapterWrapper(torch.nn.Module):
            """Enformer trunk + a small set of learnable per-track heads.

            With ``num_head_groups == 1`` (the default) there is one head per track,
            shared across every celltype -- the right choice when --celltypes holds
            strains/replicates of the same biological celltype.

            With ``num_head_groups > 1`` each group owns a full set of track heads and
            ``forward`` routes each sample to its group's heads, so layer 1 can learn
            celltype-specific signal.  The single-group parameter layout is left
            byte-identical to the original (``to_tracks.<track>``, 1-D ``scale``/``bias``)
            so existing shared-head checkpoints still load unchanged.
            """

            def __init__(self, enformer, hesc_indices, species='human', load_pretrained=True,
                         num_head_groups=1):
                super().__init__()
                self.enformer = enformer
                self.hesc_indices = hesc_indices
                self.species = species
                self.load_pretrained = load_pretrained
                self.num_head_groups = max(1, int(num_head_groups))
                self.split_heads = self.num_head_groups > 1
                self.num_tracks = len(hesc_indices)

                # Enformer trunk output is dim * 2 (e.g., 1536 * 2 = 3072)
                embedding_dim = enformer.dim * 2

                if self.split_heads:
                    # One independent head set per group: to_tracks[group][track]
                    self.to_tracks = torch.nn.ModuleList([
                        torch.nn.ModuleList([
                            torch.nn.Linear(in_features=embedding_dim, out_features=1)
                            for _ in hesc_indices
                        ])
                        for _ in range(self.num_head_groups)
                    ])
                    self.scale = torch.nn.Parameter(torch.ones(self.num_head_groups, self.num_tracks))
                    self.bias = torch.nn.Parameter(torch.zeros(self.num_head_groups, self.num_tracks))
                else:
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

                    # Copy weights/biases for the specific indices we care about.
                    # Every group starts from the same pretrained row; they diverge
                    # during training as each sees only its own celltype's samples.
                    with torch.no_grad():
                        for i, original_idx in enumerate(hesc_indices):
                            # Copy the specific row from the weight matrix
                            # Original shape: [5313, 3072] -> Extract row [3072] -> Unsqueeze to [1, 3072]
                            w = original_human_linear.weight.data[original_idx].unsqueeze(0).clone()
                            # Copy the specific bias
                            b = original_human_linear.bias.data[original_idx].unsqueeze(0).clone()
                            for head in self._track_heads_for_group_list(i):
                                head.weight.data = w.clone()
                                head.bias.data = b.clone()

                # Enformer uses Softplus activation for the final output
                self.activation = torch.nn.Softplus()

            def _track_heads_for_group_list(self, track_i):
                """Every group's head for one track (used by the pretrained copy)."""
                if self.split_heads:
                    return [self.to_tracks[g][track_i] for g in range(self.num_head_groups)]
                return [self.to_tracks[track_i]]

            def _project(self, embeddings, group):
                """Apply one group's heads to ``embeddings`` -> (B, L, num_tracks)."""
                heads = self.to_tracks[group] if self.split_heads else self.to_tracks
                track_preds = []
                for track_i, linear_layer in enumerate(heads):
                    # Project embeddings: (batch, seq_len, 1)
                    track_output = linear_layer(embeddings)
                    # Apply learnable scale and bias
                    if self.split_heads:
                        scale, bias = self.scale[group, track_i], self.bias[group, track_i]
                    else:
                        scale, bias = self.scale[track_i], self.bias[track_i]
                    track_preds.append(track_output * scale + bias)

                # Concatenate to (batch, seq_len, num_selected_tracks)
                return torch.cat(track_preds, dim=-1)

            def forward(self, x, head_group=None):
                """Forward pass.

                ``head_group`` selects which head set to use and is ignored unless the
                heads are split.  It may be a single int (whole batch shares a group)
                or a 1-D tensor of per-sample group indices; rows are routed to their
                own group's heads and reassembled in the original batch order.
                """
                # Efficiently get embeddings without computing all 5313 original heads
                # embeddings shape: (batch, seq_len, 3072)
                embeddings = self.enformer(x, return_only_embeddings=True)

                if not self.split_heads:
                    # Apply Softplus to ensure positive values (required for Poisson loss)
                    return self.activation(self._project(embeddings, None))

                if head_group is None:
                    raise ValueError(
                        'Enformer heads are split by celltype but no head_group was passed. '
                        'Every call site must supply the sample celltype/group.')

                if not torch.is_tensor(head_group):
                    head_group = torch.as_tensor(head_group, device=embeddings.device)
                head_group = head_group.reshape(-1).long()

                if head_group.numel() == 1:
                    # Whole batch shares one group: no scatter needed.
                    return self.activation(self._project(embeddings, int(head_group.item())))

                if head_group.numel() != embeddings.shape[0]:
                    raise ValueError(f'head_group has {head_group.numel()} entries for a batch of '
                                     f'{embeddings.shape[0]}.')

                # Route each group's rows through its own heads, then reassemble.  The
                # loop runs at most num_head_groups times and the heads are tiny, so the
                # cost next to the trunk forward is negligible.
                out = None
                for g in head_group.unique().tolist():
                    rows = (head_group == g).nonzero(as_tuple=True)[0]
                    sub = self._project(embeddings[rows], int(g))
                    if out is None:
                        out = sub.new_zeros((embeddings.shape[0],) + tuple(sub.shape[1:]))
                    # index_copy (out-of-place) keeps this autograd-safe.
                    out = out.index_copy(0, rows, sub)
                return self.activation(out)

        # 3. Instantiate and return
        if load_pretrained:
            print('[enformer] Initialising track heads from the pretrained Basenji head.')
        else:
            print('[enformer] Initialising track heads FROM SCRATCH (no pretrained head copy).')
        if num_head_groups > 1:
            print(f'[enformer] Splitting layer-1 track heads into {num_head_groups} celltype '
                  f'groups: {self.hparams.enformer_head_group_names}')
        return HESCHeadAdapterWrapper(enformer, hesc_indices, species=self.species,
                                      load_pretrained=load_pretrained,
                                      num_head_groups=num_head_groups).to(self.device)
        

    def layer3_1d_targets(self):
        """``(feature_names, column_indices)`` of layer 3's OWN 1D targets.

        Defaults to every ``--target-features`` track, which is the historical
        behaviour.  A subclass that owns a downstream layer overrides this so its
        targets are not also fit here -- see the layer-4 trainer, where having
        both layers predict transcription put two layers on the same task and
        pulled layer 3's latent away from Hi-C.
        """
        feats = getattr(self, 'layer3_target_features', None) \
            or list(self.hparams.output_features)
        idx = getattr(self, 'layer3_target_indices', None) \
            or list(range(len(self.hparams.output_features)))
        return feats, idx

    def get_model(self, args):
        model_name =  args.model_type
        ModelClass = getattr(corigami_models, model_name)
        num_input_features = 0
        num_input_features = len(self.hparams.input_features)

        num_target_tracks = 0
        if self.predict_1d:
            num_target_tracks = len(self.layer3_1d_targets()[0])
        print(f'Number of input genomic features: {num_input_features}')
        print(f'Number of target 1D tracks: {num_target_tracks}')

        # Instantiate the model. Layer 3 (the Hi-C model) respects --target-features:
        # when output features are requested it also reconstructs/predicts those 1D
        # tracks alongside Hi-C, exactly like the standard 8-track architecture.
        model = ModelClass(
            num_genomic_features=num_input_features, # Input features
            num_target_tracks=num_target_tracks,
            conditioning_vec_size=len(self.hparams.conditioning_vec[0].split(',')) if self.hparams.conditioning_vec is not None else None,
            mid_hidden=self.hparams.model_latent_dim,
            predict_hic=self.hparams.predict_hic,
            diploid=args.dataset_assembly2 is not None,
            predict_1d=self.predict_1d,
            target_mat_size=args.mat_size,
            target_1d_length=args.target_1d_size,
            recon_1d=args.recon_1d,
            seq_filter_size=args.seq_filter_size,
            activation_1d=None
        )
        if args.model_path is not None:
            checkpoint = torch.load(args.model_path, map_location='cuda' if torch.cuda.is_available() else 'cpu')
            model_weights = checkpoint['state_dict']

            # Edit keys
            for key in list(model_weights):
                model_weights[key.replace('model.', '')] = model_weights.pop(key)
            model.load_state_dict(model_weights)
        return model

    def forward(self, x, conditioning_vec=None):
        return self.model(x, conditioning_vec=conditioning_vec)

    def proc_batch(self, batch):
        """Unpack a batch into ``(inputs, mat, target_1d_tracks, condition_vec, celltype_idx)``.

        The dataset appends its optional fields to the END of the tuple, in this
        order: conditioning vector, then celltype index.  Strip them off first so
        the positional unpacking below only has to deal with the core fields.
        """
        batch = list(batch)
        celltype_idx = batch.pop() if self.split_enformer_heads else None
        batch = tuple(batch)  # condition_vec, if present, is now last again

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
        if celltype_idx is not None:
            celltype_idx = torch.as_tensor(celltype_idx).reshape(-1).long().to(inputs.device)
        return inputs, mat, target_1d_tracks, condition_vec, celltype_idx

    # ------------------------------------------------------------------
    # Curriculum schedule
    # ------------------------------------------------------------------
    def current_mix_prob(self):
        """Probability of feeding a *predicted* upstream track into the next layer.

        0 during the pretrain phase (each layer learns on ground-truth inputs),
        then ramps linearly to ``--max-mix-prob`` over ``--ramp-epochs``.
        """
        epoch = self.current_epoch
        pre = self.hparams.pretrain_epochs
        ramp = max(1, self.hparams.ramp_epochs)
        if epoch < pre:
            return 0.0
        if epoch >= pre + ramp:
            return self.hparams.max_mix_prob
        return self.hparams.max_mix_prob * (epoch - pre) / ramp

    # ------------------------------------------------------------------
    # Layer 1 head routing
    # ------------------------------------------------------------------
    def head_group_for(self, celltype_idx):
        """Map per-sample celltype indices to per-sample layer-1 head groups.

        Returns None when the heads are shared, which every call site forwards
        straight through to the wrapper (where it is ignored).
        """
        if not self.split_enformer_heads or celltype_idx is None:
            return None
        lut = self.celltype_to_head_group.to(celltype_idx.device)
        return lut[celltype_idx]

    # ------------------------------------------------------------------
    # Enformer tiling (memory-efficient, single-window gradient)
    # ------------------------------------------------------------------
    def _enformer_forward_tile(self, inputs, tile_start, use_grad, head_group=None):
        """Run Enformer on one ENFORMER_CONTEXT_LENGTH window whose *output*
        region begins at ``tile_start``.  Returns predictions in **linear**
        space, shape (B, 896, num_enformer_tracks).

        Only the sequence channels (first 4) are used, reordered ATCG->ACGT and
        padded with 0.25 at the genome edges, matching the inference pipeline.
        """
        ctx_start = tile_start - ENFORMER_TRIM
        ctx_end = ctx_start + ENFORMER_CONTEXT_LENGTH
        input_start = max(ctx_start, 0)
        input_end = min(ctx_end, inputs.shape[1])
        win = inputs[:, input_start:input_end, :4]
        if ctx_start < 0:
            win = F.pad(win, (0, 0, -ctx_start, 0), "constant", 0.25)
        if win.shape[1] < ENFORMER_CONTEXT_LENGTH:
            win = F.pad(win, (0, 0, 0, ENFORMER_CONTEXT_LENGTH - win.shape[1]), "constant", 0.25)
        win = win[:, :, [0, 2, 3, 1]].float()  # ATCG -> ACGT
        grad_ctx = torch.enable_grad() if use_grad else torch.no_grad()
        with grad_ctx:
            # head_group routes each row to its celltype's heads (ignored when shared).
            out = self.enformer(win, head_group=head_group)  # (B, 896, K), linear space (Softplus)
        return out

    def _tile_to_full(self, tile_out_linear, length):
        """Upsample a single tile's (B, 896, K) linear prediction to
        (B, ENFORMER_TARGET_LEN, K) in **log1p** space (model-input space)."""
        out_log = torch.log1p(tile_out_linear)
        up = F.interpolate(out_log.permute(0, 2, 1), size=ENFORMER_TARGET_LEN,
                           mode='linear', align_corners=True).permute(0, 2, 1)
        return up

    def enformer_assemble_full(self, inputs, grad_tile_idx=None, head_group=None):
        """Build a full-window (B, L, K) Enformer prediction in **log1p** space
        by tiling the 2Mb window.

        The tile at ``grad_tile_idx`` carries gradients; every other tile runs
        under ``no_grad`` (so peak memory ~= a single Enformer forward pass).
        Pass ``grad_tile_idx=None`` for a fully detached prediction (validation
        / visualisation).

        Returns
        -------
        full_log1p : (B, L, K) tensor, log1p space
        grad_out   : (B, 896, K) linear tile output at ``grad_tile_idx`` (or None)
        grad_start : output start bp of that tile (or None)
        """
        L = inputs.shape[1]
        # Build the full track as a concatenation of NON-overlapping segments
        # (avoids in-place writes into a graph tensor).  Each tile k owns the
        # segment [k*TARGET_LEN, (k+1)*TARGET_LEN); the last tile's window start
        # is clipped in-bounds, so we slice out its (offset) tail.
        segments = []
        grad_out, grad_start = None, None
        for k, ts in enumerate(self.enformer_tile_starts):
            seg_start = k * ENFORMER_TARGET_LEN
            if seg_start >= L:
                break
            seg_end = min(seg_start + ENFORMER_TARGET_LEN, L)
            offset = seg_start - ts  # 0 except for the clipped final tile
            use_grad = (grad_tile_idx is not None and k == grad_tile_idx)
            out = self._enformer_forward_tile(inputs, ts, use_grad, head_group=head_group)
            if use_grad:
                up = self._tile_to_full(out, L)
                segments.append(up[:, offset:offset + (seg_end - seg_start), :])
                grad_out, grad_start = out, ts
            else:
                with torch.no_grad():
                    up = self._tile_to_full(out, L)
                    segments.append(up[:, offset:offset + (seg_end - seg_start), :])
        full = torch.cat(segments, dim=1)  # (B, L, K)
        return full, grad_out, grad_start

    def enformer_window_loss(self, inputs, tile_out_linear, tile_start):
        """MSE (in log1p space) between the Enformer prediction for one tile and
        the ground-truth tracks over that tile's output region."""
        gt = inputs[:, tile_start:tile_start + ENFORMER_TARGET_LEN, self.enformer_gt_channels]
        gt_ds = F.interpolate(gt.permute(0, 2, 1).float(), size=tile_out_linear.shape[1],
                              mode='linear', align_corners=True).permute(0, 2, 1)
        pred_log = torch.log1p(tile_out_linear)
        return F.mse_loss(pred_log, gt_ds)

    def enformer_predict_1d(self, inputs, head_group=None):
        """Full-window Enformer prediction in **linear** space (B, L, K).

        Kept for VizCallback / diagnostics: fully detached, no gradient.
        """
        full_log1p, _, _ = self.enformer_assemble_full(inputs, grad_tile_idx=None,
                                                       head_group=head_group)
        return torch.expm1(full_log1p)


    def training_step(self, batch, batch_idx):
        """End-to-end hierarchical step.

        Every step trains all three layers with a supervised loss:
          * layer 1 (Enformer)  -> predicts non-rad21 tracks from sequence
          * layer 2 (RAD21)     -> predicts rad21 from seq + non-rad21 tracks
          * layer 3 (Hi-C)      -> predicts Hi-C from seq + all tracks

        A curriculum probability (``current_mix_prob``) controls whether an
        upstream layer's *prediction* is fed forward as the next layer's input,
        chaining the gradient all the way through:
            Hi-C loss -> layer3 -> (predicted rad21) -> layer2 -> (predicted
            tracks) -> layer1.

        Memory: the Enformer contributes gradients through a single randomly
        chosen tile per step (``enformer_assemble_full`` runs the rest under
        no_grad), so peak memory stays ~= one Enformer forward pass.
        """
        inputs, mat, target_1d_tracks, condition_vec, celltype_idx = self.proc_batch(batch)
        L = inputs.shape[1]
        abs_rad21 = 5 + self.rad21_idx if self.use_rad21 else None
        head_group = self.head_group_for(celltype_idx)
        mix_prob = self.current_mix_prob()
        self.log('mix_prob', mix_prob, on_step=False, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)

        # ---- Layer 1: Enformer -------------------------------------------
        # One random tile carries the gradient; used both for the supervised
        # loss and (when mixing) as the differentiable region fed downstream.
        grad_tile_idx = random.randrange(len(self.enformer_tile_starts))
        use_enformer_mix = random.random() < mix_prob
        # Full-window tiling is only needed when mixing whole tracks forward.
        need_full = use_enformer_mix and self.hparams.enformer_mix_mode == 'full'

        if need_full:
            enf_full_log1p, grad_out, grad_start = self.enformer_assemble_full(
                inputs, grad_tile_idx=grad_tile_idx, head_group=head_group)
        else:
            grad_start = self.enformer_tile_starts[grad_tile_idx]
            grad_out = self._enformer_forward_tile(inputs, grad_start, use_grad=True,
                                                  head_group=head_group)
            enf_full_log1p = None

        loss_enformer = self.enformer_window_loss(inputs, grad_out, grad_start)
        self.log('train_loss_enformer', loss_enformer, on_step=True, on_epoch=True,
                 prog_bar=True, logger=True, sync_dist=True)

        # Feed predicted tracks forward into layers 2 & 3 (random subset).
        if use_enformer_mix:
            inputs = inputs.clone()
            replaced = []
            for i, in_idx in enumerate(self.enformer_input_idx):
                if random.random() < self.hparams.enformer_track_mix_prob:
                    replaced.append(i)
            if not replaced:  # guarantee at least one so the tiling isn't wasted
                replaced = [random.randrange(len(self.enformer_input_idx))]
            if self.hparams.enformer_mix_mode == 'window':
                # Replace only the differentiable tile region (cheapest path).
                end = min(grad_start + ENFORMER_TARGET_LEN, L)
                grad_up = self._tile_to_full(grad_out, L)[:, :end - grad_start, :]
                for i in replaced:
                    inputs[:, grad_start:end, 5 + self.enformer_input_idx[i]] = grad_up[:, :, i]
            else:
                for i in replaced:
                    inputs[:, :, 5 + self.enformer_input_idx[i]] = enf_full_log1p[:, :, i]

        # ---- Layer 2: RAD21 predictor (skipped when rad21 is not an input) --
        # Consumes seq + (possibly Enformer-predicted) non-rad21 tracks.
        loss_rad21 = 0.0
        if self.use_rad21:
            inputs_without_rad21 = torch.cat([inputs[:, :, :abs_rad21],
                                              inputs[:, :, abs_rad21 + 1:]], dim=2)
            rad21_out = self.input_pred_model(inputs_without_rad21,
                                              conditioning_vec=condition_vec).get('1d')
            rad21_pred = rad21_out[:, :, 0]  # (B, target_1d_size)
            gt_rad21 = inputs[:, :, abs_rad21].detach()  # ground-truth rad21 (never mixed)
            gt_rad21_ds = F.interpolate(gt_rad21.unsqueeze(1), size=rad21_pred.shape[1],
                                        mode='linear', align_corners=True).squeeze(1).float()
            loss_rad21 = F.mse_loss(rad21_pred, gt_rad21_ds)
            self.log('train_loss_rad21', loss_rad21, on_step=True, on_epoch=True,
                     prog_bar=True, logger=True, sync_dist=True)

            # Feed predicted rad21 forward into layer 3.
            use_pred_rad21 = random.random() < mix_prob
            if use_pred_rad21:
                if not use_enformer_mix:
                    inputs = inputs.clone()
                rad21_up = F.interpolate(rad21_pred.unsqueeze(1), size=L,
                                         mode='linear', align_corners=True).squeeze(1).float()
                inputs[:, :, abs_rad21] = rad21_up

        # ---- Layer 3: Hi-C (and optional 1D track) predictor -------------
        outputs = self(inputs, conditioning_vec=condition_vec)
        loss_hic = 0.0
        if self.hparams.predict_hic:
            loss_hic = self.criterion(outputs.get('hic'), mat)

        # Layer 3 also reconstructs/predicts the --target-features 1D tracks,
        # matching the standard 8-track architecture.
        loss_1d = 0.0
        if target_1d_tracks is not None:
            pred_1d = outputs.get('1d')
            # A subclass may give layer 3 a NARROWER set of 1D targets than the
            # dataset loads -- the layer-4 trainer does, because transcription is
            # layer 4's job and --target-features has to name every track the
            # dataset reads for either layer.  Absent those attributes this is
            # exactly the old behaviour.
            feats_1d, idx_1d = self.layer3_1d_targets()
            target_1d_own = target_1d_tracks[:, :, idx_1d]
            loss_1d_per = F.mse_loss(pred_1d, target_1d_own, reduction='none').mean(dim=0)
            for i, feature in enumerate(feats_1d):
                track_loss = loss_1d_per[:, i].mean()
                self.log(f'train_loss_1d_{feature}', track_loss, on_step=True, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
            # Only score bins where the target is non-zero.  A window that is
            # entirely zero (telomere/centromere gap) has no valid bins at all and
            # mask.sum() == 0, so the division returns NaN -- which propagates into
            # the epoch loss.  With --no-hic this is layer 3's only loss, so a single
            # empty window is enough to NaN out the whole run.
            mask = target_1d_tracks != 0
            denom = mask.sum()
            loss_1d = (loss_1d_per * mask).sum() / denom if denom > 0 else loss_1d_per.sum() * 0.0
            self.log('train_loss_1d', loss_1d, on_step=True, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)

        total_loss = (self.hparams.training_loss_weight_enformer * loss_enformer
                      + self.hparams.training_loss_weight_rad21 * loss_rad21
                      + self.hparams.training_loss_weight_hic * loss_hic
                      + self.hparams.training_loss_weight_1d * loss_1d)

        self.log('train_loss', total_loss, on_step=True, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        self.log('train_loss_hic', loss_hic, on_step=True, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        return total_loss

    def validation_step(self, batch, batch_idx):
        inputs, mat, target_1d_tracks, condition_vec, celltype_idx = self.proc_batch(batch)
        L = inputs.shape[1]
        abs_rad21 = 5 + self.rad21_idx if self.use_rad21 else None
        head_group = self.head_group_for(celltype_idx)

        # ---- Layer 1: Enformer (fully detached full-window prediction) ----
        enf_full_log1p = self.enformer_predict_1d(inputs, head_group=head_group)   # linear (B, L, K)
        enf_full_log1p = torch.log1p(torch.clamp(enf_full_log1p, min=0))
        gt_tracks = inputs[:, :, self.enformer_gt_channels].float()
        loss_enformer = F.mse_loss(enf_full_log1p, gt_tracks)
        self.log('val_loss_enformer', loss_enformer, on_step=False, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        for i, feature in enumerate(self.enformer_tracks):
            pred = torch.clamp(torch.expm1(enf_full_log1p[..., i]), min=0)
            targ = torch.clamp(torch.expm1(gt_tracks[..., i]), min=0)
            corr = safe_corr(pred, targ)
            if corr is not None:
                self.log(f'val_enformer_corr_1d_{feature}', corr, on_step=False, on_epoch=True, logger=True, sync_dist=True)

        # With split heads, the whole point is that each celltype's head learns its
        # own signal -- so also report layer 1 per celltype.  Lightning averages each
        # metric over the batches in which it was logged, so logging only the
        # celltypes present in this batch is correct.
        if self.split_enformer_heads and celltype_idx is not None:
            for c in celltype_idx.unique().tolist():
                name = self.hparams.dataset_celltypes[c]
                rows = (celltype_idx == c).nonzero(as_tuple=True)[0]
                for i, feature in enumerate(self.enformer_tracks):
                    corr = safe_corr(
                        torch.clamp(torch.expm1(enf_full_log1p[rows][..., i]), min=0),
                        torch.clamp(torch.expm1(gt_tracks[rows][..., i]), min=0))
                    if corr is not None:
                        self.log(f'val_enformer_corr_1d_{feature}_{name}', corr,
                                 on_step=False, on_epoch=True, logger=True, sync_dist=False)

        # ---- Layer 2: RAD21 predictor (skipped when rad21 is not an input) --
        loss_rad21 = 0.0
        if self.use_rad21:
            inputs_no_rad21 = torch.cat([inputs[:, :, :abs_rad21], inputs[:, :, abs_rad21 + 1:]], dim=2)
            rad21_pred = self.input_pred_model(inputs_no_rad21, conditioning_vec=condition_vec).get('1d')[:, :, 0]
            rad21_up = F.interpolate(rad21_pred.unsqueeze(1), size=L, mode='linear', align_corners=True).squeeze(1).float()
            gt_rad21 = inputs[:, :, abs_rad21]
            loss_rad21 = F.mse_loss(rad21_up, gt_rad21)
            self.log('val_loss_rad21', loss_rad21, on_step=False, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
            rad21_corr = safe_corr(torch.clamp(torch.expm1(rad21_up), min=0),
                                   torch.clamp(torch.expm1(gt_rad21), min=0))
            if rad21_corr is not None:
                self.log('val_corr_1d_first_layer_rad21', rad21_corr, on_step=False, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)

        # ---- Layer 3: Hi-C (and optional 1D tracks) on ground-truth inputs -
        layer3_outputs = self(inputs, conditioning_vec=condition_vec)
        loss_hic = 0.0
        if self.hparams.predict_hic:
            pred_hic = layer3_outputs.get('hic')
            loss_hic = self.criterion(pred_hic, mat)
            hic_corr = safe_corr(pred_hic, mat)
            if hic_corr is not None:
                self.log('val_hic_corr', hic_corr, on_step=False, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
            # Contrast, not just agreement.  A head that has been pulled away
            # from its task collapses toward the mean: the correlation sags a
            # little while the predicted map goes flat.  Measured on the
            # mESC layer-4 checkpoint, r was 0.24 while the predicted std was
            # 2-4x BELOW the experimental one -- the flattening is the louder
            # signal, and the correlation alone hides it.  1.0 = matched spread.
            exp_std = mat.float().std()
            if torch.isfinite(exp_std) and exp_std > 0:
                self.log('val_hic_std_ratio', pred_hic.float().std() / exp_std,
                         on_step=False, on_epoch=True, prog_bar=True, logger=True,
                         sync_dist=True)

        # Layer 3's --target-features 1D reconstruction, matching the standard
        # 8-track architecture.
        loss_1d = 0.0
        if target_1d_tracks is not None:
            pred_1d = layer3_outputs.get('1d')
            feats_1d, idx_1d = self.layer3_1d_targets()
            target_1d_tracks = target_1d_tracks[:, :, idx_1d]
            loss_1d_per = F.mse_loss(pred_1d, target_1d_tracks, reduction='none').mean(dim=0)
            for i, feature in enumerate(feats_1d):
                track_loss = loss_1d_per[:, i].mean()
                self.log(f'val_loss_1d_{feature}', track_loss, on_step=False, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
                # Correlation for each 1D track (inverse log transform to linear space)
                pred_track = torch.clamp(torch.exp(pred_1d[..., i]) - 1, min=0)
                target_track = torch.clamp(torch.exp(target_1d_tracks[..., i]) - 1, min=0)
                corr = safe_corr(pred_track, target_track)
                if corr is not None:
                    self.log(f'val_corr_1d_{feature}', corr, on_step=False, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
            # Only score bins where the target is non-zero.  A window that is
            # entirely zero (telomere/centromere gap) has no valid bins at all and
            # mask.sum() == 0, so the division returns NaN -- which propagates into
            # the epoch loss.  With --no-hic this is layer 3's only loss, so a single
            # empty window is enough to NaN out the whole run.
            mask = target_1d_tracks != 0
            denom = mask.sum()
            loss_1d = (loss_1d_per * mask).sum() / denom if denom > 0 else loss_1d_per.sum() * 0.0
            self.log('val_loss_1d', loss_1d, on_step=False, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)

        # ---- Fully-chained layer 3: enformer -> (rad21) -> layer 3 --------
        # Measures the end-to-end path the perturbation pipeline uses. When
        # rad21 is absent the chain is simply enformer -> layer 3.  Runs whenever
        # layer 3 has *something* to predict, so --no-hic runs still get a chained
        # 1D metric.
        if self.hparams.predict_hic or target_1d_tracks is not None:
            chained = inputs.clone()
            for i, in_idx in enumerate(self.enformer_input_idx):
                chained[:, :, 5 + in_idx] = enf_full_log1p[:, :, i]
            if self.use_rad21:
                chained_no_rad21 = torch.cat([chained[:, :, :abs_rad21], chained[:, :, abs_rad21 + 1:]], dim=2)
                chained_rad21 = self.input_pred_model(chained_no_rad21, conditioning_vec=condition_vec).get('1d')[:, :, 0]
                chained[:, :, abs_rad21] = F.interpolate(chained_rad21.unsqueeze(1), size=L,
                                                         mode='linear', align_corners=True).squeeze(1).float()
            chained_outputs = self(chained, conditioning_vec=condition_vec)
            if self.hparams.predict_hic:
                chained_corr = safe_corr(chained_outputs.get('hic'), mat)
                if chained_corr is not None:
                    self.log('val_hic_corr_chained', chained_corr, on_step=False, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
            if target_1d_tracks is not None:
                chained_1d = chained_outputs.get('1d')
                # target_1d_tracks was already narrowed to layer 3's own targets
                # above, so enumerate the SAME feature list -- iterating over all
                # of --target-features indexes past the end of a narrowed head.
                for i, feature in enumerate(feats_1d):
                    corr = safe_corr(torch.clamp(torch.expm1(chained_1d[..., i]), min=0),
                                     torch.clamp(torch.expm1(target_1d_tracks[..., i]), min=0))
                    if corr is not None:
                        self.log(f'val_corr_1d_chained_{feature}', corr, on_step=False, on_epoch=True, logger=True, sync_dist=True)

        total_loss = (self.hparams.training_loss_weight_enformer * loss_enformer
                      + self.hparams.training_loss_weight_rad21 * loss_rad21
                      + self.hparams.training_loss_weight_hic * loss_hic
                      + self.hparams.training_loss_weight_1d * loss_1d)
        self.log('val_loss', total_loss, on_step=False, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        self.log('val_loss_hic', loss_hic, on_step=False, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        return total_loss

    def test_step(self, batch, batch_idx):
        total_loss = 0.0
        inputs, mat, target_1d_tracks, condition_vec, celltype_idx = self.proc_batch(batch)
        if condition_vec is not None:
            outputs = self(inputs, conditioning_vec=condition_vec)
        else:
            outputs = self(inputs)

        loss_hic = 0.0
        if self.hparams.predict_hic:
            pred_hic = outputs.get('hic')
            loss_hic = self.criterion(pred_hic, mat)
            total_loss += loss_hic * self.hparams.training_loss_weight_hic

        if target_1d_tracks is not None:
            pred_1d = outputs.get('1d')
            # Same narrowing as train/validation: layer 3 may own fewer 1D
            # targets than the dataset loads.
            feats_1d, idx_1d = self.layer3_1d_targets()
            target_1d_tracks = target_1d_tracks[:, :, idx_1d]
            loss_1d = torch.nn.functional.mse_loss(pred_1d, target_1d_tracks, reduction='none').mean(dim=0)
            # log each 1D track loss separately and measure correlation
            for i, feature in enumerate(feats_1d):
                track_loss = loss_1d[:, i].mean()
                self.log(f'test_loss_1d_{feature}', track_loss, on_step=False, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
     
            # Only score bins where the target is non-zero.  A window that is
            # entirely zero (telomere/centromere gap) has no valid bins at all and
            # mask.sum() == 0, so the division returns NaN -- which propagates into
            # the epoch loss.  With --no-hic this is layer 3's only loss, so a single
            # empty window is enough to NaN out the whole run.
            mask = target_1d_tracks != 0
            denom = mask.sum()
            loss_1d = (loss_1d * mask).sum() / denom if denom > 0 else loss_1d.sum() * 0.0
            total_loss += loss_1d * self.hparams.training_loss_weight_1d
            self.log('test_loss_1d', loss_1d, on_step=False, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
            
        self.log('test_loss', total_loss, on_step=False, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        self.log('test_loss_hic', loss_hic, on_step=False, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        
        return total_loss

    def configure_optimizers(self):
        enformer_params = list(self.enformer.parameters())
        model_params = list(self.model.parameters())

        # Only optimise Enformer parameters that are actually trainable
        # (trunk is frozen except the trailing fine-tuned block(s) + heads).
        enformer_params = [p for p in enformer_params if p.requires_grad]
        param_groups = [
            {'params': model_params, 'lr': self.hparams.optimizer_lr},          # layer 3 (Hi-C)
            {'params': enformer_params, 'lr': self.hparams.enformer_lr},        # layer 1 (Enformer)
        ]
        # layer 2 (RAD21) is only present when rad21 is an input feature.
        if self.input_pred_model is not None:
            param_groups.append({'params': list(self.input_pred_model.parameters()),
                                 'lr': self.hparams.rad21_lr})
        optimizer = torch.optim.AdamW(param_groups, weight_decay = 1e-6)

        from cshark.training.lr_scheduler import LinearWarmupCosineAnnealingLR
        scheduler = LinearWarmupCosineAnnealingLR(optimizer, warmup_epochs=10, max_epochs=self.args.trainer_max_epochs)
        scheduler.step()
        scheduler_config = {
            'scheduler': scheduler,
            'interval': 'epoch',
            'frequency': 1,
            'monitor': 'val_loss',
            'strict': True,
            'name': 'WarmupCosineAnnealing',
        }
        return {'optimizer' : optimizer, 'lr_scheduler' : scheduler_config}
    
    def on_before_optimizer_step(self, optimizer):
        # Compute the 2-norm for each layer
        # If using mixed precision, the gradients are already unscaled here
        norms = grad_norm(self.model, norm_type=2)
        self.log_dict(norms)

    def get_dataset(self, args, mode, celltype):

        celltype_root = f'{args.dataset_data_root}/{args.dataset_assembly}/{celltype}'
        genomic_features = {}
        for feature in args.input_features:
            genomic_features[feature] = {'file_name' : f'{feature}.bw',
                                         'norm' : 'log' if args.bigwig_log_transform else None }
        target_features = {}
        if args.output_features is not None:
            for feature in args.output_features:
                target_features[feature] = {'file_name' : f'{feature}.bw',
                                            'norm' : 'log' if args.bigwig_log_transform else None }
        if args.alt_assemblies is not None:
            alt_assemblies = args.alt_assemblies
            if len(alt_assemblies) != len(args.dataset_celltypes):
                raise ValueError('Number of alt assemblies must match number of celltypes')
            alt_assembly = alt_assemblies[args.dataset_celltypes.index(celltype)]

        if args.conditioning_vec is not None:
            conditioning_value = args.conditioning_vec[args.dataset_celltypes.index(celltype)]
            conditioning_value = np.array([float(x) for x in conditioning_value.split(',')])
            print(f'Using conditioning value {conditioning_value} for cell type {celltype}')

        dataset = genome_dataset.GenomeDataset(celltype_root, 
                                args.dataset_assembly,
                                input_feat_dicts = genomic_features, 
                                target_feat_dicts = target_features,
                                predict_hic = args.predict_hic,
                                predict_1d = (args.output_features is not None),
                                genome_assembly2 = args.dataset_assembly2,
                                alt_assembly= alt_assembly if args.alt_assemblies is not None else None,
                                target_res=args.resolution,
                                target_mat_size = args.mat_size,
                                target_1d_size = args.target_1d_size,
                                mode = mode,
                                hic_log_transform = args.hic_log_transform,
                                include_sequence = True,
                                include_genomic_features = True,
                                conditioning_vec = conditioning_value if args.conditioning_vec is not None else None,
                                # Only tag samples with their celltype when layer 1
                                # actually routes on it; otherwise the batch tuple stays
                                # exactly as every other trainer expects it.
                                celltype_index = (args.dataset_celltypes.index(celltype)
                                                  if self.split_enformer_heads else None)
                                )

        # Record length for printing validation image
        if mode == 'val':
            self.val_length = len(dataset) / args.dataloader_batch_size
            print('Validation loader length:', self.val_length)

        return dataset

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

if __name__ == '__main__':
    main()
