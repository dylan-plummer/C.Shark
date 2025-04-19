import os
import sys
import wandb
import torch
import argparse
import numpy as np
import matplotlib.pyplot as plt
import lightning as pl
import lightning.pytorch.callbacks as callbacks
from lightning.pytorch.callbacks import Callback
from lightning.pytorch.loggers.wandb import WandbLogger

from skimage.transform import resize

import cshark.model.corigami_models as corigami_models
from cshark.data import genome_dataset
import cshark.inference.utils.inference_utils as infer
from cshark.inference.utils import plot_utils 
import cshark.data.data_feature as data_feature


class VizCallback(Callback):
    def __init__(self, data_root='cshark_data/data', celltypes=['gm12878'], assembly='hg19', out_dir='deeploop_viz'):
        self.out_dir = out_dir
        os.makedirs(self.out_dir, exist_ok=True)
        self.data_root = data_root
        self.celltypes = celltypes
        self.assembly = assembly
        self.image_scale = 256  # size of each heatmap (fixed by model)
        self.loci = ['chr1:66000000', 'chr2:500000', 'chr3:145500000',
                     'chr11:1500000', 'chr2:162000000',
                     'chr10:122700000', 'chr15:59100000', 'chr12:89300000']
        self.chr_names = [s.split(':')[0] for s in self.loci]
        self.starts = [int(s.split(':')[1]) for s in self.loci]
        self.seq = f"{self.data_root}/{self.assembly}/dna_sequence"
        # https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE167200
        self.ctcf = {celltype: f"{self.data_root}/{self.assembly}/{celltype}/genomic_features/ctcf.bw" for celltype in celltypes}
        #self.atac = {celltype: None for celltype in celltypes}  # for if we are not using ATAC
        self.atac = {celltype: f"{self.data_root}/hg19/{celltype}/genomic_features/atac.bw" for celltype in celltypes}
        # /mnt/rstor/genetics/JinLab/fxj45/WWW/ssz20/bigwig
        self.h3k27ac = {celltype: f"{self.data_root}/{self.assembly}/{celltype}/genomic_features/h3k27ac.bw" for celltype in celltypes}
        self.h3k4me3 = {celltype: f"{self.data_root}/{self.assembly}/{celltype}/genomic_features/h3k4me3.bw" for celltype in celltypes}
        # https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSM733679
        self.h3k36me3 = {celltype: f"{self.data_root}/{self.assembly}/{celltype}/genomic_features/h3k36me3.bw" for celltype in celltypes}
        # from here: /mnt/rstor/genetics/JinLab/fxj45/WWW/xww/bigwig
        self.h3k4me1 = {celltype: f"{self.data_root}/{self.assembly}/{celltype}/genomic_features/h3k4me1.bw" for celltype in celltypes}
        self.h3k27me3 = {celltype: f"{self.data_root}/{self.assembly}/{celltype}/genomic_features/h3k27me3.bw" for celltype in celltypes}
        self.rad21 = {celltype: f"{self.data_root}/{self.assembly}/{celltype}/genomic_features/rad21.bw" for celltype in celltypes}

    def on_train_start(self, trainer, pl_module):
        print("Saving ground truth Hi-C example loci for reference")
        for celltype in self.celltypes:
            for chr_name, start in zip(self.chr_names, self.starts):
                locus = f"{chr_name}:{start}"
                hic = data_feature.HiCFeature(path = f'{self.data_root}/{self.assembly}/{celltype}/hic_matrix/{chr_name}.npz')
                mat = hic.get(start)
                mat = resize(mat, (self.image_scale, self.image_scale), anti_aliasing=True, preserve_range=True)
                os.makedirs(os.path.join(self.out_dir, locus), exist_ok=True)
                plot = plot_utils.MatrixPlot(os.path.join(self.out_dir, locus), mat, 'ground_truth', celltype, 
                                    chr_name, start)
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


    def on_validation_epoch_end(self, trainer, pl_module):
        print("Evaluating is starting")
        for celltype in self.celltypes:
            for chr_name, start in zip(self.chr_names, self.starts):
                try:
                    locus = f"{chr_name}:{start}"
                    #other_paths = [self.h3k27ac[celltype], self.h3k4me3[celltype]]
                    other_paths = []
                    for feature in pl_module.hparams.input_features:
                        if feature == 'h3k27ac':
                            other_paths.append(self.h3k27ac[celltype])
                        elif feature == 'h3k4me3':
                            other_paths.append(self.h3k4me3[celltype])
                        elif feature == 'h3k36me3':
                            other_paths.append(self.h3k36me3[celltype])
                        elif feature == 'h3k4me1':
                            other_paths.append(self.h3k4me1[celltype])
                        elif feature == 'h3k27me3':
                            other_paths.append(self.h3k27me3[celltype])
                        elif feature == 'rad21':
                            other_paths.append(self.rad21[celltype])
                    #other_paths = [self.h3k27me3[celltype]]
                    seq_region, ctcf_region, atac_region, other_regions = infer.load_region(chr_name, 
                        start, self.seq, self.ctcf[celltype], self.atac[celltype], other_paths)
                    inputs = infer.preprocess_default(seq_region, ctcf_region, atac_region, other_regions)
                    pl_module.model.eval()
                    outputs = pl_module.model(inputs)
                    pred = outputs.get('hic')[0].detach().cpu().numpy()
                    print('pred shape:', pred.shape)
                    pred = (pred + pred.T) * 0.5
                    os.makedirs(os.path.join(self.out_dir, locus), exist_ok=True)
                    plot = plot_utils.MatrixPlot(os.path.join(self.out_dir, locus), pred, 'prediction', celltype, 
                                        chr_name, start)
                    plot.plot()
                    tmp_plot_path = os.path.join(self.out_dir, locus, celltype, 'prediction', 'imgs', f"{chr_name}_{start}.png")
                    new_plot_path = os.path.join(self.out_dir, locus, celltype, f"{pl_module.current_epoch}.png")
                    try:
                        os.rename(tmp_plot_path, new_plot_path)
                        if pl_module.hparams.use_wandb:
                            wandb.log({locus + '_' + celltype: wandb.Image(new_plot_path)})
                    except Exception as e:
                        print(e)

                    pred_1d_tracks = outputs.get('1d')[0].permute(1, 0).detach().cpu().numpy()
                    print(pred_1d_tracks.shape)
                    if pred_1d_tracks is not None:
                        os.makedirs(os.path.join(self.out_dir, locus, celltype, '1d_tracks'), exist_ok=True)
                        # visualize 1D tracks as shaded plots
                        fig, axs = plt.subplots(len(pred_1d_tracks), 1, figsize=(10, len(pred_1d_tracks) * 2))
                        if len(pred_1d_tracks) == 1:
                            axs = [axs]
                        colors = ['blue', 'orange', 'green', 'red', 'purple', 'brown', 'pink', 'gray']
                        for i, pred_1d in enumerate(pred_1d_tracks):
                            track_name = pl_module.hparams.output_features[i]
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

def init_parser():
  parser = argparse.ArgumentParser(description='C.Origami Training Module.')

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
  # list of celltypes
  parser.add_argument('--celltypes', dest='dataset_celltypes', default=['alpha', 'beta'], nargs='+',
                        help='Cell types to train on')

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
                            default=['ctcf', 'atac'],
                            help='Input features to use')
  parser.add_argument('--target-features', dest='output_features', nargs='+',
                            default=None,
                            help='Target features to use')
  parser.add_argument('--target-feature-size', dest='target_1d_size', type=int, default=2048,
                      help='Size of output 1d track')
  parser.add_argument('--latent-dim', dest='model_latent_dim', type=int, default=256,
                      help='Latent dimension size (mid_hidden)')
  parser.add_argument('--lr', dest='optimizer_lr', type=float, default=2e-4, help='Learning rate')
  parser.add_argument('--loss-weight-hic', dest='training_loss_weight_hic', type=float, default=1.0,
                      help='Weight for Hi-C loss term')
  parser.add_argument('--loss-weight-1d', dest='training_loss_weight_1d', type=float, default=1.0,
                      help='Weight for 1D track loss term')


  args = parser.parse_args(args=None if sys.argv[1:] else ['--help'])
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
    pl_trainer = pl.Trainer(strategy='ddp',
                            accelerator="gpu" if torch.cuda.is_available() else "cpu", devices=args.trainer_num_gpu,
                            gradient_clip_val=1,
                            logger = wandb_logger if args.use_wandb else None,
                            callbacks = [VizCallback(data_root=args.dataset_data_root,
                                                     celltypes=args.dataset_celltypes,  
                                                     assembly=args.dataset_assembly),
                                         early_stop_callback,
                                         checkpoint_callback,
                                         lr_monitor],
                            max_epochs = args.trainer_max_epochs
                            )
    trainloader = pl_module.get_dataloader(args, 'train')
    valloader = pl_module.get_dataloader(args, 'val')
    testloader = pl_module.get_dataloader(args, 'test')

    for test_batch_i in range(1):
        # load a batch and visualize it for debugging
        batch = next(iter(trainloader))
        inputs, mat, target_1d_tracks = pl_module.proc_batch(batch)
        print('inputs shape:', inputs.shape) # (batch, window, 5 + num_genomic_features)
        print('mat shape:', mat.shape)  # (batch, image_scale, image_scale)
        print('target_1d_tracks shape:', target_1d_tracks.shape if target_1d_tracks is not None else None)
        
        # visualize the input genomic features
        genomic_features = inputs[:, :, 5:]
        genomic_features = genomic_features[0].detach().cpu().numpy() 
        #genomic_features = resize(genomic_features, (pl_module.hparams.target_1d_size,), anti_aliasing=True, preserve_range=True)
        
        fig, axs = plt.subplots(genomic_features.shape[1], 1, figsize=(15, 4))  
        if genomic_features.shape[1] == 1:
            axs = [axs]
        colors = ['blue', 'orange', 'green', 'red', 'purple', 'brown', 'pink', 'gray']
        for i in range(genomic_features.shape[1]):
            track = np.exp(genomic_features[:, i]) - 1  # inverse log transformation
            bin_size = int(len(track) / pl_module.hparams.target_1d_size)
            track = track.reshape(-1, bin_size).mean(axis=1)
            axs[i].plot(track, color=colors[i % len(colors)])
            axs[i].fill_between(range(len(track)), track, color=colors[i % len(colors)], alpha=0.5)
        plt.savefig(f'input_genomic_features.png_{test_batch_i}.png')
        plt.close()

        # visualize the target Hi-C matrix
        mat = mat[0].detach().cpu().numpy()
        mat = resize(mat, (pl_module.hparams.target_1d_size, pl_module.hparams.target_1d_size), anti_aliasing=True, preserve_range=True)
        plt.imshow(mat, cmap='Reds', interpolation='none')
        plt.colorbar()
        plt.title('Target Hi-C Matrix')
        plt.savefig(f'target_hic_matrix.png_{test_batch_i}.png')
        plt.close()

        # visualize the target 1D tracks
        if target_1d_tracks is not None:
            target_1d_tracks = target_1d_tracks[0].detach().cpu().numpy()
            #target_1d_tracks = resize(target_1d_tracks, (pl_module.hparams.target_1d_size,), anti_aliasing=True, preserve_range=True)
            fig, axs = plt.subplots(target_1d_tracks.shape[1], 1, figsize=(15, 4))
            if target_1d_tracks.shape[1] == 1:
                axs = [axs]
            for i in range(target_1d_tracks.shape[1]):
                track = np.exp(target_1d_tracks[:, i]) - 1  # inverse log transformation
                axs[i].plot(track, color=colors[i % len(colors)])
                axs[i].fill_between(range(len(target_1d_tracks)), track, color=colors[i % len(colors)], alpha=0.5)
            plt.title('Target 1D Tracks')
            plt.savefig(f'target_1d_tracks.png_{test_batch_i}.png')
            plt.close()

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
        

    def get_model(self, args):
        model_name =  args.model_type
        ModelClass = getattr(corigami_models, model_name)
        num_input_features = 0
        num_input_features = len(self.hparams.input_features)

        num_target_tracks = 0
        if self.predict_1d:
            num_target_tracks = len(self.hparams.output_features)

        # Instantiate the model
        model = ModelClass(
            num_genomic_features=num_input_features, # Input features
            num_target_tracks=num_target_tracks,    # Target 1D tracks
            mid_hidden=self.hparams.model_latent_dim,
            predict_hic=True,
            predict_1d=self.predict_1d,
            target_1d_length=args.target_1d_size
            # Add other necessary model args from hparams if they exist
        )
        if args.model_path is not None:
            checkpoint = torch.load(args.model_path, map_location='cuda' if torch.cuda.is_available() else 'cpu')
            model_weights = checkpoint['state_dict']

            # Edit keys
            for key in list(model_weights):
                model_weights[key.replace('model.', '')] = model_weights.pop(key)
            model.load_state_dict(model_weights)
        return model

    def forward(self, x):
        return self.model(x)

    def proc_batch(self, batch):
        target_1d_tracks = None
        if self.predict_1d:
            seq, features, mat, target_1d_tracks, start, end, chr_name, chr_idx = batch
        else:
            seq, features, mat, start, end, chr_name, chr_idx = batch
        features = torch.cat([feat.unsqueeze(2) for feat in features], dim = 2)
        inputs = torch.cat([seq, features], dim = 2)
        mat = mat.float()
        if target_1d_tracks is not None:
            target_1d_tracks = torch.stack(target_1d_tracks, dim = 2)
        target_1d_tracks = target_1d_tracks.float() if target_1d_tracks is not None else None
        return inputs, mat, target_1d_tracks
    
    def training_step(self, batch, batch_idx):
        total_loss = 0.0
        inputs, mat, target_1d_tracks = self.proc_batch(batch)
        outputs = self(inputs)

        pred_hic = outputs.get('hic')
        loss_hic = self.criterion(pred_hic, mat)
        total_loss += loss_hic * self.hparams.training_loss_weight_hic

        if target_1d_tracks is not None:
            pred_1d = outputs.get('1d')
            loss_1d = self.criterion(pred_1d, target_1d_tracks)
            total_loss += loss_1d * self.hparams.training_loss_weight_1d
            self.log('train_loss_1d', loss_1d, on_step=True, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)

        self.log('train_loss', total_loss, on_step=True, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        self.log('train_loss_hic', loss_hic, on_step=True, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        
        return total_loss

    def validation_step(self, batch, batch_idx):
        total_loss = 0.0
        inputs, mat, target_1d_tracks = self.proc_batch(batch)
        outputs = self(inputs)

        pred_hic = outputs.get('hic')
        loss_hic = self.criterion(pred_hic, mat)
        total_loss += loss_hic * self.hparams.training_loss_weight_hic

        if target_1d_tracks is not None:
            pred_1d = outputs.get('1d')
            loss_1d = self.criterion(pred_1d, target_1d_tracks)
            total_loss += loss_1d * self.hparams.training_loss_weight_1d
            self.log('val_loss_1d', loss_1d, on_step=False, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)

        self.log('val_loss', total_loss, on_step=False, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        self.log('val_loss_hic', loss_hic, on_step=False, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        
        return total_loss

    def test_step(self, batch, batch_idx):
        total_loss = 0.0
        inputs, mat, target_1d_tracks = self.proc_batch(batch)
        outputs = self(inputs)

        pred_hic = outputs.get('hic')
        loss_hic = self.criterion(pred_hic, mat)
        total_loss += loss_hic * self.hparams.training_loss_weight_hic

        if target_1d_tracks is not None:
            pred_1d = outputs.get('1d')
            loss_1d = self.criterion(pred_1d, target_1d_tracks)
            total_loss += loss_1d * self.hparams.training_loss_weight_1d
            self.log('test_loss_1d', loss_1d, on_step=False, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)

        self.log('test_loss', total_loss, on_step=False, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        self.log('test_loss_hic', loss_hic, on_step=False, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        
        return total_loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), 
                                     lr = 2e-4,
                                     weight_decay = 0)

        import pl_bolts
        scheduler = pl_bolts.optimizers.lr_scheduler.LinearWarmupCosineAnnealingLR(optimizer, warmup_epochs=10, max_epochs=self.args.trainer_max_epochs)
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

    def get_dataset(self, args, mode, celltype):

        celltype_root = f'{args.dataset_data_root}/{args.dataset_assembly}/{celltype}'
        genomic_features = {}
        for feature in args.input_features:
            genomic_features[feature] = {'file_name' : f'{feature}.bw',
                                         'norm' : 'log' }
        target_features = {}
        if args.output_features is not None:
            for feature in args.output_features:
                target_features[feature] = {'file_name' : f'{feature}.bw',
                                            'norm' : 'log' }
        dataset = genome_dataset.GenomeDataset(celltype_root, 
                                args.dataset_assembly,
                                input_feat_dicts = genomic_features, 
                                target_feat_dicts = target_features,
                                predict_hic = True,
                                predict_1d = (args.output_features is not None),
                                target_1d_size = args.target_1d_size,
                                mode = mode,
                                include_sequence = True,
                                include_genomic_features = True)

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
