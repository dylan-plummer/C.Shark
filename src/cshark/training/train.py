import os
import sys
import wandb
import torch
import argparse
import numpy as np
import matplotlib.pyplot as plt
import pytorch_lightning as pl
import pytorch_lightning.callbacks as callbacks
from pytorch_lightning.callbacks.base import Callback
from pytorch_lightning.loggers import WandbLogger

from skimage.transform import resize

import cshark.model.corigami_models as corigami_models
from cshark.data import genome_dataset
import cshark.inference.utils.inference_utils as infer
from cshark.inference.utils import plot_utils 
import cshark.data.data_feature as data_feature


class VizCallback(Callback):
    def __init__(self, celltypes=['gm12878'], out_dir='deeploop_viz'):
        self.out_dir = out_dir
        os.makedirs(self.out_dir, exist_ok=True)
        self.celltypes = celltypes
        self.image_scale = 256  # size of each heatmap (fixed by model)
        self.loci = ['chr1:66000000', 'chr2:500000', 'chr3:145500000', 'chr3:186170000', 
                     'chr11:1500000', 'chr2:162000000',
                     #'chr5:93910000', 'chr5:97000000', 'chr5:87000000', 'chr7:137000000', 'chr8:4990000', 'chr9:90930000', 'chr12:93240000', 'chr18:55160000', 'chrX:11680000',
                     'chr10:122700000', 'chr15:59100000', 'chr12:89300000', 'chr20:47000000']
        self.chr_names = [s.split(':')[0] for s in self.loci]
        self.starts = [int(s.split(':')[1]) for s in self.loci]
        #self.chr_names = ['chr1', 'chr2', 'chr3', 'chr3', 'chr10', 'chr15', 'chr12', 'chr20']
        #self.starts = [66000000, 500000, 145500000, 122700000, 59100000, 89300000, 47000000]
        self.seq = "corigami_data/data/hg19/dna_sequence"
        # https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE167200
        self.ctcf = {celltype: f"corigami_data/data/hg19/{celltype}/genomic_features/ctcf.bw" for celltype in celltypes}
        self.atac = {celltype: None for celltype in celltypes}  # for if we are not using ATAC
        #self.atac = {celltype: f"corigami_data/data/hg19/{celltype}/genomic_features/atac.bw" for celltype in celltypes}
        # /mnt/rstor/genetics/JinLab/fxj45/WWW/ssz20/bigwig
        self.h3k27ac = {celltype: f"corigami_data/data/hg19/{celltype}/genomic_features/h3k27ac.bw" for celltype in celltypes}
        self.h3k4me3 = {celltype: f"corigami_data/data/hg19/{celltype}/genomic_features/h3k4me3.bw" for celltype in celltypes}
        # https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSM733679
        self.h3k36me3 = {celltype: f"corigami_data/data/hg19/{celltype}/genomic_features/h3k36me3.bw" for celltype in celltypes}
        # from here: /mnt/rstor/genetics/JinLab/fxj45/WWW/xww/bigwig
        self.h3k4me1 = {celltype: f"corigami_data/data/hg19/{celltype}/genomic_features/h3k4me1.bw" for celltype in celltypes}
        self.h3k27me3 = {celltype: f"corigami_data/data/hg19/{celltype}/genomic_features/h3k27me3.bw" for celltype in celltypes}

    def on_train_start(self, trainer, pl_module):
        print("Saving ground truth Hi-C example loci for reference")
        for celltype in self.celltypes:
            for chr_name, start in zip(self.chr_names, self.starts):
                locus = f"{chr_name}:{start}"
                hic = data_feature.HiCFeature(path = f'corigami_data/data/hg19/{celltype}/hic_matrix/{chr_name}.npz')
                mat = hic.get(start)
                mat = resize(mat, (self.image_scale, self.image_scale), anti_aliasing=True)
                os.makedirs(os.path.join(self.out_dir, locus), exist_ok=True)
                plot = plot_utils.MatrixPlot(os.path.join(self.out_dir, locus), mat, 'ground_truth', celltype, 
                                    chr_name, start)
                plot.plot()
                tmp_plot_path = os.path.join(self.out_dir, locus, celltype, 'ground_truth', 'imgs', f"{chr_name}_{start}.png")
                new_plot_path = os.path.join(self.out_dir, locus, celltype, f"ground_truth.png")
                try:
                    os.rename(tmp_plot_path, new_plot_path)
                    wandb.log({locus + '_experimental_' + celltype: wandb.Image(new_plot_path)})
                except Exception as e:
                    print(e)

    def on_validation_epoch_end(self, trainer, pl_module):
        print("Evaluating is starting")
        for celltype in self.celltypes:
            for chr_name, start in zip(self.chr_names, self.starts):
                try:
                    locus = f"{chr_name}:{start}"
                    #other_paths = [self.h3k27ac[celltype], self.h3k4me3[celltype]]
                    other_paths = None
                    seq_region, ctcf_region, atac_region, other_regions = infer.load_region(chr_name, 
                        start, self.seq, self.ctcf[celltype], self.atac[celltype], other_paths)
                    inputs = infer.preprocess_default(seq_region, ctcf_region, atac_region, other_regions)
                    pl_module.model.eval()
                    pred = pl_module.model(inputs)[0].detach().cpu().numpy()
                    pred = (pred + pred.T) * 0.5
                    os.makedirs(os.path.join(self.out_dir, locus), exist_ok=True)
                    plot = plot_utils.MatrixPlot(os.path.join(self.out_dir, locus), pred, 'prediction', celltype, 
                                        chr_name, start)
                    plot.plot()
                    tmp_plot_path = os.path.join(self.out_dir, locus, celltype, 'prediction', 'imgs', f"{chr_name}_{start}.png")
                    new_plot_path = os.path.join(self.out_dir, locus, celltype, f"{pl_module.current_epoch}.png")
                    try:
                        os.rename(tmp_plot_path, new_plot_path)
                        wandb.log({locus + '_' + celltype: wandb.Image(new_plot_path)})
                    except Exception as e:
                        print(e)
                except Exception as e:
                    print(e)


def main():
    args = init_parser()
    init_training(args)
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
  parser.add_argument('--model-type', dest='model_type', default='ConvTransModel',
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
  
  # add flags for CTCF, ATAC, and other genomic features
  parser.add_argument('--ctcf', dest='dataset_ctcf', default=True,
                            action='store_true',
                            help='Use CTCF')
  parser.add_argument('--atac', dest='dataset_atac', default=False,
                            action='store_true',
                            help='Use ATAC')
  parser.add_argument('--other-feats', dest='dataset_other_feats', default=None,
                            help='Other genomic features to use')


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
    wandb_logger = WandbLogger(project='c.origami')
    wandb_logger.watch(pl_module.model)
    pl_trainer = pl.Trainer(strategy='ddp',
                            accelerator="gpu" if torch.cuda.is_available() else "cpu", devices=args.trainer_num_gpu,
                            gradient_clip_val=1,
                            logger = wandb_logger,
                            callbacks = [VizCallback(celltypes=args.dataset_celltypes),
                                         early_stop_callback,
                                         checkpoint_callback,
                                         lr_monitor],
                            max_epochs = args.trainer_max_epochs
                            )
    trainloader = pl_module.get_dataloader(args, 'train')
    valloader = pl_module.get_dataloader(args, 'val')
    testloader = pl_module.get_dataloader(args, 'test')
    pl_trainer.fit(pl_module, trainloader, valloader)

class TrainModule(pl.LightningModule):
    
    def __init__(self, args):
        super().__init__()
        self.model = self.get_model(args)
        self.args = args
        self.save_hyperparameters()

    def forward(self, x):
        return self.model(x)

    def proc_batch(self, batch):
        seq, features, mat, start, end, chr_name, chr_idx = batch
        features = torch.cat([feat.unsqueeze(2) for feat in features], dim = 2)
        inputs = torch.cat([seq, features], dim = 2)
        mat = mat.float()
        return inputs, mat
    
    def training_step(self, batch, batch_idx):
        inputs, mat = self.proc_batch(batch)
        outputs = self(inputs)
        criterion = torch.nn.MSELoss()
        loss = criterion(outputs, mat)

        metrics = {'train_step_loss': loss}
        self.log_dict(metrics, batch_size = inputs.shape[0], prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        ret_metrics = self._shared_eval_step(batch, batch_idx)
        return ret_metrics

    def test_step(self, batch, batch_idx):
        ret_metrics = self._shared_eval_step(batch, batch_idx)
        return ret_metrics

    def _shared_eval_step(self, batch, batch_idx):
        inputs, mat = self.proc_batch(batch)
        outputs = self(inputs)
        criterion = torch.nn.MSELoss()
        loss = criterion(outputs, mat)
        return loss

    # Collect epoch statistics
    def training_epoch_end(self, step_outputs):
        step_outputs = [out['loss'] for out in step_outputs]
        ret_metrics = self._shared_epoch_end(step_outputs)
        metrics = {'train_loss' : ret_metrics['loss']
                  }
        self.log_dict(metrics, prog_bar=True)

    def validation_epoch_end(self, step_outputs):
        ret_metrics = self._shared_epoch_end(step_outputs)
        metrics = {'val_loss' : ret_metrics['loss']
                  }
        self.log_dict(metrics, prog_bar=True)

    def _shared_epoch_end(self, step_outputs):
        loss = torch.tensor(step_outputs).mean()
        return {'loss' : loss}

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
        genomic_features = {'ctcf' : {'file_name' : 'ctcf.bw',
                                             'norm' : 'log' },
                            # 'h3k27ac' : {'file_name' : 'h3k27ac.bw',
                            #                  'norm' : 'log' },
                            # 'h3k4me3' : {'file_name' : 'h3k4me3.bw',
                            #                     'norm' : 'log' },
                            # 'h3k36me3' : {'file_name' : 'h3k36me3.bw',
                            #                     'norm' : 'log' },
                            # 'h3k4me1': {'file_name' : 'h3k4me1.bw',
                            #                     'norm' : 'log' },
                            # 'h3k27me3': {'file_name' : 'h3k27me3.bw',
                            #                     'norm' : 'log' },      
                            # 'atac' : {'file_name' : 'atac.bw',
                            #                  'norm' : 'log' }
                            }
        dataset = genome_dataset.GenomeDataset(celltype_root, 
                                args.dataset_assembly,
                                genomic_features, 
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

    def get_model(self, args):
        model_name =  args.model_type
        num_genomic_features = 1
        ModelClass = getattr(corigami_models, model_name)
        model = ModelClass(num_genomic_features, mid_hidden = 256, use_cross_attn=False)
        if args.model_path is not None:
            checkpoint = torch.load(args.model_path, map_location='cuda' if torch.cuda.is_available() else 'cpu')
            model_weights = checkpoint['state_dict']

            # Edit keys
            for key in list(model_weights):
                model_weights[key.replace('model.', '')] = model_weights.pop(key)
            model.load_state_dict(model_weights)
        return model

if __name__ == '__main__':
    main()
