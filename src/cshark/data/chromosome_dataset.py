import sys 
import os
import random
import pickle
import pandas as pd
import numpy as np

from skimage.transform import resize
from torch.utils.data import Dataset

import cshark.data.data_feature as data_feature

class ChromosomeDataset(Dataset):
    '''
    Dataloader that provide sequence, features, and HiC data. Assume input
    folder strcuture.

    Args:
        data_root (str): Directory including sequence features, and HiC matrix 
            as subdirectories.
        chr_name (str): Name of the represented chromosome (e.g. chr1)
            as ``root/DNA/chr1/DNA`` for DNA as an example.
        omit_regions (list of tuples): start and end of excluded regions
    '''
    def __init__(self, celltype_root, chr_name, omit_regions, 
                 feature_list, target_track_list,
                 predict_hic=True, predict_1d=False, 
                 celltype_root2=None,
                 target_res=10000,
                 target_mat_size=256,
                 target_1d_size=512,
                 hic_log_transform=True,
                 use_aug = True):
        self.use_aug = use_aug
        self.res = target_res # 10kb resolution
        self.target_1d_len = target_1d_size
        self.bins = 2097152 / self.res # 2M bins
        self.image_scale = target_mat_size # IMPORTANT, scale output to image scale (e.g 210 to 256)
        self.sample_bins = 500
        self.stride = 50 # bins
        self.chr_name = chr_name
        self.predict_hic = predict_hic
        self.predict_1d = predict_1d
        self.target_1d_len = target_1d_size
        self.hic_log_transform = hic_log_transform

        # print(f'Loading chromosome {chr_name}...')
        # print(f'Predicting Hi-C: {self.predict_hic}, Predicting 1D Tracks: {self.predict_1d}')
        # print(f'Using {self.res} resolution, {self.bins} bins, {self.image_scale} image scale')

        # Get the parent directory of celltype_root
        parent_dir = os.path.dirname(celltype_root)
        dna_sequence_path = os.path.join(parent_dir, 'dna_sequence', f'{chr_name}.fa.gz')

        self.seq = data_feature.SequenceFeature(path=dna_sequence_path)
        print(dna_sequence_path)

        if celltype_root2:
            parent_dir2 = os.path.dirname(celltype_root2)
            dna_sequence_path2 = os.path.join(parent_dir2, 'dna_sequence', f'{chr_name}.fa.gz')
            print(dna_sequence_path2)
            self.seq2 = data_feature.SequenceFeature(path=dna_sequence_path2)
        else:
            self.seq2 = None
        self.genomic_features = feature_list
        self.mat = data_feature.HiCFeature(path = f'{celltype_root}/hic_matrix/{chr_name}.npz')

        if self.predict_1d:
            self.target_tracks = target_track_list # Target 1D features
            if not self.target_tracks:
                 raise ValueError("predict_1d is True, but target_track_list is empty.")
        else:
            self.target_tracks = []

        self.omit_regions = omit_regions
        self.check_length() # Check data length

        self.all_intervals = self.get_active_intervals()
        self.intervals = self.filter(self.all_intervals, omit_regions)

    def __getitem__(self, idx):
        start, end = self.intervals[idx]
        target_size = int(self.bins * self.res)

        # Shift Augmentations
        if self.use_aug: 
            start, end = self.shift_aug(target_size, start, end)
        else:
            start, end = self.shift_fix(target_size, start, end)

        seq, features, mat, target_1d_tracks = self.get_data_at_interval(start, end)
        if self.seq2:
            seq2 = self.seq2.get(start, end)

        if self.use_aug:
            # Extra on sequence
            seq = self.gaussian_noise(seq, 0.1)
            # Genomic features
            features = [self.gaussian_noise(item, 0.1) for item in features]
            if self.predict_1d:
                target_1d_tracks = [self.gaussian_noise(item, 0.1) for item in target_1d_tracks]
            # Reverse complement all data
            if self.seq2:
                seq2 = self.gaussian_noise(seq2, 0.1)
                chance = np.random.rand(1)
                if chance < 0.5:
                    seq, features, mat, target_1d_tracks = self.reverse(seq, features, mat, target_1d_tracks)
                    seq2, features, mat, target_1d_tracks = self.reverse(seq2, features, mat, target_1d_tracks)
            else:
                seq, features, mat, target_1d_tracks = self.reverse(seq, features, mat, target_1d_tracks)
        if self.seq2:
            seq = np.concatenate([seq, seq2], axis = 1)
        return seq, features, mat, target_1d_tracks, start, end

    def __len__(self):
        return len(self.intervals)

    def gaussian_noise(self, inputs, std = 1):
        noise = np.random.randn(*inputs.shape) * std
        outputs = inputs + noise
        return outputs

    def reverse(self, seq, features, mat, target_1d_tracks, chance = 0.5):
        '''
        Reverse sequence and matrix
        '''
        r_bool = np.random.rand(1)
        if r_bool < chance:
            seq_r = np.flip(seq, 0).copy() # n x 5 shape
            features_r = [np.flip(item, 0).copy() for item in features] # n
            target_1d_tracks_r = [np.flip(item, 0).copy() for item in target_1d_tracks] # n
            mat_r = np.flip(mat, [0, 1]).copy() # n x n

            # Complementary sequence
            seq_r = self.complement(seq_r)
        else:
            seq_r = seq
            features_r = features
            mat_r = mat
            target_1d_tracks_r = target_1d_tracks
        return seq_r, features_r, mat_r, target_1d_tracks_r

    def complement(self, seq, chance = 0.5):
        '''
        Complimentary sequence
        '''
        r_bool = np.random.rand(1)
        if r_bool < chance:
            seq_comp = np.concatenate([seq[:, 1:2],
                                       seq[:, 0:1],
                                       seq[:, 3:4],
                                       seq[:, 2:3],
                                       seq[:, 4:5]], axis = 1)
        else:
            seq_comp = seq
        return seq_comp

    def get_data_at_interval(self, start, end):
        '''
        Slice data from arrays with transformations
        '''
        # Sequence processing
        seq = self.seq.get(start, end)
        
        # Features processing
        features = [item.get(self.chr_name, start, end) for item in self.genomic_features]
        # Hi-C matrix processing
        mat = self.mat.get(start, res=self.res)
        mat = resize(mat, (self.image_scale, self.image_scale), anti_aliasing=True, preserve_range=True)
        if self.hic_log_transform:
            mat = np.log(mat + 1)
        # Target 1D track processing
        loaded_paths = [item.path for item in self.genomic_features]
        target_1d_tracks_out = []
        if self.predict_1d:
             # Target 1D tracks also correspond to the input window [start, end]
             target_1d_tracks = []  # re-use already loaded tracks
             for item in self.target_tracks:
                if item.path in loaded_paths:
                    target_1d_tracks.append(features[loaded_paths.index(item.path)])
                else:
                    target_1d_tracks.append(item.get(self.chr_name, start, end))
             # Ensure target tracks have the expected length (padding if necessary)
             for track in target_1d_tracks:
                  bin_size = int(len(track) / self.target_1d_len)
                  resized_track = track.reshape(-1, bin_size).mean(axis=1)
                  target_1d_tracks_out.append(resized_track)
        return seq, features, mat, target_1d_tracks_out

    def get_active_intervals(self):
        '''
        Get intervals for sample data: [[start, end]]
        '''
        chr_bins = len(self.seq) / self.res
        data_size = (chr_bins - self.sample_bins) / self.stride
        starts = np.arange(0, data_size).reshape(-1, 1) * self.stride
        intervals_bin = np.append(starts, starts + self.sample_bins, axis=1)
        intervals = intervals_bin * self.res
        return intervals.astype(int)

    def filter(self, intervals, omit_regions):
        valid_intervals = []
        for start, end in intervals: 
            # Way smaller than omit or way larger than omit
            start_cond = start <= omit_regions[:, 1]
            end_cond = omit_regions[:, 0] <= end
            #import pdb; pdb.set_trace()
            if sum(start_cond * end_cond) == 0:
                valid_intervals.append([start, end])
        return valid_intervals

    def encode_seq(self, seq):
        ''' 
        encode dna to onehot (n x 5)
        '''
        seq_emb = np.zeros((len(seq), 5))
        seq_emb[np.arange(len(seq)), seq] = 1
        return seq_emb

    def shift_aug(self, target_size, start, end):
        '''
        All unit are in basepairs
        '''
        offset = random.choice(range(end - start - target_size))
        return start + offset , start + offset + target_size

    def shift_fix(self, target_size, start, end):
        offset = 0
        return start + offset , start + offset + target_size

    def check_length(self):
        # Check sequence vs first *input* feature
        if self.genomic_features:
             assert len(self.seq.seq) == self.genomic_features[0].length(self.chr_name), f'Sequence {len(self.seq)} and First feature {self.genomic_features[0].length(self.chr_name)} have different length.'
        # Check sequence vs first *target* 1D track (if applicable)
        if self.predict_1d and self.target_tracks:
             assert len(self.seq.seq) == self.target_tracks[0].length(self.chr_name), f'Sequence {len(self.seq)} and First target track {self.target_tracks[0].length(self.chr_name)} have different length.'
        # Check Hi-C length consistency (if applicable)
        if self.predict_hic and self.mat:
             assert abs(len(self.seq) / self.res -  len(self.mat)) < 2, f'Sequence {len(self.seq) / self.res} and Hi-C {len(self.mat)} have different length.' 

def get_feature_list(root_dir, feat_dicts):
    '''
    Args:
        features: a list of dicts with 
            1. file name
            2. norm status
    Returns:
        feature_list: a list of genomic features (bigwig files)
    '''
    feat_list = []
    for feat_item in feat_dicts:
        file_name = feat_item['file_name']
        file_path = f'{root_dir}/{file_name}'
        norm = feat_item['norm']
        feat_list.append(data_feature.GenomicFeature(file_path, norm))
    return feat_list

def proc_centrotelo(bed_dir):
    ''' Take a bed file indicating location, output a dictionary of items 
    by chromosome which contains a list of 2 value lists (range of loc)
    '''
    df = pd.read_csv(bed_dir , sep = '\t', names = ['chr', 'start', 'end'], usecols = [0, 1, 2])
    chrs = df['chr'].unique()
    centrotelo_dict = {}
    for chr_name in chrs:
        sub_df = df[df['chr'] == chr_name]
        regions = sub_df.drop('chr', axis = 1).to_numpy()
        centrotelo_dict[chr_name] = regions
    return centrotelo_dict
