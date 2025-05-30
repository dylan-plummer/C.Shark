import os
import pandas as pd
from torch.utils.data import Dataset
from cshark.data.chromosome_dataset import ChromosomeDataset
import cshark.data.data_feature as data_feature

class GenomeDataset(Dataset):
    '''
    Load all chromosomes
    '''
    def __init__(self, celltype_root, 
                       genome_assembly,
                       input_feat_dicts, 
                       target_feat_dicts={}, # NEW: Dictionary for target features
                       mode = 'train', 
                       include_sequence = True,
                       include_genomic_features = True,
                       predict_hic=True, # NEW: Control Hi-C loading/prediction
                       predict_1d=False,  # NEW: Control 1D track loading/prediction
                       genome_assembly2=None,
                       target_res=10000,
                       target_mat_size=256,
                       target_1d_size=512,
                       hic_log_transform=True,
                       use_aug = True):
        self.data_root = celltype_root
        self.include_sequence = include_sequence
        self.include_genomic_features = include_genomic_features
        self.predict_hic = predict_hic
        self.predict_1d = predict_1d
        self.genome_assembly = genome_assembly
        self.genome_assembly2 = genome_assembly2
        self.target_res = target_res
        self.target_mat_size = target_mat_size
        self.target_1d_size = target_1d_size
        self.hic_log_transform = hic_log_transform

        if not self.include_sequence: print('Not using sequence!')
        if not self.include_genomic_features: print('Not using genomic features!')
        if not self.predict_hic: print('Not predicting Hi-C!')
        if not self.predict_1d: print('Not predicting 1D Tracks!')
        if not predict_hic and not predict_1d:
             raise ValueError("Must predict at least Hi-C or 1D tracks.")
        self.use_aug = use_aug

        if mode != 'train': self.use_aug = False # Set augmentation

        # Assign train/val/test chromosomes
        self.chr_names = self.get_chr_names(genome_assembly)
        if mode == 'train':
            self.chr_names.remove('chr10')
            self.chr_names.remove('chr15')
            self.chr_names.remove('chrX') # chrX removed for consistency
        elif mode == 'val':
            self.chr_names = ['chr10']
        elif mode == 'test':
            self.chr_names = ['chr15']
        else:
            raise Exception(f'Unknown mode {mode}')

        # Load genomewide features
        self.genomic_features = self.load_features(f'{celltype_root}/genomic_features', input_feat_dicts)

        # Load target 1D track features
        if self.predict_1d:
             # Assume target features are also in genomic_features dir, or specify another path
             self.target_1d_features = self.load_features(f'{celltype_root}/genomic_features', target_feat_dicts)
             if not self.target_1d_features:
                  raise ValueError("predict_1d is True, but no target features were loaded. Check target_feat_dicts.")
        else:
             self.target_1d_features = []

        # Load regions to be ignored
        root_data_dir = os.path.dirname(celltype_root)
        self.centrotelo_dict = self.proc_centrotelo(f'{root_data_dir}/centrotelo.bed')
        # Load chromsome_datasets as part of the dictionary
        self.chr_data, self.lengths = self.load_chrs(self.chr_names, self.genomic_features, self.target_1d_features)
        # Build chrmosome lookup table from the genome
        self.ranges = self.get_ranges(self.lengths)

    def __getitem__(self, idx):
        # Query for chromosome name and where in the chromosome
        chr_name, chr_idx = self.get_chr_idx(idx)
        # Get data from the specific ChromosomeDataset
        data_tuple = self.chr_data[chr_name][chr_idx]

        # Unpack the tuple based on predict_hic and predict_1d flags
        # Order: seq, features, [mat], [target_1d_tracks], start, end
        seq, features, mat, target_1d_tracks, start, end = data_tuple

        # Construct the output tuple, setting optional components to None if not included
        if self.predict_1d:
            outputs = [
                seq if self.include_sequence else None,
                features if self.include_genomic_features else None,
                mat if self.predict_hic else None,
                target_1d_tracks if self.predict_1d else None,
                start,
                end,
                chr_name,
                chr_idx
            ]
        else:
            outputs = [
                seq if self.include_sequence else None,
                features if self.include_genomic_features else None,
                mat if self.predict_hic else None,
                start,
                end,
                chr_name,
                chr_idx
            ]
        

        return tuple(outputs)

    def __len__(self):
        return sum(self.lengths)
        
    def load_chrs(self, chr_names, genomic_features, target_features):
        '''
        Load chromosome data into a dictionary
        '''
        print('Loading chromosome datasets...')
        chr_data_dict = {}
        lengths = []
        for chr_name in chr_names:
            omit_regions = self.centrotelo_dict[chr_name]
            chr_data_dict[chr_name] = ChromosomeDataset(self.data_root, chr_name, omit_regions, 
                                                        genomic_features, target_features, predict_hic=True, 
                                                        predict_1d=self.predict_1d, 
                                                        celltype_root2=self.data_root.replace(self.genome_assembly, self.genome_assembly2) if self.genome_assembly2 else None,
                                                        target_res=self.target_res,
                                                        target_mat_size=self.target_mat_size,
                                                        target_1d_size=self.target_1d_size,
                                                        hic_log_transform=self.hic_log_transform,
                                                        use_aug=self.use_aug)
            lengths.append(len(chr_data_dict[chr_name]))
        print('Chromosome datasets loaded')
        return chr_data_dict, lengths

    def load_features(self, root_dir, feat_dicts):
        '''
        Args:
            features: a list of dicts with 
                1. file name
                2. norm status
        Returns:
            feature_list: a list of genomic features (bigwig files)
        '''
        feat_list = []
        for feat_item in list(feat_dicts.values()):
            file_name = feat_item['file_name']
            file_path = f'{root_dir}/{file_name}'
            norm = feat_item['norm']
            feat_list.append(data_feature.GenomicFeature(file_path, norm))
        return feat_list
        
    def get_chr_names(self, assembly):
        '''
        Get a list of all chr names. e.g. [chr1 , chr2, ...]
        '''
        print(f'Using Assembly: {assembly}')
        if assembly in ['hg38', 'hg19']:
            chrs = list(range(1, 23))
        # mouse strain genomes (mm10 is b6)
        elif str(assembly) in ['mm10', 'mm9', '129', 'pwk', 'aj', 'cast', 'nzo', 'nod', 'wsb']:
            chrs = list(range(1, 20))
        else: raise Exception(f'Assembly {assembly} unknown')
        chrs.append('X')
        #chrs.append('Y')
        chr_names = []
        for chr_num in chrs:
            chr_names.append(f'chr{chr_num}')
        return chr_names

    def get_ranges(self, lengths):
        current_start = 0
        ranges = []
        for length in lengths:
            ranges.append([current_start, current_start + length - 1])
            current_start += length
        return ranges

    def get_chr_idx(self, idx):
        '''
        Check index and return chr_name and chr index
        '''
        for i, chr_range in enumerate(self.ranges):
            start, end = chr_range
            if start <= idx <= end:
                return self.chr_names[i], idx - start

    def proc_centrotelo(self, bed_dir):
        ''' Take a bed file indicating location, output a dictionary of items 
        by chromosome which contains a list of 2 value lists (range of loc)
        '''
        df = pd.read_csv(bed_dir , sep = '\t', names = ['chr', 'start', 'end'])
        chrs = df['chr'].unique()
        centrotelo_dict = {}
        for chr_name in chrs:
            sub_df = df[df['chr'] == chr_name]
            regions = sub_df.drop('chr', axis = 1).to_numpy()
            centrotelo_dict[chr_name] = regions
        return centrotelo_dict
