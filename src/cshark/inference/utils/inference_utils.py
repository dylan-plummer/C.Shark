import os
import numpy as np
import pandas as pd
import sys
import torch

from cshark.data.data_feature import SequenceFeature, GenomicFeature
from cshark.inference.utils.model_utils import load_default

def preprocess_default(seq, ctcf, atac, other=None):
    # Process sequence
    seq = torch.tensor(seq).unsqueeze(0) 
    # Normailze ctcf and atac-seq
    ctcf = torch.tensor(np.nan_to_num(ctcf, 0)) # Important! replace nan with 0
    if atac is not None:
        atac_log = torch.tensor(atac) # Important! replace nan with 0
        # Merge inputs
        features = [ctcf, atac_log]
    else:
        features = [ctcf]
    if other is not None:
        for other_region in other:
            other_feat = torch.tensor(np.nan_to_num(other_region, 0))
            features.append(other_feat)
    features = torch.cat([feat.unsqueeze(0).unsqueeze(2) for feat in features], dim = 2)
    inputs = torch.cat([seq, features], dim = 2)
    # Move input to gpu if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    inputs = inputs.to(device)
    return inputs

## Load data ##
def load_region(chr_name, start, seq_path, ctcf_path, atac_path, other_paths=None, window = 2097152, ctcf_log2=False):
    ''' Single loading method for one region '''
    end = start + window
    seq, ctcf, atac = load_data_default(chr_name, seq_path, ctcf_path, atac_path, ctcf_log2=ctcf_log2)
    other_regions = None
    if other_paths is not None:
        other_feats = []
        other_regions = []
        for feat_path in other_paths:
            other_feats.append(GenomicFeature(path = feat_path, norm = 'log'))
            other_regions.append(other_feats[-1].get(chr_name, start, end))
    seq_region, ctcf_region, atac_region = get_data_at_interval(chr_name, start, end, seq, ctcf, atac)
    return seq_region, ctcf_region, atac_region, other_regions


def load_data_default(chr_name, seq_path, ctcf_path, atac_path, ctcf_log2=False):
    seq_chr_path = os.path.join(seq_path, f'{chr_name}.fa.gz')
    seq = SequenceFeature(path = seq_chr_path)
    ctcf = GenomicFeature(path = ctcf_path, norm = 'log' if not ctcf_log2 else 'log2')
    atac = None
    if atac_path is not None:
        atac = GenomicFeature(path = atac_path, norm = 'log')

    return seq, ctcf, atac

def get_data_at_interval(chr_name, start, end, seq, ctcf, atac):
    '''
    Slice data from arrays with transformations
    '''
    seq_region = seq.get(start, end)
    ctcf_region = ctcf.get(chr_name, start, end)
    try:
        atac_region = atac.get(chr_name, start, end)
    except RuntimeError:  # no ATAC provided
        atac_region = None
    except AttributeError:  # also no ATAC provided
        atac_region = None
    return seq_region, ctcf_region, atac_region

## Load Model ##
def prediction(seq_region, ctcf_region, atac_region, model_path, other_regions=None, record_attn=False, use_cross_attn=False, num_genomic_features=2, mid_hidden=256):
    model = load_default(model_path, record_attn=record_attn, num_genomic_features=num_genomic_features, use_cross_attn=use_cross_attn, mid_hidden=mid_hidden)
    if other_regions is None:
        inputs = preprocess_default(seq_region, ctcf_region, atac_region)
    else:
        inputs = preprocess_default(seq_region, ctcf_region, atac_region, other_regions)
    if record_attn:
        pred, attn, cross_attn = model(inputs)
        pred = pred[0].detach().cpu().numpy()
        attn = attn.detach().cpu().numpy()
        cross_attn = [c.detach().cpu().numpy() for _, c in cross_attn]
        # symmetrize
        #pred = (pred + pred.T) * 0.5
        return pred, attn, cross_attn
    else:
        pred = model(inputs)[0].detach().cpu().numpy()
        # symmetrize
        #pred = (pred + pred.T) * 0.5
        return pred