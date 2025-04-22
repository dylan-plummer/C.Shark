import os
import numpy as np
import pandas as pd
import sys
import torch
import cooler 
import pyBigWig
from scipy.sparse import coo_matrix

from cshark.data.data_feature import SequenceFeature, GenomicFeature
from cshark.inference.utils.model_utils import load_default



def write_tmp_cooler(pred, chr_name, start, res=8192, window=2097152, out_file='tmp/tmp.cool'):
    bins = pd.DataFrame()
    #bin_range = np.linspace(start, start + window - res, pred.shape[0])
    bin_range = np.arange(0, start + window + res, res)
    bins['start'] = bin_range
    bins['start'] = bins['start'].astype(int)
    bins['end'] = bins['start'] + res
    bins['end'] = bins['end'].astype(int)
    bins['chrom'] = chr_name
    # offset start bin 
    start_offset = int(start / res)

    pixels = pd.DataFrame()
    sparse_mat = coo_matrix(np.triu(pred), dtype=np.float32)
    pixels['bin1_id'] = sparse_mat.row + start_offset
    pixels['bin2_id'] = sparse_mat.col + start_offset
    pixels['count'] = sparse_mat.data 

    pixels.to_csv(out_file + '.csv')

    cooler.create_cooler(out_file, bins, pixels, dtypes={'count': np.float32})


def knockout_peaks(signal_array, threshold=2.0, min_peak_width=5, padding_factor=1.0):
    """
    Simulates knockout of peaks in a signal array by replacing peak regions with background values.
    
    Args:
        signal_array (numpy.ndarray): 1D array containing signal values.
        threshold (float): Minimum signal value to be considered part of a peak.
        min_peak_width (int): Minimum width for a region to be called a peak.
        padding_factor (float): Fraction of peak width to use for background calculation.
            
    Returns:
        numpy.ndarray: Copy of input array with peaks knocked out (replaced with background).
    """
    # Create a copy of the input array to modify
    result = np.copy(signal_array)
    array_length = len(signal_array)
    
    # Find regions above threshold
    above_threshold = signal_array >= threshold
    
    # Track peak regions
    in_peak = False
    peak_start = None
    peaks = []  # Will store (start, end) tuples
    
    # Find peaks
    for i in range(array_length):
        if above_threshold[i]:
            if not in_peak:
                # Start of a new peak
                peak_start = i
                in_peak = True
        else:
            if in_peak:
                # End of current peak
                peak_end = i
                if peak_end - peak_start >= min_peak_width:
                    peaks.append((peak_start, peak_end))
                in_peak = False
    
    # Handle case where array ends during a peak
    if in_peak and array_length - peak_start >= min_peak_width:
        peaks.append((peak_start, array_length))
    
    # Process each peak
    for peak_start, peak_end in peaks:
        peak_width = peak_end - peak_start
        
        # Calculate padding for background, but don't exceed array bounds
        padding = min(int(peak_width * padding_factor), 5)
        
        # Calculate regions before and after peak for background
        pre_start = max(0, peak_start - padding)
        pre_end = peak_start
        
        post_start = peak_end
        post_end = min(array_length, peak_end + padding)
        
        # Calculate mean of surrounding regions as background
        pre_values = signal_array[pre_start:pre_end]
        post_values = signal_array[post_start:post_end]
        
        # Handle empty regions
        pre_mean = np.mean(pre_values) if len(pre_values) > 0 else 0.0
        post_mean = np.mean(post_values) if len(post_values) > 0 else 0.0
        
        # Calculate background value as average of pre and post regions
        background_val = (pre_mean + post_mean) / 2.0
        
        # Replace peak with background value
        result[peak_start:peak_end] = background_val
    
    return result


def get_axis_range_from_bigwig(bigwig_path, chr_name, start, window=2097152, q=0.999):
    bw = pyBigWig.open(bigwig_path)
    values = np.array(bw.values(chr_name, start, start + window))
    values = np.nan_to_num(values, nan = 0.0)
    return int(np.quantile(values, q=q))


def write_tmp_chipseq_ko(bigwig_path, track_name, chr_name, start, deletion_start, deletion_width, ko_mode='zero', window=2097152):
    """
    Write a temporary ctcf bigiwg file with the deletion region perturbed based on the ko_mode
    -Open ctcf_path using pyBigWig
    -Get the region from start to start + window
    >>> bw = bw.open(ctcf_path)
    >>> bw.values(chr_name, start, start + window, numpy=True)
    -convert to log transformed values expected by knockout_peaks function
    -perform the perturbation with knockout_peaks function
    -write the new ctcf to a temporary bigwig file in tmp folder
    -then modify the tmp/tmp_tracks.ini file writing to use this file in a new bigwig track (same for tmp_tracks_diff.ini)
    """
    bw = pyBigWig.open(bigwig_path)
    values = np.array(bw.values(chr_name, start, start + window))
    values = np.nan_to_num(values, nan = 0.0)
    log_values = np.log(values + 1)
    deletion_index_start = deletion_start - start
    deletion_index_end = deletion_start + deletion_width - start
    ko_peaks = np.copy(log_values)

    if ko_mode == 'knockout':
        sub_values = log_values[deletion_index_start:deletion_index_end]
        sub_output = knockout_peaks(sub_values)
        ko_peaks[deletion_index_start:deletion_index_end] = sub_output
    
    if ko_mode == 'zero': 
        ko_peaks[deletion_index_start:deletion_index_end] = 0
    
    ko_peaks = np.exp(ko_peaks) - 1 
    header = bw.chroms().items()
    header_list =list(header)
    bw.close()

    ctcf_ko_bw = pyBigWig.open(f'tmp/{track_name}_ko.bw','w')
    ctcf_ko_bw.addHeader(header_list)
    positions = list(range(start, start+window))
    values = list(ko_peaks)

    # merge intervals
    merged_intervals = []
    prev_pos = positions[0]
    prev_val = values[0]

    for i in range(1,len(positions)):
        curr_val = values[i]
        curr_pos = positions[i]
        
        if curr_val != prev_val :
            merged_intervals.append((prev_pos, positions[i], prev_val))    
            prev_pos = curr_pos
            prev_val = curr_val

    merged_intervals.append((prev_pos, positions[-1] + 1, prev_val))

    for s,e,v in merged_intervals:
         ctcf_ko_bw.addEntries([chr_name],[s],[e],[v])


    ctcf_ko_bw.close()

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
def prediction(seq_region, ctcf_region, atac_region, model_path, other_regions=None, record_attn=False, num_genomic_features=2, mid_hidden=256, undo_log=True):
    model = load_default(model_path, record_attn=record_attn, num_genomic_features=num_genomic_features, mid_hidden=mid_hidden)
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
        pred = (pred + pred.T) * 0.5
        if undo_log:
            pred = np.expm1(pred)
        return pred, attn, cross_attn
    else:
        output = model(inputs)
        if isinstance(output, dict):
            pred = output['hic']
            pred_1d = output['1d']
            if pred_1d is not None:
                pred_1d = pred_1d[0].detach().cpu().numpy()
                if undo_log:
                    pred_1d = np.expm1(pred_1d)
            else:
                pred_1d = None
        else:
            pred = output
            pred_1d = None
        pred = pred[0].detach().cpu().numpy()
        # symmetrize
        pred = (pred + pred.T) * 0.5
        if undo_log:
            pred = np.expm1(pred)
        return {'hic': pred, '1d': pred_1d}