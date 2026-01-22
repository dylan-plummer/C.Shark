import os
import numpy as np
import pandas as pd
import sys
import torch
import cooler 
import pyBigWig
from scipy.sparse import coo_matrix
from skimage.transform import resize

from cshark.data.data_feature import SequenceFeature, GenomicFeature
from cshark.inference.utils.model_utils import load_default


def oe_normalize_cooler(c, max_strata=512, dummy=1e-7):
    """
    Performs distance (strata) based observed/expected (O/E) ratio correction
    on a matrix in a vectorized and efficient manner.

    Args:
        mat (np.ndarray): The input matrix.
        max_strata (int): The maximum diagonal distance to normalize.
        dummy (float): A small value to add to avoid division by zero.

    Returns:
        np.ndarray: The normalized matrix.
    """
    data = c.matrix(balance=False, sparse=True)
    chrom = list(c.chromnames)[0]
    # main loop
    mat = data.fetch(chrom).toarray()
    # Ensure input is a NumPy array and handle NaNs
    mat = np.nan_to_num(np.asarray(mat))
    mat[mat < 0] = 0  # Set negative values to zero
    n = mat.shape[0]

    # Ensure the matrix is square
    if n != mat.shape[1]:
        raise ValueError("Input matrix must be square.")

    # Cap max_strata to the actual number of diagonals in the matrix
    num_diags = min(max_strata, n)

    # Calculate the mean of positive values for each diagonal up to num_diags
    # A list comprehension is a clean way to do this
    averages = np.array([
        np.mean(diag[diag > 0]) if np.any(diag > 0) else 0
        for diag in (np.diagonal(mat, offset=i) for i in range(num_diags))
    ])

    # Replace any calculated zeros with 1 to avoid division by zero
    averages[averages == 0] = 1

    # --- Vectorized creation of the 'expected' matrix ---
    # Create an array of average values indexed by distance from the diagonal.
    # For distances >= num_diags, we will not normalize (i.e., divide by 1).
    avg_by_dist = np.ones(n)
    avg_by_dist[:num_diags] = averages
    # Calculate the distance from the main diagonal for each matrix element
    rows, cols = np.indices(mat.shape)
    dist = np.abs(rows - cols)
    # Build the 'expected' matrix using advanced indexing
    expected_mat = avg_by_dist[dist]
    normalized_mat = (mat + dummy) / (expected_mat + dummy)
    # only keep upper triangle
    normalized_mat = np.triu(normalized_mat)
    normalized_mat = coo_matrix(normalized_mat)
    normalized_pixels = pd.DataFrame()
    normalized_pixels['bin1_id'] = normalized_mat.row
    normalized_pixels['bin2_id'] = normalized_mat.col
    normalized_pixels['count'] = normalized_mat.data
    return normalized_pixels


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


def knockout_peaks(signal_array, threshold=2.0, min_peak_width=5, padding_factor=3.0, background_q=0.1, increase_factor=None):
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
        pre_end = peak_start - peak_width
        
        post_start = peak_end + peak_width
        post_end = min(array_length, peak_end + padding)
        
        # Calculate mean of surrounding regions as background
        pre_values = signal_array[pre_start:pre_end]
        post_values = signal_array[post_start:post_end]
        
        # Handle empty regions
        # pre_mean = np.quantile(pre_values, q=background_q) if len(pre_values) > 0 else 0.0
        # post_mean = np.quantile(post_values, q=background_q) if len(post_values) > 0 else 0.0
        pre_mean = np.mean(pre_values) if len(pre_values) > 0 else 0.0
        post_mean = np.mean(post_values) if len(post_values) > 0 else 0.0
        
        # Calculate background value as average of pre and post regions
        background_val = (pre_mean + post_mean) / 2.0

        background_val = min(background_val, 1.0)  # Cap background value to 1.0
        
        if increase_factor is not None:
            # Increase peak value by the specified factor
            result[peak_start:peak_end] = signal_array[peak_start:peak_end] * increase_factor
        else:
            # Replace peak with background value
            result[peak_start:peak_end] = background_val
            #result[peak_start:peak_end] = 1
    
    return result


def get_axis_range_from_bigwig(bigwig_path, chr_name, start, window=2097152, q=0.995):
    bw = pyBigWig.open(bigwig_path)
    values = np.array(bw.values(chr_name, start, start + window))
    values = np.nan_to_num(values, nan = 0.0)
    lim = np.quantile(values, q=q)
    return lim if lim != 0 else None


def chunk_shuffle(arr, chunk_size=1000):
    """
    Split into chunks and shuffle the chunks
    """
    arr = np.array(arr)
    n_chunks = len(arr) // chunk_size
    chunks = np.array_split(arr, n_chunks)
    np.random.shuffle(chunks)
    return np.concatenate(chunks)

def write_tmp_pred_bigwig(base_bigwig_path, pred_values, track_name, chr_name, start, suffix='pred', window=2097152):
    bw = pyBigWig.open(base_bigwig_path)
    values = np.array(bw.values(chr_name, start, start + window))
    values = np.nan_to_num(values, nan = 0.0)
    
    header = bw.chroms().items()
    header_list =list(header)
    bw.close()

    ctcf_ko_bw = pyBigWig.open(f'tmp/{track_name}_{suffix}.bw','w')
    ctcf_ko_bw.addHeader(header_list)
    positions = list(range(start, start+window))
    # extend pred_values to the same length as positions by repeating values
    pred_values = np.array(pred_values)
    if len(pred_values) < len(positions):
        pred_values = resize(pred_values, (len(positions),)).squeeze()
    values = list(pred_values)
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
         ctcf_ko_bw.addEntries([chr_name],[s],[e],[float(v)])


    ctcf_ko_bw.close()


def write_tmp_chipseq_ko(bigwig_path, track_name, chr_name, start, deletion_start, deletion_width, ko_mode='zero', window=2097152, peak_height=2.0):
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
        sub_output = knockout_peaks(sub_values, threshold=peak_height)
        ko_peaks[deletion_index_start:deletion_index_end] = sub_output

    if 'increase' in ko_mode:
        if '_' not in ko_mode:
            increase_factor = 2.0
        increase_factor = float(ko_mode.split('_')[1])
        sub_values = log_values[deletion_index_start:deletion_index_end]
        sub_output = knockout_peaks(sub_values, threshold=peak_height, increase_factor=increase_factor)
        ko_peaks[deletion_index_start:deletion_index_end] = sub_output

    if 'cluster' in ko_mode:
        # add a cluster of peaks in the region
        if '_' not in ko_mode:
            cluster_ratio = 0.05
        else:
            cluster_ratio = float(ko_mode.split('_')[1])
        cluster_indices = np.random.choice(np.arange(deletion_index_start, deletion_index_end), size=int((deletion_index_end - deletion_index_start) * cluster_ratio), replace=False)
        for idx in cluster_indices:
            # add a peak at idx
            ko_peaks[idx] = np.random.uniform(1, 5)

    
    if ko_mode == 'zero': 
        ko_peaks[deletion_index_start:deletion_index_end] = 0

    if ko_mode == 'shuffle':
        sub_values = log_values[deletion_index_start:deletion_index_end]
        sub_output = chunk_shuffle(sub_values)
        ko_peaks[deletion_index_start:deletion_index_end] = sub_output

    if ko_mode == 'knockout_shuffle':
        sub_values = log_values[deletion_index_start:deletion_index_end]
        sub_output = knockout_peaks(sub_values, threshold=peak_height)
        sub_output = chunk_shuffle(sub_output)
        ko_peaks[deletion_index_start:deletion_index_end] = sub_output
    
    if ko_mode == 'reverse' or ko_mode == 'reverse_motif':
        sub_values = log_values[deletion_index_start:deletion_index_end]
        sub_output = np.flip(sub_values)
        ko_peaks[deletion_index_start:deletion_index_end] = sub_output
    
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
    features = []
    if ctcf is not None:
        ctcf = torch.tensor(np.nan_to_num(ctcf, 0)) # Important! replace nan with 0
        features.append(ctcf)
    if atac is not None:
        atac_log = torch.tensor(atac) # Important! replace nan with 0
        # Merge inputs
        features.append(atac_log)
    if other is not None:
        for other_region in other:
            other_feat = torch.tensor(np.nan_to_num(other_region, 0))
            features.append(other_feat)
    if len(features) == 0:
        inputs = seq
    else:
        features = torch.cat([feat.unsqueeze(0).unsqueeze(2) for feat in features], dim = 2)
        inputs = torch.cat([seq, features], dim = 2)
    # Move input to gpu if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    inputs = inputs.to(device)
    return inputs

## Load data ##
def load_region(chr_name, start, seq_path, ctcf_path, atac_path, other_paths=None, seq2_path=None, window = 2097152, ctcf_log2=False,
                bigwig_log=True):
    ''' Single loading method for one region '''
    end = start + window
    seq, ctcf, atac = load_data_default(chr_name, seq_path, ctcf_path, atac_path, ctcf_log2=ctcf_log2, 
                                        bigwig_log=bigwig_log)
    other_regions = None
    if other_paths is not None:
        other_feats = []
        other_regions = []
        for feat_path in other_paths:
            other_feats.append(GenomicFeature(path = feat_path, norm = 'log' if bigwig_log else None))
            other_regions.append(other_feats[-1].get(chr_name, start, end))
    seq_region, ctcf_region, atac_region = get_data_at_interval(chr_name, start, end, seq, ctcf, atac)
    if seq2_path is not None:
        seq_chr_path = os.path.join(seq2_path, f'{chr_name}.fa.gz')
        seq2 = SequenceFeature(path = seq_chr_path)
        seq2_region = seq2.get(start, end)
        seq_region = np.concatenate((seq_region, seq2_region), axis=1)
    return seq_region, ctcf_region, atac_region, other_regions


def load_data_default(chr_name, seq_path, ctcf_path, atac_path, ctcf_log2=False,
                      bigwig_log=True):
    seq_chr_path = os.path.join(seq_path, f'{chr_name}.fa.gz')
    seq = SequenceFeature(path = seq_chr_path)
    ctcf = None
    if ctcf_path is not None:
        ctcf_log = 'log' if not ctcf_log2 else 'log2'
        if not bigwig_log:
            ctcf_log = None
        ctcf = GenomicFeature(path = ctcf_path, norm = ctcf_log)
    atac = None
    if atac_path is not None:
        atac = GenomicFeature(path = atac_path, norm = 'log' if bigwig_log else None)

    return seq, ctcf, atac

def get_data_at_interval(chr_name, start, end, seq, ctcf, atac):
    '''
    Slice data from arrays with transformations
    '''
    seq_region = seq.get(start, end)
    try:
        ctcf_region = ctcf.get(chr_name, start, end)
    except RuntimeError:  # no CTCF provided
        ctcf_region = None
    except AttributeError:  # also no CTCF provided
        ctcf_region = None
    try:
        atac_region = atac.get(chr_name, start, end)
    except RuntimeError:  # no ATAC provided
        atac_region = None
    except AttributeError:  # also no ATAC provided
        atac_region = None
    return seq_region, ctcf_region, atac_region

## Load Model ##
def prediction(seq_region, ctcf_region, atac_region, model_path, 
               other_regions=None, diploid=False, record_attn=False, 
               num_genomic_features=2, mat_size=256, mid_hidden=256, target_1d_length=8192,
               bigwig_log=True,
               undo_log=True, seq_filter_size=3, recon_1d=True,
               other_feat_names=None):
    model = load_default(model_path, record_attn=record_attn, num_genomic_features=num_genomic_features, 
                         mat_size=mat_size, diploid=diploid, mid_hidden=mid_hidden, 
                         target_1d_length=target_1d_length,
                         seq_filter_size=seq_filter_size, recon_1d=recon_1d)
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
        try:
            output = model(inputs)
        except TypeError:
            inputs = {
                    'seq': inputs[..., :5],
                    'ctcf': inputs[..., 5:6],
                    'atac': inputs[..., 6:7]
                }
            if other_feat_names is not None and other_regions is not None:
                for i, other_feat in enumerate(other_feat_names):
                    inputs[other_feat] = torch.tensor(other_regions[i]).unsqueeze(0).unsqueeze(2).to(inputs['seq'].device)
            output = model(inputs,
                predict_tracks=['ctcf', 'atac', 'rad21', 'h3k27ac', 'h3k4me3', 'h3k9me3', 'h3k36me3', 'h3k27me3', 'myc', 'nanog', 'yy1', 'polr2a', 'rnaseq', 'hic']
            )
        if isinstance(output, dict):
            pred = output['hic']
            pred_1d = output['1d']
            if pred_1d is not None:
                if isinstance(pred_1d, dict):
                    tmp_1d = [v.detach().cpu().numpy() for v in pred_1d.values()]
                    pred_1d = np.concatenate(tmp_1d, axis=0).transpose()
                else:
                    pred_1d = pred_1d[0].detach().cpu().numpy()
                if bigwig_log:
                    pred_1d = np.expm1(pred_1d)
                    pred_1d = np.clip(pred_1d, 0, None)
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