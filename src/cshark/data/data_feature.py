import gzip
import numpy as np
import pyBigWig as pbw


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

class Feature():

    def __init__(self, **kwargs):
        self.load(**kwargs)
    
    def load(self):
        raise Exception('Not implemented')

    def get(self):
        raise Exception('Not implemented')

    def __len__(self):
        raise Exception('Not implemented')

class HiCFeature(Feature):

    def load(self, path = None):
        self.hic = self.load_hic(path)

    def get(self, start, window = 2097152, res = 10000):
        start_bin = int(start / res)
        range_bin = int(window / res)
        end_bin = start_bin + range_bin
        hic_mat = self.diag_to_mat(self.hic, start_bin, end_bin)
        return hic_mat

    def load_hic(self, path):
        #print(f'Reading Hi-C: {path}')
        return dict(np.load(path))

    def diag_to_mat(self, ori_load, start, end):
        '''
        Only accessing 256 x 256 region max, two loops are okay
        '''
        square_len = end - start
        diag_load = {}
        for diag_i in range(square_len):
            diag_load[str(diag_i)] = ori_load[str(diag_i)][start : start + square_len - diag_i]
            diag_load[str(-diag_i)] = ori_load[str(-diag_i)][start : start + square_len - diag_i]
        start -= start
        end -= start

        diag_region = []
        for diag_i in range(square_len):
            diag_line = []
            for line_i in range(-1 * diag_i, -1 * diag_i + square_len):
                if line_i < 0:
                    diag_line.append(diag_load[str(line_i)][start + line_i + diag_i])
                else:
                    diag_line.append(diag_load[str(line_i)][start + diag_i])
            diag_region.append(diag_line)
        diag_region = np.array(diag_region).reshape(square_len, square_len)
        return diag_region

    def __len__(self):
        return len(self.hic['0'])

class GenomicFeatureSingleThread(Feature):

    def __init__(self, path, norm, knockout=False):
        self.path = path
        self.load(path)
        self.norm = norm
        self.knockout = knockout
        # check if path is valid
        self.track_present = True  # used to load mask values
        try:
            with pbw.open(path) as bw_file:
                bw_file.close()
        except:
            self.track_present = False
        #print(f'{path} track present: {self.track_present}')
        #print(f'Feature path: {path} \n Normalization status: {norm}')

    def load(self, path):
        self.feature = self.read_feature(path)

    def get(self, chr_name, start, end):
        feature = self.feature_to_npy(chr_name, start, end)
        if self.track_present:
            feature = np.nan_to_num(feature, 0) # Important! replace nan with 0
            if self.norm == 'log':
                feature = np.log(feature + 1)
            elif self.norm == 'log2':
                feature = np.log2(feature + 1)
            elif self.norm is None:
                feature = feature
            else:
                raise Exception(f'Norm type {self.norm} undefined')
        if self.knockout:
            feature = knockout_peaks(feature, threshold=0.5)
        return feature

    def read_feature(self, path):
        '''
        read bigwig file
        '''
        bw_file = pbw.open(path)
        return bw_file

    def feature_to_npy(self, chr_name, start, end):
        signals = self.feature.values(chr_name, start, end)
        return np.array(signals)

    def length(self, chr_name):
        return self.feature.chroms(chr_name)

class GenomicFeature(GenomicFeatureSingleThread):

    def __init__(self, path, norm, knockout=False):
        self.path = path
        self.norm = norm
        self.knockout = knockout
        # check if path is valid
        self.track_present = True  # used to load mask values
        try:
            with pbw.open(path) as bw_file:
                bw_file.close()
        except:
            self.track_present = False
        #print(f'{path} track present: {self.track_present}')
        #print(f'Feature path: {path} \n Normalization status: {norm}')

    def load(self, path):
        raise Exception('Left blank')

    def feature_to_npy(self, chr_name, start, end):
        if self.track_present:
            with pbw.open(self.path) as bw_file:
                signals = bw_file.values(chr_name, int(start), int(end))
            return np.array(signals)
        else:
            length = end - start
            return np.array([-1] * length)

    def length(self, chr_name):
        if self.track_present:
            with pbw.open(self.path) as bw_file:
                length = bw_file.chroms(chr_name)
            return length
        else:
            return 0

class SequenceFeature(Feature):

    def load(self, path = None):
        self.seq = self.read_seq(path)

    def get(self, start, end):
        seq = self.seq_to_npy(self.seq, start, end)
        onehot_seq = self.onehot_encode(seq)
        return onehot_seq

    def __len__(self):
        return len(self.seq)

    def read_seq(self, dna_dir):
        '''
        Transform fasta data to numpy array
        
        Args:
            dna_dir (str): Directory to DNA .fa path

        Returns:
            array: A numpy char array that contains DNA for a chromosome
        '''
        #print(f'Reading sequence: {dna_dir}')
        with gzip.open(dna_dir, 'r') as f:
            seq = f.read().decode("utf-8")
        seq = seq[seq.find('\n'):]
        seq = seq.replace('\n', '').lower()
        return seq
        
    def seq_to_npy(self, seq, start, end):
        '''
        Transform fasta data to integer numpy array
        
        Args:
            dna_dir (str): Directory to DNA .fa path

        Returns:
            array: A numpy char array that contains DNA for a chromosome
        '''
        seq = seq[start : end]
        en_dict = {'a' : 0, 't' : 1, 'c' : 2, 'g' : 3, 'n' : 4, 'y': 4, 'r':4, 'w':4, 's':4, 'm':4, 'k':4, 'h':4, 'b':4, 'v':4, '-':4}
        en_seq = [en_dict[ch] for ch in seq]
        np_seq = np.array(en_seq, dtype = int)
        return np_seq

    def onehot_encode(self, seq):
        ''' 
        encode integer dna array to onehot (n x 5)
        Args:
            seq (arr): Numpy array (n x 1) of dna encoded as 0-4 integers

        Returns:
            array: A numpy matrix (n x 5)
        '''
        seq_emb = np.zeros((len(seq), 5))
        seq_emb[np.arange(len(seq)), seq] = 1
        return seq_emb

if __name__ == '__main__':
    main()