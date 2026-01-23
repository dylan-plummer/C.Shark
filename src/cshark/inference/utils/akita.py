import json

from cooltools.lib.numutils import set_diag
import numpy as np
import pandas as pd
import tensorflow as tf
from basenji import seqnn
from skimage.transform import resize

# 1Mb
WINDOW_SIZE = 1048576
model_dir = '/home/dmp131/basenji/manuscripts/akita/'
params_file = model_dir+'params.json'
model_file  = model_dir+'model_best.h5'
with open(params_file) as params_open:
    params = json.load(params_open)
    params_model = params['model']
    params_train = params['train']

seqnn_model = seqnn.SeqNN(params_model)
seqnn_model.restore(model_file)
print('successfully loaded Akita')

def from_upper_triu(vector_repr, matrix_len, num_diags):
    z = np.zeros((matrix_len,matrix_len))
    triu_tup = np.triu_indices(matrix_len,num_diags)
    z[triu_tup] = vector_repr
    for i in range(-num_diags+1,num_diags):
        set_diag(z, np.nan, i)
    return z + z.T

data_dir =   '/home/dmp131/basenji/manuscripts/akita/data/'
hic_targets = pd.read_csv(data_dir+'/targets.txt',sep='\t')
hic_file_dict_num = dict(zip(hic_targets['index'].values, hic_targets['file'].values) )
hic_file_dict     = dict(zip(hic_targets['identifier'].values, hic_targets['file'].values) )
hic_num_to_name_dict = dict(zip(hic_targets['index'].values, hic_targets['identifier'].values) )

# read data parameters
data_stats_file = '%s/statistics.json' % data_dir
with open(data_stats_file) as data_stats_open:
    data_stats = json.load(data_stats_open)

seq_length = data_stats['seq_length']
target_length = data_stats['target_length']
hic_diags =  data_stats['diagonal_offset']
target_crop = data_stats['crop_bp'] // data_stats['pool_width']
target_length1 = data_stats['seq_length'] // data_stats['pool_width']
crop_bp = data_stats['crop_bp']

target_length1_cropped = target_length1 - 2*target_crop

def akita_pred(seq, model=seqnn_model, akita_res=2048, target_res=4096, akita_idx=1):
    """Make Akita prediction on a one-hot encoded sequence.

    Args:
        seq: One-hot encoded sequence of shape (L, 4) where L is the sequence length.
        model: Pre-loaded Basenji seqnn model for Akita.
    Returns:
        Dictionary with 'hic' key containing the predicted Hi-C contact matrix.
    """
    seq = np.expand_dims(seq, axis=0)  # Add batch dimension
    if seq.shape[-1] == 5:
        seq = seq[:, :, :4]  # Remove N channel if present
    # Reorder: ATCG -> ACGT
    seq = seq[:, :, [0, 2, 3, 1]]
    # predict in overlapping windows and average for final pred
    step_size = WINDOW_SIZE // 4
    final_output_size = int((seq.shape[1] / akita_res))
    target_output_size = int((seq.shape[1] / target_res))
    mat = np.zeros((final_output_size + final_output_size // 2, final_output_size+ final_output_size // 2))
    count_mat = np.zeros((final_output_size + final_output_size // 2, final_output_size+ final_output_size // 2))
    for start in range(0, seq.shape[1], step_size):
        end = min(start + WINDOW_SIZE, seq.shape[1])
        seq_chunk = seq[:, start:end, :]
        pred_chunk = model.predict(tf.convert_to_tensor(seq_chunk, dtype=tf.float32), verbose=0)
        pred_mat = from_upper_triu(pred_chunk[:,:,akita_idx], target_length1_cropped, hic_diags)
        # Map predictions to final matrix
        hic_pred = pred_mat
        chunk_size = hic_pred.shape[0]
        start_idx = int((start + crop_bp) // akita_res)
        end_idx = start_idx + chunk_size
        mat[start_idx:end_idx, start_idx:end_idx] += hic_pred
        count_mat[start_idx:end_idx, start_idx:end_idx] += 1
        if end == seq.shape[1]:
            break
    # Average overlapping predictions
    count_mat[count_mat == 0] = 1  # Prevent division by zero
    hic_pred = mat / count_mat
    mat = mat[:final_output_size, :final_output_size]
    # resize to target output size
    mat = resize(mat, (target_output_size, target_output_size), order=1, mode='reflect', anti_aliasing=True)
    return {'hic': mat}