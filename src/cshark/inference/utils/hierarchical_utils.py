"""
Hierarchical RAD21 prediction utilities for C.Shark.

Provides functions to:
  1. Load a hierarchical checkpoint and extract the inner RAD21 predictor
     (``input_pred_model``) that maps 7 input tracks → RAD21.
  2. Predict WT RAD21 from the current (unperturbed) input tracks.
  3. Predict perturbed RAD21 from modified input tracks.
  4. Compute the fold-change / additive delta between WT and perturbed RAD21.
  5. Apply that delta to the real experimental RAD21 bigwig data.

This mirrors the ``enformer_seq`` KO mode but uses the hierarchical C.Shark
inner model instead of Enformer.  The resulting perturbed RAD21 track is then
fed into the main C.Shark model for Hi-C prediction.
"""

import os
import argparse
import numpy as np
import torch
import torch.nn.functional as F

import cshark.model.corigami_models as corigami_models
from cshark.inference.utils.model_utils import get_all_track_names
from cshark.inference.utils.inference_utils import preprocess_default


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def load_hierarchical_rad21_predictor(checkpoint_path, device=None):
    """Load the RAD21 predictor (``input_pred_model``) from a hierarchical
    training checkpoint.

    The checkpoint is expected to have been produced by the
    ``hierarchical_predict.TrainModule`` which stores two sub-models:

    * ``model`` — the main Hi-C predictor (takes all 8 tracks).
    * ``input_pred_model`` — predicts RAD21 from the other 7 tracks
      (sequence + CTCF + ATAC + 5 histones, excluding RAD21).

    Parameters
    ----------
    checkpoint_path : str
        Path to the ``.ckpt`` file produced by hierarchical training.
    device : torch.device or None
        If *None*, uses CUDA when available.

    Returns
    -------
    rad21_model : torch.nn.Module
        The inner RAD21 predictor in eval mode.
    all_track_names : list[str]
        Ordered list of *all* input track names stored in the checkpoint
        (e.g. ``['ctcf', 'atac', 'rad21', 'h3k27ac', ...]``).
    rad21_idx : int
        Index of ``'rad21'`` within *all_track_names*.
    device : torch.device
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print(f"[hierarchical] Loading hierarchical checkpoint from {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    hparams = checkpoint['hyper_parameters']

    # --- Reconstruct the input_pred_model architecture ---
    model_name = hparams.get('model_type', 'MultiTaskConvTransModel')
    ModelClass = getattr(corigami_models, model_name)

    conditioning_vec = hparams.get('conditioning_vec', None)
    conditioning_vec_size = None
    if conditioning_vec is not None:
        conditioning_vec_size = len(conditioning_vec[0].split(','))

    rad21_model = ModelClass(
        num_genomic_features=7,   # 7 input tracks (all except RAD21)
        num_target_tracks=1,      # predicts RAD21
        conditioning_vec_size=conditioning_vec_size,
        mid_hidden=hparams.get('model_latent_dim', 256),
        predict_hic=False,
        diploid=hparams.get('dataset_assembly2', None) is not None,
        predict_1d=True,
        target_mat_size=hparams.get('mat_size', 512),
        target_1d_length=hparams.get('target_1d_size', 8192),
        recon_1d=hparams.get('recon_1d', True),
        seq_filter_size=hparams.get('seq_filter_size', 3),
        activation_1d=None,
    )

    # --- Load weights for the input_pred_model ---
    state_dict = checkpoint['state_dict']
    inner_weights = {}
    for key, value in state_dict.items():
        if key.startswith('input_pred_model.'):
            new_key = key.replace('input_pred_model.', '')
            inner_weights[new_key] = value

    if not inner_weights:
        raise RuntimeError(
            f"No 'input_pred_model.*' keys found in checkpoint {checkpoint_path}. "
            f"Are you sure this is a hierarchical training checkpoint?"
        )

    rad21_model.load_state_dict(inner_weights)
    rad21_model.eval()
    rad21_model.to(device)

    # --- Resolve track names ---
    all_track_names, _, input_tracks = get_all_track_names(checkpoint_path)
    if 'rad21' not in all_track_names:
        raise ValueError(
            f"'rad21' not found in checkpoint track names: {all_track_names}"
        )
    rad21_idx = all_track_names.index('rad21')

    print(f"[hierarchical] RAD21 predictor loaded successfully.")
    print(f"[hierarchical] All tracks: {all_track_names}")
    print(f"[hierarchical] RAD21 index: {rad21_idx}")
    print(f"[hierarchical] Inner model weights: {len(inner_weights)} tensors loaded.")

    return rad21_model, all_track_names, rad21_idx, device


# ---------------------------------------------------------------------------
# RAD21 prediction
# ---------------------------------------------------------------------------

def predict_rad21(rad21_model, inputs_tensor, rad21_idx, device=None):
    """Run the inner model to predict RAD21 from the input tracks
    (with RAD21 channel removed).

    Parameters
    ----------
    rad21_model : torch.nn.Module
        The ``input_pred_model`` extracted from a hierarchical checkpoint.
    inputs_tensor : torch.Tensor, shape (1, seq_len, num_channels)
        Full preprocessed input tensor (sequence + all tracks).
        The RAD21 channel will be stripped internally.
    rad21_idx : int
        Index of the RAD21 channel within the genomic feature channels
        (0-indexed *after* the 5 sequence channels, i.e. the absolute
        channel index is ``5 + rad21_idx``).
    device : torch.device or None

    Returns
    -------
    rad21_pred : np.ndarray, shape (output_bins,)
        Predicted RAD21 signal (already ``expm1``-transformed from log1p
        space, clipped to non-negative).
    """
    if device is None:
        device = next(rad21_model.parameters()).device

    inputs = inputs_tensor.to(device)

    # Remove the RAD21 channel (absolute index = 5 + rad21_idx)
    abs_idx = 5 + rad21_idx
    inputs_no_rad21 = torch.cat([
        inputs[:, :, :abs_idx],
        inputs[:, :, abs_idx + 1:]
    ], dim=2)

    with torch.no_grad():
        output = rad21_model(inputs_no_rad21)

    # output['1d'] shape: (1, output_bins, 1)
    pred_1d = output['1d']
    rad21_pred = torch.expm1(pred_1d[:, :, 0]).squeeze().cpu().numpy()
    rad21_pred = np.clip(rad21_pred, 0, None)

    return rad21_pred


# ---------------------------------------------------------------------------
# Delta computation
# ---------------------------------------------------------------------------

def compute_hierarchical_rad21_delta(rad21_model, wt_inputs, ko_inputs,
                                     rad21_idx, device=None, epsilon=0.01):
    """Compute fold-change and additive delta for RAD21 between WT and
    perturbed (KO) inputs.

    Parameters
    ----------
    rad21_model : torch.nn.Module
    wt_inputs : torch.Tensor, shape (1, seq_len, num_channels)
        Preprocessed WT input tensor.
    ko_inputs : torch.Tensor, shape (1, seq_len, num_channels)
        Preprocessed perturbed input tensor (after track modifications).
    rad21_idx : int
    device : torch.device or None
    epsilon : float

    Returns
    -------
    fold_change : np.ndarray, shape (output_bins,)
        KO / WT ratio.  >1 = gain, <1 = loss.
    delta : np.ndarray, shape (output_bins,)
        Additive difference KO − WT.
    wt_pred : np.ndarray, shape (output_bins,)
    ko_pred : np.ndarray, shape (output_bins,)
    """
    wt_pred = predict_rad21(rad21_model, wt_inputs, rad21_idx, device=device)
    ko_pred = predict_rad21(rad21_model, ko_inputs, rad21_idx, device=device)

    delta = ko_pred - wt_pred
    fold_change = ko_pred / np.clip(wt_pred, epsilon, None)

    return fold_change, delta, wt_pred, ko_pred


# ---------------------------------------------------------------------------
# Apply delta to experimental RAD21
# ---------------------------------------------------------------------------

def apply_rad21_delta(experimental_rad21, fold_change, delta,
                      mode='additive', cap=None):
    """Apply the hierarchical-model-predicted RAD21 delta to experimental
    RAD21 data.

    Parameters
    ----------
    experimental_rad21 : np.ndarray, shape (track_len,)
        Real experimental RAD21 signal (e.g. from a bigwig, already in
        the model's input space — typically log1p-transformed).
    fold_change : np.ndarray, shape (pred_bins,)
        Per-bin fold-change (KO / WT) from the hierarchical model.
    delta : np.ndarray, shape (pred_bins,)
        Per-bin additive delta (KO − WT) from the hierarchical model.
    mode : str
        ``'multiplicative'`` — multiply experimental by fold_change.
        ``'additive'``       — add the raw delta to experimental.
    cap : float
        Upper bound on fold-change (and 1/cap lower bound) to prevent
        extreme outliers.

    Returns
    -------
    perturbed_rad21 : np.ndarray, shape (track_len,)
        Modified RAD21 signal.
    """
    track_len = len(experimental_rad21)
    result = experimental_rad21.copy()

    # Resample delta/fold_change to match experimental resolution
    if len(fold_change) != track_len:
        fold_change = _resample(fold_change, track_len)
    if len(delta) != track_len:
        delta = _resample(delta, track_len)

    if mode == 'multiplicative':
        if cap is not None:
            fold_change = np.clip(fold_change, 1.0 / cap, cap)
        print(np.min(fold_change), np.mean(fold_change), np.max(fold_change))
        print(np.min(result), np.mean(result), np.max(result))
        result = np.expm1(result)  # Convert from log1p to linear space
        result = result * fold_change
        result = np.log1p(result)  # Back to log1p space
    elif mode == 'additive':
        result = result + delta
    else:
        raise ValueError(
            f"Unknown delta mode: '{mode}'. Use 'multiplicative' or 'additive'."
        )

    #result = np.clip(result, 0, None)
    return result


def _resample(signal, target_length):
    """Resample a 1D signal to *target_length* using linear interpolation."""
    t = torch.tensor(signal, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
    resampled = F.interpolate(t, size=target_length, mode='linear',
                              align_corners=True)
    return resampled.squeeze().numpy()


# ---------------------------------------------------------------------------
# High-level integration function for perturb.py
# ---------------------------------------------------------------------------

def hierarchical_rad21_update(rad21_model, rad21_idx,
                              seq_region, ctcf_region, atac_region,
                              other_regions,
                              seq_region_ko, ctcf_region_ko, atac_region_ko,
                              other_regions_ko,
                              experimental_rad21,
                              input_track_names,
                              delta_mode='additive', cap=None,
                              window=2097152):
    """End-to-end hierarchical RAD21 update for the perturbation pipeline.

    1. Build WT and KO input tensors from the provided track arrays.
    2. Predict RAD21 for both using the hierarchical inner model.
    3. Compute fold-change / delta.
    4. Apply delta to the experimental RAD21 track.
    5. Replace RAD21 in ``other_regions_ko`` with the perturbed values.

    Parameters
    ----------
    rad21_model : torch.nn.Module
        Inner RAD21 predictor from :func:`load_hierarchical_rad21_predictor`.
    rad21_idx : int
        Index of RAD21 in the all-tracks list.
    seq_region, ctcf_region, atac_region : np.ndarray
        WT (unperturbed) input arrays.
    other_regions : list[np.ndarray] or None
        WT other feature arrays (in input_track_names[2:] order).
    seq_region_ko, ctcf_region_ko, atac_region_ko : np.ndarray
        Perturbed input arrays (after applying KO to whichever tracks).
    other_regions_ko : list[np.ndarray] or None
        Perturbed other feature arrays.
    experimental_rad21 : np.ndarray
        Real experimental RAD21 signal at the model's input resolution.
    input_track_names : list[str]
        Ordered names of input tracks (e.g. ``['ctcf', 'atac', 'rad21', ...]``).
    delta_mode : str
    cap : float
    window : int

    Returns
    -------
    other_regions_ko : list[np.ndarray]
        Updated other_regions with RAD21 replaced by the perturbed prediction.
    hierarchical_results : dict
        Diagnostic info: ``fold_change``, ``delta``, ``wt_pred``, ``ko_pred``,
        ``perturbed_rad21``.
    """
    device = next(rad21_model.parameters()).device

    # Build full input tensors
    wt_inputs = preprocess_default(seq_region, ctcf_region, atac_region,
                                   other_regions).to(device)
    ko_inputs = preprocess_default(seq_region_ko, ctcf_region_ko, atac_region_ko,
                                   other_regions_ko).to(device)

    # Compute RAD21 delta
    fold_change, delta, wt_pred, ko_pred = compute_hierarchical_rad21_delta(
        rad21_model, wt_inputs, ko_inputs, rad21_idx, device=device
    )

    # Apply delta to experimental RAD21
    if delta_mode == 'prediction':
        print(f"[hierarchical] Using raw hierarchical model prediction as perturbed RAD21.")
        perturbed_rad21 = ko_pred
        #perturbed_rad21 = np.clip(perturbed_rad21, 0, None)
        perturbed_rad21 = _resample(perturbed_rad21, len(experimental_rad21))
    else:
        perturbed_rad21 = apply_rad21_delta(
            experimental_rad21, fold_change, delta,
            mode=delta_mode, cap=cap,
        )

    # Determine where RAD21 sits in other_regions
    # input_track_names is like ['ctcf', 'atac', 'rad21', 'h3k27ac', ...]
    # other_regions corresponds to input_track_names[2:] (everything after ctcf/atac)
    other_offset = 0
    if 'ctcf' in input_track_names:
        other_offset += 1
    if 'atac' in input_track_names:
        other_offset += 1
    other_track_names = input_track_names[other_offset:]

    if 'rad21' in other_track_names and other_regions_ko is not None:
        rad21_other_idx = other_track_names.index('rad21')
        other_regions_ko[rad21_other_idx] = perturbed_rad21
        print(f"[hierarchical] RAD21 replaced in other_regions_ko at index {rad21_other_idx}")
    else:
        print(f"[hierarchical] Warning: 'rad21' not found in other_track_names "
              f"{other_track_names}. RAD21 not updated.")

    hierarchical_results = {
        'fold_change': fold_change,
        'delta': delta,
        'wt_pred': wt_pred,
        'ko_pred': ko_pred,
        'perturbed_rad21': perturbed_rad21,
    }

    print(f"[hierarchical] RAD21 delta applied (mode={delta_mode}).")
    print(f"[hierarchical] WT RAD21 mean={wt_pred.mean():.4f}, "
          f"KO RAD21 mean={ko_pred.mean():.4f}, "
          f"perturbed mean={perturbed_rad21.mean():.4f}")

    return other_regions_ko, hierarchical_results


# ---------------------------------------------------------------------------
# BigWig writing for diagnostics
# ---------------------------------------------------------------------------

def write_tmp_hierarchical_rad21_bigwig(base_bigwig_path, wt_pred, ko_pred,
                                        perturbed_rad21,
                                        chr_name, start, window=2097152):
    """Write diagnostic bigwigs for the hierarchical RAD21 predictions.

    Produces three files in ``tmp/``:
    * ``rad21_hierarchical_wt_pred.bw`` — WT predicted RAD21
    * ``rad21_hierarchical_ko_pred.bw`` — KO predicted RAD21
    * ``rad21_hierarchical_perturbed.bw`` — perturbed experimental RAD21

    Parameters
    ----------
    base_bigwig_path : str
        Path to any existing bigwig (used for chromosome header).
    wt_pred, ko_pred, perturbed_rad21 : np.ndarray
    chr_name : str
    start : int
    window : int
    """
    import pyBigWig

    bw = pyBigWig.open(base_bigwig_path)
    header_list = list(bw.chroms().items())
    bw.close()

    for name, signal in [('wt_pred', wt_pred),
                         ('ko_pred', ko_pred),
                         ('perturbed', np.expm1(perturbed_rad21))]:  # Convert back from log1p for visualization
        # Resample to bp resolution for bigwig
        if len(signal) != window:
            signal = _resample(signal, window)

        out_path = f'tmp/rad21_hierarchical_{name}.bw'
        out_bw = pyBigWig.open(out_path, 'w')
        out_bw.addHeader(header_list)

        positions = list(range(start, start + window))
        values = list(signal.astype(float))

        # Merge intervals for efficient writing
        merged = []
        prev_pos = positions[0]
        prev_val = values[0]
        for i in range(1, len(positions)):
            if values[i] != prev_val:
                merged.append((prev_pos, positions[i], prev_val))
                prev_pos = positions[i]
                prev_val = values[i]
        merged.append((prev_pos, positions[-1] + 1, prev_val))

        for s, e, v in merged:
            out_bw.addEntries([chr_name], [s], [e], [float(v)])

        out_bw.close()
        print(f"[hierarchical] Wrote {out_path}")


def write_tmp_hierarchical_delta_bigwig(base_bigwig_path, delta, fc,
                                         chr_name, start, window=2097152):
    """Write the raw RAD21 delta as a bigwig (centered around zero).

    Output: ``tmp/rad21_hierarchical_delta.bw``

    Parameters
    ----------
    base_bigwig_path : str
    delta : np.ndarray
    chr_name : str
    start : int
    window : int
    """
    import pyBigWig

    bw = pyBigWig.open(base_bigwig_path)
    header_list = list(bw.chroms().items())
    bw.close()

    signal = delta.copy()
    if len(signal) != window:
        signal = _resample(signal, window)

    out_path = 'tmp/rad21_hierarchical_delta.bw'
    out_bw = pyBigWig.open(out_path, 'w')
    out_bw.addHeader(header_list)

    positions = list(range(start, start + window))
    values = list(signal.astype(float))

    merged = []
    prev_pos = positions[0]
    prev_val = values[0]
    for i in range(1, len(positions)):
        if values[i] != prev_val:
            merged.append((prev_pos, positions[i], prev_val))
            prev_pos = positions[i]
            prev_val = values[i]
    merged.append((prev_pos, positions[-1] + 1, prev_val))

    for s, e, v in merged:
        out_bw.addEntries([chr_name], [s], [e], [float(v)])

    out_bw.close()
    print(f"[hierarchical] Wrote {out_path}")

    # Also write fold-change bigwig for reference
    fc_out_path = 'tmp/rad21_hierarchical_fold_change.bw'
    out_bw = pyBigWig.open(fc_out_path, 'w')
    out_bw.addHeader(header_list)
    fc_signal = fc.copy()
    if len(fc_signal) != window:
        fc_signal = _resample(fc_signal, window)
    fc_values = list(fc_signal.astype(float))
    merged = []
    prev_pos = positions[0]
    prev_val = fc_values[0]
    for i in range(1, len(positions)):
        if fc_values[i] != prev_val:
            merged.append((prev_pos, positions[i], prev_val))
            prev_pos = positions[i]
            prev_val = fc_values[i]
    merged.append((prev_pos, positions[-1] + 1, prev_val))
    for s, e, v in merged:
        out_bw.addEntries([chr_name], [s], [e], [float(v)])
    out_bw.close()
    print(f"[hierarchical] Wrote {fc_out_path}")