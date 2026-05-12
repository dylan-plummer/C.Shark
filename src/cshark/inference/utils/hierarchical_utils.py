"""
Hierarchical prediction utilities for C.Shark.

Provides functions to:
  1. Load a hierarchical checkpoint — either an old-style RAD21-only inner
     model or a CSharkUniversalModel that can predict any missing tracks.
  2. Predict WT / KO tracks and compute deltas.
  3. Apply deltas to experimental data.
  4. Write diagnostic bigwigs for every predicted track.

When the checkpoint is a CSharkUniversalModel the user only needs to provide
CTCF + ATAC (plus any other tracks they have); all remaining tracks required
by the final 8-track Hi-C model are predicted automatically.
"""

import os
import argparse
import numpy as np
import torch
import torch.nn.functional as F

import cshark.model.corigami_models as corigami_models
from cshark.inference.utils.model_utils import get_all_track_names, load_default
from cshark.inference.utils.inference_utils import preprocess_default


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def load_hierarchical_rad21_predictor(checkpoint_path, device=None, n_input_tracks=None):
    """Load the RAD21 predictor (``input_pred_model``) from a hierarchical
    training checkpoint.

    The checkpoint is expected to have been produced by the
    ``hierarchical_predict.TrainModule`` which stores two sub-models:

    * ``model`` — the main Hi-C predictor (takes all 8 tracks).
    * ``input_pred_model`` — predicts RAD21 from the other 7 tracks
      (sequence + CTCF + ATAC + histones, excluding RAD21).

    Parameters
    ----------
    checkpoint_path : str
        Path to the ``.ckpt`` file produced by hierarchical training.
    device : torch.device or None
        If *None*, uses CUDA when available.
    n_input_tracks : int or None
        Number of genomic-feature input channels for the inner model
        (all tracks except RAD21).  If *None* (default), this is
        auto-detected from the checkpoint's stored track names.

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

    # --- Resolve track names early so we can auto-detect n_input_tracks ---
    all_track_names, _, input_tracks = get_all_track_names(checkpoint_path)
    if 'rad21' not in all_track_names:
        raise ValueError(
            f"'rad21' not found in checkpoint track names: {all_track_names}"
        )
    rad21_idx = all_track_names.index('rad21')

    if n_input_tracks is None:
        # The inner model takes all tracks except RAD21
        n_input_tracks = len(all_track_names) - 1
        print(f"[hierarchical] Auto-detected n_input_tracks={n_input_tracks} "
              f"from checkpoint tracks: {all_track_names}")

    # --- Reconstruct the input_pred_model architecture ---
    model_name = hparams.get('model_type', 'MultiTaskConvTransModel')
    ModelClass = getattr(corigami_models, model_name)

    conditioning_vec = hparams.get('conditioning_vec', None)
    conditioning_vec_size = None
    if conditioning_vec is not None:
        conditioning_vec_size = len(conditioning_vec[0].split(','))

    rad21_model = ModelClass(
        num_genomic_features=n_input_tracks,   # n_input_tracks input tracks (all except RAD21)
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
        # raise RuntimeError(
        #     f"No 'input_pred_model.*' keys found in checkpoint {checkpoint_path}. "
        #     f"Are you sure this is a hierarchical training checkpoint?"
        # )
        # provided checkpoint is already a MultiTaskConvTransModel checkpoint, not a hierarchical TrainModule checkpoint
        print(f"[hierarchical] Warning: No 'input_pred_model.*' keys found in checkpoint. Assuming checkpoint is already the inner model.")
        inner_weights = {}
        for key, value in state_dict.items():
            if key.startswith('model.'):
                new_key = key.replace('model.', '')
                inner_weights[new_key] = value

    rad21_model.load_state_dict(inner_weights)
    rad21_model.eval()
    rad21_model.to(device)

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


# ===========================================================================
# Universal hierarchical mode
# ===========================================================================

def _is_universal_checkpoint(checkpoint_path):
    """Return True if the checkpoint is a CSharkUniversalModel."""
    device = torch.device('cpu')
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    state_dict = checkpoint.get('state_dict', {})
    # CSharkUniversalModel has 'model.track_embedders.*' keys
    for key in state_dict:
        if 'track_embedders' in key or 'modality_embeddings' in key:
            return True
    return False


def load_hierarchical_predictor(checkpoint_path, device=None):
    """Auto-detect checkpoint type and load accordingly.

    If the checkpoint is a CSharkUniversalModel, load it as a universal
    predictor that can predict any missing tracks.  Otherwise, fall back
    to the legacy RAD21-only ``load_hierarchical_rad21_predictor``.

    Returns
    -------
    result : dict with keys:
        ``'model'``           — the loaded model (eval mode, on device).
        ``'is_universal'``    — bool, True if CSharkUniversalModel.
        ``'all_track_names'`` — list[str], all tracks known to the model.
        ``'input_track_names'`` — list[str], input tracks of the model.
        ``'target_track_names'`` — list[str], tracks the model can predict.
        ``'rad21_idx'``       — int or None (only for legacy mode).
        ``'device'``          — torch.device.
    """
    is_universal = _is_universal_checkpoint(checkpoint_path)

    if is_universal:
        return _load_universal_predictor(checkpoint_path, device=device)
    else:
        model, all_tracks, rad21_idx, dev = load_hierarchical_rad21_predictor(
            checkpoint_path, device=device
        )
        return {
            'model': model,
            'is_universal': False,
            'all_track_names': all_tracks,
            'input_track_names': all_tracks,  # legacy model trained on all tracks
            'target_track_names': ['rad21'],
            'rad21_idx': rad21_idx,
            'device': dev,
        }


def _load_universal_predictor(checkpoint_path, device=None):
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    all_track_names, target_track_names, input_track_names = get_all_track_names(
        checkpoint_path
    )

    print(f"[hierarchical-universal] Loading universal model from {checkpoint_path}")
    print(f"[hierarchical-universal] Input tracks:  {input_track_names}")
    print(f"[hierarchical-universal] Target tracks: {target_track_names}")

    model = load_default(
        checkpoint_path,
        num_genomic_features=len(input_track_names),
        target_1d_length=16384,  # 16k input bins at 128bp resolution for 2Mb window
    )
    model.eval()
    model.to(device)

    return {
        'model': model,
        'is_universal': True,
        'all_track_names': all_track_names,
        'input_track_names': input_track_names,
        'target_track_names': target_track_names,
        'rad21_idx': None,
        'device': device,
    }


def _predict_tracks_universal(model, seq_region, provided_tracks,
                               predict_track_names, device=None):
    """Run the universal model to predict a set of 1D tracks.

    Parameters
    ----------
    model : CSharkUniversalModel
    seq_region : np.ndarray, shape (seq_len, 5|10)
    provided_tracks : dict[str, np.ndarray]
        Track name → 1D signal for user-provided tracks.
    predict_track_names : list[str]
    device : torch.device or None

    Returns
    -------
    predictions : dict[str, np.ndarray]
        Track name → predicted signal (log1p space, matching model training).
    """
    if device is None:
        device = next(model.parameters()).device

    input_dict = {
        'seq': torch.tensor(seq_region, dtype=torch.float32)
               .unsqueeze(0).to(device),
    }
    for name, arr in provided_tracks.items():
        input_dict[name] = (
            torch.tensor(arr, dtype=torch.float32)
            .unsqueeze(0).unsqueeze(2).to(device)
        )

    with torch.no_grad():
        output = model(input_dict, predict_tracks=predict_track_names)

    pred_1d = output.get('1d', {})
    predictions = {}
    for name in predict_track_names:
        if name in pred_1d:
            predictions[name] = pred_1d[name].squeeze().cpu().numpy()

    return predictions


def _build_provided_tracks(seq_region, ctcf_region, atac_region,
                           other_regions, input_track_names):
    """Build a ``{track_name: array}`` dict from the raw region arrays."""
    other_offset = 0
    if 'ctcf' in input_track_names:
        other_offset += 1
    if 'atac' in input_track_names:
        other_offset += 1
    other_track_names = input_track_names[other_offset:]

    tracks = {}
    if ctcf_region is not None and 'ctcf' in input_track_names:
        tracks['ctcf'] = ctcf_region
    if atac_region is not None and 'atac' in input_track_names:
        tracks['atac'] = atac_region
    if other_regions is not None:
        for idx, tname in enumerate(other_track_names):
            # Keep the first occurrence for duplicated names so core inputs
            # (ctcf/atac) are not overwritten by malformed metadata.
            if tname in tracks:
                continue
            tracks[tname] = other_regions[idx]
    return tracks


def hierarchical_universal_update(
    hier_info,
    seq_region_wt, ctcf_region_wt, atac_region_wt, other_regions_wt,
    seq_region_ko, ctcf_region_ko, atac_region_ko, other_regions_ko,
    input_track_names,
    hic_model_track_names,
    user_provided_names=None,
    delta_mode='multiplicative', cap=None,
    window=2097152,
):
    """End-to-end hierarchical update using a universal model.

    Determines which tracks need predicting (tracks in *hic_model_track_names*
    that the user did not provide), predicts them for WT and KO, computes
    deltas, applies deltas to experimental data, and replaces entries in
    ``other_regions_ko``.

    Parameters
    ----------
    hier_info : dict
        Output of :func:`load_hierarchical_predictor` (universal mode).
    seq_region_wt, ctcf_region_wt, atac_region_wt : np.ndarray
    other_regions_wt : list[np.ndarray] or None
    seq_region_ko, ctcf_region_ko, atac_region_ko : np.ndarray
    other_regions_ko : list[np.ndarray] or None
    input_track_names : list[str]
        The user-provided input track names for the Hi-C model.
    hic_model_track_names : list[str]
        All track names the final Hi-C model expects.
    delta_mode, cap : str, float
    window : int

    Returns
    -------
    other_regions_ko : list[np.ndarray]
    hier_results : dict[str, dict]
        Per-track diagnostics.
    """
    model = hier_info['model']
    target_tracks = hier_info['target_track_names']

    # Tracks to predict = needed by Hi-C model but not originally provided
    # by the user.  When fill_missing_tracks has already added predicted
    # tracks to input_track_names, use user_provided_names to know what the
    # user *actually* supplied so we still compute WT/KO deltas for filled
    # tracks.
    _provided = user_provided_names if user_provided_names is not None else input_track_names
    tracks_to_predict = [t for t in hic_model_track_names
                         if t not in _provided and t in target_tracks]
    # Also always include tracks the universal model can predict that ARE in
    # the user's input list (so we capture the perturbation effect on all tracks)
    # — but ONLY if the user doesn't already provide experimental data for them
    # (the delta approach handles propagation).
    print(f"[hierarchical-universal] Tracks to predict: {tracks_to_predict}")

    if not tracks_to_predict:
        print("[hierarchical-universal] No tracks to predict, skipping.")
        return other_regions_ko, {}

    # Build WT/KO track dicts (only user-provided tracks go in)
    provided_wt = _build_provided_tracks(
        seq_region_wt, ctcf_region_wt, atac_region_wt,
        other_regions_wt, input_track_names,
    )
    provided_ko = _build_provided_tracks(
        seq_region_ko, ctcf_region_ko, atac_region_ko,
        other_regions_ko, input_track_names,
    )

    # Predict
    wt_preds = _predict_tracks_universal(model, seq_region_wt, provided_wt,
                                          tracks_to_predict)
    ko_preds = _predict_tracks_universal(model, seq_region_ko, provided_ko,
                                          tracks_to_predict)

    # Compute deltas and apply
    other_offset = 0
    if 'ctcf' in input_track_names:
        other_offset += 1
    if 'atac' in input_track_names:
        other_offset += 1
    other_track_names = input_track_names[other_offset:]

    hier_results = {}
    epsilon = 0.01

    for name in tracks_to_predict:
        if name not in wt_preds or name not in ko_preds:
            continue

        wt = wt_preds[name]
        ko = ko_preds[name]
        delta = ko - wt
        fc = ko / np.clip(wt, epsilon, None)

        # Get experimental data if available
        if name in other_track_names and other_regions_wt is not None:
            exp_idx = other_track_names.index(name)
            experimental = other_regions_wt[exp_idx].copy()
        else:
            experimental = None

        # Apply delta
        if delta_mode == 'prediction':
            perturbed = ko.copy()
            if experimental is not None and len(perturbed) != len(experimental):
                perturbed = _resample(perturbed, len(experimental))
        elif experimental is not None:
            perturbed = apply_rad21_delta(experimental, fc, delta,
                                         mode=delta_mode, cap=cap)
        else:
            # No experimental data — use raw prediction
            perturbed = ko.copy()

        # Place perturbed track into other_regions_ko
        if name in other_track_names and other_regions_ko is not None:
            tidx = other_track_names.index(name)
            other_regions_ko[tidx] = perturbed
            print(f"[hierarchical-universal] Replaced {name} in other_regions at index {tidx}")

        hier_results[name] = {
            'wt_pred': wt,
            'ko_pred': ko,
            'fold_change': fc,
            'delta': delta,
            'perturbed': perturbed,
            'experimental': experimental,
        }

        print(f"[hierarchical-universal] {name}: WT mean={wt.mean():.4f}, "
              f"KO mean={ko.mean():.4f}, perturbed mean={perturbed.mean():.4f}")

    return other_regions_ko, hier_results


def write_tmp_hierarchical_bigwigs(base_bigwig_path, hier_results,
                                   chr_name, start, window=2097152):
    """Write diagnostic bigwigs for every track predicted by the hierarchical
    model.

    For each track produces:
    * ``tmp/{track}_hierarchical_wt_pred.bw``
    * ``tmp/{track}_hierarchical_ko_pred.bw``
    * ``tmp/{track}_hierarchical_perturbed.bw``
    * ``tmp/{track}_hierarchical_delta.bw``
    * ``tmp/{track}_hierarchical_fold_change.bw``
    """
    import pyBigWig

    bw = pyBigWig.open(base_bigwig_path)
    header_list = list(bw.chroms().items())
    bw.close()

    for track_name, info in hier_results.items():
        for suffix, signal_raw in [
            ('wt_pred', info['wt_pred']),
            ('ko_pred', info['ko_pred']),
            ('perturbed', info['perturbed']),
            ('delta', info['delta']),
            ('fold_change', info['fold_change']),
        ]:
            signal = signal_raw.copy()
            if len(signal) != window:
                signal = _resample(signal, window)

            out_path = f'tmp/{track_name}_hierarchical_{suffix}.bw'
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


def fill_missing_tracks(
    hic_model_path,
    hierarchical_model_path,
    seq_region,
    ctcf_region,
    atac_region,
    other_regions,
    other_feats,
    input_track_names,
    bigwig_log=True,
    hier_info=None,
    verbose=True,
    hic_input_tracks=None,
):
    """Predict any tracks the Hi-C model requires but the user did not provide.

    Uses either:
    * a universal hierarchical model to predict any missing supported track, or
    * a legacy RAD21-only hierarchical model to fill a missing ``rad21`` track.

    Returns augmented
    ``other_regions``, ``other_feats``, and ``input_track_names`` that include
    the newly predicted tracks in the order the Hi-C model expects.

    Parameters
    ----------
    hic_model_path : str
        Path to the main Hi-C prediction checkpoint.
    hierarchical_model_path : str
        Path to the CSharkUniversalModel checkpoint.
    seq_region, ctcf_region, atac_region : np.ndarray
    other_regions : list[np.ndarray] or None
    other_feats : list[str] or None
        File paths corresponding to ``other_regions``.
    input_track_names : list[str]
        Current user-provided track names (e.g. ``['ctcf', 'atac', 'rad21', ...]``).
    bigwig_log : bool
        Whether signals are in log1p space.
    hier_info : dict or None
        Pre-loaded hierarchical predictor (from ``load_hierarchical_predictor``).
        If None, the model will be loaded from ``hierarchical_model_path``.

    Returns
    -------
    other_regions : list[np.ndarray]
    other_feats : list[str]
    input_track_names : list[str]
    num_genomic_features : int
    """
    # Determine which tracks the Hi-C model was trained with
    if hic_input_tracks is None:
        try:
            _, _, hic_input_tracks = get_all_track_names(hic_model_path)
        except Exception:
            hic_input_tracks = []

    if not hic_input_tracks:
        # Fallback: infer count from checkpoint weight shape
        device = torch.device('cpu')
        ckpt = torch.load(hic_model_path, map_location=device, weights_only=False)
        sd = {k.replace('model.', ''): v for k, v in ckpt['state_dict'].items()}
        epi_key = next((k for k in sd if 'conv_start_epi.0.weight' in k), None)
        if epi_key is not None:
            expected_count = sd[epi_key].shape[1]
        else:
            expected_count = len(input_track_names)
        del ckpt
        # If counts already match, nothing to do
        if len(input_track_names) >= expected_count:
            num_gf = 2 + (len(other_regions) if other_regions else 0)
            return other_regions, other_feats, input_track_names, num_gf
        # Without named tracks we cannot determine what to predict
        print(f"[fill-missing] Hi-C model expects {expected_count} tracks but "
              f"{len(input_track_names)} provided. Cannot determine missing "
              f"track names — checkpoint has no input_features metadata.")
        num_gf = 2 + (len(other_regions) if other_regions else 0)
        return other_regions, other_feats, input_track_names, num_gf

    missing = [t for t in hic_input_tracks if t not in input_track_names]
    if not missing:
        num_gf = 2 + (len(other_regions) if other_regions else 0)
        return other_regions, other_feats, input_track_names, num_gf

    if verbose:
        print(f"[fill-missing] Hi-C model expects tracks: {hic_input_tracks}")
        print(f"[fill-missing] User provided:             {input_track_names}")
        print(f"[fill-missing] Missing tracks to predict:  {missing}")

    # Load hierarchical model (universal or legacy RAD21-only)
    if hier_info is None:
        hier_info = load_hierarchical_predictor(hierarchical_model_path)

    provided_tracks = {}
    if ctcf_region is not None:
        provided_tracks['ctcf'] = ctcf_region
    if atac_region is not None:
        provided_tracks['atac'] = atac_region
    other_offset = 0
    if 'ctcf' in input_track_names:
        other_offset += 1
    if 'atac' in input_track_names:
        other_offset += 1
    other_track_names = input_track_names[other_offset:]
    if other_regions is not None:
        for idx, tname in enumerate(other_track_names):
            if tname in provided_tracks:
                continue
            provided_tracks[tname] = other_regions[idx]

    predictions = {}
    if hier_info['is_universal']:
        predictions = _predict_tracks_universal(
            hier_info['model'], seq_region, provided_tracks, missing,
        )
    else:
        # Legacy hierarchical model can only provide RAD21.
        if missing != ['rad21']:
            if verbose:
                print("[fill-missing] Legacy hierarchical model only predicts rad21; "
                      f"cannot fill missing tracks: {missing}")
            num_gf = 2 + (len(other_regions) if other_regions else 0)
            return other_regions, other_feats, input_track_names, num_gf

        legacy_all_tracks = hier_info.get('all_track_names', [])
        if 'rad21' not in legacy_all_tracks:
            if verbose:
                print("[fill-missing] Legacy hierarchical model has no rad21 track metadata.")
            num_gf = 2 + (len(other_regions) if other_regions else 0)
            return other_regions, other_feats, input_track_names, num_gf

        missing_inputs = [
            t for t in legacy_all_tracks
            if t != 'rad21' and t not in provided_tracks
        ]
        if missing_inputs:
            if verbose:
                print("[fill-missing] Cannot predict rad21 with legacy model; "
                      f"missing required inputs: {missing_inputs}")
            num_gf = 2 + (len(other_regions) if other_regions else 0)
            return other_regions, other_feats, input_track_names, num_gf

        ctcf_for_legacy = provided_tracks.get('ctcf')
        atac_for_legacy = provided_tracks.get('atac')
        if ctcf_for_legacy is None or atac_for_legacy is None:
            if verbose:
                print("[fill-missing] Legacy rad21 predictor requires both ctcf and atac inputs.")
            num_gf = 2 + (len(other_regions) if other_regions else 0)
            return other_regions, other_feats, input_track_names, num_gf

        # Build features in the exact order expected by preprocess_default:
        # [ctcf, atac] + other_tracks, where other_tracks includes rad21 slot.
        legacy_other_tracks = [t for t in legacy_all_tracks if t not in ('ctcf', 'atac')]
        legacy_other_regions = []
        for tname in legacy_other_tracks:
            if tname == 'rad21':
                legacy_other_regions.append(np.zeros(len(seq_region), dtype=np.float32))
            else:
                legacy_other_regions.append(provided_tracks[tname])

        legacy_inputs = preprocess_default(
            seq_region,
            ctcf_for_legacy,
            atac_for_legacy,
            legacy_other_regions,
        )
        rad21_pred = predict_rad21(
            hier_info['model'],
            legacy_inputs,
            hier_info['rad21_idx'],
            device=hier_info.get('device'),
        )

        if len(rad21_pred) != len(seq_region):
            rad21_pred = _resample(rad21_pred, len(seq_region))
        predictions['rad21'] = rad21_pred
        if verbose:
            print(f"[fill-missing] Predicted rad21 with legacy model: mean={rad21_pred.mean():.4f}")

    # Insert predicted tracks into other_regions in the order expected by the
    # Hi-C model. We reconstruct other_regions following hic_input_tracks order
    # (excluding ctcf/atac which are handled separately).
    hic_other = [t for t in hic_input_tracks if t not in ('ctcf', 'atac')]
    new_other_regions = []
    new_other_feats = []
    # Build new_input_names from scratch so it stays aligned with
    # new_other_regions (which follows hic_other order).
    new_input_names = []
    if 'ctcf' in input_track_names:
        new_input_names.append('ctcf')
    if 'atac' in input_track_names:
        new_input_names.append('atac')

    window = len(seq_region)  # use seq length for resampling target

    for tname in hic_other:
        if tname in provided_tracks:
            # Already provided by user — find it in existing other_regions
            idx = other_track_names.index(tname) if tname in other_track_names else None
            if idx is not None:
                new_other_regions.append(other_regions[idx])
                new_other_feats.append(other_feats[idx] if other_feats else f'predicted_{tname}.bw')
                new_input_names.append(tname)
            else:
                # It was ctcf or atac — skip (handled separately)
                continue
        elif tname in predictions:
            pred = predictions[tname]
            # Resample to match expected length
            if len(pred) != window:
                pred = _resample(pred, window)
            new_other_regions.append(pred)
            new_other_feats.append(f'predicted_{tname}.bw')
            new_input_names.append(tname)
            if verbose:
                print(f"[fill-missing] Predicted {tname}: mean={pred.mean():.4f}")
        else:
            # Could not predict — insert zeros
            if verbose:
                print(f"[fill-missing] WARNING: Could not predict {tname}, inserting zeros")
            new_other_regions.append(np.zeros(window, dtype=np.float32))
            new_other_feats.append(f'zero_{tname}.bw')
            new_input_names.append(tname)

    num_gf = 2 + len(new_other_regions)
    if verbose:
        print(f"[fill-missing] Final track count: {num_gf} (ctcf + atac + {len(new_other_regions)} others)")

    return new_other_regions, new_other_feats, new_input_names, num_gf