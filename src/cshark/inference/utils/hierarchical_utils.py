"""
Hierarchical prediction utilities for C.Shark (legacy RAD21 predictor).

Provides functions to:
  1. Load the RAD21-only inner model from a hierarchical training checkpoint.
  2. Predict WT / KO RAD21 tracks and compute deltas.
  3. Apply deltas to experimental RAD21 data.
  4. Write diagnostic bigwigs for the predicted tracks.

All intermediate track values are kept in log1p space for model inputs.
Bigwig outputs are converted back to linear space for visualization.
"""

import os
import numpy as np
import torch
import torch.nn.functional as F

import cshark.model.corigami_models as corigami_models
from cshark.inference.utils.model_utils import get_all_track_names
from cshark.inference.utils.inference_utils import preprocess_default


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def load_hierarchical_rad21_predictor(checkpoint_path, device=None, n_input_tracks=None):
    """Load the RAD21 predictor (``input_pred_model``) from a hierarchical
    training checkpoint.

    Parameters
    ----------
    checkpoint_path : str
        Path to the ``.ckpt`` file produced by hierarchical training.
    device : torch.device or None
    n_input_tracks : int or None
        Auto-detected from checkpoint if None.

    Returns
    -------
    rad21_model : torch.nn.Module
    all_track_names : list[str]
    rad21_idx : int
    device : torch.device
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print(f"[hierarchical] Loading hierarchical checkpoint from {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    hparams = checkpoint['hyper_parameters']

    all_track_names, _, input_tracks = get_all_track_names(checkpoint_path)
    if 'rad21' not in all_track_names:
        raise ValueError(
            f"'rad21' not found in checkpoint track names: {all_track_names}"
        )
    rad21_idx = all_track_names.index('rad21')

    if n_input_tracks is None:
        # The inner model consumes every input track EXCEPT rad21 (which it predicts).
        # Two checkpoint conventions exist and both must work:
        #   - rad21 listed only in output_features -> input_tracks already excludes it
        #   - rad21 listed in input_features too   -> subtract it here
        n_input_tracks = len(input_tracks) - (1 if 'rad21' in input_tracks else 0)
        print(f"[hierarchical] Auto-detected n_input_tracks={n_input_tracks} "
              f"from checkpoint tracks: {input_tracks}")

    model_name = hparams.get('model_type', 'MultiTaskConvTransModel')
    ModelClass = getattr(corigami_models, model_name)

    conditioning_vec = hparams.get('conditioning_vec', None)
    conditioning_vec_size = None
    if conditioning_vec is not None:
        conditioning_vec_size = len(conditioning_vec[0].split(','))

    rad21_model = ModelClass(
        num_genomic_features=n_input_tracks,
        num_target_tracks=1,
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

    state_dict = checkpoint['state_dict']
    inner_weights = {}
    for key, value in state_dict.items():
        if key.startswith('input_pred_model.'):
            inner_weights[key.replace('input_pred_model.', '')] = value

    if not inner_weights:
        print(f"[hierarchical] Warning: No 'input_pred_model.*' keys found. "
              f"Assuming checkpoint is already the inner model.")
        for key, value in state_dict.items():
            if key.startswith('model.'):
                inner_weights[key.replace('model.', '')] = value

    rad21_model.load_state_dict(inner_weights)
    rad21_model.eval()
    rad21_model.to(device)

    print(f"[hierarchical] RAD21 predictor loaded. Tracks: {all_track_names}, RAD21 idx: {rad21_idx}")
    return rad21_model, all_track_names, rad21_idx, device


# ---------------------------------------------------------------------------
# RAD21 prediction
# ---------------------------------------------------------------------------

def predict_rad21(rad21_model, inputs_tensor, rad21_idx, device=None):
    """Run the inner model to predict RAD21 from the input tracks.

    Parameters
    ----------
    rad21_idx : int or None
        Index of RAD21 in the full input tensor (will be removed before passing
        to the model).  Pass ``None`` when the RAD21 track is already absent
        from ``inputs_tensor`` (prediction-only mode).

    Returns the predicted RAD21 signal in **linear space** (expm1 applied).
    """
    if device is None:
        device = next(rad21_model.parameters()).device

    inputs = inputs_tensor.to(device)

    if rad21_idx is not None:
        # Remove the experimental RAD21 channel so the model doesn't see it
        abs_idx = 5 + rad21_idx
        inputs_no_rad21 = torch.cat([
            inputs[:, :, :abs_idx],
            inputs[:, :, abs_idx + 1:]
        ], dim=2)
    else:
        # RAD21 already excluded from inputs; pass directly
        inputs_no_rad21 = inputs

    with torch.no_grad():
        output = rad21_model(inputs_no_rad21)

    # Model outputs log1p space; convert to linear for consistent delta computation
    pred_1d = output['1d']
    rad21_pred = torch.expm1(pred_1d[:, :, 0]).squeeze().cpu().numpy()
    rad21_pred = np.clip(rad21_pred, 0, None)

    return rad21_pred  # linear space


# ---------------------------------------------------------------------------
# Delta computation
# ---------------------------------------------------------------------------

def compute_hierarchical_rad21_delta(rad21_model, wt_inputs, ko_inputs,
                                     rad21_idx, device=None, epsilon=0.01):
    """Compute fold-change and additive delta for RAD21 (both in linear space).

    Returns
    -------
    fold_change : np.ndarray  — KO / WT ratio (linear)
    delta : np.ndarray        — KO − WT additive difference (linear)
    wt_pred : np.ndarray      — linear space
    ko_pred : np.ndarray      — linear space
    """
    wt_pred = predict_rad21(rad21_model, wt_inputs, rad21_idx, device=device)
    ko_pred = predict_rad21(rad21_model, ko_inputs, rad21_idx, device=device)

    delta = ko_pred - wt_pred
    fold_change = ko_pred / np.clip(wt_pred, epsilon, None)

    return fold_change, delta, wt_pred, ko_pred


# ---------------------------------------------------------------------------
# Apply delta to experimental RAD21
# ---------------------------------------------------------------------------

def apply_rad21_delta(experimental_rad21_log1p, fold_change, delta,
                      mode='multiplicative', cap=None):
    """Apply the hierarchical-model-predicted delta to experimental RAD21 data.

    Parameters
    ----------
    experimental_rad21_log1p : np.ndarray
        Real experimental RAD21 signal in **log1p space** (as loaded from bigwig
        with log normalization).
    fold_change : np.ndarray  — linear-space fold change (KO / WT)
    delta : np.ndarray        — linear-space additive delta (KO − WT)
    mode : str
        ``'multiplicative'`` — apply fold change in linear space.
        ``'additive'``       — apply additive delta in linear space.
    cap : float or None
        Fold-change cap for multiplicative mode.

    Returns
    -------
    perturbed : np.ndarray
        Perturbed RAD21 signal in **log1p space** (for model input).
    """
    track_len = len(experimental_rad21_log1p)
    result = experimental_rad21_log1p.copy()

    if len(fold_change) != track_len:
        fold_change = _resample(fold_change, track_len)
    if len(delta) != track_len:
        delta = _resample(delta, track_len)

    # All modes: work in linear space, return log1p
    result_linear = np.expm1(result)

    if mode == 'multiplicative':
        if cap is not None:
            fold_change = np.clip(fold_change, 1.0 / cap, cap)
        print(f"[hierarchical] FC range: [{np.min(fold_change):.3f}, {np.max(fold_change):.3f}], "
              f"mean={np.mean(fold_change):.3f}")
        result_linear = result_linear * fold_change
    elif mode == 'additive':
        result_linear = result_linear + delta
    else:
        raise ValueError(f"Unknown delta mode: '{mode}'. Use 'multiplicative' or 'additive'.")

    result_linear = np.clip(result_linear, 0, None)
    return np.log1p(result_linear)  # back to log1p space


def _resample(signal, target_length):
    """Resample a 1D signal to *target_length* using linear interpolation."""
    t = torch.tensor(signal, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
    resampled = F.interpolate(t, size=target_length, mode='linear', align_corners=True)
    return resampled.squeeze().numpy()


# ---------------------------------------------------------------------------
# High-level integration function for perturb.py
# ---------------------------------------------------------------------------

def hierarchical_rad21_update(rad21_model, rad21_idx,
                              seq_region, ctcf_region, atac_region,
                              other_regions,
                              seq_region_ko, ctcf_region_ko, atac_region_ko,
                              other_regions_ko,
                              experimental_rad21_log1p,
                              input_track_names,
                              delta_mode='multiplicative', cap=None,
                              window=2097152):
    """End-to-end hierarchical RAD21 update for the perturbation pipeline.

    1. Build WT and KO input tensors.
    2. Predict RAD21 for both (linear space).
    3. Compute fold-change / delta.
    4. (If experimental RAD21 available) Apply delta; replace RAD21 in
       ``other_regions_ko`` with perturbed values (log1p).

    When ``experimental_rad21_log1p`` is ``None`` (RAD21 not in input bigwigs),
    the function operates in **prediction-only** mode: predictions are returned
    for bigwig visualization but ``other_regions_ko`` is not modified.

    The returned ``hierarchical_results['perturbed_rad21']`` is in **log1p space**
    (or ``None`` in prediction-only mode).

    Parameters
    ----------
    rad21_idx : int or None
        Position of RAD21 in the full input tensor.  ``None`` when RAD21 is
        absent from the bigwig inputs (tracks already exclude it).
    experimental_rad21_log1p : np.ndarray or None
        Experimental RAD21 in log1p space.  ``None`` → prediction-only mode.
    delta_mode : str
        ``'multiplicative'``, ``'additive'``, or ``'prediction'``.
    """
    device = next(rad21_model.parameters()).device

    wt_inputs = preprocess_default(seq_region, ctcf_region, atac_region,
                                   other_regions).to(device)
    ko_inputs = preprocess_default(seq_region_ko, ctcf_region_ko, atac_region_ko,
                                   other_regions_ko).to(device)

    fold_change, delta, wt_pred, ko_pred = compute_hierarchical_rad21_delta(
        rad21_model, wt_inputs, ko_inputs, rad21_idx, device=device
    )

    if experimental_rad21_log1p is None:
        # Prediction-only mode: no experimental track to perturb
        perturbed_rad21 = None
        print(f"[hierarchical] prediction-only mode (no experimental RAD21). "
              f"WT mean={wt_pred.mean():.4f}, KO mean={ko_pred.mean():.4f}")
    elif delta_mode == 'prediction':
        # Use raw model prediction — convert to log1p for consistent space
        perturbed_rad21 = np.log1p(np.clip(ko_pred, 0, None))
        perturbed_rad21 = _resample(perturbed_rad21, len(experimental_rad21_log1p))
    else:
        perturbed_rad21 = apply_rad21_delta(
            experimental_rad21_log1p, fold_change, delta,
            mode=delta_mode, cap=cap,
        )

    # Replace RAD21 in other_regions_ko when we have a perturbed value
    if perturbed_rad21 is not None and other_regions_ko is not None:
        other_offset = sum(1 for t in ['ctcf', 'atac'] if t in input_track_names)
        other_track_names = input_track_names[other_offset:]
        if 'rad21' in other_track_names:
            rad21_other_idx = other_track_names.index('rad21')
            other_regions_ko[rad21_other_idx] = perturbed_rad21  # log1p for model input
            print(f"[hierarchical] RAD21 replaced at index {rad21_other_idx}, "
                  f"delta_mode={delta_mode}, "
                  f"WT mean={wt_pred.mean():.4f}, KO mean={ko_pred.mean():.4f}, "
                  f"perturbed mean (log1p)={perturbed_rad21.mean():.4f}")

    hierarchical_results = {
        'fold_change': fold_change,
        'delta': delta,
        'wt_pred': wt_pred,              # linear space
        'ko_pred': ko_pred,              # linear space
        'perturbed_rad21': perturbed_rad21,  # log1p space, or None
    }

    return other_regions_ko, hierarchical_results


# ---------------------------------------------------------------------------
# BigWig writing for diagnostics
# ---------------------------------------------------------------------------

def _write_bw_signal(out_path, header_list, chr_name, start, signal, window):
    """Write a 1D signal array to a bigwig file (signal must be in linear space)."""
    import pyBigWig

    if len(signal) != window:
        signal = _resample(signal, window)

    out_bw = pyBigWig.open(out_path, 'w')
    out_bw.addHeader(header_list)

    positions = list(range(start, start + window))
    values = list(signal.astype(float))

    merged = []
    prev_pos, prev_val = positions[0], values[0]
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


def write_tmp_hierarchical_rad21_bigwig(base_bigwig_path, wt_pred, ko_pred,
                                        perturbed_rad21_log1p,
                                        chr_name, start, window=2097152):
    """Write diagnostic bigwigs for the hierarchical RAD21 predictions.

    Writes to ``tmp/``:
    * ``rad21_hierarchical_wt_pred.bw``     — WT predicted RAD21 (linear)
    * ``rad21_hierarchical_ko_pred.bw``     — KO predicted RAD21 (linear)
    * ``rad21_hierarchical_perturbed.bw``   — perturbed experimental RAD21 (linear)
                                             (skipped when ``perturbed_rad21_log1p`` is None)

    Parameters
    ----------
    wt_pred, ko_pred : np.ndarray
        Model predictions in **linear space** (from predict_rad21).
    perturbed_rad21_log1p : np.ndarray or None
        Perturbed experimental RAD21 in **log1p space**; converted to linear here.
        Pass ``None`` in prediction-only mode (no experimental RAD21 bigwig).
    """
    import pyBigWig

    bw = pyBigWig.open(base_bigwig_path)
    header_list = list(bw.chroms().items())
    bw.close()

    os.makedirs('tmp', exist_ok=True)

    # wt_pred and ko_pred are already in linear space
    for name, signal in [('wt_pred', wt_pred), ('ko_pred', ko_pred)]:
        _write_bw_signal(f'tmp/rad21_hierarchical_{name}.bw',
                         header_list, chr_name, start, signal.copy(), window)

    # perturbed_rad21 is in log1p space — convert to linear for visualization
    if perturbed_rad21_log1p is not None:
        perturbed_linear = np.expm1(np.clip(perturbed_rad21_log1p, 0, None))
        _write_bw_signal('tmp/rad21_hierarchical_perturbed.bw',
                         header_list, chr_name, start, perturbed_linear, window)


def write_tmp_hierarchical_delta_bigwig(base_bigwig_path, delta, fc,
                                        chr_name, start, window=2097152):
    """Write the RAD21 delta and fold-change as bigwigs.

    Outputs (in ``tmp/``):
    * ``rad21_hierarchical_delta.bw``       — additive delta (linear)
    * ``rad21_hierarchical_fold_change.bw`` — fold change (linear ratio)
    """
    import pyBigWig

    bw = pyBigWig.open(base_bigwig_path)
    header_list = list(bw.chroms().items())
    bw.close()

    os.makedirs('tmp', exist_ok=True)
    _write_bw_signal('tmp/rad21_hierarchical_delta.bw',
                     header_list, chr_name, start, delta.copy(), window)
    _write_bw_signal('tmp/rad21_hierarchical_fold_change.bw',
                     header_list, chr_name, start, fc.copy(), window)
