"""
Enformer-based perturbation utilities for C.Shark.

Provides functions to:
  1. Load an Enformer model (pre-trained or fine-tuned checkpoint)
  2. Predict 1D tracks from DNA sequence using a sliding-window approach
  3. Compute the delta between WT and ALT Enformer predictions
  4. Apply that delta to experimental 1D tracks (the "enformer_seq" KO mode)

This allows perturb.py to support an ``enformer_seq`` knockout mode where
sequence variants are evaluated through Enformer and the predicted
fold-change is transferred onto actual experimental bigwig data.
"""

import os
import numpy as np
import torch
import torch.nn.functional as F

# Enformer constants (matching hierarchical_predict_with_enformer.py)
ENFORMER_CONTEXT_LENGTH = 196_608
ENFORMER_TARGET_LEN = 896 * 128  # 114,688 bp
ENFORMER_TRIM = (ENFORMER_CONTEXT_LENGTH - ENFORMER_TARGET_LEN) // 2  # 40,960 bp
ENFORMER_DOWNSAMPLE_FACTOR = 128
ENFORMER_OUTPUT_BINS = 896

# Sequence encoding used in perturb.py (ATCGN)
EN_DICT = {'a': 0, 't': 1, 'c': 2, 'g': 3, 'n': 4}

# Default mapping from short track names to Enformer target descriptions.
# Matches the target_map used in TrainModule (hierarchical_predict_with_enformer.py).
# Species-specific mappings are keyed as ``(species, track_name)``.
DEFAULT_TARGET_MAP = {
    'ctcf':     {'human': 'CTCF:H1-hESC',            'mouse': 'CTCF'},
    'atac':     {'human': 'DNASE:H1-hESC',            'mouse': 'DNASE'},
    'rad21':    {'human': 'CHIP:RAD21:H1-hESC',       'mouse': 'RAD21'},
    'h3k27ac':  {'human': 'CHIP:H3K27ac:H1-hESC',     'mouse': 'H3K27ac'},
    'h3k4me3':  {'human': 'CHIP:H3K4me3:H1-hESC',     'mouse': 'H3K4me3'},
    'h3k9me3':  {'human': 'CHIP:H3K9me3:H1-hESC',     'mouse': 'H3K9me3'},
    'h3k36me3': {'human': 'CHIP:H3K36me3:H1-hESC',    'mouse': 'H3K36me3'},
    'h3k27me3': {'human': 'CHIP:H3K27me3:H1-hESC',    'mouse': 'H3K27me3'},
}


# ---------------------------------------------------------------------------
# Enformer target index resolution
# ---------------------------------------------------------------------------

def get_target_indices(species: str, target: str) -> np.ndarray:
    """Fetch numerical indices for a target description from the Basenji
    targets file.

    Mirrors ``TrainModule.get_target_indices`` but is usable standalone.

    Parameters
    ----------
    species : str
        ``'human'`` or ``'mouse'``.
    target : str
        Substring to match in the ``description`` column (case-insensitive).

    Returns
    -------
    indices : np.ndarray of int
    """
    import pandas as pd
    targets_file = (
        f"https://raw.githubusercontent.com/calico/basenji/master/"
        f"manuscripts/cross2020/targets_{species}.txt"
    )
    targets_df = pd.read_csv(targets_file, sep='\t')
    mask = targets_df['description'].str.contains(target, case=False)
    indices = targets_df[mask]['index'].values
    if len(indices) == 0:
        raise ValueError(
            f"No Enformer tracks found for target '{target}' in species "
            f"'{species}'.  Check DEFAULT_TARGET_MAP or pass custom "
            f"target descriptions."
        )
    return indices


# ---------------------------------------------------------------------------
# Adapter wrapper  (standalone version of HESCHeadAdapterWrapper)
# ---------------------------------------------------------------------------

class EnformerHeadAdapterWrapper(torch.nn.Module):
    """Lightweight adapter that sits on top of the Enformer trunk and projects
    its embeddings to a small set of user-specified tracks.

    This is a standalone version of the ``HESCHeadAdapterWrapper`` defined
    inside ``TrainModule.get_hESC_wrapper``.  It can optionally copy the
    pre-trained Enformer head weights for the selected track indices so that
    predictions are immediately meaningful without fine-tuning.

    Parameters
    ----------
    enformer : torch.nn.Module
        A loaded ``enformer_pytorch`` model instance.
    track_indices : list[int]
        Numerical indices into the Enformer output head for the tracks of
        interest (one per desired output track).
    species : str
        ``'human'`` or ``'mouse'`` — selects which Enformer head to
        initialise from.
    load_pretrained : bool
        If *True*, copy weights/biases from the original Enformer head for
        the selected indices.  Recommended for zero-shot usage.
    """

    def __init__(self, enformer, track_indices, species='human',
                 load_pretrained=True):
        super().__init__()
        self.enformer = enformer
        self.track_indices = track_indices
        self.species = species

        # Enformer trunk output dim is dim * 2 (e.g. 1536 * 2 = 3072)
        embedding_dim = enformer.dim * 2

        # Per-track linear projections
        self.to_tracks = torch.nn.ModuleList([
            torch.nn.Linear(in_features=embedding_dim, out_features=1)
            for _ in track_indices
        ])

        # Learnable per-track scale and bias
        self.scale = torch.nn.Parameter(torch.ones(len(track_indices)))
        self.bias = torch.nn.Parameter(torch.zeros(len(track_indices)))

        if load_pretrained:
            # _heads['human'] is Sequential(Linear, Softplus)
            original_linear = enformer._heads[self.species][0]
            with torch.no_grad():
                for i, original_idx in enumerate(track_indices):
                    self.to_tracks[i].weight.data = (
                        original_linear.weight.data[original_idx]
                        .unsqueeze(0).clone()
                    )
                    self.to_tracks[i].bias.data = (
                        original_linear.bias.data[original_idx]
                        .unsqueeze(0).clone()
                    )

        self.activation = torch.nn.Softplus()

    def forward(self, x):
        """Forward pass.

        Parameters
        ----------
        x : torch.Tensor, shape (batch, seq_len, 4)
            One-hot encoded sequence in ACGT order.

        Returns
        -------
        preds : torch.Tensor, shape (batch, 896, num_tracks)
        """
        embeddings = self.enformer(x, return_only_embeddings=True)

        track_preds = []
        for track_i, linear_layer in enumerate(self.to_tracks):
            track_output = linear_layer(embeddings)
            track_output = track_output * self.scale[track_i] + self.bias[track_i]
            track_preds.append(track_output)

        # (batch, 896, num_selected_tracks)
        track_preds = torch.cat(track_preds, dim=-1)
        return self.activation(track_preds)


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def load_enformer_pretrained(target_tracks=None, species='human',
                             load_pretrained_heads=True, device=None):
    """Load the pre-trained Enformer and wrap it for a specific set of tracks.

    The returned model directly outputs ``(batch, 896, len(target_tracks))``
    — no need to slice into the full 5 313-track head.

    Parameters
    ----------
    target_tracks : list[str] or None
        Short track names (keys of ``DEFAULT_TARGET_MAP``), e.g.
        ``['ctcf', 'atac', 'rad21']``.  If *None*, defaults to
        ``['ctcf', 'atac']``.
    species : str
        ``'human'`` or ``'mouse'``.
    load_pretrained_heads : bool
        If *True* (default), copy the original Enformer head weights for
        the selected tracks so predictions are usable without fine-tuning.
    device : torch.device or None

    Returns
    -------
    model : EnformerHeadAdapterWrapper
        Wrapped model in eval mode.
    track_names : list[str]
        Ordered track names matching the model's output columns.
    device : torch.device
    """
    from enformer_pytorch import from_pretrained

    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if target_tracks is None:
        target_tracks = ['ctcf', 'atac']

    # Resolve short names → Enformer target indices
    track_indices = []
    resolved_names = []
    for track_name in target_tracks:
        if track_name in DEFAULT_TARGET_MAP:
            desc = DEFAULT_TARGET_MAP[track_name].get(species)
            if desc is None:
                raise ValueError(
                    f"No target description for track '{track_name}' in "
                    f"species '{species}'.  Add it to DEFAULT_TARGET_MAP or "
                    f"use load_enformer_from_checkpoint instead."
                )
        else:
            # Treat the name itself as a target description for flexibility
            desc = track_name
        indices = get_target_indices(species, desc)
        track_indices.append(int(indices[0]))
        resolved_names.append(track_name)

    print(f"[enformer_utils] Loading pre-trained Enformer for species='{species}'")
    print(f"[enformer_utils] Target tracks: {resolved_names}")
    print(f"[enformer_utils] Enformer head indices: {track_indices}")

    enformer = from_pretrained('EleutherAI/enformer-official-rough',
                               use_tf_gamma=True)
    # Freeze all parameters — we are not fine-tuning, just predicting
    for param in enformer.parameters():
        param.requires_grad = False

    wrapper = EnformerHeadAdapterWrapper(
        enformer,
        track_indices,
        species=species,
        load_pretrained=load_pretrained_heads,
    )
    wrapper.eval()
    wrapper.to(device)
    return wrapper, resolved_names, device


def load_enformer_from_checkpoint(checkpoint_path, device=None):
    """Load a fine-tuned Enformer wrapper from a hierarchical training checkpoint.

    The checkpoint is expected to contain a ``TrainModule`` with an ``enformer``
    attribute (the ``HESCHeadAdapterWrapper``).

    Returns
    -------
    enformer_wrapper : torch.nn.Module
        The fine-tuned adapter wrapper in eval mode.
    track_names : list[str]
        Names of the output tracks (e.g. ``['ctcf', 'atac', 'h3k27ac', ...]``).
    device : torch.device
    """
    import argparse
    # Lazy import to avoid circular deps at module level
    from cshark.inference.hierarchical_predict_with_enformer import TrainModule

    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    checkpoint = torch.load(checkpoint_path, map_location=device)
    hparams = argparse.Namespace(**checkpoint['hyper_parameters'])
    module = TrainModule.load_from_checkpoint(checkpoint_path, args=hparams,
                                              map_location=device)
    module.eval()
    module.to(device)

    enformer_wrapper = module.enformer
    enformer_wrapper.eval()

    # Track names are the input features minus 'seq' (Enformer predicts the
    # non-sequence genomic signals)
    track_names = [f for f in hparams.input_features if f != 'seq']

    return enformer_wrapper, track_names, device


# ---------------------------------------------------------------------------
# Sliding-window Enformer prediction  (standalone, no TrainModule needed)
# ---------------------------------------------------------------------------

def enformer_predict_1d(model, seq_onehot, num_tracks, device=None,
                        step_fraction=0.5):
    """Predict 1D tracks from a one-hot sequence using Enformer with overlapping
    sliding windows.

    This is a standalone version of ``TrainModule.enformer_predict_1d`` that
    works directly with numpy arrays and a wrapped Enformer model.

    Parameters
    ----------
    model : torch.nn.Module
        An ``EnformerHeadAdapterWrapper`` (or compatible) that accepts a
        tensor of shape ``(batch, seq_len, 4)`` in **ACGT** order and returns
        a tensor of shape ``(batch, 896, num_tracks)``.
    seq_onehot : np.ndarray, shape (seq_len, 5) or (seq_len, 4)
        One-hot encoded sequence.  If 5 columns the last (N) is ignored.
        Expected column order: **A T C G [N]** (the perturb.py convention).
    num_tracks : int
        Number of output tracks the model produces.
    device : torch.device or None
    step_fraction : float
        Fraction of ``ENFORMER_CONTEXT_LENGTH`` to advance per window (controls
        overlap). Default 0.5 → 50% overlap.

    Returns
    -------
    preds : np.ndarray, shape (seq_len, num_tracks)
        Predicted 1D signal at base-pair resolution (averaged across
        overlapping windows).
    """
    if device is None:
        device = next(model.parameters()).device

    # Use only ATCG columns (drop N if present)
    seq_4 = seq_onehot[:, :4].astype(np.float32)
    seq_len = seq_4.shape[0]

    preds = np.zeros((seq_len, num_tracks), dtype=np.float64)
    counts = np.zeros((seq_len, num_tracks), dtype=np.float64)

    step_size = int(ENFORMER_CONTEXT_LENGTH * step_fraction)

    with torch.no_grad():
        for start in range(-ENFORMER_TRIM, seq_len, step_size):
            end = start + ENFORMER_CONTEXT_LENGTH

            if start + ENFORMER_TRIM >= seq_len:
                break

            # Extract window, pad if needed
            input_start = max(start, 0)
            input_end = min(end, seq_len)
            window_seq = seq_4[input_start:input_end]

            # Pad left
            if start < 0:
                left_pad = -start
                window_seq = np.concatenate(
                    [np.full((left_pad, 4), 0.25, dtype=np.float32), window_seq],
                    axis=0,
                )

            # Pad right
            if window_seq.shape[0] < ENFORMER_CONTEXT_LENGTH:
                right_pad = ENFORMER_CONTEXT_LENGTH - window_seq.shape[0]
                window_seq = np.concatenate(
                    [window_seq, np.full((right_pad, 4), 0.25, dtype=np.float32)],
                    axis=0,
                )

            # Reorder ATCG → ACGT (Enformer convention)
            window_seq = window_seq[:, [0, 2, 3, 1]]

            input_tensor = (torch.tensor(window_seq, dtype=torch.float32)
                            .unsqueeze(0).to(device))

            # Model forward — shape (1, 896, num_tracks)
            outputs = model(input_tensor)
            outputs_np = outputs.cpu().numpy()[0]  # (896, num_tracks)

            global_out_start = start + ENFORMER_TRIM

            for bin_idx in range(ENFORMER_OUTPUT_BINS):
                bin_start = global_out_start + bin_idx * ENFORMER_DOWNSAMPLE_FACTOR
                bin_end = bin_start + ENFORMER_DOWNSAMPLE_FACTOR

                if bin_start >= seq_len:
                    break
                bin_end = min(bin_end, seq_len)

                preds[bin_start:bin_end, :] += outputs_np[bin_idx]
                counts[bin_start:bin_end, :] += 1.0

    # Average overlapping predictions
    preds = preds / np.clip(counts, 1.0, None)
    return preds


# ---------------------------------------------------------------------------
# Sequence manipulation helpers
# ---------------------------------------------------------------------------

def apply_variant_to_seq(seq_onehot, position, alt_base):
    """Apply a single base substitution to a one-hot encoded sequence.

    Parameters
    ----------
    seq_onehot : np.ndarray, shape (seq_len, 5)
        Original sequence (A T C G N).
    position : int
        0-based position within the sequence to mutate.
    alt_base : str
        Alternate base (single character: a/t/c/g).

    Returns
    -------
    alt_seq : np.ndarray
        Copy of *seq_onehot* with the variant applied.
    """
    alt_seq = seq_onehot.copy()
    new_entry = np.zeros(alt_seq.shape[1])
    alt_idx = EN_DICT[alt_base.lower()]
    new_entry[alt_idx] = 1
    alt_seq[position, :] = new_entry
    return alt_seq


def apply_alt_sequence(seq_onehot, start, end, alt_bases):
    """Replace a stretch of bases in the one-hot sequence.

    Parameters
    ----------
    seq_onehot : np.ndarray, shape (seq_len, 5)
    start, end : int
        0-based half-open interval ``[start, end)`` to replace.
    alt_bases : str
        Replacement bases (must have length ``end - start``).

    Returns
    -------
    alt_seq : np.ndarray
        Copy with the replacement applied.
    """
    alt_seq = seq_onehot.copy()
    if len(alt_bases) != (end - start):
        raise ValueError(
            f"Alt sequence length {len(alt_bases)} does not match region "
            f"length {end - start}."
        )
    for i, base in enumerate(alt_bases.lower()):
        new_entry = np.zeros(alt_seq.shape[1])
        new_entry[EN_DICT[base]] = 1
        alt_seq[start + i, :] = new_entry
    return alt_seq


# ---------------------------------------------------------------------------
# Delta computation & application  (core of the "enformer_seq" KO mode)
# ---------------------------------------------------------------------------

def compute_enformer_delta(model, wt_seq, alt_seq, num_tracks, device=None,
                           step_fraction=0.5, epsilon=1e-6):
    """Compute the per-base-pair fold-change (ALT / WT) predicted by Enformer.

    Parameters
    ----------
    model : torch.nn.Module
        Enformer (or adapter wrapper).
    wt_seq : np.ndarray, shape (seq_len, 4 or 5)
        Wild-type one-hot sequence.
    alt_seq : np.ndarray, shape (seq_len, 4 or 5)
        Alternate one-hot sequence (with variant(s) applied).
    num_tracks : int
        Number of output tracks.
    device : torch.device or None
    step_fraction : float
    epsilon : float
        Small constant to avoid division by zero.

    Returns
    -------
    fold_change : np.ndarray, shape (seq_len, num_tracks)
        Per-bp ratio ALT / WT.  Values > 1 indicate increase, < 1 decrease.
    delta : np.ndarray, shape (seq_len, num_tracks)
        Per-bp additive difference ALT − WT.
    wt_pred : np.ndarray
    alt_pred : np.ndarray
    """
    wt_pred = enformer_predict_1d(model, wt_seq, num_tracks, device=device,
                                  step_fraction=step_fraction)
    alt_pred = enformer_predict_1d(model, alt_seq, num_tracks, device=device,
                                   step_fraction=step_fraction)

    delta = alt_pred - wt_pred
    fold_change = alt_pred / np.clip(wt_pred, epsilon, None)

    return fold_change, delta, wt_pred, alt_pred


def apply_enformer_delta_to_track(track, fold_change_1d, mode='multiplicative',
                                  cap=10.0):
    """Apply the Enformer-predicted delta to an experimental 1D track.

    Parameters
    ----------
    track : np.ndarray, shape (L,)
        Experimental signal (e.g. CTCF ChIP-seq values for the region).
    fold_change_1d : np.ndarray, shape (L,)
        Per-bp fold-change (ALT / WT) from Enformer for the matching track.
        Must already be resampled / aligned to the track's resolution.
    mode : str
        ``'multiplicative'`` — multiply track by fold_change.
        ``'additive'``       — add the raw delta to track.
    cap : float
        Upper bound on fold-change to prevent extreme outliers.

    Returns
    -------
    perturbed : np.ndarray, shape (L,)
    """
    track = track.copy()
    if mode == 'multiplicative':
        fc = np.clip(fold_change_1d, 1.0 / cap, cap)
        track = track * fc
    elif mode == 'additive':
        track = track + fold_change_1d
    else:
        raise ValueError(f"Unknown delta mode: '{mode}'. Use 'multiplicative' or 'additive'.")
    # Ensure non-negative signal
    track = np.clip(track, 0, None)
    return track


def downsample_to_track_resolution(signal, target_length):
    """Resample a base-pair resolution signal to a coarser track resolution
    using simple averaging.

    Parameters
    ----------
    signal : np.ndarray, shape (bp_length,)
    target_length : int

    Returns
    -------
    resampled : np.ndarray, shape (target_length,)
    """
    bp_length = len(signal)
    bin_size = bp_length / target_length
    resampled = np.zeros(target_length)
    for i in range(target_length):
        s = int(round(i * bin_size))
        e = int(round((i + 1) * bin_size))
        e = min(e, bp_length)
        if s < e:
            resampled[i] = np.mean(signal[s:e])
    return resampled


# ---------------------------------------------------------------------------
# High-level "enformer_seq" knockout function for perturb.py
# ---------------------------------------------------------------------------

def enformer_seq_knockout(seq_region, ctcf_region, atac_region, other_regions,
                          input_track_names,
                          enformer_model, enformer_track_names,
                          variant_positions=None, alt_bases=None,
                          ko_start=None, ko_end=None, alt_sequence=None,
                          window=2097152,
                          delta_mode='multiplicative', cap=10.0,
                          device=None):
    """Perform the **enformer_seq** knockout.

    1. Build an ALT sequence from variant(s) or a replacement stretch.
    2. Run Enformer on WT and ALT to get per-track fold-change.
    3. Apply that fold-change to the matching experimental 1D tracks.

    Parameters
    ----------
    seq_region : np.ndarray, shape (seq_len, 5)
        One-hot WT sequence (ATCGN).
    ctcf_region : np.ndarray or None
    atac_region : np.ndarray or None
    other_regions : list[np.ndarray] or None
    input_track_names : list[str]
        Names of input tracks in order, e.g. ``['ctcf', 'atac', 'rad21', ...]``.
    enformer_model : torch.nn.Module
    enformer_track_names : list[str]
        Names of tracks the Enformer model outputs, e.g. ``['ctcf', 'atac', 'h3k27ac', ...]``.
    variant_positions : list[int] or None
        0-based positions within *seq_region* for single-base variants.
    alt_bases : list[str] or None
        Alternate bases corresponding to *variant_positions*.
    ko_start, ko_end : int or None
        Region to replace with *alt_sequence* (0-based within seq_region).
    alt_sequence : str or None
        Replacement bases for ``[ko_start, ko_end)``.
    window : int
    delta_mode : str
        ``'multiplicative'`` or ``'additive'``.
    cap : float
    device : torch.device or None

    Returns
    -------
    ctcf_region, atac_region, other_regions : perturbed tracks
    enformer_results : dict
        Contains ``'fold_change'``, ``'delta'``, ``'wt_pred'``, ``'alt_pred'``
        for diagnostics / plotting.
    """
    num_tracks = len(enformer_track_names)

    # --- 1. Build ALT sequence ---
    alt_seq = seq_region.copy()
    if variant_positions is not None and alt_bases is not None:
        for pos, alt in zip(variant_positions, alt_bases):
            alt_seq = apply_variant_to_seq(alt_seq, pos, alt)
            ref_idx = seq_region[pos, :4].argmax()
            ref_base = 'ATCG'[ref_idx]
            print(f"[enformer_seq] Variant at bp {pos}: {ref_base} -> {alt.upper()}")
    if ko_start is not None and ko_end is not None and alt_sequence is not None:
        alt_seq = apply_alt_sequence(alt_seq, ko_start, ko_end, alt_sequence)
        print(f"[enformer_seq] Replaced region [{ko_start}, {ko_end}) with alt sequence "
              f"(length {len(alt_sequence)})")

    # --- 2. Compute Enformer delta ---
    fold_change, delta, wt_pred, alt_pred = compute_enformer_delta(
        enformer_model, seq_region, alt_seq, num_tracks,
        device=device, epsilon=1e-6,
    )

    # --- 3. Apply delta to each matching experimental track ---
    def _apply_to_track(track, enformer_track_idx):
        """Resample enformer delta to track resolution and apply."""
        if track is None:
            return track
        track_len = len(track)
        if delta_mode == 'multiplicative':
            fc_1d = fold_change[:, enformer_track_idx]
            fc_resampled = downsample_to_track_resolution(fc_1d, track_len)
            return apply_enformer_delta_to_track(track, fc_resampled,
                                                  mode='multiplicative', cap=cap)
        else:
            d_1d = delta[:, enformer_track_idx]
            d_resampled = downsample_to_track_resolution(d_1d, track_len)
            return apply_enformer_delta_to_track(track, d_resampled,
                                                  mode='additive', cap=cap)

    for enf_idx, enf_name in enumerate(enformer_track_names):
        if enf_name == 'ctcf' and ctcf_region is not None:
            ctcf_region = _apply_to_track(ctcf_region, enf_idx)
            print(f"[enformer_seq] Applied delta to CTCF track")
        elif enf_name == 'atac' and atac_region is not None:
            atac_region = _apply_to_track(atac_region, enf_idx)
            print(f"[enformer_seq] Applied delta to ATAC track")
        elif other_regions is not None and enf_name in input_track_names:
            # Find position in other_regions (offset by ctcf/atac)
            other_offset = 0
            if 'ctcf' in input_track_names:
                other_offset += 1
            if 'atac' in input_track_names:
                other_offset += 1
            other_track_names = input_track_names[other_offset:]
            if enf_name in other_track_names:
                other_idx = other_track_names.index(enf_name)
                other_regions[other_idx] = _apply_to_track(
                    other_regions[other_idx], enf_idx
                )
                print(f"[enformer_seq] Applied delta to {enf_name} track")

    enformer_results = {
        'fold_change': fold_change,
        'delta': delta,
        'wt_pred': wt_pred,
        'alt_pred': alt_pred,
        'enformer_track_names': enformer_track_names,
    }

    return ctcf_region, atac_region, other_regions, enformer_results


# ---------------------------------------------------------------------------
# BigWig writing for enformer-modified tracks
# ---------------------------------------------------------------------------

def write_tmp_enformer_ko_bigwig(bigwig_path, fold_change, delta,
                                  enformer_track_idx, track_name,
                                  chr_name, start, window=2_097_152,
                                  delta_mode='multiplicative', cap=10.0):
    """Read an experimental bigwig and write an enformer-perturbed version.

    The original signal is read at bp resolution, the Enformer-predicted
    fold-change (or additive delta) is applied, and the result is written
    to ``tmp/{track_name}_enformer_ko.bw``.

    Parameters
    ----------
    bigwig_path : str
        Path to the original experimental bigwig file.
    fold_change : np.ndarray, shape (bp_length, num_tracks)
        Per-bp fold-change (ALT / WT) from Enformer.
    delta : np.ndarray, shape (bp_length, num_tracks)
        Per-bp additive delta from Enformer.
    enformer_track_idx : int
        Column index into *fold_change* / *delta* for this track.
    track_name : str
        Short name (e.g. ``'ctcf'``), used for the output filename.
    chr_name : str
    start : int
    window : int
    delta_mode : str
        ``'multiplicative'`` or ``'additive'``.
    cap : float
    """
    import pyBigWig

    bw = pyBigWig.open(bigwig_path)
    original = np.array(bw.values(chr_name, start, start + window))
    original = np.nan_to_num(original, nan=0.0)
    header_list = list(bw.chroms().items())
    bw.close()

    if delta_mode == 'multiplicative':
        fc = fold_change[:, enformer_track_idx]
        # Resample if lengths differ
        if len(fc) != len(original):
            fc = downsample_to_track_resolution(fc, len(original))
        fc = np.clip(fc, 1.0 / cap, cap)
        modified = original * fc
    else:
        d = delta[:, enformer_track_idx]
        if len(d) != len(original):
            d = downsample_to_track_resolution(d, len(original))
        modified = original + d
    modified = np.clip(modified, 0, None)

    out_path = f'tmp/{track_name}_enformer_ko.bw'
    out_bw = pyBigWig.open(out_path, 'w')
    out_bw.addHeader(header_list)

    positions = list(range(start, start + window))
    values = list(modified)

    # Merge intervals for efficient writing
    merged_intervals = []
    prev_pos = positions[0]
    prev_val = values[0]
    for i in range(1, len(positions)):
        if values[i] != prev_val:
            merged_intervals.append((prev_pos, positions[i], prev_val))
            prev_pos = positions[i]
            prev_val = values[i]
    merged_intervals.append((prev_pos, positions[-1] + 1, prev_val))

    for s, e, v in merged_intervals:
        out_bw.addEntries([chr_name], [s], [e], [float(v)])

    out_bw.close()
    print(f"[enformer_seq] Wrote enformer KO bigwig: {out_path}")


def write_tmp_enformer_delta_bigwig(bigwig_path, fold_change, delta,
                                     enformer_track_idx, track_name,
                                     chr_name, start, window=2_097_152,
                                     delta_mode='multiplicative'):
    """Write the raw Enformer delta (or log2 fold-change) as a bigwig.

    This produces a track centred around zero that is suitable for a
    diverging colour map (e.g. red/blue) in a diff plot.

    * **multiplicative** mode: writes ``log2(fold_change)`` so that gains
      are positive and losses negative.
    * **additive** mode: writes the raw additive delta.

    Output file: ``tmp/{track_name}_enformer_delta.bw``

    Parameters
    ----------
    bigwig_path : str
        Path to an original experimental bigwig (used only for the
        chromosome header).
    fold_change : np.ndarray, shape (bp_length, num_tracks)
    delta : np.ndarray, shape (bp_length, num_tracks)
    enformer_track_idx : int
    track_name : str
    chr_name : str
    start : int
    window : int
    delta_mode : str
    """
    import pyBigWig

    bw = pyBigWig.open(bigwig_path)
    header_list = list(bw.chroms().items())
    bw.close()

    if delta_mode == 'multiplicative':
        fc = fold_change[:, enformer_track_idx].copy()
        fc = np.clip(fc, 1e-6, None)  # avoid log(0)
        signal = np.log2(fc)
    else:
        signal = delta[:, enformer_track_idx].copy()

    # Resample to window length if needed
    if len(signal) != window:
        signal = downsample_to_track_resolution(signal, window)

    out_path = f'tmp/{track_name}_enformer_delta.bw'
    out_bw = pyBigWig.open(out_path, 'w')
    out_bw.addHeader(header_list)

    positions = list(range(start, start + window))
    values = list(signal)

    merged_intervals = []
    prev_pos = positions[0]
    prev_val = values[0]
    for i in range(1, len(positions)):
        if values[i] != prev_val:
            merged_intervals.append((prev_pos, positions[i], prev_val))
            prev_pos = positions[i]
            prev_val = values[i]
    merged_intervals.append((prev_pos, positions[-1] + 1, prev_val))

    for s, e, v in merged_intervals:
        out_bw.addEntries([chr_name], [s], [e], [float(v)])

    out_bw.close()
    print(f"[enformer_seq] Wrote enformer delta bigwig: {out_path}")
