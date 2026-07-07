"""
AlphaGenome-based sequence-perturbation utilities for C.Shark.

This is the AlphaGenome analogue of ``enformer_utils.py``.  It powers the
``alphagenome_seq`` knockout mode, which operates exactly like ``enformer_seq``
but swaps the Enformer backbone for the AlphaGenome PyTorch port
(``alphagenome-pytorch``):

  1. Load AlphaGenome, pruning every head we do not need (contact maps + splice
     heads, and any genome-track head not requested) to save GPU memory.  This
     is the mirror image of ``alphagenome_predict.py`` (which keeps ONLY the
     contact-map head); here we keep only the requested 1D genome-track heads.
  2. Predict 1D tracks from DNA sequence with a sliding 1 Mb window.
  3. Compute the per-bp fold-change (ALT / WT) predicted by AlphaGenome.
  4. Apply that fold-change to the experimental 1D tracks.

The delta math, the delta-to-track application, the base-substitution helpers,
and the KO/delta bigwig writers are all model-agnostic, so they are imported
verbatim from ``enformer_utils`` rather than duplicated.  The only genuinely
AlphaGenome-specific pieces live here: model loading (+head pruning), track-name
-> output-head/channel resolution via the AlphaGenome metadata catalog, and the
sliding-window predictor.
"""

import numpy as np
import torch

# Reuse the model-agnostic machinery from the Enformer utilities.
from cshark.inference.utils.enformer_utils import (
    apply_enformer_delta_to_track,
    downsample_to_track_resolution,
)

# ── AlphaGenome constants ──────────────────────────────────────────────
AG_CONTEXT_LENGTH = 1_048_576      # 1 Mb DNA-only input window
AG_BIN_SIZE = 128                  # 128-bp genome-track resolution we consume
AG_OUTPUT_BINS = AG_CONTEXT_LENGTH // AG_BIN_SIZE   # 8192 bins / window

# Sequence encoding used throughout perturb.py (ATCGN).  AlphaGenome expects
# one-hot in ACGT order (A=0, C=1, G=2, T=3; N -> all-zeros), so we reorder the
# ATCG columns via this index list (identical to the Enformer path).
ATCG_TO_ACGT = [0, 2, 3, 1]

# Map short C.Shark track names to an AlphaGenome (output-head, metadata-filter)
# pair.  The filter is passed to the metadata catalog to pick the channel(s) of
# interest.  Assay heads (atac/dnase) need no filter; TF / histone ChIP heads
# are narrowed by transcription factor / histone mark.
AG_TARGET_MAP = {
    'ctcf':     ('chip_tf',      {'transcription_factor': 'CTCF'}),
    'rad21':    ('chip_tf',      {'transcription_factor': 'RAD21'}),
    'atac':     ('atac',         {}),
    'dnase':    ('dnase',        {}),
    'h3k27ac':  ('chip_histone', {'histone_mark': 'H3K27ac'}),
    'h3k4me3':  ('chip_histone', {'histone_mark': 'H3K4me3'}),
    'h3k9me3':  ('chip_histone', {'histone_mark': 'H3K9me3'}),
    'h3k36me3': ('chip_histone', {'histone_mark': 'H3K36me3'}),
    'h3k27me3': ('chip_histone', {'histone_mark': 'H3K27me3'}),
}


# ---------------------------------------------------------------------------
# Metadata catalog + track-index resolution
# ---------------------------------------------------------------------------

def _match_track(track, criteria):
    """Replicate ``NamedTrackTensor._match_track`` against a catalog track.

    ``field=None`` -> field missing/None; list -> membership; scalar -> equality.
    """
    for key, expected in criteria.items():
        actual = track.get(key)
        if expected is None:
            if actual is not None:
                return False
        elif isinstance(expected, (list, tuple, set, frozenset)):
            if actual not in expected:
                return False
        elif actual != expected:
            return False
    return True


def _load_catalog(metadata_path, species):
    """Load an AlphaGenome track-metadata catalog.

    Prefers an explicit ``metadata_path`` (parquet/csv/tsv); otherwise falls
    back to the package's built-in catalog.  Raises a helpful error if neither
    is available, since name-based track selection is impossible without it.
    """
    from alphagenome_pytorch.named_outputs import TrackMetadataCatalog

    if metadata_path is not None:
        return TrackMetadataCatalog.from_file(metadata_path)
    try:
        return TrackMetadataCatalog.load_builtin(species)
    except Exception as exc:  # builtin parquet not bundled in all releases
        raise RuntimeError(
            "AlphaGenome track metadata is required to resolve track names "
            "(e.g. 'ctcf', 'h3k27ac') to output channels, but no built-in "
            "catalog was found in this alphagenome-pytorch install. Pass a "
            "metadata file via --alphagenome-metadata (parquet/csv/tsv with "
            "columns like track_name/transcription_factor/histone_mark)."
        ) from exc


def _resolve_track_channels(catalog, target_tracks, org_idx, celltype=None):
    """Resolve each short track name to (output_head_name, channel indices).

    Returns
    -------
    resolvers : dict[str, tuple[str, np.ndarray]]
        Maps track name -> (alphagenome head name, int channel indices) into the
        *real* (padding-stripped) channels of that head's prediction tensor.
    resolved_names : list[str]
        The subset of ``target_tracks`` that were successfully resolved, in order.
    """
    # Optional biosample narrowing: if a cell-line token from ``celltype`` is
    # present in the catalog's biosample_name values we prefer those tracks,
    # otherwise we average across every track matching the assay/mark filter.
    cell_token = None
    if celltype:
        for token in ('GM12878', 'H1', 'K562', 'HepG2', 'IMR90', 'HeLa'):
            if token.lower() in celltype.lower():
                cell_token = token
                break

    resolvers = {}
    resolved_names = []
    for name in target_tracks:
        key = name.lower()
        if key not in AG_TARGET_MAP:
            print(f"[alphagenome_seq] WARNING: no AlphaGenome mapping for track "
                  f"'{name}'; skipping.")
            continue
        head_name, criteria = AG_TARGET_MAP[key]
        tracks = catalog.get_tracks(head_name, organism=org_idx)
        matches = [t for t in tracks if _match_track(t, criteria)]
        if cell_token is not None:
            narrowed = [t for t in matches
                        if cell_token.lower() in str(t.get('biosample_name') or '').lower()]
            if narrowed:
                matches = narrowed
        if not matches:
            print(f"[alphagenome_seq] WARNING: no AlphaGenome channels matched "
                  f"track '{name}' (head '{head_name}', filter {criteria}); skipping.")
            continue
        idxs = np.array([int(t.track_index) for t in matches], dtype=int)
        resolvers[name] = (head_name, idxs)
        resolved_names.append(name)
        print(f"[alphagenome_seq] '{name}' -> head '{head_name}', "
              f"{len(idxs)} channel(s){' @ ' + cell_token if cell_token else ''}")
    return resolvers, resolved_names


# ---------------------------------------------------------------------------
# Model loading  (+ aggressive head pruning to save GPU memory)
# ---------------------------------------------------------------------------

def load_alphagenome(model_path, target_tracks=None, species='human',
                     celltype=None, device=None, metadata_path=None,
                     full_precision=False):
    """Load AlphaGenome and prepare it for 1D genome-track prediction.

    Parameters
    ----------
    model_path : str
        Path to the AlphaGenome ``.safetensors`` / ``.pth`` checkpoint.
    target_tracks : list[str] or None
        Short track names to predict, e.g. ``['ctcf', 'atac', 'h3k27ac']``.
        Defaults to ``['ctcf', 'atac']``.
    species : str
        ``'human'`` or ``'mouse'`` (selects organism index 0 / 1).
    celltype : str or None
        Used for best-effort biosample narrowing of matching tracks.
    device : torch.device or None
    metadata_path : str or None
        Track-metadata catalog (parquet/csv/tsv) for name resolution.
    full_precision : bool
        Disable bf16 mixed precision (use fp32 only).

    Returns
    -------
    model : AlphaGenome (eval, pruned, on ``device``)
    track_names : list[str]  -- resolved output track names, in order
    device : torch.device
    org_idx : int            -- 0 human / 1 mouse
    resolvers : dict[str, (head_name, np.ndarray)]
    """
    import torch.nn as nn
    from alphagenome_pytorch import AlphaGenome
    from alphagenome_pytorch.config import DtypePolicy

    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if target_tracks is None:
        target_tracks = ['ctcf', 'atac']
    org_idx = 1 if species == 'mouse' else 0

    # --- resolve which output heads / channels we need BEFORE loading -------
    catalog = _load_catalog(metadata_path, species)
    resolvers, resolved_names = _resolve_track_channels(
        catalog, target_tracks, org_idx, celltype=celltype)
    if not resolvers:
        raise RuntimeError(
            "AlphaGenome could not resolve any of the requested tracks "
            f"{target_tracks}. Check --enformer-tracks and the metadata catalog."
        )
    needed_heads = {head for head, _ in resolvers.values()}

    print(f"[alphagenome_seq] Loading AlphaGenome from {model_path} ...")
    if full_precision:
        dtype_policy = DtypePolicy.full_float32()
    else:
        dtype_policy = DtypePolicy.mixed_precision()   # params fp32, compute/out bf16
    model = AlphaGenome.from_pretrained(model_path, dtype_policy=dtype_policy)
    model.set_track_metadata_catalog(catalog)

    # --- GPU memory saving: prune every head we do not need ----------------
    # Drop the contact-map + splice heads outright (mirror of, but opposite to,
    # alphagenome_predict.py), and keep only the requested genome-track heads.
    model.contact_maps_head = None
    model.splice_sites_classification_head = None
    model.splice_sites_usage_head = None
    model.splice_sites_junction_head = None
    kept, dropped = [], []
    pruned_heads = nn.ModuleDict()
    for head_name in list(model.heads.keys()):
        if head_name in needed_heads:
            pruned_heads[head_name] = model.heads[head_name]
            kept.append(head_name)
        else:
            dropped.append(head_name)
    model.heads = pruned_heads
    print(f"[alphagenome_seq] Kept genome-track heads {kept}; "
          f"pruned {dropped} + contact-map/splice heads")

    model = model.to(device).eval()
    for p in model.parameters():
        p.requires_grad = False
    return model, resolved_names, device, org_idx, resolvers


# ---------------------------------------------------------------------------
# Sliding-window AlphaGenome 1D prediction
# ---------------------------------------------------------------------------

def alphagenome_predict_1d(model, seq_onehot, resolvers, ordered_names,
                           org_idx, device=None, step_fraction=0.5):
    """Predict 1D tracks from a one-hot sequence via overlapping 1 Mb windows.

    AlphaGenome consumes a 1 Mb window and emits 128-bp-resolution tracks over
    the whole window (no center-cropping).  We slide the window across the full
    (e.g. 2 Mb) region, mean over the resolved channels of each requested track,
    expand each 128-bp bin back to bp, and average across overlapping windows.

    Parameters
    ----------
    model : AlphaGenome (pruned, eval)
    seq_onehot : np.ndarray, shape (seq_len, 4 or 5) in ATCGN order.
    resolvers : dict[str, (head_name, np.ndarray)]  from ``load_alphagenome``.
    ordered_names : list[str]  output track order.
    org_idx : int
    device : torch.device or None
    step_fraction : float  fraction of the window to advance (0.5 = 50% overlap).

    Returns
    -------
    preds : np.ndarray, shape (seq_len, num_tracks)  bp-resolution 1D signal.
    """
    if device is None:
        device = next(model.parameters()).device

    seq_acgt = seq_onehot[:, :4].astype(np.float32)[:, ATCG_TO_ACGT]
    seq_len = seq_acgt.shape[0]
    num_tracks = len(ordered_names)
    needed_heads = tuple({resolvers[n][0] for n in ordered_names})

    preds = np.zeros((seq_len, num_tracks), dtype=np.float64)
    counts = np.zeros(seq_len, dtype=np.float64)

    step_size = max(1, int(AG_CONTEXT_LENGTH * step_fraction))
    # Ensure the tail of the region is covered even if not window-aligned.
    starts = list(range(0, max(1, seq_len - AG_CONTEXT_LENGTH) + 1, step_size))
    if starts[-1] != seq_len - AG_CONTEXT_LENGTH and seq_len > AG_CONTEXT_LENGTH:
        starts.append(seq_len - AG_CONTEXT_LENGTH)

    with torch.no_grad():
        for win_start in starts:
            if device.type != 'cpu':
                torch.cuda.empty_cache()
            s = max(win_start, 0)
            e = min(win_start + AG_CONTEXT_LENGTH, seq_len)
            window_seq = seq_acgt[s:e]
            if window_seq.shape[0] < AG_CONTEXT_LENGTH:   # right-pad with N (zeros)
                pad = AG_CONTEXT_LENGTH - window_seq.shape[0]
                window_seq = np.concatenate(
                    [window_seq, np.zeros((pad, 4), dtype=np.float32)], axis=0)

            inp = torch.from_numpy(window_seq).float().unsqueeze(0).to(device)
            outputs = model.predict(inp, org_idx, named_outputs=False,
                                    resolutions=(AG_BIN_SIZE,),
                                    heads=needed_heads, channels_last=True)

            seg_len = e - s
            counts[s:e] += 1.0
            for ti, name in enumerate(ordered_names):
                head_name, idxs = resolvers[name]
                # (1, out_bins, num_raw_tracks) -> mean over resolved channels
                bin_sig = (outputs[head_name][AG_BIN_SIZE][0][:, idxs]
                           .float().mean(dim=-1).cpu().numpy())
                bp_sig = np.repeat(bin_sig, AG_BIN_SIZE)       # 128-bp -> bp
                preds[s:e, ti] += bp_sig[:seg_len]

            del outputs, inp

    preds = preds / np.clip(counts[:, None], 1.0, None)
    return preds


# ---------------------------------------------------------------------------
# High-level "alphagenome_seq" knockout  (parallels enformer_seq_knockout)
# ---------------------------------------------------------------------------

def alphagenome_seq_knockout(seq_region, ctcf_region, atac_region, other_regions,
                             input_track_names,
                             alphagenome_model, alphagenome_track_names,
                             resolvers, org_idx,
                             perturb_track_names=None,
                             alt_seq_region=None,
                             window=2097152,
                             delta_mode='multiplicative', cap=10.0,
                             track_is_log1p=True,
                             device=None,
                             epsilon=1e-6):
    """Run AlphaGenome on WT vs ALT and transfer the fold-change onto tracks.

    Mirrors ``enformer_utils.enformer_seq_knockout`` (same inputs / return
    contract), so the downstream bigwig writers and plotting code are reused
    unchanged.  The ALT sequence is expected to be supplied fully materialised
    via ``alt_seq_region`` (the planning step applies the base substitutions).
    """
    perturb_track_names = (set(perturb_track_names)
                           if perturb_track_names is not None else None)

    if alt_seq_region is not None:
        alt_seq = alt_seq_region
        print('[alphagenome_seq] Using precomputed cumulative ALT sequence')
    else:
        alt_seq = seq_region

    # --- AlphaGenome predictions on WT and ALT ---
    wt_pred = alphagenome_predict_1d(alphagenome_model, seq_region, resolvers,
                                     alphagenome_track_names, org_idx, device=device)
    alt_pred = alphagenome_predict_1d(alphagenome_model, alt_seq, resolvers,
                                      alphagenome_track_names, org_idx, device=device)

    # --- delta / fold-change (identical math to the Enformer path) ---
    delta = alt_pred - wt_pred
    fold_change = alt_pred / np.clip(wt_pred, epsilon, None)
    log1p_wt = np.log1p(np.clip(wt_pred, 0, None))
    log1p_alt = np.log1p(np.clip(alt_pred, 0, None))
    log1p_delta = log1p_alt - log1p_wt
    fold_change_log1p = np.exp(log1p_delta)

    def _apply_to_track(track, track_idx):
        if track is None:
            return track
        track_len = len(track)
        if delta_mode == 'multiplicative':
            fc = downsample_to_track_resolution(fold_change[:, track_idx], track_len)
            return apply_enformer_delta_to_track(track, fc, mode='multiplicative',
                                                 cap=cap, track_is_log1p=track_is_log1p)
        elif delta_mode == 'additive':
            d = downsample_to_track_resolution(delta[:, track_idx], track_len)
            return apply_enformer_delta_to_track(track, d, mode='additive',
                                                 cap=cap, track_is_log1p=track_is_log1p)
        else:  # raw predictions
            pred = downsample_to_track_resolution(alt_pred[:, track_idx], track_len)
            return np.log1p(np.clip(pred, 0, None)) if track_is_log1p else pred

    other_offset = sum(1 for t in ['ctcf', 'atac'] if t in input_track_names)
    other_track_names = input_track_names[other_offset:]
    for track_idx, ag_name in enumerate(alphagenome_track_names):
        if perturb_track_names is not None and ag_name not in perturb_track_names:
            continue
        if ag_name == 'ctcf' and ctcf_region is not None:
            ctcf_region = _apply_to_track(ctcf_region, track_idx)
            print('[alphagenome_seq] Applied delta to CTCF track')
        elif ag_name == 'atac' and atac_region is not None:
            atac_region = _apply_to_track(atac_region, track_idx)
            print('[alphagenome_seq] Applied delta to ATAC track')
        elif other_regions is not None and ag_name in other_track_names:
            j = other_track_names.index(ag_name)
            other_regions[j] = _apply_to_track(other_regions[j], track_idx)
            print(f'[alphagenome_seq] Applied delta to {ag_name} track')

    alphagenome_results = {
        'fold_change': fold_change,
        'fold_change_log1p': fold_change_log1p,
        'delta': delta,
        'log1p_delta': log1p_delta,
        'wt_pred': wt_pred,
        'alt_pred': alt_pred,
        'enformer_track_names': alphagenome_track_names,   # keep key name for reuse
        'perturbed_track_names': (list(perturb_track_names)
                                  if perturb_track_names is not None
                                  else list(alphagenome_track_names)),
    }
    return ctcf_region, atac_region, other_regions, alphagenome_results
