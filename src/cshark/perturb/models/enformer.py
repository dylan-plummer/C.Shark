"""Enformer-based sequence-perturbation secondary predictor (extracted from single_locus).

- ``apply_enformer_seq_ko``: when ko-mode enformer_seq was used, load Enformer,
  predict the 1D delta on the mutated sequence, rewrite the perturbed input
  track(s), and write the enformer KO/delta bigwigs (perturb.py ~925-962).
  Returns (ctcf_region, atac_region, other_regions, enformer_perturbed_track_names).
- ``rewrite_enformer_ko_tracks``: re-write the enformer-KO plotting bigwigs from
  the final in-memory inputs (perturb.py ~1021-1044). Side-effect only.

Verbatim logic from the original single_deletion.
"""
import numpy as np

from cshark.inference.utils.inference_utils import write_tmp_pred_bigwig
from cshark.inference.utils.enformer_utils import (
    load_enformer_pretrained, load_enformer_from_checkpoint, enformer_seq_knockout,
    write_tmp_enformer_ko_bigwig, write_tmp_enformer_delta_bigwig,
    downsample_to_track_resolution,
)


def apply_enformer_seq_ko(assembly, atac_region, bigwig_log_transform, celltype, chr_name, ctcf_region, enformer_delta_cap, enformer_delta_mode, enformer_model_path, enformer_seq_active, enformer_tracks, input_track_names, input_track_paths, other_regions, seq_region, seq_region_wt, start, window):
    if enformer_seq_active:
        print('[enformer_seq] Loading Enformer model for cumulative sequence perturbation...')
        enf_target_tracks = enformer_tracks if enformer_tracks is not None else ['ctcf', 'atac', 'rad21']
        enf_species = 'mouse' if 'mm10' in (assembly or '') else 'human'
        if enformer_model_path is not None:
            enformer_model, enformer_track_names, enf_device = load_enformer_from_checkpoint(
                enformer_model_path, enformer_tracks=enf_target_tracks)
        else:
            enformer_model, enformer_track_names, enf_device = load_enformer_pretrained(
                target_tracks=enf_target_tracks, species=enf_species, celltype=celltype)

        ctcf_region, atac_region, other_regions, enformer_results = enformer_seq_knockout(
            seq_region_wt, ctcf_region, atac_region, other_regions,
            input_track_names, enformer_model, enformer_track_names,
            perturb_track_names=enf_target_tracks, alt_seq_region=seq_region,
            window=window, delta_mode=enformer_delta_mode, cap=enformer_delta_cap,
            track_is_log1p=bigwig_log_transform, device=enf_device,
        )
        print(f'[enformer_seq] Delta applied: mode={enformer_delta_mode}, cap={enformer_delta_cap}.')

        perturbed_track_names = set(enformer_results.get('perturbed_track_names', []))
        enformer_perturbed_track_names = {t.lower() for t in perturbed_track_names}
        for enf_idx, enf_name in enumerate(enformer_results['enformer_track_names']):
            if enf_name in perturbed_track_names and enf_name in input_track_names:
                track_path = input_track_paths[input_track_names.index(enf_name)]
                write_tmp_enformer_ko_bigwig(
                    track_path, enformer_results['fold_change'], enformer_results['delta'],
                    enformer_results['fold_change_log1p'], enformer_results['log1p_delta'],
                    enf_idx, enf_name, chr_name, start, window=window,
                    delta_mode=enformer_delta_mode, cap=enformer_delta_cap,
                    track_is_log1p=bigwig_log_transform,
                )
                write_tmp_enformer_delta_bigwig(
                    track_path, enformer_results['fold_change'], enformer_results['delta'],
                    enformer_results['fold_change_log1p'], enformer_results['log1p_delta'],
                    enf_idx, enf_name, chr_name, start, window=window,
                    delta_mode='additive', track_is_log1p=bigwig_log_transform,
                )
    else:
        enformer_perturbed_track_names = set()
    return ctcf_region, atac_region, other_regions, enformer_perturbed_track_names


def rewrite_enformer_ko_tracks(atac_region, bigwig_log_transform, chr_name, ctcf_region, enformer_perturbed_track_names, input_track_names, input_track_paths, other_regions, start, window):
    # Rewrite Enformer KO plotting tracks from final in-memory inputs
    if enformer_perturbed_track_names:
        other_offset = sum(1 for t in ['ctcf', 'atac'] if t in input_track_names)
        other_track_names = input_track_names[other_offset:]
        for track_name in sorted(enformer_perturbed_track_names):
            if track_name not in input_track_names:
                continue
            track_path = input_track_paths[input_track_names.index(track_name)]
            if track_name == 'ctcf':
                track_values = ctcf_region
            elif track_name == 'atac':
                track_values = atac_region
            elif other_regions is not None and track_name in other_track_names:
                track_values = other_regions[other_track_names.index(track_name)]
            else:
                track_values = None
            if track_values is not None:
                # Track values are in log1p space; convert to linear for bigwig
                if bigwig_log_transform:
                    track_values = np.expm1(track_values)
                write_tmp_pred_bigwig(
                    track_path, track_values, track_name, chr_name, start,
                    suffix='enformer_ko', window=window,
                )


# ---------------------------------------------------------------------------
# Allele-specific peak redistribution (opt-in --allele-peak-split)
#
# Instead of multiplying the Enformer fold-change directly onto the bulk
# experimental track, split each bulk track into ref/alt allele tracks by the
# predicted allele ratio, in LINEAR space. Faithful port of the user's
# reference script `1.split_exp_by_allele_ratio.py`:
#     frac_ref = R/(R+A)  (R+A>0)  else 0.5     # 50/50 fallback
#     out_ref  = min(2 * E * frac_ref, CAP)
#     out_alt  = min(2 * E * frac_alt, CAP)
# where R = Enformer pred on the reference/WT sequence (wt_pred),
#       A = Enformer pred on the alt sequence (alt_pred),
#       E = the experimental bulk WT track.
# At R+A==0 (or fc==1) -> 50/50 -> out = 2*E*0.5 = E (peak preserved).
# This module is ONLY reached when --allele-peak-split is set; the existing
# direct-apply path (apply_enformer_seq_ko) is untouched.
# ---------------------------------------------------------------------------

REDISTRIBUTION_CAP = 10.0   # default output-value cap (linear), matches reference script


def redistribute_by_allele_ratio(track, pred_ref_1d, pred_alt_1d, cap=REDISTRIBUTION_CAP,
                                  track_is_log1p=True):
    """Split one experimental track ``E`` into (ref, alt) by the allele ratio.

    Parameters
    ----------
    track : np.ndarray, shape (L,)
        Experimental bulk signal (log1p space if ``track_is_log1p``).
    pred_ref_1d, pred_alt_1d : np.ndarray, shape (L,)
        Reference / alt Enformer predictions, ALREADY resampled to ``len(track)``.
    cap : float
        Upper bound on the output value in LINEAR space (reference CAP=10).
    track_is_log1p : bool
        If True the input/output are log1p-transformed; the split is done in
        linear space (expm1 -> split -> log1p), matching the rest of the engine.

    Returns
    -------
    (out_ref, out_alt) : tuple of np.ndarray, shape (L,), same space as ``track``.
    """
    ref = np.nan_to_num(np.asarray(pred_ref_1d, dtype=np.float64), nan=0.0)
    alt = np.nan_to_num(np.asarray(pred_alt_1d, dtype=np.float64), nan=0.0)
    denom = ref + alt
    with np.errstate(invalid='ignore', divide='ignore'):
        frac_ref = np.where(denom > 0, ref / denom, 0.5)   # 50/50 fallback
    frac_alt = 1.0 - frac_ref

    E_lin = np.expm1(track) if track_is_log1p else np.asarray(track, dtype=np.float64).copy()
    out_ref_lin = np.minimum(2.0 * E_lin * frac_ref, cap)
    out_alt_lin = np.minimum(2.0 * E_lin * frac_alt, cap)

    if track_is_log1p:
        out_ref = np.log1p(np.clip(out_ref_lin, 0, None))
        out_alt = np.log1p(np.clip(out_alt_lin, 0, None))
    else:
        out_ref = np.clip(out_ref_lin, 0, None)
        out_alt = np.clip(out_alt_lin, 0, None)
    return out_ref.astype(track.dtype), out_alt.astype(track.dtype)


def redistribute_enformer_alleles(ctcf_region, atac_region, other_regions, input_track_names,
                                  enformer_results, cap=REDISTRIBUTION_CAP, track_is_log1p=True):
    """Build (ref, alt) allele track sets from the WT tracks + Enformer preds.

    Mirrors the track->enformer-column mapping in ``enformer_seq_knockout``:
    only the perturbed enformer tracks are redistributed; every other track is
    copied unchanged into BOTH alleles (so non-perturbed inputs stay = bulk).

    Returns
    -------
    (ref_ctcf, ref_atac, ref_other), (alt_ctcf, alt_atac, alt_other)
    """
    wt_pred = enformer_results['wt_pred']      # (enf_len, num_tracks) -- reference allele
    alt_pred = enformer_results['alt_pred']    # (enf_len, num_tracks) -- alt allele
    enformer_track_names = enformer_results['enformer_track_names']
    perturbed = set(enformer_results.get('perturbed_track_names', enformer_track_names))

    # Default: both alleles = the WT (bulk) track. Perturbed tracks overwritten below.
    ref_ctcf = ctcf_region.copy() if ctcf_region is not None else None
    alt_ctcf = ctcf_region.copy() if ctcf_region is not None else None
    ref_atac = atac_region.copy() if atac_region is not None else None
    alt_atac = atac_region.copy() if atac_region is not None else None
    ref_other = [r.copy() for r in other_regions] if other_regions is not None else None
    alt_other = [r.copy() for r in other_regions] if other_regions is not None else None

    other_offset = sum(1 for t in ['ctcf', 'atac'] if t in input_track_names)
    other_track_names = input_track_names[other_offset:]

    for enf_idx, enf_name in enumerate(enformer_track_names):
        if enf_name not in perturbed or enf_name not in input_track_names:
            continue
        if enf_name == 'ctcf' and ctcf_region is not None:
            E = ctcf_region
        elif enf_name == 'atac' and atac_region is not None:
            E = atac_region
        elif other_regions is not None and enf_name in other_track_names:
            E = other_regions[other_track_names.index(enf_name)]
        else:
            continue
        track_len = len(E)
        ref_1d = downsample_to_track_resolution(wt_pred[:, enf_idx], track_len)
        alt_1d = downsample_to_track_resolution(alt_pred[:, enf_idx], track_len)
        out_ref, out_alt = redistribute_by_allele_ratio(
            E, ref_1d, alt_1d, cap=cap, track_is_log1p=track_is_log1p)
        if enf_name == 'ctcf':
            ref_ctcf, alt_ctcf = out_ref, out_alt
        elif enf_name == 'atac':
            ref_atac, alt_atac = out_ref, out_alt
        else:
            j = other_track_names.index(enf_name)
            ref_other[j] = out_ref
            alt_other[j] = out_alt
        print(f'[allele-peak-split] Redistributed {enf_name} track into ref/alt')
    return (ref_ctcf, ref_atac, ref_other), (alt_ctcf, alt_atac, alt_other)


def apply_enformer_peak_split(assembly, atac_region, bigwig_log_transform, celltype, chr_name,
                              ctcf_region, enformer_delta_cap, enformer_delta_mode,
                              enformer_model_path, enformer_seq_active, enformer_tracks,
                              input_track_names, other_regions, seq_region, seq_region_wt,
                              start, window):
    """Opt-in allele-specific path: run Enformer on the WT (ref) and ALT sequences
    and split each perturbed experimental track into ref/alt allele tracks.

    Loads Enformer exactly like ``apply_enformer_seq_ko`` and reuses
    ``enformer_seq_knockout`` purely to obtain ``wt_pred``/``alt_pred`` (its
    direct-applied tracks are discarded; the original WT tracks are used as E).

    Returns
    -------
    (ref_set, alt_set, enformer_perturbed_track_names, enformer_results)
        ref_set/alt_set = (ctcf, atac, other_regions) for each allele.
    """
    print('[allele-peak-split] Loading Enformer for allele-specific peak redistribution...')
    enf_target_tracks = enformer_tracks if enformer_tracks is not None else ['ctcf', 'atac', 'rad21']
    enf_species = 'mouse' if 'mm10' in (assembly or '') else 'human'
    if enformer_model_path is not None:
        enformer_model, enformer_track_names, enf_device = load_enformer_from_checkpoint(
            enformer_model_path, enformer_tracks=enf_target_tracks)
    else:
        enformer_model, enformer_track_names, enf_device = load_enformer_pretrained(
            target_tracks=enf_target_tracks, species=enf_species, celltype=celltype)

    # Run on WT (reference) vs ALT seq; we only consume wt_pred/alt_pred below.
    _, _, _, enformer_results = enformer_seq_knockout(
        seq_region_wt, ctcf_region, atac_region, other_regions,
        input_track_names, enformer_model, enformer_track_names,
        perturb_track_names=enf_target_tracks, alt_seq_region=seq_region,
        window=window, delta_mode=enformer_delta_mode, cap=enformer_delta_cap,
        track_is_log1p=bigwig_log_transform, device=enf_device,
    )

    ref_set, alt_set = redistribute_enformer_alleles(
        ctcf_region, atac_region, other_regions, input_track_names,
        enformer_results, cap=enformer_delta_cap, track_is_log1p=bigwig_log_transform)

    perturbed_track_names = set(enformer_results.get('perturbed_track_names', []))
    enformer_perturbed_track_names = {t.lower() for t in perturbed_track_names}
    print(f'[allele-peak-split] Built ref/alt allele track sets (CAP={enformer_delta_cap}).')
    return ref_set, alt_set, enformer_perturbed_track_names, enformer_results
