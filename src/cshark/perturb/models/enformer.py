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
