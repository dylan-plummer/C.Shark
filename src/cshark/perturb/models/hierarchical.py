"""Hierarchical RAD21 secondary predictor (extracted from single_locus).

Two verbatim-extracted blocks from the original ``single_deletion``:
- ``prepare_rad21_input``: when rad21 is absent from the input bigwigs, predict it
  and insert it at the position the main model expects (perturb.py ~716-760).
- ``apply_rad21_update``: compute the WT-vs-KO rad21 delta, rewrite the rad21
  channel, and write the diagnostic bigwigs (perturb.py ~977-1019).

Both return the (possibly reassigned) ``other_regions``; ``input_track_names`` /
``input_track_paths`` are mutated in place. Logic is byte-identical to the original.
"""
import os
import numpy as np

from cshark.inference.utils.hierarchical_utils import (
    hierarchical_rad21_update,
    write_tmp_hierarchical_rad21_bigwig,
    write_tmp_hierarchical_delta_bigwig,
)


def prepare_rad21_input(atac_region, ctcf_region, hierarchical_rad21_model, input_track_names, input_track_paths, model_path, other_regions, seq_region):
    if hierarchical_rad21_model is not None and 'rad21' not in input_track_names:
        from cshark.inference.utils.hierarchical_utils import predict_rad21
        from cshark.inference.utils.model_utils import get_all_track_names
        from cshark.inference.utils.inference_utils import preprocess_default

        main_all_tracks, _, _ = get_all_track_names(model_path)
        if 'rad21' in main_all_tracks:
            # Determine where rad21 sits in the "other" slots of the main model
            other_main_names = [t for t in main_all_tracks if t not in ('ctcf', 'atac')]
            rad21_other_pos = other_main_names.index('rad21')

            # Predict rad21 from the current WT inputs (no channel removal needed:
            # our input tensor already excludes rad21)
            wt_inputs_np = preprocess_default(seq_region, ctcf_region, atac_region, other_regions)
            rad21_linear = predict_rad21(hierarchical_rad21_model, wt_inputs_np,
                                         rad21_idx=None)  # inputs already exclude rad21

            # Resample from model output resolution to bigwig track resolution
            ref_track = ctcf_region if ctcf_region is not None else (
                atac_region if atac_region is not None else
                (other_regions[0] if other_regions else None))
            track_len = len(ref_track) if ref_track is not None else len(rad21_linear)
            if len(rad21_linear) != track_len:
                rad21_linear = np.interp(
                    np.linspace(0, 1, track_len),
                    np.linspace(0, 1, len(rad21_linear)),
                    rad21_linear,
                )
            rad21_log1p = np.log1p(rad21_linear)

            if other_regions is None:
                other_regions = [rad21_log1p]
            else:
                other_regions.insert(rad21_other_pos, rad21_log1p)

            # Use the WT-pred bigwig path as the "experimental" track for plotting.
            # It will be written later by write_tmp_hierarchical_rad21_bigwig.
            rad21_pred_bw = os.path.abspath('tmp/rad21_hierarchical_wt_pred.bw')
            insert_at_global = 2 + rad21_other_pos  # position in full track list
            input_track_names.insert(insert_at_global, 'rad21')
            input_track_paths.insert(insert_at_global, rad21_pred_bw)

            print(f'[hierarchical] Predicted RAD21 from {len(input_track_names) - 1} input tracks, '
                  f'inserted at other-position {rad21_other_pos}.')
    return other_regions


def apply_rad21_update(atac_region, atac_region_wt, chr_name, ctcf_region, ctcf_region_wt, hierarchical_delta_cap, hierarchical_delta_mode, hierarchical_rad21_model, input_track_names, input_track_paths, other_regions, other_regions_wt, seq_region, seq_region_wt, start, window):
    if hierarchical_rad21_model is not None and 'rad21' in input_track_names:
        other_offset = sum(1 for t in ['ctcf', 'atac'] if t in input_track_names)
        other_track_names = input_track_names[other_offset:]
        rad21_other_idx = other_track_names.index('rad21')
        experimental_rad21_log1p = other_regions_wt[rad21_other_idx].copy()

        # rad21_idx is rad21's position among ALL non-seq input channels
        # (ctcf=0, atac=1, then others in order) — this is what predict_rad21 uses
        # to remove the rad21 channel from the preprocess_default tensor.
        rad21_tensor_idx = input_track_names.index('rad21')

        # Use an existing bigwig for the chromosome header (the path stored in
        # input_track_paths for rad21 may be a not-yet-written tmp/ path when
        # rad21 was predicted/inserted rather than provided directly).
        rad21_bw_path = input_track_paths[input_track_names.index('rad21')]
        if not os.path.exists(rad21_bw_path):
            # Fall back to first available real bigwig for header
            rad21_bw_path = next((p for p in input_track_paths if os.path.exists(p)), rad21_bw_path)

        other_regions, hierarchical_results = hierarchical_rad21_update(
            hierarchical_rad21_model, rad21_tensor_idx,
            seq_region_wt, ctcf_region_wt, atac_region_wt, other_regions_wt,
            seq_region, ctcf_region, atac_region, other_regions,
            experimental_rad21_log1p,
            input_track_names,
            delta_mode=hierarchical_delta_mode,
            cap=hierarchical_delta_cap,
            window=window,
        )

        write_tmp_hierarchical_rad21_bigwig(
            rad21_bw_path,
            hierarchical_results['wt_pred'],          # linear
            hierarchical_results['ko_pred'],           # linear
            hierarchical_results['perturbed_rad21'],   # log1p or None
            chr_name, start, window=window,
        )
        write_tmp_hierarchical_delta_bigwig(
            rad21_bw_path,
            hierarchical_results['delta'],
            hierarchical_results['fold_change'],
            chr_name, start, window=window,
        )
    return other_regions


