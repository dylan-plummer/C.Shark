"""AlphaGenome-based sequence-perturbation secondary predictor.

AlphaGenome analogue of ``perturb/models/enformer.py``.  ``apply_alphagenome_seq_ko``
is the drop-in counterpart of ``apply_enformer_seq_ko``: when ko-mode
``alphagenome_seq`` was used, it loads AlphaGenome (heads pruned to save memory),
predicts the 1D delta on the mutated sequence, rewrites the perturbed input
track(s), and writes the same ``tmp/{track}_enformer_ko.bw`` / ``_enformer_delta.bw``
plotting bigwigs so the existing pyGenomeTracks pipeline is reused unchanged.

Returns (ctcf_region, atac_region, other_regions, perturbed_track_names).

The KO/delta bigwig writers (``write_tmp_enformer_ko_bigwig`` /
``write_tmp_enformer_delta_bigwig``) are model-agnostic and shared with the
Enformer path, as is ``rewrite_enformer_ko_tracks`` (imported by single_locus).
"""
from cshark.inference.utils.enformer_utils import (
    write_tmp_enformer_ko_bigwig, write_tmp_enformer_delta_bigwig,
)
from cshark.inference.utils.alphagenome_utils import (
    load_alphagenome, alphagenome_seq_knockout,
)


def apply_alphagenome_seq_ko(assembly, atac_region, bigwig_log_transform, celltype,
                             chr_name, ctcf_region, enformer_delta_cap,
                             enformer_delta_mode, enformer_tracks,
                             alphagenome_model_path, alphagenome_metadata_path,
                             alphagenome_seq_active, input_track_names,
                             input_track_paths, other_regions, seq_region,
                             seq_region_wt, start, window):
    if not alphagenome_seq_active:
        return ctcf_region, atac_region, other_regions, set()

    print('[alphagenome_seq] Loading AlphaGenome model for cumulative sequence perturbation...')
    ag_target_tracks = enformer_tracks if enformer_tracks is not None else ['ctcf', 'atac', 'rad21']
    ag_species = 'mouse' if 'mm10' in (assembly or '') else 'human'

    ag_model, ag_track_names, ag_device, org_idx, resolvers = load_alphagenome(
        alphagenome_model_path, target_tracks=ag_target_tracks, species=ag_species,
        celltype=celltype, metadata_path=alphagenome_metadata_path)

    ctcf_region, atac_region, other_regions, ag_results = alphagenome_seq_knockout(
        seq_region_wt, ctcf_region, atac_region, other_regions,
        input_track_names, ag_model, ag_track_names, resolvers, org_idx,
        perturb_track_names=ag_target_tracks, alt_seq_region=seq_region,
        window=window, delta_mode=enformer_delta_mode, cap=enformer_delta_cap,
        track_is_log1p=bigwig_log_transform, device=ag_device,
    )
    print(f'[alphagenome_seq] Delta applied: mode={enformer_delta_mode}, cap={enformer_delta_cap}.')

    perturbed_track_names = set(ag_results.get('perturbed_track_names', []))
    alphagenome_perturbed_track_names = {t.lower() for t in perturbed_track_names}
    for ag_idx, ag_name in enumerate(ag_results['enformer_track_names']):
        if ag_name in perturbed_track_names and ag_name in input_track_names:
            track_path = input_track_paths[input_track_names.index(ag_name)]
            write_tmp_enformer_ko_bigwig(
                track_path, ag_results['fold_change'], ag_results['delta'],
                ag_results['fold_change_log1p'], ag_results['log1p_delta'],
                ag_idx, ag_name, chr_name, start, window=window,
                delta_mode=enformer_delta_mode, cap=enformer_delta_cap,
                track_is_log1p=bigwig_log_transform, tool='alphagenome',
            )
            write_tmp_enformer_delta_bigwig(
                track_path, ag_results['fold_change'], ag_results['delta'],
                ag_results['fold_change_log1p'], ag_results['log1p_delta'],
                ag_idx, ag_name, chr_name, start, window=window,
                delta_mode='additive', track_is_log1p=bigwig_log_transform,
                tool='alphagenome',
            )
    return ctcf_region, atac_region, other_regions, alphagenome_perturbed_track_names


def apply_alphagenome_peak_split(assembly, atac_region, bigwig_log_transform, celltype, chr_name,
                                 ctcf_region, enformer_delta_cap, enformer_delta_mode,
                                 alphagenome_model_path, alphagenome_metadata_path,
                                 enformer_tracks, input_track_names, other_regions,
                                 seq_region, seq_region_wt, start, window):
    """AlphaGenome counterpart of ``apply_enformer_peak_split``: run AlphaGenome on the
    WT (ref) and ALT sequences and split each perturbed experimental track into ref/alt
    allele tracks. Reuses ``alphagenome_seq_knockout`` (for wt_pred/alt_pred -- same dict
    shape as the Enformer path) and the backbone-agnostic ``redistribute_enformer_alleles``.

    Returns
    -------
    (ref_set, alt_set, perturbed_track_names, ag_results)
        ref_set/alt_set = (ctcf, atac, other_regions) for each allele.
    """
    from cshark.perturb.models.enformer import redistribute_enformer_alleles

    print('[allele-peak-split] Loading AlphaGenome for allele-specific peak redistribution...')
    ag_target_tracks = enformer_tracks if enformer_tracks is not None else ['ctcf', 'atac', 'rad21']
    ag_species = 'mouse' if 'mm10' in (assembly or '') else 'human'
    ag_model, ag_track_names, ag_device, org_idx, resolvers = load_alphagenome(
        alphagenome_model_path, target_tracks=ag_target_tracks, species=ag_species,
        celltype=celltype, metadata_path=alphagenome_metadata_path)

    # alphagenome_seq_knockout REASSIGNS other_regions[i] in place (the delta-applied
    # track); pass copies so the pristine WT tracks remain as E for redistribution.
    _ctcf = ctcf_region.copy() if ctcf_region is not None else None
    _atac = atac_region.copy() if atac_region is not None else None
    _other = [r.copy() for r in other_regions] if other_regions is not None else None
    _, _, _, ag_results = alphagenome_seq_knockout(
        seq_region_wt, _ctcf, _atac, _other,
        input_track_names, ag_model, ag_track_names, resolvers, org_idx,
        perturb_track_names=ag_target_tracks, alt_seq_region=seq_region,
        window=window, delta_mode=enformer_delta_mode, cap=enformer_delta_cap,
        track_is_log1p=bigwig_log_transform, device=ag_device,
    )

    # Redistribute using the ORIGINAL (WT bulk) experimental tracks as E.
    ref_set, alt_set = redistribute_enformer_alleles(
        ctcf_region, atac_region, other_regions, input_track_names,
        ag_results, cap=enformer_delta_cap, track_is_log1p=bigwig_log_transform)
    perturbed = {t.lower() for t in ag_results.get('perturbed_track_names', [])}
    print(f'[allele-peak-split] Built ref/alt allele track sets via AlphaGenome (CAP={enformer_delta_cap}).')
    return ref_set, alt_set, perturbed, ag_results
