"""Allele-specific peak-redistribution runner (opt-in ``--allele-peak-split``).

This is a SEPARATE path, only reached from ``run_single_locus`` when
``cfg.allele_peak_split`` is set AND an ``enformer_seq`` perturbation is active.
The existing single-locus flow is left completely untouched: when the flag is
off, ``run_single_locus`` never calls into this module.

Pipeline (all redistribution math in LINEAR space; see ``models.enformer``):
  1. Enformer is run on the WT (reference) and ALT sequences; each perturbed
     experimental track is split into a ``ref`` and an ``alt`` allele track via
     ``out = min(2 * E * frac, CAP)`` (the user's reference script formula).
  2. If a hierarchical RAD21 predictor is given, RAD21 is predicted for EACH
     allele's tracks and the experimental RAD21 is split by the two alleles'
     predicted ratio (same ``2*E*frac`` math). This needs both alleles' preds,
     so it runs once for both before the per-allele model passes.
  3. The 8-track model is run once per allele and a full output set
     (Hi-C cooler, arcs, 1D tracks, pyGenomeTracks figure) is written with a
     ``_ref`` / ``_alt`` filename label.

Output helpers (``plot_prediction_matrix`` / ``write_tmp_cooler`` / ``write_arcs``
/ ``build_track_inis`` / ``run_pygenometracks`` / ``plot_pred_1d_tracks``) are the
same ones the existing path uses, so per-allele outputs match its style. Each
allele is processed fully and sequentially, so the shared ``tmp/`` scratch files
are produced and consumed before the next allele overwrites them.
"""
import os
import numpy as np
from skimage.transform import resize

from cshark.data.data_feature import HiCFeature
from cshark.inference.utils import plot_utils
from cshark.inference.utils.inference_utils import write_tmp_cooler, preprocess_default
from cshark.inference.utils.hierarchical_utils import predict_rad21

from cshark.perturb.output.arcs import write_arcs, write_regions
from cshark.perturb.output.plots import plot_prediction_matrix, plot_pred_1d_tracks
from cshark.perturb.output.tracks_ini import build_track_inis, run_pygenometracks
from cshark.perturb.models.enformer import (
    apply_enformer_peak_split, redistribute_by_allele_ratio, rewrite_enformer_ko_tracks,
)

# plotting constants (verbatim from single_locus / the original perturb.py)
font_size = 15
plot_width = 17
track_label_fraction = 0.13


def _redistribute_rad21_alleles(ref_set, alt_set, *, seq_region, other_regions_wt,
                                input_track_names, hierarchical_rad21_model,
                                cap, bigwig_log_transform):
    """Per-allele RAD21 redistribution (mirrors the enformer split, via hierarchical).

    Predict RAD21 from each allele's redistributed tracks, then split the
    experimental WT RAD21 by the two alleles' predicted ratio. Returns updated
    (ref_set, alt_set); a no-op when no hierarchical model / no rad21 input.
    """
    if hierarchical_rad21_model is None or 'rad21' not in input_track_names:
        return ref_set, alt_set

    other_offset = sum(1 for t in ['ctcf', 'atac'] if t in input_track_names)
    other_track_names = input_track_names[other_offset:]
    rad21_other_idx = other_track_names.index('rad21')
    rad21_tensor_idx = input_track_names.index('rad21')   # position among non-seq channels
    E_rad21 = other_regions_wt[rad21_other_idx]           # experimental WT rad21 (log1p)
    track_len = len(E_rad21)

    ref_ctcf, ref_atac, ref_other = ref_set
    alt_ctcf, alt_atac, alt_other = alt_set

    def _pred(ctcf_a, atac_a, other_a):
        inp = preprocess_default(seq_region, ctcf_a, atac_a, other_a)
        pred = predict_rad21(hierarchical_rad21_model, inp, rad21_tensor_idx)  # linear, model res
        pred = np.atleast_1d(np.asarray(pred, dtype=np.float64))
        if len(pred) != track_len:
            pred = np.interp(np.linspace(0, 1, track_len),
                             np.linspace(0, 1, len(pred)), pred)
        return pred

    rad21_pred_ref = _pred(ref_ctcf, ref_atac, ref_other)
    rad21_pred_alt = _pred(alt_ctcf, alt_atac, alt_other)
    out_ref, out_alt = redistribute_by_allele_ratio(
        E_rad21, rad21_pred_ref, rad21_pred_alt, cap=cap, track_is_log1p=bigwig_log_transform)
    ref_other[rad21_other_idx] = out_ref
    alt_other[rad21_other_idx] = out_alt
    print('[allele-peak-split] Redistributed RAD21 (hierarchical) into ref/alt')
    return (ref_ctcf, ref_atac, ref_other), (alt_ctcf, alt_atac, alt_other)


def _predict_and_write_allele(cfg, label, *, model, seq_region, ctcf_a, atac_a, other_a,
                              input_track_names, input_track_paths, pred_before, pred_before_1d,
                              enformer_perturbed_track_names, extra_perturbed=frozenset(),
                              plot_track_names, plot_track_paths,
                              deletion_starts, deletion_widths, res, image_scale, window):
    """Run the model on one allele's track set and write its full output set,
    with a ``_<label>`` filename prefix. Mirrors single_locus lines ~234-329 for
    the enformer_seq case (no track-deletion / chipseq-KO blocks, which never
    apply to enformer_seq)."""
    output_path = cfg.output_path
    celltype = cfg.celltype
    chr_name = cfg.chr_name
    start = cfg.start
    assembly = cfg.assembly
    model_path = cfg.model_path
    ctcf_path = cfg.ctcf_path
    bigwig_log_transform = cfg.bigwig_log_transform
    no_plots = cfg.no_plots
    silent = cfg.silent
    region = cfg.region
    plot_diff = cfg.plot_diff
    plot_pred_bigwigs = cfg.plot_pred_bigwigs if cfg.plot_pred_bigwigs is not None else []
    plot_pred_log2fc = False
    ctcf_motif_p = cfg.ctcf_motif_p
    ko_data = cfg.ko_data

    base = cfg.outname
    if base and not base.endswith('_'):
        base += '_'
    allele_outname = f'{base}{label}_'

    # Tracks shown as "<track> SNP perturb": the enformer-redistributed ones, plus
    # rad21 (hierarchical per-allele redistribution) when present -- so the perturbed
    # rad21 actually appears on the figure instead of only its WT track.
    plot_perturbed = set(enformer_perturbed_track_names) | set(extra_perturbed)

    # Write the perturbed plotting bigwigs from THIS allele's final inputs.
    rewrite_enformer_ko_tracks(
        atac_region=atac_a, bigwig_log_transform=bigwig_log_transform, chr_name=chr_name,
        ctcf_region=ctcf_a, enformer_perturbed_track_names=plot_perturbed,
        input_track_names=input_track_names, input_track_paths=input_track_paths,
        other_regions=other_a, start=start, window=window)

    # KO prediction for this allele
    pred_output = model.predict_arrays(seq_region, ctcf_a, atac_a, other_a, input_track_names[2:])
    pred = pred_output['hic']
    pred_1d = pred_output['1d']
    if not no_plots:
        plot_prediction_matrix(
            pred, os.path.join(output_path, f'{allele_outname}{celltype}_{chr_name}_{start}_pred.png'),
            f'Prediction ({label} allele)')

    # 1D track prediction bigwigs (pred_1d already linear from prediction())
    plot_pred_1d_tracks(
        celltype=celltype, chr_name=chr_name, input_track_names=input_track_names,
        input_track_paths=input_track_paths, model_path=model_path, no_plots=no_plots,
        outname=allele_outname, output_path=output_path, plot_pred_bigwigs=plot_pred_bigwigs,
        plot_pred_log2fc=plot_pred_log2fc, plot_track_names=plot_track_names, pred_1d=pred_1d,
        pred_before_1d=pred_before_1d, start=start)

    # Ground truth (best-effort, identical logic to the existing path)
    plot_ground_truth = False
    mat = None
    if not no_plots:
        try:
            ctcf_filename = os.path.basename(ctcf_path).split('.')[0]
            hic_path = ctcf_path.replace('genomic_features', 'hic_matrix').replace(
                f'/{ctcf_filename}.bw', '') + f'/{chr_name}.npz'
            hic = HiCFeature(path=hic_path)
            gt_res = 10000 if res == 8192 else (5000 if res == 4096 else res)
            mat = hic.get(start, window=int(window), res=gt_res)
            mat = resize(mat, (int(image_scale), int(image_scale)), anti_aliasing=True)
            mat += 0.01
            plot = plot_utils.MatrixPlot(output_path, mat, 'ground_truth', celltype, chr_name, start)
            plot.plot(vmin=1.0, vmax=2.5)
            plot_ground_truth = True
        except Exception as e:
            print(e)
            print('No ground truth found')
            mat = np.zeros_like(pred)

    # Coolers (shared tmp/ scratch, consumed by pyGenomeTracks below before the
    # next allele overwrites them)
    write_tmp_cooler(pred, chr_name, start, res=res)
    write_tmp_cooler(pred_before, chr_name, start, out_file='tmp/tmp_before.cool', res=res)
    if plot_ground_truth:
        write_tmp_cooler(mat, chr_name, start, window=(int(window * 2)), out_file='tmp/tmp_true.cool', res=res)
    diff = pred - pred_before
    write_tmp_cooler(diff, chr_name, start, out_file='tmp/tmp_diff.cool', res=res)

    write_regions(deletion_starts, deletion_widths, chr_name, 'tmp/regions.bed')

    region_start = int(region.split(':')[1].split('-')[0]) if region is not None else start
    region_end = int(region.split(':')[1].split('-')[1]) if region is not None else start + window
    write_arcs(pred_before, chr_name, start, res, region_start, region_end, 'tmp/arcs.bed', quantile=0.99)
    if plot_diff:
        write_arcs(diff, chr_name, start, res, region_start, region_end, 'tmp/arcs_diff.bed', two_sided=True)
    write_arcs(pred, chr_name, start, res, region_start, region_end, 'tmp/arcs_ko.bed', quantile=0.99)
    if plot_ground_truth:
        write_arcs(mat, chr_name, start, res, region_start, region_end, 'tmp/arcs_true.bed', quantile=0.99)

    # pyGenomeTracks .ini files. NOTE: hierarchical_active is forced False for the
    # allele path -- the hierarchical *diagnostic* track is not plotted here (its
    # tmp bigwigs are not written), but the RAD21 redistribution still feeds the
    # prediction. Enformer-KO tracks ARE plotted (rewritten above).
    build_track_inis(
        assembly=assembly, celltype=celltype, chr_name=chr_name, ctcf_motif_p=ctcf_motif_p,
        ctcf_path=ctcf_path, enformer_perturbed_track_names=plot_perturbed,
        enformer_seq_active=True, hierarchical_active=False,
        input_track_names=input_track_names, input_track_paths=input_track_paths, ko_data=ko_data,
        max_val_diff=cfg.max_val_diff, max_val_pred=cfg.max_val_pred, max_val_true=cfg.max_val_true,
        min_val_diff=cfg.min_val_diff, min_val_pred=cfg.min_val_pred, min_val_true=cfg.min_val_true,
        plot_bigwig_q=cfg.plot_bigwig_q, plot_diff=plot_diff, plot_ground_truth=plot_ground_truth,
        plot_pred_bigwigs=plot_pred_bigwigs, plot_pred_log2fc=plot_pred_log2fc,
        plot_track_names=plot_track_names, plot_track_paths=plot_track_paths, start=start)

    run_pygenometracks(
        region=region, celltype=celltype, chr_name=chr_name, deletion_starts=deletion_starts,
        deletion_widths=deletion_widths, font_size=font_size, no_plots=no_plots,
        outname=allele_outname, output_path=output_path, plot_diff=plot_diff,
        plot_ground_truth=plot_ground_truth, plot_width=plot_width, silent=silent, start=start,
        track_label_fraction=track_label_fraction, window=window, is_snp=True)
    print(f'[allele-peak-split] Wrote {label} allele outputs (prefix "{allele_outname}").')


def run_allele_peak_split(cfg, *, model, seq_region, seq_region_wt,
                          ctcf_region, atac_region, other_regions,
                          other_regions_wt, input_track_names, input_track_paths,
                          pred_before, pred_before_1d, plot_track_names, plot_track_paths,
                          hierarchical_rad21_model, deletion_starts, deletion_widths,
                          res, image_scale, window):
    """Entry point for the opt-in allele-specific peak-redistribution path."""
    cap = cfg.enformer_delta_cap  # output-value CAP (reference script uses 10)

    # 1. Enformer redistribution -> ref/alt allele track sets.
    ref_set, alt_set, enformer_perturbed_track_names, _ = apply_enformer_peak_split(
        assembly=cfg.assembly, atac_region=atac_region, bigwig_log_transform=cfg.bigwig_log_transform,
        celltype=cfg.celltype, chr_name=cfg.chr_name, ctcf_region=ctcf_region,
        enformer_delta_cap=cfg.enformer_delta_cap, enformer_delta_mode=cfg.enformer_delta_mode,
        enformer_model_path=cfg.enformer_model_path, enformer_seq_active=True,
        enformer_tracks=cfg.enformer_tracks, input_track_names=input_track_names,
        other_regions=other_regions, seq_region=seq_region, seq_region_wt=seq_region_wt,
        start=cfg.start, window=window)

    # 2. Per-allele RAD21 redistribution (needs both alleles' preds first).
    ref_set, alt_set = _redistribute_rad21_alleles(
        ref_set, alt_set, seq_region=seq_region, other_regions_wt=other_regions_wt,
        input_track_names=input_track_names, hierarchical_rad21_model=hierarchical_rad21_model,
        cap=cap, bigwig_log_transform=cfg.bigwig_log_transform)

    # rad21 is redistributed via the hierarchical per-allele step (not enformer), so
    # add it to the plotted perturbed-tracks set when a hierarchical model is used --
    # otherwise its perturbed track never appears (only its WT track would).
    extra_perturbed = ({'rad21'} if (hierarchical_rad21_model is not None
                                     and 'rad21' in input_track_names) else set())

    # 3. Per allele: predict + full output set.
    for label, (ctcf_a, atac_a, other_a) in (('ref', ref_set), ('alt', alt_set)):
        _predict_and_write_allele(
            cfg, label, model=model, seq_region=seq_region, ctcf_a=ctcf_a, atac_a=atac_a,
            other_a=other_a, input_track_names=input_track_names, input_track_paths=input_track_paths,
            pred_before=pred_before, pred_before_1d=pred_before_1d,
            enformer_perturbed_track_names=enformer_perturbed_track_names,
            extra_perturbed=extra_perturbed,
            plot_track_names=plot_track_names, plot_track_paths=plot_track_paths,
            deletion_starts=deletion_starts, deletion_widths=deletion_widths,
            res=res, image_scale=image_scale, window=window)
    print('[allele-peak-split] Done: ref + alt allele predictions written.')
