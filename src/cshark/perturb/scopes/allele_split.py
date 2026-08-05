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
import re
import shutil
import numpy as np
from skimage.transform import resize

from cshark.data.data_feature import HiCFeature
from cshark.inference.utils import plot_utils
from cshark.inference.utils.inference_utils import (
    write_tmp_cooler, preprocess_default, write_tmp_pred_bigwig,
)
from cshark.inference.utils.hierarchical_utils import predict_rad21
from cshark.inference.tracks_files import get_tracks

from cshark.perturb.output.arcs import write_arcs, write_regions
from cshark.perturb.output.plots import plot_prediction_matrix, plot_pred_1d_tracks
from cshark.perturb.output.tracks_ini import build_track_inis, run_pygenometracks, get_axis_range_from_bigwig
from cshark.perturb.models.enformer import (
    apply_enformer_peak_split, redistribute_by_allele_ratio, rewrite_enformer_ko_tracks,
)

# plotting constants (verbatim from single_locus / the original perturb.py)
font_size = 15
plot_width = 17
track_label_fraction = 0.13


def _redistribute_rad21_alleles(set_a, set_b, *, seq_a, seq_b, other_regions_wt,
                                input_track_names, hierarchical_rad21_model,
                                cap, bigwig_log_transform):
    """Per-allele RAD21 redistribution (mirrors the enformer split, via hierarchical).

    Predict RAD21 from EACH allele's redistributed tracks (using that allele's own
    sequence ``seq_a`` / ``seq_b``), then split the experimental WT RAD21 by the two
    alleles' predicted ratio. Returns updated (set_a, set_b); a no-op when no
    hierarchical model / no rad21 input. For function #1 pass ``seq_a == seq_b``.
    """
    if hierarchical_rad21_model is None or 'rad21' not in input_track_names:
        return set_a, set_b

    other_offset = sum(1 for t in ['ctcf', 'atac'] if t in input_track_names)
    other_track_names = input_track_names[other_offset:]
    rad21_other_idx = other_track_names.index('rad21')
    rad21_tensor_idx = input_track_names.index('rad21')   # position among non-seq channels
    E_rad21 = other_regions_wt[rad21_other_idx]           # experimental WT rad21 (log1p)
    track_len = len(E_rad21)

    a_ctcf, a_atac, a_other = set_a
    b_ctcf, b_atac, b_other = set_b

    def _pred(seq, ctcf_a, atac_a, other_a):
        inp = preprocess_default(seq, ctcf_a, atac_a, other_a)
        pred = predict_rad21(hierarchical_rad21_model, inp, rad21_tensor_idx)  # linear, model res
        pred = np.atleast_1d(np.asarray(pred, dtype=np.float64))
        if len(pred) != track_len:
            pred = np.interp(np.linspace(0, 1, track_len),
                             np.linspace(0, 1, len(pred)), pred)
        return pred

    rad21_pred_a = _pred(seq_a, a_ctcf, a_atac, a_other)
    rad21_pred_b = _pred(seq_b, b_ctcf, b_atac, b_other)
    out_a, out_b = redistribute_by_allele_ratio(
        E_rad21, rad21_pred_a, rad21_pred_b, cap=cap, track_is_log1p=bigwig_log_transform)
    a_other[rad21_other_idx] = out_a
    b_other[rad21_other_idx] = out_b
    print('[allele-split] Redistributed RAD21 (hierarchical) into both alleles')
    return (a_ctcf, a_atac, a_other), (b_ctcf, b_atac, b_other)


def _snapshot_allele_tmp(label, *, input_track_names):
    """Preserve THIS allele's shared tmp/ artifacts in tmp/<label>/ so the other
    allele's run doesn't overwrite them (ref/alt each keep their own cool/ini/bed/bigwig).

    The copied .ini files have their relative references (`tmp/...` and the un-prefixed
    `arcs_*.bed`) rewritten to ABSOLUTE paths inside tmp/<label>/, so each allele's .ini
    stays self-contained and re-renderable. Purely additive: the shared tmp/ and the
    per-allele PNGs are left untouched.
    """
    dst = os.path.join('tmp', label)
    os.makedirs(dst, exist_ok=True)
    abs_dst = os.path.abspath(dst)

    names = ['tmp.cool', 'tmp.cool.csv', 'tmp_before.cool', 'tmp_before.cool.csv',
             'tmp_diff.cool', 'tmp_diff.cool.csv', 'tmp_true.cool', 'tmp_true.cool.csv',
             'regions.bed', 'arcs.bed', 'arcs_ko.bed', 'arcs_diff.bed', 'arcs_true.bed',
             'ctcf_motif.bed', 'ctcf_motifs_detected.bed',
             'tmp_tracks.ini', 'tmp_tracks_pred.ini', 'tmp_tracks_diff.ini', 'tmp_tracks_true.ini']
    names += [f'{t}_enformer_ko.bw' for t in input_track_names]
    for n in names:
        src = os.path.join('tmp', n)
        if os.path.isfile(src):
            shutil.copy2(src, os.path.join(dst, n))

    for ini in ('tmp_tracks.ini', 'tmp_tracks_pred.ini', 'tmp_tracks_diff.ini', 'tmp_tracks_true.ini'):
        p = os.path.join(dst, ini)
        if not os.path.isfile(p):
            continue
        txt = open(p).read()
        # motif file is renamed to ctcf_motifs_detected.bed after rendering -> point there
        txt = txt.replace('file = tmp/ctcf_motif.bed', f'file = {abs_dst}/ctcf_motifs_detected.bed')
        # 'file = tmp/X' -> absolute path inside tmp/<label>/
        txt = re.sub(r'file = tmp/(\S+)', lambda m: f'file = {abs_dst}/{m.group(1)}', txt)
        # un-prefixed arcs bed (resolved by pyGenomeTracks relative to cwd) -> absolute
        txt = re.sub(r'file = (arcs\S*\.bed)', lambda m: f'file = {abs_dst}/{m.group(1)}', txt)
        with open(p, 'w') as f:
            f.write(txt)
    print(f'[allele-split] Saved {label} tmp artifacts -> {dst}/ (cool/ini/bed/bigwig, ini paths absolutized)')


def _predict_and_write_allele(cfg, label, *, model, seq_region, ctcf_a, atac_a, other_a,
                              input_track_names, input_track_paths, pred_before, pred_before_1d,
                              enformer_perturbed_track_names, extra_perturbed=frozenset(),
                              fig_kind='snp', perturb_label='SNP perturb',
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
        plot_track_names=plot_track_names, plot_track_paths=plot_track_paths, start=start,
        perturb_label=perturb_label)

    run_pygenometracks(
        region=region, celltype=celltype, chr_name=chr_name, deletion_starts=deletion_starts,
        deletion_widths=deletion_widths, font_size=font_size, no_plots=no_plots,
        outname=allele_outname, output_path=output_path, plot_diff=plot_diff,
        plot_ground_truth=plot_ground_truth, plot_width=plot_width, silent=silent, start=start,
        track_label_fraction=track_label_fraction, window=window, fig_kind=fig_kind)
    print(f'[allele-split] Wrote {label} allele outputs (prefix "{allele_outname}").')

    # Preserve this allele's tmp/ artifacts before the next allele overwrites them.
    _snapshot_allele_tmp(label, input_track_names=input_track_names)
    return pred


def _write_allele_perturb_bigwigs(label, ctcf_a, atac_a, other_a, *, input_track_names,
                                  input_track_paths, perturbed_tracks, chr_name, start, window,
                                  bigwig_log_transform):
    """Write tmp/<track>_<label>_snp_perturb.bw for each perturbed track of ONE allele,
    so both alleles' tracks are simultaneously available for the comparison .ini files."""
    other_offset = sum(1 for t in ['ctcf', 'atac'] if t in input_track_names)
    other_names = input_track_names[other_offset:]
    for tname in input_track_names:
        if tname not in perturbed_tracks:
            continue
        if tname == 'ctcf':
            arr = ctcf_a
        elif tname == 'atac':
            arr = atac_a
        elif other_a is not None and tname in other_names:
            arr = other_a[other_names.index(tname)]
        else:
            arr = None
        if arr is None:
            continue
        vals = np.expm1(arr) if bigwig_log_transform else arr
        tpath = input_track_paths[input_track_names.index(tname)]
        write_tmp_pred_bigwig(tpath, vals, tname, chr_name, start,
                              suffix=f'{label}_snp_perturb', window=window)


def _write_compare_ini(ini_path, *, order, heatmap_cool, heatmap_title, perturbed_in_order,
                       chr_name, start, ctcf_path, celltype, assembly, plot_bigwig_q,
                       min_val_diff, max_val_diff, perturb_label='SNP perturb', max_probe_allele='ref'):
    """Build one cross-allele comparison .ini: for each perturbed track, two interleaved
    rows (the two alleles in `order`), then an allele-difference Hi-C heatmap. Rows are
    coloured by position (1st red, 2nd blue); the two alleles of a track share a max so
    they are visually comparable."""
    if '/mm10/' in ctcf_path:
        assembly = 'mm10'
    elif '/hg38/' in ctcf_path:
        assembly = 'hg38'
    data_root = ctcf_path[:ctcf_path.index(f'/{assembly}/')]
    lines = [ln + '\n' for ln in get_tracks(data_root, celltype, assembly).split('\n')]
    # Colour by ROW POSITION within each pair: first row red, second row blue.
    ROW_COLORS = ('#FF0000', '#454FA5')
    with open(ini_path, 'w') as f:
        for line in lines:
            if 'arcs.bed' in line:
                line = line.replace('arcs.bed', 'arcs_ko.bed')
            if '[Genes]' in line:
                for tname in perturbed_in_order:
                    # Probe a FIXED allele for the shared y-max so both rows of a pair AND
                    # both comparison files use the same scale (matches function #1).
                    probe_bw = f'tmp/{tname}_{max_probe_allele}_snp_perturb.bw'
                    tmax = (get_axis_range_from_bigwig(probe_bw, chr_name, start, q=plot_bigwig_q)
                            if os.path.exists(probe_bw) else None)
                    for ci, allele in enumerate(order):
                        bwp = f'tmp/{tname}_{allele}_snp_perturb.bw'
                        if not os.path.exists(bwp):
                            continue
                        f.write(f'[{allele} {tname} {perturb_label}]\n')
                        f.write(f'file = {bwp}\n')
                        f.write('height = 2\n')
                        f.write(f'color = {ROW_COLORS[ci]}\n')
                        f.write(f'title = {allele} {tname} {perturb_label}\n')
                        f.write('min_value = 0\n')
                        if tmax is not None:
                            f.write(f'max_value = {tmax}\n')
                        f.write('number_of_bins = 512\n\n')
                f.write(f'[{heatmap_title}]\n')
                f.write(f'file = {heatmap_cool}\n')
                f.write(f'min_value = {min_val_diff}\n')
                f.write(f'max_value = {max_val_diff}\n')
                f.write('colormap = bwr\n')
                f.write('file_type = hic_matrix_square\n\n')
            f.write(line)
        f.write('\n')
        f.write('[deletion]')
        f.write('# bed file with regions to highlight\n')
        f.write('file = tmp/regions.bed\n')
        f.write('alpha = 0.25\n')
        f.write('type = vhighlight\n')


def _run_two_allele_outputs(cfg, *, model, alleles, labels, fig_kind, perturb_label,
                            input_track_names, input_track_paths, pred_before, pred_before_1d,
                            plot_track_names, plot_track_paths, perturbed_tracks,
                            deletion_starts, deletion_widths, res, image_scale, window):
    """Shared tail for BOTH allele paths: predict + write each allele, then the two
    cross-allele comparison figures (interleaved rows + allele-difference heatmap).

    ``alleles`` = [(label, seq, (ctcf, atac, other_regions)), ...] (two entries).
    ``labels`` = (a, b). ``perturbed_tracks`` = set of track names shown as "<t> <perturb_label>".
    File 1: rows (a, b) + heatmap (b - a); File 2: rows (b, a) + heatmap (a - b).
    """
    preds, track_sets = {}, {}
    for label, seq, track_set in alleles:
        track_sets[label] = track_set
        ctcf_a, atac_a, other_a = track_set
        preds[label] = _predict_and_write_allele(
            cfg, label, model=model, seq_region=seq, ctcf_a=ctcf_a, atac_a=atac_a, other_a=other_a,
            input_track_names=input_track_names, input_track_paths=input_track_paths,
            pred_before=pred_before, pred_before_1d=pred_before_1d,
            enformer_perturbed_track_names=perturbed_tracks, extra_perturbed=frozenset(),
            fig_kind=fig_kind, perturb_label=perturb_label,
            plot_track_names=plot_track_names, plot_track_paths=plot_track_paths,
            deletion_starts=deletion_starts, deletion_widths=deletion_widths,
            res=res, image_scale=image_scale, window=window)

    a, b = labels
    if cfg.no_plots or preds.get(a) is None or preds.get(b) is None:
        print('[allele-split] Done: both allele predictions written.')
        return preds

    # Both alleles' perturbed bigwigs (distinct names so they coexist for the comparison).
    for label in (a, b):
        ctcf_a, atac_a, other_a = track_sets[label]
        _write_allele_perturb_bigwigs(
            label, ctcf_a, atac_a, other_a, input_track_names=input_track_names,
            input_track_paths=input_track_paths, perturbed_tracks=perturbed_tracks,
            chr_name=cfg.chr_name, start=cfg.start, window=window,
            bigwig_log_transform=cfg.bigwig_log_transform)

    # Allele-difference contact maps: (b - a) and (a - b).
    cool_ba = f'tmp/tmp_{b}_minus_{a}.cool'
    cool_ab = f'tmp/tmp_{a}_minus_{b}.cool'
    write_tmp_cooler(preds[b] - preds[a], cfg.chr_name, cfg.start, out_file=cool_ba, res=res)
    write_tmp_cooler(preds[a] - preds[b], cfg.chr_name, cfg.start, out_file=cool_ab, res=res)

    perturbed_in_order = [t for t in input_track_names if t in perturbed_tracks]
    region = cfg.region if cfg.region is not None else f"{cfg.chr_name}:{cfg.start}-{cfg.start + window}"
    base = cfg.outname
    if base and not base.endswith('_'):
        base += '_'

    for order, cool, title, stem in (
        ((a, b), cool_ba, f'{b} - {a}', f'{b}_minus_{a}'),
        ((b, a), cool_ab, f'{a} - {b}', f'{a}_minus_{b}'),
    ):
        ini = f'tmp/tmp_tracks_compare_{stem}.ini'
        _write_compare_ini(
            ini, order=order, heatmap_cool=cool, heatmap_title=title,
            perturbed_in_order=perturbed_in_order, chr_name=cfg.chr_name, start=cfg.start,
            ctcf_path=cfg.ctcf_path, celltype=cfg.celltype, assembly=cfg.assembly,
            plot_bigwig_q=cfg.plot_bigwig_q, min_val_diff=cfg.min_val_diff,
            max_val_diff=cfg.max_val_diff, perturb_label=perturb_label, max_probe_allele=a)
        tag = f'{fig_kind}_compare_{stem}'
        out_png = os.path.join(cfg.output_path, f'{base}{cfg.celltype}_{cfg.chr_name}_{cfg.start}_{tag}.png')
        cmd = (f"pyGenomeTracks --tracks {ini} -o {out_png} --region {region} "
               f"--fontSize {font_size} --plotWidth {plot_width} --trackLabelFraction {track_label_fraction}")
        if cfg.silent:
            cmd += ' > /dev/null 2>&1'
        os.system(cmd)
    print(f'[allele-split] Wrote 2 cross-allele comparison .ini + figures ({b}-{a}, {a}-{b}).')
    print('[allele-split] Done: both allele predictions written.')
    return preds


def run_allele_peak_split(cfg, *, model, seq_region, seq_region_wt,
                          ctcf_region, atac_region, other_regions,
                          other_regions_wt, input_track_names, input_track_paths,
                          pred_before, pred_before_1d, plot_track_names, plot_track_paths,
                          hierarchical_rad21_model, deletion_starts, deletion_widths,
                          res, image_scale, window, alphagenome_seq_active=False):
    """Function #1: allele-specific peak redistribution from an IN-ENGINE sequence-model
    run (ref = WT seq, alt = SNP-mutated seq). Triggered by --allele-peak-split with an
    enformer_seq OR alphagenome_seq perturbation. The backbone that predicts the ref/alt
    ratio is chosen here (AlphaGenome vs Enformer); everything downstream (RAD21
    hierarchical split, per-allele prediction, outputs) is backbone-agnostic and shared."""
    cap = cfg.enformer_delta_cap  # output-value CAP (reference script uses 10)

    # Backbone dispatch: alphagenome_seq -> AlphaGenome, else Enformer. Both return the
    # same (ref_set, alt_set) via redistribute_enformer_alleles on their wt_pred/alt_pred.
    if alphagenome_seq_active:
        from cshark.perturb.models.alphagenome import apply_alphagenome_peak_split
        ref_set, alt_set, enformer_perturbed_track_names, _ = apply_alphagenome_peak_split(
            assembly=cfg.assembly, atac_region=atac_region, bigwig_log_transform=cfg.bigwig_log_transform,
            celltype=cfg.celltype, chr_name=cfg.chr_name, ctcf_region=ctcf_region,
            enformer_delta_cap=cfg.enformer_delta_cap, enformer_delta_mode=cfg.enformer_delta_mode,
            alphagenome_model_path=cfg.alphagenome_model_path,
            alphagenome_metadata_path=cfg.alphagenome_metadata_path,
            enformer_tracks=cfg.enformer_tracks, input_track_names=input_track_names,
            other_regions=other_regions, seq_region=seq_region, seq_region_wt=seq_region_wt,
            start=cfg.start, window=window)
    else:
        ref_set, alt_set, enformer_perturbed_track_names, _ = apply_enformer_peak_split(
            assembly=cfg.assembly, atac_region=atac_region, bigwig_log_transform=cfg.bigwig_log_transform,
            celltype=cfg.celltype, chr_name=cfg.chr_name, ctcf_region=ctcf_region,
            enformer_delta_cap=cfg.enformer_delta_cap, enformer_delta_mode=cfg.enformer_delta_mode,
            enformer_model_path=cfg.enformer_model_path, enformer_seq_active=True,
            enformer_tracks=cfg.enformer_tracks, input_track_names=input_track_names,
            other_regions=other_regions, seq_region=seq_region, seq_region_wt=seq_region_wt,
            start=cfg.start, window=window)

    # Each allele is predicted from ITS OWN sequence: ref = the unmodified --seq
    # genome, alt = the same window after the variant substitution (or, with
    # --alt-fasta, the whole alternate genome). Earlier versions fed seq_region to
    # both, which was near-harmless for a 1 bp SNP (0.013% max change in the
    # predicted Hi-C, r = 1.00000000) but wrong for --alt-fasta, where the two
    # sequences differ by ~1,941 bases per 2 Mb window.
    ref_set, alt_set = _redistribute_rad21_alleles(
        ref_set, alt_set, seq_a=seq_region_wt, seq_b=seq_region, other_regions_wt=other_regions_wt,
        input_track_names=input_track_names, hierarchical_rad21_model=hierarchical_rad21_model,
        cap=cap, bigwig_log_transform=cfg.bigwig_log_transform)

    extra = ({'rad21'} if (hierarchical_rad21_model is not None and 'rad21' in input_track_names) else set())
    perturbed_tracks = set(enformer_perturbed_track_names) | extra

    _run_two_allele_outputs(
        cfg, model=model,
        alleles=[('ref', seq_region_wt, ref_set), ('alt', seq_region, alt_set)],
        labels=('ref', 'alt'), fig_kind='snp', perturb_label='SNP perturb',
        input_track_names=input_track_names, input_track_paths=input_track_paths,
        pred_before=pred_before, pred_before_1d=pred_before_1d,
        plot_track_names=plot_track_names, plot_track_paths=plot_track_paths,
        perturbed_tracks=perturbed_tracks, deletion_starts=deletion_starts,
        deletion_widths=deletion_widths, res=res, image_scale=image_scale, window=window)


def run_allele_haplotype(cfg):
    """Function #2: haplotype peak redistribution from PROVIDED maternal/paternal prediction
    bigwigs (no in-engine Enformer). Single-locus; each allele uses its own haplotype sequence."""
    import cshark.inference.utils.inference_utils as infer
    from cshark.inference.utils.hierarchical_utils import load_hierarchical_rad21_predictor
    from cshark.perturb.config import WINDOW
    from cshark.perturb.models.base import CSharkModel
    from cshark.perturb.models.hierarchical import prepare_rad21_input
    from cshark.perturb.models.enformer import redistribute_from_provided_preds

    if cfg.start is None:
        raise SystemExit('[allele-haplotype] requires --start (single-locus mode).')
    if not cfg.maternal_seq_path or not cfg.paternal_seq_path:
        raise SystemExit('[allele-haplotype] requires --maternal-seq and --paternal-seq.')

    window, res, image_scale = WINDOW, cfg.resolution, cfg.mat_size
    chr_name, start, cap = cfg.chr_name, cfg.start, cfg.enformer_delta_cap
    os.makedirs(cfg.output_path, exist_ok=True)
    os.makedirs('tmp', exist_ok=True)

    # WT region: reference seq (for the WT baseline) + WT bulk tracks E (for redistribution).
    seq_ref, ctcf_region, atac_region, other_regions = infer.load_region(
        chr_name, start, cfg.seq_path, cfg.ctcf_path, cfg.atac_path, cfg.other_feats,
        seq2_path=None, window=window, bigwig_log=cfg.bigwig_log_transform)

    input_track_names, input_track_paths = [], []
    if cfg.ctcf_path is not None:
        input_track_names.append('ctcf'); input_track_paths.append(cfg.ctcf_path)
    if cfg.atac_path is not None:
        input_track_names.append('atac'); input_track_paths.append(cfg.atac_path)
    if cfg.other_feats is not None:
        for feat in cfg.other_feats:
            input_track_names.append(os.path.basename(feat).split('.')[0]); input_track_paths.append(feat)

    hierarchical_rad21_model = None
    if cfg.hierarchical_model_path is not None:
        hierarchical_rad21_model, _, _, _ = load_hierarchical_rad21_predictor(cfg.hierarchical_model_path)
    other_regions = prepare_rad21_input(
        atac_region=atac_region, ctcf_region=ctcf_region, hierarchical_rad21_model=hierarchical_rad21_model,
        input_track_names=input_track_names, input_track_paths=input_track_paths,
        model_path=cfg.model_path, other_regions=other_regions, seq_region=seq_ref)

    num_genomic_features = 2 if other_regions is None else 2 + len(other_regions)
    if atac_region is None:
        num_genomic_features -= 1
    if ctcf_region is None:
        num_genomic_features -= 1
    model = CSharkModel(cfg, num_genomic_features=num_genomic_features, diploid=False)

    pred_before_output = model.predict_arrays(seq_ref, ctcf_region, atac_region, other_regions, input_track_names[2:])
    pred_before, pred_before_1d = pred_before_output['hic'], pred_before_output['1d']

    plot_track_names, plot_track_paths = [], []
    for pt in (cfg.plot_bigwigs or []):
        if pt not in input_track_names:
            plot_track_names.append(pt); plot_track_paths.append(input_track_paths[0].replace(input_track_names[0], pt))
    for pt in (cfg.plot_pred_bigwigs or []):
        if pt not in plot_track_names:
            plot_track_names.append(pt); plot_track_paths.append(input_track_paths[0].replace(input_track_names[0], pt))

    # Each allele's HAPLOID haplotype sequence (only the seq array is used; tracks come from E).
    mat_seq = infer.load_region(chr_name, start, cfg.maternal_seq_path, cfg.ctcf_path, cfg.atac_path,
                                cfg.other_feats, seq2_path=None, window=window, bigwig_log=cfg.bigwig_log_transform)[0]
    pat_seq = infer.load_region(chr_name, start, cfg.paternal_seq_path, cfg.ctcf_path, cfg.atac_path,
                                cfg.other_feats, seq2_path=None, window=window, bigwig_log=cfg.bigwig_log_transform)[0]

    # Maternal/paternal track sets from the provided prediction bigwigs (WT bulk where no pred).
    mat_set, pat_set, redistributed = redistribute_from_provided_preds(
        ctcf_region, atac_region, other_regions, input_track_names,
        cfg.maternal_pred, cfg.paternal_pred, chr_name, start, window,
        cap=cap, track_is_log1p=cfg.bigwig_log_transform)

    # Per-allele RAD21 redistribution (each allele's own sequence).
    mat_set, pat_set = _redistribute_rad21_alleles(
        mat_set, pat_set, seq_a=mat_seq, seq_b=pat_seq, other_regions_wt=other_regions,
        input_track_names=input_track_names, hierarchical_rad21_model=hierarchical_rad21_model,
        cap=cap, bigwig_log_transform=cfg.bigwig_log_transform)

    extra = ({'rad21'} if (hierarchical_rad21_model is not None and 'rad21' in input_track_names) else set())
    perturbed_tracks = set(redistributed) | extra

    _run_two_allele_outputs(
        cfg, model=model,
        alleles=[('maternal', mat_seq, mat_set), ('paternal', pat_seq, pat_set)],
        labels=('maternal', 'paternal'), fig_kind='hap', perturb_label='perturb',
        input_track_names=input_track_names, input_track_paths=input_track_paths,
        pred_before=pred_before, pred_before_1d=pred_before_1d,
        plot_track_names=plot_track_names, plot_track_paths=plot_track_paths,
        perturbed_tracks=perturbed_tracks, deletion_starts=[], deletion_widths=[],
        res=res, image_scale=image_scale, window=window)
    print('[allele-haplotype] Done.')
