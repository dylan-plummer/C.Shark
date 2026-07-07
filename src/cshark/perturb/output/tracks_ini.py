"""pyGenomeTracks .ini builder for single-locus perturbation output.

``build_track_inis`` is a verbatim extraction of the 4-ini-file construction from
the original ``single_deletion`` (perturb.py ~1230-1597 / single_locus.py 598-965):
it writes tmp/tmp_tracks.ini, tmp_tracks_true.ini, tmp_tracks_pred.ini and
tmp_tracks_diff.ini. Body is byte-identical to the original; only wrapped in a
function whose parameters are the locals it reads.

TrackSpec is retained for a future spec-driven rewrite (Step 6); the current
builder still emits ini text inline as the original did.
"""
import os
import pandas as pd
from importlib.resources import files

from cshark.inference.utils.inference_utils import get_axis_range_from_bigwig
from cshark.inference.tracks_files import get_tracks
from dataclasses import dataclass
from typing import Optional


@dataclass
class TrackSpec:
    """A single track to render in a pyGenomeTracks view (reserved for Step 6)."""
    name: str
    path: str
    title: str = ''
    color: str = '#666666'
    min_value: Optional[float] = None
    max_value: Optional[float] = None
    kind: str = 'bigwig'


def build_track_inis(assembly, celltype, chr_name, ctcf_motif_p, ctcf_path, enformer_perturbed_track_names, enformer_seq_active, hierarchical_active, input_track_names, input_track_paths, ko_data, max_val_diff, max_val_pred, max_val_true, min_val_diff, min_val_pred, min_val_true, plot_bigwig_q, plot_diff, plot_ground_truth, plot_pred_bigwigs, plot_pred_log2fc, plot_track_names, plot_track_paths, start, perturb_label='SNP perturb', ko_tool='enformer'):
    if '/mm10/' in ctcf_path:
        assembly = 'mm10'
    elif '/hg38/' in ctcf_path:
        assembly = 'hg38'
    assembly_idx = ctcf_path.index(f'/{assembly}/')
    data_root = ctcf_path[:assembly_idx]
    tracks = get_tracks(data_root, celltype, assembly)
    lines = tracks.split('\n')
    lines = [line + '\n' for line in lines]
    colors = ['red', 'blue', 'green', 'orange', 'purple', 'pink', 'brown', 'gray', 'cyan', 'magenta',
              'yellow', 'red', 'blue', 'green', 'orange', 'purple', 'pink']
    ko_track_names = {track_name.lower() for track_name in ko_data}

    def resolve_ko_display_track(track_name, track_path):
        canonical = track_name.lower()
        display_path = track_path
        if hierarchical_active and canonical == 'rad21':
            # Don't redirect to perturbed.bw here — the hierarchical track blocks
            # below write WT pred / KO pred / Perturbed (model input) explicitly.
            # Keeping display_path == track_path suppresses the confusing "rad21 KO"
            # entry that would otherwise duplicate the perturbed track.
            pass
        elif canonical in ko_track_names and os.path.exists(f'tmp/{track_name}_ko.bw'):
            display_path = f'tmp/{track_name}_ko.bw'
        return canonical, display_path

    with open('tmp/tmp_tracks.ini', 'w') as f:
        for line in lines:
            if 'arcs.bed' in line:
                line = line.replace('arcs.bed', 'arcs_ko.bed')

            if '[Genes]' in line:
                for track_i, (track_name, track_path) in enumerate(
                        zip(input_track_names + plot_track_names, input_track_paths + plot_track_paths)):
                    track_name = os.path.basename(track_path).split('.')[0]
                    canonical_track_name, display_track_path = resolve_ko_display_track(track_name, track_path)
                    track_max = None
                    is_model_input = track_name in input_track_names
                    final_track_path = display_track_path if os.path.exists(display_track_path) else None
                    wt_track_path = track_path if os.path.exists(track_path) else None
                    if is_model_input and wt_track_path is not None:
                        f.write(f'[{track_name} WT]\n')
                        f.write(f'file = {wt_track_path}\n')
                        f.write('height = 2\n')
                        f.write(f'color = {colors[track_i]}\n')
                        f.write(f'title = {track_name} WT\n')
                        f.write('min_value = 0\n')
                        wt_track_max = get_axis_range_from_bigwig(wt_track_path, chr_name, start, q=plot_bigwig_q)
                        if wt_track_max is not None:
                            f.write(f'max_value = {wt_track_max}\n')
                        f.write('number_of_bins = 512\n\n')
                        track_max = wt_track_max
                    if final_track_path is not None and (not is_model_input or final_track_path != wt_track_path):
                        final_title = f'{track_name} KO' if is_model_input else track_name
                        f.write(f'[{track_name}]\n')
                        f.write(f'file = {final_track_path}\n')
                        f.write('height = 2\n')
                        f.write(f'color = {colors[track_i]}\n')
                        f.write(f'title = {final_title}\n')
                        f.write('min_value = 0\n')
                        final_track_max = get_axis_range_from_bigwig(final_track_path, chr_name, start, q=plot_bigwig_q)
                        if final_track_max is not None:
                            f.write(f'max_value = {final_track_max}\n')
                        f.write('number_of_bins = 512\n\n')
                        track_max = final_track_max
                    elif track_max is None and final_track_path is not None:
                        track_max = get_axis_range_from_bigwig(final_track_path, chr_name, start, q=plot_bigwig_q)
                    if final_track_path is not None or wt_track_path is not None:
                        if canonical_track_name == 'ctcf' and ctcf_motif_p is not None:
                            f.write('[CTCF motif]\n')
                            motif_file = str(files("cshark").joinpath(f"static/ctcf_motifs_jaspar.bed"))
                            motifs = pd.read_csv(motif_file, sep='\t', names=['chrom', 'start', 'end', 'strand', 'q'])
                            motifs = motifs.loc[motifs['q'] > ctcf_motif_p].reset_index(drop=True)
                            motifs.to_csv('tmp/ctcf_motif.bed', sep='\t', header=False, index=False)
                            f.write('file = tmp/ctcf_motif.bed\n')
                            f.write('file_type = bed\n')
                            f.write('fontsize = 10\n')
                            f.write('display = interleaved\n')
                        if (enformer_seq_active and canonical_track_name in enformer_perturbed_track_names and
                                os.path.exists(f'tmp/{track_name}_{ko_tool}_ko.bw')):
                            f.write(f'[{track_name} {perturb_label}]\n')
                            f.write(f'file = tmp/{track_name}_{ko_tool}_ko.bw\n')
                            f.write('height = 2\n')
                            f.write(f'color = {colors[track_i]}\n')
                            f.write(f'title = {track_name} {perturb_label}\n')
                            f.write('min_value = 0\n')
                            if track_max is not None:
                                f.write(f'max_value = {track_max}\n')
                            f.write('number_of_bins = 512\n\n')
                        if hierarchical_active and canonical_track_name == 'rad21':
                            if os.path.exists('tmp/rad21_hierarchical_wt_pred.bw'):
                                f.write('[RAD21 Hier. WT pred]\n')
                                f.write('file = tmp/rad21_hierarchical_wt_pred.bw\n')
                                f.write('height = 2\n')
                                f.write('color = darkgreen\n')
                                f.write('title = RAD21 Hier. WT pred\n')
                                f.write('min_value = 0\n')
                                if track_max is not None:
                                    f.write(f'max_value = {track_max}\n')
                                f.write('number_of_bins = 512\n\n')
                            if os.path.exists('tmp/rad21_hierarchical_ko_pred.bw'):
                                f.write('[RAD21 Hier. KO pred]\n')
                                f.write('file = tmp/rad21_hierarchical_ko_pred.bw\n')
                                f.write('height = 2\n')
                                f.write('color = darkorange\n')
                                f.write('title = RAD21 Hier. KO pred\n')
                                f.write('min_value = 0\n')
                                if track_max is not None:
                                    f.write(f'max_value = {track_max}\n')
                                f.write('number_of_bins = 512\n\n')
                            if os.path.exists('tmp/rad21_hierarchical_perturbed.bw'):
                                f.write('[RAD21 Perturbed (model input)]\n')
                                f.write('file = tmp/rad21_hierarchical_perturbed.bw\n')
                                f.write('height = 2\n')
                                f.write('color = crimson\n')
                                f.write('title = RAD21 Perturbed (model input)\n')
                                f.write('min_value = 0\n')
                                if track_max is not None:
                                    f.write(f'max_value = {track_max}\n')
                                f.write('number_of_bins = 512\n\n')
                    if track_name in plot_pred_bigwigs:
                        pred_ko_file = f'tmp/{track_name}_pred_KO.bw'
                        if (not plot_pred_log2fc and hierarchical_active and
                                track_name.lower() == 'rad21' and
                                os.path.exists('tmp/rad21_hierarchical_ko_pred.bw')):
                            pred_ko_file = 'tmp/rad21_hierarchical_ko_pred.bw'
                        if os.path.exists(pred_ko_file):
                            f.write(f'[{track_name} pred]\n')
                            f.write(f'file = {pred_ko_file}\n')
                            f.write('height = 2\n')
                            if plot_pred_log2fc:
                                f.write(f'title = {track_name} log2FC\n')
                                f.write('color = red\n')
                                f.write('negative_color = blue\n')
                                f.write('min_value = -1\n')
                                f.write('max_value = 1\n')
                            else:
                                f.write(f'title = {track_name} pred KO\n')
                                f.write(f'color = {colors[track_i]}\n')
                                f.write('min_value = 0\n')
                                if track_max is not None:
                                    f.write(f'max_value = {track_max}\n')
                            f.write('number_of_bins = 512\n\n')

                f.write('[KO pred]\n')
                f.write('file = tmp/tmp.cool\n')
                f.write(f'min_value = {min_val_pred}\n')
                if max_val_pred is not None:
                    f.write(f'max_value = {max_val_pred}\n')
                f.write('colormap =  [ (1.0, 1.0, 1.0), (1.0, 0.92, 0.92),(1.0, 0.8, 0.8),(1.0, 0.6, 0.6), (1.0, 0.4, 0.4),(1.0, 0.294, 0.294)]\n')
                f.write('file_type = hic_matrix_square\n\n')

            f.write(line)

        f.write('\n')
        f.write('[deletion]')
        f.write('# bed file with regions to highlight\n')
        f.write('file = tmp/regions.bed\n')
        f.write('alpha = 0.25\n')
        f.write('type = vhighlight\n')

    if plot_ground_truth:
        lines = tracks.split('\n')
        lines = [line + '\n' for line in lines]
        with open('tmp/tmp_tracks_true.ini', 'w') as f:
            for line in lines:
                if 'arcs.bed' in line:
                    line = line.replace('arcs.bed', 'arcs_true.bed')
                if '[Genes]' in line:
                    for track_i, (track_name, track_path) in enumerate(
                            zip(input_track_names + plot_track_names, input_track_paths + plot_track_paths)):
                        track_name = os.path.basename(track_path).split('.')[0]
                        try:
                            track_max = get_axis_range_from_bigwig(track_path, chr_name, start, q=plot_bigwig_q)
                        except Exception as e:
                            print(f'Error getting axis range for {track_path}: {e}')
                            continue
                        f.write(f'[{track_name}]\n')
                        f.write(f'file = {track_path}\n')
                        f.write('height = 2\n')
                        f.write(f'color = {colors[track_i]}\n')
                        f.write(f'title = {track_name}\n')
                        f.write('min_value = 0\n')
                        if track_max is not None:
                            f.write(f'max_value = {track_max}\n')
                        f.write('number_of_bins = 512\n\n')
                    f.write('[deeploop]\n')
                    f.write('file = tmp/tmp_true.cool\n')
                    f.write(f'min_value = {min_val_true}\n')
                    if max_val_true is not None:
                        f.write(f'max_value = {max_val_true}\n')
                    f.write('colormap =  [ (1.0, 1.0, 1.0), (1.0, 0.92, 0.92),(1.0, 0.8, 0.8),(1.0, 0.6, 0.6), (1.0, 0.4, 0.4),(1.0, 0.294, 0.294)]\n')
                    f.write('file_type = hic_matrix_square\n\n')
                f.write(line)

    lines = tracks.split('\n')
    lines = [line + '\n' for line in lines]
    with open('tmp/tmp_tracks_pred.ini', 'w') as f:
        for line in lines:
            if '[Genes]' in line:
                for track_i, (track_name, track_path) in enumerate(
                        zip(input_track_names + plot_track_names, input_track_paths + plot_track_paths)):
                    track_name = os.path.basename(track_path).split('.')[0]
                    canonical_track_name = track_name.lower()
                    if os.path.exists(track_path):
                        f.write(f'[{track_name}]\n')
                        f.write(f'file = {track_path}\n')
                        f.write('height = 2\n')
                        f.write(f'color = {colors[track_i]}\n')
                        f.write(f'title = {track_name}\n')
                        f.write('min_value = 0\n')
                        track_max = get_axis_range_from_bigwig(track_path, chr_name, start, q=plot_bigwig_q)
                        if track_max is not None:
                            f.write(f'max_value = {track_max}\n')
                        f.write('number_of_bins = 512\n\n')
                    else:
                        track_max = None
                    if track_name in plot_pred_bigwigs:
                        pred_wt_file = f'tmp/{track_name}_pred_WT.bw'
                        if (hierarchical_active and track_name.lower() == 'rad21' and
                                os.path.exists('tmp/rad21_hierarchical_wt_pred.bw')):
                            pred_wt_file = 'tmp/rad21_hierarchical_wt_pred.bw'
                        f.write(f'[{track_name} pred]\n')
                        f.write(f'file = {pred_wt_file}\n')
                        f.write('height = 2\n')
                        f.write(f'color = {colors[track_i]}\n')
                        f.write(f'title = {track_name} pred\n')
                        f.write('min_value = 0\n')
                        if track_max is not None:
                            f.write(f'max_value = {track_max}\n')
                        f.write('number_of_bins = 512\n\n')
                    if hierarchical_active and canonical_track_name == 'rad21':
                        if os.path.exists('tmp/rad21_hierarchical_wt_pred.bw'):
                            f.write('[RAD21 Hier. WT pred]\n')
                            f.write('file = tmp/rad21_hierarchical_wt_pred.bw\n')
                            f.write('height = 2\n')
                            f.write('color = darkgreen\n')
                            f.write('title = RAD21 Hier. WT pred\n')
                            f.write('min_value = 0\n')
                            if track_max is not None:
                                f.write(f'max_value = {track_max}\n')
                            f.write('number_of_bins = 512\n\n')
                f.write('[WT pred]\n')
                f.write('file = tmp/tmp_before.cool\n')
                f.write(f'min_value = {min_val_pred}\n')
                if max_val_pred is not None:
                    f.write(f'max_value = {max_val_pred}\n')
                f.write('colormap =  [ (1.0, 1.0, 1.0), (1.0, 0.92, 0.92),(1.0, 0.8, 0.8),(1.0, 0.6, 0.6), (1.0, 0.4, 0.4),(1.0, 0.294, 0.294)]\n')
                f.write('file_type = hic_matrix_square\n\n')
            f.write(line)

    if plot_diff:
        lines = tracks.split('\n')
        lines = [line + '\n' for line in lines]
        with open('tmp/tmp_tracks_diff.ini', 'w') as f:
            for line in lines:
                if 'arcs.bed' in line:
                    line = line.replace('arcs.bed', 'arcs_diff.bed')
                    f.write(line)
                    f.write('line_width = 1\n')
                    f.write('color = bwr\n')
                    f.write('alpha = 0.5\n')
                    f.write('height = 3\n')
                    f.write('file_type = links\n')
                    f.write('links_type = arcs\n')
                    f.write('orientation = inverted\n')
                    break

                if '[Genes]' in line:
                    for track_i, (track_name, track_path) in enumerate(
                            zip(input_track_names + plot_track_names, input_track_paths + plot_track_paths)):
                        track_name = os.path.basename(track_path).split('.')[0]
                        canonical_track_name = track_name.lower()
                        track_max = None
                        if os.path.exists(track_path):
                            f.write(f'[{track_name}]\n')
                            f.write(f'file = {track_path}\n')
                            f.write('height = 2\n')
                            f.write(f'color = {colors[track_i]}\n')
                            f.write(f'title = {track_name}\n')
                            f.write('min_value = 0\n')
                            track_max = get_axis_range_from_bigwig(track_path, chr_name, start, q=plot_bigwig_q)
                            if track_max is not None:
                                f.write(f'max_value = {track_max}\n')
                            f.write('number_of_bins = 512\n\n')
                            if track_name in ko_data:
                                f.write(f'[{track_name} KO]\n')
                                f.write(f'file = tmp/{track_name}_ko.bw\n')
                                f.write('height = 2\n')
                                f.write(f'color = {colors[track_i]}\n')
                                f.write(f'title = {track_name} KO\n')
                                f.write('min_value = 0\n')
                                if track_max is not None:
                                    f.write(f'max_value = {track_max}\n')
                                f.write('number_of_bins = 512\n\n')
                            if (enformer_seq_active and canonical_track_name in enformer_perturbed_track_names and
                                    os.path.exists(f'tmp/{track_name}_{ko_tool}_delta.bw')):
                                enformer_ko_file = f'tmp/{track_name}_{ko_tool}_ko.bw'
                                if os.path.exists(enformer_ko_file):
                                    f.write(f'[{track_name} {perturb_label}]\n')
                                    f.write(f'file = {enformer_ko_file}\n')
                                    f.write('height = 2\n')
                                    f.write(f'color = {colors[track_i]}\n')
                                    f.write(f'title = {track_name} {perturb_label}\n')
                                    f.write('min_value = 0\n')
                                    if track_max is not None:
                                        f.write(f'max_value = {track_max}\n')
                                    f.write('number_of_bins = 512\n\n')
                                f.write(f'[{track_name} SNP delta]\n')
                                f.write(f'file = tmp/{track_name}_{ko_tool}_delta.bw\n')
                                f.write('height = 2\n')
                                f.write('color = red\n')
                                f.write('negative_color = blue\n')
                                f.write(f'title = {track_name} SNP delta\n')
                                f.write('min_value = -0.5\n')
                                f.write('max_value = 0.5\n')
                                f.write('number_of_bins = 512\n\n')
                            if hierarchical_active and canonical_track_name == 'rad21':
                                if os.path.exists('tmp/rad21_hierarchical_delta.bw'):
                                    f.write('[RAD21 Hier. Delta]\n')
                                    f.write('file = tmp/rad21_hierarchical_delta.bw\n')
                                    f.write('height = 2\n')
                                    f.write('color = red\n')
                                    f.write('negative_color = blue\n')
                                    f.write('title = RAD21 Hier. Delta\n')
                                    f.write('min_value = -0.5\n')
                                    f.write('max_value = 0.5\n')
                                    f.write('number_of_bins = 512\n\n')
                                if os.path.exists('tmp/rad21_hierarchical_perturbed.bw'):
                                    f.write('[RAD21 Perturbed (model input)]\n')
                                    f.write('file = tmp/rad21_hierarchical_perturbed.bw\n')
                                    f.write('height = 2\n')
                                    f.write('color = crimson\n')
                                    f.write('title = RAD21 Perturbed (model input)\n')
                                    f.write('min_value = 0\n')
                                    if track_max is not None:
                                        f.write(f'max_value = {track_max}\n')
                                    f.write('number_of_bins = 512\n\n')
                        if track_name in plot_pred_bigwigs:
                            pred_diff_file = f'tmp/{track_name}_pred_diff.bw'
                            if (plot_pred_log2fc and os.path.exists(f'tmp/{track_name}_pred_KO.bw')):
                                pred_diff_file = f'tmp/{track_name}_pred_KO.bw'
                            if (not plot_pred_log2fc and hierarchical_active and
                                    track_name.lower() == 'rad21' and
                                    os.path.exists('tmp/rad21_hierarchical_delta.bw')):
                                pred_diff_file = 'tmp/rad21_hierarchical_delta.bw'
                            if os.path.exists(pred_diff_file):
                                f.write(f'[{track_name} pred diff]\n')
                                f.write(f'file = {pred_diff_file}\n')
                                f.write('height = 2\n')
                                f.write('color = red\n')
                                f.write('negative_color = blue\n')
                                if plot_pred_log2fc:
                                    f.write(f'title = {track_name} pred log2FC\n')
                                else:
                                    f.write(f'title = {track_name} pred delta\n')
                                f.write('min_value = -1\n')
                                f.write('max_value = 1\n')
                                f.write('number_of_bins = 512\n\n')

                    f.write('[Diff]\n')
                    f.write('file = tmp/tmp_diff.cool\n')
                    f.write(f'min_value = {min_val_diff}\n')
                    if max_val_diff is not None:
                        f.write(f'max_value = {max_val_diff}\n')
                    f.write('colormap = bwr\n')
                    f.write('file_type = hic_matrix_square\n\n')
                f.write(line)


def run_pygenometracks(region, celltype, chr_name, deletion_starts, deletion_widths, font_size, no_plots, outname, output_path, plot_diff, plot_ground_truth, plot_width, silent, start, track_label_fraction, window, fig_kind=None):
    """Render the .ini files to PNGs via pyGenomeTracks (extracted verbatim).

    ``fig_kind`` selects the output filename tags by perturbation type:
    ``None`` -> legacy ``ctcf_ko_*`` (KO / deletion runs);
    ``'snp'`` -> ``snp_perturb_*`` (enformer_seq / allele-peak-split);
    ``'hap'`` -> ``hap_perturb_*`` (haplotype redistribution). Default None preserves
    the legacy names.
    """
    if not no_plots:
        try:
            region = region if region is not None else f"{chr_name}:{start}-{start + window}"

            if fig_kind == 'snp':
                tag_main, tag_pred, tag_true, tag_diff = (
                    'snp_perturb_tracks', 'snp_pred_tracks', 'snp_true_tracks', 'snp_perturb_tracks_diff')
            elif fig_kind == 'hap':
                tag_main, tag_pred, tag_true, tag_diff = (
                    'hap_perturb_tracks', 'hap_pred_tracks', 'hap_true_tracks', 'hap_perturb_tracks_diff')
            else:
                tag_main, tag_pred, tag_true, tag_diff = (
                    'ctcf_ko_tracks', 'ctcf_pred_tracks', 'ctcf_true_tracks', 'ctcf_ko_tracks_diff')

            if plot_diff:
                tracks_cmd = f"pyGenomeTracks --tracks tmp/tmp_tracks_diff.ini -o {os.path.join(output_path, f'{outname}{celltype}_{chr_name}_{start}_{tag_diff}.png')} --region {region} --fontSize {font_size} --plotWidth {plot_width} --trackLabelFraction {track_label_fraction}"
                if silent:
                    tracks_cmd += ' > /dev/null 2>&1'
                os.system(tracks_cmd)
            if plot_ground_truth:
                tracks_cmd = f"pyGenomeTracks --tracks tmp/tmp_tracks_true.ini -o {os.path.join(output_path, f'{outname}{celltype}_{chr_name}_{start}_{tag_true}.png')} --region {region} --fontSize {font_size} --plotWidth {plot_width} --trackLabelFraction {track_label_fraction}"
                if silent:
                    tracks_cmd += ' > /dev/null 2>&1'
                os.system(tracks_cmd)
            tracks_cmd = f"pyGenomeTracks --tracks tmp/tmp_tracks_pred.ini -o {os.path.join(output_path, f'{outname}{celltype}_{chr_name}_{start}_{tag_pred}.png')} --region {region} --fontSize {font_size} --plotWidth {plot_width} --trackLabelFraction {track_label_fraction}"
            if silent:
                tracks_cmd += ' > /dev/null 2>&1'
            os.system(tracks_cmd)
            if deletion_starts is not None and deletion_widths is not None:
                tracks_cmd = f"pyGenomeTracks --tracks tmp/tmp_tracks.ini -o {os.path.join(output_path, f'{outname}{celltype}_{chr_name}_{start}_{tag_main}.png')} --region {region} --fontSize {font_size} --plotWidth {plot_width} --trackLabelFraction {track_label_fraction}"
                if silent:
                    tracks_cmd += ' > /dev/null 2>&1'
                os.system(tracks_cmd)
        except Exception as e:
            print(e)

        try:
            os.rename('tmp/ctcf_motif.bed', 'tmp/ctcf_motifs_detected.bed')
        except Exception:
            pass
