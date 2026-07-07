"""Plotting helpers for perturbation outputs.

``plot_prediction_matrix`` is the de-duplicated form of the WT/KO imshow blocks
(perturb.py lines 779-783 and 1060-1064). The WT-vs-KO scatter (full-chrom),
1D-track log2FC plotting, and ``visualize_force_directed_structure`` (the
function copy-pasted across the hierarchical_predict* siblings) will land here
as those paths are ported.
"""
import matplotlib
matplotlib.use('Agg')           # headless: write files without a display
import matplotlib.pyplot as plt
import os
import numpy as np
from cshark.inference.utils import model_utils
from cshark.inference.utils.inference_utils import write_tmp_pred_bigwig


def plot_prediction_matrix(matrix, out_path, title, cmap='Reds', dpi=300):
    """Save a single contact matrix heatmap (faithful to the original imshow)."""
    plt.imshow(matrix, cmap=cmap)
    plt.colorbar()
    plt.title(title)
    plt.savefig(out_path, dpi=dpi)
    plt.close()


def plot_wt_vs_ko_scatter(*args, **kwargs):
    raise NotImplementedError("Port WT-vs-KO scatter from full-chrom branch (perturb.py ~589).")


def plot_pred_1d_log2fc(*args, **kwargs):
    raise NotImplementedError("Port 1D-track log2FC plotting (perturb.py ~1087-1127).")


def plot_pred_1d_tracks(celltype, chr_name, input_track_names, input_track_paths, model_path, no_plots, outname, output_path, plot_pred_bigwigs, plot_pred_log2fc, plot_track_names, pred_1d, pred_before_1d, start):
    """Per-track WT/KO/log2FC plots + predicted bigwigs (extracted verbatim)."""
    track_names = model_utils.get_1d_track_names(model_path)
    if track_names is None:
        track_names = []
    for track_idx, track_name in enumerate(track_names):
        try:
            ctcf_pred_before = pred_before_1d[:, track_idx]
            ctcf_pred = pred_1d[:, track_idx]
        except (IndexError, TypeError):
            break
        if track_name not in plot_track_names and track_name not in input_track_names:
            continue
        ctcf_log2fc = np.log2((ctcf_pred + 1e-5) / (ctcf_pred_before + 1e-5))
        log2fc_norm = ctcf_log2fc * ctcf_pred_before
        ymax = max(np.max(ctcf_pred_before), np.max(ctcf_pred))

        if not no_plots and track_name in plot_track_names:
            fig, axs = plt.subplots(3, 1, figsize=(10, 5))
            axs[0].plot(ctcf_pred_before, label='Before', color='blue')
            axs[0].fill_between(range(len(ctcf_pred_before)), ctcf_pred_before, 0, color='blue', alpha=0.2)
            axs[0].set_ylim(0, ymax)
            axs[1].plot(ctcf_pred, label='After', color='orange')
            axs[1].fill_between(range(len(ctcf_pred)), ctcf_pred, 0, color='orange', alpha=0.2)
            axs[1].set_ylim(0, ymax)
            axs[2].plot(log2fc_norm, label='Log2FC', color='green')
            axs[2].fill_between(range(len(log2fc_norm)), log2fc_norm, 0, color='green', alpha=0.2)
            axs[0].set_title(f'{track_name.upper()} Before')
            axs[1].set_title(f'{track_name.upper()} After')
            axs[2].set_title(f'{track_name.upper()} Log2FC * original signal')
            plt.tight_layout()
            plt.savefig(os.path.join(output_path, f'{outname}{celltype}_{chr_name}_{start}_{track_name}_log2fc.png'), dpi=300)
            plt.close(fig)
        if track_name in plot_pred_bigwigs:
            # pred_1d values are in linear space (expm1 applied in prediction())
            write_tmp_pred_bigwig(input_track_paths[0], ctcf_pred_before, track_name, chr_name, start, suffix='pred_WT')
            write_tmp_pred_bigwig(input_track_paths[0], ctcf_pred - ctcf_pred_before, track_name, chr_name, start, suffix='pred_diff')
            if plot_pred_log2fc:
                ctcf_log2fc_clipped = np.clip(ctcf_log2fc, -1, 1)
                write_tmp_pred_bigwig(input_track_paths[0], ctcf_log2fc_clipped, track_name, chr_name, start, suffix='pred_KO')
            else:
                write_tmp_pred_bigwig(input_track_paths[0], ctcf_pred, track_name, chr_name, start, suffix='pred_KO')
