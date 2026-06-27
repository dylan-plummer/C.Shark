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
