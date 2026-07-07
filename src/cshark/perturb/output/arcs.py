"""Arc / region BED writers for high-contact visualisation.

The original ``single_deletion`` wrote four near-identical arc files
(``arcs.bed`` / ``arcs_diff.bed`` / ``arcs_ko.bed`` / ``arcs_true.bed``,
perturb.py lines 1182-1228) -- the same nested-pixel loop with a different
matrix and cutoff. ``write_arcs`` is the single de-duplicated implementation;
``write_regions`` writes the simple deletion-span bed (lines 1167-1170).
"""
import numpy as np


def write_arcs(matrix, chr_name, start, res, region_start, region_end, out_path,
               quantile=0.99, two_sided=False):
    """Write BED-PE-like arcs for pixels above threshold within the region.

    ``two_sided`` (used for the diff matrix) keeps pixels above the 0.99
    quantile of positive values OR below the 0.01 quantile of negative values;
    otherwise keeps pixels above the ``quantile`` cutoff. Pixel coordinates and
    the strict region bounds match the original exactly.
    """
    if two_sided:
        gain = np.quantile(matrix[matrix > 0], 0.99) if np.sum(matrix > 0) > 0 else 0.0
        loss = np.quantile(matrix[matrix < 0], 0.01) if np.sum(matrix < 0) > 0 else 0.0
        keep = lambda v: v > gain or v < loss
    else:
        cutoff = np.quantile(matrix, quantile)
        keep = lambda v: v > cutoff
    with open(out_path, 'w') as f:
        for i in range(matrix.shape[0]):
            for j in range(matrix.shape[1]):
                pixel_start_i = i * res + start
                pixel_end_i = i * res + start + res
                pixel_start_j = j * res + start
                pixel_end_j = j * res + start + res
                if (keep(matrix[i, j]) and
                        pixel_start_i > region_start and pixel_end_i < region_end and
                        pixel_start_j > region_start and pixel_end_j < region_end):
                    f.write(f'{chr_name}\t{pixel_start_i}\t{pixel_end_i}\t'
                            f'{chr_name}\t{pixel_start_j}\t{pixel_end_j}\t{matrix[i, j]}\n')


def write_regions(deletion_starts, deletion_widths, chr_name, out_path):
    """Write the deletion-span BED (perturb.py lines 1167-1170)."""
    with open(out_path, 'w') as f:
        if deletion_starts is not None and deletion_widths is not None:
            for deletion_start, deletion_width in zip(deletion_starts, deletion_widths):
                f.write(f'{chr_name}\t{deletion_start}\t{deletion_start + deletion_width}\n')
