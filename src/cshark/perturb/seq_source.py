"""Shared helper for the ``--alt-fasta`` whole-window ALT sequence source.

Both the single-locus and full-chromosome scopes replace the WT sequence of a
prediction window with the bases from an alternate per-chromosome ``.fa.gz``
directory (same layout as ``--seq``). This helper loads that ALT window with the
exact slice + one-hot encoding used for the WT sequence in ``infer.load_region``
so the result is drop-in compatible with the WT ``seq_region``.
"""
import os

import numpy as np

from cshark.data.data_feature import SequenceFeature


def load_alt_fasta_region(alt_fasta_dir, chr_name, start, window, n_alleles):
    """Load the whole-window ALT sequence for ``chr_name`` starting at ``start``.

    ``n_alleles`` is ``seq_region.shape[1] // 5`` of the WT region; for diploid
    inputs the ALT sequence is tiled across every allele so all bases fed to the
    model come from the alternate genome. Returns a one-hot ``(window, 5 * n_alleles)``
    array.
    """
    alt_chr_path = os.path.join(alt_fasta_dir, f'{chr_name}.fa.gz')
    if not os.path.exists(alt_chr_path):
        raise FileNotFoundError(
            f'--alt-fasta: expected per-chromosome file {alt_chr_path} but it is missing.')
    alt_region = SequenceFeature(path=alt_chr_path).get(start, start + window)  # (rows, 5)
    if n_alleles > 1:
        alt_region = np.concatenate([alt_region] * n_alleles, axis=1)
    return alt_region


def align_alt_to_wt(alt_region, seq_region):
    """Crop/zero-pad ``alt_region`` to the WT ``seq_region`` row count.

    Guards against an alt chromosome whose length differs from the WT assembly
    (e.g. indels): keeps row-parity with the WT window so downstream shapes match.
    """
    if alt_region.shape[0] == seq_region.shape[0]:
        return alt_region
    n = min(alt_region.shape[0], seq_region.shape[0])
    aligned = np.zeros_like(seq_region)
    aligned[:n] = alt_region[:n]
    return aligned
