"""Cross-track deletion-with-padding operator.

Faithful port of the ``del`` / ``deletion`` / ``delete`` branch from the
original ``deletion_with_padding`` (lines 1648-1661). Unlike the other
operators, a deletion removes the span from *every* track at once and re-pads
both ends so the window length is preserved, so it operates on all tracks
together rather than a single channel.
"""
import numpy as np


def delete_with_padding(seq_region, ctcf_region, atac_region, other_regions,
                        start, deletion_start, deletion_width,
                        left_del_pad, right_del_pad):
    """Excise ``[deletion_start, deletion_start+deletion_width)`` from all tracks.

    ``left_del_pad`` / ``right_del_pad`` are tuples
    ``(seq_pad, ctcf_pad, atac_pad, other_pads)`` prepended/appended to keep the
    total length constant. Returns ``(seq_region, ctcf_region, atac_region,
    other_regions)``.
    """
    rel_start = deletion_start - start
    rel_end = deletion_start - start + deletion_width
    left_seq_pad, left_ctcf_pad, left_atac_pad, left_other_pads = left_del_pad
    right_seq_pad, right_ctcf_pad, right_atac_pad, right_other_pads = right_del_pad
    print(left_seq_pad.shape, seq_region.shape, seq_region[:rel_start, :].shape)
    seq_region = np.concatenate((left_seq_pad, seq_region[:rel_start, :],
                                 seq_region[rel_end:, :], right_seq_pad), axis=0)
    ctcf_region = np.concatenate((left_ctcf_pad, ctcf_region[:rel_start],
                                  ctcf_region[rel_end:], right_ctcf_pad), axis=0)
    atac_region = np.concatenate((left_atac_pad, atac_region[:rel_start],
                                  atac_region[rel_end:], right_atac_pad), axis=0)
    if other_regions is not None:
        for i in range(len(other_regions)):
            other_regions[i] = np.concatenate((left_other_pads[i], other_regions[i][:rel_start],
                                               other_regions[i][rel_end:], right_other_pads[i]), axis=0)
    return seq_region, ctcf_region, atac_region, other_regions
