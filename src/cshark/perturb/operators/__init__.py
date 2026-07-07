"""Perturbation operators and the top-level dispatcher.

``deletion_with_padding`` is the faithful port of the original function of the
same name (lines 1631-1758 of ``inference/perturb.py``): same signature, same
return tuple, same behaviour. It delegates the leaf logic to ``track_ops`` /
``seq_ops`` / ``deletion``.
"""
import numpy as np

from cshark.perturb.operators.track_ops import track_ko
from cshark.perturb.operators.seq_ops import seq_perturb, seq_region_ko
from cshark.perturb.operators.deletion import delete_with_padding
from cshark.perturb.operators.base import (
    SUPPORTED_KO_MODES, TRACK_MODES, SEQ_MODES, DELETION_MODES, canonical_mode,
)

__all__ = [
    'deletion_with_padding', 'track_ko', 'seq_perturb', 'seq_region_ko',
    'delete_with_padding', 'SUPPORTED_KO_MODES', 'canonical_mode',
]


def deletion_with_padding(chr_name, start, deletion_start, deletion_width, seq_region, ctcf_region, atac_region,
                          other_regions=None, ko_data=('ctcf',), ko_channels=(0,), channel_offset=0, ko_mode=('zero',),
                          peak_height=2.0, left_del_pad=None, right_del_pad=None):
    """Apply each ``(track, mode, channel)`` perturbation to the region in place.

    Faithful port of the original ``deletion_with_padding``. ``ko_data``/
    ``ko_mode``/``ko_channels`` are parallel lists applied in order. Returns
    ``(seq_region, ctcf_region, atac_region, other_regions)``.
    """
    rel_start = deletion_start - start
    rel_end = deletion_start - start + deletion_width
    for track_name, knockout_mode, channel_idx in zip(ko_data, ko_mode, ko_channels):
        if track_name == 'ctcf':
            ctcf_region = track_ko(rel_start, rel_end, ctcf_region, ko_mode=knockout_mode, peak_height=peak_height)
        elif track_name == 'atac':
            atac_region = track_ko(rel_start, rel_end, atac_region, ko_mode=knockout_mode, peak_height=peak_height)
        elif track_name == 'seq':
            if knockout_mode in DELETION_MODES:
                seq_region, ctcf_region, atac_region, other_regions = delete_with_padding(
                    seq_region, ctcf_region, atac_region, other_regions,
                    start, deletion_start, deletion_width, left_del_pad, right_del_pad)
            else:
                seq_region = seq_region_ko(seq_region, chr_name, start, deletion_start, deletion_width, knockout_mode)
        elif other_regions is not None:
            original = other_regions[channel_idx - channel_offset].copy()
            other_regions[channel_idx - channel_offset] = track_ko(
                rel_start, rel_end, other_regions[channel_idx - channel_offset],
                ko_mode=knockout_mode, peak_height=peak_height)
            if np.array_equal(original, other_regions[channel_idx - channel_offset]):
                print(f'Warning: {track_name} KO did not change the signal. Check the KO mode.')
    return seq_region, ctcf_region, atac_region, other_regions
