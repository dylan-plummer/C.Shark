"""1D track perturbation operators.

Verbatim port of ``track_ko`` from the original ``cshark/inference/perturb.py``
(lines 1761-1786). Operates on a single 1D signal array in place (and returns
it truncated to ``window``). Reuses ``knockout_peaks`` / ``chunk_shuffle`` from
``cshark.inference.utils.inference_utils`` -- not duplicated here.
"""
import numpy as np

from cshark.inference.utils.inference_utils import knockout_peaks, chunk_shuffle

# KO modes handled directly by track_ko (excludes seq-only and deletion modes).
TRACK_KO_MODES = (
    'zero', 'mean', 'knockout', 'increase', 'cluster',
    'shuffle', 'knockout_shuffle', 'reverse', 'reverse_motif',
)


def track_ko(start, end, track, window=2097152, ko_mode='zero', peak_height=2.0):
    """Apply a knockout ``ko_mode`` to ``track[start:end]`` in place.

    ``increase_<factor>`` and ``cluster_<ratio>`` carry a numeric suffix.
    Returns the (mutated) track truncated to ``window``.
    """
    if ko_mode == 'zero':
        track[start:end] = 0
    elif ko_mode == 'mean':
        mean = np.mean(np.concatenate([track[:start], track[end:]]))
        track[start:end] = mean
    elif ko_mode == 'knockout':
        track[start:end] = knockout_peaks(track[start:end], threshold=peak_height)
    elif 'increase' in ko_mode:
        increase_factor = float(ko_mode.split('_')[1]) if '_' in ko_mode else 2.0
        track[start:end] = knockout_peaks(track[start:end], threshold=peak_height, increase_factor=increase_factor)
    elif 'cluster' in ko_mode:
        cluster_ratio = float(ko_mode.split('_')[1]) if '_' in ko_mode else 0.05
        cluster_indices = np.random.choice(np.arange(start, end), size=int((end - start) * cluster_ratio), replace=False)
        for idx in cluster_indices:
            track[idx] = np.random.uniform(1, 5)
    elif ko_mode == 'shuffle':
        track[start:end] = chunk_shuffle(track[start:end])
    elif ko_mode == 'knockout_shuffle':
        track[start:end] = knockout_peaks(track[start:end], threshold=peak_height)
        track[start:end] = chunk_shuffle(track[start:end])
    elif ko_mode in ('reverse', 'reverse_motif'):
        track[start:end] = track[start:end][::-1]
    else:
        raise ValueError('ko_mode must be one of: zero, mean, knockout, increase, cluster, shuffle, knockout_shuffle, reverse, reverse_motif')
    return track[:window]
