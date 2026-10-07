"""Cross-track deletion operators.

``delete_with_padding`` is the faithful port of the ``del`` / ``deletion`` /
``delete`` branch from the original ``deletion_with_padding`` (lines 1648-1661),
kept for the legacy callers. It excises one span at a time, so it cannot apply
several deletions correctly (every excision shifts the coordinates of the next).

The single-locus engine uses the multi-site functions below instead:

* ``validate_deletion_request`` -- reject unsupported deletion requests before
  any model is loaded.
* ``apply_deletions`` -- excise ALL deletion intervals (given in WT coordinates)
  from every track at once, padding both window ends from the contiguous flanking
  genome so the window length is preserved. Returns a ``DeletionResult`` whose
  ``coord_map`` gives the WT genomic coordinate of every bp of the edited window.
* ``build_remap_matrix`` / ``align_matrix_to_wt`` / ``align_1d_to_wt`` -- map the
  prediction made on the edited window back onto WT bins (overlap-weighted), so it
  can be plotted on genomic coordinates and subtracted from the WT prediction.
  Bins that are 100% deleted are left blank.
* ``write_deletion_ko_bigwigs`` -- write the final model-input tracks, mapped back
  to WT coordinates (deleted bp = 0), as ``tmp/{track}_ko.bw`` for the KO panels.
"""
import os
from dataclasses import dataclass, field
from typing import List, Tuple

import numpy as np
from scipy.sparse import csr_matrix, diags

from cshark.perturb.config import WINDOW
from cshark.perturb.operators.base import DELETION_MODES


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


# ---------------------------------------------------------------------------
# Multi-site deletion
# ---------------------------------------------------------------------------

@dataclass
class DeletionResult:
    """Bookkeeping of one ``apply_deletions`` call."""
    intervals: List[Tuple[int, int]]          # merged [start, end) in WT coordinates
    left_pad_bp: int                          # genome prepended before ``start``
    right_pad_bp: int                         # genome appended after ``start + window``
    coord_map: np.ndarray = field(repr=False) # WT coordinate of every bp of the edited window


def deletion_indices(ko_mode):
    """Indices of the perturbation entries whose ``--ko-mode`` is a deletion."""
    return [i for i, mode in enumerate(ko_mode or []) if mode in DELETION_MODES]


def validate_deletion_request(cfg):
    """Reject unsupported deletion requests early (before any model is loaded).

    No-op unless at least one ``--ko-mode`` is ``del`` / ``deletion`` / ``delete``.
    """
    del_idx = deletion_indices(cfg.ko_mode)
    if not del_idx:
        return
    ko_data = list(cfg.ko_data or [])

    bad = [(i, ko_data[i] if i < len(ko_data) else None) for i in del_idx
           if i >= len(ko_data) or ko_data[i] != 'seq']
    if bad:
        raise ValueError(
            f"--ko-mode deletion removes the DNA together with ALL tracks, so it must be "
            f"paired with --ko seq (got --ko {', '.join(str(k) for _, k in bad)} at perturbation "
            f"index(es) {[i for i, _ in bad]}). Use e.g. '--ko seq --ko-mode deletion'. To remove "
            f"only a track's signal, use '--ko <track> --ko-mode zero' or '--ko-mode knockout'.")

    if cfg.start is None:
        raise ValueError('Deletions are only supported in single-locus mode: pass --start. '
                         'Full-chromosome mode does not support --ko-mode deletion.')
    for flag, enabled in (('--allele-haplotype', cfg.allele_haplotype),
                          ('--allele-peak-split', cfg.allele_peak_split),
                          ('--alt-fasta', cfg.alt_fasta is not None)):
        if enabled:
            raise ValueError(f'--ko-mode deletion cannot be combined with {flag}.')

    if cfg.deletion_start is None or cfg.deletion_width is None:
        raise ValueError('Deletions need explicit coordinates: give --ko-start and --ko-width '
                         '(one entry per perturbation, WT coordinates).')
    if max(del_idx) >= len(cfg.deletion_start) or max(del_idx) >= len(cfg.deletion_width):
        raise ValueError(
            f'Mismatched perturbation list lengths: --ko-mode has {len(cfg.ko_mode)} entries but '
            f'--ko-start has {len(cfg.deletion_start)} and --ko-width has {len(cfg.deletion_width)}. '
            f'Give one entry per perturbation in each list.')

    window_end = cfg.start + WINDOW
    for i in del_idx:
        d_start, d_width = cfg.deletion_start[i], cfg.deletion_width[i]
        if d_width < 1:
            raise ValueError(f'Deletion {i}: --ko-width must be >= 1 (got {d_width}).')
        if d_start < cfg.start or d_start + d_width > window_end:
            raise ValueError(
                f'Deletion {i} ({cfg.chr_name}:{d_start}-{d_start + d_width}) must lie inside the '
                f'prediction window {cfg.chr_name}:{cfg.start}-{window_end}.')

    bigwigs = cfg.bigwigs or {}
    if cfg.hierarchical_model_path is not None and 'rad21' not in bigwigs:
        # Without a rad21 bigwig the hierarchical model would have to predict the rad21
        # input track, and that predicted track has no flanking genome to pad deletions.
        from cshark.inference.utils.model_utils import get_all_track_names
        main_all_tracks, _, _ = get_all_track_names(cfg.model_path)
        if 'rad21' in main_all_tracks:
            raise ValueError(
                'Deletions with --hierarchical-model need an experimental RAD21 track: add '
                'rad21=<path> to --bigwigs (the predicted RAD21 track cannot be padded for deletions).')


def merge_deletion_intervals(starts, widths):
    """Sort ``(start, width)`` deletions and merge overlapping/adjacent ones.

    Returns a list of ``(start, end)`` half-open intervals in WT coordinates.
    """
    intervals = sorted((int(s), int(s) + int(w)) for s, w in zip(starts, widths))
    merged = []
    for s, e in intervals:
        if merged and s <= merged[-1][1]:
            print(f'Warning: deletion {s}-{e} overlaps or touches {merged[-1][0]}-{merged[-1][1]}; '
                  f'merging them into one deletion.')
            merged[-1] = (merged[-1][0], max(merged[-1][1], e))
        else:
            merged.append((s, e))
    return merged


def _chrom_length(chr_name, bigwig_paths):
    """Chromosome length from the header of the first readable bigwig, else None."""
    import pyBigWig
    for path in bigwig_paths:
        if path is None or not os.path.exists(path):
            continue
        bw = pyBigWig.open(path)
        try:
            length = bw.chroms(chr_name)
        finally:
            bw.close()
        if length:
            return int(length)
    return None


def apply_deletions(chr_name, start, window, deletions, seq_region, ctcf_region, atac_region,
                    other_regions, seq_path, ctcf_path, atac_path, other_feats=None,
                    seq2_path=None, bigwig_log=True):
    """Excise every deletion from all tracks at once, keeping the window length.

    ``deletions`` is a list of ``(start, width)`` in WT coordinates. With ``T`` the
    total deleted length, the window is extended by ``T // 2`` bp of genome on the
    left and ``T - T // 2`` bp on the right (loaded contiguously from the genome), and
    the deleted spans are removed from the extended window, leaving exactly ``window``
    bp. Returns ``(seq_region, ctcf_region, atac_region, other_regions, DeletionResult)``.
    """
    import cshark.inference.utils.inference_utils as infer

    intervals = merge_deletion_intervals([d[0] for d in deletions], [d[1] for d in deletions])
    total = sum(e - s for s, e in intervals)
    left_bp = total // 2
    right_bp = total - left_bp

    if start - left_bp < 0:
        raise ValueError(f'Deletions remove {total} bp, but only {start} bp of {chr_name} lie '
                         f'left of the window to pad it with {left_bp} bp.')
    chrom_len = _chrom_length(chr_name, [ctcf_path, atac_path] + list(other_feats or []))
    if chrom_len is not None and start + window + right_bp > chrom_len:
        raise ValueError(f'Deletions remove {total} bp, but the window end {start + window} plus '
                         f'{right_bp} bp of padding runs past the end of {chr_name} ({chrom_len} bp).')

    def load_flank(flank_start, flank_len):
        if flank_len == 0:
            return None
        flank = infer.load_region(chr_name, flank_start, seq_path, ctcf_path, atac_path, other_feats,
                                  seq2_path=seq2_path, window=flank_len, bigwig_log=bigwig_log)
        if flank[0].shape[0] != flank_len:
            raise ValueError(f'Could not load {flank_len} bp of {chr_name} at {flank_start} to pad '
                             f'the deletions (got {flank[0].shape[0]} bp).')
        return flank

    left = load_flank(start - left_bp, left_bp)
    right = load_flank(start + window, right_bp)

    def extend(index, region):
        parts = [p for p in (left[index] if left else None, region, right[index] if right else None)
                 if p is not None]
        return np.concatenate(parts, axis=0)

    ext_len = window + total
    keep = np.ones(ext_len, dtype=bool)
    for s, e in intervals:
        keep[s - start + left_bp:e - start + left_bp] = False
    coord_map = (np.arange(ext_len, dtype=np.int64) + (start - left_bp))[keep]
    assert coord_map.shape[0] == window, (coord_map.shape, window)

    seq_region = extend(0, seq_region)[keep]
    ctcf_region = extend(1, ctcf_region)[keep] if ctcf_region is not None else None
    atac_region = extend(2, atac_region)[keep] if atac_region is not None else None
    if other_regions is not None:
        other_regions = [np.concatenate([p for p in (left[3][i] if left else None, region,
                                                     right[3][i] if right else None)
                                         if p is not None], axis=0)[keep]
                         for i, region in enumerate(other_regions)]

    spans = ', '.join(f'{chr_name}:{s}-{e}' for s, e in intervals)
    print(f'[deletion] Removed {len(intervals)} interval(s), {total} bp in total ({spans}); '
          f'padded {left_bp} bp on the left and {right_bp} bp on the right.')
    result = DeletionResult(intervals=intervals, left_pad_bp=left_bp, right_pad_bp=right_bp,
                            coord_map=coord_map)
    return seq_region, ctcf_region, atac_region, other_regions, result


def build_remap_matrix(coord_map, start, window, n_bins):
    """Overlap matrix from edited-window bins to WT bins.

    ``M[j, i]`` = number of bp of edited-window bin ``i`` whose WT coordinate falls
    in WT bin ``j`` (padding bp fall outside the WT window and are dropped). Rows
    are normalised to sum to 1. Returns ``(M, fully_deleted)``, where
    ``fully_deleted[j]`` is True for WT bins none of whose bp survived.
    """
    n_bp = coord_map.shape[0]
    ko_bin = np.arange(n_bp, dtype=np.int64) * n_bins // n_bp
    wt_pos = coord_map - start
    valid = (wt_pos >= 0) & (wt_pos < window)
    wt_bin = wt_pos[valid] * n_bins // window
    counts = csr_matrix((np.ones(int(valid.sum())), (wt_bin, ko_bin[valid])),
                        shape=(n_bins, n_bins))   # duplicates are summed
    row_sum = np.asarray(counts.sum(axis=1)).ravel()
    fully_deleted = row_sum == 0
    inv = np.zeros(n_bins)
    inv[~fully_deleted] = 1.0 / row_sum[~fully_deleted]
    return diags(inv) @ counts, fully_deleted


def align_matrix_to_wt(pred, coord_map, start, window):
    """Map a contact matrix predicted on the edited window back onto WT bins.

    Fully deleted WT bins (rows and columns) are NaN.
    """
    remap, fully_deleted = build_remap_matrix(coord_map, start, window, pred.shape[0])
    aligned = np.asarray(remap @ (remap @ pred).T).T
    aligned[fully_deleted, :] = np.nan
    aligned[:, fully_deleted] = np.nan
    return aligned


def align_1d_to_wt(pred_1d, coord_map, start, window):
    """Map ``(n_bins, n_tracks)`` 1D predictions back onto WT bins.

    Fully deleted WT bins are 0 (no DNA left, so no signal).
    """
    remap, fully_deleted = build_remap_matrix(coord_map, start, window, pred_1d.shape[0])
    aligned = np.asarray(remap @ pred_1d)
    aligned[fully_deleted, :] = 0.0
    return aligned


def track_to_wt(track, coord_map, start, window):
    """Place an edited-window track back on WT coordinates; deleted bp become 0."""
    wt_track = np.zeros(window, dtype=float)
    wt_pos = coord_map - start
    valid = (wt_pos >= 0) & (wt_pos < window)
    wt_track[wt_pos[valid]] = track[valid]
    return wt_track


def _write_linear_bigwig(out_path, header_path, chr_name, start, signal):
    """Write a bp-resolution linear signal starting at ``start`` (runs of equal values merged)."""
    import pyBigWig
    bw = pyBigWig.open(header_path)
    header = list(bw.chroms().items())
    bw.close()
    signal = np.asarray(signal, dtype=float)
    change = np.flatnonzero(np.diff(signal)) + 1
    run_starts = np.concatenate(([0], change))
    run_ends = np.concatenate((change, [signal.shape[0]]))
    out = pyBigWig.open(out_path, 'w')
    out.addHeader(header)
    out.addEntries([chr_name] * run_starts.shape[0], (run_starts + start).tolist(),
                   ends=(run_ends + start).tolist(), values=signal[run_starts].tolist())
    out.close()


def write_deletion_ko_bigwigs(deletion, input_track_names, input_track_paths, ctcf_region,
                              atac_region, other_regions, chr_name, start, window,
                              bigwig_log=True, hierarchical_active=False):
    """Write the final (deleted) model-input tracks as ``tmp/{track}_ko.bw`` on WT coordinates.

    These feed the KO panels, so they show exactly what the model received: every
    earlier perturbation plus the deletions (deleted bp = 0). With the hierarchical
    RAD21 model active, ``tmp/rad21_hierarchical_perturbed.bw`` (the "model input"
    panel) is rewritten the same way.
    """
    other_offset = sum(1 for t in ('ctcf', 'atac') if t in input_track_names)
    for idx, (name, path) in enumerate(zip(input_track_names, input_track_paths)):
        if name == 'ctcf' and idx < other_offset:
            track = ctcf_region
        elif name == 'atac' and idx < other_offset:
            track = atac_region
        else:
            track = other_regions[idx - other_offset] if other_regions is not None else None
        if track is None or not os.path.exists(path):
            continue
        wt_track = track_to_wt(track, deletion.coord_map, start, window)
        if bigwig_log:
            wt_track = np.expm1(np.clip(wt_track, 0, None))
        _write_linear_bigwig(f'tmp/{name}_ko.bw', path, chr_name, start, wt_track)
        if hierarchical_active and name == 'rad21' and os.path.exists('tmp/rad21_hierarchical_perturbed.bw'):
            _write_linear_bigwig('tmp/rad21_hierarchical_perturbed.bw', path, chr_name, start, wt_track)
    print(f'[deletion] Wrote KO input tracks with the deletions applied: '
          f'{", ".join(f"tmp/{n}_ko.bw" for n in input_track_names)}')
