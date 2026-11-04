"""Accumulate binned model predictions across windows and write them as bigwigs.

WHY AN ACCUMULATOR AND NOT A STREAMING WRITER
---------------------------------------------
The evaluators in this package do not sweep the genome left to right.  Layer-1
peak windows are centred on called peaks and each covers 114 kb of output, so
neighbouring peaks produce heavily overlapping predictions; layer-3 windows are
anchored at gene starts with a 10% margin, so they overlap too.  A bigwig needs
each interval written once, in coordinate order, so the predictions have to be
reduced to one value per bin *before* anything is written.  This class holds a
running sum and count per (chromosome bin, track) and divides at the end, so a
bin covered by three windows is written as the mean of the three -- the same
reduction ``predict_genome`` applies to its sliding window.  Windows also start
at arbitrary coordinates rather than on the output grid, so each one is
area-resampled onto that grid on the way in instead of being rounded to the
nearest bin, which would displace the whole window by up to half a bin.

WHAT GOES IN AND WHAT COMES OUT
-------------------------------
``add`` writes predictions **in whatever space the caller hands over**.  The
callers in this package hand over LINEAR signal (the layer-1 Enformer heads emit
linear; layer-3's 1D output is ``expm1``'d first when ``bigwig_log_transform``),
so the files are directly comparable with the measured coverage bigwigs and can
be divided to form ratios.  Deliberately NOT applied: library-depth scaling and
log2 median centering.  Those are properties of a *comparison between*
celltypes, not of a track -- baking them in would produce files that no longer
match the model's own output space or each other across runs.

Bins no window covered are left out of the file rather than written as zeros: a
gap in a bigwig reads as "no data", which is the truth for a region nothing was
run on, whereas a zero would claim the model predicted no signal there.

MEMORY
------
Six bytes per (bin, track) held until ``write`` -- ``float32`` sum plus
``uint16`` count.  The count is per track and not per bin because the callers
fill one celltype/mode at a time, so different columns of the same bin can have
different divisors.  At 64 bp bins, 18 tracks and 235 Mb of sequence that is
~400 MB; a coarser ``bin_bp`` divides it down proportionally.
"""

import os

import numpy as np

#: pyBigWig materialises a Python list per call, so a chromosome-long run is
#: written in chunks rather than as one 2-million element list.
_CHUNK_BINS = 500_000

_COUNT_MAX = np.iinfo(np.uint16).max


def _natural_key(chrom):
    body = chrom[3:] if chrom.lower().startswith('chr') else chrom
    return (0, int(body), '') if body.isdigit() else (1, 0, body)


def _resample_area(values, start_bp, src_bin_bp, out_bin_bp):
    """Length-weighted rebin of ``values`` onto the global ``out_bin_bp`` grid.

    ``values`` is ``(n, C)`` piecewise-constant over
    ``[start_bp + i*src_bin_bp, start_bp + (i+1)*src_bin_bp)``.  Returns
    ``(first_out_bin, (m, C))`` where each output row is the mean of the signal
    over the part of that output bin the window covers -- so a bin the window
    only half reaches gets the mean of that half, not a value diluted by the
    uncovered part.

    Done by evaluating the running integral of the signal at the output bins'
    edges.  The integral is piecewise linear with knots at the source bin
    edges, so an edge falling inside source bin ``i`` is exact by interpolation,
    and the construction is correct whether the output bins are wider than the
    source bins, equal to them, or merely out of phase with them.
    """
    n, C = values.shape
    end_bp = start_bp + n * src_bin_bp
    # Running integral at the source knots: F[i] = integral up to knot i.
    F = np.empty((n + 1, C), dtype=np.float64)
    F[0] = 0.0
    np.cumsum(values * src_bin_bp, axis=0, out=F[1:])

    j0 = int(np.floor(start_bp / out_bin_bp))
    j1 = int(np.ceil(end_bp / out_bin_bp))
    edges = np.arange(j0, j1 + 1, dtype=np.float64) * out_bin_bp
    # Clip to the window, then split each edge into (source bin, fraction).
    t = np.clip(edges, start_bp, end_bp)
    pos = (t - start_bp) / src_bin_bp
    i = np.clip(np.floor(pos).astype(np.int64), 0, n - 1)
    frac = pos - i
    Ft = F[i] + (frac * src_bin_bp)[:, None] * values[i]

    # Every bin from j0 to j1-1 overlaps the window by a positive length: the
    # first contains start_bp and the last contains end_bp - 1, so no empty
    # rows can arise and the block stays contiguous.
    width = np.diff(t)
    return j0, np.diff(Ft, axis=0) / np.maximum(width, 1e-9)[:, None]


class BinnedTrackWriter:
    """Bin-grid accumulator for one model layer's predicted tracks.

    Parameters
    ----------
    out_dir : str
        Directory the ``.bw`` files are written into.
    bin_bp : int
        Bin width the files are written at.  Any positive width works -- the
        source bins are area-resampled onto it -- but a multiple of the model's
        native bin width keeps each output bin an exact aggregate of whole
        source bins wherever the two grids are in phase.
    chrom_lengths : dict[str, int]
        Lengths for the bigwig header; every chromosome passed to :meth:`add`
        must appear here.
    track_labels : list[str]
        One label per track.  The label is the filename stem, so it should
        already carry the celltype and the input mode, e.g.
        ``layer3_predicted_alpha_total_rna_plus``.
    """

    def __init__(self, out_dir, bin_bp, chrom_lengths, track_labels, verbose=True):
        if bin_bp <= 0:
            raise ValueError(f'bin_bp must be positive, got {bin_bp}')
        if not track_labels:
            raise ValueError('track_labels is empty')
        self.out_dir = out_dir
        self.bin_bp = int(bin_bp)
        self.chrom_lengths = dict(chrom_lengths)
        self.labels = list(track_labels)
        self._col = {lab: i for i, lab in enumerate(self.labels)}
        self.verbose = verbose
        self._acc = {}              # chrom -> [sum (n_bins, T), count (n_bins, T)]
        self.n_added = 0
        self._saturated = False

    # -- accumulation ----------------------------------------------------
    def _arrays(self, chrom):
        if chrom not in self._acc:
            if chrom not in self.chrom_lengths:
                raise KeyError(f'No length registered for {chrom}; cannot size '
                               'its bin grid.')
            n = int(self.chrom_lengths[chrom]) // self.bin_bp + 1
            self._acc[chrom] = [np.zeros((n, len(self.labels)), dtype=np.float32),
                                np.zeros((n, len(self.labels)), dtype=np.uint16)]
        return self._acc[chrom]

    def add(self, chrom, start_bp, values, src_bin_bp, labels=None):
        """Add one window's prediction.

        ``values`` is ``(n_src_bins, n_cols)`` covering
        ``[start_bp, start_bp + n_src_bins * src_bin_bp)``.  ``labels`` names
        those columns in order; omitted, the writer's whole track list is meant.

        The window's bins need not line up with the output grid, and usually do
        not: a peak-centred window begins wherever its peak's midpoint puts it
        and a layer-3 window wherever its first gene does.  Each output bin is
        therefore given the LENGTH-WEIGHTED MEAN of the source signal over its
        own span (:func:`_resample_area`), not the value of whichever source bin
        happens to be nearest.  Rounding to the nearest bin instead would shift
        a whole window by up to half a bin in genomic coordinates, which is the
        one error a browser track cannot absorb.
        """
        values = np.asarray(values, dtype=np.float64)
        if values.ndim != 2:
            raise ValueError(f'values must be 2-D (bins, cols), got {values.shape}')
        src_bin_bp = int(src_bin_bp)
        if src_bin_bp <= 0:
            raise ValueError(f'src_bin_bp must be positive, got {src_bin_bp}')
        cols = ([self._col[lab] for lab in labels] if labels is not None
                else list(range(len(self.labels))))
        if values.shape[1] != len(cols):
            raise ValueError(f'{values.shape[1]} value columns for {len(cols)} labels')
        if values.shape[0] == 0:
            return

        b0, resampled = _resample_area(values, float(start_bp), src_bin_bp,
                                       self.bin_bp)
        sums, counts = self._arrays(chrom)
        lo, hi = max(0, b0), min(sums.shape[0], b0 + resampled.shape[0])
        if hi <= lo:
            return
        idx = np.ix_(np.arange(lo, hi), cols)
        sums[idx] += resampled[lo - b0:hi - b0].astype(np.float32)
        c = counts[idx]
        if c.size and c.max() >= _COUNT_MAX:
            self._saturated = True
        counts[idx] = np.minimum(c.astype(np.int32) + 1, _COUNT_MAX)
        self.n_added += 1

    # -- output ----------------------------------------------------------
    def bytes_estimate(self):
        """Bytes held once every registered chromosome has been touched."""
        return sum((int(L) // self.bin_bp + 1) * 6 * len(self.labels)
                   for L in self.chrom_lengths.values())

    def write(self):
        """Divide by the per-bin counts and write one bigwig per track label."""
        import pyBigWig

        if not self._acc:
            if self.verbose:
                print('[bigwig] nothing accumulated; no files written')
            return []
        if self._saturated:
            print(f'[bigwig] WARNING: >{_COUNT_MAX} windows over some bin; the '
                  f'mean there is over the first {_COUNT_MAX} only.')
        chroms = sorted(self._acc, key=_natural_key)
        header = [(c, int(self.chrom_lengths[c])) for c in chroms]
        paths = []
        for ti, label in enumerate(self.labels):
            path = os.path.join(self.out_dir, f'{label}.bw')
            bw = pyBigWig.open(path, 'w')
            n_written = 0
            try:
                bw.addHeader(header)
                for chrom in chroms:
                    sums, counts = self._acc[chrom]
                    cnt = counts[:, ti]
                    covered = cnt > 0
                    if not covered.any():
                        continue
                    vals = np.zeros(sums.shape[0], dtype=np.float64)
                    vals[covered] = sums[covered, ti] / cnt[covered].astype(np.float64)
                    vals = np.nan_to_num(vals, nan=0.0, posinf=0.0, neginf=0.0)
                    # Contiguous runs of covered bins, each written with the
                    # span/step form so one call covers a whole run.
                    bounds = np.concatenate(
                        [[0], np.flatnonzero(np.diff(covered.astype(np.int8))) + 1,
                         [covered.size]])
                    for s, e in zip(bounds[:-1], bounds[1:]):
                        if not covered[s]:
                            continue
                        for c0 in range(s, e, _CHUNK_BINS):
                            c1 = min(e, c0 + _CHUNK_BINS)
                            bw.addEntries(chrom, int(c0 * self.bin_bp),
                                          values=[float(x) for x in vals[c0:c1]],
                                          span=self.bin_bp, step=self.bin_bp)
                            n_written += c1 - c0
            finally:
                bw.close()
            paths.append(path)
            if self.verbose:
                print(f'[bigwig] {path}  ({n_written:,} x {self.bin_bp} bp bins)')
        return paths

    def free(self):
        self._acc.clear()
