"""Imprinting-locus filtering for allele-specific benchmarks.

Allele-specific peaks inside an imprinted domain differ between the two alleles
for *epigenetic* reasons -- parent-of-origin marks laid down in the germline --
not because the two haplotypes differ in sequence.  A sequence-only model is
given no way to tell which parent a haplotype came from, so scoring it on those
peaks measures something it cannot in principle predict.  This module reads the
integrated imprinting reference and marks peaks falling in imprinted regions so
that they can be excluded from (or reported separately in) a benchmark.

Reference file layout
---------------------
``imprinting.full.list.tsv`` is the exploded join of several region definitions:
one row per LOC x REG x LOOP combination, with ``<PREFIX>_chr/_start/_end/_ID``
column groups and ``NA`` where a group does not apply to that row.

======  ====================================================================
LOC     Merged imprinting domain (+/-5 kb), spanning BOTH ends of an
        imprinting loop -- the widest definition.
BLOCK   Merged nearby loop anchors; considers only ONE end of the loop.
REG     BLOCK intersected with the other modalities; also ONE end.
CTCF    Per-modality evidence: the interval in which that modality was
ATAC    called imprinted.  ``NA`` when the modality showed nothing for the
RNA     row's region.  ``LOOP`` is the Hi-C (loop) evidence.
LOOP
======  ====================================================================

Because rows are exploded, modality evidence has to be aggregated per region
unit (grouping by ``<SCOPE>_ID``) before it can be counted -- that count is what
``min_modalities`` thresholds.  A unit with evidence from a single modality is a
weaker imprinting call than one corroborated by several, hence the choice
between the full list (``min_modalities=1``) and the corroborated subset.

Coordinates are reference (mm10) coordinates, matching both the allele-specific
peak files and the SNP-substituted pseudo-genomes.

Caveats worth knowing when picking a policy
-------------------------------------------
* ``scope='loc'`` spans both loop ends, so a LOC can be megabases wide -- the
  full LOC list covers ~218 Mb (~8% of mm10) and removes ~17% of a typical
  CTCF/ATAC allele-specific peak set.  ``min_modalities=2`` cuts that to ~1-2%.
* Every BLOCK is by construction a loop anchor, so LOOP evidence is present for
  100% of BLOCK units; counting it there inflates the tally by one for every
  unit.  ``load_imprinting_intervals`` warns when a modality is degenerate like
  this for the chosen scope.
"""
import os

import numpy as np
import pandas as pd

#: Region definitions selectable as the filtering scope, widest first.
SCOPES = ('loc', 'reg', 'block')
#: Evidence channels that can count toward the ``min_modalities`` tally.
MODALITIES = ('ctcf', 'atac', 'rna', 'loop')


def _non_na(series):
    """Boolean mask of "this column group applies to this row"."""
    return series.notna() & (series.astype(str).str.upper() != 'NA')


def _merge_intervals(df):
    """Merge overlapping/abutting intervals per chromosome.

    Returns ``{chrom: (starts, ends)}`` with both arrays sorted ascending and
    the intervals disjoint, which is what makes the ``searchsorted`` lookup in
    ``ImprintingIndex.overlaps`` a single comparison.
    """
    merged = {}
    for chrom, grp in df.groupby('chr', sort=False):
        order = np.argsort(grp['start'].values, kind='stable')
        starts = grp['start'].values[order]
        ends = grp['end'].values[order]
        out_s, out_e = [], []
        cur_s = cur_e = None
        for s, e in zip(starts, ends):
            if cur_s is None:
                cur_s, cur_e = s, e
            elif s <= cur_e:                      # overlapping or abutting
                cur_e = max(cur_e, e)
            else:
                out_s.append(cur_s); out_e.append(cur_e)
                cur_s, cur_e = s, e
        if cur_s is not None:
            out_s.append(cur_s); out_e.append(cur_e)
        merged[str(chrom)] = (np.asarray(out_s, dtype=np.int64),
                              np.asarray(out_e, dtype=np.int64))
    return merged


class ImprintingIndex:
    """Merged imprinted intervals with a vectorised overlap test."""

    def __init__(self, intervals, stats):
        self._iv = intervals
        self.stats = stats

    @property
    def chroms(self):
        return set(self._iv)

    @property
    def n_intervals(self):
        return int(sum(len(s) for s, _ in self._iv.values()))

    @property
    def total_bp(self):
        return int(sum(int((e - s).sum()) for s, e in self._iv.values()))

    def overlaps(self, chrom, start, end):
        """Boolean mask: does each ``[start, end)`` interval hit an imprinted region?

        ``chrom``/``start``/``end`` are equal-length array-likes.
        """
        chrom = np.asarray(chrom, dtype=object)
        start = np.asarray(start, dtype=np.int64)
        end = np.asarray(end, dtype=np.int64)
        hit = np.zeros(len(start), dtype=bool)
        for c in np.unique(chrom):
            iv = self._iv.get(str(c))
            if iv is None:
                continue
            starts, ends = iv
            sel = np.where(chrom == c)[0]
            # Last interval that begins strictly before this peak ends; a merged
            # index is disjoint and start-sorted, so only that one can overlap.
            # 'left' is required: intervals are half-open, so an interval starting
            # exactly at the peak's end does not overlap it.
            j = np.searchsorted(starts, end[sel], side='left') - 1
            ok = j >= 0
            if ok.any():
                idx = sel[ok]
                hit[idx] = ends[j[ok]] > start[idx]
        return hit

    def annotate(self, df, chrom_col='chr', start_col='start', end_col='end',
                 out_col='imprinted'):
        """Return ``df`` with a boolean ``out_col`` marking imprinted peaks."""
        out = df.copy()
        if len(out) == 0:
            out[out_col] = np.zeros(0, dtype=bool)
            return out
        out[out_col] = self.overlaps(out[chrom_col].values,
                                     out[start_col].values, out[end_col].values)
        return out


def _unit_evidence(df, scope, modalities):
    """Per region unit of ``scope``: which modalities showed imprinting.

    Returns a boolean DataFrame indexed by unit ID with one column per modality.
    """
    prefix = scope.upper()
    id_col = f'{prefix}_ID'
    rows = df[_non_na(df[id_col])]
    cols = {}
    for m in modalities:
        mod_col = f'{m.upper()}_ID'
        cols[m] = _non_na(rows[mod_col]).groupby(rows[id_col]).any()
    return pd.DataFrame(cols)


def modality_sensitivity(path, scope='loc', modalities=MODALITIES, flank=0):
    """How many units / bp survive each ``min_modalities`` cut, for reporting.

    Returns a list of ``{'min_modalities', 'n_units', 'n_intervals', 'total_bp'}``
    so a caller can show the user what the threshold choice costs before they
    commit to one.
    """
    df = _read_reference(path)
    ev = _unit_evidence(df, scope, modalities)
    counts = ev.sum(axis=1)
    rows = []
    for k in range(1, len(modalities) + 1):
        keep = set(counts.index[counts >= k])
        idx = _intervals_for_units(df, scope, keep, flank)
        rows.append({'min_modalities': k, 'n_units': len(keep),
                     'n_intervals': idx.n_intervals, 'total_bp': idx.total_bp})
    return rows


def _read_reference(path):
    if not os.path.exists(path):
        raise SystemExit(f'Imprinting reference not found: {path}')
    df = pd.read_csv(path, sep='\t', dtype=str)
    missing = [f'{p}_{s}' for p in ('LOC', 'REG', 'BLOCK') for s in ('chr', 'start', 'end', 'ID')
               if f'{p}_{s}' not in df.columns]
    if missing:
        raise SystemExit(f'Imprinting reference {path} is missing columns: {missing}')
    return df


def _intervals_for_units(df, scope, keep_ids, flank):
    """Merged ``ImprintingIndex`` over the ``scope`` intervals of ``keep_ids``."""
    prefix = scope.upper()
    rows = df[_non_na(df[f'{prefix}_ID']) & df[f'{prefix}_ID'].isin(keep_ids)]
    iv = rows[[f'{prefix}_chr', f'{prefix}_start', f'{prefix}_end']].drop_duplicates()
    iv.columns = ['chr', 'start', 'end']
    if iv.empty:
        return ImprintingIndex({}, {})
    iv['start'] = np.maximum(0, iv['start'].astype(np.int64) - int(flank))
    iv['end'] = iv['end'].astype(np.int64) + int(flank)
    return ImprintingIndex(_merge_intervals(iv), {})


def load_imprinting_intervals(path, scope='loc', min_modalities=2,
                              modalities=MODALITIES, flank=0, verbose=True):
    """Build the imprinted-region index for one filtering policy.

    Parameters
    ----------
    path : str
        ``imprinting.full.list.tsv``.
    scope : {'loc', 'reg', 'block'}
        Which region definition to exclude on.  ``loc`` is the widest (both loop
        ends); ``reg``/``block`` cover a single anchor.
    min_modalities : int
        Keep only region units with imprinting evidence from at least this many
        of ``modalities``.  ``1`` is the full list; ``2`` is the corroborated
        subset.
    modalities : sequence of str
        Which evidence channels count toward the tally.
    flank : int
        Extra bp added to each side of every kept interval.

    Returns
    -------
    ImprintingIndex
        With a populated ``.stats`` dict describing the policy.
    """
    scope = scope.lower()
    if scope not in SCOPES:
        raise SystemExit(f'--imprinting-scope must be one of {SCOPES}, got "{scope}".')
    modalities = [m.lower() for m in modalities]
    bad = [m for m in modalities if m not in MODALITIES]
    if bad:
        raise SystemExit(f'Unknown imprinting modalities {bad}; choose from {MODALITIES}.')
    if not modalities:
        raise SystemExit('--imprinting-modalities cannot be empty.')
    if not 1 <= min_modalities <= len(modalities):
        raise SystemExit(f'--imprinting-min-modalities must be in [1, {len(modalities)}] '
                         f'for modalities {modalities}, got {min_modalities}.')

    df = _read_reference(path)
    ev = _unit_evidence(df, scope, modalities)
    counts = ev.sum(axis=1)
    keep = set(counts.index[counts >= min_modalities])
    index = _intervals_for_units(df, scope, keep, flank)
    index.stats = {
        'path': path,
        'scope': scope,
        'min_modalities': int(min_modalities),
        'modalities': list(modalities),
        'flank': int(flank),
        'n_units_total': int(len(ev)),
        'n_units_kept': int(len(keep)),
        'n_intervals_merged': index.n_intervals,
        'total_bp': index.total_bp,
        'units_per_modality': {m: int(ev[m].sum()) for m in modalities},
        'unit_modality_histogram': {int(k): int(v) for k, v
                                    in counts.value_counts().sort_index().items()},
    }

    if verbose:
        s = index.stats
        print(f'[imprinting] {os.path.basename(path)}: scope={scope} '
              f'min_modalities={min_modalities} modalities={",".join(modalities)} '
              f'flank={flank}')
        print(f'[imprinting] kept {s["n_units_kept"]}/{s["n_units_total"]} region units '
              f'-> {s["n_intervals_merged"]} merged intervals covering '
              f'{s["total_bp"] / 1e6:.1f} Mb')
        print(f'[imprinting] units with evidence per modality: {s["units_per_modality"]}')
        print(f'[imprinting] units by modality count: {s["unit_modality_histogram"]}')
        for m in modalities:
            if len(ev) and ev[m].all():
                print(f'[imprinting] NOTE: every {scope.upper()} unit carries {m.upper()} '
                      f'evidence (it is part of how {scope.upper()} is defined), so it '
                      f'adds 1 to every tally; consider dropping it from '
                      f'--imprinting-modalities or raising --imprinting-min-modalities.')
    return index


def filter_report(df, index, label='', verbose=True):
    """Annotate ``df`` and split it into (retained, imprinted) frames.

    ``df`` needs ``chr``/``start``/``end`` columns.  Peaks on a chromosome absent
    from the reference can never match, which is reported because it matters:
    chrX allele-specific signal is dominated by X-inactivation -- another
    epigenetic effect a sequence model cannot see -- and the reference carries no
    chrX calls.
    """
    ann = index.annotate(df)
    if verbose and len(ann):
        missing = sorted(set(ann['chr'].astype(str)) - index.chroms)
        if missing:
            n = int(ann['chr'].astype(str).isin(missing).sum())
            print(f'[imprinting] {label}: {n} peaks on chromosomes with no imprinting '
                  f'calls in the reference ({",".join(missing)}) -- kept as-is.')
        print(f'[imprinting] {label}: excluding {int(ann["imprinted"].sum())}/{len(ann)} '
              f'peaks ({100 * ann["imprinted"].mean():.1f}%) as imprinted.')
    return ann[~ann['imprinted']].copy(), ann[ann['imprinted']].copy()
