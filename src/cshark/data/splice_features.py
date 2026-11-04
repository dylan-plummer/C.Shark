"""Splice-site and gene annotation targets, derived from a GENCODE GTF.

Layer 4 predicts transcription, and the single largest source of structure in an
RNA-seq profile is where the exons are: coverage steps up and down at splice
sites by an order of magnitude.  A coverage-only model has to infer those steps
from the sequence with no direct supervision, which is exactly the gap
AlphaGenome closed by adding splice heads.  This module builds those targets.

Two kinds of supervision come out of an annotation alone, needing no extra
experiment:

``splice site class`` (per base, 5 classes)
    donor+, acceptor+, donor-, acceptor-, background -- taken from every
    annotated exon boundary, strand-aware.  This is the SpliceAI target.
``gene intervals``
    for the gene-level expression head, and for masking coverage losses to
    transcribed regions when wanted.

A third, ``splice site usage`` (what fraction of transcripts use each site), is
*not* derivable from an annotation -- it needs junction counts from real data.
:class:`SpliceSiteAnnotation` therefore exposes usage as an optional overlay
loaded from a junction table, and the trainer simply skips the usage head when
none is supplied rather than fitting a constant.

Coordinate conventions
----------------------
GTF is 1-based inclusive; everything here is converted to 0-based half-open to
match the rest of C.Shark.  For a ``+`` strand transcript with exons
``[e1_start, e1_end), [e2_start, e2_end)``, the donor is the first *intronic*
base after exon 1 (``e1_end``) and the acceptor is the last intronic base before
exon 2 (``e2_start - 1``).  On the ``-`` strand transcription runs the other way,
so the roles swap: the donor sits at ``e_start - 1`` and the acceptor at
``e_end``.  Getting this backwards is silent and costs the head everything, so
:func:`build_splice_sites` is written to be read against that description.

The first and last exon boundaries of a transcript are transcript ends, not
splice sites, and are excluded.
"""
import gzip
import os
import pickle

import numpy as np

#: Class indices for the 5-way per-base splice target.
DONOR_PLUS, ACCEPTOR_PLUS, DONOR_MINUS, ACCEPTOR_MINUS, BACKGROUND = 0, 1, 2, 3, 4
SPLICE_CLASSES = ('donor+', 'acceptor+', 'donor-', 'acceptor-', 'background')
NUM_SPLICE_CLASSES = 5


def _open(path):
    return gzip.open(path, 'rt') if path.endswith('.gz') else open(path, 'r')


def _attr(attributes, key):
    """Pull one value out of a GTF attribute string."""
    token = f'{key} "'
    i = attributes.find(token)
    if i < 0:
        return None
    i += len(token)
    j = attributes.find('"', i)
    return attributes[i:j] if j > i else None


def build_splice_sites(gtf_path, chroms=None, gene_types=None, verbose=True):
    """Parse a GTF into per-chromosome splice sites and gene intervals.

    Returns ``(sites, genes)`` where ``sites[chrom]`` is a dict with arrays
    ``position`` (0-based) and ``cls`` (one of the class indices above), and
    ``genes[chrom]`` is a list of ``(start, end, strand, gene_name)``.

    ``gene_types`` optionally restricts to e.g. ``{'protein_coding'}``; the
    default keeps everything, because lncRNAs and antisense transcripts are
    real transcription that the RNA-seq target contains whether or not the
    annotation subset does.
    """
    exons_by_tx = {}
    genes = {}
    with _open(gtf_path) as fh:
        for line in fh:
            if not line or line[0] == '#':
                continue
            f = line.rstrip('\n').split('\t')
            if len(f) < 9:
                continue
            chrom, feature, start, end, strand, attrs = f[0], f[2], f[3], f[4], f[6], f[8]
            if chroms is not None and chrom not in chroms:
                continue
            if gene_types is not None:
                gt = _attr(attrs, 'gene_type')
                if gt is not None and gt not in gene_types:
                    continue
            if feature == 'exon':
                tx = _attr(attrs, 'transcript_id')
                if tx is None:
                    continue
                # GTF 1-based inclusive -> 0-based half-open.
                exons_by_tx.setdefault((chrom, tx, strand), []).append(
                    (int(start) - 1, int(end)))
            elif feature == 'gene':
                genes.setdefault(chrom, []).append(
                    (int(start) - 1, int(end), strand,
                     _attr(attrs, 'gene_name') or _attr(attrs, 'gene_id') or '.'))

    sites = {}
    for (chrom, _tx, strand), exons in exons_by_tx.items():
        if len(exons) < 2:
            continue                       # single-exon transcript: no junctions
        exons.sort()
        pos, cls = sites.setdefault(chrom, ([], []))
        for k in range(len(exons) - 1):
            intron_start = exons[k][1]          # first intronic base
            intron_end = exons[k + 1][0] - 1    # last intronic base
            if intron_end < intron_start:
                continue                        # overlapping/degenerate exons
            if strand == '+':
                pos.append(intron_start); cls.append(DONOR_PLUS)
                pos.append(intron_end);   cls.append(ACCEPTOR_PLUS)
            else:
                # Transcription runs right-to-left: the intron's right edge is
                # the donor and its left edge is the acceptor.
                pos.append(intron_end);   cls.append(DONOR_MINUS)
                pos.append(intron_start); cls.append(ACCEPTOR_MINUS)

    out = {}
    for chrom, (pos, cls) in sites.items():
        p = np.asarray(pos, dtype=np.int64)
        c = np.asarray(cls, dtype=np.int8)
        # Transcripts of the same gene share boundaries; deduplicate so a site
        # used by 20 isoforms does not count 20 times.
        key = p * NUM_SPLICE_CLASSES + c
        _, uniq = np.unique(key, return_index=True)
        uniq = np.sort(uniq)
        order = np.argsort(p[uniq], kind='stable')
        out[chrom] = {'position': p[uniq][order], 'cls': c[uniq][order]}
    if verbose:
        n = sum(len(v['position']) for v in out.values())
        print(f'[splice] {gtf_path}: {n} unique splice sites over {len(out)} '
              f'chromosomes; {sum(len(v) for v in genes.values())} genes')
    return out, genes


class SpliceSiteAnnotation:
    """Windowed access to splice-site labels, with an on-disk parse cache.

    Parsing a full GENCODE GTF takes tens of seconds, which is fine once and
    intolerable in every dataloader worker, so the parsed arrays are cached next
    to the GTF (or at ``cache_path``) and memory-mapped thereafter.
    """

    def __init__(self, gtf_path, cache_path=None, chroms=None, gene_types=None,
                 verbose=True):
        self.gtf_path = gtf_path
        cache_path = cache_path or (gtf_path + '.cshark_splice.pkl')
        if os.path.exists(cache_path) and \
                os.path.getmtime(cache_path) >= os.path.getmtime(gtf_path):
            with open(cache_path, 'rb') as fh:
                self.sites, self.genes = pickle.load(fh)
            if verbose:
                n = sum(len(v['position']) for v in self.sites.values())
                print(f'[splice] loaded {n} splice sites from cache {cache_path}')
        else:
            self.sites, self.genes = build_splice_sites(
                gtf_path, chroms=chroms, gene_types=gene_types, verbose=verbose)
            try:
                with open(cache_path, 'wb') as fh:
                    pickle.dump((self.sites, self.genes), fh)
                if verbose:
                    print(f'[splice] wrote parse cache {cache_path}')
            except OSError as exc:      # read-only reference directory
                print(f'[splice] could not write cache ({exc}); parsing each run')
        self.usage = None

    # -- targets ---------------------------------------------------------
    def class_labels(self, chrom, start, end):
        """``(end - start,)`` int8 array of class indices, background-filled."""
        n = int(end - start)
        labels = np.full(n, BACKGROUND, dtype=np.int8)
        s = self.sites.get(chrom)
        if s is None or not len(s['position']):
            return labels
        lo = np.searchsorted(s['position'], start, 'left')
        hi = np.searchsorted(s['position'], end, 'left')
        if hi > lo:
            labels[s['position'][lo:hi] - start] = s['cls'][lo:hi]
        return labels

    def site_positions(self, chrom, start, end, max_sites=None):
        """Annotated site offsets within the window, split by class.

        Returns ``(positions, classes)``; the junction head needs candidate
        donor/acceptor positions as an *input*, and at inference time those come
        from the classification head instead.
        """
        s = self.sites.get(chrom)
        if s is None or not len(s['position']):
            return np.zeros(0, np.int64), np.zeros(0, np.int8)
        lo = np.searchsorted(s['position'], start, 'left')
        hi = np.searchsorted(s['position'], end, 'left')
        pos = s['position'][lo:hi] - start
        cls = s['cls'][lo:hi]
        if max_sites is not None and len(pos) > max_sites:
            keep = np.linspace(0, len(pos) - 1, max_sites).astype(int)
            pos, cls = pos[keep], cls[keep]
        return pos, cls

    def genes_in(self, chrom, start, end, min_overlap=1):
        """Annotated genes overlapping the window, as window-relative intervals."""
        out = []
        for g_start, g_end, strand, name in self.genes.get(chrom, []):
            lo, hi = max(g_start, start), min(g_end, end)
            if hi - lo >= min_overlap:
                out.append((int(lo - start), int(hi - start), strand, name))
        return out


def splice_class_counts(labels):
    """Per-class counts, for sanity-checking a batch's class balance."""
    return {name: int((labels == i).sum()) for i, name in enumerate(SPLICE_CLASSES)}
