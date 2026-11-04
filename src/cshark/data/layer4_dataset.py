"""Dataset wrapper adding layer-4 transcription targets to a ``GenomeDataset``.

``GenomeDataset`` already yields sequence, input tracks, the Hi-C matrix and the
1D target tracks for a 2,097,152 bp window.  Layer 4 needs three more things,
all of which are cheap to derive on the fly and none of which belong in the
shared dataset (they would change the tuple every other trainer unpacks):

* **splice-site class labels** at 1 bp over the central crop, from the GTF;
* **gene intervals** in latent-bin units, plus a per-gene expression target;
* optional **candidate donor/acceptor positions** for the junction head.

The gene-level expression target is deliberately derived from the *same*
experimental RNA coverage the coverage head is fit against -- the mean sense-strand
signal over the gene body.  Taking it from an external quantification instead
would introduce a second normalisation (and a second set of gene boundaries) that
the coverage target knows nothing about, so the two heads would be fit against
subtly inconsistent definitions of the same quantity.
"""
import numpy as np
import torch
from torch.utils.data import Dataset

from cshark.data.splice_features import (
    SpliceSiteAnnotation, DONOR_PLUS, DONOR_MINUS,
    ACCEPTOR_PLUS, ACCEPTOR_MINUS,
)

#: Index of each field in the ``GenomeDataset`` tuple when predict_hic and
#: predict_1d are both on (which layer-4 training always requires).
_SEQ, _FEATS, _MAT, _TGT1D, _START, _END, _CHROM, _CHRIDX = range(8)


class Layer4Dataset(Dataset):
    """Wrap a ``GenomeDataset`` and append the layer-4 targets.

    Parameters
    ----------
    base : GenomeDataset
        Must be built with ``predict_hic=True`` and ``predict_1d=True``.
    splice : SpliceSiteAnnotation
    window, splice_crop, bp_per_bin : int
        Geometry, matching :class:`~cshark.model.layer4_models.HierarchicalExpressionModel`.
    coverage_indices : list[int]
        Which columns of the base's stacked 1D targets are the layer-4 coverage
        tracks, in head order.
    sense_strand_index : dict
        ``{'+': col, '-': col}`` into ``coverage_indices`` -- which coverage track
        carries sense-strand transcription for each gene orientation, used for
        the gene-level target.
    max_genes : int
        Genes per window are padded/truncated to this many (a 2 Mb window holds
        ~25 in mouse); truncation keeps the longest, which are the ones with
        enough covered bins for a stable target.
    """

    def __init__(self, base, splice, window=2_097_152, splice_crop=131_072,
                 bp_per_bin=4096, target_1d_size=32768, coverage_indices=(0, 1),
                 sense_strand_index=None, max_genes=64, max_splice_sites=0,
                 min_gene_bp=1000):
        # Layer 4 reads seq/features/mat/target_1d out of the base tuple by fixed
        # position, which is only valid when the base emits both.  Check the flags
        # rather than the tuple length: GenomeDataset appends optional trailing
        # fields (conditioning vector, celltype index), so a length test can be
        # satisfied by the wrong combination and silently misalign every field --
        # e.g. --no-hic + predict_1d + celltype_index also gives an 8-tuple, where
        # position 2 is target_1d rather than the contact map.
        if not getattr(base, 'predict_hic', True):
            raise ValueError(
                'Layer4Dataset needs a GenomeDataset built with predict_hic=True '
                '(layer 4 consumes the contact map). Drop --no-hic for layer-4 runs, '
                'or use --layer4-hic-mode none if you want the no-3D ablation while '
                'still loading Hi-C.')
        if not getattr(base, 'predict_1d', False):
            raise ValueError(
                'Layer4Dataset needs a GenomeDataset built with predict_1d=True '
                '(the coverage targets come from the 1D tracks); pass --target-features.')
        self.base = base
        self.splice = splice
        self.window = window
        self.splice_crop = splice_crop
        self.bp_per_bin = bp_per_bin
        self.target_1d_size = target_1d_size
        self.bp_per_1d = window // target_1d_size
        self.coverage_indices = list(coverage_indices)
        self.sense_strand_index = sense_strand_index or {'+': 0, '-': 1}
        self.max_genes = max_genes
        self.max_splice_sites = max_splice_sites
        self.min_gene_bp = min_gene_bp

    def __len__(self):
        return len(self.base)

    def _crop_bounds(self, start):
        c = start + self.window // 2
        return c - self.splice_crop // 2, c + self.splice_crop // 2

    def _gene_targets(self, chrom, start, target_1d):
        """Gene intervals (in latent bins) plus a log-expression target each."""
        genes = [g for g in self.splice.genes_in(chrom, start, start + self.window)
                 if g[1] - g[0] >= self.min_gene_bp]
        genes.sort(key=lambda g: g[1] - g[0], reverse=True)
        genes = genes[:self.max_genes]

        bins = np.zeros((self.max_genes, 2), dtype=np.int64)
        strand = np.zeros(self.max_genes, dtype=np.float32)
        mask = np.zeros(self.max_genes, dtype=bool)
        target = np.zeros(self.max_genes, dtype=np.float32)
        for i, (lo, hi, gstrand, _name) in enumerate(genes):
            bins[i, 0] = lo // self.bp_per_bin
            bins[i, 1] = max(bins[i, 0] + 1, -(-hi // self.bp_per_bin))
            strand[i] = 1.0 if gstrand == '+' else -1.0
            mask[i] = True
            col = self.coverage_indices[self.sense_strand_index[
                gstrand if gstrand in self.sense_strand_index else '+']]
            s1d, e1d = lo // self.bp_per_1d, max(lo // self.bp_per_1d + 1,
                                                 -(-hi // self.bp_per_1d))
            e1d = min(e1d, self.target_1d_size)
            seg = target_1d[s1d:e1d, col]
            # Targets are ln(1+x); undo before averaging so this is a mean of
            # signal rather than a mean of logs.  Mean density, not the total:
            # a total scales with gene length and would put the target on an
            # O(10) log scale that swamps every other head at initialisation.
            # The head is given log gene length, so it can recover a total.
            dens = np.expm1(np.clip(seg, 0, 30)).mean() if len(seg) else 0.0
            target[i] = float(np.log1p(dens))
        return bins, strand, mask, target

    def _splice_targets(self, chrom, start):
        lo, hi = self._crop_bounds(start)
        labels = self.splice.class_labels(chrom, lo, hi).astype(np.int64)
        out = {'splice_class': labels,
               'splice_mask': np.ones_like(labels, dtype=bool)}
        if self.max_splice_sites:
            pos, cls = self.splice.site_positions(chrom, lo, hi)
            donors = pos[(cls == DONOR_PLUS) | (cls == DONOR_MINUS)]
            acceptors = pos[(cls == ACCEPTOR_PLUS) | (cls == ACCEPTOR_MINUS)]

            def pad(a):
                b = np.zeros(self.max_splice_sites, dtype=np.int64)
                v = np.zeros(self.max_splice_sites, dtype=bool)
                n = min(len(a), self.max_splice_sites)
                b[:n] = a[:n]
                v[:n] = True
                return b, v

            out['donor_pos'], out['donor_valid'] = pad(donors)
            out['acceptor_pos'], out['acceptor_valid'] = pad(acceptors)
        return out

    def __getitem__(self, idx):
        item = self.base[idx]
        if len(item) < 8:
            raise ValueError(
                'Layer4Dataset needs a GenomeDataset built with predict_hic=True '
                f'and predict_1d=True (got a {len(item)}-tuple).')
        seq, feats, mat = item[_SEQ], item[_FEATS], item[_MAT]
        tgt1d, start, chrom = item[_TGT1D], int(item[_START]), item[_CHROM]

        # GenomeDataset returns the 1D targets as a list of per-track arrays.
        target_1d = np.stack([np.asarray(t) for t in tgt1d], axis=1) \
            if isinstance(tgt1d, (list, tuple)) else np.asarray(tgt1d)

        out = {
            'seq': torch.as_tensor(np.asarray(seq), dtype=torch.float32),
            'features': torch.as_tensor(
                np.stack([np.asarray(f) for f in feats], axis=1)
                if isinstance(feats, (list, tuple)) else np.asarray(feats),
                dtype=torch.float32),
            'mat': torch.as_tensor(np.asarray(mat), dtype=torch.float32),
            'target_1d': torch.as_tensor(target_1d, dtype=torch.float32),
            'start': int(start),
            'chrom': str(chrom),
        }
        # Carry the base dataset's celltype tag through, when it has one.  Read off
        # the dataset rather than the tuple so it stays correct no matter which
        # optional trailing fields GenomeDataset appended.  Layer 1 needs it to pick
        # the right celltype head (--enformer-split-heads-by-celltype).
        base_celltype = getattr(self.base, 'celltype_index', None)
        if base_celltype is not None:
            out['celltype_idx'] = int(base_celltype)
        bins, strand, gmask, gtarget = self._gene_targets(chrom, start, target_1d)
        out['gene_bins'] = torch.as_tensor(bins)
        out['gene_strand'] = torch.as_tensor(strand)
        out['gene_mask'] = torch.as_tensor(gmask)
        out['gene_target'] = torch.as_tensor(gtarget)
        for k, v in self._splice_targets(chrom, start).items():
            out[k] = torch.as_tensor(v)
        return out


def build_splice_annotation(gtf_path, cache_path=None, verbose=True):
    """Convenience constructor so the trainer keeps one import."""
    return SpliceSiteAnnotation(gtf_path, cache_path=cache_path, verbose=verbose)
