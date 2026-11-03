"""Layer-1-only training data: EVERY celltype's tracks for every window.

Used by ``train_hierarchical_with_enformer.py --layer1-finetune``.

WHY A SEPARATE DATASET
----------------------
The hierarchical trainer samples (2 Mb window, ONE celltype) and back-propagates
through one random Enformer tile of it.  With celltype-split heads that means
each head is updated only when its own celltype is drawn -- ~4/33 of steps at
batch size 4 -- while the shared trunk block is updated on every step.  The
trunk then learns the shared accessibility and the heads barely move from their
initialisation, so the model never learns which celltype a peak belongs to.

Here a sample is one Enformer-native window (196,608 bp of sequence -> 896 x
128 bp bins) carrying the targets of ALL celltypes.  The trunk runs once per
window and every head is supervised on every step, against the very same
sequence -- which is the only setting in which "celltype A is open here and B
is not" is something the loss can actually see.

TARGETS
-------
``log1p`` of the 128 bp bin MEAN of the raw bigwig, computed once into a
memory-mapped cache.  This is what every evaluator compares a head against (the
head emits linear signal; the evaluator divides it by the measured bin mean), so
training and evaluation now measure the same quantity.  Two departures from the
hierarchical path, both deliberate:

  * no Gaussian noise on the targets.  ChromosomeDataset adds N(0, 0.1) per bp to
    every feature track, and layer 1's target is read from those channels; on the
    lab's SC-islet ATAC that noise variance (0.01) is 4-7x the ENTIRE
    celltype-specific part of the loss;
  * a true bin mean rather than F.interpolate's 128x linear downsample, which
    reads ~1 bp per bin.

Augmentation is Enformer's own: a shift of the window (in whole 128 bp bins, so
targets stay exact) and reverse complement.

THE CACHE
---------
``<cache>/seq/<chrom>.npy``                          uint8 codes, A C G T = 0-3, other = 4
``<cache>/tracks/<celltype>/<track>/<chrom>.npy``    float32 128 bp bin means, linear
``<cache>/tracks/<celltype>/<track>/meta.json``      source bigwig path / size / mtime

Built once (in parallel) and reused by every later run that points at the same
directory; a bigwig whose size or mtime changed is re-binned.  The arrays are
opened with ``mmap_mode='r'``, so DataLoader workers share the page cache rather
than each holding a copy.  For 33 hg38 celltypes it is ~3.3 GB.
"""

import gzip
import json
import os
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import pandas as pd
import torch

from cshark.data.data_feature import bigwig_values

ENFORMER_CONTEXT_LENGTH = 196_608
ENFORMER_TARGET_LEN = 896 * 128            # 114,688 bp of output per window
ENFORMER_TRIM = (ENFORMER_CONTEXT_LENGTH - ENFORMER_TARGET_LEN) // 2   # 40,960
ENFORMER_OUTPUT_BINS = 896
BIN_BP = 128
#: Whole-chromosome bigwig reads are done in chunks of this many bp (a multiple
#: of BIN_BP), so a worker never holds chr1 at bp resolution (~1 GB) at once.
_READ_CHUNK = 8_388_608

# A C G T -> 0 1 2 3, everything else (N, IUPAC, '-') -> 4.  Case-insensitive.
_LUT = np.full(256, 4, dtype=np.uint8)
for _i, _ch in enumerate('ACGT'):
    _LUT[ord(_ch)] = _i
    _LUT[ord(_ch.lower())] = _i
# One-hot rows in Enformer's ACGT order.  Code 4 is an all-zero row: that is how
# the inference pipeline (load_window_acgt) presents an N, so training matches.
_ONEHOT = np.vstack([np.eye(4, dtype=np.float32), np.zeros((1, 4), np.float32)])


# ---------------------------------------------------------------------------
# Splits -- identical to GenomeDataset, so layer-1 numbers stay held out
# ---------------------------------------------------------------------------
def all_chroms(assembly):
    if str(assembly).startswith('hg'):
        nums = list(range(1, 23))
    else:
        nums = list(range(1, 20))
    return [f'chr{n}' for n in nums] + ['chrX']


def split_chroms(assembly, mode):
    """val = chr10, test = chr15, train = the rest minus chrX (as GenomeDataset)."""
    if mode == 'val':
        return ['chr10']
    if mode == 'test':
        return ['chr15']
    return [c for c in all_chroms(assembly) if c not in ('chr10', 'chr15', 'chrX')]


# ---------------------------------------------------------------------------
# Cache construction
# ---------------------------------------------------------------------------
def _atomic_save(path, arr):
    tmp = f'{path}.tmp{os.getpid()}.npy'
    np.save(tmp, arr)
    os.replace(tmp, path)


def _seq_job(fa_path, out_path):
    with gzip.open(fa_path, 'rb') as fh:
        raw = fh.read()
    if raw.startswith(b'>'):
        raw = raw[raw.find(b'\n') + 1:]
    raw = raw.replace(b'\n', b'').replace(b'\r', b'')
    _atomic_save(out_path, _LUT[np.frombuffer(raw, dtype=np.uint8)])
    return out_path


def _track_job(bw_path, chrom, out_path):
    import pyBigWig
    bw = pyBigWig.open(bw_path)
    try:
        length = bw.chroms(chrom)
        if length is None:
            raise KeyError(f'{chrom} is not in {bw_path}')
        n_bins = -(-int(length) // BIN_BP)
        out = np.zeros(n_bins, dtype=np.float32)
        for s in range(0, int(length), _READ_CHUNK):
            e = min(int(length), s + _READ_CHUNK)
            v = np.nan_to_num(bigwig_values(bw, chrom, s, e),
                              nan=0.0, posinf=0.0, neginf=0.0)
            np.clip(v, 0, None, out=v)
            # Pad the final partial bin with zeros: it is the chromosome's last
            # <128 bp and never inside a training window (the context would run
            # off the end).
            pad = (-len(v)) % BIN_BP
            if pad:
                v = np.concatenate([v, np.zeros(pad, dtype=v.dtype)])
            b0 = s // BIN_BP
            binned = v.reshape(-1, BIN_BP).mean(axis=1)
            out[b0:b0 + len(binned)] = binned
    finally:
        bw.close()
    _atomic_save(out_path, out)
    return out_path


def _bigwig_meta(path):
    st = os.stat(path)
    return {'source': os.path.realpath(path), 'size': st.st_size,
            'mtime': int(st.st_mtime), 'bin_bp': BIN_BP}


def build_cache(data_root, assembly, celltypes, tracks, cache_dir, chroms=None,
                num_workers=8, verbose=True):
    """Bin every (celltype, track) bigwig and every chromosome's sequence once.

    Idempotent: existing arrays whose source bigwig is unchanged are kept.
    Raises before doing any work if a celltype lacks a track -- layer-1
    finetuning scores every celltype on every window, so there is no "missing"
    case to mask.
    """
    chroms = list(chroms or all_chroms(assembly))
    root = os.path.join(data_root, assembly)
    seq_dir = os.path.join(root, 'dna_sequence')

    missing = [f'{c}/{t}' for c in celltypes for t in tracks
               if not os.path.isfile(os.path.join(root, c, 'genomic_features', f'{t}.bw'))]
    if missing:
        raise FileNotFoundError(
            f'--layer1-finetune scores every celltype on every window, so every '
            f'celltype needs every --enformer-tracks bigwig. Missing under {root}: '
            f'{missing}')
    absent_fa = [c for c in chroms if not os.path.isfile(os.path.join(seq_dir, f'{c}.fa.gz'))]
    if absent_fa:
        raise FileNotFoundError(f'No sequence for {absent_fa} under {seq_dir}')

    jobs = []
    os.makedirs(os.path.join(cache_dir, 'seq'), exist_ok=True)
    for chrom in chroms:
        out = os.path.join(cache_dir, 'seq', f'{chrom}.npy')
        if not os.path.isfile(out):
            jobs.append((_seq_job, (os.path.join(seq_dir, f'{chrom}.fa.gz'), out), f'seq {chrom}'))

    pending_meta = []
    for c in celltypes:
        for t in tracks:
            bw_path = os.path.join(root, c, 'genomic_features', f'{t}.bw')
            d = os.path.join(cache_dir, 'tracks', c, t)
            os.makedirs(d, exist_ok=True)
            meta_path = os.path.join(d, 'meta.json')
            meta = _bigwig_meta(bw_path)
            stale = True
            if os.path.isfile(meta_path):
                with open(meta_path) as fh:
                    stale = json.load(fh) != meta
            for chrom in chroms:
                out = os.path.join(d, f'{chrom}.npy')
                if stale or not os.path.isfile(out):
                    jobs.append((_track_job, (bw_path, chrom, out), f'{c}/{t} {chrom}'))
            pending_meta.append((meta_path, meta, stale))

    if jobs:
        if verbose:
            print(f'[layer1 cache] building {len(jobs)} arrays in {cache_dir} '
                  f'with {num_workers} workers (one-off; later runs reuse them)')
        done = 0
        with ProcessPoolExecutor(max_workers=max(1, num_workers)) as pool:
            futs = {pool.submit(fn, *a): label for fn, a, label in jobs}
            for fut in as_completed(futs):
                fut.result()                     # re-raise a worker's failure here
                done += 1
                if verbose and (done % 50 == 0 or done == len(jobs)):
                    print(f'[layer1 cache]   {done}/{len(jobs)}')
    # Meta last, so an interrupted build is redone rather than trusted.
    for meta_path, meta, stale in pending_meta:
        if stale or not os.path.isfile(meta_path):
            with open(meta_path, 'w') as fh:
                json.dump(meta, fh)
    if verbose:
        print(f'[layer1 cache] ready: {len(celltypes)} celltypes x {tracks} x '
              f'{len(chroms)} chromosomes')
    return cache_dir


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------
def _read_omit(bed_path):
    if not bed_path or not os.path.isfile(bed_path):
        return {}
    df = pd.read_csv(bed_path, sep='\t', header=None, usecols=[0, 1, 2],
                     names=['chr', 'start', 'end'])
    return {c: g[['start', 'end']].to_numpy() for c, g in df.groupby('chr')}


def _overlaps(regions, s, e):
    return regions is not None and len(regions) and bool(
        ((regions[:, 0] < e) & (regions[:, 1] > s)).any())


class Layer1WindowDataset(torch.utils.data.Dataset):
    """One Enformer window per item, with every celltype's targets.

    Returns ``(seq, target)``:
      * ``seq``    -- (196608, 4) float32 one-hot, ACGT, N = zero row;
      * ``target`` -- (896, K, C) float32 LINEAR 128 bp bin means, K tracks in
        ``tracks`` order and C celltypes in ``celltypes`` order (the order the
        trainer's head lookup is built in).

    Windows tile each chromosome at a stride of one output window.  A window is
    dropped when its output region touches a centromere/telomere interval or is
    more than ``max_n_frac`` N, and windows whose 196 kb context would run off
    the chromosome are never made -- so the model never trains on padding.
    """

    def __init__(self, cache_dir, chroms, celltypes, tracks, mode='train',
                 omit_bed=None, max_n_frac=0.5, shift=True, reverse_complement=True):
        self.cache_dir = cache_dir
        self.celltypes = list(celltypes)
        self.tracks = list(tracks)
        self.train = mode == 'train'
        self.shift = shift and self.train
        self.rc = reverse_complement and self.train
        omit = _read_omit(omit_bed)

        self.seq, self.tgt, self.windows, self.limits = {}, {}, [], {}
        for chrom in chroms:
            seq = np.load(os.path.join(cache_dir, 'seq', f'{chrom}.npy'), mmap_mode='r')
            arrs = [[np.load(os.path.join(cache_dir, 'tracks', c, t, f'{chrom}.npy'),
                             mmap_mode='r') for c in self.celltypes] for t in self.tracks]
            # Usable length: the shortest of the sequence and every track, so a
            # bigwig built on a slightly different patch can never index past
            # its end.
            length = min([len(seq)] + [len(a) * BIN_BP for row in arrs for a in row])
            self.seq[chrom], self.tgt[chrom] = seq, arrs
            regions = omit.get(chrom)
            # Output starts on the 128 bp grid (ENFORMER_TRIM is 320 bins), so
            # targets are whole cached bins with no resampling.
            last = length - ENFORMER_TRIM - ENFORMER_TARGET_LEN
            self.limits[chrom] = (ENFORMER_TRIM, last, regions)
            for s in range(ENFORMER_TRIM, last + 1, ENFORMER_TARGET_LEN):
                e = s + ENFORMER_TARGET_LEN
                if _overlaps(regions, s, e):
                    continue
                if np.count_nonzero(np.asarray(seq[s:e]) == 4) > max_n_frac * ENFORMER_TARGET_LEN:
                    continue
                self.windows.append((chrom, s))
        if not self.windows:
            raise ValueError(f'No usable {mode} windows on {list(chroms)}')

    def __len__(self):
        return len(self.windows)

    def __getitem__(self, idx):
        chrom, s = self.windows[idx]
        if self.shift:
            # Up to one output window, in whole bins, falling back to the
            # unshifted window if the shift would leave the chromosome or reach
            # an omitted region.
            lo, last, regions = self.limits[chrom]
            s2 = s + int(np.random.randint(0, ENFORMER_OUTPUT_BINS)) * BIN_BP
            if s2 <= last and not _overlaps(regions, s2, s2 + ENFORMER_TARGET_LEN):
                s = s2
        codes = np.asarray(self.seq[chrom][s - ENFORMER_TRIM:s + ENFORMER_TARGET_LEN + ENFORMER_TRIM])
        seq = _ONEHOT[codes]                                     # (196608, 4)
        b0 = s // BIN_BP
        tgt = np.stack([np.stack([np.asarray(a[b0:b0 + ENFORMER_OUTPUT_BINS])
                                  for a in row], axis=-1)
                        for row in self.tgt[chrom]], axis=1)     # (896, K, C)
        if self.rc and np.random.rand() < 0.5:
            # Reverse the strand: flip positions, and A<->T / C<->G, which in
            # ACGT channel order is simply reversing the channels.
            seq = seq[::-1, ::-1]
            tgt = tgt[::-1]
        return (torch.from_numpy(np.ascontiguousarray(seq, dtype=np.float32)),
                torch.from_numpy(np.ascontiguousarray(tgt, dtype=np.float32)))
