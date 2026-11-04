#!/usr/bin/env python
"""Build real-genome locus datasets in the ``example_loci.tsv`` format.

Every row is one transcript laid over a window of reference sequence:

    column 1  locus name
    column 2  DNA (ACGT), in the transcript's own orientation (reverse
              complemented for '-' strand genes, so splicing always reads 5'->3')
    column 3  track, same length: '-' untranscribed, 'E' exon, 'I' intron.
              An intron spans [donor GT, past acceptor AG), i.e. exactly the
              GTF gap between two consecutive exons.
    column 4  variants "pos:ref>alt:label" comma-separated (may be empty).
              pos is 0-based into column 2; ref/alt are on the column-2 strand.

Each dataset is written as ``<dataset>.train.tsv`` / ``<dataset>.test.tsv``
(split by chromosome) plus ``<dataset>.meta.tsv`` carrying genomic coordinates
and provenance for every row, and ``summary.json`` with counts and drop reasons.

Datasets
--------
canonical_pc   one canonical transcript per protein-coding gene
canonical_lnc  one canonical transcript per lncRNA gene (weaker splice grammar)
one_intron     canonical pc/lnc transcripts with exactly one intron
two_intron     canonical pc/lnc transcripts with exactly two introns
compact        canonical pc/lnc transcripts whose whole window is <= --compact-len
alt_5ss        isoform pairs differing only by an alternative donor
alt_3ss        isoform pairs differing only by an alternative acceptor
exon_skip      isoform pairs, one includes a cassette exon, one skips it
retained_intron isoform pairs, one splices an intron, the other retains it
intergenic     gene-free windows, all '-' (negative control for false sites)
clinvar_splice canonical_pc loci carrying >=1 labelled ClinVar splice-region SNV
               (hg38 only, needs --clinvar-vcf)

Isoform-pair datasets put both isoforms of an event over the *same* window, so
the two rows share column 2 and differ only in column 3.  Only introns inside
the span both isoforms cover are compared, so differing TSS/TES do not hide an
event.

Label hygiene
-------------
A row's track only marks its own transcript, so any other annotated splice
site in the window is an unlabelled positive.  By default a locus is dropped if
another gene has a splice site on the same strand inside the window
(``--keep-other-gene-sites`` disables this).  Alternative sites of the *same*
gene cannot be avoided this way; their count is reported per row in the meta
file (``n_unlabelled_same_gene_sites``) so they can be filtered downstream.
Opposite-strand genes are kept (their sites read as CT/AC in this orientation)
and counted in ``n_antisense_sites``.

With ``--splice-motif strict`` (the default) a transcript is kept only if every
intron is GT..AG, matching the machine's convention; GC-AG and U12 AT-AC
introns (~1%) are dropped and counted.

Test chromosomes default to SpliceAI's (chr1,3,5,7,9).  Paralogs are *not*
removed across the split, so treat the test set as chromosome-held-out, not
homology-held-out.

Bacterial genomes
-----------------
``--assembly`` also takes a few RefSeq reference bacteria (or ``bacteria`` for
all of them), spanning ~32-66% GC.  Their FASTA/GFF are fetched from NCBI on
first use into ``data/external/bacteria/`` and md5-checked.  Bacteria have no
spliceosomal introns, so these rows never contain 'I': they are a test of
whether a model invents splice sites in intron-free sequence, and of gene-body
calling across very different base compositions.

genes          one row per annotated (non-pseudo) gene, gene +/- --flank
tiles          the whole genome cut into --tile-len windows, both orientations

In both, column 3 marks *every* annotated gene body (gene, pseudogene, RNA) on
the row's strand as 'E', not just the focal one: bacterial genomes are ~88%
coding and operons are dense, so marking only one gene would leave its
neighbours as unlabelled positives.  The gene span is used, not the CDS parts,
because NCBI encodes programmed frameshifts and frameshifted pseudogenes as
split CDS records that would otherwise read as 1-2 bp introns.  Caveat: the
annotation has no UTRs or operon structure, so '-' means "no annotated gene",
not "untranscribed".

With one chromosome there is nothing to hold out by chromosome, so the last
--bacteria-test-frac of each replicon is the test region; windows that cross
the train/test boundary are dropped.

Examples
--------
    python self_play/build_real_loci.py --assembly hg38 \\
        --clinvar-vcf self_play/data/external/clinvar.vcf.gz
    python self_play/build_real_loci.py --assembly bacteria
    python self_play/build_real_loci.py --assembly mm10 --max-len 5000 \\
        --datasets one_intron two_intron compact alt_5ss alt_3ss
"""
import argparse
import bisect
import gzip
import json
import os
import pickle
import random
import re
import sys
import time
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor

import numpy as np

DATA_ROOT = '/mnt/jinstore/JinLab04/dmp131/cshark_data/data'
ASSEMBLIES = {
    # GENCODE v38 (Ensembl 104) and vM25 (Ensembl 100): the annotations the
    # multi-track models are trained/evaluated against.
    'hg38': dict(gtf=f'{DATA_ROOT}/hg38/hg38_genes.gtf',
                 fasta_dir=f'{DATA_ROOT}/hg38/dna_sequence',
                 chroms=[f'chr{i}' for i in range(1, 23)] + ['chrX']),
    'mm10': dict(gtf=f'{DATA_ROOT}/mm10/mm10_genes.gtf',
                 fasta_dir=f'{DATA_ROOT}/mm10/dna_sequence',
                 chroms=[f'chr{i}' for i in range(1, 20)] + ['chrX']),
}
DEFAULT_TEST_CHROMS = ['chr1', 'chr3', 'chr5', 'chr7', 'chr9']

# RefSeq reference genomes: small, single-chromosome, well annotated, and
# chosen to span base composition.
NCBI_GENOMES = 'https://ftp.ncbi.nlm.nih.gov/genomes/all'
BACTERIA = {
    'mgen_g37': dict(accession='GCF_000027325.1_ASM2732v1',
                     organism='Mycoplasma genitalium G37'),        # 0.58 Mb, 32% GC
    'bsub_168': dict(accession='GCF_000009045.1_ASM904v1',
                     organism='Bacillus subtilis 168'),            # 4.2 Mb, 44% GC
    'ecoli_k12': dict(accession='GCF_000005845.2_ASM584v2',
                      organism='Escherichia coli K-12 MG1655'),    # 4.6 Mb, 51% GC
    'paer_pao1': dict(accession='GCF_000006765.1_ASM676v1',
                      organism='Pseudomonas aeruginosa PAO1'),     # 6.3 Mb, 66% GC
}
BACTERIAL_DATASETS = ['genes', 'tiles']

ALL_DATASETS = ['canonical_pc', 'canonical_lnc', 'one_intron', 'two_intron',
                'compact', 'alt_5ss', 'alt_3ss', 'exon_skip', 'retained_intron',
                'intergenic', 'clinvar_splice'] + BACTERIAL_DATASETS
EVENT_DATASETS = ('alt_5ss', 'alt_3ss', 'exon_skip', 'retained_intron')
# GENCODE merged these into 'lncRNA' in v31/vM22-ish; vM25 still splits them.
LNC_TYPES = {'lncRNA', 'lincRNA', 'antisense', 'sense_intronic',
             'sense_overlapping', 'processed_transcript', 'macro_lncRNA',
             'bidirectional_promoter_lncRNA', '3prime_overlapping_ncRNA'}
CANONICAL_TYPES = {'protein_coding'} | LNC_TYPES
INCOMPLETE_TAGS = {'mRNA_start_NF', 'mRNA_end_NF', 'cds_start_NF', 'cds_end_NF'}

COMP = str.maketrans('ACGTN', 'TGCAN')


def revcomp(s):
    return s.translate(COMP)[::-1]


# --------------------------------------------------------------------------
# GTF
# --------------------------------------------------------------------------
_ATTR_RE = re.compile(r'(\S+) "([^"]*)"')


def parse_gtf(path, chroms):
    """Return {transcript_id: dict} with 0-based half-open, sorted exons."""
    chroms = set(chroms)
    tx = {}
    opener = gzip.open if path.endswith('.gz') else open
    with opener(path, 'rt') as fh:
        for line in fh:
            if line.startswith('#'):
                continue
            f = line.split('\t', 8)
            if f[2] not in ('exon', 'transcript') or f[0] not in chroms:
                continue
            attrs = f[8]
            tid = attrs[attrs.find('transcript_id "') + 15:].split('"', 1)[0]
            if f[2] == 'transcript':
                a, tags = {}, set()
                for k, v in _ATTR_RE.findall(attrs):
                    if k == 'tag':
                        tags.add(v)
                    else:
                        a[k] = v
                t = tx.setdefault(tid, {'exons': []})
                t.update(chrom=f[0], strand=f[6], gene_id=a['gene_id'],
                         gene_name=a.get('gene_name', a['gene_id']),
                         gene_type=a.get('gene_type', ''),
                         transcript_type=a.get('transcript_type', ''),
                         tsl=a.get('transcript_support_level', 'NA'),
                         tags=tags)
            else:
                tx.setdefault(tid, {'exons': []})['exons'].append(
                    (int(f[3]) - 1, int(f[4])))
    for t in tx.values():
        t['exons'].sort()
        ex = t['exons']
        t['start'], t['end'] = ex[0][0], ex[-1][1]
        t['introns'] = [(ex[i][1], ex[i + 1][0]) for i in range(len(ex) - 1)]
    return tx


def load_gtf(path, chroms, cache_dir):
    os.makedirs(cache_dir, exist_ok=True)
    st = os.stat(path)
    cache = os.path.join(cache_dir, f'{os.path.basename(path)}.{st.st_size}.'
                                    f'{int(st.st_mtime)}.loci.pkl')
    if os.path.exists(cache):
        with open(cache, 'rb') as fh:
            tx = pickle.load(fh)
    else:
        t0 = time.time()
        tx = parse_gtf(path, ASSEMBLY_ALL_CHROMS)
        with open(cache, 'wb') as fh:
            pickle.dump(tx, fh, protocol=pickle.HIGHEST_PROTOCOL)
        log(f'parsed {path} ({len(tx)} transcripts) in {time.time() - t0:.0f}s')
    chroms = set(chroms)
    return {k: v for k, v in tx.items() if v['chrom'] in chroms}


ASSEMBLY_ALL_CHROMS = {f'chr{i}' for i in range(1, 23)} | {'chrX', 'chrY'}


def canonical_rank(t):
    """Higher is more canonical.  mm10 vM25 predates Ensembl_canonical, so the
    APPRIS/CCDS/basic/TSL tags carry the choice there."""
    tags = t['tags']
    return ('MANE_Select' in tags,
            'Ensembl_canonical' in tags,
            any(x.startswith('appris_principal') for x in tags),
            'CCDS' in tags,
            'basic' in tags,
            t['tsl'] == '1',
            sum(e - s for s, e in t['exons']),
            t['transcript_id'])


def is_complete(t):
    return not (t['tags'] & INCOMPLETE_TAGS)


# --------------------------------------------------------------------------
# Splice-site index (for the unlabelled-site filter)
# --------------------------------------------------------------------------
class SiteIndex:
    """Sorted intron-boundary coordinates per (chrom, strand) with gene ids."""

    def __init__(self, tx):
        buckets = defaultdict(set)
        for t in tx.values():
            if 'readthrough_transcript' in t['tags']:
                continue  # spans two genes and would veto both of them
            for s, e in t['introns']:
                buckets[(t['chrom'], t['strand'])].add((s, t['gene_id']))
                buckets[(t['chrom'], t['strand'])].add((e, t['gene_id']))
        self.pos, self.gene = {}, {}
        for k, v in buckets.items():
            v = sorted(v)
            self.pos[k] = np.array([p for p, _ in v], dtype=np.int64)
            self.gene[k] = [g for _, g in v]

    def query(self, chrom, strand, lo, hi):
        """[(pos, gene_id)] with lo <= pos <= hi."""
        k = (chrom, strand)
        if k not in self.pos:
            return []
        p = self.pos[k]
        i, j = np.searchsorted(p, lo, 'left'), np.searchsorted(p, hi, 'right')
        return list(zip(p[i:j].tolist(), self.gene[k][i:j]))


# --------------------------------------------------------------------------
# ClinVar
# --------------------------------------------------------------------------
_STARS = {'practice_guideline': 4, 'reviewed_by_expert_panel': 3,
          'criteria_provided,_multiple_submitters,_no_conflicts': 2,
          'criteria_provided,_single_submitter': 1}
_CODING_MC = ('missense_variant', 'nonsense', 'stop_gained', 'stop_lost',
              'frameshift', 'initiator_codon_variant', 'start_lost',
              'inframe_')


def clinvar_label(sig):
    sig = sig.lower()
    if sig in ('pathogenic', 'likely_pathogenic', 'pathogenic/likely_pathogenic'):
        return 'pathogenic'
    if sig in ('benign', 'likely_benign', 'benign/likely_benign'):
        return 'benign'
    return None


def load_clinvar(path, chroms, min_stars, drop_coding):
    """{chrom: (sorted pos0 array, [(pos0, ref, alt, label, id)])} for SNVs."""
    out = defaultdict(list)
    n = Counter()
    with gzip.open(path, 'rt') as fh:
        for line in fh:
            if line.startswith('#'):
                continue
            f = line.split('\t', 8)
            chrom = 'chr' + f[0]
            if chrom not in chroms:
                continue
            ref, alt = f[3], f[4]
            if len(ref) != 1 or len(alt) != 1 or ref not in 'ACGT' or alt not in 'ACGT':
                continue
            info = dict(x.split('=', 1) for x in f[7].rstrip().split(';') if '=' in x)
            label = clinvar_label(info.get('CLNSIG', ''))
            if label is None:
                n['unlabelled_sig'] += 1
                continue
            if _STARS.get(info.get('CLNREVSTAT', ''), 0) < min_stars:
                n['below_min_stars'] += 1
                continue
            if drop_coding and any(c in info.get('MC', '') for c in _CODING_MC):
                # a missense call near a splice site is not a splicing label
                n['coding_consequence'] += 1
                continue
            out[chrom].append((int(f[1]) - 1, ref, alt, label, f[2]))
            n[label] += 1
    res = {}
    for c, v in out.items():
        v.sort()
        res[c] = (np.array([x[0] for x in v], dtype=np.int64), v)
    log(f'clinvar SNVs kept: {dict(n)}')
    return res


# --------------------------------------------------------------------------
# Locus jobs
# --------------------------------------------------------------------------
def make_job(dataset, name, txs, flank, chrom_len, extra=None):
    """One row: transcript ``txs[0]`` over the window spanning all of ``txs``."""
    t = txs[0]
    ws = max(0, min(x['start'] for x in txs) - flank)
    we = min(chrom_len, max(x['end'] for x in txs) + flank)
    return dict(dataset=dataset, name=name, chrom=t['chrom'], strand=t['strand'],
                ws=ws, we=we, tx=t, extra=extra or {})


def intron_events(a, b):
    """Classify the A/B difference on introns both isoforms span.  Returns a
    list of (event_type, role_a, role_b, key) -- usually zero or one."""
    lo, hi = max(a['start'], b['start']), min(a['end'], b['end'])
    ia = {i for i in a['introns'] if i[0] >= lo and i[1] <= hi}
    ib = {i for i in b['introns'] if i[0] >= lo and i[1] <= hi}
    da, db = sorted(ia - ib), sorted(ib - ia)
    plus = a['strand'] == '+'
    if len(da) == 1 and len(db) == 1:
        (as_, ae), (bs, be) = da[0], db[0]
        if as_ == bs and ae != be:      # shared left boundary
            et = 'alt_3ss' if plus else 'alt_5ss'
        elif ae == be and as_ != bs:    # shared right boundary
            et = 'alt_5ss' if plus else 'alt_3ss'
        else:
            return []
        return [(et, 'ssA', 'ssB', (et, tuple(sorted([da[0], db[0]]))))]
    for inc, skp, flip in ((da, db, False), (db, da, True)):
        if len(inc) == 2 and len(skp) == 1 and skp[0] == (inc[0][0], inc[1][1]):
            key = ('exon_skip', (inc[0][1], inc[1][0]))
            return [('exon_skip', 'skip', 'inc', key) if flip
                    else ('exon_skip', 'inc', 'skip', key)]
    for spl, ret_t, ret_d, flip in ((da, b, db, False), (db, a, da, True)):
        if len(spl) == 1 and not ret_d:
            s, e = spl[0]
            if any(xs < s and xe > e for xs, xe in ret_t['exons']):
                key = ('retained_intron', spl[0])
                return [('retained_intron', 'retained', 'spliced', key) if flip
                        else ('retained_intron', 'spliced', 'retained', key)]
    return []


def build_jobs(tx, args, chrom_lens):
    by_gene = defaultdict(list)
    for tid, t in tx.items():
        t['transcript_id'] = tid
        by_gene[t['gene_id']].append(t)
    want = set(args.datasets)
    jobs = []
    drops = Counter()

    # ---- canonical families ------------------------------------------------
    for gid, ts in by_gene.items():
        gtype = ts[0]['gene_type']
        if gtype not in CANONICAL_TYPES:
            continue
        t = max(ts, key=canonical_rank)
        if 'readthrough_transcript' in t['tags']:
            drops['canonical:readthrough'] += 1
            continue
        if not is_complete(t):
            drops['canonical:incomplete_5_or_3_end'] += 1
            continue
        if not t['introns']:
            drops['canonical:single_exon'] += 1
            continue
        if min(e - s for s, e in t['introns']) < args.min_intron:
            drops['canonical:tiny_intron'] += 1
            continue
        clen = chrom_lens[t['chrom']]
        name = f"{t['gene_name']}_{t['transcript_id']}"
        ni = len(t['introns'])
        cands = []
        if gtype == 'protein_coding':
            cands.append('canonical_pc')
            cands.append('clinvar_splice')
        else:
            cands.append('canonical_lnc')
        if ni == 1:
            cands.append('one_intron')
        if ni == 2:
            cands.append('two_intron')
        cands.append('compact')
        for ds in cands:
            if ds in want:
                jobs.append(make_job(ds, name, [t], args.flank, clen,
                                     {'selection': 'canonical'}))

    # ---- isoform-pair events ---------------------------------------------
    if want & set(EVENT_DATASETS):
        for gid, ts in by_gene.items():
            if ts[0]['gene_type'] not in CANONICAL_TYPES:
                continue
            ok = [t for t in ts if t['introns']
                  and 'readthrough_transcript' not in t['tags']
                  and ('basic' in t['tags'] or t['tsl'] in ('1', '2'))
                  and min(e - s for s, e in t['introns']) >= args.min_intron]
            ok.sort(key=canonical_rank, reverse=True)
            seen, per_gene = set(), Counter()
            for i in range(len(ok)):
                for j in range(i + 1, len(ok)):
                    a, b = ok[i], ok[j]
                    if a['start'] >= b['end'] or b['start'] >= a['end']:
                        continue
                    for et, ra, rb, key in intron_events(a, b):
                        if et not in want or key in seen:
                            continue
                        if per_gene[et] >= args.max_events_per_gene:
                            drops[f'{et}:over_per_gene_cap'] += 1
                            continue
                        seen.add(key)
                        per_gene[et] += 1
                        eid = f"{a['gene_name']}_{et}{per_gene[et]}"
                        clen = chrom_lens[a['chrom']]
                        for t, o, role in ((a, b, ra), (b, a, rb)):
                            jobs.append(make_job(
                                et, f"{eid}_{role}_{t['transcript_id']}",
                                [t, o], args.flank, clen,
                                {'event_id': eid, 'role': role,
                                 'partner': o['transcript_id']}))
    return jobs, drops


def intergenic_jobs(tx, args, chrom_lens, lengths, rng):
    spans = defaultdict(list)
    for t in tx.values():
        spans[t['chrom']].append((t['start'] - args.intergenic_margin,
                                  t['end'] + args.intergenic_margin))
    merged = {}
    for c, v in spans.items():
        v.sort()
        m = [list(v[0])]
        for s, e in v[1:]:
            if s <= m[-1][1]:
                m[-1][1] = max(m[-1][1], e)
            else:
                m.append([s, e])
        merged[c] = ([x[0] for x in m], [x[1] for x in m])
    chroms = list(chrom_lens)
    w = np.array([chrom_lens[c] for c in chroms], dtype=float)
    w /= w.sum()
    jobs, tries = [], 0
    # oversample: windows with N are only rejected once sequence is loaded;
    # write_outputs trims back to --n-intergenic
    while len(jobs) < 2 * args.n_intergenic and tries < args.n_intergenic * 400:
        tries += 1
        L = int(rng.choice(lengths))
        c = chroms[rng.choices(range(len(chroms)), weights=w)[0]]
        if chrom_lens[c] <= L:
            continue
        s = rng.randrange(0, chrom_lens[c] - L)
        e = s + L
        st, en = merged.get(c, ([], []))
        k = bisect.bisect_right(st, e) - 1   # last span starting before e
        if k >= 0 and en[k] > s:
            continue
        strand = rng.choice('+-')
        jobs.append(dict(dataset='intergenic',
                         name=f'intergenic_{c}_{s}_{e}_{"p" if strand == "+" else "m"}',
                         chrom=c, strand=strand, ws=s, we=e, tx=None, extra={}))
    return jobs


# --------------------------------------------------------------------------
# Per-chromosome worker
# --------------------------------------------------------------------------
_G = {}


def _init_worker(site_index, clinvar, args):
    _G.update(sites=site_index, clinvar=clinvar, args=args)


def read_fasta(path):
    with gzip.open(path, 'rt') as fh:
        fh.readline()
        return ''.join(l.rstrip() for l in fh).upper()


def chrom_len_from_fasta(path):
    n = 0
    with gzip.open(path, 'rt') as fh:
        fh.readline()
        for l in fh:
            n += len(l.rstrip())
    return n


def local(p, ws, we, strand):
    """Genomic 0-based base -> 0-based index in the oriented locus."""
    return p - ws if strand == '+' else we - 1 - p


def render(job, seq_chrom):
    """Return (row, meta) or (None, drop_reason)."""
    args, sites = _G['args'], _G['sites']
    ws, we, strand, chrom = job['ws'], job['we'], job['strand'], job['chrom']
    L = we - ws
    if L > args.max_len:
        return None, 'window_over_max_len'
    if job['dataset'] == 'compact' and L > args.compact_len:
        return None, 'window_over_compact_len'
    seq = seq_chrom[ws:we]
    if len(seq) != L:
        return None, 'off_chrom_end'
    if args.max_n_frac == 0:
        if set(seq) - set('ACGT'):
            return None, 'non_ACGT'
    elif sum(ch not in 'ACGT' for ch in seq) > args.max_n_frac * L:
        return None, 'non_ACGT'
    track = ['-'] * L
    t = job['tx']
    meta = dict(name=job['name'], dataset=job['dataset'], chrom=chrom,
                win_start=ws, win_end=we, strand=strand, locus_len=L)
    variants = []
    if t is not None:
        for s, e in t['exons']:
            track[s - ws:e - ws] = 'E' * (e - s)
        for s, e in t['introns']:
            track[s - ws:e - ws] = 'I' * (e - s)
        # motifs, in transcript orientation
        bad = []
        for s, e in t['introns']:
            intr = seq_chrom[s:e]
            if strand == '-':
                intr = revcomp(intr)
            bad.append(intr[:2] + '-' + intr[-2:])
        motifs = Counter(bad)
        if args.splice_motif == 'strict' and set(motifs) != {'GT-AG'}:
            return None, 'non_GT_AG_intron'
        # unlabelled splice sites
        own = {p for i in t['introns'] for p in i}
        same, other = 0, 0
        for p, g in sites.query(chrom, strand, ws, we):
            if g == t['gene_id']:
                same += p not in own
            else:
                other += 1
        if other and not args.keep_other_gene_sites:
            return None, 'other_gene_sites_in_window'
        anti = len(sites.query(chrom, '-' if strand == '+' else '+', ws, we))
        meta.update(gene_id=t['gene_id'], gene_name=t['gene_name'],
                    gene_type=t['gene_type'], transcript_id=t['transcript_id'],
                    transcript_type=t['transcript_type'], tsl=t['tsl'],
                    n_exons=len(t['exons']), n_introns=len(t['introns']),
                    intron_motifs=';'.join(f'{k}x{v}' for k, v in motifs.items()),
                    n_unlabelled_same_gene_sites=same,
                    n_other_gene_sites=other, n_antisense_sites=anti,
                    tags=','.join(sorted(t['tags'])))
        # ClinVar splice-region SNVs for this transcript
        cv = _G['clinvar'].get(chrom) if _G['clinvar'] else None
        if cv is not None:
            variants = clinvar_for(t, cv, seq, ws, we, args)
    else:
        anti = sum(len(sites.query(chrom, s, ws, we)) for s in '+-')
        meta.update(n_other_gene_sites=anti)
    seq_o = seq if strand == '+' else revcomp(seq)
    track_o = ''.join(track) if strand == '+' else ''.join(reversed(track))
    meta.update(job['extra'])
    meta['n_variants'] = len(variants)
    meta['n_pathogenic'] = sum(v.endswith(':pathogenic') for v in variants)
    return (job['name'], seq_o, track_o, ','.join(variants)), meta


def clinvar_for(t, cv, seq, ws, we, args):
    pos, recs = cv
    i, j = np.searchsorted(pos, t['start']), np.searchsorted(pos, t['end'])
    if i == j:
        return []
    # splice-region: within exon_margin on the exon side or intron_margin on the
    # intron side of any junction of this transcript (genomic, either strand)
    region = []
    for s, e in t['introns']:
        region.append((s - args.variant_exon_margin, s + args.variant_intron_margin))
        region.append((e - args.variant_intron_margin, e + args.variant_exon_margin))
    out = []
    for p, ref, alt, label, vid in recs[i:j]:
        if not any(lo <= p < hi for lo, hi in region):
            continue
        if seq[p - ws] != ref:
            continue  # reference mismatch: skip rather than guess
        lp = local(p, ws, we, t['strand'])
        if t['strand'] == '-':
            ref, alt = ref.translate(COMP), alt.translate(COMP)
        out.append(f'{lp}:{ref}>{alt}:{label}')
    out.sort(key=lambda v: int(v.split(':', 1)[0]))
    return out


def process_chrom(chrom, fasta_path, jobs):
    seq = read_fasta(fasta_path)
    rows, drops = [], Counter()
    for job in jobs:
        row, meta = render(job, seq)
        if row is None:
            drops[f"{job['dataset']}:{meta}"] += 1
        else:
            rows.append((row, meta))
    return chrom, rows, drops


# --------------------------------------------------------------------------
# Bacteria
# --------------------------------------------------------------------------
def fetch_bacterium(key, ext_dir):
    """Download (once) and md5-check the RefSeq FASTA + GFF3 for ``key``."""
    import hashlib
    import urllib.request
    acc = BACTERIA[key]['accession']
    url = f'{NCBI_GENOMES}/{acc[:3]}/{acc[4:7]}/{acc[7:10]}/{acc[10:13]}/{acc}'
    d = os.path.join(ext_dir, acc)
    os.makedirs(d, exist_ok=True)
    paths = {}
    md5s = None
    for kind, suffix in (('fna', '_genomic.fna.gz'), ('gff', '_genomic.gff.gz')):
        fn = acc + suffix
        path = os.path.join(d, fn)
        if not os.path.exists(path):
            if md5s is None:
                with urllib.request.urlopen(f'{url}/md5checksums.txt') as r:
                    md5s = {l.split()[1].lstrip('./'): l.split()[0]
                            for l in r.read().decode().splitlines() if l.strip()}
            log(f'downloading {url}/{fn}')
            tmp = path + '.part'
            urllib.request.urlretrieve(f'{url}/{fn}', tmp)
            with open(tmp, 'rb') as fh:
                got = hashlib.md5(fh.read()).hexdigest()
            if got != md5s.get(fn):
                os.remove(tmp)
                raise RuntimeError(f'md5 mismatch for {fn}: {got} != {md5s.get(fn)}')
            os.replace(tmp, path)
        paths[kind] = path
    return paths


def read_multi_fasta(path):
    seqs, name, buf = {}, None, []
    with gzip.open(path, 'rt') as fh:
        for line in fh:
            if line.startswith('>'):
                if name:
                    seqs[name] = ''.join(buf).upper()
                name, buf = line[1:].split()[0], []
            else:
                buf.append(line.rstrip())
    if name:
        seqs[name] = ''.join(buf).upper()
    return seqs


def parse_bacterial_gff(path):
    """Gene and pseudogene features -> {replicon: [gene dict]} (0-based)."""
    genes = defaultdict(list)
    with gzip.open(path, 'rt') as fh:
        for line in fh:
            if line.startswith('#'):
                continue
            f = line.rstrip('\n').split('\t')
            if len(f) < 9 or f[2] not in ('gene', 'pseudogene'):
                continue
            a = dict(x.split('=', 1) for x in f[8].split(';') if '=' in x)
            genes[f[0]].append(dict(
                chrom=f[0], start=int(f[3]) - 1, end=int(f[4]), strand=f[6],
                gene_id=a.get('ID', ''), locus_tag=a.get('locus_tag', ''),
                gene_name=a.get('Name', a.get('locus_tag', '')),
                gene_type=a.get('gene_biotype', f[2]),
                pseudo=f[2] == 'pseudogene' or a.get('pseudo') == 'true'))
    for v in genes.values():
        v.sort(key=lambda g: (g['start'], g['end']))
    return genes


def bacterial_rows(key, paths, args):
    """All rows for one bacterium, with split already assigned."""
    seqs = read_multi_fasta(paths['fna'])
    genes = parse_bacterial_gff(paths['gff'])
    drops, out = Counter(), []
    for rep_, seq in seqs.items():
        L = len(seq)
        gs = [g for g in genes.get(rep_, []) if g['end'] <= L]
        drops['genes:wraps_origin'] += len(genes.get(rep_, [])) - len(gs)
        mask = {s: np.zeros(L, dtype=bool) for s in '+-'}
        for g in gs:
            mask[g['strand']][g['start']:g['end']] = True
        boundary = int(L * (1 - args.bacteria_test_frac))

        def emit(dataset, name, ws, we, strand, focal=None):
            if we - ws > args.max_len:
                drops[f'{dataset}:window_over_max_len'] += 1
                return
            if ws < boundary < we:
                drops[f'{dataset}:straddles_train_test_boundary'] += 1
                return
            sq = seq[ws:we]
            if set(sq) - set('ACGT'):
                drops[f'{dataset}:non_ACGT'] += 1
                return
            m = mask[strand][ws:we]
            track = np.where(m, 'E', '-')
            if strand == '-':
                sq, track = revcomp(sq), track[::-1]
            meta = dict(name=name, dataset=dataset, chrom=rep_, win_start=ws,
                        win_end=we, strand=strand, locus_len=we - ws,
                        split='test' if ws >= boundary else 'train',
                        n_genes_marked=sum(g['strand'] == strand and g['start'] < we
                                           and g['end'] > ws for g in gs),
                        frac_marked=round(float(m.mean()), 4),
                        gc=round((sq.count('G') + sq.count('C')) / len(sq), 4),
                        n_exons=0, n_introns=0, n_variants=0, n_pathogenic=0)
            if focal:
                meta.update(gene_id=focal['gene_id'], gene_name=focal['gene_name'],
                            gene_type=focal['gene_type'],
                            locus_tag=focal['locus_tag'], n_exons=1)
            out.append(((name, sq, ''.join(track), ''), meta))

        if 'genes' in args.datasets:
            for g in gs:
                if g['pseudo']:
                    drops['genes:pseudogene'] += 1
                    continue
                ws, we = max(0, g['start'] - args.flank), min(L, g['end'] + args.flank)
                emit('genes', f"{key}_{g['locus_tag'] or g['gene_id']}_{g['gene_name']}",
                     ws, we, g['strand'], g)
        if 'tiles' in args.datasets:
            for ws in range(0, L - args.tile_len + 1, args.tile_stride or args.tile_len):
                we = ws + args.tile_len
                for strand in '+-':
                    emit('tiles', f"{key}_{rep_}_{ws}_{we}_{'p' if strand == '+' else 'm'}",
                         ws, we, strand)
    return out, drops


def build_bacteria(keys, args, ext_dir):
    import copy
    for key in keys:
        paths = fetch_bacterium(key, ext_dir)
        a = copy.copy(args)
        a.assembly = key
        src = dict(gtf=paths['gff'], fasta_dir=paths['fna'],
                   organism=f"{BACTERIA[key]['organism']} ({BACTERIA[key]['accession']})",
                   split_desc=f'test = last {args.bacteria_test_frac:g} of each replicon',
                   track_desc="#   column 3  'E' every annotated gene body on this strand, "
                              "'-' no annotated gene (UTRs/operons unannotated); never 'I'")
        rows, drops = bacterial_rows(key, paths, a)
        out_dir = os.path.join(args.out_dir, key)
        os.makedirs(out_dir, exist_ok=True)
        log(f'== {key}: {src["organism"]}')
        write_outputs(rows, a, src, out_dir, drops, len(rows))
        log(f'wrote {out_dir}')


# --------------------------------------------------------------------------
# Output
# --------------------------------------------------------------------------
META_COLS = ['name', 'dataset', 'split', 'chrom', 'win_start', 'win_end',
             'strand', 'locus_len', 'gene_id', 'gene_name', 'gene_type',
             'transcript_id', 'transcript_type', 'tsl', 'n_exons', 'n_introns',
             'intron_motifs', 'event_id', 'role', 'partner',
             'n_unlabelled_same_gene_sites', 'n_other_gene_sites',
             'n_antisense_sites', 'n_variants', 'n_pathogenic', 'selection', 'tags',
             'locus_tag', 'n_genes_marked', 'frac_marked', 'gc']


def header(assembly, dataset, split, args, src):
    return '\n'.join([
        f'# REAL DATA -- {assembly} {dataset} ({split}), built by self_play/build_real_loci.py',
        f'# annotation: {src["gtf"]}',
        f'# genome:     {src["fasta_dir"]}',
    ] + ([f'# organism:   {src["organism"]}'] if 'organism' in src else []) + [
        '#   column 1  locus name (coordinates/provenance in the matching .meta.tsv)',
        "#   column 2  DNA (ACGT), transcript orientation ('-' strand genes reverse-complemented)",
        src.get('track_desc', "#   column 3  '-' untranscribed, 'E' exon, 'I' intron  "
                              "(intron = [donor GT, past acceptor AG))"),
        '#   column 4  variants "pos:ref>alt:label", 0-based into column 2, on the column-2 strand',
        f'# flank={args.flank} max_len={args.max_len} splice_motif={args.splice_motif} '
        + src.get('split_desc', f'test_chroms={",".join(args.test_chroms)}'),
    ]) + '\n'


def write_outputs(results, args, src, out_dir, drops, n_jobs):
    by_ds = defaultdict(list)
    for row, meta in results:
        if 'split' not in meta:
            meta['split'] = 'test' if meta['chrom'] in args.test_chroms else 'train'
        by_ds[meta['dataset']].append((row, meta))
    rng = random.Random(args.seed)
    summary = {'assembly': args.assembly, 'args': {k: v for k, v in vars(args).items()},
               'datasets': {}, 'drops': dict(sorted(drops.items()))}
    for ds in args.datasets:
        items = by_ds.get(ds, [])
        if ds == 'intergenic' and len(items) > args.n_intergenic:
            items = rng.sample(items, args.n_intergenic)
        if ds == 'clinvar_splice':
            items = [x for x in items if x[1]['n_variants'] > 0]
        if ds in EVENT_DATASETS:
            # both members of a pair must survive, or the pair is meaningless
            cnt = Counter(m['event_id'] for _, m in items)
            items = [x for x in items if cnt[x[1]['event_id']] == 2]
        items.sort(key=lambda x: (x[1]['chrom'], x[1]['win_start'], x[1]['name']))
        if args.max_loci and len(items) > args.max_loci:
            if ds in EVENT_DATASETS:
                eids = sorted({m['event_id'] for _, m in items})
                keep = set(rng.sample(eids, args.max_loci // 2))
                items = [x for x in items if x[1]['event_id'] in keep]
            else:
                items = sorted(rng.sample(items, args.max_loci),
                               key=lambda x: (x[1]['chrom'], x[1]['win_start']))
        if not items:
            log(f'{ds}: no loci')
            summary['datasets'][ds] = {'n': 0}
            continue
        for split in ('train', 'test'):
            with open(os.path.join(out_dir, f'{ds}.{split}.tsv'), 'w') as fh:
                fh.write(header(args.assembly, ds, split, args, src))
                for row, meta in items:
                    if meta['split'] == split:
                        fh.write('\t'.join(row) + '\n')
        with open(os.path.join(out_dir, f'{ds}.meta.tsv'), 'w') as fh:
            fh.write('\t'.join(META_COLS) + '\n')
            for _, meta in items:
                fh.write('\t'.join(str(meta.get(c, '')) for c in META_COLS) + '\n')
        lens = np.array([m['locus_len'] for _, m in items])
        sp = Counter(m['split'] for _, m in items)
        d = dict(n=len(items), train=sp['train'], test=sp['test'],
                 len_median=int(np.median(lens)), len_min=int(lens.min()),
                 len_max=int(lens.max()),
                 n_rows_with_variants=sum(m['n_variants'] > 0 for _, m in items),
                 n_variants=sum(m['n_variants'] for _, m in items),
                 n_pathogenic=sum(m['n_pathogenic'] for _, m in items))
        summary['datasets'][ds] = d
        log(f'{ds:16s} n={d["n"]:6d} (train {d["train"]}, test {d["test"]}) '
            f'len median {d["len_median"]} [{d["len_min"]}, {d["len_max"]}]'
            + (f' variants {d["n_variants"]} ({d["n_pathogenic"]} path)'
               if d['n_variants'] else ''))
    summary['n_jobs'] = n_jobs
    with open(os.path.join(out_dir, 'summary.json'), 'w') as fh:
        json.dump(summary, fh, indent=1, default=str)


def log(msg):
    print(f'[build_real_loci] {msg}', file=sys.stderr, flush=True)


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--assembly', required=True,
                   choices=sorted(ASSEMBLIES) + sorted(BACTERIA) + ['bacteria'],
                   help="'bacteria' builds every genome in BACTERIA")
    p.add_argument('--gtf', help='override the default GENCODE GTF')
    p.add_argument('--fasta-dir', help='override the per-chromosome chrN.fa.gz directory')
    p.add_argument('--out-dir', default=os.path.join(here, 'data', 'real'),
                   help='written to <out-dir>/<assembly>/')
    p.add_argument('--cache-dir', default=os.path.join(here, 'data', 'cache'))
    p.add_argument('--datasets', nargs='+', default=None, choices=ALL_DATASETS,
                   help='default: all (clinvar_splice only with --clinvar-vcf)')
    p.add_argument('--chroms', nargs='+', help='default: autosomes + chrX')
    p.add_argument('--test-chroms', nargs='+', default=DEFAULT_TEST_CHROMS)
    p.add_argument('--flank', type=int, default=100,
                   help="untranscribed '-' bases either side of the transcript(s)")
    p.add_argument('--max-len', type=int, default=8192,
                   help='drop loci whose window is longer than this')
    p.add_argument('--compact-len', type=int, default=3000,
                   help='window cap for the compact dataset')
    p.add_argument('--min-intron', type=int, default=30,
                   help='drop transcripts with an intron shorter than this '
                        '(GENCODE uses tiny gaps to encode frameshift indels)')
    p.add_argument('--splice-motif', choices=['strict', 'any'], default='strict',
                   help='strict: every intron must be GT..AG')
    p.add_argument('--keep-other-gene-sites', action='store_true',
                   help='keep windows containing another gene\'s same-strand splice sites')
    p.add_argument('--max-n-frac', type=float, default=0.0,
                   help='max fraction of non-ACGT bases (default 0: drop any)')
    p.add_argument('--max-events-per-gene', type=int, default=3)
    p.add_argument('--n-intergenic', type=int, default=2000)
    p.add_argument('--intergenic-margin', type=int, default=5000,
                   help='intergenic windows keep this far from any annotated transcript')
    p.add_argument('--max-loci', type=int, default=0,
                   help='random subsample per dataset (0 = keep all)')
    p.add_argument('--clinvar-vcf', help='ClinVar GRCh38 VCF (hg38 only)')
    p.add_argument('--min-stars', type=int, default=1,
                   help='minimum ClinVar review stars')
    p.add_argument('--keep-coding-variants', action='store_true',
                   help='keep ClinVar SNVs whose consequence is missense/nonsense/etc.')
    p.add_argument('--variant-exon-margin', type=int, default=3)
    p.add_argument('--variant-intron-margin', type=int, default=50)
    p.add_argument('--tile-len', type=int, default=5000,
                   help='bacteria: window length for the tiles dataset')
    p.add_argument('--tile-stride', type=int, default=0,
                   help='bacteria: tile step (0 = --tile-len, non-overlapping)')
    p.add_argument('--bacteria-test-frac', type=float, default=0.2,
                   help='bacteria: fraction at the end of each replicon held out as test')
    p.add_argument('--external-dir', default=os.path.join(here, 'data', 'external'),
                   help='bacterial genomes are downloaded to <external-dir>/bacteria/')
    p.add_argument('--workers', type=int, default=8)
    p.add_argument('--seed', type=int, default=0)
    args = p.parse_args()

    if args.assembly == 'bacteria' or args.assembly in BACTERIA:
        if args.clinvar_vcf:
            p.error('--clinvar-vcf is GRCh38; only valid with --assembly hg38')
        args.datasets = args.datasets or BACTERIAL_DATASETS
        bad = set(args.datasets) - set(BACTERIAL_DATASETS)
        if bad:
            p.error(f'not available for bacteria: {sorted(bad)}')
        keys = sorted(BACTERIA) if args.assembly == 'bacteria' else [args.assembly]
        build_bacteria(keys, args, os.path.join(args.external_dir, 'bacteria'))
        return
    if args.datasets and set(args.datasets) & set(BACTERIAL_DATASETS):
        p.error(f'{BACTERIAL_DATASETS} are bacteria-only datasets')

    src = dict(ASSEMBLIES[args.assembly])
    if args.gtf:
        src['gtf'] = args.gtf
    if args.fasta_dir:
        src['fasta_dir'] = args.fasta_dir
    chroms = args.chroms or src['chroms']
    if args.clinvar_vcf and args.assembly != 'hg38':
        p.error('--clinvar-vcf is GRCh38; only valid with --assembly hg38')
    if args.datasets is None:
        args.datasets = [d for d in ALL_DATASETS if d not in BACTERIAL_DATASETS
                         and (d != 'clinvar_splice' or args.clinvar_vcf)]
    elif 'clinvar_splice' in args.datasets and not args.clinvar_vcf:
        p.error('clinvar_splice needs --clinvar-vcf')
    out_dir = os.path.join(args.out_dir, args.assembly)
    os.makedirs(out_dir, exist_ok=True)

    fasta = {c: os.path.join(src['fasta_dir'], f'{c}.fa.gz') for c in chroms}
    missing = [c for c, f in fasta.items() if not os.path.exists(f)]
    if missing:
        p.error(f'missing FASTA for {missing} in {src["fasta_dir"]}')
    lens_cache = os.path.join(args.cache_dir, f'{args.assembly}.chrom_sizes.json')
    os.makedirs(args.cache_dir, exist_ok=True)
    chrom_lens = json.load(open(lens_cache)) if os.path.exists(lens_cache) else {}
    todo = [c for c in chroms if c not in chrom_lens]
    if todo:
        log(f'measuring {len(todo)} chromosome lengths')
        with ProcessPoolExecutor(args.workers) as ex:
            for c, n in zip(todo, ex.map(chrom_len_from_fasta, [fasta[c] for c in todo])):
                chrom_lens[c] = n
        json.dump(chrom_lens, open(lens_cache, 'w'), indent=1)
    chrom_lens = {c: chrom_lens[c] for c in chroms}

    log(f'loading {src["gtf"]}')
    tx = load_gtf(src['gtf'], chroms, args.cache_dir)
    log(f'{len(tx)} transcripts on {len(chroms)} chromosomes')
    for tid, t in tx.items():
        t['transcript_id'] = tid

    jobs, drops = build_jobs(tx, args, chrom_lens)
    if 'intergenic' in args.datasets:
        rng = random.Random(args.seed)
        lengths = [j['we'] - j['ws'] for j in jobs if j['dataset'] == 'canonical_pc'
                   and j['we'] - j['ws'] <= args.max_len] or [5000]
        jobs += intergenic_jobs(tx, args, chrom_lens, lengths, rng)
    log(f'{len(jobs)} candidate rows: '
        + ', '.join(f'{k}={v}' for k, v in Counter(j["dataset"] for j in jobs).items()))

    clinvar = None
    if args.clinvar_vcf:
        clinvar = load_clinvar(args.clinvar_vcf, set(chroms), args.min_stars,
                               not args.keep_coding_variants)
    sites = SiteIndex(tx)

    per_chrom = defaultdict(list)
    for j in jobs:
        # no window can pass max_len; skip before loading sequence
        if j['we'] - j['ws'] > args.max_len:
            drops[f"{j['dataset']}:window_over_max_len"] += 1
            continue
        per_chrom[j['chrom']].append(j)
    results = []
    order = sorted(per_chrom, key=lambda c: -chrom_lens[c])
    with ProcessPoolExecutor(args.workers, initializer=_init_worker,
                             initargs=(sites, clinvar, args)) as ex:
        futs = [ex.submit(process_chrom, c, fasta[c], per_chrom[c]) for c in order]
        for f in futs:
            c, rows, d = f.result()
            results.extend(rows)
            drops.update(d)
            log(f'{c}: {len(rows)} rows')
    write_outputs(results, args, src, out_dir, drops, len(jobs))
    log(f'wrote {out_dir}')


if __name__ == '__main__':
    main()
