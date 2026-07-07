"""DNA sequence perturbation operators.

Faithful port of the sequence-handling code from the original
``cshark/inference/perturb.py``:

- ``seq_perturb``     (lines 1789-1812): single-base variant (name preserved).
- ``seq_region_ko``   <- the ``track_name == 'seq'`` branch of
  ``deletion_with_padding`` (lines 1644-1749), excluding the cross-track
  ``del`` mode which lives in ``deletion.py``.

Operates on a one-hot ``seq_region`` array of shape ``(L, 5)`` in place.
"""
import numpy as np
import matplotlib.pyplot as plt

from cshark.perturb.dna import en_dict, reverse_complement


def seq_perturb(start, alt, seq, window=2097152):
    """Simulate a single-nucleotide DNA variant at ``start`` -> ``alt``.

    Verbatim port of the original ``seq_perturb``. Returns ``seq`` truncated to
    ``window``.
    """
    new_entry = np.zeros(5)
    alt_idx = en_dict[alt.lower()]
    new_entry[alt_idx] = 1
    ref_entry = seq[start, :]
    ref = ref_entry.argmax()
    ref_base = list(en_dict.keys())[list(en_dict.values()).index(ref)]
    print(f'Pos: {start}, Alt: {alt}, Ref: {ref_base.upper()}')
    if ref == alt_idx:
        print('No change')
    ref_bases = []
    for i in range(start - 10, start + 10):
        if i == start:
            ref_bases.append('*')
        if i < 0 or i >= len(seq):
            ref_bases.append('N')
        else:
            ref_bases.append(list(en_dict.keys())[list(en_dict.values()).index(seq[i].argmax())])
        if i == start:
            ref_bases.append('*')
    print(''.join(ref_bases).upper())
    seq[start, :] = new_entry
    return seq[:window]


def seq_region_ko(seq_region, chr_name, start, deletion_start, deletion_width, ko_mode):
    """Apply an in-place sequence KO to ``seq_region`` over the deletion span.

    Handles the non-cross-track sequence modes: zero/knockout (mask to N),
    shuffle, random, reverse (reverse-complement), and reverse_motif (CTCF
    motif scan + reverse-complement of matches). The cross-track ``del`` mode
    is handled separately in ``deletion.delete_with_padding``.

    Returns the (mutated) ``seq_region``.
    """
    rel_start = deletion_start - start
    rel_end = deletion_start - start + deletion_width
    if ko_mode == 'knockout' or ko_mode == 'zero':
        seq_region[rel_start:rel_end, :] = 0
        seq_region[rel_start:rel_end, 4] = 1
    elif ko_mode == 'shuffle':
        idxs = np.arange(seq_region[rel_start:rel_end, :].shape[0])
        np.random.shuffle(idxs)
        seq_region[rel_start:rel_end, :] = seq_region[rel_start:rel_end, :][idxs, :]
    elif ko_mode == 'random':
        rand_bases = np.random.choice(4, size=(deletion_width,))
        rand_seq = np.zeros((deletion_width, 5), dtype=np.float32)
        for i in range(4):
            rand_seq[:, i] = (rand_bases == i).astype(np.float32)
        if rel_start >= 0 and rel_end <= seq_region.shape[0]:
            seq_region[rel_start:rel_end, :] = rand_seq
    elif ko_mode == 'reverse':
        seq_region[rel_start:rel_end, :] = reverse_complement(seq_region[rel_start:rel_end, :])
    elif ko_mode == 'reverse_motif':
        _reverse_ctcf_motifs(seq_region, chr_name, start, deletion_start, deletion_width)
    return seq_region


def _reverse_ctcf_motifs(seq_region, chr_name, start, deletion_start, deletion_width):
    """Scan the deletion span for CTCF motifs (JASPAR) and reverse-complement them.

    Verbatim port of the ``reverse_motif`` block (lines 1675-1749). Writes
    diagnostic ``tmp/ctcf_corr.png`` and ``tmp/ctcf_motif.bed``.
    """
    from pyjaspar import jaspardb
    jdb_obj = jaspardb(release='JASPAR2024')
    motifs = jdb_obj.fetch_motifs(
        collection=['CORE'], tf_name='CTCF',
        tax_group=['Vertebrates'], species=['9606'], all_versions=False)
    motif = motifs[0]
    matrix_dict = motif.counts.normalize()
    matrix = []
    for base in ['A', 'T', 'C', 'G', 'N']:
        if base in matrix_dict:
            matrix.append(list(matrix_dict[base]))
        else:
            matrix.append([0] * len(matrix_dict['A']))
    matrix = np.array(matrix).T
    rel_start = deletion_start - start
    rel_end = deletion_start - start + deletion_width
    seq_scan = seq_region[rel_start:rel_end, :]
    corrs = []
    is_reverse = []
    for i in range(seq_scan.shape[0]):
        try:
            corr = np.dot(seq_scan[i: i + matrix.shape[0], :].flatten(), matrix.flatten()) / matrix.shape[0]
            rc_seq = reverse_complement(seq_scan[i: i + matrix.shape[0], :])
            corr_reverse = np.dot(rc_seq.flatten(), matrix.flatten()) / matrix.shape[0]
            is_reverse.append(corr_reverse > corr)
            corr = max(corr, corr_reverse)
            corrs.append(corr)
        except Exception:
            break
    corrs = np.array(corrs)
    max_idx = np.argmax(corrs)
    ref_bases = []
    for i in range(rel_start + max_idx, rel_start + max_idx + matrix.shape[0]):
        if i < 0 or i >= len(seq_region):
            ref_bases.append('N')
        else:
            ref_bases.append(list(en_dict.keys())[list(en_dict.values()).index(seq_region[i].argmax())])
    ref_bases = ''.join(ref_bases).upper()
    top_n = np.sum(corrs > 0.65)
    if top_n > 0:
        max_idxs = np.argsort(corrs)[-top_n:]
        forward_motif_xs = []
        forward_motif_ys = []
        reverse_motif_xs = []
        reverse_motif_ys = []
        for i in max_idxs:
            ref_bases = []
            for j in range(rel_start + i, rel_start + i + matrix.shape[0]):
                if j < 0 or j >= len(seq_region):
                    ref_bases.append('N')
                else:
                    ref_bases.append(list(en_dict.keys())[list(en_dict.values()).index(seq_region[j].argmax())])
            ref_bases = ''.join(ref_bases).upper()
            print(f'{chr_name}:{deletion_start + i} - {ref_bases} {"<" if is_reverse[i] else ">"} (corr: {corrs[i]:.3f})')
            if is_reverse[i]:
                reverse_motif_xs.append(i)
                reverse_motif_ys.append(corrs[i])
            else:
                forward_motif_xs.append(i)
                forward_motif_ys.append(corrs[i])
            motif_seq = seq_region[rel_start + i: rel_start + i + matrix.shape[0], :]
            rc_motif_seq = reverse_complement(motif_seq)
            seq_region[rel_start + i: rel_start + i + matrix.shape[0], :] = rc_motif_seq

        fig = plt.figure(figsize=(15, 4))
        plt.plot(corrs)
        plt.scatter(forward_motif_xs, forward_motif_ys, color='blue', marker='>', label='Forward motif')
        plt.scatter(reverse_motif_xs, reverse_motif_ys, color='red', marker='<', label='Reverse motif')
        plt.savefig('tmp/ctcf_corr.png')
        plt.close()

        with open('tmp/ctcf_motif.bed', 'w') as f:
            for i in max_idxs:
                f.write(f'{chr_name}\t{deletion_start + i}\t{deletion_start + i + matrix.shape[0]}\t{"<" if is_reverse[i] else ">"}\t{corrs[i]:.3f}\n')
    else:
        print('No motifs found with correlation > 0.65')
