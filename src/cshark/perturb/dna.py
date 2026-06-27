"""DNA one-hot helpers shared across the perturbation engine.

Verbatim ports of the small helpers that lived at module scope in the original
``cshark/inference/perturb.py`` (``en_dict`` and ``reverse_complement``).
"""
import numpy as np

# Base -> one-hot channel index. Channel 4 is 'N' (ambiguous / masked).
en_dict = {'a': 0, 't': 1, 'c': 2, 'g': 3, 'n': 4}


def reverse_complement(seq):
    """Reverse-complement a one-hot encoded sequence array of shape ``(L, 5)``.

    Reverses along the length axis and swaps complementary channels
    (a<->t, c<->g), leaving N (channel 4) in place. Verbatim port of the helper
    from the original ``perturb.py``.
    """
    seq = np.flip(seq, 0)
    seq_comp = np.concatenate([seq[:, 1:2],
                               seq[:, 0:1],
                               seq[:, 3:4],
                               seq[:, 2:3],
                               seq[:, 4:5]], axis=1)
    return seq_comp


def base_at(seq, i):
    """Return the uppercase base char at row ``i`` of one-hot ``seq``.

    Out-of-range indices return 'N'. Mirrors the inline lookup the original
    code repeated in several places via ``en_dict``.
    """
    if i < 0 or i >= len(seq):
        return 'N'
    idx = seq[i].argmax()
    return list(en_dict.keys())[list(en_dict.values()).index(idx)].upper()
