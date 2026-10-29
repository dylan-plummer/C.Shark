"""Perturbation planning for single-locus runs (extracted from single_locus).

``plan_perturbations`` is the verbatim ko-mode loop from the original
``single_deletion`` (perturb.py ~818-923): it applies seq / enformer_seq /
alphagenome_seq base substitutions to the sequence, flags the active
sequence-model mode, and collects the pending
track perturbations. Deletions (``del`` / ``deletion`` / ``delete``) are only
collected into ``pending_deletions``; they are applied once, after every other
perturbation, by ``operators.deletion.apply_deletions``. Returns the (possibly
reassigned) seq_region and deletion_widths plus the collected plan.
"""
import numpy as np

from cshark.perturb.dna import en_dict, reverse_complement
from cshark.perturb.operators.seq_ops import seq_perturb

# Substitutions up to this length go base-by-base through seq_perturb (which prints
# each base's context); longer ones are written as one block with a one-line summary.
PER_BASE_MAX_BP = 50


def plan_perturbations(alt_bp, atac_path, bigwig_log_transform, channel_offset, chr_name, ctcf_path, deletion_starts, deletion_widths, hierarchical_rad21_model, input_track_names, ko_data_types, ko_mode, other_feats, peak_height, seq2_path, seq_path, seq_region, start, window, seed=None):
    # Resolve --alt list
    alt_bp_list = alt_bp if alt_bp is not None else []
    _alt_keywords = {'reverse', 'shuffle', 'random', 'mask'}
    # --seed: a dedicated RNG for the shuffle/random keywords only; without it the
    # global numpy RNG is used as before.
    rng = np.random.default_rng(seed) if seed is not None else np.random
    if deletion_starts is not None and deletion_widths is None:
        deletion_widths = []
        for i, ds in enumerate(deletion_starts):
            if i < len(alt_bp_list) and alt_bp_list[i].lower() not in _alt_keywords:
                deletion_widths.append(max(1, len(alt_bp_list[i])))
            else:
                deletion_widths.append(1)

    # Guard against SILENTLY dropping perturbations: the ko lists are zipped
    # together below, so mismatched lengths would truncate to the shortest and
    # skip the extra sites (e.g. giving --ko-mode once with two --ko-start would
    # only perturb the first SNP). Require them to align -- for N sites give N
    # entries in each of --ko / --ko-mode / --ko-start / --ko-width / --alt.
    if deletion_starts is not None:
        n = len(deletion_starts)
        lengths = {'--ko-start': n,
                   '--ko-width': len(deletion_widths) if deletion_widths is not None else n,
                   '--ko': len(ko_data_types),
                   '--ko-mode': len(ko_mode)}
        if len(set(lengths.values())) > 1:
            raise ValueError(
                f"Mismatched perturbation list lengths {lengths}: --ko / --ko-mode / --ko-start / "
                f"--ko-width must each have the SAME number of entries (one per perturbation), "
                f"otherwise extra sites are silently dropped. For N SNPs give N entries in each, e.g. "
                f"--ko seq seq --ko-mode enformer_seq enformer_seq --ko-start P1 P2 --ko-width 1 1 --alt B1 B2"
            )
        missing_alt = [i for i, m in enumerate(ko_mode)
                       if m in ('seq', 'enformer_seq', 'alphagenome_seq') and i >= len(alt_bp_list)]
        if missing_alt:
            raise ValueError(
                f"--alt missing for seq/enformer_seq/alphagenome_seq site index(es) {missing_alt}: give "
                f"one --alt base per perturbation (aligned with --ko-start), otherwise those sites "
                f"silently become 'N'."
            )

    def _resolve_alt_string(raw_alt, rel_start, rel_end, current_seq_region, label):
        """Return ``(alt_string, rel_start)``; keyword spans are clipped to the window."""
        idx_to_base = {0: 'a', 1: 't', 2: 'c', 3: 'g', 4: 'n'}
        raw_alt_lower = raw_alt.lower()
        if raw_alt_lower not in _alt_keywords:
            return raw_alt, rel_start
        # A negative rel_start (e.g. --ko-start == --start, shifted by the 1-based -1)
        # would make current_seq_region[rel_start:rel_end] an empty slice.
        rel_start = max(rel_start, 0)
        rel_end = min(rel_end, current_seq_region.shape[0])
        if raw_alt_lower == 'mask':
            print(f'[{label}] Masking {chr_name}:{start + rel_start}-{start + rel_end} with N')
            return 'n' * max(0, rel_end - rel_start), rel_start
        if raw_alt_lower == 'reverse':
            rc = reverse_complement(current_seq_region[rel_start:rel_end, :])
            alt_string = ''.join(idx_to_base[row.argmax()] for row in rc)
            print(f'[{label}] Using reverse-complement of {chr_name}:{start + rel_start}-{start + rel_end}')
            return alt_string, rel_start
        if raw_alt_lower == 'shuffle':
            sub = current_seq_region[rel_start:rel_end, :].copy()
            idxs = np.arange(sub.shape[0])
            rng.shuffle(idxs)
            sub = sub[idxs, :]
            alt_string = ''.join(idx_to_base[row.argmax()] for row in sub)
            print(f'[{label}] Using shuffled bases of {chr_name}:{start + rel_start}-{start + rel_end}')
            return alt_string, rel_start
        if raw_alt_lower == 'random':
            bases = 'acgt'
            alt_string = ''.join(rng.choice(list(bases)) for _ in range(rel_end - rel_start))
            print(f'[{label}] Using random bases for {chr_name}:{start + rel_start}-{start + rel_end}')
            return alt_string, rel_start

    def _substitute(seq_region, rel_start, alt_string, label):
        """Write ``alt_string`` into every allele of ``seq_region`` from ``rel_start``."""
        if len(alt_string) <= PER_BASE_MAX_BP:
            print(f'[{label}] Substituting {len(alt_string)} base(s) at '
                  f'{chr_name}:{start + rel_start} (rel {rel_start}): {alt_string.upper()}')
            for bp_offset, base in enumerate(alt_string):
                abs_pos = rel_start + bp_offset
                if 0 <= abs_pos < seq_region.shape[0]:
                    if seq2_path is not None:
                        seq1 = seq_region[:, :seq_region.shape[1] // 2]
                        seq2 = seq_region[:, seq_region.shape[1] // 2:]
                        seq1 = seq_perturb(abs_pos, base, seq1)
                        seq2 = seq_perturb(abs_pos, base, seq2)
                        seq_region = np.concatenate((seq1, seq2), axis=1)
                    else:
                        seq_region = seq_perturb(abs_pos, base, seq_region)
            return seq_region
        lo = max(rel_start, 0)
        hi = min(rel_start + len(alt_string), seq_region.shape[0])
        print(f'[{label}] Substituting {len(alt_string)} bases as one block at '
              f'{chr_name}:{start + lo}-{start + hi} (rel {lo}-{hi}): '
              f'{alt_string[:PER_BASE_MAX_BP].upper()}...')
        if lo >= hi:
            return seq_region
        idx = np.array([en_dict[b] for b in alt_string[lo - rel_start:hi - rel_start].lower()])
        block = np.zeros((hi - lo, 5), dtype=seq_region.dtype)
        block[np.arange(hi - lo), idx] = 1
        seq_region[lo:hi, :] = np.tile(block, (1, seq_region.shape[1] // 5))
        return seq_region

    # Apply perturbations
    enformer_seq_active = False
    alphagenome_seq_active = False
    hierarchical_active = hierarchical_rad21_model is not None
    pending_track_perturbations = []
    pending_deletions = []

    if deletion_starts is not None and deletion_widths is not None:
        for ko_idx, (deletion_start, deletion_width, ko_data_type, knockout_mode, ko_height) in enumerate(
                zip(deletion_starts, deletion_widths, ko_data_types, ko_mode, peak_height)):
            raw_alt = alt_bp_list[ko_idx] if ko_idx < len(alt_bp_list) else 'n' * deletion_width

            if knockout_mode in ('enformer_seq', 'alphagenome_seq'):
                # Both modes build the ALT sequence identically; they differ only
                # in which backbone (Enformer / AlphaGenome) predicts the delta.
                label = knockout_mode
                deletion_start -= 1
                if knockout_mode == 'enformer_seq':
                    enformer_seq_active = True
                else:
                    alphagenome_seq_active = True
                rel_start = deletion_start - start
                rel_end = rel_start + deletion_width
                alt_string, rel_start = _resolve_alt_string(raw_alt, rel_start, rel_end, seq_region, label)
                seq_region = _substitute(seq_region, rel_start, alt_string, label)
                continue

            if knockout_mode == 'seq':
                deletion_start -= 1
                rel_start = deletion_start - start
                rel_end = rel_start + deletion_width
                alt_string, rel_start = _resolve_alt_string(raw_alt, rel_start, rel_end, seq_region, 'seq')
                seq_region = _substitute(seq_region, rel_start, alt_string, 'seq')
                continue

            if knockout_mode in ('del', 'deletion', 'delete'):
                # WT coordinates; excised together after all other perturbations.
                pending_deletions.append((deletion_start, deletion_width))
                continue

            if ko_data_type in input_track_names:
                ko_channel = input_track_names.index(ko_data_type)
            else:
                ko_channel = -1
            left_del_pad = None
            right_del_pad = None
            pending_track_perturbations.append((
                deletion_start, deletion_width, ko_data_type, ko_channel,
                channel_offset, knockout_mode, ko_height, left_del_pad, right_del_pad,
            ))
    return seq_region, deletion_widths, pending_track_perturbations, pending_deletions, enformer_seq_active, alphagenome_seq_active, hierarchical_active
