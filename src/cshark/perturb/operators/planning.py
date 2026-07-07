"""Perturbation planning for single-locus runs (extracted from single_locus).

``plan_perturbations`` is the verbatim ko-mode loop from the original
``single_deletion`` (perturb.py ~818-923): it applies seq / enformer_seq /
alphagenome_seq base substitutions to the sequence, flags the active
sequence-model mode, and collects the pending
track perturbations (with del-padding loaded). Returns the (possibly reassigned)
seq_region and deletion_widths plus the collected plan.
"""
import numpy as np

import cshark.inference.utils.inference_utils as infer
from cshark.perturb.dna import reverse_complement
from cshark.perturb.operators.seq_ops import seq_perturb


def plan_perturbations(alt_bp, atac_path, bigwig_log_transform, channel_offset, chr_name, ctcf_path, deletion_starts, deletion_widths, hierarchical_rad21_model, input_track_names, ko_data_types, ko_mode, other_feats, peak_height, seq2_path, seq_path, seq_region, start, window):
    # Resolve --alt list
    alt_bp_list = alt_bp if alt_bp is not None else []
    _alt_keywords = {'reverse', 'shuffle', 'random'}
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
        idx_to_base = {0: 'a', 1: 't', 2: 'c', 3: 'g', 4: 'n'}
        raw_alt_lower = raw_alt.lower()
        if raw_alt_lower == 'reverse':
            rc = reverse_complement(current_seq_region[rel_start:rel_end, :])
            alt_string = ''.join(idx_to_base[row.argmax()] for row in rc)
            print(f'[{label}] Using reverse-complement of {chr_name}:{start + rel_start}-{start + rel_end}')
            return alt_string
        if raw_alt_lower == 'shuffle':
            sub = current_seq_region[rel_start:rel_end, :].copy()
            idxs = np.arange(sub.shape[0])
            np.random.shuffle(idxs)
            sub = sub[idxs, :]
            alt_string = ''.join(idx_to_base[row.argmax()] for row in sub)
            print(f'[{label}] Using shuffled bases of {chr_name}:{start + rel_start}-{start + rel_end}')
            return alt_string
        if raw_alt_lower == 'random':
            bases = 'acgt'
            alt_string = ''.join(np.random.choice(list(bases)) for _ in range(rel_end - rel_start))
            print(f'[{label}] Using random bases for {chr_name}:{start + rel_start}-{start + rel_end}')
            return alt_string
        return raw_alt

    # Apply perturbations
    enformer_seq_active = False
    alphagenome_seq_active = False
    hierarchical_active = hierarchical_rad21_model is not None
    pending_track_perturbations = []

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
                alt_string = _resolve_alt_string(raw_alt, rel_start, rel_end, seq_region, label)
                print(f'[{label}] Queued {len(alt_string)} base(s) at '
                      f'{chr_name}:{deletion_start} (rel {rel_start}): {alt_string.upper()}')
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
                continue

            if knockout_mode == 'seq':
                deletion_start -= 1
                rel_start = deletion_start - start
                rel_end = rel_start + deletion_width
                alt_string = _resolve_alt_string(raw_alt, rel_start, rel_end, seq_region, 'seq')
                print(f'[seq] Substituting {len(alt_string)} base(s) at '
                      f'{chr_name}:{deletion_start} (rel {rel_start}): {alt_string.upper()}')
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
                continue

            if ko_data_type in input_track_names:
                ko_channel = input_track_names.index(ko_data_type)
            else:
                ko_channel = -1
            left_del_pad = None
            right_del_pad = None
            if knockout_mode in ('del', 'deletion', 'delete'):
                left_pad_bp = deletion_width // 2
                right_pad_bp = deletion_width - left_pad_bp
                left_pad_seq, left_pad_ctcf, left_pad_atac, left_pad_other = infer.load_region(chr_name,
                    start - left_pad_bp, seq_path, ctcf_path, atac_path, other_feats,
                    seq2_path=seq2_path, window=left_pad_bp, bigwig_log=bigwig_log_transform)
                left_del_pad = (left_pad_seq, left_pad_ctcf, left_pad_atac, left_pad_other)
                right_pad_seq, right_pad_ctcf, right_pad_atac, right_pad_other = infer.load_region(chr_name,
                    start + window + right_pad_bp, seq_path, ctcf_path, atac_path, other_feats,
                    seq2_path=seq2_path, window=right_pad_bp, bigwig_log=bigwig_log_transform)
                right_del_pad = (right_pad_seq, right_pad_ctcf, right_pad_atac, right_pad_other)
            pending_track_perturbations.append((
                deletion_start, deletion_width, ko_data_type, ko_channel,
                channel_offset, knockout_mode, ko_height, left_del_pad, right_del_pad,
            ))
    return seq_region, deletion_widths, pending_track_perturbations, enformer_seq_active, alphagenome_seq_active, hierarchical_active
