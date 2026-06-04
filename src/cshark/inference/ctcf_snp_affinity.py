import argparse
import gzip
import os
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.font_manager import FontProperties
from matplotlib.patches import PathPatch
from matplotlib.textpath import TextPath
from matplotlib.transforms import Affine2D


BASE_TO_INDEX = {'A': 0, 'T': 1, 'C': 2, 'G': 3, 'N': 4}
RC_TRANS = str.maketrans('ATCGN', 'TAGCN')
LOGO_BASES = ('A', 'C', 'G', 'T')
LOGO_COLORS = {
    'A': '#2ca02c',
    'C': '#1f77b4',
    'G': '#ff7f0e',
    'T': '#d62728',
}
LOGO_FONT = FontProperties(family='DejaVu Sans', weight='bold')


def parse_args():
    parser = argparse.ArgumentParser(
        description='Score approximate CTCF motif affinity changes for SNPs using a JASPAR PWM.',
        epilog=(
            'Provide one chromosome plus matching --positions and --alts lists. '
            'Positions are 1-based by default.'
        ),
    )
    parser.add_argument('--fasta', required=True,
                        help='Reference FASTA path (.fa, .fasta, optionally gzipped)')
    parser.add_argument('--chrom', required=True,
                        help='Chromosome name for all queried SNPs')
    parser.add_argument('--positions', nargs='+', required=True, type=int,
                        help='One or more SNP positions on --chrom')
    parser.add_argument('--alts', nargs='+', required=True,
                        help='Alternate bases matching --positions order')
    parser.add_argument('--out-prefix', required=True,
                        help='Output prefix for result table and plots')
    parser.add_argument('--scan-flank', type=int, default=20,
                        help='Flank size around each SNP to scan for the best reference motif hit (default: 20)')
    parser.add_argument('--jaspar-release', default='JASPAR2024',
                        help='JASPAR release to query (default: %(default)s)')
    parser.add_argument('--species', default='9606',
                        help='NCBI taxonomy id used when fetching the CTCF motif (default: %(default)s)')
    parser.add_argument('--tf-name', default='CTCF',
                        help='TF name to fetch from JASPAR (default: %(default)s)')
    parser.add_argument('--zero-based', action='store_true',
                        help='Interpret SNP positions as 0-based instead of 1-based')
    return parser.parse_args()


def open_text_maybe_gzip(path):
    return gzip.open(path, 'rt') if str(path).endswith('.gz') else open(path, 'rt')


def load_fasta(path):
    sequences = {}
    chrom = None
    chunks = []
    with open_text_maybe_gzip(path) as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            if line.startswith('>'):
                if chrom is not None:
                    sequences[chrom] = ''.join(chunks).upper()
                chrom = line[1:].split()[0]
                chunks = []
                continue
            chunks.append(line)
    if chrom is not None:
        sequences[chrom] = ''.join(chunks).upper()
    if not sequences:
        raise ValueError(f'No sequences found in FASTA: {path}')
    return sequences


def build_variant_table(chrom, positions, alts):
    if len(positions) != len(alts):
        raise ValueError(
            f'Expected the same number of --positions and --alts values, got {len(positions)} and {len(alts)}'
        )

    records = []
    for idx, (pos, alt) in enumerate(zip(positions, alts), start=1):
        alt = str(alt).upper()
        records.append({
            'id': f'variant_{idx}',
            'chrom': str(chrom),
            'pos': int(pos),
            'alt': alt,
        })
    return pd.DataFrame(records)


def reverse_complement_str(seq):
    return seq.translate(RC_TRANS)[::-1]


def fetch_pwm(jaspar_release='JASPAR2024', species='9606', tf_name='CTCF'):
    try:
        from pyjaspar import jaspardb
    except ImportError as exc:
        raise ImportError(
            'pyjaspar is required for PWM lookup. Install it with `pip install pyjaspar`.'
        ) from exc

    jdb_obj = jaspardb(release=jaspar_release)
    motifs = jdb_obj.fetch_motifs(
        collection=['CORE'],
        tf_name=tf_name,
        tax_group=['Vertebrates'],
        species=[species],
        all_versions=False,
    )
    if not motifs:
        raise ValueError(
            f'No motifs returned for tf_name={tf_name}, species={species}, release={jaspar_release}'
        )

    motif = motifs[0]
    matrix_dict = motif.counts.normalize()
    motif_len = len(matrix_dict['A'])
    pwm_rows = []
    for base in ['A', 'T', 'C', 'G', 'N']:
        if base in matrix_dict:
            pwm_rows.append(list(matrix_dict[base]))
        else:
            pwm_rows.append([0.0] * motif_len)
    pwm = np.array(pwm_rows, dtype=float).T
    return motif, pwm


def sequence_to_onehot(seq):
    onehot = np.zeros((len(seq), 5), dtype=float)
    for idx, base in enumerate(seq.upper()):
        base_idx = BASE_TO_INDEX.get(base, 4)
        onehot[idx, base_idx] = 1.0
    return onehot


def pwm_score(seq, pwm):
    if len(seq) != pwm.shape[0]:
        raise ValueError(f'Sequence length {len(seq)} does not match motif length {pwm.shape[0]}')
    encoded = sequence_to_onehot(seq)
    return float(np.dot(encoded.ravel(), pwm.ravel()) / pwm.shape[0])


def pwm_logo_heights(pwm):
    probabilities = np.column_stack([pwm[:, BASE_TO_INDEX[base]] for base in LOGO_BASES])
    row_sums = probabilities.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0.0] = 1.0
    probabilities = probabilities / row_sums
    with np.errstate(divide='ignore', invalid='ignore'):
        entropy_terms = np.where(probabilities > 0.0, probabilities * np.log2(probabilities), 0.0)
    info_content = np.clip(np.log2(len(LOGO_BASES)) + entropy_terms.sum(axis=1), 0.0, None)
    return probabilities * info_content[:, None]


def draw_logo_letter(ax, base, x0, y0, height):
    if height <= 0.0:
        return

    text = TextPath((0, 0), base, size=1, prop=LOGO_FONT)
    bounds = text.get_extents()
    width = 0.9
    scale_x = width / bounds.width
    scale_y = height / bounds.height
    transform = Affine2D().scale(scale_x, scale_y).translate(
        x0 + (1.0 - width) / 2.0 - bounds.x0 * scale_x,
        y0 - bounds.y0 * scale_y,
    )
    ax.add_patch(
        PathPatch(
            text,
            transform=transform + ax.transData,
            facecolor=LOGO_COLORS[base],
            edgecolor='none',
        )
    )


def draw_pwm_logo(ax, pwm):
    heights = pwm_logo_heights(pwm)
    max_height = 0.0
    for pos_idx in range(pwm.shape[0]):
        letters = sorted(
            ((LOGO_BASES[base_idx], heights[pos_idx, base_idx]) for base_idx in range(len(LOGO_BASES))),
            key=lambda item: item[1],
        )
        y_offset = 0.0
        for base, height in letters:
            draw_logo_letter(ax, base, pos_idx, y_offset, height)
            y_offset += height
        max_height = max(max_height, y_offset)

    tick_positions = np.arange(pwm.shape[0]) + 0.5
    ax.set_xlim(0, pwm.shape[0])
    ax.set_ylim(0.0, max(1.25, max_height * 1.25))
    ax.set_xticks(tick_positions)
    ax.set_xticklabels([str(idx) for idx in range(1, pwm.shape[0] + 1)])
    ax.set_ylabel('Bits')
    ax.set_xlabel('Motif position')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    return ax.get_ylim()[1]


def apply_alt_allele(reference_seq, pos0, alt, ref=None):
    alt = alt.upper()
    if any(base not in BASE_TO_INDEX for base in alt):
        raise ValueError(f'ALT contains unsupported bases: {alt}')

    if ref is None:
        ref = reference_seq[pos0:pos0 + len(alt)]
    else:
        ref = ref.upper()
        if any(base not in BASE_TO_INDEX for base in ref):
            raise ValueError(f'REF contains unsupported bases: {ref}')

    if len(ref) != len(alt):
        raise ValueError(
            'Only SNPs or same-length substitutions are supported so the motif window stays fixed.'
        )

    observed_ref = reference_seq[pos0:pos0 + len(ref)].upper()
    if observed_ref != ref:
        raise ValueError(f'REF mismatch at position {pos0}: expected {ref}, observed {observed_ref}')

    return reference_seq[:pos0] + alt + reference_seq[pos0 + len(ref):]


def find_best_reference_motif(reference_seq, positions0, pwm, scan_flank):
    motif_len = pwm.shape[0]
    if len(reference_seq) < motif_len:
        raise ValueError('Reference sequence is shorter than the motif length')

    if not positions0:
        raise ValueError('At least one SNP position is required to find a motif')

    min_pos0 = min(positions0)
    max_pos0 = max(positions0)

    search_start = max(0, min_pos0 - scan_flank - motif_len + 1)
    search_end = min(len(reference_seq) - motif_len, max_pos0 + scan_flank)
    if search_start > search_end:
        raise ValueError('Search region is outside the chromosome bounds')

    best_hit = None
    for motif_start in range(search_start, search_end + 1):
        motif_end = motif_start + motif_len
        if not all(motif_start <= pos0 < motif_end for pos0 in positions0):
            continue

        motif_seq = reference_seq[motif_start:motif_end]
        forward_score = pwm_score(motif_seq, pwm)
        reverse_seq = reverse_complement_str(motif_seq)
        reverse_score = pwm_score(reverse_seq, pwm)

        if reverse_score > forward_score:
            score = reverse_score
            strand = '-'
            oriented_seq = reverse_seq
        else:
            score = forward_score
            strand = '+'
            oriented_seq = motif_seq

        if best_hit is None or score > best_hit['ref_score']:
            best_hit = {
                'motif_start0': motif_start,
                'motif_end0': motif_end,
                'strand': strand,
                'ref_score': score,
                'ref_motif_seq': motif_seq,
                'ref_scored_seq': oriented_seq,
            }

    if best_hit is None:
        raise ValueError('No motif window overlapping all SNP positions could be evaluated')
    return best_hit


def apply_alt_alleles(reference_seq, variants, zero_based=False):
    alt_seq = reference_seq
    for variant in variants:
        pos = int(variant['pos'])
        pos0 = pos if zero_based else pos - 1
        ref = variant.get('ref')
        alt_seq = apply_alt_allele(alt_seq, pos0, variant['alt'], ref=ref)
    return alt_seq


def build_variant_annotations(variants, positions0, ref_scored_seq, alt_scored_seq,
                              motif_start0, motif_end0, strand, pwm):
    annotations = []
    motif_len = pwm.shape[0]

    for variant, pos0 in zip(variants, positions0):
        allele_len = len(variant['alt'])
        if strand == '-':
            scored_start = motif_end0 - (pos0 + allele_len)
        else:
            scored_start = pos0 - motif_start0

        for offset in range(allele_len):
            rel_pos = scored_start + offset
            ref_base = ref_scored_seq[rel_pos]
            alt_base = alt_scored_seq[rel_pos]
            genomic_pos = int(variant['pos']) + (offset if strand == '+' else allele_len - 1 - offset)
            ref_contribution = pwm[rel_pos, BASE_TO_INDEX[ref_base]] / motif_len
            alt_contribution = pwm[rel_pos, BASE_TO_INDEX[alt_base]] / motif_len
            annotations.append({
                'genomic_pos': genomic_pos,
                'motif_pos': rel_pos + 1,
                'ref_base': ref_base,
                'alt_base': alt_base,
                'delta_score': alt_contribution - ref_contribution,
            })

    annotations.sort(key=lambda item: item['motif_pos'])
    return annotations


def score_variants(reference_by_chrom, variants, pwm, scan_flank, zero_based=False, result_id=None):
    if not variants:
        raise ValueError('At least one variant is required for scoring')

    chrom = variants[0]['chrom']
    if chrom not in reference_by_chrom:
        raise ValueError(f'Chromosome {chrom} not found in FASTA')

    chrom_seq = reference_by_chrom[chrom]
    positions = []
    for variant in variants:
        if variant['chrom'] != chrom:
            raise ValueError('All variants in a combined score must be on the same chromosome')
        pos = int(variant['pos'])
        pos0 = pos if zero_based else pos - 1
        if pos0 < 0 or pos0 >= len(chrom_seq):
            raise ValueError(f'Position {pos} is outside chromosome {chrom} length {len(chrom_seq)}')
        positions.append(pos0)

    best_hit = find_best_reference_motif(chrom_seq, positions, pwm, scan_flank)
    alt_seq = apply_alt_alleles(chrom_seq, variants, zero_based=zero_based)

    motif_start0 = best_hit['motif_start0']
    motif_end0 = best_hit['motif_end0']
    ref_window = chrom_seq[motif_start0:motif_end0]
    alt_window = alt_seq[motif_start0:motif_end0]
    if best_hit['strand'] == '-':
        alt_scored_seq = reverse_complement_str(alt_window)
    else:
        alt_scored_seq = alt_window
    alt_score = pwm_score(alt_scored_seq, pwm)

    ref_alleles = []
    alt_alleles = []
    pos_labels = []
    for variant, pos0 in zip(variants, positions):
        alt_len = len(variant['alt'])
        rel_pos = pos0 - motif_start0
        ref_alleles.append(ref_window[rel_pos:rel_pos + alt_len])
        alt_alleles.append(variant['alt'])
        pos_labels.append(str(variant['pos']))

    if result_id is None:
        result_id = variants[0]['id'] if len(variants) == 1 else 'all_snps'

    variant_annotations = build_variant_annotations(
        variants,
        positions,
        best_hit['ref_scored_seq'],
        alt_scored_seq,
        motif_start0,
        motif_end0,
        best_hit['strand'],
        pwm,
    )

    return {
        'id': result_id,
        'analysis_type': 'single' if len(variants) == 1 else 'combined',
        'chrom': chrom,
        'pos': ','.join(pos_labels),
        'ref': ','.join(ref_alleles),
        'alt': ','.join(alt_alleles),
        'motif_start': motif_start0 + 1,
        'motif_end': motif_end0,
        'motif_strand': best_hit['strand'],
        'ref_motif_seq': ref_window,
        'alt_motif_seq': alt_window,
        'ref_scored_seq': best_hit['ref_scored_seq'],
        'alt_scored_seq': alt_scored_seq,
        'ref_score': best_hit['ref_score'],
        'alt_score': alt_score,
        'delta_score': alt_score - best_hit['ref_score'],
        'variant_annotations': variant_annotations,
    }


def plot_scores(results_df, out_path, tf_name):
    labels = [f'{row.id}\n{row.chrom}:{row.pos}' for row in results_df.itertuples(index=False)]
    x = np.arange(len(results_df))
    width = 0.38
    fig_width = max(10, 0.75 * len(results_df) + 4)
    fig, axes = plt.subplots(
        2, 1,
        figsize=(fig_width, 8),
        sharex=True,
        gridspec_kw={'height_ratios': [2, 1]},
    )

    axes[0].bar(x - width / 2, results_df['ref_score'], width=width,
                color='steelblue', label='Reference')
    axes[0].bar(x + width / 2, results_df['alt_score'], width=width,
                color='darkorange', label='Alternate')
    axes[0].set_ylabel('PWM correlation score')
    axes[0].set_title(f'{tf_name} motif affinity proxy')
    axes[0].legend(frameon=False)

    delta_colors = ['firebrick' if val < 0 else 'seagreen' for val in results_df['delta_score']]
    axes[1].axhline(0.0, color='black', linewidth=1, alpha=0.6)
    axes[1].bar(x, results_df['delta_score'], color=delta_colors)
    axes[1].set_ylabel('Alt - Ref')
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(labels, rotation=45, ha='right')

    fig.tight_layout()
    fig.savefig(out_path, dpi=300)
    plt.close(fig)


def plot_variant_motif_logos(results, pwm, out_path, tf_name):
    if not results:
        raise ValueError('At least one result is required for motif logo plotting')

    fig_width = max(10, 0.55 * pwm.shape[0] + 4)
    fig_height = max(3.5, 3.0 * len(results))
    fig, axes = plt.subplots(len(results), 1, figsize=(fig_width, fig_height), squeeze=False)
    axes = axes.ravel()

    for ax, record in zip(axes, results):
        draw_pwm_logo(ax, pwm)
        ax.set_title(
            f"{record['id']} | {record['chrom']}:{record['pos']} | strand {record['motif_strand']} | total Δ={record['delta_score']:+.3f}",
            fontsize=10,
            loc='left',
        )

        for annotation_idx, annotation in enumerate(record['variant_annotations']):
            x0 = annotation['motif_pos'] - 1
            x_center = x0 + 0.5
            delta = annotation['delta_score']
            color = 'firebrick' if delta < 0 else 'seagreen' if delta > 0 else 'dimgray'
            label_offset = annotation_idx % 2

            ax.axvspan(x0, x0 + 1, color=color, alpha=0.18, linewidth=0)
            ax.axvline(x_center, color=color, linestyle='--', linewidth=1.1, alpha=0.8)
            ax.text(
                x_center,
                1.02 + 0.08 * label_offset,
                str(annotation['genomic_pos']),
                color=color,
                fontsize=8,
                fontweight='bold',
                ha='center',
                va='bottom',
                transform=ax.get_xaxis_transform(),
                clip_on=False,
            )
            ax.text(
                x_center,
                -0.20 - 0.12 * label_offset,
                f"{annotation['ref_base']}>{annotation['alt_base']}\nΔ={delta:+.2f}",
                color=color,
                fontsize=8,
                ha='center',
                va='top',
                transform=ax.get_xaxis_transform(),
                clip_on=False,
            )

    fig.suptitle(
        f'{tf_name} motif logo with variant placement',
        fontsize=13,
        y=0.995,
    )
    fig.text(
        0.5,
        0.01,
        'Shaded motif positions mark altered bases; color reflects the local ALT-REF PWM contribution change.',
        ha='center',
        fontsize=9,
    )
    fig.subplots_adjust(top=0.9, bottom=0.12, hspace=1.0)
    fig.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close(fig)


def main():
    args = parse_args()
    out_prefix = Path(args.out_prefix)
    out_prefix.parent.mkdir(parents=True, exist_ok=True)

    reference_by_chrom = load_fasta(args.fasta)
    snps = build_variant_table(args.chrom, args.positions, args.alts)
    motif, pwm = fetch_pwm(
        jaspar_release=args.jaspar_release,
        species=args.species,
        tf_name=args.tf_name,
    )

    records = []
    failures = []
    for row in snps.itertuples(index=False):
        try:
            variant = {
                'id': row.id,
                'chrom': row.chrom,
                'pos': row.pos,
                'alt': row.alt,
            }
            records.append(
                score_variants(
                    reference_by_chrom,
                    [variant],
                    pwm,
                    scan_flank=args.scan_flank,
                    zero_based=args.zero_based,
                )
            )
        except Exception as exc:
            failures.append({'id': getattr(row, 'id', 'unknown'), 'error': str(exc)})

    if len(snps) > 1:
        try:
            combined_variants = snps.to_dict(orient='records')
            records.append(
                score_variants(
                    reference_by_chrom,
                    combined_variants,
                    pwm,
                    scan_flank=args.scan_flank,
                    zero_based=args.zero_based,
                    result_id='all_snps',
                )
            )
        except Exception as exc:
            failures.append({'id': 'all_snps', 'error': str(exc)})

    if not records:
        failure_msg = pd.DataFrame(failures).to_string(index=False) if failures else 'No successful records.'
        raise RuntimeError(f'Failed to score all variants.\n{failure_msg}')

    tabular_records = []
    for record in records:
        export_record = record.copy()
        export_record.pop('variant_annotations', None)
        tabular_records.append(export_record)

    results_df = pd.DataFrame(tabular_records)
    results_path = out_prefix.with_name(f'{out_prefix.name}_scores.tsv')
    plot_path = out_prefix.with_name(f'{out_prefix.name}_scores.png')
    motif_plot_path = out_prefix.with_name(f'{out_prefix.name}_motif_logo.png')
    results_df.to_csv(results_path, sep='\t', index=False)
    plot_scores(results_df, plot_path, motif.name)
    plot_variant_motif_logos(records, pwm, motif_plot_path, motif.name)

    print(f'Saved {len(results_df)} scored variants to {results_path}')
    print(f'Saved plot to {plot_path}')
    print(f'Saved motif logo plot to {motif_plot_path}')
    if failures:
        failure_path = out_prefix.with_name(f'{out_prefix.name}_failures.tsv')
        pd.DataFrame(failures).to_csv(failure_path, sep='\t', index=False)
        print(f'Skipped {len(failures)} variants; details written to {failure_path}')


if __name__ == '__main__':
    main()