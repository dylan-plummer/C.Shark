import argparse
import gzip
import os
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import numpy as np
import pandas as pd
from matplotlib.font_manager import FontProperties
from matplotlib.lines import Line2D
from matplotlib.patches import Patch, PathPatch
from matplotlib.textpath import TextPath
from matplotlib.transforms import Affine2D


# ---------------------------------------------------------------------------
# Publication style, ported from figures_lib/_palettes_HiC.r (theme_pub,
# the PUB_* size/linewidth constants, and the allele / RdBu palettes) so the
# motif logo matches the manuscript figures.
# ---------------------------------------------------------------------------
def _pick_pub_font_family():
    # HELVETICA_FAMILY <- "Arial" in the R lib; fall back to metric-compatible
    # Arial clones (Liberation/Nimbus Sans) when Arial itself is not installed.
    available = {f.name for f in fm.fontManager.ttflist}
    for name in ('Arial', 'Helvetica', 'Arimo', 'Liberation Sans', 'Nimbus Sans'):
        if name in available:
            return name
    return 'DejaVu Sans'


PUB_FONT_FAMILY = _pick_pub_font_family()
PUB_BASE_TEXT_SIZE = 7        # PUB_BASE_TEXT_SIZE
PUB_AXIS_TEXT_SIZE = 6        # PUB_AXIS_TEXT_SIZE
PUB_TITLE_TEXT_SIZE = 7       # PUB_TITLE_TEXT_SIZE
PUB_DENSE_TEXT_SIZE = 5       # annotation text (~PUB_COMPACT_TEXT_SIZE)
PUB_AXIS_LINEWIDTH = 0.5      # ~PUB_AXIS_LINEWIDTH (ggplot 0.2 unit)
PUB_REFERENCE_LINEWIDTH = 0.6

plt.rcParams.update({
    'font.family': PUB_FONT_FAMILY,
    'pdf.fonttype': 42,        # keep text editable in vector output
    'svg.fonttype': 'none',
    'axes.linewidth': PUB_AXIS_LINEWIDTH,
    'xtick.major.width': PUB_AXIS_LINEWIDTH,
    'ytick.major.width': PUB_AXIS_LINEWIDTH,
    'text.color': 'black',
    'axes.edgecolor': 'black',
    'axes.labelcolor': 'black',
    'xtick.color': 'black',
    'ytick.color': 'black',
})

BASE_TO_INDEX = {'A': 0, 'T': 1, 'C': 2, 'G': 3, 'N': 4}
RC_TRANS = str.maketrans('ATCGN', 'TAGCN')
LOGO_BASES = ('A', 'C', 'G', 'T')
# Nucleotide palette deliberately excludes red and blue so those two hues are
# reserved exclusively for the maternal (red) / paternal (blue) annotations.
LOGO_COLORS = {
    'A': '#2CA02C',   # green
    'C': '#E69F00',   # orange
    'G': '#7B3FA0',   # purple (DeltaLike hue from _palettes_HiC.r)
    'T': '#8C564B',   # brown
}
LOGO_FONT = FontProperties(family=PUB_FONT_FAMILY, weight='bold')
# ALLELE_SCATTER_PALETTE from the R lib: maternal = RdBu high (red),
# paternal = RdBu low (blue). Red/blue are reserved for the allele letters.
MATERNAL_COLOR = '#B2182B'
PATERNAL_COLOR = '#2166AC'
PUB_NEUTRAL_GREY = '#7F7F7F'
# Effect-direction scale (green = gain / Δ>0, red = loss / Δ<0), used for the
# SNP highlight band (light tint) and the rsID / Δ annotation text (saturated).
DELTA_POS_COLOR = '#2E8B57'        # seagreen
DELTA_NEG_COLOR = '#B22222'        # firebrick
HIGHLIGHT_POS_COLOR = '#C7E9C0'    # light green band (Δ>0)
HIGHLIGHT_NEG_COLOR = '#FBC9C4'    # light red band (Δ<0)
HIGHLIGHT_ZERO_COLOR = '#E0E0E0'   # light grey band (Δ==0)


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
    parser.add_argument('--rs-ids', nargs='+', default=None,
                        help='Optional rsID for each SNP, matching --positions order (e.g. rs12345)')
    parser.add_argument('--maternal-alleles', nargs='+', default=None,
                        help='Maternal allele base for each SNP, matching --positions order')
    parser.add_argument('--paternal-alleles', nargs='+', default=None,
                        help='Paternal allele base for each SNP, matching --positions order')
    parser.add_argument('--genome-build', default='hg38',
                        help='Genome assembly/build label shown on plots for --positions (default: %(default)s)')
    parser.add_argument('--out-prefix', required=True,
                        help='Output prefix for result table and plots')
    parser.add_argument('--plot-format', default='png', choices=['png', 'pdf', 'both'],
                        help='Image format for the plots: png (raster), pdf (vector, '
                             'editable text), or both (default: %(default)s)')
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


def build_variant_table(chrom, positions, alts, rs_ids=None, maternal_alleles=None, paternal_alleles=None):
    if len(positions) != len(alts):
        raise ValueError(
            f'Expected the same number of --positions and --alts values, got {len(positions)} and {len(alts)}'
        )
    if rs_ids is not None and len(rs_ids) != len(positions):
        raise ValueError(
            f'Expected the same number of --positions and --rs-ids values, got {len(positions)} and {len(rs_ids)}'
        )
    if maternal_alleles is not None and len(maternal_alleles) != len(positions):
        raise ValueError(
            f'Expected the same number of --positions and --maternal-alleles values, '
            f'got {len(positions)} and {len(maternal_alleles)}'
        )
    if paternal_alleles is not None and len(paternal_alleles) != len(positions):
        raise ValueError(
            f'Expected the same number of --positions and --paternal-alleles values, '
            f'got {len(positions)} and {len(paternal_alleles)}'
        )

    records = []
    for idx, (pos, alt) in enumerate(zip(positions, alts), start=1):
        alt = str(alt).upper()
        records.append({
            'id': f'variant_{idx}',
            'chrom': str(chrom),
            'pos': int(pos),
            'alt': alt,
            'rsid': str(rs_ids[idx - 1]) if rs_ids else '',
            'maternal_allele': str(maternal_alleles[idx - 1]).upper() if maternal_alleles else '',
            'paternal_allele': str(paternal_alleles[idx - 1]).upper() if paternal_alleles else '',
        })
    return pd.DataFrame(records)


def reverse_complement_str(seq):
    return seq.translate(RC_TRANS)[::-1]


def reverse_complement_pwm(pwm):
    bases_by_index = sorted(BASE_TO_INDEX, key=BASE_TO_INDEX.get)
    complement_columns = [BASE_TO_INDEX[base.translate(RC_TRANS)] for base in bases_by_index]
    return pwm[::-1, :][:, complement_columns]


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
    ax.set_ylabel('Bits', fontsize=PUB_AXIS_TEXT_SIZE)
    ax.set_xlabel('Motif position', fontsize=PUB_AXIS_TEXT_SIZE)
    # theme_pub look: classic L-shaped axes, thin black lines, small ticks.
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.tick_params(labelsize=PUB_AXIS_TEXT_SIZE, width=PUB_AXIS_LINEWIDTH, length=2.5, pad=2)
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


def apply_alleles(reference_seq, variants, allele_key, zero_based=False):
    seq = reference_seq
    for variant in variants:
        pos = int(variant['pos'])
        pos0 = pos if zero_based else pos - 1
        ref = variant.get('ref')
        seq = apply_alt_allele(seq, pos0, variant[allele_key], ref=ref)
    return seq


def apply_alt_alleles(reference_seq, variants, zero_based=False):
    return apply_alleles(reference_seq, variants, 'alt', zero_based=zero_based)


def build_variant_annotations(variants, positions0, ref_scored_seq, alt_scored_seq,
                              motif_start0, motif_end0, strand, pwm,
                              maternal_scored_seq=None, paternal_scored_seq=None):
    annotations = []
    motif_len = pwm.shape[0]
    has_parental = maternal_scored_seq is not None and paternal_scored_seq is not None

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
            annotation = {
                'genomic_pos': genomic_pos,
                'motif_pos': rel_pos + 1,
                'ref_base': ref_base,
                'alt_base': alt_base,
                'rsid': variant.get('rsid', ''),
                'delta_score': alt_contribution - ref_contribution,
            }

            if has_parental:
                maternal_base = maternal_scored_seq[rel_pos]
                paternal_base = paternal_scored_seq[rel_pos]
                maternal_contribution = pwm[rel_pos, BASE_TO_INDEX[maternal_base]] / motif_len
                paternal_contribution = pwm[rel_pos, BASE_TO_INDEX[paternal_base]] / motif_len
                annotation['maternal_base'] = maternal_base
                annotation['paternal_base'] = paternal_base
                annotation['parental_delta_score'] = paternal_contribution - maternal_contribution

            annotations.append(annotation)

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
    has_parental_alleles = all(
        variant.get('maternal_allele') and variant.get('paternal_allele') for variant in variants
    )
    if has_parental_alleles:
        maternal_seq = apply_alleles(chrom_seq, variants, 'maternal_allele', zero_based=zero_based)
        paternal_seq = apply_alleles(chrom_seq, variants, 'paternal_allele', zero_based=zero_based)

    motif_start0 = best_hit['motif_start0']
    motif_end0 = best_hit['motif_end0']
    ref_window = chrom_seq[motif_start0:motif_end0]
    alt_window = alt_seq[motif_start0:motif_end0]
    if best_hit['strand'] == '-':
        alt_scored_seq = reverse_complement_str(alt_window)
    else:
        alt_scored_seq = alt_window
    alt_score = pwm_score(alt_scored_seq, pwm)

    if has_parental_alleles:
        maternal_window = maternal_seq[motif_start0:motif_end0]
        paternal_window = paternal_seq[motif_start0:motif_end0]
        if best_hit['strand'] == '-':
            maternal_scored_seq = reverse_complement_str(maternal_window)
            paternal_scored_seq = reverse_complement_str(paternal_window)
        else:
            maternal_scored_seq = maternal_window
            paternal_scored_seq = paternal_window
        maternal_score = pwm_score(maternal_scored_seq, pwm)
        paternal_score = pwm_score(paternal_scored_seq, pwm)
    else:
        maternal_scored_seq = paternal_scored_seq = None
        maternal_score = paternal_score = None

    ref_alleles = []
    alt_alleles = []
    pos_labels = []
    rsid_labels = []
    maternal_labels = []
    paternal_labels = []
    for variant, pos0 in zip(variants, positions):
        alt_len = len(variant['alt'])
        rel_pos = pos0 - motif_start0
        ref_alleles.append(ref_window[rel_pos:rel_pos + alt_len])
        alt_alleles.append(variant['alt'])
        pos_labels.append(str(variant['pos']))
        rsid_labels.append(variant.get('rsid') or '')
        maternal_labels.append(variant.get('maternal_allele') or '')
        paternal_labels.append(variant.get('paternal_allele') or '')

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
        maternal_scored_seq=maternal_scored_seq,
        paternal_scored_seq=paternal_scored_seq,
    )

    return {
        'id': result_id,
        'analysis_type': 'single' if len(variants) == 1 else 'combined',
        'chrom': chrom,
        'pos': ','.join(pos_labels),
        'rsid': ','.join(rsid_labels),
        'ref': ','.join(ref_alleles),
        'alt': ','.join(alt_alleles),
        'maternal': ','.join(maternal_labels),
        'paternal': ','.join(paternal_labels),
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
        'maternal_score': maternal_score,
        'paternal_score': paternal_score,
        'parental_delta_score': (paternal_score - maternal_score) if has_parental_alleles else None,
        'variant_annotations': variant_annotations,
    }


def plot_scores(results_df, out_path, tf_name, genome_build):
    labels = [
        f'{row.id}' + (f' ({row.rsid})' if getattr(row, 'rsid', '') else '')
        + f'\n{row.chrom}:{row.pos} ({genome_build})'
        for row in results_df.itertuples(index=False)
    ]
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


def _delta_direction_color(delta):
    # Effect direction for the rsID / Δ annotation text: green gain / red loss.
    if delta > 0:
        return DELTA_POS_COLOR
    if delta < 0:
        return DELTA_NEG_COLOR
    return PUB_NEUTRAL_GREY


def _delta_band_color(delta):
    # Light tint of the direction colour for the SNP highlight band.
    if delta > 0:
        return HIGHLIGHT_POS_COLOR
    if delta < 0:
        return HIGHLIGHT_NEG_COLOR
    return HIGHLIGHT_ZERO_COLOR


def plot_variant_motif_logos(results, pwm, out_path, tf_name, genome_build):
    if not results:
        raise ValueError('At least one result is required for motif logo plotting')

    motif_len = pwm.shape[0]
    has_any_parental = any(record.get('parental_delta_score') is not None for record in results)

    # Publication-sized figure (theme_pub / PUB_* conventions from _palettes_HiC.r).
    # Logo panel deliberately kept short (half the previous height). The inter-panel
    # gap is sized in fixed inches (not relative to the short axes) so each panel's
    # allele annotations + x-label clear the next panel's title.
    n_panels = len(results)
    fig_width = max(4.2, 0.26 * motif_len + 1.6)
    panel_height_inches = 0.95
    inter_gap_inches = 1.3 if has_any_parental else 1.05
    top_reserved_inches = 0.8
    bottom_reserved_inches = 0.95 if has_any_parental else 0.8
    fig_height = (
        panel_height_inches * n_panels
        + inter_gap_inches * (n_panels - 1)
        + top_reserved_inches
        + bottom_reserved_inches
    )
    fig, axes = plt.subplots(n_panels, 1, figsize=(fig_width, fig_height), squeeze=False)
    axes = axes.ravel()

    rc_pwm = reverse_complement_pwm(pwm)

    for ax, record in zip(axes, results):
        is_minus_strand = record['motif_strand'] == '-'
        display_pwm = rc_pwm if is_minus_strand else pwm
        draw_pwm_logo(ax, display_pwm)
        rsid_suffix = f" ({record['rsid']})" if record.get('rsid') else ''
        if record.get('parental_delta_score') is not None:
            delta_title = f"Δ(pat−mat)={record['parental_delta_score']:+.3f}"
        else:
            delta_title = f"Δ={record['delta_score']:+.3f}"
        ax.set_title(
            f"{record['id']}{rsid_suffix}  |  {genome_build} {record['chrom']}:{record['pos']}  |  strand {record['motif_strand']}  |  total {delta_title}",
            fontsize=PUB_BASE_TEXT_SIZE,
            loc='left',
            pad=26,
        )

        for annotation_idx, annotation in enumerate(record['variant_annotations']):
            has_parental = 'parental_delta_score' in annotation
            if is_minus_strand:
                display_pos = motif_len - annotation['motif_pos'] + 1
                ref_base = annotation['ref_base'].translate(RC_TRANS)
                alt_base = annotation['alt_base'].translate(RC_TRANS)
                if has_parental:
                    maternal_base = annotation['maternal_base'].translate(RC_TRANS)
                    paternal_base = annotation['paternal_base'].translate(RC_TRANS)
            else:
                display_pos = annotation['motif_pos']
                ref_base = annotation['ref_base']
                alt_base = annotation['alt_base']
                if has_parental:
                    maternal_base = annotation['maternal_base']
                    paternal_base = annotation['paternal_base']

            x0 = display_pos - 1
            x_center = x0 + 0.5
            delta = annotation['parental_delta_score'] if has_parental else annotation['delta_score']
            color = _delta_direction_color(delta)
            label_offset = annotation_idx % 2
            pos_label = (
                f"{annotation['rsid']} · {annotation['genomic_pos']}"
                if annotation.get('rsid') else str(annotation['genomic_pos'])
            )

            # Mark the SNP column with a light diverging band (green Δ>0 / red Δ<0),
            # no line; drawn behind the logo letters (zorder=0) so they stay crisp.
            ax.axvspan(x0, x0 + 1, color=_delta_band_color(delta), alpha=0.7,
                       linewidth=0, zorder=0)
            # Offsets are in fixed points (not axes-fraction) so they stay a small,
            # constant distance from the axis regardless of per-panel axes height.
            ax.annotate(
                pos_label,
                xy=(x_center, 1.0),
                xycoords=ax.get_xaxis_transform(),
                xytext=(0, 4 + 8 * label_offset),
                textcoords='offset points',
                color=color,
                fontsize=PUB_DENSE_TEXT_SIZE,
                fontweight='bold',
                ha='center',
                va='bottom',
                clip_on=False,
            )

            label_y = -24 - 12 * label_offset
            line_step = 8
            if has_parental:
                # Two rows: small grey mat/pat tag + coloured allele letter,
                # reading maternal (red) over paternal (blue); Δ below.
                for row, (tag, base_letter, base_color) in enumerate((
                    ('mat', maternal_base, MATERNAL_COLOR),
                    ('pat', paternal_base, PATERNAL_COLOR),
                )):
                    row_y = label_y - row * line_step
                    ax.annotate(
                        tag,
                        xy=(x_center, 0.0),
                        xycoords=ax.get_xaxis_transform(),
                        xytext=(-1, row_y),
                        textcoords='offset points',
                        color=PUB_NEUTRAL_GREY,
                        fontsize=PUB_DENSE_TEXT_SIZE,
                        ha='right',
                        va='top',
                        clip_on=False,
                    )
                    ax.annotate(
                        base_letter,
                        xy=(x_center, 0.0),
                        xycoords=ax.get_xaxis_transform(),
                        xytext=(3, row_y),
                        textcoords='offset points',
                        color=base_color,
                        fontsize=PUB_DENSE_TEXT_SIZE,
                        fontweight='bold',
                        ha='left',
                        va='top',
                        clip_on=False,
                    )
                ax.annotate(
                    f"Δ {delta:+.2f}",
                    xy=(x_center, 0.0),
                    xycoords=ax.get_xaxis_transform(),
                    xytext=(0, label_y - 2 * line_step),
                    textcoords='offset points',
                    color=color,
                    fontsize=PUB_DENSE_TEXT_SIZE,
                    ha='center',
                    va='top',
                    clip_on=False,
                )
            else:
                ax.annotate(
                    f"{ref_base}>{alt_base}\n{delta:+.2f}",
                    xy=(x_center, 0.0),
                    xycoords=ax.get_xaxis_transform(),
                    xytext=(0, label_y),
                    textcoords='offset points',
                    color=color,
                    fontsize=PUB_DENSE_TEXT_SIZE,
                    ha='center',
                    va='top',
                    clip_on=False,
                )

    fig.suptitle(
        f'{tf_name} motif logo',
        fontsize=PUB_TITLE_TEXT_SIZE,
        y=1.0,
        va='top',
    )

    # Legend, mirroring the R "legend system" style: small colored keys, no frame.
    band_handles = [
        Patch(facecolor=HIGHLIGHT_POS_COLOR, edgecolor='none', label='Δ>0 (green band)'),
        Patch(facecolor=HIGHLIGHT_NEG_COLOR, edgecolor='none', label='Δ<0 (red band)'),
    ]
    if has_any_parental:
        handles = [
            Patch(facecolor=MATERNAL_COLOR, edgecolor='none', label='Maternal allele'),
            Patch(facecolor=PATERNAL_COLOR, edgecolor='none', label='Paternal allele'),
        ] + band_handles
        caption = ('Alleles below each panel read maternal → paternal; '
                   'Δ = paternal − maternal PWM contribution; band colours the SNP by Δ sign. '
                   '"−"-strand loci are reverse-complemented to the genomic + strand.')
    else:
        handles = band_handles
        caption = ('Alleles below each panel read ref>alt; Δ = alt − ref PWM contribution; '
                   'band colours the SNP by Δ sign. '
                   '"−"-strand loci are reverse-complemented to the genomic + strand.')

    # Reserve fixed physical space so nothing collides regardless of panel count.
    # hspace is expressed relative to the (short) axes height so that the absolute
    # inter-panel gap stays at inter_gap_inches.
    top_frac = 1 - top_reserved_inches / fig_height
    bottom_frac = bottom_reserved_inches / fig_height
    fig.subplots_adjust(top=top_frac, bottom=bottom_frac,
                        hspace=inter_gap_inches / panel_height_inches)

    fig.legend(
        handles=handles,
        loc='upper center',
        bbox_to_anchor=(0.5, -0.08 / fig_height),
        ncol=len(handles),
        frameon=False,
        fontsize=PUB_AXIS_TEXT_SIZE,
        handlelength=1.1,
        handleheight=1.1,
        columnspacing=1.6,
        handletextpad=0.5,
    )
    fig.text(
        0.5,
        -0.30 / fig_height,
        caption,
        ha='center',
        va='top',
        fontsize=PUB_DENSE_TEXT_SIZE,
        color='black',
    )
    fig.savefig(out_path, dpi=600, bbox_inches='tight')
    plt.close(fig)


def main():
    args = parse_args()
    out_prefix = Path(args.out_prefix)
    out_prefix.parent.mkdir(parents=True, exist_ok=True)

    reference_by_chrom = load_fasta(args.fasta)
    snps = build_variant_table(
        args.chrom, args.positions, args.alts, rs_ids=args.rs_ids,
        maternal_alleles=args.maternal_alleles, paternal_alleles=args.paternal_alleles,
    )
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
                'rsid': row.rsid,
                'maternal_allele': row.maternal_allele,
                'paternal_allele': row.paternal_allele,
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
    results_df.to_csv(results_path, sep='\t', index=False)
    print(f'Saved {len(results_df)} scored variants to {results_path}')

    extensions = ['.png', '.pdf'] if args.plot_format == 'both' else [f'.{args.plot_format}']
    for ext in extensions:
        plot_path = out_prefix.with_name(f'{out_prefix.name}_scores{ext}')
        motif_plot_path = out_prefix.with_name(f'{out_prefix.name}_motif_logo{ext}')
        plot_scores(results_df, plot_path, motif.name, args.genome_build)
        plot_variant_motif_logos(records, pwm, motif_plot_path, motif.name, args.genome_build)
        print(f'Saved plot to {plot_path}')
        print(f'Saved motif logo plot to {motif_plot_path}')
    if failures:
        failure_path = out_prefix.with_name(f'{out_prefix.name}_failures.tsv')
        pd.DataFrame(failures).to_csv(failure_path, sep='\t', index=False)
        print(f'Skipped {len(failures)} variants; details written to {failure_path}')


if __name__ == '__main__':
    main()