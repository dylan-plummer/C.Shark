"""
AlphaGenome full-chromosome Hi-C contact map prediction.

Sliding-window prediction analogous to akita_predict.py but using the
AlphaGenome PyTorch port, which takes 1 Mb (1,048,576 bp) DNA-only input
and produces 64×64 contact maps at 2,048 bp resolution (per window).

Overlapping windows are averaged and written to a cooler file.
Optionally produces pyGenomeTracks plots.
"""

import os
import argparse
import sys

import numpy as np
import pandas as pd
import cooler
import torch
import matplotlib.pyplot as plt
import seaborn as sns
import pyfaidx
from tqdm import tqdm
from scipy.sparse import coo_matrix

from alphagenome_pytorch import AlphaGenome
from alphagenome_pytorch.utils.sequence import sequence_to_onehot_tensor

# ── Constants ──────────────────────────────────────────────────────────
WINDOW = 1_048_576              # 1 Mb input window
CONTACT_BIN_SIZE = 2048         # bp per contact-map bin
CONTACT_HEADS = 28              # number of contact-map output tracks
FONT_SIZE = 15
PLOT_WIDTH = 17
TRACK_LABEL_FRACTION = 0.13


# ── Cooler helpers ─────────────────────────────────────────────────────

def write_cooler(mat, chr_name, start, res, out_file='tmp/tmp.cool',
                 window=WINDOW):
    """Write a square numpy matrix as a single-resolution cooler file."""
    os.makedirs(os.path.dirname(out_file) or '.', exist_ok=True)
    bin_range = np.arange(0, start + window + res, res)
    bins = pd.DataFrame({
        'chrom': chr_name,
        'start': bin_range.astype(int),
        'end': (bin_range + res).astype(int),
    })
    start_offset = int(start / res)
    sparse = coo_matrix(np.triu(mat), dtype=np.float32)
    pixels = pd.DataFrame({
        'bin1_id': sparse.row + start_offset,
        'bin2_id': sparse.col + start_offset,
        'count': sparse.data,
    })
    cooler.create_cooler(out_file, bins, pixels,
                         dtypes={'count': np.float32})


def pixels_to_dense(res_df, res):
    """
    Reconstruct a dense contact matrix from the aggregated pixel DataFrame.
    Returns (matrix, mat_start) where mat_start is the genomic start of bin 0.
    """
    df = res_df.dropna(subset=['start1', 'start2']).copy()
    if len(df) == 0:
        return np.zeros((1, 1)), 0
    min_start = int(min(df['start1'].min(), df['start2'].min()))
    max_end   = int(max(df['end1'].max(), df['end2'].max()))
    n_bins = int((max_end - min_start) / res)
    mat = np.zeros((n_bins, n_bins), dtype=np.float32)
    rows = ((df['start1'].values - min_start) / res).astype(int)
    cols = ((df['start2'].values - min_start) / res).astype(int)
    vals = df['pred'].values.astype(np.float32)
    # clamp to valid range
    valid = (rows >= 0) & (rows < n_bins) & (cols >= 0) & (cols < n_bins)
    rows, cols, vals = rows[valid], cols[valid], vals[valid]
    mat[rows, cols] = vals
    # symmetrise
    mat = mat + mat.T - np.diag(np.diag(mat))
    return mat, min_start


# ── Arcs (links) for pyGenomeTracks ───────────────────────────────────

def write_arcs(mat, chr_name, start, res, out_file, quantile=0.99,
               region_start=None, region_end=None):
    """
    Write top-scoring pixels as a links/arcs BED file for pyGenomeTracks.
    """
    nonzero = mat[mat != 0]
    if len(nonzero) == 0:
        cutoff = 0.0
    else:
        cutoff = np.quantile(nonzero, quantile)
    if region_start is None:
        region_start = start
    if region_end is None:
        region_end = start + mat.shape[0] * res
    with open(out_file, 'w') as f:
        for i in range(mat.shape[0]):
            for j in range(i, mat.shape[1]):
                val = mat[i, j]
                if val <= cutoff:
                    continue
                si = i * res + start
                ei = si + res
                sj = j * res + start
                ej = sj + res
                if si >= region_start and ej <= region_end:
                    f.write(f'{chr_name}\t{si}\t{ei}\t'
                            f'{chr_name}\t{sj}\t{ej}\t{val}\n')


# ── pyGenomeTracks INI generation ─────────────────────────────────────

def write_tracks_ini(cool_file, chr_name, assembly, out_ini,
                     genes_gtf=None, min_val=0.0, max_val=None,
                     arcs_file=None, title='AlphaGenome pred'):
    """
    Write a pyGenomeTracks .ini file showing the cooler heatmap,
    gene annotations, and optionally arcs.
    """
    with open(out_ini, 'w') as f:
        f.write('[x-axis]\nwhere = top\n\n')

        if genes_gtf and os.path.exists(genes_gtf):
            f.write('[Genes]\n')
            f.write(f'file = {genes_gtf}\n')
            f.write('title = Genes\n')
            f.write('prefered_name = gene_name\n')
            f.write('height = 4\n')
            f.write('merge_transcripts = True\n')
            f.write('labels = True\n')
            f.write('max_labels = 100\n')
            f.write('all_labels_inside = True\n')
            f.write('style = UCSC\n')
            f.write('gene_rows = 10\n')
            f.write('file_type = gtf\n')
            f.write('fontsize = 10\n\n')

        f.write(f'[{title}]\n')
        f.write(f'file = {cool_file}\n')
        f.write(f'min_value = {min_val}\n')
        if max_val is not None:
            f.write(f'max_value = {max_val}\n')
        f.write('colormap = [ (1.0, 1.0, 1.0), (1.0, 0.92, 0.92),'
                '(1.0, 0.8, 0.8),(1.0, 0.6, 0.6),'
                '(1.0, 0.4, 0.4),(1.0, 0.294, 0.294)]\n')
        f.write('file_type = hic_matrix_square\n\n')

        if arcs_file and os.path.exists(arcs_file):
            f.write('[arcs]\n')
            f.write(f'file = {arcs_file}\n')
            f.write('line_width = 1\n')
            f.write('color = red\n')
            f.write('height = 3\n')
            f.write('file_type = links\n')
            f.write('links_type = arcs\n')
            f.write('orientation = inverted\n\n')


# ── Core prediction wrapper ───────────────────────────────────────────

def alphagenome_predict(seq_str, model, organism_index=0, contact_head=0,
                        device='cpu'):
    """
    Run AlphaGenome on a 1 Mb DNA string and return the contact matrix
    for the requested head.

    Returns:
        np.ndarray of shape (S, S) – predicted contact matrix.
    """
    dna_onehot = sequence_to_onehot_tensor(seq_str).unsqueeze(0).to(device)
    preds = model.predict(dna_onehot, organism_index=organism_index)
    # pair_activations: (B, S, S, 28)  channels_last
    contacts = preds['pair_activations']
    mat = contacts[0, :, :, contact_head].float().cpu().numpy()
    # undo log transform 
    mat = np.exp(mat)
    return mat


# ── Main ───────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description='AlphaGenome sliding-window full-chromosome Hi-C prediction.'
    )
    # Required
    parser.add_argument('--genome', required=True,
                        help='Path to reference genome FASTA (.fa) file')
    parser.add_argument('--chr', dest='chr_name', required=True,
                        help='Chromosome name (e.g. chr1)')
    parser.add_argument('--model', dest='model_path', required=True,
                        help='Path to AlphaGenome .safetensors checkpoint')
    parser.add_argument('--out-file', dest='out_file', required=True,
                        help='Output .tsv path for contact predictions')

    # Optional
    parser.add_argument('--out', dest='output_path', default='outputs',
                        help='Directory for auxiliary outputs (plots, etc.)')
    parser.add_argument('--outname', default='',
                        help='Output prefix for file names')
    parser.add_argument('--celltype', default='alphagenome',
                        help='Label used in output file naming')
    parser.add_argument('--assembly', default='hg38',
                        help='Genome assembly (hg19, hg38, mm10)')
    parser.add_argument('--organism', dest='organism_index', type=int,
                        default=0, help='Organism index: 0=human, 1=mouse')
    parser.add_argument('--contact-head', dest='contact_head', type=int,
                        default=0,
                        help=f'Contact map head index (0-{CONTACT_HEADS-1})')
    parser.add_argument('--region', default=None,
                        help='Restrict to region, e.g. chr1:10000000-20000000')
    parser.add_argument('--n-overlap-pred', dest='n_overlap_preds', type=int,
                        default=2,
                        help='Overlapping predictions per pixel '
                             '(controls step size)')
    parser.add_argument('--device', default=None,
                        help='Torch device (default: auto)')
    parser.add_argument('--no-plots', dest='no_plots', action='store_true',
                        help='Skip pyGenomeTracks / matplotlib plots')
    parser.add_argument('--silent', action='store_true',
                        help='Suppress pyGenomeTracks stdout/stderr')

    # Plot colour-scale
    parser.add_argument('--min-val', dest='min_val', type=float, default=0.0,
                        help='Min colour-scale value for Hi-C plot')
    parser.add_argument('--max-val', dest='max_val', type=float, default=None,
                        help='Max colour-scale value for Hi-C plot')

    # Genes GTF (for pyGenomeTracks)
    parser.add_argument('--genes-gtf', dest='genes_gtf', default=None,
                        help='Path to genes GTF for pyGenomeTracks '
                             '(auto-detected from assembly if omitted)')

    args = parser.parse_args(args=None if sys.argv[1:] else ['--help'])

    # ── Device ─────────────────────────────────────────────────────────
    if args.device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    else:
        device = args.device
    print(f'Using device: {device}')

    # ── Load model ─────────────────────────────────────────────────────
    print(f'Loading AlphaGenome from {args.model_path} ...')
    model = AlphaGenome.from_pretrained(args.model_path)
    model = model.to(device).eval()
    total_params = sum(p.numel() for p in model.parameters())
    print(f'Model loaded - {total_params / 1e6:.1f}M parameters')

    # ── Open genome ────────────────────────────────────────────────────
    genome = pyfaidx.Fasta(args.genome)
    chr_name = args.chr_name
    chr_length = len(genome[chr_name])
    print(f'{chr_name} length: {chr_length:,} bp')

    # ── Sliding-window coordinates ─────────────────────────────────────
    step_size = int(WINDOW / args.n_overlap_preds)
    res = CONTACT_BIN_SIZE

    region_start = None
    region_end = None
    if args.region is not None:
        region_str = args.region
        if ':' in region_str:
            _, coords = region_str.split(':')
        else:
            coords = region_str
        region_start, region_end = (int(x) for x in coords.split('-'))
        starts = np.arange(
            max(0, region_start - step_size),
            min(region_end + step_size, chr_length - WINDOW),
            step_size,
        )
    else:
        starts = np.arange(0, chr_length - WINDOW, step_size)
        args.no_plots = True  # skip plots for sub-regions

    ends = starts + WINDOW
    print(f'Step size: {step_size:,} bp  |  {len(starts)} windows')

    # ── Predict ────────────────────────────────────────────────────────
    os.makedirs('tmp', exist_ok=True)
    results = {'a1': [], 'a2': [], 'pred': []}
    bins_list = []

    for start, end in tqdm(zip(starts, ends), desc='Predicting',
                           total=len(starts)):
        seq_str = str(genome[chr_name][start:end])

        mat = alphagenome_predict(
            seq_str, model,
            organism_index=args.organism_index,
            contact_head=args.contact_head,
            device=device,
        )

        # Write per-window cooler to extract bin/pixel DataFrames
        write_cooler(mat, chr_name, start, res=res)
        c = cooler.Cooler('tmp/tmp.cool')
        pixels = c.pixels()[:].rename(columns={'count': 'pred'})
        results['a1'].extend(pixels['bin1_id'].tolist())
        results['a2'].extend(pixels['bin2_id'].tolist())
        results['pred'].extend(pixels['pred'].tolist())
        bins_list.append(c.bins()[:])

    genome.close()

    # ── Aggregate overlapping windows ──────────────────────────────────
    res_df = (pd.DataFrame(results)
              .groupby(['a1', 'a2']).mean().reset_index())
    res_df['a1'] = 'A_' + res_df['a1'].astype(str)
    res_df['a2'] = 'A_' + res_df['a2'].astype(str)

    bins_df = (pd.concat(bins_list, ignore_index=True)
               .drop_duplicates().reset_index(drop=True))
    bins_df['bin_id'] = 'A_' + bins_df.index.astype(str)

    chr_map   = bins_df.set_index('bin_id')['chrom'].to_dict()
    start_map = bins_df.set_index('bin_id')['start'].to_dict()
    end_map   = bins_df.set_index('bin_id')['end'].to_dict()

    res_df['chrom1'] = res_df['a1'].map(chr_map)
    res_df['chrom2'] = res_df['a2'].map(chr_map)
    res_df['start1'] = res_df['a1'].map(start_map)
    res_df['start2'] = res_df['a2'].map(start_map)
    res_df['end1']   = res_df['a1'].map(end_map)
    res_df['end2']   = res_df['a2'].map(end_map)
    res_df = res_df[['chrom1', 'start1', 'end1', 'a1',
                      'chrom2', 'start2', 'end2', 'a2', 'pred']]

    # Clip to requested region
    if region_start is not None:
        res_df = res_df[
            (res_df['start1'] >= region_start) &
            (res_df['end1']   <= region_end) &
            (res_df['start2'] >= region_start) &
            (res_df['end2']   <= region_end)
        ]
        bins_df = bins_df[
            (bins_df['start'] >= region_start) &
            (bins_df['end']   <= region_end)
        ]

    # ── Write TSV output ───────────────────────────────────────────────
    out_file = args.out_file
    if not out_file.endswith('.tsv'):
        out_file += '.tsv'
    try:
        os.makedirs(os.path.dirname(out_file), exist_ok=True)
    except Exception:
        pass

    res_df['pred'] = res_df['pred'].round(4)
    res_df.fillna(0, inplace=True)
    res_df.to_csv(out_file, sep='\t', header=True, index=False)
    bins_df.to_csv(out_file.replace('.tsv', '_bins.tsv'),
                   sep='\t', header=False, index=False)
    print(f'Wrote {len(res_df)} pixels -> {out_file}')

    # ── Plots ──────────────────────────────────────────────────────────
    if args.no_plots:
        return
    # ── Reconstruct dense matrix and write cooler ──────────────────────
    dense_mat, mat_start = pixels_to_dense(res_df, res)
    mat_window = dense_mat.shape[0] * res
    cool_file = out_file.replace('.tsv', '.cool')
    write_cooler(dense_mat, chr_name, mat_start, res=res,
                 out_file=cool_file, window=mat_window)
    print(f'Cooler -> {cool_file}')

    

    os.makedirs(args.output_path, exist_ok=True)
    prefix = f'{args.outname}_' if args.outname else ''
    celltype = args.celltype
    assembly = args.assembly

    # --- Histogram QC ---
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.hist(res_df['pred'].values, bins=100)
    ax.set_xlabel('Predicted contact score')
    ax.set_ylabel('Count')
    ax.set_title(f'AlphaGenome head {args.contact_head} - {chr_name}')
    hist_path = os.path.join(
        args.output_path,
        f'{prefix}{celltype}_{chr_name}_head{args.contact_head}_hist.png',
    )
    plt.savefig(hist_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f'Histogram -> {hist_path}')

    # --- pyGenomeTracks ---
    # Try to locate a genes GTF
    genes_gtf = args.genes_gtf
    if genes_gtf is None:
        # Convention: cshark_data/data/<assembly>/<assembly>_genes.gtf
        candidate = os.path.join('cshark_data', 'data', assembly,
                                 f'{assembly}_genes.gtf')
        if os.path.exists(candidate):
            genes_gtf = candidate

    # Write arcs file from the dense matrix
    arcs_file = 'tmp/arcs_alphagenome.bed'
    try:
        write_arcs(dense_mat, chr_name, mat_start, res, arcs_file,
                   quantile=0.99,
                   region_start=region_start, region_end=region_end)
    except Exception as e:
        print(f'Warning: could not write arcs file: {e}')
        arcs_file = None

    # Write tracks INI
    ini_file = 'tmp/tmp_tracks_alphagenome.ini'
    write_tracks_ini(cool_file, chr_name, assembly, ini_file,
                     genes_gtf=genes_gtf,
                     min_val=args.min_val, max_val=args.max_val,
                     arcs_file=arcs_file,
                     title='AlphaGenome pred')

    # Run pyGenomeTracks
    if region_start is not None and region_end is not None:
        plot_region = f'{chr_name}:{region_start}-{region_end}'
    else:
        # default: first 5 Mb or whole chromosome
        plot_end = min(chr_length, 5_000_000)
        plot_region = f'{chr_name}:0-{plot_end}'

    pgt_out = os.path.join(
        args.output_path,
        f'{prefix}{celltype}_{chr_name}_head{args.contact_head}_tracks.png',
    )
    tracks_cmd = (
        f'pyGenomeTracks --tracks {ini_file} '
        f'-o {pgt_out} '
        f'--region {plot_region} '
        f'--fontSize {FONT_SIZE} '
        f'--plotWidth {PLOT_WIDTH} '
        f'--trackLabelFraction {TRACK_LABEL_FRACTION}'
    )
    if args.silent:
        tracks_cmd += ' > /dev/null 2>&1'
    print(f'Running: {tracks_cmd}')
    os.system(tracks_cmd)
    print(f'Tracks plot -> {pgt_out}')


if __name__ == '__main__':
    main()
