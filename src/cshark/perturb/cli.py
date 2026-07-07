"""Command-line entry point for the perturbation engine.

Thin: build the argparse parser (ported verbatim from the original
``perturb.py main()`` argparse block, lines 59-204), resolve ``--device``
(folding in the old ``perturb_cpu.py``), construct a ``PerturbConfig``, build the
model + secondary predictors via ``models.registry.build_model``, and dispatch
to ``scopes.single_locus`` (``--start`` given) or ``scopes.full_chrom``.

IMPORTANT: ``--device`` must be resolved BEFORE importing any torch-backed
module, so ``CUDA_VISIBLE_DEVICES`` is set while CUDA is still uninitialised.
The heavy imports (models / scopes, which pull in torch via inference_utils)
are therefore deferred into ``main`` *after* ``resolve_device``.
"""
import argparse
import os
import sys

from cshark.perturb.config import PerturbConfig


class ParseKwargs(argparse.Action):
    """``--bigwigs ctcf=a.bw atac=b.bw`` -> {'ctcf': 'a.bw', 'atac': 'b.bw'}.

    Verbatim port of the helper action from the original perturb.py.
    """
    def __call__(self, parser, namespace, values, option_string=None):
        setattr(namespace, self.dest, dict())
        for value in values:
            key, value = value.split('=')
            getattr(namespace, self.dest)[key] = value


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description='C.Shark Editing / Perturbation Module.')

    parser.add_argument('--out', dest='output_path', default='outputs',
                        help='output path for storing results (default: %(default)s)')
    parser.add_argument('--celltype', dest='celltype',
                        help='Sample cell type for prediction, used for output separation', required=True)
    parser.add_argument('--assembly', dest='assembly', default='hg19',
                        help='Genome assembly version (hg19, hg38, mm10)')
    parser.add_argument('--outname', dest='outname', default='',
                        help='Output prefix for saving plots and predictions')
    parser.add_argument('--chr', dest='chr_name', help='Chromosome for prediction', required=True)
    parser.add_argument('--start', dest='start', type=int,
                        help='Starting point for prediction (single-locus mode)', required=False)
    parser.add_argument('--model', dest='model_path', help='Path to the model checkpoint', required=True)

    # NEW: replaces the separate perturb_cpu.py (which hard-set CUDA_VISIBLE_DEVICES=-1).
    parser.add_argument('--device', dest='device', default='auto', choices=['auto', 'cpu', 'cuda'],
                        help="Compute device: auto (default), cpu (sets CUDA_VISIBLE_DEVICES=-1), or cuda")

    parser.add_argument('--resolution', dest='resolution', type=int, default=8192,
                        help='Resolution (bp) of output Hi-C matrix')
    parser.add_argument('--resolution-1d', dest='resolution_1d', type=int, default=256,
                        help='Resolution (bp) of output 1D tracks')
    parser.add_argument('--matrix-size', dest='mat_size', type=int, default=256,
                        help='Size of output Hi-C matrix')
    parser.add_argument('--latent_size', dest='mid_hidden', type=int, default=256, required=False, help='')
    parser.add_argument('--seq-filter-size', dest='seq_filter_size', type=int, default=3, required=False,
                        help='Size of the 1D conv filter for sequence data')
    parser.add_argument('--no-recon', dest='recon_1d', action='store_false',
                        help='Whether to reconstruct 1D tracks from full features or from sequence only')
    parser.add_argument('--no-hic-log-transform', dest='hic_log_transform', action='store_false',
                        help='Whether to apply log transformation to Hi-C matrices')
    parser.add_argument('--no-bigwig-log-transform', dest='bigwig_log_transform', action='store_false',
                        help='Whether to apply log transformation to bigwig tracks')
    parser.add_argument('--oe-norm', dest='oe_norm', action='store_true',
                        help='Whether to apply observed/expected normalization to Hi-C matrices')

    parser.add_argument('--out-file', dest='out_file', required=False,
                        help='Path to the output file for full chromosome prediction')
    parser.add_argument('--seq', dest='seq_path', required=True,
                        help='Path to the folder where the sequence .fa.gz files are stored')
    parser.add_argument('--seq2', dest='seq2_path', required=False,
                        help='Path to the second allele sequence .fa.gz files')
    parser.add_argument('--bigwigs', nargs='*', required=False, action=ParseKwargs,
                        help='Paths to the bigwig files for genomic features (key=value)')
    parser.add_argument('--plot-bigwigs', dest='plot_bigwigs', nargs='+', required=False,
                        help='Names of bigwig tracks to plot (not inputs)')
    parser.add_argument('--plot-pred-bigwigs', dest='plot_pred_bigwigs', nargs='+', required=False,
                        help='Names of predicted bigwig tracks to plot')
    parser.add_argument('--ko', dest='ko_data', type=str, nargs='+', default=[], required=False,
                        help='name of data modalities to knockout')
    parser.add_argument('--ko-mode', dest='ko_mode', type=str, nargs='+', default=['zero'], required=False,
                        help='how we simulate the knockout (zero, mean, knockout, shuffle, knockout_shuffle, '
                             'reverse, reverse_motif, seq, enformer_seq, alphagenome_seq, del)')
    parser.add_argument('--ko-start', dest='deletion_start', nargs='+', type=int, required=False,
                        help='Starting points for deletion.')
    parser.add_argument('--ko-width', dest='deletion_width', nargs='+', type=int, required=False,
                        help='Width for deletion.')
    parser.add_argument('--peak-height', dest='peak_height', nargs='+', type=float, default=2.0,
                        help='Peak height threshold for knockout.')
    parser.add_argument('--alt', dest='alt_bp', type=str, nargs='+', required=False,
                        help='Alt base(s) for seq / enformer_seq / alphagenome_seq ko-modes.')
    parser.add_argument('--padding', dest='end_padding_type', default='zero',
                        help='Padding type, either zero or follow (default: %(default)s)')
    parser.add_argument('--hide-line', dest='hide_deletion_line', action='store_true',
                        help='Remove the line showing deletion site')
    parser.add_argument('--whitespace', dest='whitespace', action='store_true',
                        help='Add whitespace around the deletion site for better visualization')
    parser.add_argument('--region', '--locus', dest='region', required=False,
                        help='specific region to visualize, otherwise full 2Mb window')

    # Screening params
    parser.add_argument('--screen-start', dest='screen_start', type=int, required=False,
                        help='Starting point for screening.')
    parser.add_argument('--screen-end', dest='screen_end', type=int, required=False,
                        help='Ending point for screening.')
    parser.add_argument('--step-size', dest='step_size', type=int, default=1000, required=False,
                        help='step size of perturbations in screening.')
    parser.add_argument('--n-top-sites', dest='n_top_sites', type=int, default=5, required=False,
                        help='number of most impactful sites to visualize after screening')
    parser.add_argument('--plot-diff', dest='plot_diff', action='store_true',
                        help='plot the difference heatmap instead of comparisons')
    parser.add_argument('--silent', dest='silent', action='store_true',
                        help='do not print out pyGenomeTracks logs')
    parser.add_argument('--load-screen', dest='load_screen', action='store_true',
                        help='load the screen results from a saved bedgraph')
    parser.add_argument('--n-overlap-pred', dest='n_overlap_preds', type=int, default=2, required=False,
                        help='Number of predictions for each pixel (controls step size of sliding window)')

    # Plotting params
    parser.add_argument('--min-val-true', dest='min_val_true', type=float, default=0.5, required=False,
                        help='min value for color scale of ground truth data')
    parser.add_argument('--max-val-true', dest='max_val_true', type=float, default=None, required=False,
                        help='max value for color scale of ground truth data')
    parser.add_argument('--min-val-pred', dest='min_val_pred', type=float, default=0.1, required=False,
                        help='min value for color scale of prediction data')
    parser.add_argument('--max-val-pred', dest='max_val_pred', type=float, default=None, required=False,
                        help='max value for color scale of prediction data')
    parser.add_argument('--min-val-diff', dest='min_val_diff', type=float, default=-0.5, required=False,
                        help='min value for color scale of diff matrix')
    parser.add_argument('--max-val-diff', dest='max_val_diff', type=float, default=0.5, required=False,
                        help='max value for color scale of diff matrix')
    parser.add_argument('--plot-bigwig-q', dest='plot_bigwig_q', type=float, default=0.995, required=False,
                        help='Quantile cutoff for bigwig plot max values (default: %(default)s)')
    parser.add_argument('--ctcf-motif-p', dest='ctcf_motif_p', type=int, default=None, required=False,
                        help='max p-value (transformed) to display CTCF motif')
    parser.add_argument('--no-plots', dest='no_plots', action='store_true', help='do not generate plots')

    # Enformer-based perturbation params
    parser.add_argument('--enformer-model', dest='enformer_model_path', type=str, default=None,
                        help='Path to a fine-tuned Enformer checkpoint (.ckpt).')
    parser.add_argument('--enformer-delta-mode', dest='enformer_delta_mode', type=str, default='multiplicative',
                        help='How to apply Enformer-predicted delta: multiplicative, additive, or prediction')
    parser.add_argument('--enformer-delta-cap', dest='enformer_delta_cap', type=float, default=10.0,
                        help='Cap on fold-change values when using enformer_seq mode (default: 10.0)')
    parser.add_argument('--enformer-tracks', dest='enformer_tracks', type=str, nargs='+', default=['ctcf', 'atac'],
                        help='Target track names for Enformer/AlphaGenome delta predictions')

    # AlphaGenome-based perturbation params (ko-mode alphagenome_seq).
    # Operates exactly like enformer_seq but with the AlphaGenome backbone; it
    # reuses --enformer-tracks / --enformer-delta-mode / --enformer-delta-cap for
    # track selection and delta application.
    parser.add_argument('--alphagenome-model', dest='alphagenome_model_path', type=str, default=None,
                        help='Path to the AlphaGenome checkpoint (.safetensors/.pth); required for ko-mode alphagenome_seq.')
    parser.add_argument('--alphagenome-metadata', dest='alphagenome_metadata_path', type=str, default=None,
                        help='Path to the AlphaGenome track-metadata catalog (parquet/csv/tsv) used to resolve '
                             'track names (ctcf, h3k27ac, ...) to output channels. Falls back to the built-in '
                             'catalog if omitted and available.')

    parser.add_argument('--allele-peak-split', dest='allele_peak_split', action='store_true',
                        help='Allele-specific peak redistribution: instead of applying the Enformer fold-change directly to the bulk track, split each bulk track into ref/alt allele tracks (shares 1/(1+fc) and fc/(1+fc), x2) and emit a separate Hi-C prediction per allele. Opt-in; default off keeps the direct-apply behavior.')

    # Haplotype peak-redistribution from PROVIDED predictions (function #2)
    parser.add_argument('--allele-haplotype', dest='allele_haplotype', action='store_true',
                        help='Haplotype peak redistribution from PROVIDED whole-genome Enformer '
                             'predictions: split each WT bulk track into maternal/paternal alleles '
                             'using --maternal-pred/--paternal-pred ratios (2*E*frac, CAP), run '
                             'hierarchical RAD21 per allele, and predict per allele using each '
                             'haplotype sequence (--maternal-seq/--paternal-seq). Single-locus (--start) only.')
    parser.add_argument('--maternal-pred', dest='maternal_pred', nargs='*', action=ParseKwargs,
                        help='Maternal Enformer prediction bigwigs (key=value: track=bigwig). Tracks '
                             'lacking both maternal+paternal preds keep their WT bulk track in both alleles.')
    parser.add_argument('--paternal-pred', dest='paternal_pred', nargs='*', action=ParseKwargs,
                        help='Paternal Enformer prediction bigwigs (key=value: track=bigwig).')
    parser.add_argument('--maternal-seq', dest='maternal_seq_path', required=False,
                        help='Per-chromosome .fa.gz dir for the maternal haplotype (main-model seq for the maternal allele).')
    parser.add_argument('--paternal-seq', dest='paternal_seq_path', required=False,
                        help='Per-chromosome .fa.gz dir for the paternal haplotype.')

    # Hierarchical RAD21 predictor params
    parser.add_argument('--hierarchical-model', dest='hierarchical_model_path', type=str, default=None,
                        help='Path to the hierarchical RAD21 predictor checkpoint (.ckpt).')
    parser.add_argument('--hierarchical-delta-mode', dest='hierarchical_delta_mode', type=str, default='multiplicative',
                        help='How to apply the hierarchical RAD21 delta: multiplicative, additive, or prediction')
    parser.add_argument('--hierarchical-delta-cap', dest='hierarchical_delta_cap', type=float, default=None,
                        help='Cap on fold-change values in hierarchical multiplicative mode')
    return parser


def resolve_device(device: str) -> str:
    """Apply the ``--device`` choice to the process environment.

    ``cpu`` hides all GPUs (equivalent to the old ``perturb_cpu.py``). Must be
    called before any torch import so CUDA is never initialised on GPU.
    Returns the resolved device string.
    """
    if device == 'cpu':
        os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
    return device


def main():
    parser = build_parser()
    args = parser.parse_args(args=None if sys.argv[1:] else ['--help'])

    resolve_device(args.device)          # MUST precede the heavy (torch) imports below
    os.makedirs('tmp', exist_ok=True)
    cfg = PerturbConfig.from_args(args)

    # Deferred imports: these pull in torch via cshark.inference.utils.
    if cfg.allele_haplotype:
        from cshark.perturb.scopes.allele_split import run_allele_haplotype
        return run_allele_haplotype(cfg)
    if cfg.start is not None:
        from cshark.perturb.scopes.single_locus import run_single_locus
        return run_single_locus(cfg)
    from cshark.perturb.scopes.full_chrom import run_full_chrom
    return run_full_chrom(cfg)


if __name__ == '__main__':
    main()
