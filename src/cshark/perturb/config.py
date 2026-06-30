"""Typed configuration for a perturbation run.

``PerturbConfig`` is the parsed-argument struct produced by ``cli.py`` and
threaded through ``scopes`` / ``models`` / ``output`` -- replacing the original
practice of passing the raw argparse ``args`` namespace plus ~30 positional
parameters into ``single_deletion``.

Field set mirrors the argparse options of the original ``perturb.py main()``
(lines 59-204), with one addition: ``device`` (folds in ``perturb_cpu.py``).
"""
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

# Module-level constants (verbatim from the original perturb.py).
WINDOW = 2097152
RES = 8192
IMAGE_SCALE = 256


@dataclass
class PerturbConfig:
    # --- core ---
    celltype: str
    chr_name: str
    model_path: str
    seq_path: str
    output_path: str = 'outputs'
    outname: str = ''
    assembly: str = 'hg19'
    start: Optional[int] = None
    device: str = 'auto'                 # auto | cpu | cuda  (replaces perturb_cpu.py)

    # --- Hi-C / model output geometry ---
    resolution: int = 8192
    resolution_1d: int = 256
    mat_size: int = 256                  # --matrix-size
    mid_hidden: int = 256                # --latent_size
    seq_filter_size: int = 3
    recon_1d: bool = True                # --no-recon -> False
    hic_log_transform: bool = True       # --no-hic-log-transform -> False
    bigwig_log_transform: bool = True    # --no-bigwig-log-transform -> False
    oe_norm: bool = False                # --oe-norm
    n_overlap_preds: int = 2

    # --- inputs ---
    seq2_path: Optional[str] = None      # diploid second allele
    bigwigs: Dict[str, str] = field(default_factory=dict)
    out_file: Optional[str] = None       # full-chrom TSV output

    # --- perturbation ---
    ko_data: List[str] = field(default_factory=list)
    ko_mode: List[str] = field(default_factory=lambda: ['zero'])
    deletion_start: Optional[List[int]] = None   # --ko-start
    deletion_width: Optional[List[int]] = None   # --ko-width
    peak_height: Any = 2.0                       # scalar or list (argparse nargs='+')
    allele_peak_split: bool = False               # opt-in allele-specific peak redistribution (ref/alt)
    allele_haplotype: bool = False                # opt-in haplotype redistribution from PROVIDED preds (maternal/paternal)
    maternal_pred: Dict[str, str] = field(default_factory=dict)   # --maternal-pred track=bigwig (whole-genome Enformer preds)
    paternal_pred: Dict[str, str] = field(default_factory=dict)   # --paternal-pred track=bigwig
    maternal_seq_path: Optional[str] = None       # --maternal-seq (per-chrom .fa.gz dir for the maternal haplotype)
    paternal_seq_path: Optional[str] = None       # --paternal-seq
    alt_bp: Optional[List[str]] = None           # --alt
    end_padding_type: str = 'zero'               # --padding

    # --- screening ---
    screen_start: Optional[int] = None
    screen_end: Optional[int] = None
    step_size: int = 1000
    n_top_sites: int = 5
    load_screen: bool = False
    region: Optional[str] = None                 # --region / --locus

    # --- plotting ---
    plot_bigwigs: Optional[List[str]] = None
    plot_pred_bigwigs: Optional[List[str]] = None
    hide_deletion_line: bool = False             # --hide-line
    whitespace: bool = False
    plot_diff: bool = False
    silent: bool = False
    no_plots: bool = False
    min_val_true: float = 0.5
    max_val_true: Optional[float] = None
    min_val_pred: float = 0.1
    max_val_pred: Optional[float] = None
    min_val_diff: float = -0.5
    max_val_diff: float = 0.5
    plot_bigwig_q: float = 0.995
    ctcf_motif_p: Optional[int] = None

    # --- secondary predictors ---
    enformer_model_path: Optional[str] = None
    enformer_delta_mode: str = 'multiplicative'
    enformer_delta_cap: float = 10.0
    enformer_tracks: List[str] = field(default_factory=lambda: ['ctcf', 'atac'])
    hierarchical_model_path: Optional[str] = None
    hierarchical_delta_mode: str = 'multiplicative'
    hierarchical_delta_cap: Optional[float] = None

    # --- derived (computed in from_args) ---
    ctcf_path: Optional[str] = None
    atac_path: Optional[str] = None
    other_feats: Optional[List[str]] = None

    @classmethod
    def from_args(cls, args) -> "PerturbConfig":
        """Build a config from an argparse Namespace, computing derived fields.

        Mirrors the post-parse setup in the original ``main()`` (lines 206-227):
        split ``bigwigs`` into ctcf/atac/other paths and normalise ko lists.
        """
        bigwigs = dict(args.bigwigs) if getattr(args, 'bigwigs', None) else {}
        ko_data = args.ko_data
        ko_mode = args.ko_mode
        if isinstance(ko_data, str):
            ko_data = [ko_data]
        if isinstance(ko_mode, str):
            ko_mode = [ko_mode]
        other_feats = [v for k, v in bigwigs.items() if k not in ('ctcf', 'atac')]
        if not other_feats:
            other_feats = None
        maternal_pred = dict(args.maternal_pred) if getattr(args, 'maternal_pred', None) else {}
        paternal_pred = dict(args.paternal_pred) if getattr(args, 'paternal_pred', None) else {}

        # Map only the known dataclass fields from the namespace.
        known = {f for f in cls.__dataclass_fields__}
        kwargs = {k: v for k, v in vars(args).items() if k in known}
        kwargs.update(
            bigwigs=bigwigs,
            ko_data=ko_data,
            ko_mode=ko_mode,
            ctcf_path=bigwigs.get('ctcf'),
            atac_path=bigwigs.get('atac'),
            other_feats=other_feats,
            maternal_pred=maternal_pred,
            paternal_pred=paternal_pred,
        )
        return cls(**kwargs)
