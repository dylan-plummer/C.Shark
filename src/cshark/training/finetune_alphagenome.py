#!/usr/bin/env python
"""Fine-tune AlphaGenome (``alphagenome-pytorch``) on custom bigwig tracks.

This is the AlphaGenome analogue of ``train_hierarchical_with_enformer.py``: it
exists so that a *fine-tuned* AlphaGenome can be benchmarked head-to-head with
C.Shark on the same cell types, the same tracks and -- importantly -- the same
held-out chromosomes (val = chr10, test = chr15, matching
``cshark/data/genome_dataset.py``).

IMPORTANT: this module deliberately imports NOTHING from ``cshark``.  It runs in
the ``alphagenome`` conda env (which has ``alphagenome-pytorch``, torch, pyfaidx
and pyBigWig, but not C.Shark), so run it by path rather than with ``-m``:

    conda activate alphagenome
    python /path/to/src/cshark/training/finetune_alphagenome.py --help

See ``examples/finetune_alphagenome_mouse_islet_rna.sh`` for the mouse islet
RNA-seq invocation.

What it does
------------
1. Tiles the genome into fixed-length training/validation intervals (BED),
   excluding a blacklist (centromeres/telomeres) and any window without enough
   bigwig coverage (which is how the leading N-blocks get dropped).
2. Builds AlphaGenome, strips every pretrained head, loads the pretrained trunk.
3. Attaches ONE new ``GenomeTracksHead`` sized to ``--bigwig`` (one track per
   file) and applies the requested transfer mode (``linear`` / ``lora`` /
   ``full`` / ``ia3`` / ``houlsby``, combinable).
4. Trains with AlphaGenome's own multinomial (positional + count) loss and
   reports per-track Pearson r on log1p signal, which is the metric C.Shark
   logs for its 1D tracks.
5. Saves a small delta checkpoint (adapters + head only, ~MBs) each time val
   improves, plus a ``last.delta.pth`` for resuming.

The upstream ``agt finetune`` CLI described in the docs is not present in the
released ``alphagenome-pytorch`` wheel (0.3.1), so this driver calls the same
public Python API (``TransferConfig`` / ``prepare_for_transfer`` /
``GenomicDataset`` / ``compute_finetuning_loss`` / ``save_delta_checkpoint``)
directly.  Two things it adds that the library's stock ``train_epoch`` cannot
do, and which matter here:

  * ``--organism-index 1`` -- the stock loops hardcode organism 0 (human).
    Mouse data must go through the mouse organism embedding.
  * pair-embedding removal -- the trunk always builds the (B, S, S, 128) pair
    embedding for the contact-map head.  We deleted that head, so the pair
    embedder is stubbed out by default (``--pair-embeddings`` puts it back),
    which saves a large chunk of activation memory.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.amp import autocast
from torch.utils.data import DataLoader
from tqdm import tqdm

import pyBigWig

from alphagenome_pytorch import AlphaGenome
from alphagenome_pytorch.extensions.finetuning import (
    TransferConfig,
    collate_genomic,
    compute_finetuning_loss,
    create_lr_scheduler,
    load_delta_checkpoint,
    save_delta_checkpoint,
    TrainingLogger,
)
from alphagenome_pytorch.extensions.finetuning.datasets import (
    CachedGenome,
    GenomicDataset,
    compute_track_means,
)
from alphagenome_pytorch.extensions.finetuning.transfer import (
    load_trunk,
    prepare_for_transfer,
    remove_all_heads,
)

HEAD_NAME = "finetune"


# ---------------------------------------------------------------------------
# Interval (BED) construction
# ---------------------------------------------------------------------------

def read_chrom_sizes(fasta_path: str) -> dict[str, int]:
    """Read chromosome sizes from a FASTA index (``<fasta>.fai``)."""
    fai = Path(f"{fasta_path}.fai")
    if not fai.exists():
        raise FileNotFoundError(
            f"{fai} not found. Index the genome first: samtools faidx {fasta_path}"
        )
    sizes: dict[str, int] = {}
    with open(fai) as fh:
        for line in fh:
            name, length = line.split("\t")[:2]
            sizes[name] = int(length)
    return sizes


def read_blacklist(bed_path: str | None) -> dict[str, list[tuple[int, int]]]:
    """Read a BED of regions to avoid (centromeres/telomeres/etc)."""
    blacklist: dict[str, list[tuple[int, int]]] = {}
    if bed_path is None:
        return blacklist
    with open(bed_path) as fh:
        for line in fh:
            if not line.strip() or line.startswith(("#", "track", "browser")):
                continue
            fields = line.split()
            blacklist.setdefault(fields[0], []).append((int(fields[1]), int(fields[2])))
    for chrom in blacklist:
        blacklist[chrom].sort()
    return blacklist


def _overlaps(start: int, end: int, regions: list[tuple[int, int]]) -> bool:
    return any(r_start < end and start < r_end for r_start, r_end in regions)


def build_intervals(
    chrom_sizes: dict[str, int],
    chroms: list[str],
    seq_len: int,
    stride: int,
    blacklist: dict[str, list[tuple[int, int]]],
    coverage_bw: str | None,
    min_coverage: float,
    max_intervals: int | None,
) -> list[tuple[str, int, int]]:
    """Tile ``chroms`` into ``seq_len`` windows, dropping blacklisted/empty ones."""
    bw = pyBigWig.open(coverage_bw) if coverage_bw and min_coverage > 0 else None
    bw_chroms = set(bw.chroms()) if bw is not None else set()

    intervals: list[tuple[str, int, int]] = []
    try:
        for chrom in chroms:
            size = chrom_sizes.get(chrom)
            if size is None or size < seq_len:
                continue
            banned = blacklist.get(chrom, [])
            for start in range(0, size - seq_len + 1, stride):
                end = start + seq_len
                if _overlaps(start, end, banned):
                    continue
                if bw is not None and chrom in bw_chroms:
                    stats = bw.stats(chrom, start, end, type="coverage")
                    cov = stats[0] if stats and stats[0] is not None else 0.0
                    if cov < min_coverage:
                        continue
                intervals.append((chrom, start, end))
    finally:
        if bw is not None:
            bw.close()

    if max_intervals is not None and len(intervals) > max_intervals:
        step = max(1, len(intervals) // max_intervals)
        intervals = intervals[::step][:max_intervals]
    return intervals


def write_bed(path: Path, intervals: list[tuple[str, int, int]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as fh:
        for chrom, start, end in intervals:
            fh.write(f"{chrom}\t{start}\t{end}\n")


def make_split_beds(args: argparse.Namespace, out_dir: Path) -> tuple[Path, Path]:
    """Create (or reuse) train/val interval BEDs for the requested split."""
    train_bed = out_dir / "train_intervals.bed"
    val_bed = out_dir / "val_intervals.bed"
    if train_bed.exists() and val_bed.exists() and not args.rebuild_beds:
        print(f"[beds] reusing {train_bed} and {val_bed}")
        return train_bed, val_bed

    chrom_sizes = read_chrom_sizes(args.genome_fasta)
    blacklist = read_blacklist(args.blacklist_bed)

    # Chromosome split, mirroring cshark/data/genome_dataset.py: everything
    # except the val/test/excluded chromosomes trains.
    held_out = set(args.val_chroms) | set(args.test_chroms) | set(args.exclude_chroms)
    if args.train_chroms:
        train_chroms = list(args.train_chroms)
    else:
        train_chroms = [c for c in args.all_chroms if c not in held_out]

    coverage_bw = args.bigwig[0] if args.min_signal_coverage > 0 else None
    print(f"[beds] train chroms: {' '.join(train_chroms)}")
    print(f"[beds] val chroms:   {' '.join(args.val_chroms)}")

    train_intervals = build_intervals(
        chrom_sizes, train_chroms, args.sequence_length, args.train_stride,
        blacklist, coverage_bw, args.min_signal_coverage, args.max_train_intervals,
    )
    val_intervals = build_intervals(
        chrom_sizes, args.val_chroms, args.sequence_length, args.val_stride,
        blacklist, coverage_bw, args.min_signal_coverage, args.max_val_intervals,
    )
    if not train_intervals or not val_intervals:
        raise RuntimeError(
            "Empty train or val interval set -- check --sequence-length, the "
            "chromosome names, and --min-signal-coverage."
        )

    write_bed(train_bed, train_intervals)
    write_bed(val_bed, val_intervals)
    print(f"[beds] wrote {len(train_intervals)} train / {len(val_intervals)} val "
          f"intervals of {args.sequence_length} bp")
    return train_bed, val_bed


# ---------------------------------------------------------------------------
# Model construction
# ---------------------------------------------------------------------------

class _NoPairEmbedder(nn.Module):
    """Stub for ``model.embedder_pair`` when contact maps are not being trained.

    The trunk unconditionally builds the (B, S, S, 128) pair embedding for the
    contact-map head.  With that head removed the tensor is pure overhead, and
    at 1 Mb it is the single largest activation in the graph.
    """

    def forward(self, *_args, **_kwargs):  # noqa: D102
        return None


def build_model(args: argparse.Namespace, track_means: torch.Tensor | None):
    """Build AlphaGenome with a fresh head and the requested transfer mode."""
    model = AlphaGenome(
        num_organisms=2,
        gradient_checkpointing=args.gradient_checkpointing,
    )
    # Drop every pretrained head (genome tracks, contact maps, splicing) before
    # loading, so nothing but the trunk is instantiated or restored.
    remove_all_heads(model)
    print(f"[model] loading pretrained trunk from {args.pretrained_weights}")
    model = load_trunk(model, args.pretrained_weights, exclude_heads=True, strict=False)

    if not args.pair_embeddings:
        model.embedder_pair = _NoPairEmbedder()

    head_cfg = {
        "modality": args.modality,
        "num_tracks": len(args.bigwig),
        "resolutions": list(args.resolutions),
        "num_organisms": 1,
    }
    if track_means is not None:
        head_cfg["track_means"] = track_means

    config = TransferConfig(
        mode=list(args.mode),
        lora_rank=args.lora_rank,
        lora_alpha=args.lora_alpha,
        lora_targets=list(args.lora_targets),
        unfreeze_norm=args.unfreeze_norm,
        new_heads={HEAD_NAME: head_cfg},
        learning_rate=args.lr,
    )
    model = prepare_for_transfer(model, config)

    n_total = sum(p.numel() for p in model.parameters())
    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[model] mode={args.mode} trainable {n_train:,} / {n_total:,} params "
          f"({100 * n_train / max(1, n_total):.2f}%)")
    return model, config


# ---------------------------------------------------------------------------
# Train / validate
# ---------------------------------------------------------------------------

def forward_batch(model, head, sequences, trunk_org, head_org, resolutions):
    """Trunk -> embeddings -> new head. Returns {resolution: (B, S, T)}."""
    outputs = model(
        sequences,
        trunk_org,
        embeddings_only=True,
        resolutions=tuple(resolutions),
        channels_last=False,
    )
    embeddings = {}
    if 1 in resolutions and outputs.get("embeddings_1bp") is not None:
        embeddings[1] = outputs["embeddings_1bp"]
    if 128 in resolutions and outputs.get("embeddings_128bp") is not None:
        embeddings[128] = outputs["embeddings_128bp"]
    return head(embeddings, head_org)


def _organism_indices(batch_size: int, device: torch.device, organism_index: int):
    """Trunk index selects human/mouse; the new head only ever has organism 0."""
    trunk = torch.full((batch_size,), organism_index, dtype=torch.long, device=device)
    head = torch.zeros(batch_size, dtype=torch.long, device=device)
    return trunk, head


def train_one_epoch(model, head, loader, optimizer, scheduler, device, args,
                    epoch, logger) -> float:
    model.train()
    head.train()
    amp = (autocast(device_type="cuda", dtype=torch.bfloat16)
           if args.amp and device.type == "cuda" else nullcontext())

    total_loss, n_batches = 0.0, 0
    optimizer.zero_grad(set_to_none=True)
    pbar = tqdm(loader, desc=f"epoch {epoch}", dynamic_ncols=True)
    for batch_idx, (sequences, targets) in enumerate(pbar):
        sequences = sequences.to(device, non_blocking=True)
        targets = {res: t.to(device, non_blocking=True)
                   for res, t in targets.items() if res in args.resolutions}
        trunk_org, head_org = _organism_indices(sequences.shape[0], device,
                                                args.organism_index)

        with amp:
            predictions = forward_batch(model, head, sequences, trunk_org, head_org,
                                        args.resolutions)
            loss, _ = compute_finetuning_loss(
                predictions=predictions,
                targets=targets,
                resolution_weights=args.resolution_weights,
                positional_weight=args.positional_weight,
                device=device,
                channels_last=True,
            )

        (loss / args.accumulation_steps).backward()

        if (batch_idx + 1) % args.accumulation_steps == 0:
            if args.max_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(
                    [p for p in model.parameters() if p.requires_grad],
                    args.max_grad_norm,
                )
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            scheduler.step()

        total_loss += loss.item()
        n_batches += 1
        if batch_idx % args.log_every == 0:
            lr = scheduler.get_last_lr()[0]
            pbar.set_postfix({"loss": f"{loss.item():.4f}", "lr": f"{lr:.2e}"})
            if logger is not None:
                logger.log_step({"train_loss": loss.item(), "lr": lr, "epoch": epoch})

        if args.max_steps_per_epoch and n_batches >= args.max_steps_per_epoch:
            break

    # Flush a trailing partial accumulation window.
    if n_batches % args.accumulation_steps != 0:
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        scheduler.step()

    return total_loss / max(1, n_batches)


def _pearson(pred: np.ndarray, target: np.ndarray) -> float:
    """Pearson r between two 1-D arrays; NaN when either side is constant."""
    if pred.size < 2:
        return float("nan")
    pred = pred - pred.mean()
    target = target - target.mean()
    denom = np.sqrt((pred ** 2).sum() * (target ** 2).sum())
    if denom <= 0:
        return float("nan")
    return float((pred * target).sum() / denom)


@torch.no_grad()
def validate(model, head, loader, device, args) -> tuple[float, dict[str, float]]:
    """Return (mean val loss, per-track log1p Pearson r at the coarsest resolution)."""
    model.eval()
    head.eval()
    amp = (autocast(device_type="cuda", dtype=torch.bfloat16)
           if args.amp and device.type == "cuda" else nullcontext())

    metric_res = max(args.resolutions)
    total_loss, n_batches = 0.0, 0
    # Per-track correlations are averaged over windows, matching how C.Shark
    # reports val_enformer_corr_1d_* (per-sample r, then mean).
    corr_sums = np.zeros(len(args.bigwig))
    corr_counts = np.zeros(len(args.bigwig))

    for sequences, targets in tqdm(loader, desc="val", dynamic_ncols=True):
        sequences = sequences.to(device, non_blocking=True)
        targets = {res: t.to(device, non_blocking=True)
                   for res, t in targets.items() if res in args.resolutions}
        trunk_org, head_org = _organism_indices(sequences.shape[0], device,
                                                args.organism_index)

        with amp:
            predictions = forward_batch(model, head, sequences, trunk_org, head_org,
                                        args.resolutions)
            loss, _ = compute_finetuning_loss(
                predictions=predictions,
                targets=targets,
                resolution_weights=args.resolution_weights,
                positional_weight=args.positional_weight,
                device=device,
                channels_last=True,
            )
        total_loss += loss.item()
        n_batches += 1

        pred = np.log1p(predictions[metric_res].float().clamp(min=0).cpu().numpy())
        obs = np.log1p(targets[metric_res].float().clamp(min=0).cpu().numpy())
        for b in range(pred.shape[0]):
            for t in range(pred.shape[2]):
                r = _pearson(pred[b, :, t], obs[b, :, t])
                if not np.isnan(r):
                    corr_sums[t] += r
                    corr_counts[t] += 1

    per_track = {
        f"val_corr_{name}": float(corr_sums[i] / corr_counts[i]) if corr_counts[i] else float("nan")
        for i, name in enumerate(args.track_names)
    }
    finite = [v for v in per_track.values() if not np.isnan(v)]
    per_track["val_corr_mean"] = float(np.mean(finite)) if finite else float("nan")
    return total_loss / max(1, n_batches), per_track


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Fine-tune AlphaGenome on custom bigwig tracks.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    data = p.add_argument_group("data")
    data.add_argument("--genome-fasta", required=True,
                      help="Reference FASTA (must have a .fai next to it).")
    data.add_argument("--bigwig", nargs="+", required=True,
                      help="Target bigwigs, one per output track.")
    data.add_argument("--track-names", nargs="+", default=None,
                      help="Names for the tracks (default: bigwig basenames). "
                           "Used for logging and saved with the checkpoint.")
    data.add_argument("--modality", default="rna_seq",
                      choices=["rna_seq", "atac", "dnase", "procap", "cage",
                               "chip_tf", "chip_histone"],
                      help="Assay type of the head; rna_seq applies AlphaGenome's "
                           "power-law squashing, the others do not.")
    data.add_argument("--sequence-length", type=int, default=131_072,
                      help="Input window. AlphaGenome's native context is 1048576, "
                           "but that only fits on large-memory GPUs.")
    data.add_argument("--resolutions", type=int, nargs="+", default=[128],
                      choices=[1, 128],
                      help="Output resolutions to train. Adding 1 runs the decoder "
                           "and costs far more memory than 128 alone.")

    split = p.add_argument_group("chromosome split (defaults match C.Shark)")
    split.add_argument("--all-chroms", nargs="+",
                       default=[f"chr{i}" for i in range(1, 20)],
                       help="Candidate chromosomes (default: mouse autosomes).")
    split.add_argument("--train-chroms", nargs="+", default=None,
                       help="Explicit train chromosomes; default is --all-chroms "
                            "minus the val/test/excluded ones.")
    split.add_argument("--val-chroms", nargs="+", default=["chr10"])
    split.add_argument("--test-chroms", nargs="+", default=["chr15"],
                       help="Held out entirely; never used by this script.")
    split.add_argument("--exclude-chroms", nargs="+", default=["chrX", "chrY", "chrM"])
    split.add_argument("--blacklist-bed", default=None,
                       help="BED of regions to skip (e.g. data/<assembly>/centrotelo.bed).")
    split.add_argument("--train-stride", type=int, default=None,
                       help="Step between train windows (default: --sequence-length, "
                            "i.e. non-overlapping tiling).")
    split.add_argument("--val-stride", type=int, default=None,
                       help="Step between val windows (default: 4x --sequence-length).")
    split.add_argument("--max-train-intervals", type=int, default=None)
    split.add_argument("--max-val-intervals", type=int, default=256)
    split.add_argument("--min-signal-coverage", type=float, default=0.5,
                       help="Drop windows where the first bigwig covers less than "
                            "this fraction of bases. Removes N-blocks and unmappable "
                            "regions. Set 0 to disable.")
    split.add_argument("--rebuild-beds", action="store_true",
                       help="Regenerate the interval BEDs even if they exist.")

    model = p.add_argument_group("model / transfer")
    model.add_argument("--pretrained-weights", required=True,
                       help="AlphaGenome .safetensors / .pth checkpoint.")
    model.add_argument("--mode", nargs="+", default=["lora"],
                       choices=["full", "linear", "lora", "locon", "ia3", "houlsby"],
                       help="Transfer mode(s). 'lora' is the recommended default; "
                            "'linear' trains the head only; 'full' cannot be combined.")
    model.add_argument("--lora-rank", type=int, default=8)
    model.add_argument("--lora-alpha", type=float, default=16)
    model.add_argument("--lora-targets", nargs="+", default=["q_proj", "v_proj"])
    model.add_argument("--unfreeze-norm", action="store_true",
                       help="Also train LayerNorm/BatchNorm when using adapters.")
    model.add_argument("--organism-index", type=int, default=0, choices=[0, 1],
                       help="Trunk organism embedding: 0=human, 1=mouse.")
    model.add_argument("--pair-embeddings", action="store_true",
                       help="Keep the (B,S,S,128) pair embedding. Off by default "
                            "since the contact-map head is removed; only needed if "
                            "you plan to re-attach it.")
    model.add_argument("--gradient-checkpointing", action="store_true",
                       help="Trade compute for activation memory in the "
                            "encoder/tower/decoder.")

    train = p.add_argument_group("optimisation")
    train.add_argument("--epochs", type=int, default=10)
    train.add_argument("--batch-size", type=int, default=1)
    train.add_argument("--accumulation-steps", type=int, default=8)
    train.add_argument("--lr", type=float, default=1e-4)
    train.add_argument("--weight-decay", type=float, default=0.0)
    train.add_argument("--warmup-steps", type=int, default=200)
    train.add_argument("--schedule", default="cosine", choices=["cosine", "constant"])
    train.add_argument("--positional-weight", type=float, default=1.0,
                       help="Weight of the positional term of the multinomial loss "
                            "relative to the total-count term.")
    train.add_argument("--resolution-weight", type=float, nargs="+", default=None,
                       help="Loss weight per entry of --resolutions (default: 1.0 each).")
    train.add_argument("--max-grad-norm", type=float, default=1.0)
    train.add_argument("--no-amp", dest="amp", action="store_false",
                       help="Disable bfloat16 autocast.")
    train.add_argument("--num-workers", type=int, default=4)
    train.add_argument("--cache-signals", action="store_true",
                       help="Preload every bigwig into RAM (~4 bytes per base per "
                            "track, so ~10 GB per track for a mouse genome).")
    train.add_argument("--cache-genome", action="store_true",
                       help="Preload the one-hot genome into RAM (~4 bytes per base "
                            "for the chromosomes in use). Shared by train and val.")
    train.add_argument("--max-steps-per-epoch", type=int, default=None,
                       help="Cap training batches per epoch (smoke tests).")
    train.add_argument("--track-means-samples", type=int, default=200,
                       help="Windows used to estimate each track's nonzero mean, "
                            "which sets the head's output scale. 0 to skip.")

    out = p.add_argument_group("output")
    out.add_argument("--output-dir", required=True)
    out.add_argument("--run-name", default=None)
    out.add_argument("--log-every", type=int, default=25)
    out.add_argument("--save-full-checkpoint", action="store_true",
                     help="Also write the full ~1GB state dict, not just the delta.")
    out.add_argument("--resume", default=None,
                     help="Delta checkpoint to resume from (e.g. .../last.delta.pth).")
    out.add_argument("--use-wandb", action="store_true")
    out.add_argument("--wandb-project", default="alphagenome-finetune")
    out.add_argument("--wandb-entity", default=None)
    out.add_argument("--seed", type=int, default=0)

    args = p.parse_args(argv)

    args.resolutions = sorted(set(args.resolutions))
    if args.modality in ("chip_tf", "chip_histone") and args.resolutions != [128]:
        p.error(f"--modality {args.modality} only supports --resolutions 128")
    for res in args.resolutions:
        if args.sequence_length % res:
            p.error(f"--sequence-length must be divisible by resolution {res}")
    if args.sequence_length % 128:
        p.error("--sequence-length must be a multiple of 128 (the trunk's bin size)")

    if args.resolution_weight is None:
        args.resolution_weights = {res: 1.0 for res in args.resolutions}
    elif len(args.resolution_weight) != len(args.resolutions):
        p.error("--resolution-weight needs one value per --resolutions entry")
    else:
        args.resolution_weights = dict(zip(args.resolutions, args.resolution_weight))

    if args.track_names is None:
        args.track_names = [Path(b).stem for b in args.bigwig]
    if len(args.track_names) != len(args.bigwig):
        p.error("--track-names must have one entry per --bigwig")

    if args.train_stride is None:
        args.train_stride = args.sequence_length
    if args.val_stride is None:
        args.val_stride = 4 * args.sequence_length
    if "full" in args.mode and len(args.mode) > 1:
        p.error("--mode full cannot be combined with other modes")
    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[setup] device={device} tracks={args.track_names}")

    train_bed, val_bed = make_split_beds(args, out_dir)

    # --- head output scale -------------------------------------------------
    means_path = out_dir / "track_means.pt"
    track_means = None
    if args.track_means_samples > 0:
        if means_path.exists() and not args.rebuild_beds:
            track_means = torch.load(means_path, weights_only=True)
            print(f"[means] loaded {means_path}: {track_means.tolist()}")
        else:
            print(f"[means] estimating nonzero means from {args.track_means_samples} windows")
            track_means = compute_track_means(
                bigwig_files=args.bigwig,
                bed_file=str(train_bed),
                sequence_length=args.sequence_length,
                resolution=1,
                max_samples=args.track_means_samples,
            )
            torch.save(track_means, means_path)

    # --- data --------------------------------------------------------------
    if args.cache_genome:
        # One shared cache for both splits, restricted to the chromosomes in play.
        used_chroms = {line.split("\t")[0]
                       for bed in (train_bed, val_bed)
                       for line in bed.read_text().splitlines() if line}
        genome = CachedGenome(args.genome_fasta, chromosomes=used_chroms)
    else:
        genome = args.genome_fasta
    common = dict(
        genome_fasta=genome,
        bigwig_files=args.bigwig,
        resolutions=tuple(args.resolutions),
        sequence_length=args.sequence_length,
        cache_signals=args.cache_signals,
    )
    train_ds = GenomicDataset(bed_file=str(train_bed), **common)
    val_ds = GenomicDataset(bed_file=str(val_bed), **common)
    print(f"[data] {len(train_ds)} train windows, {len(val_ds)} val windows")

    loader_kwargs = dict(
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        collate_fn=collate_genomic,
        pin_memory=device.type == "cuda",
        persistent_workers=args.num_workers > 0,
    )
    train_loader = DataLoader(train_ds, shuffle=True, drop_last=True, **loader_kwargs)
    val_loader = DataLoader(val_ds, shuffle=False, drop_last=False, **loader_kwargs)

    # --- model -------------------------------------------------------------
    model, transfer_config = build_model(args, track_means)
    model = model.to(device)
    head = model.heads[HEAD_NAME]

    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=args.lr, weight_decay=args.weight_decay)
    steps_per_epoch = max(1, len(train_loader) // args.accumulation_steps)
    if args.max_steps_per_epoch:
        steps_per_epoch = max(1, args.max_steps_per_epoch // args.accumulation_steps)
    scheduler = create_lr_scheduler(
        optimizer,
        warmup_steps=args.warmup_steps,
        total_steps=steps_per_epoch * args.epochs,
        schedule=args.schedule,
    )

    start_epoch, best_val = 1, float("inf")
    if args.resume:
        print(f"[resume] {args.resume}")
        _, meta = load_delta_checkpoint(
            args.resume, model, optimizer=optimizer, scheduler=scheduler,
            verify_hash=False, skip_prepare=True,
        )
        start_epoch = int(meta.get("epoch", 0)) + 1
        best_val = float(meta.get("best_val_loss", meta.get("val_loss", float("inf"))))
        print(f"[resume] continuing at epoch {start_epoch} (best val {best_val:.4f})")

    run_config = {k: (str(v) if isinstance(v, Path) else v)
                  for k, v in vars(args).items()}
    (out_dir / "config.json").write_text(json.dumps(run_config, indent=2, default=str))
    logger = TrainingLogger(
        output_dir=out_dir,
        use_wandb=args.use_wandb,
        wandb_project=args.wandb_project,
        wandb_entity=args.wandb_entity,
        run_name=args.run_name or out_dir.name,
        config=run_config,
    )

    # --- train -------------------------------------------------------------
    try:
        for epoch in range(start_epoch, args.epochs + 1):
            t0 = time.time()
            train_loss = train_one_epoch(model, head, train_loader, optimizer,
                                         scheduler, device, args, epoch, logger)
            val_loss, metrics = validate(model, head, val_loader, device, args)
            is_best = val_loss < best_val
            if is_best:
                best_val = val_loss

            print(f"[epoch {epoch}] train {train_loss:.4f} | val {val_loss:.4f} | "
                  + " | ".join(f"{k.replace('val_corr_', 'r_')} {v:.3f}"
                               for k, v in metrics.items())
                  + f" | {time.time() - t0:.0f}s"
                  + ("  <- best" if is_best else ""))

            logger.log_epoch(epoch, train_loss=train_loss, val_loss=val_loss,
                             lr=scheduler.get_last_lr()[0], is_best=is_best,
                             extra=metrics)

            ckpt_meta = dict(epoch=epoch, train_loss=train_loss, val_loss=val_loss,
                             best_val_loss=best_val, track_names=args.track_names,
                             organism_index=args.organism_index,
                             sequence_length=args.sequence_length,
                             resolutions=list(args.resolutions), **metrics)
            save_delta_checkpoint(out_dir / "last.delta.pth", model, transfer_config,
                                  optimizer=optimizer, scheduler=scheduler, **ckpt_meta)
            if is_best:
                save_delta_checkpoint(out_dir / "best.delta.pth", model,
                                      transfer_config, **ckpt_meta)
                if args.save_full_checkpoint:
                    torch.save({"model_state_dict": model.state_dict(), **ckpt_meta},
                               out_dir / "best.full.pth")
    finally:
        logger.finish()

    print(f"[done] best val loss {best_val:.4f}; checkpoints in {out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
