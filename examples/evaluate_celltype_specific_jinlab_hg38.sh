#!/usr/bin/env bash
# Evaluate celltype-specific ATAC on the lab's hg38 SC-islet differentiation
# scATAC: AlphaLike, BetaLike, DeltaLike and Stem.
#
#   sequence --(layer 1: ONE Enformer head PER CELLTYPE)--> celltype ATAC
#
# DATA LAYOUT -- how the lab tracks reach $DATA.
# The 33 differentiation celltypes are organised under
#   /mnt/jinstore/JinLab04/dmp131/cshark_data/jinlab_finetune/hg38/<celltype>/
#       genomic_features/atac.bw -> Archive01/jiacs/finetuning/export/bigwig/
#                                   <celltype>.fragment.CPM.bw
# and each <celltype> folder is symlinked into $DATA (data/hg38/<celltype>), the
# same way the hg19 islet folders are linked into data/hg19. $DATA supplies the
# rest: dna_sequence/ and hg38_genes.gtf. So --celltypes takes the folder names
# as they are, and they must ALSO be head-group names in the checkpoint -- a
# checkpoint whose heads are called anything else (e.g. islet_alpha) exits at
# startup listing the heads it does have. To add a celltype from the 33, add its
# name to 'celltypes' below; the link is already there.
#
# THE THING THAT MAKES THIS DIFFERENT FROM THE ALLELE-SPECIFIC BENCHMARKS:
# every celltype shares one genome, so the input sequence is byte-identical
# across celltypes. Layer 1 cannot read the difference off the sequence the way
# it can with SNPs -- every celltype-specific prediction comes from that
# celltype's own head. Two things follow:
#   * the trunk runs ONCE per window and every head projects off it, so this
#     costs ~1x a single-celltype pass, however many celltypes are scored;
#   * "median |log2| predicted" is reported next to every correlation. Heads that
#     collapsed onto each other predict ratio ~0 everywhere, which scores r~0 --
#     the same number as being wrong, but a completely different failure. Read
#     the compression factor before the correlations.
#
# LIBRARY DEPTH -- MOSTLY HANDLED UPSTREAM HERE, BUT NOT FOR THE MODEL.
# These bigwigs are CPM-normalised fragment tracks, so their totals already
# match: 2.01-2.08e8 across AlphaLike / BetaLike / DeltaLike (checked when the
# data was linked). The evaluator's depth normalisation is therefore close to a
# no-op on the MEASURED side -- watch the '[layer1] library ... scale=' lines;
# anything far from 1.0 means a track is not what this comment assumes.
# --center-log2 median still matters: it removes whatever per-celltype offset
# the HEADS learned, which on the islet data reached ~1 log2 even where the
# measured tracks were depth-matched. Every run prints the offsets it removed.
#
# PREDICTED TRACKS AS BIGWIGS (--write-bigwigs, ON HERE).
# Every prediction is also dumped to a bigwig under ${out}/bigwigs/, one file
# per (layer, celltype, track), so the tracks behind the metrics can be loaded
# into a browser beside the measured bigwigs. This costs NO extra forward
# passes -- the tracks are accumulated from the passes the evaluation already
# runs. Which tracks appear is decided by the CHECKPOINT: these celltypes have
# ATAC only, so expect layer1_<celltype>_atac.bw and nothing else.
#
# STAGE 1 (layer1): celltype-specific ATAC peaks.
#   Peaks are called from the union (max across celltypes) of the selected
#   celltypes' ATAC bigwigs so the peak set does not favour a celltype -- a site
#   open only in DeltaLike is exactly what this benchmark is about, and a
#   BetaLike-only peak call would drop it. For each celltype pair the measured
#   log2 ratio at each peak is compared with the ratio of the two heads'
#   predictions. Stem is the reference point: pairs against it ask whether the
#   heads learned the differentiation, pairs among the three endocrine-like
#   types ask whether they learned the lineage split.
#
# --stages rna is NOT run: there is no RNA under these folders (atac.bw only).
#
REPO=/home/dmp131/C.Shark
export PYTHONPATH="$REPO/src${PYTHONPATH:+:$PYTHONPATH}"

CKPT=/mnt/jinstore/JinLab04/dmp131/C.Shark/checkpoints
model=$CKPT/jinlab_finetune_hg38.ckpt
celltypes="AlphaLike BetaLike DeltaLike Stem"   # folder names = head names

# Genes drawn as gene_loci/<gene>.png (measured | predicted, a row per
# celltype). Hormone + lineage-TF markers, two per celltype:
#   BetaLike INS MAFA   AlphaLike GCG ARX   DeltaLike SST HHEX   Stem POU5F1 NANOG
# All resolve in hg38_genes.gtf. Only HHEX is on a held-out chromosome (chr10);
# the rest are on TRAINING chromosomes, so those panels show how well the heads
# FIT a known celltype-specific locus, not whether they generalise -- the figure
# titles say which. A file path also works here: one gene per line, '#'
# comments allowed.
viz_genes="INS MAFA GCG ARX SST HHEX POU5F1 NANOG"

DATA=/mnt/jinstore/JinLab04/dmp131/cshark_data/data/hg38
seq=$DATA/dna_sequence
gtf=$DATA/hg38_genes.gtf                       # chr-prefixed, like the bigwigs
#chroms="chr1 chr2 chr3 chr4 chr5 chr6 chr7 chr8 chr9 chr10 chr11 chr12 chr13 chr14 chr15 chr16 chr17 chr18 chr19"
#chroms="chr10 chr15"
chroms="chr22"

# The union peak call is cached in ${out} as called_peaks_<chrom>_q<q>_b<bin>.bed
# and reused on every later run. THE CACHE NAME DOES NOT RECORD WHICH CELLTYPES
# THE UNION WAS TAKEN OVER: change 'celltypes' and keep this out-dir, and the new
# set is silently scored on the old set's peaks. Change the out-dir with the
# celltypes (or pass --recall-peaks) -- with 33 celltypes to choose from, this
# is the easy mistake to make.
out=outputs/celltype_eval_jinlab_finetune
mkdir -p ${out}

python -u -m cshark.inference.evaluate_celltype_specific \
    --checkpoint     ${model} \
    --features-root  ${DATA} \
    --seq-dir        ${seq} \
    --gtf            ${gtf} \
    --gtf-cache      ${REPO}/examples/outputs/hg38_gene_cache.pkl \
    --chroms ${chroms} \
    --celltypes ${celltypes} \
    --stages layer1 \
    --input-modes experimental predicted \
    --peak-quantile 0.99 \
    --peak-bin-bp 128 \
    --peak-min-width 256 \
    --min-abs-log2 1.0 \
    --min-coverage 1.0 \
    --top-n 200 \
    --volcano-rank effect \
    --volcano-label-n 12 \
    --viz-loci 8 \
    --viz-flank 20000 \
    --viz-genes ${viz_genes} \
    --viz-gene-flank 100000 \
    --agg mean \
    --batch-size 4 \
    --progress-every 10 \
    --write-bigwigs \
    --out-dir ${out}
echo "EXIT_CODE=$?"

# ---------------------------------------------------------------------------
# Outputs under ${out} (outputs/celltype_eval_jinlab_finetune/)
# ---------------------------------------------------------------------------
#   layer1_peaks_per_celltype.tsv   per peak: measured + predicted ATAC, per celltype
#   layer1_peak_predictions.tsv     the same, reshaped one row per (peak, pair)
#   rna_per_gene_predictions.tsv    per gene: measured RNA per celltype, layer-3 RNA
#                                   per celltype for BOTH input modes, plus the ATAC
#                                   that went in and the ATAC layer 3 returned
#   rna_de_<A>_vs_<B>_<mode>.tsv    the differential subset with effect sizes and
#                                   both sides' p-values / FDRs
#   metrics_summary.json            everything below, machine-readable
#                                   plus 'bigwigs': the list of files written
#
#   bigwigs/layer1_<celltype>_<track>.bw
#                                One Enformer head's prediction. 128 bp bins,
#                                LINEAR signal. Covers the union of the peak
#                                windows (stage 1) and the tiles the layer-3
#                                windows pulled (stage 2) -- with both stages and
#                                --input-modes predicted that is nearly the whole
#                                chromosome, because the gene windows very nearly
#                                tile it.
#   bigwigs/layer3_<mode>_<celltype>_<track>.bw
#                                Layer 3's own output, one file per input mode.
#                                64 bp bins here (window / target_1d_size),
#                                LINEAR signal, covering the gene windows. The
#                                pair to look at first is
#                                layer3_experimental_* against layer3_predicted_*
#                                for one celltype and track: the difference
#                                between them IS the 'gap' the input-mode
#                                comparison figure reports, shown per base
#                                instead of per gene.
#
#   The bigwigs are in the MODEL'S OWN OUTPUT SPACE: linear signal, with NO
#   depth normalisation and NO log2 median centering. Both corrections described
#   at the top of this file are properties of a COMPARISON between celltypes,
#   not of a track, so the files are left uncorrected and match the model rather
#   than the metrics. The practical consequence: do not eyeball one celltype
#   against another in a browser and conclude it is more open -- the library
#   difference the heads learned is still in there. The factors that make the
#   LAYER-1 files comparable are written out: 'layer1.size_factors.predicted' in
#   metrics_summary.json, one multiplier per celltype. Scale by those (or read
#   the corrected numbers from the TSVs) before comparing celltypes.
#
#   Bins no window covered are absent from the file (a gap, not a zero). Where
#   windows overlap, the mean of their predictions is written, so a peak-body
#   value read back out of a layer-1 bigwig agrees with layer1_peaks_per_celltype
#   .tsv to a few percent rather than exactly: the TSV aggregates whole model
#   bins on the peak-centred window's own grid, the bigwig is on a global 128 bp
#   grid and averages in every other window covering that bin. Layer-3 per-gene
#   values agree to <1% because those windows barely overlap. The TSVs remain the
#   source of truth for every number in metrics_summary.json.
#
#   layer1_atac_<A>_vs_<B>.png   Per pair: effect-size concordance, direction ROC
#                                (all peaks and the |log2|>=1 subset), predicted
#                                bias split by measured direction, accuracy vs
#                                effect size, and -- the panel specific to this
#                                benchmark -- the measured and predicted
#                                effect-size DISTRIBUTIONS on one axis with the
#                                compression factor printed.
#   layer1_summary.png           Differential skill per pair, absolute ATAC skill
#                                per celltype, and how much celltype-specific
#                                signal the heads actually produce.
#   gene_loci/<gene>.png         One per --viz-genes gene: a row per celltype,
#                                EXPERIMENTAL ATAC left, PREDICTED (layer 1)
#                                right, over the gene body +/- --viz-gene-flank,
#                                with a gene-annotation strip under each column.
#                                Same depth + size-factor normalisation as the
#                                locus figure below. Each column shares ONE
#                                y-axis across celltypes, so compare heights
#                                DOWN a column; the two columns are on different
#                                scales (coverage vs head output), so compare
#                                SHAPES across a row. Each panel also prints its
#                                gene-body mean on that normalised scale.
#   layer1_top_differential_loci.png
#                                The most differential peaks: every celltype's
#                                predicted profile over the locus, peak body
#                                shaded, plus measured-vs-predicted aggregates.
#                                EVERYTHING IN THIS FIGURE IS SIZE-FACTOR
#                                NORMALISED ACROSS CELLTYPES -- the N-celltype
#                                form of the --center-log2 median correction the
#                                metrics use pairwise (median-of-ratios, fitted
#                                on all peaks; the factors are printed on the
#                                figure and stored under
#                                'layer1.size_factors'). Without it the profiles
#                                are separated by library depth at every base,
#                                the deepest celltype is on top everywhere, and
#                                the one peak that is genuinely celltype-specific
#                                cannot be seen. The SELECTION of which peaks to
#                                show is made on the normalised spread too,
#                                otherwise "most differential" degenerates into
#                                "tallest peak in the deepest celltype".
#                                The bars are then each scaled to their own max
#                                ON TOP of that, because measured coverage and
#                                head output are still not a common unit even
#                                once the celltypes are comparable.
#                                --center-log2 none turns all of this off and the
#                                figure says so in its title.
#   rna_absolute_performance.png READ THIS FIRST. Predicted vs measured per-gene
#                                sense coverage, per celltype and input mode, on
#                                log axes. A model that cannot predict
#                                transcription at all cannot be expected to
#                                predict which celltype transcribes more.
#   rna_de_<A>_vs_<B>_<mode>.png Measured and predicted volcanoes with a shared
#                                three-way highlight (top measured only / top
#                                predicted only / top in both), effect-size
#                                concordance, and the up/down/either overlap
#                                against the hypergeometric expectation.
#   rna_celltype_chain.png       THE diagnostic figure. Median |log2 celltype
#                                ratio| at each stage -- ATAC in, layer 3's ATAC,
#                                layer 3's RNA -- with the measured RNA difference
#                                as a reference line, plus each stage's Spearman
#                                against the measured RNA ratio. Where the
#                                celltype difference dies is the whole result.
#   rna_input_mode_comparison.png Measured chromatin vs layer-1 chromatin as the
#                                layer-3 input, side by side per pair.
#
# ---------------------------------------------------------------------------
# READ THE DIAGNOSTICS BEFORE THE HEADLINE METRICS
# ---------------------------------------------------------------------------
#  * 'layer1.pairs.<pair>.normalisation.log2_offset_removed_*' -- the global offset
#    taken off each side. A large PREDICTED offset means the model learned the
#    library-depth difference; that is expected and is why centering is on.
#  * 'layer1.size_factors' -- the same correction in its N-celltype form
#    (median-of-ratios), fitted on all peaks and used by the locus figure. The
#    measured and predicted factors should be SIMILAR: the measured ones are
#    what is left of the library difference after depth normalisation, the
#    predicted ones are how much of it the heads learned. If they differ a lot,
#    the heads are not reproducing the depth structure they were fit on, which
#    is worth knowing before reading any locus.
#  * 'layer1.pairs.<pair>.all.effect_compression' -- predicted median |log2| over
#    measured. Near 0 means the per-celltype heads have collapsed onto each other
#    and layer 1 is emitting one celltype-agnostic ATAC track; every downstream
#    number is then bounded by that, and the 'predicted' input mode cannot beat
#    the shared-head baseline whatever layer 3 does.
#  * 'layer1.per_celltype.<ct>.spearman_r' -- absolute ATAC skill. Differential
#    skill is bounded by this: heads that do not predict ATAC well cannot predict
#    which celltype has more of it.
#  * 'rna.absolute.<mode>|<ct>' -- absolute transcription skill. If this is weak
#    with EXPERIMENTAL ATAC in, layer 3 is the bottleneck, not layer 1.
#  * 'rna.chain.<pair>|<mode>.stages' -- the magnitude and the information at each
#    stage. A stage with magnitude but no Spearman is passing noise through; a
#    stage with neither has dropped the signal.
#  * The two volcano y-axes are NOT the same statistic and the figure says so.
#    There is one bigwig per celltype, so the measured side has no replicates and
#    its p-value is an empirical null over the effect distribution -- a RANKING,
#    not calibrated significance. Highlighted sets are therefore chosen by EFFECT
#    SIZE (--volcano-rank effect); --volcano-rank fdr switches to the gated
#    version if you want it.
#
# Knobs worth knowing:
#  * 'checkpoint_head_index' in metrics_summary.json records which checkpoint
#    head each evaluated celltype came from, beside the full
#    'checkpoint_head_groups' list -- the provenance for every number in it. A
#    checkpoint trained on all 33 celltypes still evaluates only the four named
#    here; the other heads run (shared trunk) and are discarded.
#  * --chroms defaults to chr10 chr15, the loader's val and test chromosomes, so
#    every number is held out. Widen it for more genes and lose that.
#  * --stages layer1  runs only the peak evaluation (minutes, no layer 3).
#  * --max-peaks 200 --max-genes 100  is the smoke test.
#  * --peak-quantile controls how many peaks: 0.99 of covered 128 bp bins is a
#    few thousand per chromosome. Lower it for a broader, noisier set.
#  * --peaks-bed <file>  scores your own peak calls instead of the union call.
#  * fp32 is the default and should stay that way: celltype differences here are
#    small enough that bf16 rounding is a sizeable fraction of them. --allow-amp
#    overrides.
#  * Cost is dominated by layer 3: 2 modes x 3 celltypes = 6 forward passes per
#    2 Mb window, against ~19 Enformer tiles shared by all three celltypes.
#    Drop to --input-modes experimental to halve it.
#  * CUDA OOM on the Enformer forward? Lower --batch-size (1 works alongside a
#    training job on the same card). Measured with a 16 GB trainer resident, this
#    needed --batch-size 1-2.
#  * The peak call reads whole chromosomes off all three ATAC bigwigs (~2.4 GB
#    here) and does not depend on the model, so it is CACHED as
#    called_peaks_<chrom>_q<q>_b<bin>.bed in --out-dir. Re-running against another
#    checkpoint with the same --out-dir skips it; --recall-peaks forces the redo.
#    Copy that BED into a new --out-dir to reuse it there.
#  * The first run on a cold filesystem is I/O bound, not GPU bound: the peak call
#    alone took ~20 min over NFS here and seconds on the second run.
#  * --write-bigwigs holds its accumulator until BOTH stages finish, because the
#    layer-1 files are filled from stage 1 and stage 2 alike. Drop the flag if
#    the ~0.4 GiB matters, or coarsen with --bigwig-bin-bp 128 / 256.
#    --bigwig-dir puts the files somewhere other than <out-dir>/bigwigs, which is
#    what you want when re-running against a second checkpoint and the peak cache
#    is being shared via --out-dir.
#  * --stages layer1 --write-bigwigs writes ONLY the layer-1 files, and covers
#    just the peak windows rather than most of the chromosome. Use it to eyeball
#    the heads quickly; use both stages for tracks worth keeping.
