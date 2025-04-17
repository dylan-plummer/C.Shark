checkpoint=../checkpoints/deeploop_hESC_EP_CTCF.ckpt
seq=../cshark_data/data/hg19/dna_sequence


python ../src/cshark/inference/seq_perturb.py \
    --celltype hESC_WT_50pct \
    --outname seq_perturb \
    --chr chr10 \
    --start 14000000 \
    --model $checkpoint \
    --seq $seq \
    --region chr10:14980000-15200000 \
    --var-pos 15033505 15033497 \
    --alt A G \
    --min-val-pred 0.2 \
    --ctcf ../cshark_data/data/hg19/hESC_WT_50pct/genomic_features/ctcf.bw \
    --h3k27ac ../cshark_data/data/hg19/hESC_WT_50pct/genomic_features/h3k27ac.bw \
    --h3k4me3 ../cshark_data/data/hg19/hESC_WT_50pct/genomic_features/h3k4me3.bw


python ../src/cshark/inference/seq_perturb.py \
    --celltype hESC_WT_50pct \
    --outname seq_perturb \
    --chr chr10 \
    --start 14000000 \
    --model $checkpoint \
    --seq $seq \
    --region chr10:14980000-15200000 \
    --var-pos 15033505 15033497 \
    --alt A G \
    --plot-diff \
    --min-val-pred -0.0001 \
    --max-val-pred 0.0001 \
    --ctcf ../cshark_data/data/hg19/hESC_WT_50pct/genomic_features/ctcf.bw \
    --h3k27ac ../cshark_data/data/hg19/hESC_WT_50pct/genomic_features/h3k27ac.bw \
    --h3k4me3 ../cshark_data/data/hg19/hESC_WT_50pct/genomic_features/h3k4me3.bw

# python ../src/cshark/inference/seq_perturb.py \
#     --celltype hESC_WT_50pct \
#     --outname seq_perturb \
#     --chr chr11 \
#     --start 9000000 \
#     --model $checkpoint \
#     --seq $seq \
#     --region chr11:9733614-10791870 \
#     --var-pos 9833614 \
#     --alt T \
#     --min-val-pred 0.2 \
#     --ctcf ../cshark_data/data/hg19/hESC_WT_50pct/genomic_features/ctcf.bw \
#     --h3k27ac ../cshark_data/data/hg19/hESC_WT_50pct/genomic_features/h3k27ac.bw \
#     --h3k4me3 ../cshark_data/data/hg19/hESC_WT_50pct/genomic_features/h3k4me3.bw


# python ../src/cshark/inference/seq_perturb.py \
#     --celltype hESC_WT_50pct \
#     --outname seq_perturb \
#     --chr chr11 \
#     --start 9000000 \
#     --model $checkpoint \
#     --seq $seq \
#     --region chr11:9733614-10791870 \
#     --var-pos 9833614 9833616 \
#     --alt T A \
#     --plot-diff \
#     --min-val-pred -0.001 \
#     --max-val-pred 0.001 \
#     --ctcf ../cshark_data/data/hg19/hESC_WT_50pct/genomic_features/ctcf.bw \
#     --h3k27ac ../cshark_data/data/hg19/hESC_WT_50pct/genomic_features/h3k27ac.bw \
#     --h3k4me3 ../cshark_data/data/hg19/hESC_WT_50pct/genomic_features/h3k4me3.bw