checkpoint=checkpoints/deeploop_hESC_EP_CTCF.ckpt
seq=cshark_data/data/hg19/dna_sequence
ko_mode=knockout  # either zero, mean, or knockout


# heatmap region: chr11:9,733,614-10,791,870， FIN region: chr11:10,422,643-10,438,132
# first do a screen of the whole region
python src/cshark/inference/ctcf.py --celltype hESC_WT_50pct \
                 --outname ctcf_screen \
                 --chr chr11 \
                 --model $checkpoint \
                 --latent_size 256 \
                 --seq $seq \
                 --ko-mode $ko_mode \
                 --out-file outputs/chr11_ctcf_KO.tsv \
                 --ctcf cshark_data/data/hg19/hESC_WT_50pct/genomic_features/ctcf.bw \
                 --h3k27ac cshark_data/data/hg19/hESC_WT_50pct/genomic_features/h3k27ac.bw \
                 --h3k4me3 cshark_data/data/hg19/hESC_WT_50pct/genomic_features/h3k4me3.bw