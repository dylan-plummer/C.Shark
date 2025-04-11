python3 src/cshark/training/train.py \
    --data-root cshark_data/data \
    --celltypes hESC_WT_50pct \
    --input-features ctcf h3k27ac h3k4me3 \
    --target-features ctcf h3k27ac h3k4me3 h3k9me3 \
    --latent-dim 64 \
    --assembly hg19 \
    --num-gpu 1 \
    --batch-size 2 \
    --num-workers 8