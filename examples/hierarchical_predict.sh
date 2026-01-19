data_root="/mnt/jinstore/JinLab03/xxl1432/HiCorr_Deeploop/11.C.shark_checkpoints/cshark_data/data/"

python ../src/cshark/inference/hierarchical_predict.py \
  --model "/mnt/jinstore/JinLab02/dmp131/C.Shark/checkpoints/mESC_hierarchical_64bp_simplified.ckpt" \
  --out "outputs/chr7_rad21_ko_results.tsv" \
  --chrom "chr7" \
  --locus chr7:126656644-127322620 \
  --assembly "mm10" \
  --seq "${data_root}/mm10/dna_sequence" \
  --ctcf "${data_root}/mm10/Hsieh_CTCF_WT_5kb_norm_10/genomic_features/ctcf.bw" \
  --atac "${data_root}/mm10/Hsieh_CTCF_WT_5kb_norm_10/genomic_features/atac.bw" \
  --hic-wt /mnt/jinstore/JinLab03/xxl1432/HiCorr_Deeploop/11.C.shark_checkpoints/preprocessing/1.to_cooler/7.Hsieh_WT_5kb/Hsieh_CTCF_WT_5kb_deeploop.cool \
  --hic-ko /mnt/jinstore/JinLab03/xxl1432/HiCorr_Deeploop/11.C.shark_checkpoints/preprocessing/1.to_cooler/8.Hsieh_KO_5kb/Hsieh_CTCF_KO_5kb_deeploop.cool \
  --window 2097152 \
  --step-size 1000000