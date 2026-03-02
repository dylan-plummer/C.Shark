data_root="/mnt/jinstore/JinLab03/xxl1432/HiCorr_Deeploop/11.C.shark_checkpoints/cshark_data/data/"
model=/mnt/jinstore/JinLab02/dmp131/C.Shark/checkpoints/mESC_hierarchical_full.ckpt

chrom="chr18"
region="chr18:37616977-38325119"

python ../src/cshark/inference/hierarchical_predict_with_enformer.py \
  --model ${model} \
  --out "outputs/chr18_alt_a_results.tsv" \
  --chrom ${chrom} \
  --locus ${region} \
  --assembly "mm10" \
  --seq "${data_root}/mm10/dna_sequence" \
  --ctcf "${data_root}/mm10/Hsieh_CTCF_WT_5kb_norm_10/genomic_features/ctcf.bw" \
  --atac "${data_root}/mm10/Hsieh_CTCF_WT_5kb_norm_10/genomic_features/atac.bw" \
  --window 2097152 \
  --vmin 0.1

chrom=chr7
region=chr7:126656644-127322620

python ../src/cshark/inference/hierarchical_predict_with_enformer.py \
  --model ${model} \
  --out "outputs/chr7_alt_a_results.tsv" \
  --chrom ${chrom} \
  --locus ${region} \
  --assembly "mm10" \
  --seq "${data_root}/mm10/dna_sequence" \
  --ctcf "${data_root}/mm10/Hsieh_CTCF_WT_5kb_norm_10/genomic_features/ctcf.bw" \
  --atac "${data_root}/mm10/Hsieh_CTCF_WT_5kb_norm_10/genomic_features/atac.bw" \
  --window 2097152 \
  --vmin 0.1