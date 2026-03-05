data_root="/mnt/jinstore/JinLab03/xxl1432/HiCorr_Deeploop/11.C.shark_checkpoints/cshark_data/data/"
model=/mnt/jinstore/JinLab02/dmp131/C.Shark/checkpoints/mESC_hierarchical_full_finetune.ckpt

chrom="chr18"
region="chr18:37616977-38325119"

python ../src/cshark/inference/hierarchical_predict_with_enformer.py \
  --model ${model} \
  --out "outputs/pcdh_CTCF_KO.tsv" \
  --chrom ${chrom} \
  --locus ${region} \
  --assembly "mm10" \
  --seq "${data_root}/mm10/dna_sequence" \
  --ctcf "${data_root}/mm10/Hsieh_CTCF_WT_5kb_norm_10/genomic_features/ctcf.bw" \
  --atac "${data_root}/mm10/Hsieh_CTCF_WT_5kb_norm_10/genomic_features/atac.bw" \
  --window 2097152 \
  --ko ctcf \
  --ko-mode knockout \
  --vmin 0.1

chrom=chr7
region=chr7:126656644-127322620

python ../src/cshark/inference/hierarchical_predict_with_enformer.py \
  --model ${model} \
  --out "outputs/cdiptos_CTCF_KO.tsv" \
  --chrom ${chrom} \
  --locus ${region} \
  --assembly "mm10" \
  --seq "${data_root}/mm10/dna_sequence" \
  --ctcf "${data_root}/mm10/Hsieh_CTCF_WT_5kb_norm_10/genomic_features/ctcf.bw" \
  --atac "${data_root}/mm10/Hsieh_CTCF_WT_5kb_norm_10/genomic_features/atac.bw" \
  --window 2097152 \
  --ko ctcf \
  --ko-mode knockout \
  --vmin 0.1

chrom=chr12
region=chr12:56391707-56856901

python ../src/cshark/inference/hierarchical_predict_with_enformer.py \
  --model ${model} \
  --out "outputs/chr12_rad21_ko.tsv" \
  --chrom ${chrom} \
  --locus ${region} \
  --assembly "mm10" \
  --seq "${data_root}/mm10/dna_sequence" \
  --ctcf "${data_root}/mm10/Hsieh_CTCF_WT_5kb_norm_10/genomic_features/ctcf.bw" \
  --atac "${data_root}/mm10/Hsieh_CTCF_WT_5kb_norm_10/genomic_features/atac.bw" \
  --window 2097152 \
  --ko rad21 \
  --ko-mode knockout \
  --vmin 0.1 \
  --vmax 1.2

# chr1:4,424,705-4,836,352
chrom=chr1
region=chr1:4424705-4836352

python ../src/cshark/inference/hierarchical_predict_with_enformer.py \
  --model ${model} \
  --out "outputs/chr1_rad21_ko.tsv" \
  --chrom ${chrom} \
  --locus ${region} \
  --assembly "mm10" \
  --seq "${data_root}/mm10/dna_sequence" \
  --ctcf "${data_root}/mm10/Hsieh_CTCF_WT_5kb_norm_10/genomic_features/ctcf.bw" \
  --atac "${data_root}/mm10/Hsieh_CTCF_WT_5kb_norm_10/genomic_features/atac.bw" \
  --window 2097152 \
  --ko rad21 \
  --ko-mode knockout \
  --vmin 0.1 \
  --vmax 1.2

python ../src/cshark/inference/hierarchical_predict_with_enformer.py \
  --model ${model} \
  --out "outputs/chr19_alt_a_results.tsv" \
  --chrom chr19 \
  --assembly "mm10" \
  --seq "${data_root}/mm10/dna_sequence" \
  --ctcf "${data_root}/mm10/Hsieh_CTCF_WT_5kb_norm_10/genomic_features/ctcf.bw" \
  --atac "${data_root}/mm10/Hsieh_CTCF_WT_5kb_norm_10/genomic_features/atac.bw" \
  --window 2097152 \
  --vmin 0.1