# perturb.py 重构 — 进度与续作说明 (RESUME)

> 分支：`refactor-perturb`。旧 `src/cshark/inference/perturb.py` / `perturb_cpu.py` 未改动。
> 新代码全部在 `src/cshark/perturb/`。包名暂留 `cshark`。

## 当前状态（已完成 + 已验证）

`single_deletion`（旧单点位主流程）已**程序化逐字节转写**为 `src/cshark/perturb/scopes/single_locus.py:run_single_locus(cfg)`，新 CLI 入口 `python -m cshark.perturb.cli` 对 `--start` 分支调用它。

已在**最复杂路径**（single-locus + `enformer_seq` SNP + hierarchical RAD21 + 8 tracks + `--plot-diff`）、**两个独立位点 JAK2 与 ZFAT** 上验证：新引擎 ③ 与 Archive03 旧脚本 ② 的全部输出**逐字节/数值完全一致**（47 个 tmp 文件 cooler/bigwig/bed/ini 全 `diff=0`，`*pred_tracks.png`/`*ko_tracks.png` md5 相同）。

「深化重构」**Step 1（output helper）已完成并复验 diff=0**：`single_locus.py` 里两处 `plt.imshow`→`plot_prediction_matrix`，regions→`write_regions`，4 段 arcs→`write_arcs`。

## 关键背景

- **Archive03 是重构目标**。用户存档的旧结果是用 `/mnt/jinstore/JinLab05/xxl1332/C.Shark`（2026-04 旧快照）跑的，其 enformer/hierarchical 与 Archive03 不同（Archive03 之后有 a052e2f / 83f9f40 / 8dfafeb / 198ba3e 等修复）。所以验证基准是 **my_old(Archive03) vs my_new**，不是 xxl1332 的结果。
- 环境：`cshark` 是 **非 editable 安装**（site-packages 是拷贝）。测试一律 `PYTHONPATH=<repo>/src`、**从 repo 根目录**运行。GPU 0 可用，enformer 权重已在 HF 缓存。

## 验证方法（每步回归循环）

基准固定（Archive03 旧引擎产物）：`<scratch>/jak2_old/tmp`、`<scratch>/zfat_old/tmp`。
每改一处 → 重跑新引擎两位点 → 比对 → 必须 **47/47 一致 + 2 张 PNG md5 匹配** → `git commit` → 下一步。
> scratchpad 是会话专属目录；若丢失，用旧 `inference/perturb.py` 重跑两位点重建基准。

测试位点（公共参数：`--celltype GM12878_80pct_deeploop_5kb_encode_norm_10 --matrix-size 512 --resolution 4096 --seq-filter-size 15 --latent_size 256 --ko seq --ko-mode enformer_seq --ko-width 1 --plot-diff --ctcf-motif-p 500 --silent --enformer-delta-mode multiplicative --hierarchical-delta-mode multiplicative --enformer-tracks ctcf atac rad21 h3k27ac h3k4me3 h3k9me3 h3k36me3 h3k27me3`，8 条 bigwig）：

| 位点 | --chr | --start | --ko-start (snp) | --alt | --region |
|---|---|---|---|---|---|
| JAK2 | chr9 | 4950000 | 5186616 | C | chr9:4950000-5520000 |
| ZFAT | chr8 | 134437757 | 134538345 | T | chr8:134437757-134707757 |

路径：
- model `…/models_hg38/3.GM12878_80pct_5kb_8_inputs_norm_10_8_training_remove_no_recon_hg38/models/GM12878_80pct_5kb_8_inputs_norm_10_8_training_remove_no_recon_hg38.ckpt`
- hierarchical `…/models_hg38/3.GM12878_WT_layer2_5kb_norm_10_7_tracks_pred_rad21/models/GM12878_WT_layer2_5kb_norm_10_7_tracks_pred_rad21.ckpt`
- data root `/mnt/jinstore/JinLab03/xxl1432/HiCorr_Deeploop/11.C.shark_checkpoints/cshark_data/data`（seq: `hg38/dna_sequence`；bigwig: `hg38/<cell_line>/genomic_features/{ctcf,atac,rad21,h3k27ac,h3k4me3,h3k9me3,h3k36me3,h3k27me3}.bw`）

旧/新运行命令模板见 `verify_refactor.sh`（回归脚手架，会跑两位点新引擎并对比）。对比脚本 `compare_dirs.py`（cooler 矩阵 / bigwig intervals / bed-ini 文本）。

## 明天从这里继续 —— 深化重构剩余步骤（逐步、逐测）

2. **【下一步】CSharkModel（加载一次，修性能 bug）**：在 `single_locus.py` 的 `num_genomic_features`（约 200–204 行）算完后构造一次 `model = CSharkModel(cfg, num_genomic_features=num_genomic_features, diploid=diploid)`；把两处 `infer.prediction(...)`（WT ~207–215、KO ~482–491）换成 `model.predict_arrays(seq_region, ctcf_region, atac_region, other_regions, input_track_names[2:])`。需给 `models/base.py` 的 `CSharkModel` 加一个公开 `predict_arrays`（即现有 `_predict_arrays`）。CSharkModel 的 `load_default` 参数已与 `infer.prediction` 内部一致。
3. **tracks_ini 构建器**（~370 行，`single_locus.py` ~667–960）→ `output/tracks_ini.py`。
4. **hierarchical** 两块（补 rad21 + 算 delta/写 bigwig）→ `models/hierarchical.py` 函数。
5. **enformer_seq** 块 → `models/enformer.py` 函数。
6. 把 `run_single_locus` 收成 ~80 行编排者；接好 `registry`；随后**移植 full_chrom**（复用上面抽出的 CSharkModel/enformer/hierarchical/output）。
