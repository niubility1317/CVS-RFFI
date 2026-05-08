# 5.8 训练日志分析报告

日期：2026-05-08

## 分析范围

- 日志来源：`5.8/logs`。
- 已解析实验日志：19 个；完整结束：19 个。
- 队列结构：B/C/D 被合并为 `PRE` 大队列以占满 8 张 GPU；E/SGC 在 PRE 完成后最后运行。
- 所有实验固定 seed=1337。
- 严谨性提示：`FINAL-PRIMARY` 与 Phase-E source 自动选择使用了 test 指标。因此 5.8 结果应作为开发集证据，不应直接作为最终无偏 test 结果报告。

## 启动器选择

- SRC-P: `finalist_runs/D1_domain_enhancer_off_seed1337/best_model_primary_ood.pth` from `D1_domain_enhancer_off_seed1337_seed1337`.
- SRC-S: `finalist_runs/D4_domain_no_pa_no_stats_seed1337/best_model_primary_ood.pth` from `D4_domain_no_pa_no_stats_seed1337_seed1337`.
- 启动器失败记录：0。

## Primary 指标排序

|name|phase|final_primary_score|final_primary_overall|final_primary_strict|best_worst_rx|best_worst_rx_name|sat_avg|final_primary_val|
|---|---|---|---|---|---|---|---|---|
|E2_residual_only_std_res001|E|88.24|90.70|86.92|86.99|test_rx_8|41.58|98.90|
|E3_no_res_control|E|88.12|90.76|86.69|86.27|test_rx_7|40.86|98.90|
|E1_residual_only_std|E|87.91|90.46|86.53|86.63|test_rx_7|41.20|98.85|
|D1_domain_enhancer_off_seed1337_seed1337|D|87.87|90.52|86.45|85.12|test_rx_8|40.99|98.91|
|E0_no_adapter_continue|E|87.84|90.37|86.47|86.51|test_rx_7|40.85|98.96|
|D4_domain_no_pa_no_stats_seed1337_seed1337|D|87.65|90.75|85.99|86.13|test_rx_8|42.76|98.89|
|B4_A2_ls002_seed1337_seed1337|B|87.61|90.35|86.14|85.62|test_rx_8|42.25|98.99|
|E4_full_sgc_mild_res001|E|87.61|90.30|86.16|85.95|test_rx_8|40.63|98.87|
|D6_no_orth_seed1337_seed1337|D|87.49|90.40|85.92|84.71|test_rx_8|41.67|98.95|
|C2_A1_ecc003_satmain_seed1337_seed1337|C|87.34|90.07|85.86|85.45|test_rx_8|43.16|98.85|
|B3_A1_ls002_seed1337_seed1337|B|87.25|90.23|85.64|86.03|test_rx_8|43.56|98.96|
|B1_A1_mild_seed1337_seed1337|B|87.24|89.96|85.78|85.41|test_rx_7|43.63|98.97|
|B2_A2_light_seed1337_seed1337|B|87.23|89.82|85.84|85.09|test_rx_8|42.19|98.88|
|D2_domain_rcn020_seed1337_seed1337|D|87.19|90.20|85.56|85.02|test_rx_8|41.82|98.94|
|D5_no_grl_adv_seed1337_seed1337|D|87.06|90.14|85.40|85.31|test_rx_8|41.30|98.99|
|C3_A1_ecc003_conservative_seed1337_seed1337|C|87.05|90.21|85.35|84.62|test_rx_8|43.65|98.90|
|C1_A1_ecc002_sat_seed1337_seed1337|C|86.98|90.06|85.32|85.60|test_rx_8|43.30|98.94|
|C4_A2_ecc003_satmain_seed1337_seed1337|C|86.81|89.95|85.11|84.54|test_rx_8|40.07|98.88|
|D3_domain_branch_same_seed1337_seed1337|D|86.08|89.37|84.31|85.14|test_rx_7|41.50|98.95|

## SAT 鲁棒性排序

|name|phase|sat_avg|sat_min|final_primary_score|final_primary_strict|best_worst_rx|
|---|---|---|---|---|---|---|
|C3_A1_ecc003_conservative_seed1337_seed1337|C|43.65|39.27|87.05|85.35|84.62|
|B1_A1_mild_seed1337_seed1337|B|43.63|39.89|87.24|85.78|85.41|
|B3_A1_ls002_seed1337_seed1337|B|43.56|39.26|87.25|85.64|86.03|
|C1_A1_ecc002_sat_seed1337_seed1337|C|43.30|39.02|86.98|85.32|85.60|
|C2_A1_ecc003_satmain_seed1337_seed1337|C|43.16|38.78|87.34|85.86|85.45|
|D4_domain_no_pa_no_stats_seed1337_seed1337|D|42.76|38.56|87.65|85.99|86.13|
|B4_A2_ls002_seed1337_seed1337|B|42.25|38.15|87.61|86.14|85.62|
|B2_A2_light_seed1337_seed1337|B|42.19|38.20|87.23|85.84|85.09|
|D2_domain_rcn020_seed1337_seed1337|D|41.82|38.34|87.19|85.56|85.02|
|D6_no_orth_seed1337_seed1337|D|41.67|38.10|87.49|85.92|84.71|
|E2_residual_only_std_res001|E|41.58|37.27|88.24|86.92|86.99|
|D3_domain_branch_same_seed1337_seed1337|D|41.50|37.99|86.08|84.31|85.14|
|D5_no_grl_adv_seed1337_seed1337|D|41.30|37.62|87.06|85.40|85.31|
|E1_residual_only_std|E|41.20|36.84|87.91|86.53|86.63|
|D1_domain_enhancer_off_seed1337_seed1337|D|40.99|36.90|87.87|86.45|85.12|
|E3_no_res_control|E|40.86|35.61|88.12|86.69|86.27|
|E0_no_adapter_continue|E|40.85|36.04|87.84|86.47|86.51|
|E4_full_sgc_mild_res001|E|40.63|36.12|87.61|86.16|85.95|
|C4_A2_ecc003_satmain_seed1337_seed1337|C|40.07|36.26|86.81|85.11|84.54|

## 按验证集选 checkpoint 的视角

|name|phase|best_val_tx|paired_test_at_best_val|final_best_val|final_best_overall|final_primary_score|
|---|---|---|---|---|---|---|
|E0_no_adapter_continue|E|98.96|90.37|98.96|90.37|87.84|
|D4_domain_no_pa_no_stats_seed1337_seed1337|D|99.00|90.31|99.00|90.31|87.65|
|D5_no_grl_adv_seed1337_seed1337|D|99.08|90.05|99.08|90.05|87.06|
|B4_A2_ls002_seed1337_seed1337|B|99.06|90.03|99.06|90.03|87.61|
|E1_residual_only_std|E|98.97|89.86|98.97|89.86|87.91|
|D6_no_orth_seed1337_seed1337|D|99.05|89.80|99.05|89.80|87.49|
|C3_A1_ecc003_conservative_seed1337_seed1337|C|99.05|89.71|99.05|89.71|87.05|
|E4_full_sgc_mild_res001|E|98.93|89.61|98.93|89.61|87.61|
|C2_A1_ecc003_satmain_seed1337_seed1337|C|99.07|89.59|99.07|89.59|87.34|
|E2_residual_only_std_res001|E|98.97|89.54|98.97|89.54|88.24|
|C4_A2_ecc003_satmain_seed1337_seed1337|C|99.08|89.48|99.08|89.48|86.81|
|D1_domain_enhancer_off_seed1337_seed1337|D|99.04|89.47|99.04|89.47|87.87|
|D2_domain_rcn020_seed1337_seed1337|D|99.03|89.47|99.03|89.47|87.19|
|B3_A1_ls002_seed1337_seed1337|B|99.05|89.35|99.05|89.35|87.25|
|B1_A1_mild_seed1337_seed1337|B|99.04|89.25|99.04|89.25|87.24|
|B2_A2_light_seed1337_seed1337|B|99.05|89.18|99.05|89.18|87.23|
|C1_A1_ecc002_sat_seed1337_seed1337|C|99.05|89.14|99.05|89.14|86.98|
|E3_no_res_control|E|98.95|88.99|98.95|88.99|88.12|
|D3_domain_branch_same_seed1337_seed1337|D|99.04|88.96|99.04|88.96|86.08|

## 关键结论

1. 5.8 开发集 primary 最强的是 `E2_residual_only_std_res001`：score 88.24，overall 90.70，strict UDU 86.92，worst-RX 86.99，SAT Avg 41.58。这是当前最强候选，但它继承了 Phase-E 基于 test-derived primary 指标选择 source 的问题。
2. PRE 阶段最强 source 是 `D1_domain_enhancer_off`：score 87.87，overall 90.52，strict UDU 86.45。它说明在当前 split 下，第二骨干的 RCN stats enhancer 可能存在过度域化，使发射机判别特征被域特征牵引。
3. `D4_domain_no_pa_no_stats` 同样很强：score 87.65，overall 90.75，strict UDU 85.99，worst-RX 86.13。它挑战了“PA auxiliary 必须保留”的旧判断，但需要 validation-only 和多 seed 复核。
4. label smoothing 对 light SAT 路线有帮助：`B4_A2_ls002` 相比 B1 primary +0.37、strict UDU +0.36，但 SAT Avg 下降约 1.38。
5. ECC 没有成为默认赢家。`C2_A1_ecc003_satmain` 只比 B1 primary 高 +0.10，而 `C4_A2_ecc003_satmain` 让 SAT Avg 比 B1 低约 3.56。ECC 应暂时保留为探索项，需要减弱或重新安排作用时段。
6. residual-only SGC 明确有用，full SGC 当前不建议。相对 `E0_no_adapter_continue`，`E2` 提升 primary +0.40、strict UDU +0.45、worst-RX +0.48、SAT Avg +0.73；而 `E4_full_sgc_mild_res001` primary -0.23、worst-RX -0.56。
7. `E3_no_res_control` primary 88.12、overall 90.76 也很强，但 SAT Avg 和 worst-RX 弱于 E2；它适合作为控制组，不是优先部署候选。

## 下一步推荐路线

- 开发路线赢家：继续研究 `E2_residual_only_std_res001`。
- 架构方向：Lite-B no-DAC，domain enhancer 关闭或显著减弱，SGC 只保留 residual-only，并设置 `lambda_res=0.01`；保留 conservative MixStyle 与 Fishr。
- 严谨性修正：下一轮必须用 validation-only 选择 checkpoint/source，或建立全新的 final holdout。当前 5.8 primary 排名不能作为最终无偏 test 结论。
- 下一轮 shortlist：`D1`、`D4`、`B4`、`E0`、`E2`、`E3`；`C2` 仅作为 ECC 参考组保留。

## 已管理文件

- `5.8/metrics/experiment_metrics.csv`：扁平指标表，适合直接打开对比。
- `5.8/metrics/experiment_metrics.json`：完整解析指标，适合后续脚本复用。
- `5.8/reports/5_8_training_analysis_20260508.md`：本分析报告。
- `5.8/README.md`：文件夹索引与解释说明。
