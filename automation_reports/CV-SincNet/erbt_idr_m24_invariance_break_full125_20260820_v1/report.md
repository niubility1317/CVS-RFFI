# ERBT-IDR M2.4非等价机制完整125实验报告

日期：2026-08-20

run ID：`erbt_idr_m24_invariance_break_full125_20260820_v1`

当前状态：`LANDED / REAL-CACHE-SMOKE-PASS / READY_TO_LAUNCH`

## 一、目标与机制

本实验针对提交`8d712e7cb35e4c908f9975357132c10656f26a76`复盘发现的代数等价问题：旧`M24-D1-REFIT`把IF256补零后重新送入P2-A1拟合器，实际仍使用同一特征、中心、共享协方差和LDA目标，注册后172500个query相对去RF32的D92 E0得到`N_help=0`、`N_harm=0`。

本轮实现提交为`703b7d07a2ec77e40f4f9e29e4b534af98c5dc34`。新路线不调用P2-A1协方差/LDA拟合器：G1冻结50%identity/50%FFT的平衡IF256余弦原型头；G2增加support-only、类中心张成空间正交的rank-1硬投影；G3增加类别对称的不确定性惩罚；G4在K≥5时增加确定性双原型和按类归一化log-mean-exp。K1直接使用自己的冻结原型头，K2使用投影单原型，不再强制退回历史F1。

## 二、完整矩阵

|arm|身份|
|---|---|
|G0|`M24-D0-HISTORICAL-F1`，当前D92 E0去RF32主基线|
|G1|`M24-G1-FROZEN-BALANCED-PROTOTYPE`|
|G2|`M24-G2-ORTHOGONAL-NUISANCE`|
|G3|`M24-G3-CLASS-UNCERTAINTY`|
|G4|`M24-G4-LOCAL-DUAL-PROTOTYPE`|

- receiver：`20-1`、`3-19`、`7-14`、`7-7`、`8-8`；
- method seed：`7282101`至`7282105`；
- 条件：`K1/new20`、`K2/new20`、`K5/new20`、`K10/new20`、`K10/new5`；
- 每个arm完整125组，共625个方法行、1875个场景单元；
- 所有arm复用相同`capsule_id`、`split_id`、support/query物理身份和固定received IQ。

## 三、协议与本地验证

- `protocol_schema=p2_min_v1`，`phase2_data_status=VALIDATED_ONCE`；
- 数据身份未改变，不因方法和头状态变化重验数据；
- 拟合API不接收query或truth，query逐样本对所有注册类独立决策；
- prediction完整后才允许独立scorer连接truth；
- 48项聚焦回归、Python编译、矩阵/scorer静态闭合和`git diff --check`通过；
- 一次独立P0/P1审查结论为`NO_P0_P1`；
- `REJECTED_EXTRA_GATE`：非等价检查仅作为科学诊断和停止规则，不增加发布审核门。

## 四、N607输入、输出与资源

|字段|路径|
|---|---|
|既有feature root|`/home/szu2070436088/2510044040/CV-SincNet/stage2_inputs/cvs_full_ablation_phase2c_feature_t1_20260730_v3_47212437/artifacts/features`|
|补充feature root|`/home/szu2070436088/2510044040/CV-SincNet/stage2_inputs/erbt_idr_m24_d1_refit_full125_20260820_v1_supplement_features/artifacts/features`|
|既有scoring root|`/home/szu2070436088/2510044040/CV-SincNet/stage2_inputs/cvs_full_ablation_phase2c_sidecar_t1_20260730_v3_47212437/artifacts/sidecars`|
|补充scoring root|`/home/szu2070436088/2510044040/CV-SincNet/stage2_inputs/erbt_idr_m24_d1_refit_full125_20260820_v1_supplement_packages`|
|release root|`/home/szu2070436088/2510044040/CV-SincNet/releases/erbt_idr_m24_invariance_break_full125_20260820_v1`|
|run root|`/home/szu2070436088/2510044040/CV-SincNet/runs/erbt_idr_m24_invariance_break_full125_20260820_v1`|
|log root|`/home/szu2070436088/2510044040/CV-SincNet/logs/erbt_idr_m24_invariance_break_full125_20260820_v1`|
|Python|`/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python`|
|资源|CPU，`--max-workers 2`；不占用GPU，不干预既有Phase1训练|

N607只读预检已确认项目根、两组feature root和两组scoring root可见，新run/log根不存在。

单一release归档本地与N607的SHA-256均为`ef29aad0e47a7c635b050d7be6efad66fa74804d6ba65cc2924da7ea9cff53fd`，远端编译通过。真实cache无query smoke使用`rx3-19/m7282101/K1/new20`，G0–G4共5行全部生成`PREDICTIONS_COMPLETE_TRUTH_UNOPENED` prediction并返回`PASS`；smoke位于本run的独立`smoke`子目录，不进入625行正式矩阵。

## 五、冻结命令

prediction命令：

```bash
cd /home/szu2070436088/2510044040/CV-SincNet/releases/erbt_idr_m24_invariance_break_full125_20260820_v1/code
PYTHONPATH=. /home/szu2070436088/.conda/envs/CVS-RFFI/bin/python -u scripts/run_m24_invariance_breaking_full125.py --run-id erbt_idr_m24_invariance_break_full125_20260820_v1 --feature-root /home/szu2070436088/2510044040/CV-SincNet/stage2_inputs/cvs_full_ablation_phase2c_feature_t1_20260730_v3_47212437/artifacts/features --supplemental-feature-root /home/szu2070436088/2510044040/CV-SincNet/stage2_inputs/erbt_idr_m24_d1_refit_full125_20260820_v1_supplement_features/artifacts/features --output-root /home/szu2070436088/2510044040/CV-SincNet/runs/erbt_idr_m24_invariance_break_full125_20260820_v1/predictions --device cpu --max-workers 2
```

prediction闭合后运行truth-last scorer：

```bash
PYTHONPATH=. /home/szu2070436088/.conda/envs/CVS-RFFI/bin/python -u scripts/score_m24_invariance_breaking_full125.py --matrix-index /home/szu2070436088/2510044040/CV-SincNet/runs/erbt_idr_m24_invariance_break_full125_20260820_v1/predictions/matrix_index.json --scoring-root /home/szu2070436088/2510044040/CV-SincNet/stage2_inputs/cvs_full_ablation_phase2c_sidecar_t1_20260730_v3_47212437/artifacts/sidecars --supplemental-scoring-root /home/szu2070436088/2510044040/CV-SincNet/stage2_inputs/erbt_idr_m24_d1_refit_full125_20260820_v1_supplement_packages --output-root /home/szu2070436088/2510044040/CV-SincNet/runs/erbt_idr_m24_invariance_break_full125_20260820_v1/scores --bootstrap-repeats 2000
```

## 六、停止规则与预期artifact

仅在协议/query泄漏、错误矩阵身份、输出碰撞、错误checkout、无法启动、prediction不闭合、scorer连接错误或至少两行相同确定性prediction前异常时停止并保留证据。低性能不得停止。

prediction闭合要求：`matrix_index.status=PREDICTIONS_COMPLETE_TRUTH_UNOPENED`、`row_count=625`、`paired_input_identity_count=125`且G0–G4各125行。闭合后生成625个same-row score、625个four-state score、500个paired-vs-G0结果、500个`F_within/F_std`结果，并汇总总体、K/new、receiver、seed、scene、old/new、class、margin、中心角距、help/harm、状态差异和资源指标。

## 七、结果

待N607完整prediction与truth-last评分后填写。当前没有性能结论。
