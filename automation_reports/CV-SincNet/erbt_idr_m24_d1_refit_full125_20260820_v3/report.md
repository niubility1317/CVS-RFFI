# ERBT-IDR M2.4 D1-REFIT完整125修复重跑报告

日期：2026-08-20

run ID：`erbt_idr_m24_d1_refit_full125_20260820_v3`

当前状态：`LOCAL_VERIFIED`

## 一、目标与修复身份

本run是v2启动路径技术失败后的不可覆盖继任run。算法与代码不变，继续验证D1源精度保持修复：

- FP32注册前源头使用F0 IF256主状态；
- 已量化F3注册后源头直接裁剪前256维code/scale，不重新量化；
- 注册前／后R1相对R0必须逐query零差异；
- D92 E0主基线固定为去RF32的`P2-A1_NO_RF32`。

实现提交：`1ca297dc1d5c44f6ec993abc58c8c1dc4208e89b`。

冻结release：`/home/szu2070436088/2510044040/CV-SincNet/releases/erbt_idr_m24_d1_refit_full125_20260820_v2`，对应archive SHA-256：`c5974d71fff4c04a3e2fed81c9a73ffddc29152f038e7f6673b84847731480a1`。该release已完成本地／远端SHA一致性和远端编译验证，不因仅修正输入根路径而重新发布。

## 二、完整矩阵

- receiver：`20-1`、`3-19`、`7-14`、`7-7`、`8-8`；
- method seed：`7282101`至`7282105`；
- 条件：`K1/new20`、`K2/new20`、`K5/new20`、`K10/new20`、`K10/new5`；
- R0/R1/R2各125行，总计375行、1125个场景单元；
- 禁止跨run复用旧prediction，完整重算375行。

## 三、协议与验证

- `protocol_schema=p2_min_v1`；
- `phase2_data_status=VALIDATED_ONCE`；
- 数据身份未改变，不重验received IQ或split；
- prediction不读取truth，scorer仅在375行闭合后连接truth；
- 本地57项相关回归、Python编译和`git diff --check`通过；
- 独立P0/P1审查及定点复审PASS；
- v2真实失败行smoke已达到before/after 0/1560差异；
- v3启动前重新执行同一smoke并使用独立输出根。

## 四、路径与命令

|字段|路径|
|---|---|
|既有feature root|`/home/szu2070436088/2510044040/CV-SincNet/stage2_inputs/cvs_full_ablation_phase2c_feature_t1_20260730_v3_47212437/artifacts/features`|
|补充feature root|`/home/szu2070436088/2510044040/CV-SincNet/stage2_inputs/erbt_idr_m24_d1_refit_full125_20260820_v1_supplement_features/artifacts/features`|
|既有scoring root|`/home/szu2070436088/2510044040/CV-SincNet/stage2_inputs/cvs_full_ablation_phase2c_sidecar_t1_20260730_v3_47212437/artifacts/sidecars`|
|补充scoring root|`/home/szu2070436088/2510044040/CV-SincNet/stage2_inputs/erbt_idr_m24_d1_refit_full125_20260820_v1_supplement_packages`|
|run root|`/home/szu2070436088/2510044040/CV-SincNet/runs/erbt_idr_m24_d1_refit_full125_20260820_v3`|
|log root|`/home/szu2070436088/2510044040/CV-SincNet/logs/erbt_idr_m24_d1_refit_full125_20260820_v3`|
|Python|`/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python`|
|设备|`cpu`，最多2个worker|

完整prediction命令使用`run_m24_d1_refit_matrix.py --run-id erbt_idr_m24_d1_refit_full125_20260820_v3`及上述两个feature root，输出到v3的`predictions`目录。

## 五、停止规则与预期artifact

仅在协议/query泄漏、错误矩阵身份、输出碰撞、错误checkout、无法启动、无prediction闭合或至少两行相同确定性prediction前异常时停止；低性能不得停止。

prediction闭合标准：

- `matrix_index.status=PREDICTIONS_COMPLETE_TRUTH_UNOPENED`；
- `row_count=375`；
- `paired_input_identity_count=125`；
- R0/R1/R2各125；
- R1 before/after disagreement均为0。

闭合后运行truth-last scorer，并生成总体、K/new、receiver、seed、scene、old/new、class、margin、中心角距、help/harm、`F_within`和`F_std`完整分析。

## 六、结果

待N607实验闭合后追加。
