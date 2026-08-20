# ERBT-IDR M2.4 D1-REFIT完整125修复重跑报告

日期：2026-08-20

run ID：`erbt_idr_m24_d1_refit_full125_20260820_v2`

当前状态：`STOPPED_EARLY_SYSTEMIC_TECHNICAL_FAILURE / NO_PERFORMANCE_RESULT`

## 一、修复目标与根因

v1在375个row receipt全部生成后因R1注册前编译一致性失败而停止，truth始终未打开。唯一差异位于`rx20-1/m7282101/K10/new20/leo_rain_weak`的query`qid_29b3f575fa647024e1bf91c23e216fcdb002e7ce0bfb422c5893eed0852115a6`。

分数级只读诊断确认：

- 历史注册前P2-A1_NO_RF32头为FP32 affine状态；
- 正确类相对竞争类的历史margin为`0.00015592575073242188`；
- 旧R1将该FP32头重新编译为F3 INT8/FP16后，margin变为`-0.00002002716064453125`并发生翻转；
- 使用F0保持源FP32精度时，对该场景历史预测为0差异，最大score误差为1个FP32 ULP；
- 注册后历史头本身已是F3状态，旧实现解码后再次量化虽未在该row翻转，但不满足“无损裁剪”的实现语义。

本轮修复遵循源存储精度：

- 历史源为FP32时，D1使用F0 IF256主状态，不经过FP16 bias或INT8降精度；
- 历史源为F3时，直接裁剪恒零RF32对应的前256维code/scale前缀，不解码后重新量化；
- 冻结256维log-diag和逐样本归一化顺序保持不变；
- 注册前/后任一预测差异仍为技术失败，不使用容差、手工改预测或历史结果回退；
- D1资源按注册前/后实际最大状态计费。

## 二、冻结矩阵

|字段|值|
|---|---|
|R0|`M24-D0-HISTORICAL-F1`，D92 E0默认去RF32版本|
|R1|`M24-D1-COMPILE-PARITY`，本轮修复对象|
|R2|`M24-D1-REFIT`|
|receiver|`20-1`、`3-19`、`7-14`、`7-7`、`8-8`|
|method seed|`7282101`至`7282105`|
|条件|`K1/new20`、`K2/new20`、`K5/new20`、`K10/new20`、`K10/new5`|
|完整规模|125个输入身份×3方法=375行，1125个场景单元|
|复用裁决|v1 artifact缺少显式`capsule_id/split_id/support seed`绑定，因此禁止跨run复用，v2完整重算375行|

## 三、协议与版本

- `protocol_schema=p2_min_v1`；
- `phase2_data_status=VALIDATED_ONCE`；
- received IQ、物理ID、receiver/TX、场景、K、support/query split均不变，不触发数据重验；
- prediction阶段不读取truth，不使用query role、配额或全局重分配；
- 375行prediction全部闭合并生成`matrix_index.json`后，独立scorer才连接truth；
- 分支：`work/m24-safe-residual`；
- 修复提交：`1ca297dc1d5c44f6ec993abc58c8c1dc4208e89b`；
- 本地验证：57项M2.3/M2.4回归通过，相关Python编译通过，`git diff --check`通过；
- 独立P0/P1审查：首次发现跨run复用缺少support/split直接绑定的1个P1；删除复用入口并固定完整重算375行后，定点复审PASS，P0=0、P1=0。

## 四、N607路径

|字段|路径|
|---|---|
|release|`/home/szu2070436088/2510044040/CV-SincNet/releases/erbt_idr_m24_d1_refit_full125_20260820_v2`|
|既有feature root|`/home/szu2070436088/2510044040/CV-SincNet/stage2_inputs/cvs_full_ablation_phase2c_feature_t1_20260730_v3_47212437/artifacts/features`|
|补充feature root|`/home/szu2070436088/2510044040/CV-SincNet/stage2_inputs/erbt_idr_m24_d1_refit_full125_20260820_v1_supplement_features/artifacts/features`|
|既有scoring root|`/home/szu2070436088/2510044040/CV-SincNet/stage2_inputs/cvs_full_ablation_phase2c_sidecar_t1_20260730_v3_47212437/artifacts/sidecars`|
|补充scoring root|`/home/szu2070436088/2510044040/CV-SincNet/stage2_inputs/erbt_idr_m24_d1_refit_full125_20260820_v1_supplement_packages`|
|run root|`/home/szu2070436088/2510044040/CV-SincNet/runs/erbt_idr_m24_d1_refit_full125_20260820_v2`|
|log root|`/home/szu2070436088/2510044040/CV-SincNet/logs/erbt_idr_m24_d1_refit_full125_20260820_v2`|
|Python|`/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python`|
|设备|`cpu`，最多2个worker|

## 五、执行顺序与停止规则

1.发布一个Git archive release，只进行一次本地/远端archive SHA比较和一次远端编译。
2.先对v1唯一失败输入执行真实checkpoint、无truth的R1 smoke，要求注册前和注册后差异均为0。
3.smoke PASS后立即启动完整375行prediction，不运行局部性能矩阵。
4.启动后核对PID、CWD、cmdline、run root和日志增长。
5.仅在协议/query泄漏、错误矩阵身份、输出碰撞、错误checkout、无法启动、无prediction闭合或至少两行相同确定性prediction前异常时技术停止；低性能不得停止。
6.严格确认`matrix_index.status=PREDICTIONS_COMPLETE_TRUTH_UNOPENED`、`row_count=375`、`paired_input_identity_count=125`且R0/R1/R2各125后，运行truth-last scorer。
7.评分后生成总体、K/new、receiver、seed、scene、old/new、class、margin、中心角距、help/harm、`F_within`和`F_std`完整分析，并更新D92 E0总报告。

## 六、预期artifact

- `smoke/.../row_execution_receipt.json`；
- `predictions/matrix_index.json`及375个方法行的prediction、receipt和truth-blind diagnostics；
- `scores/scored_matrix_index.json`及375个same-row/four-state结果；
- `results_summary.json`、本报告和D92 E0总报告更新。

## 七、结果

真实失败行smoke已完成，注册前／后均为0/1560差异，证明编译修复有效。随后完整controller在生成任何row receipt前退出：启动命令把补充feature root错误写成补充构建根，缺少末级`artifacts/features`，因此无法定位seed`7282104/7282105`缓存。

- prediction父PID：`3864001`，已退出；
- 完整矩阵receipt：0；
- `matrix_index.json`：不存在；
- truth：未打开；
- v2输出与日志原地保留，不删除、不覆盖、不复用run ID；
- 后续使用新run ID`erbt_idr_m24_d1_refit_full125_20260820_v3`和纠正后的补充feature root重新启动完整375行。
