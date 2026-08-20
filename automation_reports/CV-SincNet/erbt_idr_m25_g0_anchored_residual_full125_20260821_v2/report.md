# ERBT-IDR M2.5 G0锚定交叉拟合残差完整125实验报告

日期：2026-08-21

run ID：`erbt_idr_m25_g0_anchored_residual_full125_20260821_v2`

当前状态：`LOCAL_VERIFIED / PREREGISTERED / N607_NOT_LAUNCHED`

实现分支：`work/m24-safe-residual`

## 一、问题与方法裁决

既有`erbt_idr_m24_invariance_break_full125_20260820_v1`已证明，G1–G4虽然打破了R2的代数等价，但整体替换D92 E0稳健中心、共享尺度、full/block融合和全类LDA校准后，H从0.537558降至0.297636/0.297678/0.285538/0.278228，四个候选在125/125个同row身份上均低于G0。本轮不再构造替代分类头，而以去RF32的D92 E0/R1作为不可破坏主分数：

\[
s_c(q)=s_c^{G0}(q)+\lambda g(q)\bar r_c(q).
\]

其中，\(\bar r_c\)由合法support构造的局部证据跨类中心化并归一化至\([-1,1]\)；\(g(q)\)只在G0 top-2 margin不超过0.10时开启；\(\lambda\in\{0,0.02,0.04,0.08\}\)只由support内部留一局部证据选择。任一query只读取自身G0分数和冻结局部状态，不更新模型、阈值、强度、原型或其他query状态。

本实现没有取得D92内部每个fold的完整held-support logits。强度选择使用固定全support G0锚点分数与留一重建的局部证据，因此属于`G0_FIXED_ANCHOR_PLUS_LOCAL_JACKKNIFE`，不是对指导中“复用D92 held logits”的严格同构实现。该差异写入证据边界，不影响support-only和query只读合法性。

## 二、完整矩阵

|arm|定义|
|---|---|
|B0|`M24-D1-COMPILE-PARITY`，去RF32的D92 E0/R1等价基线|
|B1|`M25-B1-G0-BOUNDED-LOCAL-RESIDUAL`，G0＋有界单原型局部残差|
|B2|`M25-B2-G0-SHRINKAGE-RADIUS-RESIDUAL`，B1＋query依赖的收缩类半径|
|B3|`M25-B3-G0-STABLE-DUAL-PROTOTYPE-RESIDUAL`，B2＋稳定性门控、按簇大小加权双原型|

每个arm运行5个receiver×5个method seed×5组K/new条件，共125个输入身份。总矩阵为500个方法行、1500个LEO弱场景单元。K/new条件固定为K1/new20、K2/new20、K5/new20、K10/new20和K10/new5。K1/K2的B1–B3强度固定为0，必须与B0逐query一致。

## 三、协议与本地状态

- `protocol_schema=p2_min_v1`；
- 复用`phase2_data_status=VALIDATED_ONCE`且匹配`capsule_id/split_id`的既有固定received IQ，不因方法变化重验；
- predictor只能读取冻结Phase1 bundle、当前row合法support和与数据无关的冻结配置；
- query truth、old/new真实角色、query batch统计、类别配额和跨query重排均不可达；
- prediction全部闭合后，独立scorer才连接truth；
- 本地聚焦与相邻回归56项通过，五个生产脚本/模块编译通过，`git diff --check`通过；
- 独立P0/P1审查初审发现3项P1，均已定点修复；定点复审结论为`NO_P0_P1`；
- v1在352/500个partial receipt后因局部原型MAC统计引用未定义`IF_DIM`而退出，未生成matrix index，truth未打开且无性能结果；v1全部证据原位保留；
- 新增非零残差MAC分支回归测试，定点修复改为使用当前row的真实`feature_dim`；57项聚焦与相邻回归通过；
- 实验代码固定提交：`e847e0d41883d39c70f9633292e6f87aabcb7349`；远端`origin/work/m24-safe-residual`已独立回读为同一OID。

## 四、N607路径与资源

|字段|路径|
|---|---|
|既有feature root|`/home/szu2070436088/2510044040/CV-SincNet/stage2_inputs/cvs_full_ablation_phase2c_feature_t1_20260730_v3_47212437/artifacts/features`|
|补充feature root|`/home/szu2070436088/2510044040/CV-SincNet/stage2_inputs/erbt_idr_m24_d1_refit_full125_20260820_v1_supplement_features/artifacts/features`|
|既有scoring root|`/home/szu2070436088/2510044040/CV-SincNet/stage2_inputs/cvs_full_ablation_phase2c_sidecar_t1_20260730_v3_47212437/artifacts/sidecars`|
|补充scoring root|`/home/szu2070436088/2510044040/CV-SincNet/runs/erbt_idr_m24_d1_refit_full125_20260820_v3/scoring_root_repaired_v3`|
|release root|`/home/szu2070436088/2510044040/CV-SincNet/releases/erbt_idr_m25_g0_anchored_residual_full125_20260821_v2`|
|run root|`/home/szu2070436088/2510044040/CV-SincNet/runs/erbt_idr_m25_g0_anchored_residual_full125_20260821_v2`|
|log root|`/home/szu2070436088/2510044040/CV-SincNet/logs/erbt_idr_m25_g0_anchored_residual_full125_20260821_v2`|
|Python|`/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python`|
|资源|CPU，`--max-workers 2`；不占用GPU，不干预既有训练|

v2必须使用新的单一release归档、新release/run/log根和新的真实cache无truth smoke；不得续写或复用v1 partial输出。

## 五、冻结命令

prediction：

```bash
cd /home/szu2070436088/2510044040/CV-SincNet/releases/erbt_idr_m25_g0_anchored_residual_full125_20260821_v2/code
PYTHONPATH=. /home/szu2070436088/.conda/envs/CVS-RFFI/bin/python -u scripts/run_m25_anchored_residual_full125.py --run-id erbt_idr_m25_g0_anchored_residual_full125_20260821_v2 --feature-root /home/szu2070436088/2510044040/CV-SincNet/stage2_inputs/cvs_full_ablation_phase2c_feature_t1_20260730_v3_47212437/artifacts/features --supplemental-feature-root /home/szu2070436088/2510044040/CV-SincNet/stage2_inputs/erbt_idr_m24_d1_refit_full125_20260820_v1_supplement_features/artifacts/features --output-root /home/szu2070436088/2510044040/CV-SincNet/runs/erbt_idr_m25_g0_anchored_residual_full125_20260821_v2/predictions --device cpu --max-workers 2
```

truth-last scorer：

```bash
PYTHONPATH=. /home/szu2070436088/.conda/envs/CVS-RFFI/bin/python -u scripts/score_m25_anchored_residual_full125.py --matrix-index /home/szu2070436088/2510044040/CV-SincNet/runs/erbt_idr_m25_g0_anchored_residual_full125_20260821_v2/predictions/matrix_index.json --scoring-root /home/szu2070436088/2510044040/CV-SincNet/stage2_inputs/cvs_full_ablation_phase2c_sidecar_t1_20260730_v3_47212437/artifacts/sidecars --supplemental-scoring-root /home/szu2070436088/2510044040/CV-SincNet/runs/erbt_idr_m24_d1_refit_full125_20260820_v3/scoring_root_repaired_v3 --output-root /home/szu2070436088/2510044040/CV-SincNet/runs/erbt_idr_m25_g0_anchored_residual_full125_20260821_v2/scores --bootstrap-repeats 2000
```

summary：

```bash
PYTHONPATH=. /home/szu2070436088/.conda/envs/CVS-RFFI/bin/python -u scripts/summarize_m25_anchored_residual_full125.py --prediction-root /home/szu2070436088/2510044040/CV-SincNet/runs/erbt_idr_m25_g0_anchored_residual_full125_20260821_v2/predictions --score-root /home/szu2070436088/2510044040/CV-SincNet/runs/erbt_idr_m25_g0_anchored_residual_full125_20260821_v2/scores --output /home/szu2070436088/2510044040/CV-SincNet/runs/erbt_idr_m25_g0_anchored_residual_full125_20260821_v2/results_summary.json
```

## 六、闭合、停止与分析要求

prediction闭合必须同时满足：

- `matrix_index.status=PREDICTIONS_COMPLETE_TRUTH_UNOPENED`；
- `row_count=500`；
- `paired_input_identity_count=125`；
- B0、B1、B2、B3各125行；
- 500个prediction和row receipt均存在；
- K1/K2的B1–B3相对B0 disagreement均为0。

仅在协议/query泄漏、错误矩阵、输出碰撞、错误checkout、无法启动、prediction不闭合、scorer连接错误、进程归属不清或至少两行相同确定性prediction前异常时停止并保留证据。低性能不得停止。

评分后生成总体、K/new、receiver、seed、scene、四状态、old/new、class、margin、中心角距、help/harm、`F_within/F_std`、残差强度分布、门控比例、双原型接受率和资源分析，并与去RF32 D92 E0、M2.4 R2及G1–G4完整125结果同表比较。

## 七、证据边界

本报告当前只证明设计冻结与本地实现状态，不构成N607落地、prediction闭合、性能提升、部署效率、Phase3或真实在轨证据。最终结论必须由本run的完整500行truth-last结果更新。
