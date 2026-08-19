# ERBT-IDR M2.4／F1-SafeResidual实验报告

## 启动前登记

|字段|值|
|---|---|
|run ID|`erbt_idr_m24_safe_residual_diagnostic_20260820_v3`|
|当前状态|`LOCAL_VERIFIED / PREREGISTERED / NOT_YET_LANDED`|
|代码提交|`e4f2e97a63ac673ae955193b3341e903c576def7`，分支`work/m24-safe-residual`|
|前序|v2因K10 D1出现14/660个预测差异而在truth接入前停止；v3使用修复后的全新release与不可覆盖输出根|
|候选／矩阵|receiver=`3-19`，method seed=`7282101`；`K1/new20`与`K10/new5`两条row；每条按D0→D10固定顺序运行11臂及三个`leo_*_weak`场景|
|科学停止规则|K1与K10的D1均必须与历史F1逐query完全一致；任一不一致即停止且不评分|
|协议|`p2_min_v1`；复用匹配的`VALIDATED_ONCE` capsule/split和固定received IQ；query不更新状态，prediction冻结后才允许独立scorer接入truth|
|环境／CWD|本地`ssr-gpu`；N607使用`/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python`与本run不可变release CWD|
|资源|闭式head在CPU执行，不新增GPU计算进程；既有GPU任务保持不动|
|输出|`/home/szu2070436088/2510044040/CV-SincNet/runs/erbt_idr_m24_safe_residual_diagnostic_20260820_v3`；prediction、score和log目录不可覆盖|
|技术停止规则|仅协议／query越界、错误row或checkout、输出覆盖、非PSD、无prediction闭合、scorer接线错误或至少两个执行单元的相同确定性pre-prediction故障；不因低性能停止|
|预期artifact|2个suite index、22个arm execution receipt、22个不可变prediction artifact、22个same-row score、D1 parity、配对诊断、资源与量化收据|

## 本地验证

- 修复后`compileall`通过；M2.4聚焦和M2.3相邻回归42项通过。
- 使用真实K10 feature／overlay cache、保持truth未打开的D1回归：660/660个query与历史F1一致。
- D1紧凑推理态7677B，包含256维冻结log-diag和量化仿射头；`persistent_update_state_bytes=0`。
- 独立定点复审工具受本机旧CLI兼容问题影响，结果为`UNKNOWN`；此前候选级独立P0/P1审查已完成，v3只修复实际D1 parity故障。

## 扩展矩阵预设

两条D1都通过后才执行truth-last评分，并从D2–D10中选择至多2个相对D1无明确同row伤害的候选，与D1进入现有合法cache交集上的扩展筛选。目标覆盖5个receiver、至少3个support/query seed或draw、四个K／新类条件和三个场景；每个证据维度按真实身份标注，不把重复draw伪称为method-seed。

## 证据边界

本轮是研发诊断，不是fresh confirmation、完整125结论、Phase3开放世界能力或星载部署证明。最终结果必须保持receiver、seed、draw、K、新类数、场景和候选同row，不拼接跨row最优值。
