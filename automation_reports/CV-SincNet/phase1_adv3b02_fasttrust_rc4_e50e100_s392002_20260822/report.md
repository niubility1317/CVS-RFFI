# FastTrust-RC4压缩训练实验报告

## 当前状态

`LOCAL_VERIFIED`

run_id=`phase1_adv3b02_fasttrust_rc4_e50e100_s392002_20260822`

用户明确要求把训练预算从E200压缩为E50/E100。本轮不启动旧E200矩阵，使用8行最小矩阵同时回答“RC4是否优于无U身份/仅H”和“50epoch是否足够、100epoch是否更稳”。

## 协议与固定口径

- source-only，`L_s/U_s/V_cal/V_select=0.07/0.63/0.15/0.15`。
- 所有行固定seed392002、Core90初始化、U batch256、相同source split、每epoch完整U物理样本覆盖、相同batch/labeled replay口径；target不选模、不反馈。
- H/P/N按完整U batch归一化，不设置identity fill下限；R不承担身份梯度，但保留all-U domain/GRL/self。
- 最终checkpoint必须闭合clean与`leo_clear_weak/leo_low_elev_weak/leo_rain_weak`逐场景评估。

## 8行压缩矩阵

|GPU|预算|候选|伪标签机制|问题|
|---:|---:|---|---|---|
|0|50|E50_P0_NO_U_ID|无U identity|E50控制|
|1|50|E50_P3_DUAL_H|校准双教师H|严格伪标签在E50是否有效|
|2|50|E50_P6_RC4|H＋P＋N＋class×receiver cap|低/中置信U能否在E50形成增量|
|3|50|E50_P7_RC4_SAT|P6＋仅H星地strong view|星地辅助净增量|
|4|100|E100_P0_NO_U_ID|无U identity|E100控制|
|5|100|E100_P3_DUAL_H|校准双教师H|严格伪标签在E100是否有效|
|6|100|E100_P6_RC4|H＋P＋N＋class×receiver cap|完整RC4在E100的增量|
|7|100|E100_P7_RC4_SAT|P6＋仅H星地strong view|星地辅助净增量|

## 压缩调度

|预算|准备期|身份启动|V_cal冻结更新|末段冻结/身份降权|MUSE阶段边界|
|---:|---|---:|---|---:|
|50|E1–E3|E4|E1/E11/E24/E41|E46–E50，identity×0.4|E5/E11/E18/E41/E46|
|100|E1–E5|E6|E1/E21/E46/E81|E91–E100，identity×0.4|E9/E21/E35/E81/E91|

更新epoch按E200设计比例压缩并阶段内冻结；不是逐epoch校准。ready条件不满足时不补样本，H/P/N可以为空。

## 训练加速

- EMA两个弱视图拼接为一次teacher前向；anchor一次冻结前向。
- H/P/N共享一次全U student strong前向；只有两条P7把H星地行拼接进同一次student调用。
- P0不构造U星地视图；P/P/N仅在logit上计算。
- `V_cal`小型logistic只在4个调度点使用Newton求解；E50/E100相对E200理论epoch预算分别减少75%/50%。

## 本地验证与版本

- RC4及相邻SAT测试26项通过；Python编译、launcher语法、JSON、diff检查通过。
- 8行dry-run必须逐行读回`--epochs 50/100`、对应压缩阶段边界与clean＋三LEO输出。
- 实现基线提交：`b71ef00c3abb0d610bc10283fa73aa0721f1b4d9`；资源队列提交：`a4ee344fc84aa534b30911c618947b238cac07ca`。本报告与压缩配置提交待本轮完成后记录。

## N607与停止规则

2026-08-22 18:41 CST只读preflight为`VERIFIED`，home剩余约7.3TiB。GPU1当前已有2个compute app，因此GPU1行将由资源感知dispatcher等待；其余有槽位行可立即启动。任何GPU均不超过2个compute app，不干预SIDFFT96或SAT-Anchor。

只因协议/query泄漏、错误预算/split/seed、输出覆盖、错误checkout、无prediction闭合、重复确定性异常或进程归属不清停止；不得因中期性能差停止。当前没有性能结果，不能声称E50/E100或RC4已优于控制。
