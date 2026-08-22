# FastTrust-RC4压缩训练实验报告

## 当前状态

`RUNNING`

有效run_id=`phase1_adv3b02_fasttrust_rc4_e50e100_s392002_20260822_r2`

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

- 最终RC4、MUSE训练集成、协议、launcher与加速测试92项通过；Python编译、launcher语法、JSON、diff检查通过。
- 本地与远端8行dry-run均逐行读回`--epochs 50/100`、对应压缩阶段边界与clean＋三LEO输出。
- 实现基线提交：`b71ef00c3abb0d610bc10283fa73aa0721f1b4d9`；资源队列提交：`a4ee344fc84aa534b30911c618947b238cac07ca`；E50/E100压缩矩阵提交：`8eaeb549f32aea4421845e329e4301d647dbdcbe`。
- AMP均衡权重类型修复提交：`55db9d54c1ccd8ed7d199e7554426654002d6b0d`；无计算图零分量遥测修复提交：`53ef057dbcbb9711ff65a6b60af348051bdf0801`；最终release提交：`2186cc592860df1fca37a14e571de2d85fbe482c`。
- 真实Core90无query CUDA AMP smoke：严格checkpoint重建成功，H/P/N/R=`8/3/14/7`，loss=`0.11721120`，学生有限梯度tensor63组，冻结教师梯度0组，结论`VERIFIED`。

## 发布与启动读回

- 本地归档：`E:/type10-7/release_artifacts/phase1_fasttrust_rc4_e50e100_2186cc59.tar.gz`。
- 远端归档：`/home/szu2070436088/2510044040/CV-SincNet/releases/incoming/phase1_fasttrust_rc4_e50e100_2186cc59.tar.gz`。
- 本地/远端唯一归档SHA-256：`ff1b09f1f431f6f2bc92bb5266f5a445a744cd24ec08c5c25d7204a1c602fa93`，远端编译与launcher语法检查`VERIFIED`。
- release root：`/home/szu2070436088/2510044040/CV-SincNet/releases/phase1_fasttrust_rc4_e50e100_2186cc59`。
- 有效run root：`/home/szu2070436088/2510044040/CV-SincNet/runs/phase1_adv3b02_fasttrust_rc4_e50e100_s392002_20260822_r2`。
- dispatcher PID=`1055384`；CWD与cmdline均绑定上述release，状态`RUNNING`。
- GPU0、2、3、4、5、6、7上的7行已完成E001；训练日志由约6.8KB增长到约12.5KB，异常指纹数0。GPU1已有SIDFFT96与SAT-Anchor两个compute app，因此`E50_P3_DUAL_H`保持资源等待，未超过每卡2个训练实验。

## 技术停止与恢复记录

- 首次run_id=`phase1_adv3b02_fasttrust_rc4_e50e100_s392002_20260822`在prediction前有5行复现同一AMP类型异常：FP16目标接收FP32均衡权重。精确停止dispatcher PID1039063及32个后代，`STOP_VERIFIED=YES`；输出与日志完整保留，状态`STOPPED_EARLY_SYSTEMIC_TECHNICAL_FAILURE / NO_PERFORMANCE_RESULT`。
- 第二次run_id=`phase1_adv3b02_fasttrust_rc4_e50e100_s392002_20260822_r1`的两条P7在prediction前复现同一图无关零值遥测异常。精确停止dispatcher PID1048780及93个后代，`STOP_VERIFIED=YES`；输出与日志完整保留，状态`STOPPED_EARLY_SYSTEMIC_TECHNICAL_FAILURE / NO_PERFORMANCE_RESULT`。
- 两次停止均未接触SIDFFT96、SAT-Anchor或其他run；有效实验只认`_r2`，不得把前两次partial输出纳入性能比较。

## N607与停止规则

2026-08-22 18:41 CST只读preflight为`VERIFIED`，home剩余约7.3TiB。GPU1当前已有2个compute app，因此GPU1行将由资源感知dispatcher等待；其余有槽位行可立即启动。任何GPU均不超过2个compute app，不干预SIDFFT96或SAT-Anchor。

只因协议/query泄漏、错误预算/split/seed、输出覆盖、错误checkout、无prediction闭合、重复确定性异常或进程归属不清停止；不得因中期性能差停止。当前仅有启动与E001健康证据，没有最终clean/三LEO同row性能结果，不能声称E50/E100或RC4已优于控制。
