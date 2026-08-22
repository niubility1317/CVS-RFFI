# FastTrust-RC4压缩训练实验报告

## 当前状态

`PARTIAL_TECHNICAL_FAILURE / RUNNING`

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

## 2026-08-22 23:23 CST进度与中期数据

本节来自对8行`metrics_epoch.jsonl`和完整`train.log`的只读解析，以及PID/GPU/status/artifact读回；所有JSONL均可完整解析。当前矩阵不是8行全部健康：1行已闭合、3行仍运行、4行因同一技术异常终止。

|候选|状态|进度|最新source-val星地均值/%|最新source-val最差接收机/%|伪标签身份利用|
|---|---|---:|---:|---:|---|
|E50_P0_NO_U_ID|`ARTIFACTS_COMPLETE`|50/50|86.397|84.611|0|
|E50_P3_DUAL_H|`RUNNING`|47/50|第46轮89.151|第46轮87.365|身份期平均21.79/批，仅H|
|E50_P6_RC4|`TRAIN_FAILED`|5/50|88.526|87.024|身份期平均253.74/批，H/P/N/R=6.64/245.09/2.01/1.91|
|E50_P7_RC4_SAT|`TRAIN_FAILED`|5/50|88.529|87.143|身份期平均253.79/批，H/P/N/R=14.50/237.20/2.09/1.86|
|E100_P0_NO_U_ID|`RUNNING`|67/100|第66轮88.331|第66轮86.540|0|
|E100_P3_DUAL_H|`RUNNING`|68/100|85.913|84.008|身份期平均34.23/批，仅H|
|E100_P6_RC4|`TRAIN_FAILED`|7/100|88.317|86.857|身份期平均253.72/批，H/P/N/R=25.97/222.23/5.52/1.93|
|E100_P7_RC4_SAT|`TRAIN_FAILED`|7/100|89.153|87.643|身份期平均253.72/批，H/P/N/R=18.99/232.48/2.25/1.93|

### 已闭合的E50_P0最终结果

最终checkpoint为第50轮，严格重建成功；clean和每个LEO场景均在`test_unseen_day_unseen_rx`的60000条样本上完成。

|接收机|clean/%|clear/%|low-elev/%|rain/%|三LEO均值/%|
|---|---:|---:|---:|---:|---:|
|RX7（20-1）|83.133|65.167|60.267|60.442|61.958|
|RX8（3-19）|77.658|48.808|46.975|46.133|47.306|
|RX9（7-14）|96.033|85.517|83.383|82.075|83.658|
|RX10（7-7）|89.908|76.383|72.283|70.808|73.158|
|RX11（8-8）|76.000|68.117|66.083|65.633|66.611|
|汇总|84.547|68.798|65.798|65.018|66.538|

相对clean，clear、low-elev、rain分别下降15.748、18.748、19.528个百分点；RX8是最明显短板，三LEO均值仅47.306%。

### 同epoch可比诊断

- E50第46轮：P3相对P0的source-val星地均值`+0.511`个百分点、最差接收机`+0.579`个百分点；P3每批仅使用4.84条H样本，当前只是中期验证集诊断，不能替代最终测试。
- E100第66轮：P3相对P0的source-val星地均值`-1.974`个百分点、最差接收机`-2.214`个百分点；P3每批使用47.27条H样本。由此不能声称P3稳定优于P0。
- 身份阶段均值显示P3只使用约21.79/256（E50）和34.23/256（E100）条无标签样本；RC4可将约253.7/256条分入H/P/N身份路由，说明利用率目标在路由层面达到，但尚未形成有效参数更新和最终性能证据。

### RC4技术失败与证据边界

四条P6/P7均触发`RuntimeError: FASTTRUST_CONSECUTIVE_ZERO_OPTIMIZER_STEP_EPOCHS`。身份机制启动后的两个完整epoch中，`train_skipped_nonfinite_loss=1.0`、`train_optimizer_step_applied=0.0`，表明非有限总损失使所有优化步被跳过；这不是“RC4性能较差”，而是`STOPPED_EARLY_SYSTEMIC_TECHNICAL_FAILURE / NO_PERFORMANCE_RESULT`。终止前的H/P/N/R和coverage只能用于定位路由/数值问题，不得用于晋级或性能结论。

### 资源、ETA与当前限制

- 23:23:58 CST时dispatcher PID1055384仍存活；E50_P3、E100_P0、E100_P3主进程分别绑定GPU1、GPU4、GPU5，未干预既有SAT-Anchor任务。
- GPU1/4/5利用率分别为98%/96%/99%，显存约4695/5983/5781MiB；GPU7空闲不构成重启授权，本次仅监控。
- 近5轮平均单epoch约261–279秒。E50_P3训练预计约13分钟后结束；E100两行训练预计还需约2.4–2.6小时，之后仍需clean和三LEO逐场景评估。ETA受并发负载影响，仅作区间估计。
- 当前唯一完整性能行是E50_P0。P3尚无最终测试，P6/P7无有效训练结果，因此现在不能回答“RC4是否提升伪标签利用后的最终准确率”，也不能晋级任何伪标签方案。

## 2026-08-23 00:58 CST完成状态与epoch压缩判断

### 当前完成状态

实验尚未全部完成。E50_P0和E50_P3均已达到50/50并完成clean与三LEO严格checkpoint重建评估；E100_P0为87/100，E100_P3为89/100，仍在训练且无Traceback、RuntimeError、OOM或NaN/Inf记录；四条P6/P7继续保持`TRAIN_FAILED / NO_PERFORMANCE_RESULT`。因此当前是2行完整闭合、2行运行、4行技术失败，不得标记整个8行矩阵完成。

按近5轮速度，E100_P0剩余训练约63分钟，E100_P3约51分钟；训练后仍需四场景评估，完整闭合时间晚于训练结束。

### E50_P3相对E50_P0最终同row增益

|指标|E50_P0/%|E50_P3/%|P3-P0/pp|
|---|---:|---:|---:|
|clean|84.547|87.153|+2.607|
|leo_clear_weak|68.798|70.650|+1.852|
|leo_low_elev_weak|65.798|67.952|+2.153|
|leo_rain_weak|65.018|66.633|+1.615|
|三LEO均值|66.538|68.412|+1.873|
|四场景receiver-cell floor|46.133|48.833|+2.700|

四个聚合场景、LEO均值和receiver-cell floor均为正增益，说明50epoch足以完成一次有判别力的严格H伪标签试验，并能在相同seed、相同split下区分P3与P0。

### 收敛曲线与预算判定

- E50末10轮相对前10轮仍上升：P0的source-val星地均值约`+0.705pp`、floor约`+0.770pp`；P3分别约`+0.654pp`和`+0.774pp`。E50结束时尚不能认定完全平台化。
- E100到当前末10轮相对前10轮上升更明显：P0星地均值约`+1.855pp`、floor约`+2.092pp`；P3分别约`+1.746pp`和`+1.964pp`。E100_P3当前第89轮达到星地均值91.653%，仍处于本轮上升区间，必须等待第100轮和最终四场景测试。
- 结论分层：`50epoch=可行的快速筛选预算`，可用于淘汰无效路线和验证方向；`100epoch=当前最小确认预算`，用于候选最终比较更稳妥。现有证据不支持把50epoch直接设为最终模型训练预算，也尚未证明100epoch可以完全取代历史E200确认。
- 该判断目前只对已完成的P0/P3路线成立。RC4的P6/P7因非有限损失没有形成参数更新，无法判断RC4在50或100epoch下的收敛与收益；修复数值问题后应先跑50epoch最小证伪，再决定是否进入100epoch确认。

## 后续默认预算与自动完成分析

- 用户于2026-08-23明确确认：今后未在当前请求中另行覆盖时，正式Phase1训练默认恢复为200epoch；50/100epoch仅作为本轮压缩试验和后续候选筛选/确认预算，不修改已经启动的本轮矩阵。
- 01:05 CST只读读回：E100_P0为88/100，近5轮平均288.5秒；E100_P3为90/100，近5轮平均280.6秒。预计P3约01:52结束训练、P0约02:03结束训练；参考E50评估耗时，全部可完成分支预计约02:05形成四场景artifact。
- 已创建当前任务heartbeat自动化“FastTrust训练完成分析”（automation_id=`fasttrust`），每15分钟只读检查一次。两条E100完整闭合后，自动解析全部8行完整日志和结构化指标，完成50/100epoch、P0/P3、伪标签利用率、RC4失败、逐接收机/逐场景和资源耗时分析，追加详细中文报告，提交并push；发布验证后自动暂停，避免重复分析。
