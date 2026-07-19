# D63跨折稳定Fisher行拼接探针报告

## 1.执行前登记

- ID：`d63_jackknife_stable_fisher_row_splice_probe_20260719`；操作者：Codex`/root`；状态：`READY_TO_RUN_LOCAL_DEVELOPMENT_CELL`。
- 目标：在D62总体TP/FP Pareto门上增加固定的leave-one-inner-fold全类别不伤害约束，剔除跨fold不稳定行，同时保留稳定D61 Fisher残差收益。
- 比较目标D46：before92.22%、after81.67%、new84.67%、H82.33%、forget10.56pp、joint23.33%、min-before80%、min-after53.33%、min-new73.33%、混淆25/8/15。
- 当前聚合最强D62：before92.78%、after82.22%、new84.67%、H82.62%、forget10.56pp、joint26.67%、min-before80%、min-after53.33%、min-new73.33%、混淆23/8/15；但low forgetting+3.33pp、rain before−1.67pp，非可晋升版本。
- cell：receiver`20-1`、seed`713101`、K10/new5、3场景×5 outer fold、实际K8；复用匹配`VALIDATED_ONCE/p2_min_v1`的D18 capsule，不重验数据。
- 版本：预注册`4aa0a8ed`；实现`8cd2385d`；worktree`E:\type10-7\code\snapshots\d63wt` detached clean；脚本SHA256=`d47b51eeadece4184ef16b6b73fa069d6859ac0eaadc232efa1afa6c0cfa5264`。
- 验证：py_compile通过；D62＋D63单元测试15/15通过；D43–D63整链回归240/240通过；diff check通过。
- 本轮只在本地`ssr-gpu`执行，不访问N607，无远端PID/GPU/sync。

## 2.机制、资源与判门

D63保持D62的full/block组件、Fisher残差和匿名类行编译不变。每个候选行先通过总体严格Pareto门，再要求K个leave-one-fold子集对所有类别TP不降且FP不升；联合行替换也必须在总体和全部子集原子安全。没有阈值、角色、场景、receiver、rank、gain或alpha扫描，超参数计数0。最终只持久化一个int8/FP16 affine state，query额外MAC/state为0。

相对D46必须保持总体和三场景before/after/new/H/forgetting、floor/joint、混淆与量化，并至少严格改善after、forgetting、floor或joint之一。即使通过也不直接运行125；失败则停止D63，不放松稳定门或改成场景/角色mask。

## 3.精确执行命令

```powershell
& 'C:\Users\lh594\.conda\envs\ssr-gpu\python.exe' `
  'E:\type10-7\code\snapshots\d63wt\code\scripts\probe_d63_jackknife_stable_fisher_row_splice.py' `
  --d63-arm jackknife_stable_fisher_row_splice `
  --runtime-root 'E:\type10-7\code\snapshots\d41wt' --probe-root 'E:\type10-7\code\snapshots\d63wt' `
  --before-root 'E:\type10-7\automation_reports\CV-SincNet\d18_formal_k10_new5_rx20_1_seed713101_20260717_085303\phase2_capsule_k10_new5\predictor\before\enrollment_only' `
  --before-seal 'E:\type10-7\automation_reports\CV-SincNet\d18_formal_k10_new5_rx20_1_seed713101_20260717_085303\phase2_capsule_k10_new5\seals\before_enrollment.seal.json' --before-seal-sha256 53ace2863c9da6c2f6cc855d602c99f581df6de3d30a9a3ecb89eb6b6f0d9f75 `
  --before-formal-policy 'E:\type10-7\automation_reports\CV-SincNet\d18_formal_k10_new5_rx20_1_seed713101_20260717_085303\formal_execution_policy.json' `
  --before-formal-policy-authorization 'E:\type10-7\automation_reports\CV-SincNet\d18_formal_k10_new5_rx20_1_seed713101_20260717_085303\runtime_authorization_k10_new5\before_formal_policy_authorization.v2.json' `
  --before-signed-policy-authorization-envelope 'E:\type10-7\automation_reports\CV-SincNet\d18_formal_k10_new5_rx20_1_seed713101_20260717_085303\runtime_authorization_k10_new5\before_signed_policy_authorization_envelope.v2.json' --before-signed-policy-authorization-envelope-sha256 31a2ad9918f061b25d5a7ed0cc135df70ae02460c094b2f396bf314817bceb0e `
  --after-root 'E:\type10-7\automation_reports\CV-SincNet\d18_formal_k10_new5_rx20_1_seed713101_20260717_085303\phase2_capsule_k10_new5\predictor\after\enrollment_only' `
  --after-seal 'E:\type10-7\automation_reports\CV-SincNet\d18_formal_k10_new5_rx20_1_seed713101_20260717_085303\phase2_capsule_k10_new5\seals\after_enrollment.seal.json' --after-seal-sha256 c70aedf3a8f059e756806201758c1933a2f3e1ba4df415e69a1c776b1a2b50ff `
  --after-formal-policy 'E:\type10-7\automation_reports\CV-SincNet\d18_formal_k10_new5_rx20_1_seed713101_20260717_085303\formal_execution_policy.json' `
  --after-formal-policy-authorization 'E:\type10-7\automation_reports\CV-SincNet\d18_formal_k10_new5_rx20_1_seed713101_20260717_085303\runtime_authorization_k10_new5\after_formal_policy_authorization.v2.json' `
  --after-signed-policy-authorization-envelope 'E:\type10-7\automation_reports\CV-SincNet\d18_formal_k10_new5_rx20_1_seed713101_20260717_085303\runtime_authorization_k10_new5\after_signed_policy_authorization_envelope.v2.json' --after-signed-policy-authorization-envelope-sha256 a2483d6e9c9c362d89397029ff1e43f48358be3bdb3a05d717ee112b70a0be76 `
  --component-dir 'E:\type10-7\automation_reports\CV-SincNet\d22_int8_anchor_lifecycle_20260717\input\int8_component' --component-manifest-sha256 15b5e144f9af3989421d8e925c17758479c327be47e79222f6363dc63994629c `
  --class-binding 'E:\type10-7\github_publish\CVS-RFFI-repo\analysis\d19_adv3b02_class_binding_20260717.json' --class-binding-sha256 bb89a1dbb831acb374fccfc596ae98b660b496b449bdca577dabb962121c901f `
  --output 'E:\type10-7\automation_reports\CV-SincNet\d63_jackknife_stable_fisher_row_splice_probe_20260719\jackknife_stable_fisher_row_splice' `
  --device auto --mode development_select_unverified_component --candidate-set d42_v1
```

## 4.待完成结果

运行后必须写入105行闭包、7候选、3场景、11类、15fold、总体候选行、jackknife剔除行、联合atomic fallback、D46/D61/D62/D63比较、量化、训练、资源和artifact。

## 5.执行闭包与总判定

- runner完成105/105行、7候选×3场景×5fold、Runner125.5655s、query0、1080个组件fit；进程exit0，RECEIPT、metadata、training log闭包通过。
- D63总体：before93.33%、after82.78%、seen-new82.00%、同rowH81.65%、forgetting10.56pp、joint23.33%、min-before80%、min-after53.33%、min-new63.33%、混淆21/11/16。
- 相对D62：before+0.56pp、after+0.56pp，但new−2.67pp、H−0.97pp、joint−3.33pp、min-new−10pp；old→new−2，new→old+3，new→new+1。5/15 final prediction SHA变化。
- 结论：旧类小幅上升以新类、H、joint和新类floor显著下降为代价；low和rain仍存在场景伤害。状态`COMPLETED_DIAGNOSTIC_NEGATIVE_NOT_PROMOTABLE`，停止D63，不跑第二seed或125。D62仍是当前聚合最强开发点，但D62也不满足项目门槛。

## 6.七候选同row性能

|候选|before|after|new|H|forget|joint|min-before|min-after|min-new|混淆|判定|
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
|B3|87.78%|75.56%|72.67%|73.35%|12.22pp|23.33%|80.00%|60.00%|40.00%|33/22/19|弱比较器|
|D40-HNBR|85.56%|85.00%|15.33%|25.16%|0.56pp|0%|66.67%|63.33%|0%|2/0/0|新类塌缩|
|D41-BEC|86.11%|20.56%|78.67%|31.50%|65.56pp|0%|76.67%|0%|36.67%|142/0/32|旧类塌缩|
|ProtoNet/Z0|71.11%|48.33%|52.67%|48.97%|22.78pp|0%|33.33%|13.33%|3.33%|0/0/0|基线|
|D63 FP32|93.33%|82.78%|82.00%|81.65%|10.56pp|23.33%|80.00%|53.33%|63.33%|21/11/16|matched参考|
|D63 INT8|93.33%|82.78%|82.00%|81.65%|10.56pp|23.33%|80.00%|53.33%|63.33%|21/11/16|诊断阴性|

## 7.分场景表现

|场景|before|after|new|H|forget|joint|min-before|min-after|min-new|混淆|相对D62|
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
|clear|98.33%|91.67%|96.00%|93.57%|6.67pp|50.00%|90.00%|70.00%|90.00%|3/2/0|new−2、H−0.87pp|
|low-elev|95.00%|80.00%|70.00%|73.94%|15.00pp|10.00%|90.00%|60.00%|30.00%|6/7/8|before+3.33、after+1.67，但new−6、H−2.04、forget+1.67、joint−10pp|
|rain|86.67%|76.67%|80.00%|77.45%|10.00pp|10.00%|50.00%|30.00%|70.00%|12/2/8|before−1.67、forget−1.67pp，其余相同|

相对D46，clear after+1.67pp且forget−1.67pp，但new−2pp；low before+6.67pp、after+1.67pp，却new−6pp、H−2.04pp、forget+5pp、joint−10pp；rain before−3.33pp。三个场景均未形成无交换改善。

## 8.逐类性能

|类别|O0|O1|O2|O3|O4|O5|
|---|---:|---:|---:|---:|---:|---:|
|before|96.67%|96.67%|96.67%|80.00%|96.67%|93.33%|
|after|90.00%|90.00%|93.33%|53.33%|76.67%|93.33%|

|类别|N0|N1|N2|N3|N4|
|---|---:|---:|---:|---:|---:|
|seen-new|63.33%|93.33%|76.67%|90.00%|86.67%|

O3仍是after-old瓶颈53.33%；N0由D62的73.33%降至63.33%，构成全局min-new下降10pp的直接来源。low场景N0仅30%、N2仅50%；方法保护旧类时挤压了弱新类的决策空间。

## 9.十五fold同row性能

|场景-fold|before|after|new|H|forget|joint|floor(b/a/n)|混淆|
|---|---:|---:|---:|---:|---:|---:|---:|---:|
|clear-0|100.00%|100.00%|90.00%|94.74%|0pp|50%|100/100/50|0/1/0|
|clear-1|100.00%|83.33%|100.00%|90.91%|16.67pp|0%|100/0/100|1/0/0|
|clear-2|91.67%|83.33%|90.00%|86.54%|8.33pp|50%|50/50/50|1/1/0|
|clear-3|100.00%|100.00%|100.00%|100.00%|0pp|100%|100/100/100|0/0/0|
|clear-4|100.00%|91.67%|100.00%|95.65%|8.33pp|50%|100/50/100|1/0/0|
|low-0|100.00%|66.67%|80.00%|72.73%|33.33pp|50%|100/50/50|4/1/1|
|low-1|83.33%|58.33%|60.00%|59.15%|25.00pp|0%|50/50/0|1/1/3|
|low-2|91.67%|91.67%|70.00%|79.38%|0pp|0%|50/50/0|0/2/1|
|low-3|100.00%|100.00%|70.00%|82.35%|0pp|0%|100/100/0|0/1/2|
|low-4|100.00%|83.33%|70.00%|76.09%|16.67pp|0%|100/50/0|1/2/1|
|rain-0|75.00%|83.33%|60.00%|69.77%|−8.33pp|0%|0/50/0|2/0/4|
|rain-1|100.00%|66.67%|90.00%|76.60%|33.33pp|0%|100/0/50|4/1/0|
|rain-2|91.67%|83.33%|80.00%|81.63%|8.33pp|50%|50/50/50|1/0/2|
|rain-3|83.33%|75.00%|90.00%|81.82%|8.33pp|0%|50/0/50|2/0/1|
|rain-4|83.33%|75.00%|80.00%|77.42%|8.33pp|0%|50/50/0|3/1/1|

相对D62有5/15个final prediction SHA变化：clear-1与rain-3汇总指标不变；clear-2的new−10pp、H−4.37pp；low-1的new−10pp、H−4.48pp；low-4虽before/after各+8.33pp，却new−20pp、H−5.73pp、joint−50pp。收益与伤害集中在同一批不稳定弱类折，不能以旧类均值覆盖新类退化。

## 10.门控、量化、训练与资源

- INT8 before：总体候选32行、jackknife稳定18、剔除14，10/15 fit激活；final：总体候选45行、稳定18、剔除27，10/15 fit激活。两个阶段联合原子回退均为0。
- 分场景before稳定行clear/low/rain为3/10/5，final为8/9/1。全INT8＋FP32的60个fit审计共有154个总体候选行、稳定72、剔除82、40个fit激活。
- 机制解释：D63把D62经常触发的整fit原子回退改成逐行稳定筛选，保留行数反而从D62的46增至72；support内部分类Pareto稳定不能保证注册后的old/new联合边界稳定。
- 量化：before/final outer argmax变化0/0，margin sign flip0，support argmax变化0/0；最大score误差0.002003，量化门通过。
- 训练：基础D42仍20epoch/20step；epoch1 loss1.0320、support acc95.14%，epoch20 loss0.1027、support acc100%，所有epoch query rows总和0。D63自身0参数、0额外optimizer step。
- 资源：72次LDA fit、18,000,009,216 LDA MAC；额外36次组件fit与16,934,178,816 LDA MAC，Fisher稠密代数上界6,879,707,136 MAC，D62基础门10,048标量MAC，D63 jackknife门80,384标量MAC；总适配24,891,304,354 MAC。query6,624MAC、参数2,016、持久态8,583B、registry941B、峰值CUDA22,886,912B。
- artifact：`full_performance_summary.json`91,981B，SHA256=`52e90442fdbe75f7fb0cbc2788c2a86dcef5c6b9ec3061d4d3c823d2997a3c91`；`D63_PROBE_METADATA.json`2,281B，SHA256=`2f126e14b7ef23e9da5441472512c982943cf540d3744aa58588d082bfaa3c3e`。

## 11.与项目门槛差距及下一步

D63相对K10目标仍差after9.22pp、min-old34.67pp、new5 10.00pp；且只是一receiver、一seed开发单元，不能作正式性能声明。D63停止后必须执行D61–D63三轮回顾，重新核对目标、`项目.md`、对话索引、完整报告与日志，再选择不同机制；不得把jackknife强度改成可调阈值继续扫描。

## 12.D61–D63强制技术复盘

### 12.1权威文件、索引与完整日志

复盘重新完整读取活动goal objective、`项目.md`、`PROJECT_PROTOCOL.md`、`PHASE2_DATA_VALIDATION_APPENDIX.md`、`STAGE2_METHOD_RESEARCH_GOAL.md`和历史普查。conversation index于2026-07-19刷新为1008条，并搜索`D63`、`crossfitted Fisher`、`D58 D59 D60 D61 D62`、`support held query 泛化 old new 注册 遗忘`。当前活动会话尚未形成可检索的D61–D63摘要，因此索引只用于排除历史重复路线，当前性能以落盘report、training log、summary和metadata为权威。

我们递归枚举D61-R1、D62、D63三个run目录并完整解析全部结构化artifact。每个run均包含7个JSON、1个JSONL、105条training row、7候选、15条INT8目标row和完整epoch1–20；三轮共解析315条training row，没有JSON错误、NaN或Inf，所有目标trace的query rows总和均为0。D61/D62的完整summary stdout未出现Traceback、RuntimeError、Exception、OOM、Killed或warning marker；D63进程exit0且RECEIPT/metadata闭包。没有只读tail或抽样替代全日志解析。

历史普查中“D36只有设计/core证据”已被当前D36主报告推翻：本地`ssr-gpu`105行筛选已完成，D36-A/B/C均为性能负结果。后续路线以当前报告为准，不再把D36联合对角/rank-2连续margin作为未验证候选重复执行。

### 12.2三轮同row比较

|版本|单一机制|before|after|new|H|forget|joint|min-before|min-after|min-new|混淆|资源/量化|判定|
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---|
|D46|类级inner-LOO full/block可靠度融合|92.22%|81.67%|84.67%|82.33%|10.56pp|23.33%|80.00%|53.33%|73.33%|25/8/15|约1.07B LDA MAC；量化0/0/0|三轮基准|
|D61|共享identity-primary Fisher残差|90.00%|83.33%|76.00%|78.96%|6.67pp|26.67%|76.67%|60.00%|43.33%|18/16/20|约7.96B MAC；final argmax翻转1|旧类保护、新类塌缩|
|D62|inner-held TP/FP Pareto匿名行拼接|92.78%|82.22%|84.67%|82.62%|10.56pp|26.67%|80.00%|53.33%|73.33%|23/8/15|约24.89B MAC；量化0/0/0|当前聚合最强，场景交换|
|D63|leave-one-fold稳定Pareto行拼接|93.33%|82.78%|82.00%|81.65%|10.56pp|23.33%|80.00%|53.33%|63.33%|21/11/16|约24.89B MAC；量化0/0/0|旧类小升、新类下尾退化|

D61证明共享Fisher方向能减少旧类遗忘和old→new侵入，但同一方向把N0压到43.33%，不能作为联合空间主变换。D62最有价值的正证据来自clear-fold3：after91.67%→100%、forget8.33pp→0、joint50%→100%，new保持100%；然而low forgetting恶化3.33pp、rain before下降1.67pp，说明总体support TP/FP门没有场景泛化能力。D63将154个总体候选行剔除82个并消除联合原子回退，却保留72行、比D62的46行更多；稳定的support分类行仍使N0下降10pp。三轮共同否证“继续改support安全门即可解决held old/new联合边界”。

### 12.3协议与目标核对

- 三轮均复用匹配`VALIDATED_ONCE/p2_min_v1`的固定接收IQ，不访问clean/source，不生成第二LEO观测。
- 所有公式对类别置换等变，无class ID、old/new角色、场景、receiver、query真值、真实batch类数、quota或global assignment分支。
- query逐样本面对全部注册类；query fit/use计数为0；target-old/new走同一正式int8/FP16 affine state。
- 三轮都同时报告before-old、after-old、new、H、forgetting、全部旧/新类、三场景、15fold和混淆，没有只优化Stage2-B或只优化Stage2-C。
- D62离K10正式门仍差after9.78pp、min-old34.67pp、new5 7.33pp；没有版本满足启动125的研发门，125继续不运行。

### 12.4停止路线与下一机制

停止共享Fisher全量替换、TP/FP匿名行拼接、jackknife强度/阈值扫描、角色或场景mask。结合D58–D60复盘，一并停止类score幅度/截距校准、full↔block协方差位置扫描和跨块谱扫描。D36的联合对角/rank-2梯度头也已由当前105行报告否证，不能当作未测路线重启。

下一轮D64改换为全注册类连续局部判别机制：对每一对匿名类别使用完全相同的3-block auto-shrinkage二类LDA，在该pair support上以margin RMS无参数归一化，再将全部有向pair margin平均编译回每类一个affine row。它与D46的单一主要差异是“全局共享协方差＋逐类full/block权重”改为“全部pair局部协方差＋连续全pair tournament”，不是hard gate、行替换或角色路由。最终仍只有单一int8/FP16 affine state，query不保留pair图、不做联合优化。预期信号是同时降低old→new/new→old/new→new并抬升O3/N0/N2下尾；若任一场景或new/H/floor相对D62/D46受损，D64立即停止，不扫描pair阈值、投票权重或full/block变体。
