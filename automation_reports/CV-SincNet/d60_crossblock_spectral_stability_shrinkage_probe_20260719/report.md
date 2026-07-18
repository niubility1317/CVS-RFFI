# D60跨块谱稳定性收缩探针报告

## 1.执行前登记

- ID：`d60_crossblock_spectral_stability_shrinkage_probe_20260719`；操作者：Codex`/root`；状态：`COMPLETED_DIAGNOSTIC_NEGATIVE_NOT_PROMOTABLE`。
- 目标：用support内部physical-rank留一折稳定度连续收缩跨z160/FFT96/RF32谱模态，同时保持所有类别共享同一协方差与单一query state。
- 比较目标D46：before92.22%、after81.67%、new84.67%、H82.33%、forget10.56pp、joint23.33%、min-before80%、min-after53.33%、min-new73.33%、混淆25/8/15。
- cell：receiver`20-1`、seed`713101`、K10/new5、3场景×5 outer fold，实际K8；复用匹配`VALIDATED_ONCE/p2_min_v1`的D18 capsule，不重验数据。
- 版本：预注册`3990667f`；实现`a8c96f56`；worktree`E:\type10-7\code\snapshots\d60wt` detached clean；脚本SHA256=`0591d8598795bead21631ed60597d409ae9f00f1f3468221e24430517ab15f35`。
- 文件：实现`code/scripts/probe_d60_crossblock_spectral_stability_shrinkage.py`；测试`tests/test_probe_d60_crossblock_spectral_stability_shrinkage.py`；追溯`analysis/d60_crossblock_spectral_stability_shrinkage_traceability_20260719.md`。
- 验证：py_compile通过；D42–D46＋D59–D60回归118/118通过；diff check通过。N607不访问，无远端PID/GPU/sync。

## 2.机制、资源与判门

`R=B^(-1/2)(F−B)B^(-1/2)=Vdiag(λ)V^T`；每个inner rank fold计算`q_rj`，稳定度`s_j=mean(q_rj)^2/mean(q_rj^2)`；最终`G=B^(1/2)[I+Vdiag(s⊙λ)V^T]B^(1/2)`。无threshold/rank/ridge/权重扫描，无class/role/scene分支。K8时before/final各1个main fit＋8个inner covariance fit，总18个LDA fit；只持久化一套int8/FP16 state。

相对D46必须同时满足总体、三场景、三类floor/joint、forgetting、混淆和量化不退化，并至少一个final floor严格提高；即使通过也不直接运行125。完成后必须写入7候选、3场景、11类、15fold、谱稳定度、训练、量化、资源和artifact。

## 3.精确执行命令

```powershell
& 'C:\Users\lh594\.conda\envs\ssr-gpu\python.exe' `
  'E:\type10-7\code\snapshots\d60wt\code\scripts\probe_d60_crossblock_spectral_stability_shrinkage.py' `
  --d60-arm crossblock_spectral_stability_shrinkage `
  --runtime-root 'E:\type10-7\code\snapshots\d41wt' --probe-root 'E:\type10-7\code\snapshots\d60wt' `
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
  --output 'E:\type10-7\automation_reports\CV-SincNet\d60_crossblock_spectral_stability_shrinkage_probe_20260719\crossblock_spectral_stability_shrinkage' `
  --device auto --mode development_select_unverified_component --candidate-set d42_v1
```

预期artifact为training/support/selection/receipt/metadata/geometry/resource；风险是inner covariance拟合增加host FP64成本。任何SPD、分区、source或资源闭包失败均fail closed，不允许隐式ridge或降级。

## 4.执行闭包与结论

- 105/105行、exit0、Runner`60.4345s`；7候选×3场景×5fold完整。query0，source/clean/role/count/quota/global assignment均不可达；ground int8逐bit不变；60个active fit audit、每fit8个inner exact-once分区和18-fit资源闭包通过。
- 总体：before`91.11%`、after`81.11%`、new`83.33%`、同rowH`81.45%`、forgetting`10.00pp`、joint`23.33%`、min-before`80.00%`、min-after`50.00%`、min-new`73.33%`，混淆`27/9/16`。
- 相对D46：before`−1.11pp`、after`−0.56pp`、new`−1.33pp`、H`−0.88pp`、forgetting`−0.56pp`、joint不变、min-after`−3.33pp`，混淆`+2/+1/+1`；8/15个prediction SHA变化。
- 结论：稳定谱收缩的forgetting改善不足以抵消before/after/new/H、旧类floor和三类混淆全面退化；D60不晋级、不跑第二seed、不运行125。当前最强仍为D46。

## 5.七候选同row性能

|候选|before|after|new|H|forget|joint|min-before|min-after|min-new|混淆|判定|
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
|B3_SINGLE_IQ_DIAG_FFTRF|87.78%|75.56%|72.67%|73.35%|12.22pp|23.33%|80.00%|60.00%|40.00%|33/22/19|弱比较器|
|D40 HNBR negative|85.56%|85.00%|15.33%|25.16%|0.56pp|0%|66.67%|63.33%|0%|2/0/0|新类不可达|
|D41 BEC negative|86.11%|20.56%|78.67%|31.50%|65.56pp|0%|76.67%|0%|36.67%|142/0/32|旧类崩溃|
|ProtoNet CDA|71.11%|48.33%|52.67%|48.97%|22.78pp|0%|33.33%|13.33%|3.33%|0/0/0|弱基线|
|D60 FP32 matched|91.11%|81.11%|83.33%|81.45%|10.00pp|23.33%|80.00%|50.00%|73.33%|27/9/16|量化对照|
|D60 INT8|91.11%|81.11%|83.33%|81.45%|10.00pp|23.33%|80.00%|50.00%|73.33%|27/9/16|主候选，负结果|
|Z0_SUPPORT_ONLY|71.11%|48.33%|52.67%|48.97%|22.78pp|0%|33.33%|13.33%|3.33%|0/0/0|control|

FP32汇总与INT8相同，但量化并非逐fold完全一致，见第9节。unknown/coverage/rollback/defer均为N/A。

## 6.逐场景表现

|场景|before|after|new|H|forget|joint|min-before|min-after|min-new|混淆|相对D46|
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
|clear|98.33%|90.00%|96.00%|92.70%|8.33pp|40.00%|90%|70%|90%|4/2/0|new−2、H−0.87pp，new→old+1|
|low-elev|86.67%|75.00%|76.00%|74.25%|11.67pp|20.00%|80%|50%|50%|11/4/8|before−1.67、after−3.33、H−1.74、forget+1.67pp|
|rain|88.33%|78.33%|78.00%|77.41%|10.00pp|10.00%|60%|30%|70%|12/3/8|after+1.67、forget−3.33pp，但before/new各−1.67/−2pp|

rain的after/forgetting正信号以before、新类和new→old退化交换；low-elev则旧类适应和forgetting同时变差。没有场景形成old/new联合改善。

## 7.全部匿名类性能

|旧类|before→after|D46差异|
|---|---:|---|
|O0`1f33`|90.00→86.67%|after−3.33pp|
|O1`33bb`|93.33→90.00%|before/after各−3.33pp|
|O2`75aa`|93.33→90.00%|before−3.33pp|
|O3`8b02`|80.00→50.00%|after−3.33pp，旧类floor|
|O4`a53c`|100.00→76.67%|after+3.33pp|
|O5`f8df`|90.00→93.33%|相同|

|新类|D60|D46|变化|
|---|---:|---:|---:|
|N0`09f8`|73.33%|73.33%|0|
|N1`1c2a`|93.33%|93.33%|0|
|N2`b8fb`|73.33%|76.67%|−3.33pp|
|N3`d3af`|86.67%|90.00%|−3.33pp|
|N4`f608`|90.00%|90.00%|0|

D60保住D46的新类最低值，但没有提高它；同时O3旧类floor降至50%，N2/N3各丢1个样本。

## 8.十五fold完整表现

|场景|fold|before|after|new|H|forget|joint|before/after/new floor|混淆|
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
|clear|0|100|100|90|94.74|0|50|100/100/50|0/1/0|
|clear|1|100|83.33|100|90.91|16.67|0|100/0/100|1/0/0|
|clear|2|91.67|83.33|90|86.54|8.33|50|50/50/50|1/1/0|
|clear|3|100|91.67|100|95.65|8.33|50|100/50/100|1/0/0|
|clear|4|100|91.67|100|95.65|8.33|50|100/50/100|1/0/0|
|low|0|91.67|66.67|80|72.73|25|50|50/50/50|4/1/1|
|low|1|66.67|58.33|70|63.64|8.33|0|50/50/0|2/0/3|
|low|2|91.67|83.33|60|69.77|8.33|0|50/50/0|1/2/2|
|low|3|100|91.67|70|79.38|8.33|0|100/50/0|1/1/2|
|low|4|83.33|75|100|85.71|8.33|50|50/50/100|3/0/0|
|rain|0|83.33|83.33|60|69.77|0|0|50/50/0|2/1/3|
|rain|1|91.67|66.67|90|76.60|25|0|50/0/50|4/0/1|
|rain|2|91.67|83.33|80|81.63|8.33|50|50/50/50|1/0/2|
|rain|3|91.67|83.33|80|81.63|8.33|0|50/0/50|2/1/1|
|rain|4|83.33|75|80|77.42|8.33|0|50/50/0|3/1/1|

表内accuracy/H/joint/floor均为百分数，forget为pp；每行指标来自同一fold。

## 9.谱稳定性、训练、量化与资源

- before稳定度全模态min/mean/max=`0/0.3919/0.9991`，每fit平均173.8/288个模态因fold Rayleigh均值精确为0而收回block；final为`0/0.6247/0.9993`，平均106.6个零模态。没有模态达到预注册的数值near-one条件。
- 收缩协方差condition number before均值`136,561`、final`103,430`；30个fold-Rayleigh SHA均唯一且所有inner分区exact-once。机制真实激活，不是D59或block回退。
- 训练epoch1 loss/support-acc均值`1.0320/95.14%`，epoch20为`0.1027/100%`；300条trace finite、20epoch/20step、query rows0。
- 量化：before argmax变化0，final argmax变化1，margin翻转1，support变化0/0；max score误差min/mean/max=`0.0181/0.0362/0.0563`。汇总整数偶然抵消，仍触发0/0/0硬门失败。
- 资源：18次LDA fit，LDA MAC`532,915,200`；其中inner16次/`472,449,024` MAC；谱代数保守上界`1,337,720,832`，总适配`1,875,612,672`；query`6,624` MAC、参数`2,016`、state`8,583B`、20epoch/20step、CUDA metric峰值`22,886,912B`。host FP64峰值未实测。

## 10.预注册门与artifact

总体before/after/new/H、min-after、三场景联合、三类混淆及量化门均FAIL；forgetting、joint、min-before、min-new和prediction变化门PASS；final floor无严格改善。最终为`COMPLETED_DIAGNOSTIC_NEGATIVE_NOT_PROMOTABLE`。

|Artifact|Bytes|SHA256|
|---|---:|---|
|training_log.jsonl|4,714,053|`2ae820c383fc22daa3478af0a692c65104209cf82a343a32c3cdcd7c8ae83fed`|
|support_audit.json|313,388|`9f29d3a0c6b5369be7e350874c2f47f98bd045526b1746622634834a1541ec87`|
|selection.json|2,991|`83f405752f3d8dbe23e5f5beaf67348152204e8e65921074b757b0b44f17b2f0`|
|RECEIPT.json|4,750|`b785d63d43846c86df61dc26e0ff2e99e0af89bc386608b22b3075885fff6d7f`|
|D60_PROBE_METADATA.json|1,664|`d3d72e0cf362e8970d243433477df4a7d2b7d7a0c168a7839b951064195b5d1f`|
|full_performance_summary.json|71,666|`5b7a10423aff3b833d7f73f3bd38aa72acad409bc8254f9a1f8e29b2627f02f8`|

## 11.D58–D60强制复盘

复盘前已重新读取active objective/项目协议（本轮启动时已完成），刷新conversation index至1008条并搜索`D58 D59 D60 score calibration covariance crossblock stability D46`。索引只命中较早、不同协议的source-logit校准失败与receiver-conditioned support-head经验；它不能直接导入当前`p2_min_v1`，尤其不能带入source/clean或receiver-ID依赖。

三轮经验：D58按类score二次校准使inner-held变好却outer全面崩塌；D59共享SPD中点数值正确但只复现D45的旧/新交换；D60按fold稳定性选择跨块谱模态仍同时损失old/new并出现量化翻转。共同结论是：当前瓶颈不是“如何在full/block协方差之间更聪明地选位置”，而是D46的类别局部边界证据与统一部署state之间缺少低方差、可泛化的约束。

停止路线：所有类别幅度/截距校准；full/block固定或数据依赖位置扫描；跨块谱threshold/rank/指数扫描。下一轮必须换机制族，继续同等检查before/after/new/H、逐类old、forgetting与混淆，并保持LEO_weak-only、support-only、no clean/source、no query truth/role/quota/count/global assignment。D61启动前应先审计此前未重复的“共享低秩判别子空间＋单位尺度原型”路线，而不是再修D60。
