# D62交叉拟合Fisher行级Pareto拼接探针报告

## 1.执行前登记

- ID：`d62_crossfitted_fisher_row_splice_probe_20260719`；操作者：Codex`/root`；状态：`READY_TO_RUN_LOCAL_DEVELOPMENT_CELL`。
- 目标：仅在support inner-held证明匿名类的D61仿射行相对D46满足positive不降、false-positive不增且严格改善时替换该行；联合交互不安全则原子回退D46。
- 比较目标D46：before92.22%、after81.67%、new84.67%、H82.33%、forget10.56pp、joint23.33%、min-before80%、min-after53.33%、min-new73.33%、混淆25/8/15。
- cell：receiver`20-1`、seed`713101`、K10/new5、3场景×5 outer fold、实际K8；复用匹配`VALIDATED_ONCE/p2_min_v1`的D18 capsule，不重验数据。
- 版本：预注册`a526219d`；实现`44eea565`；worktree`E:\type10-7\code\snapshots\d62wt` detached clean；脚本SHA256=`38ae1114a06d135bca806f470417cd28a634fec0da449888665c6843615d4a20`。
- 验证：py_compile通过；D43–D46、D56、D57、D61、D62联合回归71/71通过；已知pytest exit0后的临时junction清理噪声不影响结论；diff check通过。
- 本轮只在本地`ssr-gpu`执行，不访问N607，无远端PID/GPU/sync。

## 2.机制、资源与判门

D62在每个fit额外完成full/block各1个outer和K个inner组件fit，同时生成D46/D61分数；K8时before/final共36个额外组件fit。类行门对class ID置换等变，无old/new/scene/receiver分支，无alpha、rank、gain、threshold或顺序扫描。最终只保存一套int8/FP16 affine state，query额外MAC/state为0。

相对D46必须同时保持总体、三场景、三类floor/joint、forgetting、混淆和量化，并至少严格提高after/forget/floor之一且改变≥1折；即使通过也不直接跑125。失败则停止D62，不放宽TP/FP门或改成角色/场景mask。

## 3.精确执行命令

```powershell
& 'C:\Users\lh594\.conda\envs\ssr-gpu\python.exe' `
  'E:\type10-7\code\snapshots\d62wt\code\scripts\probe_d62_crossfitted_fisher_row_splice.py' `
  --d62-arm crossfitted_fisher_row_splice `
  --runtime-root 'E:\type10-7\code\snapshots\d41wt' --probe-root 'E:\type10-7\code\snapshots\d62wt' `
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
  --output 'E:\type10-7\automation_reports\CV-SincNet\d62_crossfitted_fisher_row_splice_probe_20260719\crossfitted_fisher_row_splice' `
  --device auto --mode development_select_unverified_component --candidate-set d42_v1
```

## 4.待完成结果

运行后必须写入105行闭包、7候选、3场景、11类、15fold、接受行/atomic fallback、D46/D61/D62比较、量化、训练、资源和artifact。

## 5.执行闭包与结论

- runner完成105/105行、7候选×3场景×5fold、Runner124.7751s、query0、1080个额外组件fit。Codex前台等待在124秒超时，但落地检查确认D62进程为0、RECEIPT/metadata/105行完整，因此没有重跑。
- 总体D62：before92.78%、after82.22%、seen-new84.67%、同rowH82.62%、forgetting10.56pp、joint26.67%、min-before80%、min-after53.33%、min-new73.33%、混淆23/8/15。
- 相对D46：before+0.56pp、after+0.56pp、H+0.29pp、joint+3.33pp、old→new−2；new、forgetting、三项全局class floor、new→old/new→new均不变。2/15 final prediction变化，量化0/0/0。
- 结论：D62为当前聚合最强开发点，但rain before−1.67pp，low forgetting+3.33pp，违反三场景不交换伤害门；且离正式K10门仍很远。状态`COMPLETED_AGGREGATE_BEST_DIAGNOSTIC_NOT_PROMOTABLE`，不跑第二seed或125。

## 6.七候选同row性能

|候选|before|after|new|H|forget|joint|min-before|min-after|min-new|混淆|判定|
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
|B3|87.78%|75.56%|72.67%|73.35%|12.22pp|23.33%|80.00%|60.00%|40.00%|33/22/19|弱比较器|
|D40-HNBR|85.56%|85.00%|15.33%|25.16%|0.56pp|0%|66.67%|63.33%|0%|2/0/0|新类塌缩|
|D41-BEC|86.11%|20.56%|78.67%|31.50%|65.56pp|0%|76.67%|0%|36.67%|142/0/32|旧类塌缩|
|ProtoNet/Z0|71.11%|48.33%|52.67%|48.97%|22.78pp|0%|33.33%|13.33%|3.33%|0/0/0|基线|
|D62 FP32|92.78%|82.22%|84.67%|82.62%|10.56pp|26.67%|80.00%|53.33%|73.33%|23/8/15|matched参考|
|D62 INT8|92.78%|82.22%|84.67%|82.62%|10.56pp|26.67%|80.00%|53.33%|73.33%|23/8/15|聚合最强，场景门失败|

## 7.分场景表现

|场景|before|after|new|H|forget|joint|min-before|min-after|min-new|混淆|相对D46|
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
|clear|98.33%|91.67%|98.00%|94.44%|6.67pp|50.00%|90.00%|70.00%|90.00%|2/1/0|after+1.67、H+0.87、forget−1.67、joint+10pp|
|low-elev|91.67%|78.33%|76.00%|75.98%|13.33pp|20.00%|80.00%|60.00%|50.00%|8/5/7|before+3.33，但forget+3.33pp|
|rain|88.33%|76.67%|80.00%|77.45%|11.67pp|10.00%|60.00%|30.00%|70.00%|13/2/8|before−1.67、forget−1.67，其余不变|

## 8.逐类与15fold性能

|类别|O0|O1|O2|O3|O4|O5|
|---|---:|---:|---:|---:|---:|---:|
|before|96.67%|96.67%|96.67%|80.00%|93.33%|93.33%|
|after|90.00%|90.00%|93.33%|53.33%|73.33%|93.33%|

|类别|N0|N1|N2|N3|N4|
|---|---:|---:|---:|---:|---:|
|seen-new|73.33%|93.33%|76.67%|90.00%|90.00%|

|场景-fold|before|after|new|H|forget|joint|floor(b/a/n)|混淆|
|---|---:|---:|---:|---:|---:|---:|---:|---:|
|clear-0|100.00%|100.00%|90.00%|94.74%|0pp|50%|100/100/50|0/1/0|
|clear-1|100.00%|83.33%|100.00%|90.91%|16.67pp|0%|100/0/100|0/0/0|
|clear-2|91.67%|83.33%|100.00%|90.91%|8.33pp|50%|50/50/100|1/0/0|
|clear-3|100.00%|100.00%|100.00%|100.00%|0pp|100%|100/100/100|0/0/0|
|clear-4|100.00%|91.67%|100.00%|95.65%|8.33pp|50%|100/50/100|1/0/0|
|low-0|100.00%|66.67%|80.00%|72.73%|33.33pp|50%|100/50/50|4/1/1|
|low-1|83.33%|58.33%|70.00%|63.64%|25.00pp|0%|50/50/0|1/0/3|
|low-2|83.33%|91.67%|70.00%|79.38%|−8.33pp|0%|50/50/0|0/2/1|
|low-3|100.00%|100.00%|70.00%|82.35%|0pp|0%|100/100/0|0/1/2|
|low-4|91.67%|75.00%|90.00%|81.82%|16.67pp|50%|50/50/50|3/1/0|
|rain-0|83.33%|83.33%|60.00%|69.77%|0pp|0%|50/50/0|2/0/4|
|rain-1|100.00%|66.67%|90.00%|76.60%|33.33pp|0%|100/0/50|4/1/0|
|rain-2|91.67%|83.33%|80.00%|81.63%|8.33pp|50%|50/50/50|1/0/2|
|rain-3|83.33%|75.00%|90.00%|81.82%|8.33pp|0%|50/0/50|3/0/1|
|rain-4|83.33%|75.00%|80.00%|77.42%|8.33pp|0%|50/50/0|3/1/1|

相对D46仅clear-1与clear-3的final prediction SHA变化。clear-1汇总指标不变；clear-3把after从91.67%提高到100%、forget从8.33pp降到0、joint从50%提高到100%，且new保持100%。其余13折final完全等价D46。before的low/rain变化则造成场景交换。

## 9.门控、量化、训练与资源

- INT8 before：6/15 fit激活、接受17行；7个fit联合原子回退、2个无行通过。final：3/15 fit激活、接受6行；12个fit联合原子回退。FP/TP门真实过滤了D61，未整体替换。
- 场景激活：before clear/low/rain为1/4/1个fit、接受3/12/2行；final为2/1/0个fit、接受5/1/0行。rain final全部回退D46。
- Fisher：before rank恒5、gain0.4793–0.9567；final rank恒10、gain0.2446–0.9124。
- 量化：before/final outer argmax变化0/0，margin sign flip0，support argmax变化0/0；最大score误差0.001915，量化门通过。
- 训练：基础D42仍20epoch/20step；epoch1 loss1.0320、support acc95.14%，epoch20 loss0.1027、support acc100%，query rows总和0。D62自身0参数、0额外optimizer step。
- 资源：72次LDA fit、18,000,009,216 LDA MAC；D62额外36次fit与16,934,178,816 LDA MAC，Fisher稠密代数上界6,879,707,136 MAC，门控10,048标量MAC；总适配24,891,223,970 MAC。query6,624MAC、参数2,016、持久态8,583B、registry941B、峰值CUDA22,886,912B。
- artifact：`full_performance_summary.json`81,020B，SHA256=`04e5d7cc0a3268a6475aa75a07356d894d99c53d06add69a33f1e7c1dcc02252`；`D62_PROBE_METADATA.json`1,880B，SHA256=`0d41559f310d1b5d930c3c44cf071ee2f93cf835f58180f98a1641749c4d6419`。
