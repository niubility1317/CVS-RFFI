# D61 identity-primary共享Fisher残差探针报告

## 1.执行前登记

- ID：`d61_identity_primary_fisher_residual_probe_20260719`；操作者：Codex`/root`；状态：`READY_TO_RUN_LOCAL_DEVELOPMENT_CELL`。
- 目标：在D46的full/block每个outer与inner组件fit中，从该fit可见support闭式估计`A=I+Udiag(b/(b+w))U^T`，保留identity正交补并编译回单一仿射state。
- 比较目标D46：before92.22%、after81.67%、new84.67%、H82.33%、forget10.56pp、joint23.33%、min-before80%、min-after53.33%、min-new73.33%、混淆25/8/15。
- cell：receiver`20-1`、seed`713101`、K10/new5、3场景×5 outer fold，实际K8；复用匹配`VALIDATED_ONCE/p2_min_v1`的D18 capsule，不重验数据。
- 版本：预注册`710133f5`；实现`759be372`；worktree`E:\type10-7\code\snapshots\d61wt` detached clean；脚本SHA256=`8a243a0306a1334e3d3fdca7a190422b3756ef792aa86acf1055890da09292b8`。
- 文件：`analysis/d61_identity_primary_fisher_residual_traceability_20260719.md`、`code/scripts/probe_d61_identity_primary_fisher_residual.py`、`tests/test_probe_d61_identity_primary_fisher_residual.py`。
- 验证：py_compile通过；D42–D46＋D61回归91/91通过；已知pytest临时junction清理`PermissionError`发生在exit0之后，不影响测试结论；diff check通过。N607不访问，无远端PID/GPU/sync。

## 2.机制、资源与判门

所有类别共享同一`A`，其特征值锁在`[1,2]`、正交补严格为1；无rank、alpha、shrinkage、gain指数或阈值扫描，无训练参数、epoch、逐类校准、角色/场景分支。D46每个inner折只用该折train support重新估计`A`。K8时before/final共36个组件变换fit；只持久化编译后的int8/FP16 affine state，query额外MAC/state为0。

相对D46必须同时满足总体、三场景、floor/joint、forgetting、混淆和量化不退化，并至少一个final floor严格提高；即使通过也不直接运行125。完成后必须写入7候选、3场景、11类、15fold、Fisher rank/gain、混淆、量化、资源和artifact。

## 3.精确执行命令

```powershell
& 'C:\Users\lh594\.conda\envs\ssr-gpu\python.exe' `
  'E:\type10-7\code\snapshots\d61wt\code\scripts\probe_d61_identity_primary_fisher_residual.py' `
  --d61-arm identity_primary_fisher_residual `
  --runtime-root 'E:\type10-7\code\snapshots\d41wt' --probe-root 'E:\type10-7\code\snapshots\d61wt' `
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
  --output 'E:\type10-7\automation_reports\CV-SincNet\d61_identity_primary_fisher_residual_probe_20260719\identity_primary_fisher_residual' `
  --device auto --mode development_select_unverified_component --candidate-set d42_v1
```

预期artifact为training/support/selection/receipt/metadata/geometry/resource。任何变换边界、折内独立拟合、source closure、编译等价或资源闭包失败均fail closed，不允许隐式回退或调参。

## 4.首次执行失败

锁定实现`759be372`运行33.7s后在首个真实block组件fail closed：`D43 structured covariance is not positive definite`。原因边界是全局Fisher旋转后再强制三块协方差破坏了block组件的结构假设；未完成105行、未生成可评分指标，不得把本次失败描述为性能结果。失败目录原样保留，不覆盖、不删除。

R1已在追溯文档预注册：先按原坐标拟合D46 full/block组件，再把同fit support闭式估计的`A`编译进组件系数`W=W0A^T`；不再改动协方差坐标。性能门、协议门、无扫描和详细报告要求全部不变；R1另用`identity_primary_fisher_residual_r1`输出目录。

## 5.R1执行前锁定

- 预注册修复：`826bb33e`；实现：`6d432927`；worktree：`E:\type10-7\code\snapshots\d61r1wt` detached clean；脚本SHA256=`e557e0a50f2ebf812b111a4630d368e82d004adf4afbb90f242a51d5b7625b3f`。
- 验证：D42–D46＋D61回归91/91，diff check通过；唯一机制变化是协方差在原坐标拟合，随后编译共享Fisher残差。
- 精确命令复用第3节所有capsule、seal、policy、component、class-binding与device参数，只将脚本根替换为`d61r1wt`、输出替换为`identity_primary_fisher_residual_r1`；执行时完整命令由shell历史和本报告共同闭合。
- R1仍需完成105/105行和1080个组件fit；若再次结构/数值失败或任一性能门失败，D61路线停止，不做R2。

## 6.R1执行闭包与结论

- runner完成105/105行、7候选×3场景×5fold，耗时80.5443s；query0，clean/source/role/count/quota/global assignment均不可达。每个D42 target row记录36个LDA fit和36个D61组件变换fit；两类target共1080个组件fit。
- runner后验证器先后因D43/D61组合audit命名空间以及D61附加资源MAC未从D46基准视图扣除而fail closed。修复提交`e32febfe`、`f57ec7d9`、`da0f2a8a`只执行离线验证，不重跑runner、不改`training_log.jsonl`；最终30个D46 row、60个D61 outer audit、105行、query0和source closure全部通过。
- 总体D61：before90.00%、after83.33%、seen-new76.00%、同rowH78.96%、forgetting6.67pp、joint26.67%、min-before76.67%、min-after60.00%、min-new43.33%、混淆18/16/20。
- 相对D46：after+1.67pp、forgetting−3.89pp、joint+3.33pp、min-after+6.67pp、old→new−7；但before−2.22pp、new−8.67pp、H−3.38pp、min-before−3.33pp、min-new−30.00pp，new→old+8、new→new+5。15/15 prediction SHA变化。
- 结论：D61强化了旧类保护，却显著压制新类判别，违反域适应与新类注册等权原则；聚合、new floor、三场景、混淆和量化门均失败。状态`COMPLETED_DIAGNOSTIC_NEGATIVE_NOT_PROMOTABLE`，停止D61，不跑第二seed或125；当前最强仍为D46。

## 7.七候选同row性能

|候选|before|after|new|H|forget|joint|min-before|min-after|min-new|混淆|判定|
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
|B3|87.78%|75.56%|72.67%|73.35%|12.22pp|23.33%|80.00%|60.00%|40.00%|33/22/19|弱比较器|
|D40-HNBR|85.56%|85.00%|15.33%|25.16%|0.56pp|0%|66.67%|63.33%|0%|2/0/0|新类塌缩|
|D41-BEC|86.11%|20.56%|78.67%|31.50%|65.56pp|0%|76.67%|0%|36.67%|142/0/32|旧类塌缩|
|ProtoNet/Z0|71.11%|48.33%|52.67%|48.97%|22.78pp|0%|33.33%|13.33%|3.33%|0/0/0|基线|
|D61 FP32|90.00%|83.33%|76.00%|78.96%|6.67pp|26.67%|76.67%|60.00%|43.33%|18/15/21|matched参考|
|D61 INT8|90.00%|83.33%|76.00%|78.96%|6.67pp|26.67%|76.67%|60.00%|43.33%|18/16/20|本轮目标，NO-GO|

## 8.分场景表现

|场景|before|after|new|H|forget|joint|min-before|min-after|min-new|混淆|主要表现|
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
|clear|91.67%|86.67%|90.00%|88.15%|5.00pp|60.00%|70.00%|60.00%|70.00%|4/2/3|总体尚稳，但全面低于D46均值|
|low-elev|91.67%|85.00%|70.00%|76.36%|6.67pp|10.00%|90.00%|70.00%|10.00%|5/8/7|旧类改善，新类N0近乎失效|
|rain|86.67%|78.33%|68.00%|72.36%|8.33pp|10.00%|50.00%|30.00%|40.00%|9/6/10|旧/新均弱，新类内部混淆最高|

## 9.逐类与15fold表现

|类别|O0|O1|O2|O3|O4|O5|
|---|---:|---:|---:|---:|---:|---:|
|before|96.67%|93.33%|96.67%|76.67%|93.33%|83.33%|
|after|93.33%|93.33%|96.67%|60.00%|80.00%|76.67%|

|类别|N0|N1|N2|N3|N4|
|---|---:|---:|---:|---:|---:|
|seen-new|43.33%|93.33%|90.00%|90.00%|63.33%|

|场景-fold|before|after|new|H|forget|joint|floor(b/a/n)|混淆(o2n/n2o/n2n)|
|---|---:|---:|---:|---:|---:|---:|---:|---:|
|clear-0|100.00%|83.33%|80.00%|81.63%|16.67pp|50%|100/50/50|2/1/1|
|clear-1|91.67%|83.33%|80.00%|81.63%|8.33pp|50%|50/50/50|0/1/1|
|clear-2|83.33%|75.00%|90.00%|81.82%|8.33pp|50%|50/50/50|2/0/1|
|clear-3|83.33%|91.67%|100.00%|95.65%|−8.33pp|50%|50/50/100|0/0/0|
|clear-4|100.00%|100.00%|100.00%|100.00%|0pp|100%|100/100/100|0/0/0|
|low-0|100.00%|91.67%|70.00%|79.38%|8.33pp|0%|100/50/0|1/1/2|
|low-1|83.33%|75.00%|60.00%|66.67%|8.33pp|0%|50/50/0|2/1/3|
|low-2|83.33%|91.67%|90.00%|90.83%|−8.33pp|50%|50/50/50|0/1/0|
|low-3|100.00%|91.67%|60.00%|72.53%|8.33pp|0%|100/50/0|0/2/2|
|low-4|91.67%|75.00%|70.00%|72.41%|16.67pp|0%|50/50/0|2/3/0|
|rain-0|75.00%|75.00%|60.00%|66.67%|0pp|0%|0/0/0|2/0/4|
|rain-1|100.00%|83.33%|80.00%|81.63%|16.67pp|50%|100/50/50|2/2/0|
|rain-2|91.67%|83.33%|70.00%|76.09%|8.33pp|0%|50/50/0|1/0/3|
|rain-3|83.33%|75.00%|50.00%|60.00%|8.33pp|0%|50/0/0|2/3/2|
|rain-4|83.33%|75.00%|80.00%|77.42%|8.33pp|0%|50/50/0|2/1/1|

## 10.Fisher、量化、训练与资源

- Fisher：before rank恒5、gain范围0.4793–0.9567、均值0.7873；final rank恒10、gain范围0.2446–0.9124、均值0.6108。before/final各15个transform SHA全唯一，condition number均<1.96，identity-primary和原协方差坐标保持30/30通过。
- 编译：FP32相对分数漂移最大`8.44e-7`；D61 INT8与matched FP32的before argmax变化0、final outer argmax变化1、margin sign flip0、support argmax变化0/0；最大score误差`8.74e-4`。量化门因final outer argmax变化1失败。
- 训练：基础D42 adapter仍为20epoch/20step；epoch1平均loss1.0320、support acc95.14%，epoch20 loss0.1027、support acc100%，query rows总和0。D61本身0可训练参数、0额外optimizer step。
- 资源：36个LDA fit、1,065,830,400 LDA MAC；36个Fisher transform fit、稠密代数保守上界6,879,707,136 MAC；总适配估计7,957,035,106 MAC。query6,624MAC、参数2,016、持久态8,583B、registry941B、峰值CUDA22,886,912B；单一int8/FP16仿射state。
- artifact：`full_performance_summary.json`82,916B，SHA256=`3b2a1f4772a1241c89e71279e070fa1441cc45cbfdd5e449740be3fb5d19b1cc`；`D61_PROBE_METADATA.json`1,460B，SHA256=`ba7a3aa6616964a70c76802609d54ac16416feb4c1954de82f5862b862437b49`。
