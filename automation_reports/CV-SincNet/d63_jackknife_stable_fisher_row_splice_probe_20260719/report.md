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

运行后必须写入105行闭包、7候选、3场景、11类、15fold、总体候选行、jackknife剔除行、联合atomic fallback、D46/D61/D62/D63比较、量化、训练、资源和artifact；当前不作D63性能声明。
