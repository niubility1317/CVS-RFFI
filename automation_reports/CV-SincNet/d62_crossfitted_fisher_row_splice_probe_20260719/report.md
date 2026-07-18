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

运行后必须写入105行闭包、7候选、3场景、11类、15fold、接受行/atomic fallback、D46/D61/D62比较、量化、训练、资源和artifact；当前不作性能声明。
