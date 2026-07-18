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
