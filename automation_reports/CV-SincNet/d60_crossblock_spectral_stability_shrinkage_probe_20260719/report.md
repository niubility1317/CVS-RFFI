# D60跨块谱稳定性收缩探针报告

## 1.执行前登记

- ID：`d60_crossblock_spectral_stability_shrinkage_probe_20260719`；操作者：Codex`/root`；状态：`PREREGISTERED_NOT_YET_EXECUTED`。
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
