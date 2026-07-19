# D65冻结Stage2-B Block-LDA追加式注册探针报告

## 1.执行前登记

- ID：`d65_frozen_stage2b_blocklda_append_only_probe_20260719`；操作者：Codex`/root`；状态：`READY_TO_RUN_LOCAL_DEVELOPMENT_CELL`。
- 目标：只在Stage2-B从旧类support学习一次目标域3-block covariance，Stage2-C冻结该几何并以完全相同的`Sigma_B^-1 mu_c`公式追加新类row，验证注册时不重写旧row能否同时保护after-old并保持new。
- D64教训：before92.78%但after74.44%、new77.33%、遗忘18.33pp；pair support100%却held混淆37/16/18，根因是6类到11类扩图重写旧类几何。D65没有pair图或pair权重。
- 当前聚合最强D62：before92.78%、after82.22%、new84.67%、H82.62%、forgetting10.56pp、joint26.67%、min-before80%、min-after53.33%、min-new73.33%、混淆23/8/15；仍不可晋升。
- cell：receiver`20-1`、seed`713101`、K10/new5、3场景×5 outer fold、实际K8；复用匹配`VALIDATED_ONCE/p2_min_v1`的D18 capsule，不重验数据。
- 版本：预注册`26f326ae`；实现`364a56c4`；worktree`E:\type10-7\code\snapshots\d65wt` detached clean；脚本SHA256=`bc0c6e14191e09f773e12e7e9f194e097204c7183d244b1e5d867a339f5e4acb`。
- 验证：py_compile通过；D43＋D65专项15/15通过，含真实D42 K5 FP32/INT8逐bit追加集成测试；D42–D65整链25文件299/299通过，用时85.8s；diff check通过。pytest退出后仅有Windows临时目录清理权限噪声，命令exit0。
- 本轮只在本地确认的`C:\Users\lh594\.conda\envs\ssr-gpu\python.exe`执行，不访问N607，无远端PID/GPU/sync。

## 2.机制、协议与资源预期

Stage2-B用旧类support估计equal-prior auto-shrinkage covariance并保留z160/FFT96/RF32三个对角块；去掉equal-prior公共常数后，每类row为`w_c=Sigma_B^-1 mu_c`、`b_c=-0.5 mu_c^T w_c`。Stage2-C不更新`Sigma_B`，旧类FP32 row、int8两段码、block scale和FP16 intercept必须逐bit不变，只用相同公式追加5个新类row。query仍是单次全注册类affine argmax，不读取注册时序或old/new角色。

每个target row只做1次Stage2-B covariance fit和5个新row求解；预计covariance MAC28,366,848、append MAC429,125，连同基础metric后总适配约33,772,613MAC。query6,624MAC、额外state/MAC0；参数、epoch、step、状态沿D42正式面。没有freeze强度、协方差混合、阈值、角色、场景、receiver或class ID分支。

必须相对D62完整比较总体、三场景、11类、15fold、混淆、量化、训练和资源。旧row任一逐bit不一致直接fail closed；任一主指标、场景、floor或混淆退化则停止D65，不扫描freeze系数或full/block混合。即使本cell通过，也不直接运行125。

## 3.精确执行命令

```powershell
& 'C:\Users\lh594\.conda\envs\ssr-gpu\python.exe' `
  'E:\type10-7\code\snapshots\d65wt\code\scripts\probe_d65_frozen_stage2b_blocklda_append_only.py' `
  --d65-arm frozen_stage2b_blocklda_append_only `
  --runtime-root 'E:\type10-7\code\snapshots\d41wt' --probe-root 'E:\type10-7\code\snapshots\d65wt' `
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
  --output 'E:\type10-7\automation_reports\CV-SincNet\d65_frozen_stage2b_blocklda_append_only_probe_20260719\frozen_stage2b_blocklda_append_only' `
  --device auto --mode development_select_unverified_component --candidate-set d42_v1
```

## 4.待完成证据

运行后必须闭合105/105行、7候选、3场景、11类、15fold、30组before/final冻结协方差、FP32/INT8旧row逐bit不变、query0、量化、训练、资源和artifact，并与D46/D61/D62/D63/D64做同row比较。最终报告必须详细说明每项性能与行为，不能只陈述缺陷。
