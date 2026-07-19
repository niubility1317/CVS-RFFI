# D64全pair局部Block-LDA连续锦标赛探针报告

## 1.执行前登记

- ID：`d64_allpairs_blocklda_tournament_probe_20260719`；操作者：Codex`/root`；状态：`READY_TO_RUN_LOCAL_DEVELOPMENT_CELL`。
- 目标：以全部匿名类别pair的局部3-block二类LDA连续margin替代D46/D62的全局共享协方差与行筛选，让O3、N0、N2等局部冲突边界分别获得support-only判别方向，同时保持单次全类affine query。
- 当前聚合最强D62：before92.78%、after82.22%、new84.67%、H82.62%、forgetting10.56pp、joint26.67%、min-before80%、min-after53.33%、min-new73.33%、混淆23/8/15；它仍有low遗忘恶化和rain-before下降，状态为不可晋升。
- cell：receiver`20-1`、seed`713101`、K10/new5、3场景×5 outer fold、实际K8；复用匹配`VALIDATED_ONCE/p2_min_v1`的D18 capsule，不重验数据。
- 版本：预注册`7ba296b9`；实现`4a598539`；worktree`E:\type10-7\code\snapshots\d64wt` detached clean；脚本SHA256=`b369347c7307fe64b4b4eee133dfbee7a4bf0bbba3281c898ddd56f7c22ee9e6`。
- 本地验证：py_compile通过；D43＋D64专项测试14/14通过；D42–D64整链回归247/247通过，用时97.8s；diff check通过。
- 本轮只在本地确认的`C:\Users\lh594\.conda\envs\ssr-gpu\python.exe`执行，不访问N607，无远端PID/GPU/sync。

## 2.机制、协议与资源预期

对每个匿名类别pair `(c,d)`，D64仅使用该pair的support，在固定3-block centered特征上拟合equal-prior、auto-shrinkage二类LDA。将有向margin除以pair-support margin RMS，再对每个类别关联的全部pair连续margin取平均；最终中心化共同仿射项，并编译为一个`C×288` FP32 affine后进入既有target-old/new统一的残差int8系数＋FP16截距生命周期。

每个target row在before阶段有6类、15个pair，final阶段有11类、55个pair，共70次pair fit；30条目标row预期2100次pair fit。query不保留pair图、不做投票、阈值、图推理、batch优化或query-dependent适配，仍只执行一次全注册类affine评分，query额外MAC/state为0。方法没有角色、场景、receiver、class ID、query真值、真实batch类数、quota或global reassignment分支。

D64必须与D62/D46比较总体、三场景、11类、15fold、混淆、量化、训练和资源。只有before/after/new/H/forgetting、floor/joint以及混淆形成无交换改善时才继续第二开发seed；任一场景、new/H或floor显著受损则停止D64，不扫描pair阈值、投票权重或full/block变体。即使本cell通过，也不直接运行125。

## 3.精确执行命令

```powershell
& 'C:\Users\lh594\.conda\envs\ssr-gpu\python.exe' `
  'E:\type10-7\code\snapshots\d64wt\code\scripts\probe_d64_allpairs_blocklda_tournament.py' `
  --d64-arm allpairs_blocklda_tournament `
  --runtime-root 'E:\type10-7\code\snapshots\d41wt' --probe-root 'E:\type10-7\code\snapshots\d64wt' `
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
  --output 'E:\type10-7\automation_reports\CV-SincNet\d64_allpairs_blocklda_tournament_probe_20260719\allpairs_blocklda_tournament' `
  --device auto --mode development_select_unverified_component --candidate-set d42_v1
```

## 4.待完成证据

运行后必须闭合105/105行、7候选、3场景、11类、15fold、2100次pair fit、query0、FP32/INT8等价性、训练过程、适配/query资源、artifact哈希，并与D46/D61/D62/D63做同row比较。最终报告需详细说明每项性能与行为，不能只陈述缺陷。
