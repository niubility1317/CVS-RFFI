# D59 full/block SPD几何中点探针报告

## 1.身份与执行前状态

- 实验ID：`d59_full_block_spd_geodesic_midpoint_probe_20260719`
- 操作者：Codex`/root`
- 时间：2026-07-19（Asia/Hong_Kong）
- 状态：`PREREGISTERED_NOT_YET_EXECUTED`
- 目标：检验full auto-shrinkage与z160/FFT96/RF32三块协方差的SPD仿射不变几何中点，能否在保持所有类别共享logit尺度的前提下，兼顾D43 full的新类/rain保护和3-block的聚合/旧类收益。
- 比较目标：当前最强合法development点D46，before92.22%、after81.67%、seen-new84.67%、同rowH82.33%、forgetting10.56pp、joint23.33%、min-before80.00%、min-after53.33%、min-new73.33%、混淆25/8/15。
- development cell：receiver`20-1`、seed`713101`、K10/new5、3场景×5个physical-rank held折；实际outer fit K8。
- 数据：复用匹配`VALIDATED_ONCE/p2_min_v1`的D18 capsule，不重建、不重验。
- N607：本轮不访问；使用本地CUDA Runner取得development support-held证据。

## 2.假设与机制锁

令`F`为完整等先验auto-shrinkage协方差，`B=blockdiag(F)`。唯一候选为：

`G=B^(1/2)·(B^(-1/2)·F·B^(-1/2))^(1/2)·B^(1/2)`。

所有注册类共享同一个`G`；最终仍是等先验LDA线性score，只删除类公共仿射项。无按类斜率、按类截距、full/block score权重、geodesic位置、ridge、floor、threshold或任何扫描。K1/rank0/零残差精确回退D42单位协方差。query仍是一套int8 residual coefficient＋FP16 intercept state上的全registry独立argmax。

## 3.版本、文件与验证

|项目|证据|
|---|---|
|预注册提交|`acd70450`|
|实现提交|`bb3be85d`|
|执行worktree|`E:\type10-7\code\snapshots\d59wt`，detached clean|
|执行HEAD|`bb3be85d4ca1f5d9da4089aab7703a6beb311655`|
|执行脚本SHA256|`d9b8a94d43f8cd3887f5f22ee653add68ee8acf3e1ca1d1a96dc667b94836a9e`|
|实现|`code/scripts/probe_d59_full_block_spd_geodesic_midpoint.py`|
|单测|`tests/test_probe_d59_full_block_spd_geodesic_midpoint.py`|
|追溯|`analysis/d59_full_block_spd_geodesic_midpoint_traceability_20260719.md`|
|验证|`py_compile`通过；D42–D46＋D59定向回归104/104通过；`git diff --check`通过|

`E:\type10-7`根目录不是Git仓库；正式代码、测试、追溯和本报告进入`github_publish/CVS-RFFI-repo`，根目录只保留报告镜像。未修改N607文件，无sync destination/PID/GPU allocation。

## 4.预注册性能门

D59相对D46必须同时满足：105/105行和query0；协议、source、ground、lifecycle、state、resource、artifact闭包；量化before/final argmax变化与margin翻转0/0/0；聚合before/after/new/H与三类floor/joint不退化、forgetting不增加且至少一个final floor严格提高；三场景联合不退化；混淆不超过25/8/15；15fold至少一个outer prediction变化。即使全部通过也只允许进入另行正式候选验证，不直接运行125。

## 5.精确执行命令

```powershell
& 'C:\Users\lh594\.conda\envs\ssr-gpu\python.exe' `
  'E:\type10-7\code\snapshots\d59wt\code\scripts\probe_d59_full_block_spd_geodesic_midpoint.py' `
  --d59-arm full_block_spd_geodesic_midpoint `
  --runtime-root 'E:\type10-7\code\snapshots\d41wt' `
  --probe-root 'E:\type10-7\code\snapshots\d59wt' `
  --before-root 'E:\type10-7\automation_reports\CV-SincNet\d18_formal_k10_new5_rx20_1_seed713101_20260717_085303\phase2_capsule_k10_new5\predictor\before\enrollment_only' `
  --before-seal 'E:\type10-7\automation_reports\CV-SincNet\d18_formal_k10_new5_rx20_1_seed713101_20260717_085303\phase2_capsule_k10_new5\seals\before_enrollment.seal.json' `
  --before-seal-sha256 53ace2863c9da6c2f6cc855d602c99f581df6de3d30a9a3ecb89eb6b6f0d9f75 `
  --before-formal-policy 'E:\type10-7\automation_reports\CV-SincNet\d18_formal_k10_new5_rx20_1_seed713101_20260717_085303\formal_execution_policy.json' `
  --before-formal-policy-authorization 'E:\type10-7\automation_reports\CV-SincNet\d18_formal_k10_new5_rx20_1_seed713101_20260717_085303\runtime_authorization_k10_new5\before_formal_policy_authorization.v2.json' `
  --before-signed-policy-authorization-envelope 'E:\type10-7\automation_reports\CV-SincNet\d18_formal_k10_new5_rx20_1_seed713101_20260717_085303\runtime_authorization_k10_new5\before_signed_policy_authorization_envelope.v2.json' `
  --before-signed-policy-authorization-envelope-sha256 31a2ad9918f061b25d5a7ed0cc135df70ae02460c094b2f396bf314817bceb0e `
  --after-root 'E:\type10-7\automation_reports\CV-SincNet\d18_formal_k10_new5_rx20_1_seed713101_20260717_085303\phase2_capsule_k10_new5\predictor\after\enrollment_only' `
  --after-seal 'E:\type10-7\automation_reports\CV-SincNet\d18_formal_k10_new5_rx20_1_seed713101_20260717_085303\phase2_capsule_k10_new5\seals\after_enrollment.seal.json' `
  --after-seal-sha256 c70aedf3a8f059e756806201758c1933a2f3e1ba4df415e69a1c776b1a2b50ff `
  --after-formal-policy 'E:\type10-7\automation_reports\CV-SincNet\d18_formal_k10_new5_rx20_1_seed713101_20260717_085303\formal_execution_policy.json' `
  --after-formal-policy-authorization 'E:\type10-7\automation_reports\CV-SincNet\d18_formal_k10_new5_rx20_1_seed713101_20260717_085303\runtime_authorization_k10_new5\after_formal_policy_authorization.v2.json' `
  --after-signed-policy-authorization-envelope 'E:\type10-7\automation_reports\CV-SincNet\d18_formal_k10_new5_rx20_1_seed713101_20260717_085303\runtime_authorization_k10_new5\after_signed_policy_authorization_envelope.v2.json' `
  --after-signed-policy-authorization-envelope-sha256 a2483d6e9c9c362d89397029ff1e43f48358be3bdb3a05d717ee112b70a0be76 `
  --component-dir 'E:\type10-7\automation_reports\CV-SincNet\d22_int8_anchor_lifecycle_20260717\input\int8_component' `
  --component-manifest-sha256 15b5e144f9af3989421d8e925c17758479c327be47e79222f6363dc63994629c `
  --class-binding 'E:\type10-7\github_publish\CVS-RFFI-repo\analysis\d19_adv3b02_class_binding_20260717.json' `
  --class-binding-sha256 bb89a1dbb831acb374fccfc596ae98b660b496b449bdca577dabb962121c901f `
  --output 'E:\type10-7\automation_reports\CV-SincNet\d59_full_block_spd_geodesic_midpoint_probe_20260719\full_block_spd_geodesic_midpoint' `
  --device auto --mode development_select_unverified_component --candidate-set d42_v1
```

## 6.预期输出与完成后检查

- 输出目录：本报告目录下`full_block_spd_geodesic_midpoint`。
- 预期artifact：`training_log.jsonl`、`support_audit.json`、`selection.json`、`RECEIPT.json`、`D59_PROBE_METADATA.json`、`geometry_audit.json`、`resource_audit.json`。
- 完成后必须补充：启动/完成状态、elapsed、全部7候选、3场景、11类、15fold、混淆、量化、20epoch trace、SPD几何、资源、artifact SHA、相对D46门判定和下一实验。
- 风险：SPD中点计算可能增加host FP64时间；若数值闭包失败应fail closed，不得降级为隐式ridge或修改端点。若性能失败，不扫描geodesic位置。
