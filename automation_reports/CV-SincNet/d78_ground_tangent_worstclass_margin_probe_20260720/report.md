# D78地面域切向最差类边界实验报告

## 1.实验身份

|字段|值|
|---|---|
|实验ID|`d78_ground_tangent_worstclass_margin_probe_20260720`|
|候选|`ground_tangent_worstclass_top2_margin`|
|operator|Codex `/root`|
|状态|`PREREGISTERED_NOT_RUN`|
|目标|用地面int8域×类压缩中心形成低秩域切向基，在target support内直接改善最差类top-2边界|
|matched baseline|D62：`B/A/N/H/F/J=92.78/82.22/84.67/82.62/10.56/26.67%`|

## 2.假设与单一主要差异

D77对角预条件只降低连续CE，outer prediction变化为0/15。D78保留地面跨坐标域残差的最多13维联合方向，并把D62 final rows的修正限制在该子空间；优化目标改为class-symmetric smooth worst-class top-2 logistic margin。相对D62仅增加一个直接编译的低秩边界残差，不改数据、基线组件、候选集合或评测协议。

## 3.协议与资格边界

- 复用D18 `VALIDATED_ONCE/p2_min_v1`，receiver`20-1`、seed`713101`、K10/new5、3场景×5fold、actual K8。
- 单LEO_weak observation、support-only、query独立全类argmax；clean/source/query truth/role/quota/global reassignment访问0。
- 地面组件84个cell、逻辑状态25,428B，当前manifest为`formal_phase2_eligible=false`和`UNVERIFIED_UNDER_CURRENT_PROTOCOL`，因此D78只做development diagnostic。

## 4.预注册性能门

|candidate|机制|receiver/TX|K/seed|B|A|N|H|F|J|min-B/A/N|状态/MAC|判定|
|---|---|---|---|---:|---:|---:|---:|---:|---:|---|---|---|
|D78|D62＋ground tangent smooth-worst top-2 residual|20-1/new5|K10/713101|待跑|待跑|待跑|待跑|待跑|待跑|待跑|待审计|待跑|

相对D62要求`A/N/H/min-A/min-N`均不退化、`F`不升且至少一项严格改善，三类混淆不得交换伤害。失败即关闭D78，不扫参数、不运行第二seed、125或N607。

## 5.计划实现、验证与运行

|文件|作用|
|---|---|
|`code/cvsrffi/stage2_d78_ground_tangent_worstclass_margin.py`|地面域切向SVD、8折OOF top-2数据、smooth-worst目标、20步低秩优化|
|`code/scripts/probe_d78_ground_tangent_worstclass_margin.py`|D62 final-row集成、INT8/FP32编译、协议/资源/105行闭包|
|`tests/test_stage2_d78_ground_tangent_worstclass_margin.py`|置换等变、目标单调、top-2 margin、K1回退与确定性|
|`tests/test_probe_d78_ground_tangent_worstclass_margin.py`|公式锁、资源上限、ground只读和协议字段|

`E:\type10-7`根不是Git仓库；上述代码、追溯和本报告进入`E:\type10-7\github_publish\CVS-RFFI-repo`的Git工作流，根报告同步镜像。实现、测试、clean worktree、命令、PID、完整性能与artifact SHA将在运行前后补录。

## 6.本地实现与验证

- core SHA256=`0139e315e0fda570c2f96a572c61de4be68f899074eba197e18d9a856baac49f`；probe SHA256=`2c656afa386495a374103162d330b452b17f4a3748dc7ef71168315e22561669`。
- `ssr-gpu`下core/probe/test `py_compile`通过；专项9/9通过。
- D42-D78邻接47文件390项全部通过，用时83.4秒。pytest退出码为0；结束后的Windows临时目录`pytest-current`清理出现一次`PermissionError`，属于atexit清理噪声，不是测试失败。
- 真实ground组件烟测：26个registry domain中14个完整有效域、84个cell；切向rank13，保留残差能量77.7513%，basis只读；组件formal资格仍为false。

## 7.运行锁

- clean detached worktree：`E:\type10-7\code\snapshots\d78wt`；本地`cuda:0`运行，不同步或启动N607。
- 输出：`E:\type10-7\automation_reports\CV-SincNet\d78_ground_tangent_worstclass_margin_probe_20260720\ground_tangent_worstclass_top2_margin`；stdout/stderr独立保存在报告根。
- 预期：105行、30个target fit、1,080个D62 component execution；每target row8个OOF LDA、88个held行、rank13、20个接受步；query0。
- 精确命令如下；进程参数固定，禁止覆盖已有输出：

```powershell
& 'C:\Users\lh594\.conda\envs\ssr-gpu\python.exe' `
  'E:\type10-7\code\snapshots\d78wt\code\scripts\probe_d78_ground_tangent_worstclass_margin.py' `
  --d78-arm ground_tangent_worstclass_top2_margin `
  --ground-component-dir 'E:\type10-7\automation_reports\CV-SincNet\d22_int8_anchor_lifecycle_20260717\input\int8_component' `
  --ground-manifest-sha256 15b5e144f9af3989421d8e925c17758479c327be47e79222f6363dc63994629c `
  --runtime-root 'E:\type10-7\code\snapshots\d41wt' `
  --probe-root 'E:\type10-7\code\snapshots\d78wt' `
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
  --output 'E:\type10-7\automation_reports\CV-SincNet\d78_ground_tangent_worstclass_margin_probe_20260720\ground_tangent_worstclass_top2_margin' `
  --device auto --mode development_select_unverified_component --candidate-set d42_v1
```
