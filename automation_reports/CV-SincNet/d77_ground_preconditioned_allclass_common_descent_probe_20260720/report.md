# D77地面预条件全类共同下降实验报告

## 1.实验身份

|字段|值|
|---|---|
|实验ID|`d77_ground_preconditioned_allclass_common_descent_probe_20260720`|
|候选|`ground_preconditioned_allclass_common_descent`|
|operator|Codex `/root`|
|状态|`PREREGISTERED_NOT_RUN`|
|目标|高效利用地面int8域×类原型定义优化几何，以全注册类target-support OOF共同下降直接修正D62最终边界|
|matched baseline|D62：`B/A/N/H/F/J=92.78/82.22/84.67/82.62/10.56/26.67%`|

## 2.假设与单一主要差异

D66证明静态地面可靠性缩放能略微保护旧类，却压低新类与new floor。D77不把地面统计应用到特征后重新拟合，而把它作为target OOF多类梯度的正定预条件器；地面决定坐标可信度，11类target support共同决定方向。相对D62只增加一个直接编译到final rows的地面预条件共同下降residual。

## 3.协议与资格边界

- 复用D18 `VALIDATED_ONCE/p2_min_v1`，receiver`20-1`、seed`713101`、K10/new5、3场景×5fold、actual K8。
- 单LEO_weak observation、support-only、query独立全类argmax；clean/source/query truth/role/quota/global reassignment访问0。
- 当前D19历史地面组件SHA为`3c08c823d2e8a13c4233f0060ac67c332ecc8d6e8abec7352de975fead0267d7`，84个有效cell、逻辑状态25,428B，但manifest仍为`formal_phase2_eligible=false`和`UNVERIFIED_UNDER_CURRENT_PROTOCOL`。因此本轮只能是development diagnostic，不产生formal性能声明。

## 4.开发门与结果占位

|candidate|机制|receiver/TX|K/seed|B|A|N|H|F|J|min-B/A/N|状态/MAC|判定|
|---|---|---|---|---:|---:|---:|---:|---:|---:|---|---|---|
|D77|D62＋ground-M预条件11类OOF共同下降|20-1/new5|K10/713101|待跑|待跑|待跑|待跑|待跑|待跑|待跑|待审计|待跑|

相对D62要求`A/N/H/min-A/min-N`均不退化、`F`不升且至少一项严格改善，三类混淆不得交换伤害。失败即关闭D77，不扫参数、不运行第二seed或125。

## 5.版本与运行占位

`E:\type10-7`根目录不是Git仓库；代码、追溯和Git版报告进入`E:\type10-7\github_publish\CVS-RFFI-repo`，根报告镜像到本目录。实现后补录commit、clean worktree、测试、运行命令、PID/GPU、完整105行、逐场景/逐类/15fold/混淆/量化/资源表和最终判定。

## 6.本地实现与验证

|文件|作用|
|---|---|
|`code/cvsrffi/stage2_d77_ground_preconditioned_common_descent.py`|地面正定预条件器、8折OOF类梯度、20步M-Frank-Wolfe、解析步长与trust cap|
|`code/scripts/probe_d77_ground_preconditioned_allclass_common_descent.py`|D62 final-row集成、INT8/FP32编译、协议/资源/105行闭包|
|`tests/test_stage2_d77_ground_preconditioned_common_descent.py`|确定性、逐类CE安全、类置换等变、K1回退|
|`tests/test_probe_d77_ground_preconditioned_allclass_common_descent.py`|公式、固定20步、MAC加总、34,011B状态和协议字段|

- `ssr-gpu`下core/probe `py_compile`通过；专项9/9通过。
- D42–D77相邻47文件、424项全部通过，用时85.4秒。
- D25旧测试的2个源码字符串断言在D76未修改干净worktree同样失败；D77未改D25 runner，属于既有基线漂移。

## 7.运行锁

- clean worktree：`E:\type10-7\code\snapshots\d77wt`，commit`0831101802b5590a848fc62ca3b569629272698d`。
- core SHA256：`cb771f843d83b6fb11c1d373183421cd400d33a1636d4fc05d5be4fec69f603e`；probe SHA256：`198ecbd65bb91a83571bea123d1a5d28377ad5c779deaba050a56cd3ce7a51a3`。
- 环境：`C:\Users\lh594\.conda\envs\ssr-gpu\python.exe`，`--device auto`；本地运行，不同步或启动N607。
- 输出：`E:\type10-7\automation_reports\CV-SincNet\d77_ground_preconditioned_allclass_common_descent_probe_20260720\ground_preconditioned_allclass_common_descent`；stdout/stderr位于实验报告根。
- 预期：105行、30个target row、30次top fit、1,080次D62 component execution；每target row8个OOF LDA、88个held行、11个类梯度、20步Frank-Wolfe；query0。

```powershell
& 'C:\Users\lh594\.conda\envs\ssr-gpu\python.exe' `
  'E:\type10-7\code\snapshots\d77wt\code\scripts\probe_d77_ground_preconditioned_allclass_common_descent.py' `
  --d77-arm ground_preconditioned_allclass_common_descent `
  --ground-component-dir 'E:\type10-7\automation_reports\CV-SincNet\d22_int8_anchor_lifecycle_20260717\input\int8_component' `
  --ground-manifest-sha256 15b5e144f9af3989421d8e925c17758479c327be47e79222f6363dc63994629c `
  --runtime-root 'E:\type10-7\code\snapshots\d41wt' --probe-root 'E:\type10-7\code\snapshots\d77wt' `
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
  --output 'E:\type10-7\automation_reports\CV-SincNet\d77_ground_preconditioned_allclass_common_descent_probe_20260720\ground_preconditioned_allclass_common_descent' `
  --device auto --mode development_select_unverified_component --candidate-set d42_v1
```
