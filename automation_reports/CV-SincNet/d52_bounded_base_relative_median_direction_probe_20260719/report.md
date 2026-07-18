# D52有界base-relative median方向开发报告

## 1.状态

- run ID：`d52_bounded_base_relative_median_direction_probe_20260719`
- operator：Codex
- 状态：`IMPLEMENTED_AND_TESTED_PRE_RUN`
- 范围：本地receiver20-1、seed713101、K10/new5开发单元；不访问N607、不运行125。
- 当前最强合法开发点：D46，但不promotable。

## 2.目标、假设与比较对象

D51证明coordinate-median相对mean的稳健方向能改善rain old，但全局小RMS把部分修正放大到2.51，造成low-elev/new交换伤害。D52只检验三轮回顾预注册的假设：方向有效，尺度失败。唯一公式为：

```text
u_c = coordinate_median_r(x_rc) - mean_r(x_rc)
v_c = u_c / max(||u_c||_2, eps)
gamma_c = 1 - ||mean_r(x_rc / ||x_rc||_2)||_2
s_c = ||W_D45,c - mean_j(W_D45,j)||_2
DeltaW_c = gamma_c * s_c * v_c
W_D52,c = W_D45,c + DeltaW_c
b_D52,c = b_D45,c
```

直接比较D45、D46、D51；不得仅按rain单项或独立极值晋级。

## 3.协议与数据锁

- `protocol_schema=p2_min_v1`，复用已`VALIDATED_ONCE`胶囊；方法变化不触发数据重验证。
- 固定`leo_clear_weak/leo_low_elev_weak/leo_rain_weak`×5 outer folds；标称K10，实际每outer fit K8；new5。
- 只用support；query及其视图test-only，禁止truth/role/count/quota/global reassignment/query-dependent optimization。
- 禁止clean/source、dense query graph、class-ID/场景/receiver/handle分支。
- before/final使用同一公式；K1/K2精确D45 fallback；无alpha、threshold、clip或scan。

## 4.本地文件和验证

|文件|用途|
|---|---|
|`code/scripts/probe_d52_bounded_base_relative_median_direction.py`|探针、closure verifier、资源账|
|`tests/test_probe_d52_bounded_base_relative_median_direction.py`|公式、边界、对称性、资源测试|
|`analysis/d52_bounded_base_relative_median_direction_traceability_20260719.md`|设计–验证追踪|
|本报告|运行锁与完整性能账本|

验证结果：`py_compile`通过；D52定向10/10通过；D45–D52联合116/116通过；D40–D52执行闭包256/256通过。pytest退出后的临时目录清理出现一次`PermissionError`提示，但主进程exit0，不影响项目断言。更宽的历史测试面另有3个与D52无关的既有断言漂移：1项候选列表仍只锁到D35，2项仍要求D25 schema literal位于`run()`函数体；当前runner已经扩展到D42且schema构造位置已变化，因此不为D52改写。执行前还需clean worktree、输入hash与输出不存在检查。

## 5.预注册成功/停止标准

|指标|最低要求|
|---|---|
|预测实质性|相对D45至少1/15 outer预测SHA变化|
|联合表现|总体和各场景after/new/H/joint及逐类floor不得出现不可解释退化|
|对当前最强D46|至少保持new84.67%、min-new73.33%，同时改善old侧或遗忘|
|rain修复|after至少78.33%，forget不高于10pp，且不能以low-elev/new伤害换取|
|协议与量化|query/role/quota/count/global/clean/source为0/false；FP32/int8 argmax和margin翻转为0|

任一联合门失败即`COMPLETED_DIAGNOSTIC_NEGATIVE_NOT_PROMOTABLE`：不扫尺度、不clip、不加门控、不跑第二seed、不formalize、不运行125。

## 6.计划运行和产物

- 环境：`C:\Users\lh594\.conda\envs\ssr-gpu\python.exe`；本地`device=auto`；串行单进程。
- runtime root：`E:\type10-7\code\snapshots\d41wt`（只读bootstrap）。
- probe root：待实现提交后创建`E:\type10-7\code\snapshots\d52wt`clean detached worktree。
- output：`E:\type10-7\automation_reports\CV-SincNet\d52_bounded_base_relative_median_direction_probe_20260719\bounded_base_relative_median_direction`。
- 预期：`training_log.jsonl`、`selection.json`、`support_audit.json`、`geometry_audit.json`、`resource_audit.json`、`RECEIPT.json`、`D52_PROBE_METADATA.json`，完成后追加`full_performance_summary.json`。
- 风险：bounded scale可能过小而不改变决策；base范数可能仍把某些类推过边界；raw coordinate median方向不具旋转等变性。三者均须用完整同row指标判断。

## 7.性能报告承诺

实验完成后，本报告必须补充：7候选总体表、3场景表、old/new逐类表、15个outer行、相对D45/D46/D51的同row差值与预测变化、混淆、完整20epoch训练轨迹摘要、几何修正范数、int8/FP32误差、资源与全部artifact SHA。不得只写缺陷或只报单项最好值。

## 8.执行锁与exact command

- Git承载仓库分支`codex/cvs-rffi-release-20260626`；实现提交`422dfbd9`。根目录`E:\type10-7`不是Git仓库，本报告同步保留Git版与根目录镜像。
- clean detached worktree：`E:\type10-7\code\snapshots\d52wt`，状态`## HEAD (no branch)`；探针SHA256`32a37c732ce54bc236b999defb755f2f1c466d04a8c807241db0d169cff7846e`。
- clean worktree内`py_compile`和D52定向10/10再次通过。
- before/after seal SHA为`53ace286…d9f75`/`c70aedf3…b50ff`；签名授权envelope为`31a2ad99…ceb0e`/`a2483d6e…be76`；int8 manifest为`15b5e144…629c`；class binding为`bb89a1db…c901f`。全部与既有锁一致。
- runtime root存在；输出目录启动前不存在。本地串行`device=auto`，无N607连接。

```powershell
& 'C:\Users\lh594\.conda\envs\ssr-gpu\python.exe' `
  'E:\type10-7\code\snapshots\d52wt\code\scripts\probe_d52_bounded_base_relative_median_direction.py' `
  --d52-arm bounded_base_relative_median_direction `
  --runtime-root 'E:\type10-7\code\snapshots\d41wt' `
  --probe-root 'E:\type10-7\code\snapshots\d52wt' `
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
  --output 'E:\type10-7\automation_reports\CV-SincNet\d52_bounded_base_relative_median_direction_probe_20260719\bounded_base_relative_median_direction' `
  --device auto --mode development_select_unverified_component --candidate-set d42_v1
```
