# D74类中心正交nuisance方向删除实验报告

## 1.实验身份与状态

|字段|值|
|---|---|
|实验ID|`d74_orthogonal_nuisance_direction_removal_probe_20260720`|
|候选|`orthogonal_nuisance_direction_removal`|
|operator|Codex `/root`|
|状态|`PREREGISTERED_NOT_RUN`|
|目标|删除一个不承载类中心差异、但具有最大类内残差能量的非可逆方向，检验能否突破D62/D73等价边界|
|比较目标|D62：`B/A/N/H/F/J=92.78/82.22/84.67/82.62/10.56/26.67%`|

## 2.协议与机制锁

- `p2_min_v1`、D18匹配`VALIDATED_ONCE` capsule、receiver`20-1`、seed`713101`、K10/new5、3场景×5fold，outer-fit实际K8。
- 单LEO_weak观测、support-only、query逐样本全注册类argmax；无clean/source/query truth/role/quota/global assignment。
- before精确D62；final在D42特征中删除一个与中心化类均值span正交的最大类内残差方向，D62 refit后把投影编译进单一int8头。
- rank固定1，无阈值、强度、场景、类、角色或结果扫描；地面组件输入0。

## 3.开发门

相对D62要求`A/N/H/min-A/min-N`不退化且至少一项严格提高，同时`B/F`、场景和混淆无交换伤害。失败即负向关闭，不开第二seed或125矩阵。

## 4.版本、验证、运行和结果占位

`E:\type10-7`不是Git仓库；所有代码、测试、追溯和完整报告进入`E:\type10-7\github_publish\CVS-RFFI-repo`，根目录只保留同步镜像。实现后补录commit、clean worktree、source SHA、完整命令、PID/GPU/log/output、105行闭包、7候选、3场景、11类、15fold、机制、训练、量化、资源、artifact、缺陷和最终判定。

|candidate|机制|receiver/TX|K/seed|B|A|N|H|F|J|min-B/A/N|混淆|资源|判定|
|---|---|---|---|---:|---:|---:|---:|---:|---:|---|---|---|---|
|D74|rank-1非可逆nuisance删除＋D62 refit|20-1/new5|K10/713101|待跑|待跑|待跑|待跑|待跑|待跑|待跑|待跑|待跑|待跑|

## 5.实现锁定

|文件|用途|SHA256|
|---|---|---|
|`code/cvsrffi/stage2_d74_orthogonal_nuisance_removal.py`|中心span、类内残差SVD、rank-1投影和不变量审计|`6584e14a918b2217e96093feb2ffefbf60009257d16674913588931b8e455444`|
|`code/scripts/probe_d74_orthogonal_nuisance_direction_removal.py`|D62包装、`W(I−uuT)`编译、资源/source/闭包|`3661618f848f94d29c3a188d68b6eba8de22ca0ad014cb55ceb9502db81ed375`|
|`tests/test_stage2_d74_orthogonal_nuisance_removal.py`|非可逆、中心保护、置换等变、K1/fail-closed|`b292166f4278d683251e0e5f0a7ef18158867b76b4943c5703b4062ea10f5e5d`|
|`tests/test_probe_d74_orthogonal_nuisance_direction_removal.py`|D62继承、资源公式、调用和协议闭包|`a4053995f901adb0a35ab61ea35fbe63b9a69798cc137e851cbf2667170187e5`|

`ssr-gpu`专项测试8/8通过。D74不增加optimizer step/epoch；每个K8 final fit新增36个closed-form component fit，投影方向编译后不持久化，query额外MAC/state0。

## 6.完整验证与运行锁

- 实现commit=`eb22322c9e2e6d24817cbcee0ba0778e5d424df2`；clean worktree=`E:\type10-7\code\snapshots\d74wt`，detached HEAD且clean。
- 主工作树与clean worktree的D42–D74相邻42文件、385项测试均通过，用时82.7/82.9秒；core/probe `py_compile`通过。
- clean执行SHA：probe=`e65db3025fc9bd834ff530544b23f9d5b8a935e5567a8b5675b20533f7056fe4`、core=`2f098c8c3311ce0da9a62ace354c3c005d68da1161a82a265e70976d221e0f2f`、D62 helper=`38ae1114a06d135bca806f470417cd28a634fec0da449888665c6843615d4a20`。
- 01:22:35启动前输出目录不存在；GPU0 RTX5070Ti显存`954/16303MiB`、利用率0%。本轮本地执行，不访问N607。
- 预期闭包：105行、30目标行、30 top fit、30额外D62 final refit、1620 component execution、30份rank-1投影audit、ground/query-fit/clean/source/role/quota访问0。

```powershell
& 'C:\Users\lh594\.conda\envs\ssr-gpu\python.exe' `
  'E:\type10-7\code\snapshots\d74wt\code\scripts\probe_d74_orthogonal_nuisance_direction_removal.py' `
  --d74-arm orthogonal_nuisance_direction_removal `
  --runtime-root 'E:\type10-7\code\snapshots\d41wt' --probe-root 'E:\type10-7\code\snapshots\d74wt' `
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
  --output 'E:\type10-7\automation_reports\CV-SincNet\d74_orthogonal_nuisance_direction_removal_probe_20260720\orthogonal_nuisance_direction_removal' `
  --device auto --mode development_select_unverified_component --candidate-set d42_v1
```

## 7.启动与监控

- 2026-07-20 01:24:12启动唯一执行，PID`23556`；只读命令行与锁定参数一致，stderr 0B。
- 当前只读离散监控，不重复启动；进程退出后验证105行、projection audit、RECEIPT和metadata。
