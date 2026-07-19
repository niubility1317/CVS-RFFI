# D70交叉拟合原子生命周期行替换探针

## 1.执行前登记

- 实验ID：`d70_crossfitted_atomic_lifecycle_row_replacement_probe_20260719`；operator：Codex；状态：`PREREGISTERED_IMPLEMENTATION_VALIDATED_PERFORMANCE_PENDING`。
- 当前联合最强D62：B/A/N/H/F/J=92.78/82.22/84.67/82.62/10.56/26.67，min-B/A/N=80.00/53.33/73.33，混淆23/8/15。
- D69完整结果为92.78/81.67/74.67/77.39/11.11/30.00，min-N53.33%、混淆27/23/15；全旧行冻结跨坐标系交换已否决。D67–D69正式复盘见D69报告第15节，提交`6c5f924e`。
- 根目录`E:\type10-7`非Git；本报告镜像、代码、测试和追踪进入`E:\type10-7\github_publish\CVS-RFFI-repo`。其他工作树改动与D70无关，只暂存D70拥有路径。

## 2.方法锁

K>=2时使用两个按physical rank预定的互斥support-held fold。每折在train部分分别拟合D62 before-old和D62 final-joint；held全部类上，以final-joint为base，逐个旧行测试before行替换。单行要求本类TP不降、FP不增且至少一项严格改善；全部初选行联合替换后，必须对11类逐类TP不降且FP不增，否则mask全清零。full support只按mask在D62 final head中替换旧行，新行始终为final joint行。K1精确D62 fallback。

没有连续权重、center/scale、符号、温度、offset、class名单、scene/receiver或query角色分支。所有候选旧行使用同一计数公式；最终是一个全注册类affine head。

## 3.目标、停止条件与完整报告

- before、空mask fallback必须精确D62；两折partition exact-once，gate联合TP/FP原子安全，旧/新类评价同等。
- 相对D62必须无A/N/H/J/min-A/min-N/场景floor交换，并至少改善A/F/J/floor之一；否则`COMPLETED_DIAGNOSTIC_NEGATIVE_NOT_PROMOTABLE`。
- INT8相对FP32的argmax变化和margin sign flip为0；资源、query独立性和状态上限通过。
- 真实105行后完整报告7候选、3场景、11类、15fold、mask/TP/FP、训练、量化、资源、artifact及D62/D65/D66/D67/D68/D69对照。失败不跑第二seed/125。

## 4.数据与协议

固定receiver`20-1`、seed`713101`、K10/new5、三场景×五outer fold、实际K8；复用D18`VALIDATED_ONCE/p2_min_v1`enrollment-only capsule，不重验数据。query只测试，no clean/source、query truth/role/quota/global assignment、class-ID规则或dense query graph。ground输入锁0：D22仍`formal_phase2_eligible=false`，D66的84-cell接入为负交换。

## 5.实施计划

新增独立D70 partition/gate/lifecycle core、probe和专项测试，不修改D62/D69历史实现。先做合成partition、原子gate、置换、空mask精确fallback、K1、旧行选择、新行恒定、compiled state、禁止访问和资源闭包测试；再跑D42–D70完整链、提交、干净worktree复跑，最后才登记真实105行命令。

## 6.实现与本地验证

- `code/cvsrffi/stage2_d70_atomic_lifecycle.py`：两折rank partition、TP/FP计数、coordinate gate、all-class atomic gate和Stage2-B/Stage2-C配对生命周期。
- `code/scripts/probe_d70_crossfitted_atomic_lifecycle_row_replacement.py`：复用锁定D62与D42 runner，记录60次top-level fit、30对生命周期、120次inner D62和2280条component fit；单独计入inner LDA/Fisher/held-score/gate MAC。
- 两个测试文件共10项，覆盖partition exact-once、原子安全、置换等变、K1精确D62、选择性旧行、新行joint不变、support漂移拒绝、source closure和禁止分支。
- 专项10/10通过；D42–D70完整链345/345通过，用时81.5s，34个测试文件，包含D42 integration20项。
- 主工作树source SHA：core`f2e67c142ba8fbe797a019e724435a86b67db8446efb9ba49c96abb593b47459`；probe`ff74748be440648ade9c45c60d12c53ea71e149d74180b30a4c1570a257072c2`。

当前只有代码/合成验证，不能声明性能。下一步提交精确文件，建立干净worktree复跑345项；干净链通过后才登记真实105行命令。

## 7.干净版本与真实运行锁

- 实现提交`10536c01`；干净worktree`E:\type10-7\code\snapshots\d70wt`为detached HEAD且`git status -sb`仅`## HEAD (no branch)`。
- 干净D42–D70完整链345/345通过，用时82.8s；运行目录`E:\type10-7\local_artifacts\d70_clean_full_chain_345`。
- 实际checkout SHA：probe`1024ce5bcc4abed430a19acda811bb0fedca422ba76206acb60984e819d87ecc`、core`94d50258db904a5e289fefe8300966435895c171961892cbb07490c4e2027003`、D62 helper`38ae1114a06d135bca806f470417cd28a634fec0da449888665c6843615d4a20`。
- 本地运行，不用N607。输出目录登记时不存在，禁止覆盖或原目录重跑。

```powershell
& 'C:\Users\lh594\.conda\envs\ssr-gpu\python.exe' `
  'E:\type10-7\code\snapshots\d70wt\code\scripts\probe_d70_crossfitted_atomic_lifecycle_row_replacement.py' `
  --d70-arm crossfitted_atomic_lifecycle_row_replacement `
  --runtime-root 'E:\type10-7\code\snapshots\d41wt' --probe-root 'E:\type10-7\code\snapshots\d70wt' `
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
  --output 'E:\type10-7\automation_reports\CV-SincNet\d70_crossfitted_atomic_lifecycle_row_replacement_probe_20260719\crossfitted_atomic_lifecycle_row_replacement' `
  --device auto --mode development_select_unverified_component --candidate-set d42_v1
```

预期闭包：105行、30个目标row、60个D70 fit audit、30对生命周期、120个inner D62 fit、2280条component fit。每个final audit两折覆盖88个support row exact-once；active mask必须all-class atomic safe，空mask精确D62；ground/query/clean/source/role/quota访问0。
