# D72物理rank留一联合头bagging开发报告

## 1.实验登记

- 实验ID：`d72_physical_rank_leave_one_head_bagging_probe_20260719`；operator：Codex；状态：`READY_TO_RUN_LOCAL_DEVELOPMENT_CELL`。
- 当前最强D62：B/A/N/H/F/J=`92.78/82.22/84.67/82.62/10.56/26.67`，min-B/A/N=`80.00/53.33/73.33`，混淆old→new/new→old/new→new=`23/8/15`。
- 目标：保持D62旧域metric和全注册类统一评分语义，只降低K-shot联合LDA与D62行选择对单个physical-rank的方差；同时提高注册后旧类、新类、H、joint或通用floor，不得用注册前下降伪造低遗忘。
- development cell固定receiver`20-1`、seed`713101`、K10/new5、3场景×5outer fold；D18 capsule实际每类K8。复用`VALIDATED_ONCE/p2_min_v1` enrollment-only数据，不重新验证未变化数据。

## 2.唯一机制与公式

先按D62完整流程得到旧类metric `log_diag`，Stage2-C期间保持冻结。对before-old或final all-registered support，在固定变换 `z=x*exp(log_diag)` 下，按每类support内部物理rank顺序构造K个leave-one子集。第r个子集对每个匿名类恰好删除rank r的一条物理样本，只用剩余K-1条调用完整D62联合仿射头：

```
(W_r,b_r)=D62(z_{rank!=r},y)
W_bag=(1/K) sum_r W_r
b_bag=(1/K) sum_r b_r
```

`(W_bag,b_bag)`统一做类公共仿射中心化后，分别编译为D42两级residual-int8/FP16正式状态和matched FP32状态。query仍只执行一次`all-registered argmax`，不保留K个sidecar头，不增加query MAC或dense graph。K≤2精确回退D62。

这不是D63的jackknife稳定门：D72不按类选择、不拼接行、不看TP/FP门，只平均完整匿名联合头。这也不是D67：没有D65专家、连续alpha、score标准化或角色生命周期融合；也不是D50–D54的median prototype/score残差。

## 3.协议与地面组件边界

- 每个support物理样本仍只有一个固定LEO_weak观测；leave-one只是训练子集重用，不生成新物理样本或新view。
- 每个inner fit的held rank与train rank交集为0；K个held分区覆盖每条support恰好一次。outer-held/query不参与fit、选择、平均或中心化。
- before只读取target-old support；final读取全部已注册target-old/new support，所有类别使用完全相同公式。无class ID、old/new role、scene、receiver、query truth、真实batch类数、quota或global assignment分支。
- D22地面int8组件目前`formal_phase2_eligible=false`且`UNVERIFIED_UNDER_CURRENT_PROTOCOL`；D66真实使用ground后仍为负。D72锁定`ground_int8_component_input_count=0`，不得借研究叙述绕过协议资格。

## 4.预注册判门与停止规则

相对D62，D72必须：

1. B/A/N/H/J、min-B/A/N、3场景同类指标和三向混淆不发生交换伤害，并至少严格改善A、N、H、J、F或任一floor；
2. F改善必须同时满足A不降，不能仅由B下降产生；
3. INT8与matched FP32的support argmax变化、outer prediction变化和margin flip均为0；
4. K个leave-one分区exact-once，所有inner fit只见K-1 rank；最终只持久化一个int8/FP16 affine state；
5. 参数≤80k、optimizer step≤50、state≤256KB、query额外MAC=0，并据实报告额外闭式LDA/Fisher运算；
6. 完成真实105行后报告7候选、3场景、11类、15fold、bagging离散度、训练20epoch、量化、资源、artifact和D62/D65–D71同row对照。

失败即`COMPLETED_DIAGNOSTIC_NEGATIVE_NOT_PROMOTABLE`，停止leave-one head平均、trim/median/权重/温度/子采样率扫描，不跑第二seed或125。成功也先运行第二development seed，不直接启动125。D72完成后立即执行D70–D72三轮强制回顾。

## 5.版本与执行计划

- 根目录`E:\type10-7`不是Git仓库；本报告同步镜像到Git承载面`E:\type10-7\github_publish\CVS-RFFI-repo`。
- 计划新增：
  - `code/cvsrffi/stage2_d72_leave_one_head_bagging.py`；
  - `code/scripts/probe_d72_physical_rank_leave_one_head_bagging.py`；
  - `tests/test_stage2_d72_leave_one_head_bagging.py`；
  - `tests/test_probe_d72_physical_rank_leave_one_head_bagging.py`；
  - `analysis/d72_physical_rank_leave_one_head_bagging_traceability_20260719.md`。
- 本地验证必须显式激活`ssr-gpu`，先专项测试，再运行D42–D72相邻完整链；运行前提交、建立clean worktree、记录脚本SHA和精确命令。
- 不访问N607；真实development cell使用本地锁定Runner。输出目录必须在启动前不存在，失败目录不覆盖。

## 6.实现与主工作树验证

- 已实现D72 core、独立probe及两组测试；D62、D71及历史artifact均未修改。
- K8时每个top-level fit新增before8次＋final8次D62 leave-one联合头，共16次；两个阶段各物理rank恰好held一次，inner K7。平均后只保留一个D42 residual-int8/FP16 affine state，query额外MAC/state均为0。
- 专项测试首轮10/11通过；唯一失败是源码文本测试把必需运行时函数名`_bootstrap`误判为算法bootstrap。断言已收窄到不存在的`bootstrap_sample_indices`，算法、数据、分区、公式和资源路径均未改变；retry1为11/11通过。
- D42–D72相邻37文件完整链通过，退出码0，用时80.9s。覆盖exact-once、类置换等变、K1精确回退、D62继承闭包、无角色/query分支及资源计数。下一步提交实现并在clean worktree复跑。

## 7.clean验证、运行锁与精确命令

- 实现提交：`ba204176 implement D72 leave-one head bagging`；clean worktree：`E:\type10-7\code\snapshots\d72wt`，detached HEAD且`git status -sb`无改动。
- clean验证：D42–D72相邻37文件、348项全部通过，退出码0，用时81.0s；`py_compile`通过。
- 执行源SHA256：probe=`d6a7f4e2731a90cac394c9f33432d992f82d27400ab98ed93b52e82cdb3ec9ff`；core=`edf7940856b1f6b0dcd8577acbd4d967341d1b639a381a4b8f83cc3559389d69`；D62 helper=`38ae1114a06d135bca806f470417cd28a634fec0da449888665c6843615d4a20`。
- 环境：`C:\Users\lh594\.conda\envs\ssr-gpu\python.exe`；working directory为`E:\type10-7\code\snapshots\d72wt`；本地`device=auto`，不访问N607。
- 预期闭包：105行、30个target row、30次top fit、480次inner D62 fit、8760条component execution；before/final各30份8折exact-once audit，ground/query-fit/clean/source/role/quota访问0。

```powershell
& 'C:\Users\lh594\.conda\envs\ssr-gpu\python.exe' `
  'E:\type10-7\code\snapshots\d72wt\code\scripts\probe_d72_physical_rank_leave_one_head_bagging.py' `
  --d72-arm physical_rank_leave_one_head_bagging `
  --runtime-root 'E:\type10-7\code\snapshots\d41wt' --probe-root 'E:\type10-7\code\snapshots\d72wt' `
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
  --output 'E:\type10-7\automation_reports\CV-SincNet\d72_physical_rank_leave_one_head_bagging_probe_20260719\physical_rank_leave_one_head_bagging' `
  --device auto --mode development_select_unverified_component --candidate-set d42_v1
```
