# D72物理rank留一联合头bagging开发报告

## 1.实验登记

- 实验ID：`d72_physical_rank_leave_one_head_bagging_probe_20260719`；operator：Codex；状态：`COMPLETED_DIAGNOSTIC_NEGATIVE_NOT_PROMOTABLE`。
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

## 8.启动与监控

- 2026-07-19 23:45:21启动，PID`7996`。首次前台工具调用在14s达到工具时限并返回124，但只读进程检查确认同一精确命令的Python子进程仍存活；这不是实验失败或成功证据，没有重试、重启或覆盖输出。
- 当前转为只读离散监控；只有PID退出后才解析输出目录、105行闭包、RECEIPT和全部artifact。若最终失败，原目录保留并按实际阶段报告。

## 9.执行完成与artifact闭包

- PID`7996`于23:58:28前退出；Runner实测`elapsed_seconds=779.7196`。首次工具层124超时没有中断子进程，最终仅有一次执行、一个输出目录。
- RECEIPT：`DEVELOPMENT_SUPPORT_ONLY_DIAGNOSTIC_NEGATIVE_NOT_PROMOTABLE`，`selected_candidate_id=Z0_SUPPORT_ONLY`，`selected_positive_route=false`，`query_opened=false`，formal/performance claim均false。
- 闭包：105/105行、7候选×15fold、30个target row、30次top fit、480次inner D62 fit、8760条component execution；before/final leave-one各240次，ground输入0，query/clean/source/role/quota访问0。
- 完整摘要：`d72_full_performance_summary.json`，125,890B，SHA256`52cb75f53ec2c188a893d64d2a91fea998c8b3b2f7965ffaa50a4087aa75e177`。

|artifact|字节|SHA256|
|---|---:|---|
|`training_log.jsonl`|14,900,838|`138c3dd4c78e11e88c484211e3a6792d017b8b32ce3f2666f943854a71f19a08`|
|`support_audit.json`|313,677|`b2ce19efc3530f43a5b7f5b0d6a6c615b9f1eaebf7f6a3c51af70c380084e1ae`|
|`resource_audit.json`|6,498|`00f364e567e0462feb955321e6d414c0dc95493dab35d3ed00dfacd71a8d6e2b`|
|`geometry_audit.json`|5,132|`ae4b735a45fdae38eaf8bb6bfd23e57df9d60560144027a1e3e33937556300dc`|
|`selection.json`|2,990|`7109c4bfa70a858655d9c8e8bbc741cfb4fa282b900b26508b3bad8ebc73e936`|
|`RECEIPT.json`|5,028|`b756017039267c5ab219dd448938e55cd7271e8e5082db9854530eb12e9fbb2b`|
|`D72_PROBE_METADATA.json`|2,314|`ee8fc7580914b70d6572c2c9e0494afe5cbdde3f2851ba971335ae6a6c9d5cda`|

## 10.七候选同排性能

|候选|B|A|N|H|F|J|min-B/A/N|row-floor B/A/N|混淆old→new/new→old/new→new|判定|
|---|---:|---:|---:|---:|---:|---:|---|---|---|---|
|B3|87.78|75.56|72.67|73.35|12.22|23.33|80.00/60.00/40.00|53.33/33.33/36.67|33/22/19|比较器|
|D40-HNBR|85.56|85.00|15.33|25.16|0.56|0.00|66.67/63.33/0.00|40.00/40.00/0.00|2/0/0|新类不可达|
|D41-BEC|86.11|20.56|78.67|31.50|65.56|0.00|76.67/0.00/36.67|46.67/0.00/26.67|142/0/32|旧类崩溃|
|ProtoNet-CDA/Z0|71.11|48.33|52.67|48.97|22.78|0.00|33.33/13.33/3.33|13.33/0.00/0.00|0/0/0|负对照|
|D72 FP32 matched|93.33|82.78|82.67|81.59|10.56|26.67|80.00/53.33/70.00|73.33/50.00/46.67|22/11/15|与INT8同判|
|D72 INT8|93.33|82.78|82.67|81.59|10.56|26.67|80.00/53.33/70.00|73.33/50.00/46.67|22/11/15|负向，不晋级|

B/A/N/H/F/J均为15fold同row均值或同row统计，不拼接边际极值。D72虽把B和A各提高0.56pp，但N下降2.00pp、H下降1.03pp、min-N下降3.33pp，违反无交换门。

## 11.三场景性能

|场景|B|A|N|H|F|J|min-B/A/N|row-floor B/A/N|混淆|
|---|---:|---:|---:|---:|---:|---:|---|---|---|
|clear|98.33|91.67|98.00|94.44|6.67|50.00|90.00/70.00/90.00|90.00/60.00/90.00|2/1/0|
|low-elev|91.67|80.00|70.00|72.89|11.67|20.00|80.00/60.00/40.00|70.00/60.00/20.00|7/7/8|
|rain|90.00|76.67|80.00|77.45|13.33|10.00|60.00/30.00/70.00|60.00/30.00/30.00|13/3/7|

相对D62，clear完全不变；low-elev A+1.67pp但N−6.00pp、H−3.10pp、min-N−10pp；rain只使B+1.67pp，A/N/H不变，因此rain遗忘反而+1.67pp。bagging的有效变化集中在弱场景，但方向偏向旧类而非联合改善。

## 12.逐类性能

|类|阶段/角色|准确率|
|---|---|---:|
|O1|before/after|93.33/90.00|
|O2|before/after|96.67/93.33|
|O3|before/after|96.67/93.33|
|O4|before/after|80.00/53.33|
|O5|before/after|100.00/73.33|
|O6|before/after|93.33/93.33|
|N1|seen-new|70.00|
|N2|seen-new|93.33|
|N3|seen-new|76.67|
|N4|seen-new|90.00|
|N5|seen-new|83.33|

旧类瓶颈仍是O4的53.33%，新类瓶颈N1由D62的73.33%降到70.00%。这不是难类定向调参依据，只用于验证统一方法没有改善通用下尾。

## 13.十五个outer fold同排明细

|场景/fold|B|A|N|H|F|J|floor B/A/N|混淆|
|---|---:|---:|---:|---:|---:|---:|---|---|
|clear/0|100.00|100.00|90.00|94.74|0.00|50.00|100/100/50|0/1/0|
|clear/1|100.00|83.33|100.00|90.91|16.67|0.00|100/0/100|0/0/0|
|clear/2|91.67|83.33|100.00|90.91|8.33|50.00|50/50/100|1/0/0|
|clear/3|100.00|100.00|100.00|100.00|0.00|100.00|100/100/100|0/0/0|
|clear/4|100.00|91.67|100.00|95.65|8.33|50.00|100/50/100|1/0/0|
|low/0|100.00|75.00|80.00|77.42|25.00|50.00|100/50/50|3/1/1|
|low/1|75.00|58.33|60.00|59.15|16.67|0.00|50/50/0|1/1/3|
|low/2|91.67|91.67|70.00|79.38|0.00|0.00|50/50/0|0/2/1|
|low/3|100.00|100.00|50.00|66.67|0.00|0.00|100/100/0|0/2/3|
|low/4|91.67|75.00|90.00|81.82|16.67|50.00|50/50/50|3/1/0|
|rain/0|83.33|83.33|60.00|69.77|0.00|0.00|50/50/0|2/0/4|
|rain/1|100.00|66.67|90.00|76.60|33.33|0.00|100/0/50|4/1/0|
|rain/2|91.67|83.33|80.00|81.63|8.33|50.00|50/50/50|1/0/2|
|rain/3|91.67|75.00|90.00|81.82|16.67|0.00|50/0/50|3/1/0|
|rain/4|83.33|75.00|80.00|77.42|8.33|0.00|50/50/0|3/1/1|

相对D62有5/15个prediction hash变化：low0的A+8.33pp；low1的B−8.33pp且N−10pp；low3的N−20pp；rain3只提高B8.33pp；rain4预测变化但汇总指标不变。净结果是旧→新−1次，却new→old+3次，说明平均头把边界偏向旧类。

## 14.bagging机制行为

- before/final各15/15 fit激活，共120＋120个outer目标阶段leave-one头；每个audit含8个K7分区，exact-once=true，train-held最大交集0。
- support预测变化before/final均为0；before support base/bagged准确率均100%，final均值均99.848%。因此support自拟合面无法区分D62与D72，不能把“稳定”解释为held安全。
- before系数离散度RMS min/mean/max=`0.2952/0.3339/0.3805`，最大row L2=`7.6607/10.4667/14.3949`；final为`0.3476/0.3868/0.4389`和`9.8485/12.3848/17.3046`。头间差异真实存在，但算术平均主要平滑到旧类方向。
- inner D62 before状态：active64、atomic fallback45、no-row11，平均接受1.30行；final为active22、atomic fallback95、no-row3，平均接受0.533行。final大部分K7头的Fisher行证据不稳定，平均这些完整头无法恢复新类可达性。
- ground组件输入严格为0。D72结果不能归功于地面压缩原型；D66仍是唯一当前真实读取ground int8的相邻负向版本。

## 15.训练、量化与资源

- 旧域metric继承D42的20epoch/20step：epoch1 loss均值1.0320、support acc95.14%、grad norm1.0838；epoch20 loss0.10268、support acc100%、grad norm0.13535；所有epoch query rows总和0。
- INT8与matched FP32：before/outer argmax变化0、support argmax变化0、margin sign flip0；score最大绝对误差min/mean/max=`0.000216/0.000797/0.001565`。量化不是本轮性能缺陷来源。
- 每row共584次闭式component fit；D72额外512次。D72新增适配MAC-equivalent为255,515,405,384，总适配280,406,629,354；query仍6,624MAC/样本，额外0。
- 参数2,016，persistent state8,583B，registry941B，峰值CUDA22,886,912B，20epoch/20step，dense query graph0。部署状态满足硬上限，但适配计算相对D62显著增大而性能更差。

## 16.相邻版本同排比较与目标差距

|版本|B|A|N|H|F|J|min-B/A/N|混淆|主要行为|
|---|---:|---:|---:|---:|---:|---:|---|---|---|
|D62|92.78|82.22|84.67|82.62|10.56|26.67|80.00/53.33/73.33|23/8/15|当前联合最强|
|D65|92.22|86.11|59.33|67.12|6.11|16.67|80.00/70.00/46.67|16/28/33|旧row冻结、新类塌缩|
|D66|93.33|83.33|83.33|82.59|10.00|23.33|80.00/53.33/66.67|20/9/16|ground真实输入，交换floor|
|D67|92.78|82.78|83.33|82.16|10.00|26.67|80.00/53.33/73.33|22/11/14|连续堆叠交换新类|
|D68|58.89|51.67|14.00|18.66|7.22|0.00|50.00/43.33/0.00|20/118/11|标尺崩溃|
|D69|92.78|81.67|74.67|77.39|11.11|30.00|80.00/53.33/53.33|27/23/15|冻结旧行追加失败|
|D70|92.78|82.22|84.67|82.62|10.56|26.67|80.00/53.33/73.33|25/8/15|几乎全回退D62|
|D71|91.11|82.22|84.00|82.33|8.89|26.67|83.33/53.33/73.33|23/9/15|低F来自B下降|
|D72|93.33|82.78|82.67|81.59|10.56|26.67|80.00/53.33/70.00|22/11/15|bagging偏旧，伤新类|

D72距K10/new5门：A差9.22pp、min-A差34.67pp、N差9.33pp；没有资格启动第二seed、125或确认矩阵。D62仍为当前最强版本。

## 17.最终判定

D72状态为`COMPLETED_DIAGNOSTIC_NEGATIVE_NOT_PROMOTABLE`。它证明physical-rank leave-one完整头存在明显参数方差，并能轻微提高旧类B/A，但support自拟合面0变化无法预测outer边界方向；最终以新类准确率、H、min-new和new→old混淆为代价。按预注册停止leave-one头平均及其trim/median/权重/温度/子采样率变体，不跑第二seed或125。

## 18.D70–D72三轮强制回顾

回顾重新读取goal objective（SHA256`92f5f155939505dc45b51a1bdde77e606ce56380e706e09127218a1287bda29e`）、`项目.md`（`45683ac1e4f031a8307ac4fcb7745922ed965483975aa7b1258f78d3f6fd4920`）及active method goal，并检索conversation index。索引未出现当前活跃对话内的D70–D72条目，只返回历史qKNN/ground int8和goal创建记录，因此本次技术结论以三份当前完整summary和原始日志为准：D70/D71/D72 summary SHA分别为`2f6abd70ec456576849bc2a9c93e69004f8c4fbd84bd8defba9405df1affadea`、`7181b45081774b46eee11262fd6faf818a44f032e094129032e2fdcc664ea05d`、`52cb75f53ec2c188a893d64d2a91fea998c8b3b2f7965ffaa50a4087aa75e177`。

|轮次|假设|真实结果|保留/淘汰|
|---|---|---|---|
|D70|support-held原子门选择整条生命周期行|14/15fold精确回退，仅1行接受；总体等于D62且混淆更差|淘汰生命周期行替换和更松门|
|D71|top-2局部pair可减少旧→新且不扰动全局|只在clear3折激活；B下降、新类下降，low/rain全回退|淘汰pair重排、阈值/温度变体|
|D72|leave-one完整头平均可降低少样本方差|B/A小升但N/H/min-N下降，new→old+3；support预测0变化|淘汰head bagging及统计聚合变体|

共同失败不是量化、资源或协议，而是**support自拟合/内部非劣证据对outer old/new竞争方向缺乏辨识力**：D70/D71以门控回退为主，无法触达弱场景；D72去掉门后确实改变弱场景，但系统性向旧类偏移。三轮均保持LEO_weak-only、无clean/source/query truth/role/quota、全注册逐样本决策和ground输入0，协议无越界；也都同时报告before/after/new/H、逐类和forgetting，因此证据完整但性能未晋级。

下一轮D73不再做行、pair、score或head的post-hoc选择/平均，也不扫描这些路线。最有价值的新方向是回到表示学习本身：在D42强Stage2-B metric上加入一次**Stage2-C全注册support的类对称小步联合metric更新**，用统一的old+new分类/半径间隔目标，同时以注册前旧支持分数做连续蒸馏约束；最终仍重拟合单一D62 int8头。它必须把A与N/H同时作为一个锁定机制检验，不能用support gate选择是否启用；在实现前先核对D21-M6、D31、D36和D61，确保与既有低秩/联合头负路线有实质差异。
