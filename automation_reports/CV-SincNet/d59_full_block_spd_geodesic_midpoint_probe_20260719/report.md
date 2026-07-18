# D59 full/block SPD几何中点探针报告

## 1.身份与执行前状态

- 实验ID：`d59_full_block_spd_geodesic_midpoint_probe_20260719`
- 操作者：Codex`/root`
- 时间：2026-07-19（Asia/Hong_Kong）
- 状态：`COMPLETED_DIAGNOSTIC_NEGATIVE_NOT_PROMOTABLE`
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

## 7.真实执行与闭包

- 完成：105/105行，exit0，Runner elapsed`39.7936s`；7候选×3场景×5fold完整。
- selector：`selected_candidate_id=Z0_SUPPORT_ONLY`、`selected_positive_route=false`、`formal_metric_claim_allowed=false`；未执行selected-only full-K10。
- 协议：query opened/rows/labels均0；source/clean访问false；role Oracle、class quota、true batch class count、global assignment、query-dependent optimization均false；ground int8 entry/exit逐bit不变。
- D59 verifier：30条INT8/FP32 target row、60个before/final midpoint audit全部通过；Riccati相对Frobenius残差最大`1.1406e-14`，半程测地距离误差最大`1.5772e-12`。
- N607：未访问，无远端PID/GPU/log/sync；本轮是本地CUDA development证据。

## 8.结论先行

D59没有超过D46。总体before-old`92.22%`、after-old`82.22%`、seen-new`84.00%`、同rowH`82.16%`、forgetting`10.00pp`、joint`23.33%`、min-before`80.00%`、min-after`53.33%`、min-new`70.00%`，混淆`24/8/16`。

相对D46，D59的after-old提高`0.56pp`、forgetting降低`0.56pp`、old→new减少1次，但seen-new降低`0.67pp`、H降低`0.18pp`、最低新类降低`3.33pp`且new-new混淆增加1次。15fold中3个prediction SHA变化：low-elev fold0多保住1个旧类，low-elev fold3丢失1个新类，clear fold2只改变错误归属而不改变任何统计量。收益与损失直接交换，未形成joint/floor改善。

D59与D45的全部聚合、场景、类别floor和混淆计数完全相同，仅clear fold2的错误归属不同。因此SPD几何中点在这个cell上落到了D45全局LOO score融合的同一性能台阶，而没有解决D46的low-elev新类floor。当前最强仍为D46；D59不formalize、不运行第二seed、不生成125。

## 9.七候选完整同row性能

|候选|before|after|new|H|forget|joint|min-before|min-after|min-new|混淆old→new/new→old/new-new|判定|
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
|B3_SINGLE_IQ_DIAG_FFTRF|87.78%|75.56%|72.67%|73.35%|12.22pp|23.33%|80.00%|60.00%|40.00%|33/22/19|合法弱比较器|
|D42-D40-HNBR-INT8-NEGATIVE|85.56%|85.00%|15.33%|25.16%|0.56pp|0.00%|66.67%|63.33%|0.00%|2/0/0|新类不可达|
|D42-D41-BEC-INT8-NEGATIVE|86.11%|20.56%|78.67%|31.50%|65.56pp|0.00%|76.67%|0.00%|36.67%|142/0/32|旧类崩溃|
|D42-PROTOnet-CDA-ZID160|71.11%|48.33%|52.67%|48.97%|22.78pp|0.00%|33.33%|13.33%|3.33%|0/0/0|弱基线|
|D59 FP32 matched|92.22%|82.22%|84.00%|82.16%|10.00pp|23.33%|80.00%|53.33%|70.00%|24/8/16|与INT8完全一致|
|D59 INT8|92.22%|82.22%|84.00%|82.16%|10.00pp|23.33%|80.00%|53.33%|70.00%|24/8/16|主候选，负结果|
|Z0_SUPPORT_ONLY|71.11%|48.33%|52.67%|48.97%|22.78pp|0.00%|33.33%|13.33%|3.33%|0/0/0|identity control|

unknown、coverage、rollback、defer均为N/A；每行指标来自同一candidate的15个outer rows，没有拼接不同run的单项极值。

## 10.相对关键版本的完整表现

|版本|before|after|new|H|forget|joint|min-after|min-new|混淆|
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
|D42 full|90.56%|81.67%|81.33%|80.63%|8.89pp|23.33%|50.00%|70.00%|26/10/18|
|D43 block|92.22%|83.89%|82.67%|82.30%|8.33pp|30.00%|56.67%|66.67%|19/10/16|
|D45 global LOO|92.22%|82.22%|84.00%|82.16%|10.00pp|23.33%|53.33%|70.00%|24/8/16|
|D46 classwise LOO|92.22%|81.67%|84.67%|82.33%|10.56pp|23.33%|53.33%|73.33%|25/8/15|
|D58 OVR校准|80.00%|74.44%|69.33%|70.92%|5.56pp|3.33%|26.67%|33.33%|16/26/20|
|D59 SPD中点|92.22%|82.22%|84.00%|82.16%|10.00pp|23.33%|53.33%|70.00%|24/8/16|

相对D42，D59提高before`1.67pp`、after`0.56pp`、new`2.67pp`、H`1.53pp`，min-before/min-after各提高`3.33pp`且三类混淆各减少2次；但forgetting增加`1.11pp`。相对D43 block，D59以after`−1.67pp`、forgetting`+1.67pp`、joint`−6.67pp`换取new`+1.33pp`和min-new`+3.33pp`。这说明连续协方差几何确实处于full/block权衡之间，但中点不是D46所需的联合最优点。

## 11.逐场景性能与表现

|场景|before|after|new|H|forget|joint|min-before|min-after|min-new|混淆|相对D46|
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
|clear|98.33%|90.00%|98.00%|93.57%|8.33pp|40.00%|90.00%|70.00%|90.00%|4/1/0|全部指标相同；fold2仅错误归属变化|
|low-elev|88.33%|80.00%|74.00%|75.45%|8.33pp|20.00%|80.00%|60.00%|40.00%|7/5/8|after`+1.67pp`、forget`−1.67pp`，但new`−2pp`、H`−0.53pp`、min-new`−10pp`|
|rain|90.00%|76.67%|80.00%|77.45%|13.33pp|10.00%|60.00%|30.00%|70.00%|13/2/8|与D46全部指标相同|

失败集中在low-elev：D59在fold0把after-old从66.67%提高到75.00%，但在fold3把seen-new从70%降到60%；这不是整体迁移改善，而是旧/新边界上的一进一退。

## 12.全部匿名类性能

类名按opaque handle排序匿名为O0–O5/N0–N4，仅用于报告，不参与方法。

|旧类|before|after|相对D46|
|---|---:|---:|---|
|O0`cls_1f33`|90.00%|90.00%|相同|
|O1`cls_33bb`|96.67%|93.33%|相同|
|O2`cls_75aa`|96.67%|90.00%|相同|
|O3`cls_8b02`|80.00%|53.33%|相同，仍为旧类floor|
|O4`cls_a53c`|100.00%|73.33%|相同|
|O5`cls_f8df`|90.00%|93.33%|相同|

|新类|D59|D46|变化|
|---|---:|---:|---:|
|N0`cls_09f8`|70.00%|73.33%|−3.33pp，D59新类floor|
|N1`cls_1c2a`|93.33%|93.33%|0|
|N2`cls_b8fb`|76.67%|76.67%|0|
|N3`cls_d3af`|90.00%|90.00%|0|
|N4`cls_f608`|90.00%|90.00%|0|

D59与D46所有旧类聚合准确率完全一致，唯一类别级退化是N0。这解释了为什么after平均有小幅收益却没有min-after改善，而min-new直接失败。

## 13.十五fold完整表现

|场景|fold|before|after|new|H|forget|joint|before/after/new floor|混淆|
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
|clear|0|100.00%|100.00%|90.00%|94.74%|0.00pp|50.00%|100/100/50%|0/1/0|
|clear|1|100.00%|83.33%|100.00%|90.91%|16.67pp|0.00%|100/0/100%|1/0/0|
|clear|2|91.67%|83.33%|100.00%|90.91%|8.33pp|50.00%|50/50/100%|1/0/0|
|clear|3|100.00%|91.67%|100.00%|95.65%|8.33pp|50.00%|100/50/100%|1/0/0|
|clear|4|100.00%|91.67%|100.00%|95.65%|8.33pp|50.00%|100/50/100%|1/0/0|
|low|0|91.67%|75.00%|80.00%|77.42%|16.67pp|50.00%|50/50/50%|3/1/1|
|low|1|66.67%|58.33%|70.00%|63.64%|8.33pp|0.00%|50/50/0%|1/0/3|
|low|2|91.67%|91.67%|70.00%|79.38%|0.00pp|0.00%|50/50/0%|0/2/1|
|low|3|100.00%|100.00%|60.00%|75.00%|0.00pp|0.00%|100/100/0%|0/1/3|
|low|4|91.67%|75.00%|90.00%|81.82%|16.67pp|50.00%|50/50/50%|3/1/0|
|rain|0|83.33%|83.33%|60.00%|69.77%|0.00pp|0.00%|50/50/0%|2/0/4|
|rain|1|100.00%|66.67%|90.00%|76.60%|33.33pp|0.00%|100/0/50%|4/1/0|
|rain|2|91.67%|83.33%|80.00%|81.63%|8.33pp|50.00%|50/50/50%|1/0/2|
|rain|3|91.67%|75.00%|90.00%|81.82%|16.67pp|0.00%|50/0/50%|3/0/1|
|rain|4|83.33%|75.00%|80.00%|77.42%|8.33pp|0.00%|50/50/0%|3/1/1|

## 14.SPD几何、训练、量化与资源

- 30/30个before/final fit均激活几何中点，无回退。before midpoint condition number均值`117,977`，位于block`108,246`与full`137,470`之间；final均值`93,566`，位于block`88,237`与full`103,739`之间。
- 跨块Frobenius能量占比很小：before均值`3.10e-6`、final`2.43e-6`；但协方差病态白化后，block→full仿射不变距离分别为`4.463/4.053`。因此“原矩阵能量小”不等于“判别几何影响小”。
- 中点到两端距离严格各为总距离一半；60个Riccati残差约`1e-14`，实现数值稳定。性能失败不是SPD闭包或退化回退造成。
- 训练正常：epoch1 loss/support-acc均值`1.0320/95.14%`，epoch20为`0.1027/100%`；所有300条INT8 trace finite，20epoch/20step完整，query rows始终0。
- 量化：before/final outer argmax变化`0/0`，support变化`0/0`，margin翻转0；max score绝对误差min/mean/max=`0.0190/0.0348/0.0604`。误差幅度高于D46，但没有改变当前15fold决策。
- 资源：2次closed-form LDA、60,466,176 LDA MAC；D59 SPD稠密代数保守上界`1,911,029,760` MAC-equivalent，总适配`1,976,472,576`；query额外0，总query`6,624` MAC；参数`2,016`、state`8,583B`、registry`941B`、CUDA metric峰值`22,886,912B`、20epoch/20step。host FP64 covariance峰值未实测。
- 虽然保守适配MAC上界高于D46，真实Runner耗时`39.79s`低于D46的`76.98s`，因为D59只有2次主LDA且没有36次inner-LOO fit；MAC-equivalent是稠密代数上界，不是wall-clock测量。

## 15.预注册门判定

|门|结果|证据|
|---|---|---|
|105行、query0、协议/source/ground/artifact闭包|PASS|Runner与D59 verifier全部通过|
|量化0/0/0|PASS|before/final argmax和margin翻转全0|
|聚合before/after不退化|PASS|before相同，after+0.56pp|
|聚合new/H不退化|FAIL|new−0.67pp、H−0.18pp|
|forgetting不增加|PASS|−0.56pp|
|joint/min-before/min-after不退化|PASS|全部相同|
|min-new不退化|FAIL|73.33%→70.00%|
|final floor至少一项严格改善|FAIL|joint/min-after相同，min-new退化|
|三场景联合不退化|FAIL|low-elev new/H/min-new退化|
|混淆不增加|FAIL|new-new 15→16|
|至少1个prediction变化|PASS|3/15相对D46变化|

最终：`COMPLETED_DIAGNOSTIC_NEGATIVE_NOT_PROMOTABLE`。不运行125。

## 16.Artifact闭包

|Artifact|Bytes|SHA256|
|---|---:|---|
|`training_log.jsonl`|3,964,132|`e026adb333fff1f85da6dd069aa62dd370c59592425cf27b89b8db766f1faff4`|
|`support_audit.json`|313,278|`155387627f2054c4b9093bc1335d0484c88c138f6496b82449a6fb6a3f15597b`|
|`selection.json`|2,990|`127125f4455ef465553c4930a435532b618f0516aff93704bd738a4fe366b8a8`|
|`RECEIPT.json`|4,655|`c1781f87e2c41dfc6c9eca1db001b32b80896a6b1e1b4462e233868fb7717e42`|
|`D59_PROBE_METADATA.json`|1,816|`0ee567e2e1acbb697a3c5a9e2b5578746e9da9e0b8d08dfcf4a29bd3c69ba116`|
|`geometry_audit.json`|5,132|`ae4b735a45fdae38eaf8bb6bfd23e57df9d60560144027a1e3e33937556300dc`|
|`resource_audit.json`|6,498|`00f364e567e0462feb955321e6d414c0dc95493dab35d3ed00dfacd71a8d6e2b`|
|`full_performance_summary.json`|106,936|`7acbf1b903684a435cef1f6318790ffdda1ba909b2e8a7572ec119f15119593b`|

## 17.下一轮研发决策

D59证明：在协方差流形上做无参数中点，能可靠地连续连接full与block并保持共享类别尺度，但它仍复现D45的旧/新交换，没有形成下尾提升。停止以下路线：扫描geodesic位置、给中点加ridge/floor、把D59再与full/block做固定score融合；这些都会退化成D44–D50的权重搜索。

下一轮D60应使用新的共享证据：不再在full/block一维路径上找位置，而是审计跨块相关性的稳定子空间。候选方向是support-only、类对称的跨块canonical-correlation低秩保留：保留可由独立physical-rank支持稳定复现的跨块方向，其他跨块项回到block covariance；同一投影作用于所有类别，无class/role/scene分支。D60完成后将是D58–D60三轮，必须先做记录化复盘再启动D61。
