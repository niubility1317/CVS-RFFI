# D73新旧任务冲突投影联合度量实验报告

## 1.实验身份与当前状态

|字段|值|
|---|---|
|实验ID|`d73_conflict_projected_joint_metric_probe_20260720`|
|候选|`conflict_projected_joint_metric`|
|时间|2026-07-20|
|operator|Codex `/root`|
|状态|`COMPLETED_DIAGNOSTIC_NEGATIVE_NOT_PROMOTABLE`|
|目标|在不使用地面组件和query信息的条件下，以旧类保持/新类注册等权冲突投影的一次共享metric更新同时改善D62的注册后old与new|
|比较目标|D62：`B/A/N/H/F/J=92.78/82.22/84.67/82.62/10.56/26.67%`|

## 2.假设与非重复性

D70–D72说明support内score/head后处理不能可靠推断outer新旧方向。D73检验更窄的假设：从D42/D62强metric出发，将旧类与新类support任务梯度单位化并在冲突时对称投影，是否能用一个无扫描的Stage2-C步改善共享表示。它不同于D21-M6的低秩arm选择、D31的新类suffix、D36的12步软损失加权和D61的固定Fisher残差。

## 3.协议、数据与运行锁

- `protocol_schema=p2_min_v1`；复用D18的`VALIDATED_ONCE` capsule，不因方法变化重验数据。
- receiver`20-1`、seed`713101`、K10/new5、`clear/low_snr/rain`×5 folds；outer-fit实际每类K8。
- 每个物理IQ只有一个固定`leo_*_weak`观测；support/query物理ID隔离。
- support-only拟合；query逐样本、一次性、全注册类argmax；无query truth/role、batch类数、quota或global assignment。
- D22地面int8清单当前`formal_phase2_eligible=false`；D73地面输入、更新和状态均为0。
- before保持D62；final只执行一次确定性对角metric步、一次D62统一头refit和一次int8编译。

## 4.锁定机制与开发门

旧类与新类support分别在all-registered leave-one prototype softmax上形成任务梯度。负余弦时对称PCGrad，非负时等权合成；去除共同缩放方向后按`||delta||_2=sqrt(K/(K+288))`执行一次更新。无步长、温度、rank、权重、阈值或场景扫描。

正向门要求相对D62的`A/N/H/min-A/min-N`全部不退化且至少一项严格提高，并且`B/F`、逐场景联合表现和混淆无交换伤害。失败即停止本路线，不开第二seed或125矩阵。

## 5.版本、文件、验证与同步

|项目|预注册值|
|---|---|
|Git仓库|`E:\type10-7\github_publish\CVS-RFFI-repo`|
|分支|`codex/cvs-rffi-release-20260626`|
|根目录状态|`E:\type10-7`不是Git仓库；本报告同时镜像到Git承载面|
|本地改动|预注册追溯、core、probe、测试、汇总脚本与本报告；逐阶段补充commit/hash|
|本地环境|`ssr-gpu`|
|N607同步|本轮开发单元计划本地执行；无需SSH/SCP|

## 6.计划命令、日志与资源

|字段|计划值|
|---|---|
|working directory|Git干净快照worktree，待实现后锁定|
|Python|`C:\Users\lh594\.conda\envs\ssr-gpu\python.exe`|
|主命令|待实现和source-hash锁定后补录，不覆盖既有输出|
|输出目录|`E:\type10-7\automation_reports\CV-SincNet\d73_conflict_projected_joint_metric_probe_20260720\conflict_projected_joint_metric`|
|训练日志|输出目录下`training_log.jsonl`|
|GPU|本地单GPU；启动前记录占用与PID|
|期望artifact|support/resource/geometry/selection/receipt/metadata及完整性能汇总|
|资源上限|参数≤80k、epoch≤30、optimizer step≤50、状态≤256KB、无dense query graph|

## 7.完成后必须补录

必须补录：完整105行闭包、总体同row指标、3场景、6旧类、5新类、15fold、三类混淆、任务损失/梯度余弦/投影/一阶变化、训练trace、int8-vs-FP32、资源、artifact大小/SHA、相对D62与目标差距、缺陷、最终判定和下一实验。禁止用跨候选的边际最大值拼成“最佳性能”。

## 8.结果表占位

|candidate|机制|receiver/TX|K/seed|B old|A old|seen-new|H|forgetting|joint|min-B/A/N|混淆O→N/N→O/N→N|量化|资源|判定|
|---|---|---|---|---:|---:|---:|---:|---:|---:|---|---|---|---|---|
|D73|等权PCGrad单步共享metric|20-1/new5|K10/713101|92.78|82.22|84.67|82.62|10.56|26.67|80/53.33/73.33|23/8/15|0 flip|21step/8,583B|负向，不晋级|

## 9.实现锁定（2026-07-20 00:24）

|文件|用途|SHA256|
|---|---|---|
|`code/cvsrffi/stage2_d73_conflict_projected_joint_metric.py`|解析leave-one prototype CE梯度、等权对称冲突投影、唯一一次metric更新与审计|`8d59e1c92887930cecebbd59e534e572299a64b875d1add559238eb9177391d5`|
|`code/scripts/probe_d73_conflict_projected_joint_metric.py`|D62包装、统一int8编译、资源/训练/source闭包和真实Runner入口|`c2086e86d23119acb6253a9bb365b1e43c2effc27fdfe4f4eea867b1b995f5a0`|
|`tests/test_stage2_d73_conflict_projected_joint_metric.py`|确定性、任务安全、组内类置换、K1和fail-closed测试|`623d15ebdad5e4c93c6a2c8dcf148c56019113ac204240785940c78cfceb7b64`|
|`tests/test_probe_d73_conflict_projected_joint_metric.py`|D62包装、资源公式、调用闭包和协议字段测试|`f769d331a0fed1a75055abea2fce7f44ac12bfba4ed7a411fa0990eb175d4d79`|

专项验证命令：`python -m pytest -q tests/test_stage2_d73_conflict_projected_joint_metric.py tests/test_probe_d73_conflict_projected_joint_metric.py`，在`ssr-gpu`中通过`9/9`。首次core测试收集失败仅因测试误写为`code.cvsrffi`导入路径；改为项目既有`cvsrffi`包路径后通过，不涉及机制、公式或运行代码变化。

D42–D73相邻完整链覆盖40个文件、377项测试，全部通过，退出码0，用时82.8秒。pytest在全部测试完成后的atexit清理阶段对全局`pytest-current`临时链接报`WinError 5`，不影响退出码或任何测试结果；clean worktree复验将显式使用仓内`--basetemp`消除该主机清理噪声。

## 10.clean验证与运行锁（2026-07-20 00:29）

- clean worktree：`E:\type10-7\code\snapshots\d73wt`，detached HEAD=`762baecbf525db1714d73a00490f6562423a0128`，`git status -sb`仅显示`HEAD (no branch)`。
- clean验证：D42–D73共40文件、377项全部通过，用时82.7秒；显式仓外`--basetemp`后无清理告警；D73 core/probe的`py_compile`通过。
- 执行源SHA256：probe=`e3fd5b536528780a772d36cba9f0a86136123b3280183e127bec325d93f758b9`；core=`bdb104ceb82c9f069499dd920b88599a455a65defdd3f622184bbe8dfbe2bd63`；D62 helper=`38ae1114a06d135bca806f470417cd28a634fec0da449888665c6843615d4a20`。
- 启动前GPU0为RTX5070Ti，显存`1106/16303MiB`、利用率0%；输出目录不存在。使用本地GPU，不访问N607。
- 预期闭包：105行、30个target row、30次top fit、30次额外D62 final refit、1620条D62 residual component execution；每个目标行21步、ground/query-fit/clean/source/query-role/quota访问0。

```powershell
& 'C:\Users\lh594\.conda\envs\ssr-gpu\python.exe' `
  'E:\type10-7\code\snapshots\d73wt\code\scripts\probe_d73_conflict_projected_joint_metric.py' `
  --d73-arm conflict_projected_joint_metric `
  --runtime-root 'E:\type10-7\code\snapshots\d41wt' --probe-root 'E:\type10-7\code\snapshots\d73wt' `
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
  --output 'E:\type10-7\automation_reports\CV-SincNet\d73_conflict_projected_joint_metric_probe_20260720\conflict_projected_joint_metric' `
  --device auto --mode development_select_unverified_component --candidate-set d42_v1
```

## 11.启动与监控

- 2026-07-20 00:30:24启动唯一一次真实执行，PID`20132`；Python命令行与锁定命令一致，working directory为clean worktree。
- 启动后只读检查确认PID存活、输出目录已创建，launcher stdout/stderr暂为0B；这属于Runner早期阶段，不是性能或完成证据。
- 当前转为离散只读监控；不重试、不覆盖输出。PID退出后才解析RECEIPT、105行闭包和完整artifact。

## 12.首次启动失败与R1修复

- PID`20132`在00:31:24前退出；输出目录保留但为空，training log、RECEIPT和任何可评分artifact均不存在。因此这是`FAILED_BEFORE_FIRST_OUTER_ROW`，不是完成实验，性能不得报告。
- launcher stderr精确失败：wrapper在Runner生成row级`total_optimizer_steps`之前访问该字段，触发`KeyError: 'total_optimizer_steps'`。D42 result此阶段已有`optimizer_steps`，而Runner随后以它生成`total_optimizer_steps`。
- 同时代码审计发现row级`complete_loss_trace`由`result.training_trace`生成；R1除去对尚不存在字段的提前更新，并把新增Stage2-C trace显式写回冻结dataclass的`training_trace`。机制公式、数据、梯度、步长、D62 refit、量化和资源数值均不变。
- 新增源码测试锁定：包装器必须回写`training_trace=tuple(trace)`，提前更新块不得含`total_optimizer_steps`。R1专项9/9通过。
- 原失败目录`conflict_projected_joint_metric`和launcher日志保留，不覆盖。完成全链、提交与新clean worktree后，只允许使用新目录`conflict_projected_joint_metric_retry1`执行一次R1。

R1验证与运行锁：主工作树40文件、377项全部通过，用时82.0秒；clean worktree`E:\type10-7\code\snapshots\d73r1wt`锁定commit`e319aa7e4c7b016e4294d361dd9117ca9914e72f`，同一377项全部通过，用时82.8秒，`py_compile`通过且worktree clean。R1执行SHA：probe=`b7d0584298285b4a05fb3d0d0dc733e210ab78ba1af4486858075a0f441e2180`、core=`bdb104ceb82c9f069499dd920b88599a455a65defdd3f622184bbe8dfbe2bd63`、D62 helper=`38ae1114a06d135bca806f470417cd28a634fec0da449888665c6843615d4a20`。00:37:42检查新输出目录不存在；GPU0显存`1106/16303MiB`、利用率0%。R1使用与第10节完全相同的数据/策略参数，只把`probe-root/script`改为`d73r1wt`、输出改为`conflict_projected_joint_metric_retry1`。

R1于2026-07-20 00:38:53启动，PID`25576`。只读检查确认命令行、clean worktree、新输出目录和锁定参数一致，stderr为0B；转入只读离散监控，不再次启动。

R1 Runner于00:41:51完成105/105行和6个标准artifact，RECEIPT状态为`DEVELOPMENT_SUPPORT_ONLY_DIAGNOSTIC_NEGATIVE_NOT_PROMOTABLE`、query未打开、elapsed=172.5498秒；随后probe末尾以`D73 cap/training trace drift`拒绝生成metadata。检查目标行发现资源步数均正确为21，但`complete_loss_trace`只有D73第21步：R1从此阶段尚为空的`resource.complete_loss_trace`起始，而非D42 dataclass内已有的20步`result.training_trace`。

该目录具备评分行但缺少完整20+1训练轨迹，故仍标记`COMPLETED_RUNNER_OUTPUT_REJECTED_BY_PROBE_INCOMPLETE_TRACE`，不得据此报告或选择性能。R2唯一修复为从`result.training_trace`复制原20步再追加第21步；metric、D62 refit、int8状态、预测与资源公式不变。R1目录和stderr原样保留，R2将使用新clean worktree和`conflict_projected_joint_metric_retry2`。

R2验证与运行锁：专项9/9、主工作树D42–D73完整链377/377均通过（82.1秒）；clean worktree`E:\type10-7\code\snapshots\d73r2wt`锁定commit`3a5e3d759eceeff69ca200fcf241385b309d6dfe`，专项9/9与`py_compile`通过且worktree clean。执行SHA：probe=`eeb297bddf103d4e009e67b4895ccfb584bfa6dbe17c1bd148f26b4fd0e37123`、core=`bdb104ceb82c9f069499dd920b88599a455a65defdd3f622184bbe8dfbe2bd63`、D62 helper=`38ae1114a06d135bca806f470417cd28a634fec0da449888665c6843615d4a20`。00:46:42检查retry2目录不存在；GPU0显存`954/16303MiB`、利用率0%。除`probe-root/script=d73r2wt`和新输出目录外，命令参数与第10节一致。

R2于2026-07-20 00:47:34启动，PID`27204`；只读命令行检查与锁定参数一致，stderr 0B，转为只读监控。

R2于00:50:32完成105/105行、7个artifact和metadata，stderr 0B，probe闭包通过。但在正式汇总前的生命周期审计发现：final metric实际改变，geometry却继承D42的`metric_frozen_during_stage2c=true`及旧final hash；资源也未把2,240,688个D73解析梯度MAC加入`estimated_metric_adaptation_macs`。这不会改变R2预测，却会形成错误的量化生命周期和资源声明，因此R2标记`COMPLETED_OUTPUT_REJECTED_STALE_LIFECYCLE_AUDIT`，其指标不得作为最终采纳结果。

R3只修复审计：对base/updated final log-diagonal分别写SHA并要求不同，更新final hash和Stage2-C非冻结标志，补计metric MAC并在sanitize中对称回滚，verifier同时硬检查hash、标志与资源等式。core、metric delta、D62 refit、int8编译和预测路径均不变；使用新clean worktree与`conflict_projected_joint_metric_retry3`。

R3验证与运行锁：专项9/9、主工作树D42–D73完整链377/377均通过（82.6秒）；clean worktree`E:\type10-7\code\snapshots\d73r3wt`锁定commit`7c02518e3365f102c1bde82c15473fef6f5bebbe`，专项9/9及core/probe/summarizer `py_compile`通过且worktree clean。执行SHA：probe=`8367f7fba9617067e56ed69157828ed7091a0b04f0c51f8905ec7056fd832802`、core=`bdb104ceb82c9f069499dd920b88599a455a65defdd3f622184bbe8dfbe2bd63`、D62 helper=`38ae1114a06d135bca806f470417cd28a634fec0da449888665c6843615d4a20`、summarizer=`1cf357df1a15944a921fc6f7c9d68787fa67300007ed24c7ef2a6d8205551e3e`。01:01:03检查retry3目录不存在；GPU0显存`954/16303MiB`、利用率0%。除`probe-root/script=d73r3wt`和新输出目录外，命令参数与第10节一致。

R3于2026-07-20 01:02:00启动，PID`1332`；只读命令行检查与锁定参数一致，stderr 0B，转为只读监控。

## 13.R3完成状态与artifact闭包

- PID`1332`于01:05:22前退出，Runner实测`elapsed_seconds=172.5560`，launcher stderr为0B。
- RECEIPT=`DEVELOPMENT_SUPPORT_ONLY_DIAGNOSTIC_NEGATIVE_NOT_PROMOTABLE`，`selected_candidate_id=Z0_SUPPORT_ONLY`，`selected_positive_route=false`，`query_opened=false`；无formal/performance claim。
- 闭包：105/105行、7候选×15fold、30个目标量化/FP32行、30次top fit、30次额外D62 final refit、1620次D62 residual component execution；目标行21/21完整训练步，ground/query-fit/clean/source/query-role/quota访问0。
- lifecycle闭包：15/15个INT8 fold的base/updated metric SHA不同，final SHA均等于updated SHA；`metric_frozen_during_stage2c=false`、`stage2c_log_diag_frozen=false`。metric MAC等式为`4,976,640+2,240,688=7,217,328`。
- 完整摘要：`d73_r3_full_performance_summary.json`，87,448B，SHA256=`8810fb11322a83bc8f745d0d29c3cb4a51063d1e3fd587d25be8be97dc356cf9`。

|artifact|字节|SHA256|
|---|---:|---|
|`training_log.jsonl`|14,774,335|`e111126fed3de1dc92ad7845dffd5bce42439a69012a96c8c71cbe210db9fbe3`|
|`support_audit.json`|313,672|`b4cd3dacb1d25bc566b0621950cd6c5567815b9cea779cb05dc4a3b90803a3b4`|
|`resource_audit.json`|6,498|`00f364e567e0462feb955321e6d414c0dc95493dab35d3ed00dfacd71a8d6e2b`|
|`geometry_audit.json`|5,132|`ae4b735a45fdae38eaf8bb6bfd23e57df9d60560144027a1e3e33937556300dc`|
|`selection.json`|2,992|`0bc6d11f7e1a2c26e765eefb58e12e172c6cb6b453de23f5e62be25a574173df`|
|`RECEIPT.json`|5,030|`1f3b8bb0096f3516d4cc866c22c4316f680c84999befeb1b7618160f09d6c70f`|
|`D73_PROBE_METADATA.json`|2,485|`d10486306c008cee30fcd4e45c46de5251ab78b12f4e8fb8ec06744e99e561a5`|

## 14.同row总体结果与开发门

|candidate|机制|receiver/TX|K/seed|B old|A old|seen-new|unknown|H|forgetting|joint|min-B/A/N|row-floor B/A/N|混淆O→N/N→O/N→N|coverage/rollback/defer|量化|资源|判定|
|---|---|---|---|---:|---:|---:|---|---:|---:|---:|---|---|---|---|---|---|---|
|D73 INT8|等权旧/新梯度单步共享metric＋D62 refit|20-1/new5|K10实际fit K8/713101|92.78|82.22|84.67|N/A，本开发单元无unknown query|82.62|10.56|26.67|80.00/53.33/73.33|73.33/50.00/46.67|23/8/15|N/A/0/0|INT8=FP32 argmax|见第21节|负向，不晋级|
|D62 INT8|冻结D42 metric＋crossfitted Fisher row splice|20-1/new5|同上|92.78|82.22|84.67|N/A|82.62|10.56|26.67|80.00/53.33/73.33|73.33/50.00/46.67|23/8/15|N/A/0/0|INT8=FP32 argmax|更低|当前最强|

D73与D62的15/15 outer prediction SHA完全相同，所有总体、场景、类、fold、floor和混淆指标逐项相同；D73开发正向门失败。相对活动K10目标：`A`差9.78pp、`min-A`差34.67pp、`new5`差7.33pp。不得运行第二seed、K1/K5/K20或125矩阵。

## 15.三场景表现

|场景|B|A|N|H|F|J|min-B/A/N|row-floor B/A/N|混淆O→N/N→O/N→N|主要表现|
|---|---:|---:|---:|---:|---:|---:|---|---|---|---|
|LEO clear weak|98.33|91.67|98.00|94.44|6.67|50.00|90.00/70.00/90.00|90.00/60.00/90.00|2/1/0|新类接近饱和，但fold1旧类floor为0|
|LEO low elev weak|91.67|78.33|76.00|75.98|13.33|20.00|80.00/60.00/50.00|70.00/60.00/20.00|8/5/7|旧/新双向混淆，new floor明显不足|
|LEO rain weak|88.33|76.67|80.00|77.45|11.67|10.00|60.00/30.00/70.00|60.00/30.00/30.00|13/2/8|旧类侵入新类最严重，min-A仅30%|

## 16.逐类总体准确率

类编号按Runner注册顺序，仅用于报告；算法不含类ID专用公式。

|类|before-old|after-old|遗忘/变化|
|---|---:|---:|---:|
|O1|96.67|90.00|−6.67pp|
|O2|96.67|90.00|−6.67pp|
|O3|96.67|93.33|−3.33pp|
|O4|80.00|53.33|−26.67pp|
|O5|93.33|73.33|−20.00pp|
|O6|93.33|93.33|0.00pp|

|类|seen-new准确率|
|---|---:|
|N1|73.33|
|N2|93.33|
|N3|76.67|
|N4|90.00|
|N5|90.00|

旧类最弱仍是O4=53.33%，新类最弱仍是N1=73.33%。D73没有修复D62的下尾类瓶颈。

## 17.15个outer fold完整同row表

|场景|fold|B|A|N|H|F|J|floor B/A/N|混淆O→N/N→O/N→N|
|---|---:|---:|---:|---:|---:|---:|---:|---|---|
|clear|0|100.00|100.00|90.00|94.74|0.00|50.00|100/100/50|0/1/0|
|clear|1|100.00|83.33|100.00|90.91|16.67|0.00|100/0/100|0/0/0|
|clear|2|91.67|83.33|100.00|90.91|8.33|50.00|50/50/100|1/0/0|
|clear|3|100.00|100.00|100.00|100.00|0.00|100.00|100/100/100|0/0/0|
|clear|4|100.00|91.67|100.00|95.65|8.33|50.00|100/50/100|1/0/0|
|low|0|100.00|66.67|80.00|72.73|33.33|50.00|100/50/50|4/1/1|
|low|1|83.33|58.33|70.00|63.64|25.00|0.00|50/50/0|1/0/3|
|low|2|83.33|91.67|70.00|79.38|−8.33|0.00|50/50/0|0/2/1|
|low|3|100.00|100.00|70.00|82.35|0.00|0.00|100/100/0|0/1/2|
|low|4|91.67|75.00|90.00|81.82|16.67|50.00|50/50/50|3/1/0|
|rain|0|83.33|83.33|60.00|69.77|0.00|0.00|50/50/0|2/0/4|
|rain|1|100.00|66.67|90.00|76.60|33.33|0.00|100/0/50|4/1/0|
|rain|2|91.67|83.33|80.00|81.63|8.33|50.00|50/50/50|1/0/2|
|rain|3|83.33|75.00|90.00|81.82|8.33|0.00|50/0/50|3/0/1|
|rain|4|83.33|75.00|80.00|77.42|8.33|0.00|50/50/0|3/1/1|

## 18.机制激活、训练与为什么没有降低遗忘

- 15/15个fold均生成不同的metric方向；`||delta||2=0.164399`，RMS=0.009687，最大绝对坐标变化均值0.076769。base/updated state SHA均不同，机制真实激活。
- 旧/新任务梯度余弦全为正，范围0.7265–0.8532、均值0.8002，因此设计中的冲突投影分支0/15次激活；实际执行的是两个一致方向的等权平均。
- leave-one prototype CE在support内同时下降：旧类均值1.53155→1.51140（−0.02014），新类1.62904→1.61128（−0.01776）。旧类prototype support准确率68.61→68.89%，新类保持61.83%。这证明优化目标被执行，但support代理改善没有转化为outer性能。
- D42 Stage2-B训练完整保留：epoch1 loss=1.031996、support acc=95.14%、grad=1.08376；epoch20 loss=0.102685、support acc=100%、grad=0.13535。epoch21仅使用support，query rows=0。
- D73 final D62 gate为3/15 active、共接纳7行；D62原始为3/15 active、接纳6行。新增1个row没有改变任何outer argmax。
- 失败机理是表示重参数化被后续D62重新拟合基本吸收：共享对角metric改变了support内原型距离，但统一LDA/自动shrinkage在新坐标中重估系数，最终15fold判别边界argmax与D62完全一致。因而`F=10.56pp`没有下降，更不存在地面原型带来的遗忘收益。

## 19.与近期版本的matched比较

|比较|ΔB|ΔA|ΔN|ΔH|ΔF|ΔJ|Δmin-B/A/N|prediction hash变化|混淆ΔO→N/N→O/N→N|解释|
|---|---:|---:|---:|---:|---:|---:|---|---:|---|---|
|D73−D62|0.00|0.00|0.00|0.00|0.00|0.00|0/0/0|0/15|0/0/0|完全等价但资源更高|
|D73−D72|−0.56|−0.56|+2.00|+1.03|0.00|0.00|0/0/+3.33|5/15|+1/−3/0|恢复D62的新类，但不是新进步|
|D73−D71|+1.67|0.00|+0.67|+0.29|+1.67|0.00|−3.33/0/0|1/15|0/−1/0|更高B导致F数值更大，A未改善|
|D73−D61|+2.78|−1.11|+8.67|+3.67|+3.89|0.00|+3.33/−6.67/+30.00|15/15|+5/−8/−5|D61低F来自旧类保护并牺牲新类；D73仍不达目标|

## 20.量化表现

- D73 INT8与matched FP32的before outer、final outer、before support、final support argmax变化均为0；margin符号翻转0。
- 最大score绝对量化误差：fold最小0.000519、均值0.000867、最大0.001762。
- 最差margin分布仍含明显负值：old-new最小−2.0896、new-old最小−4.8601、new-new最小−1.2095。量化不是失败原因，边界本身仍混杂。

## 21.资源表现

|资源|D73|D62|增量/说明|
|---|---:|---:|---|
|trainable parameters|2,016|2,016|峰值不增加；D73瞬时metric维度288|
|optimizer steps/epochs|21/21|20/20|+1 Stage2-C步|
|closed-form component fits|108|72|+36，增加50%|
|LDA fit MAC|35,811,735,552|18,000,009,216|+17,811,726,336|
|metric adaptation MAC|7,217,328|4,976,640|+2,240,688|
|total adaptation MAC|46,145,052,306|24,891,223,970|+21,253,828,336，增加85.39%|
|query MAC|6,624|6,624|额外0|
|persistent/registry state|8,583/941B|8,583/941B|额外0|
|peak CUDA memory|22,886,912B|22,886,912B|不增加|
|dense query graph|0B|0B|通过|
|ground int8 component input|0|0|D22未获正式资格|

D73满足≤80k参数、≤30epoch、≤50step、≤256KB状态和无dense query graph的资源上限，但在性能完全等价D62的同时增加85.39%适配计算，因此被D62严格支配。

## 22.缺陷、结论与下一步

1.核心缺陷不是任务冲突：旧/新梯度高度同向，PCGrad从未激活；继续扫描投影顺序或冲突阈值没有依据。
2.核心缺陷是目标错位/重参数化吸收：support prototype CE对metric敏感，但D62 refit把这类变化吸收，outer边界不变。
3.旧类O4/O5和弱场景仍是主要遗忘源；D73没有使用地面压缩原型，因为D22当前不具正式资格。D66已经验证合法读取84个地面int8单元也为负，不能为降低F而越过协议。
4.停止D73的步长、温度、PCGrad顺序、任务权重、多步、rank、场景/类/角色门和第二seed；不运行125。
5.下一轮应避开“共享可逆metric＋重新拟合统一LDA”的等价类，直接研究不会被头部refit抵消的注册竞争信息，例如以D62固定头为教师、在类对称且query零开销条件下编译一个受约束的非可逆低维竞争残差；仍须同时约束old/new outer代理并保持ground输入0。

最终判定：`COMPLETED_DIAGNOSTIC_NEGATIVE_NOT_PROMOTABLE`。当前最强协议合法开发版本仍为D62，而不是D73。
