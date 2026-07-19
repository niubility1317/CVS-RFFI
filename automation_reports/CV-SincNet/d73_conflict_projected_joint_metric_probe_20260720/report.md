# D73新旧任务冲突投影联合度量实验报告

## 1.实验身份与当前状态

|字段|值|
|---|---|
|实验ID|`d73_conflict_projected_joint_metric_probe_20260720`|
|候选|`conflict_projected_joint_metric`|
|时间|2026-07-20|
|operator|Codex `/root`|
|状态|`PREREGISTERED_NOT_RUN`|
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
|D73|等权PCGrad单步共享metric|20-1/new5|K10/713101|待跑|待跑|待跑|待跑|待跑|待跑|待跑|待跑|待跑|待跑|待跑|

## 9.实现锁定（2026-07-20 00:24）

|文件|用途|SHA256|
|---|---|---|
|`code/cvsrffi/stage2_d73_conflict_projected_joint_metric.py`|解析leave-one prototype CE梯度、等权对称冲突投影、唯一一次metric更新与审计|`8d59e1c92887930cecebbd59e534e572299a64b875d1add559238eb9177391d5`|
|`code/scripts/probe_d73_conflict_projected_joint_metric.py`|D62包装、统一int8编译、资源/训练/source闭包和真实Runner入口|`c2086e86d23119acb6253a9bb365b1e43c2effc27fdfe4f4eea867b1b995f5a0`|
|`tests/test_stage2_d73_conflict_projected_joint_metric.py`|确定性、任务安全、组内类置换、K1和fail-closed测试|`623d15ebdad5e4c93c6a2c8dcf148c56019113ac204240785940c78cfceb7b64`|
|`tests/test_probe_d73_conflict_projected_joint_metric.py`|D62包装、资源公式、调用闭包和协议字段测试|`f769d331a0fed1a75055abea2fce7f44ac12bfba4ed7a411fa0990eb175d4d79`|

专项验证命令：`python -m pytest -q tests/test_stage2_d73_conflict_projected_joint_metric.py tests/test_probe_d73_conflict_projected_joint_metric.py`，在`ssr-gpu`中通过`9/9`。首次core测试收集失败仅因测试误写为`code.cvsrffi`导入路径；改为项目既有`cvsrffi`包路径后通过，不涉及机制、公式或运行代码变化。

D42–D73相邻完整链覆盖40个文件、377项测试，全部通过，退出码0，用时82.8秒。pytest在全部测试完成后的atexit清理阶段对全局`pytest-current`临时链接报`WinError 5`，不影响退出码或任何测试结果；clean worktree复验将显式使用仓内`--basetemp`消除该主机清理噪声。
