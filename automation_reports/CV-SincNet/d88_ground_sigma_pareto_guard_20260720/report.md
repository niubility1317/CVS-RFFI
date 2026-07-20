# D88地面sigma逐类Pareto保护实验报告

## 预注册设计

- 实验ID：`d88_ground_sigma_pareto_guard_20260720`
- 状态：`PLANNED_LOCAL_DEVELOPMENT_DIAGNOSTIC`
- 目标：修复D87虽使注册后旧类准确率由82.78%升至85.00%、遗忘由10.00%降至7.78%，但新类准确率由84.67%降至83.33%的同row冲突。
- 假设：D87的地面半径sigma方向具有真实旧域适应信号；新类退化来自每步只约束聚合smooth-worst sigma风险，允许个别类的clean OOF CE上升。若把同一方向投影到所有已注册类clean OOF CE共同非增锥，再做逐类精确回溯，可保留部分旧类收益并消除新类系统性回退。
- 单一主要差异：相对D87只增加逐类、角色无关的common-descent cone projection与相对未更新D62点的exact per-class line-search guard；地面v2组件、14个方向、半径幅度、sigma权重、20步、rank13空间、D78 trust ball、D79中心化仿射编译全部不变。
- 类对称性：当前row全部11个注册类使用同一公式；不读取old/new角色、class ID、query标签、query组成或class quota。
- 数据边界：复用匹配`VALIDATED_ONCE`的固定单LEO弱观测；反事实sigma view仅为同一received IQ的数学视图，不增加K；不读取clean/source样本。
- 预期可观察结果：全部目标row均满足`max_class_oof_clean_ce_delta<=数值容差`；D87发生变化的4/15个outer row中，clear/fold2的新类损失应被抑制，同时尽量保留low/fold0和rain/fold3的旧类收益。
- 失败/停止条件：若残差15/15全部回退为零，则说明D87地面sigma方向没有全类共同下降空间；若仍有`seen_new_acc<84.67%`或注册后旧类不高于D85的82.78%，本路线不进入seed2/125；不扫描权重、半径或trust参数。
- 最小验证矩阵：先跑核心与probe单测；通过后锁定development seed、K10/new5、3场景×5fold×INT8/FP32 matched的105行完整probe，并报告同row总体、逐场景、15行、逐类、混淆、量化和资源。
- Phase1组件声明：当前v2组件仍为`PHASE1_COMPONENT_PENDING_OUTER_JOINT_SEAL`，因此D88即使性能改善也只能作为forced development diagnostic，不能直接晋升正式确认。

## 版本与执行计划

- 本地Git工作树：`E:\type10-7\code\snapshots\d81wt`
- 核心：`code/cvsrffi/stage2_d88_ground_sigma_pareto_guard.py`
- probe：`code/scripts/probe_d88_ground_sigma_pareto_guard.py`
- 测试：`tests/test_stage2_d88_ground_sigma_pareto_guard.py`、`tests/test_probe_d88_ground_sigma_pareto_guard.py`
- 环境：`ssr-gpu`
- 工作目录：本地开发cell；本轮无需N607，不占用GPU，不创建SSH连接。
- 输出根：`E:\type10-7\automation_reports\CV-SincNet\d88_ground_sigma_pareto_guard_20260720\ground_sigma_pareto_guard_centered_head`
- 预期artifact：`training_log.jsonl`、`predictions.jsonl`、`predictions.receipt.json`、`D88_PROBE_METADATA.json`、完整性能汇总与本报告结果段。

## 最终状态

`COMPLETED_DIAGNOSTIC_OVERCONSTRAINED_GUARD_REGRESSION_NOT_PROMOTABLE`

retry3于2026-07-20 08:48:06 CST启动，PID=`13764`，约140秒完成105/105行；`stderr=0B`，receipt、D88 metadata及全性能账本均通过。共7候选×3场景×5outer fold；目标INT8/FP32各15行、fit使用每类8个inner support，outer held physical rank不参与fit。组件仍为`PHASE1_COMPONENT_PENDING_OUTER_JOINT_SEAL`，即使性能通过也只能是development diagnostic。

## 七候选总体性能

数值均为%；`B/A/N/H/F/J`依次为注册前旧类、注册后旧类、seen-new、同row调和均值、遗忘、joint floor。

|candidate|机制|B|A|N|H|F|J|min B/A/N|row floor B/A/N|o→n/n→o/n→wrong-n|判定|
|---|---|---:|---:|---:|---:|---:|---:|---|---|---|---|
|D42-USLDA-INT8|D88地面sigma Pareto保护|92.78|82.22|84.67|82.62|10.56|26.67|80.00/53.33/73.33|73.33/50.00/46.67|23/8/15|主候选，负结果|
|D42-USLDA-FP32-MATCHED|D88 FP32 matched|92.78|82.22|84.67|82.62|10.56|26.67|80.00/53.33/73.33|73.33/50.00/46.67|23/8/15|与INT8完全同预测|
|B3_SINGLE_IQ_DIAG_FFTRF|B3诊断比较器|87.78|75.56|72.67|73.35|12.22|23.33|80.00/60.00/40.00|53.33/33.33/36.67|33/22/19|弱基线|
|D42-D40-HNBR-INT8-NEGATIVE|HNBR负对照|85.56|85.00|15.33|25.16|0.56|0|66.67/63.33/0|40.00/40.00/0|2/0/0|新类不可达|
|D42-D41-BEC-INT8-NEGATIVE|BEC负对照|86.11|20.56|78.67|31.50|65.56|0|76.67/0/36.67|46.67/0/26.67|142/0/32|旧类崩溃|
|D42-PROTOnet-CDA-ZID160|ProtoNet CDA|71.11|48.33|52.67|48.97|22.78|0|33.33/13.33/3.33|13.33/0/0|0/0/0|弱基线|
|Z0_SUPPORT_ONLY|identity control|71.11|48.33|52.67|48.97|22.78|0|33.33/13.33/3.33|13.33/0/0|0/0/0|选择回退|

## 与D87、D85的matched差异

|比较|ΔA|ΔN|ΔH|ΔF|ΔJ|Δmin-A|Δrow-A-floor|Δrow-N-floor|Δo→n/n→o/n→wrong-n|
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
|D88−D87|-2.78|+1.33|-0.95|+2.78|-3.33|-6.67|-3.33|+3.33|+5/-2/0|
|D88−D85|-0.56|0|-0.31|+0.56|0|0|0|0|+1/0/0|

D88确实消除了D87的新类总体回退，并把clear/fold2从`N=90%`恢复到100%；但同时撤销了D87在low/fold0、rain/fold1、rain/fold3的大部分旧类收益。相对D85仅low/fold0发生离散变化：`A=75.00%→66.67%`，多1次old→new；其余14/15行预测相同。严格逐类support CE保护没有转化为held query保护。

## 三场景性能

|场景|B/A/N/H/F/J|min B/A/N|row floor B/A/N|o→n/n→o/n→wrong-n|相对D85|
|---|---|---|---|---|---|
|clear|98.33/91.67/98.00/94.44/6.67/50.00|90/70/90|90/60/90|2/1/0|完全同指标|
|low-elev|91.67/78.33/76.00/75.98/13.33/20.00|80/60/50|70/60/20|8/5/7|A−1.67pp、F+1.67pp|
|rain|88.33/76.67/80.00/77.45/11.67/10.00|60/30/70|60/30/30|13/2/8|完全同指标|

## 15个outer row

|scene/fold|B/A/N/H/F/J|floor B/A/N|o→n/n→o/n→wrong-n|
|---|---|---|---|
|clear/0|100/100/90/94.74/0/50|100/100/50|0/1/0|
|clear/1|100/83.33/100/90.91/16.67/0|100/0/100|0/0/0|
|clear/2|91.67/83.33/100/90.91/8.33/50|50/50/100|1/0/0|
|clear/3|100/100/100/100/0/100|100/100/100|0/0/0|
|clear/4|100/91.67/100/95.65/8.33/50|100/50/100|1/0/0|
|low/0|100/66.67/80/72.73/33.33/50|100/50/50|4/1/1|
|low/1|83.33/58.33/70/63.64/25/0|50/50/0|1/0/3|
|low/2|83.33/91.67/70/79.38/−8.33/0|50/50/0|0/2/1|
|low/3|100/100/70/82.35/0/0|100/100/0|0/1/2|
|low/4|91.67/75/90/81.82/16.67/50|50/50/50|3/1/0|
|rain/0|83.33/83.33/60/69.77/0/0|50/50/0|2/0/4|
|rain/1|100/66.67/90/76.60/33.33/0|100/0/50|4/1/0|
|rain/2|91.67/83.33/80/81.63/8.33/50|50/50/50|1/0/2|
|rain/3|83.33/75/90/81.82/8.33/0|50/0/50|3/0/1|
|rain/4|83.33/75/80/77.42/8.33/0|50/50/0|3/1/1|

## 逐类性能

|TX|角色|B|D85 A/N|D87 A/N|D88 A/N|D88缺陷|
|---|---|---:|---:|---:|---:|---|
|14-10|旧|96.67|93.33|93.33|93.33|遗忘3.34pp|
|14-7|旧|80.00|53.33|60.00|53.33|最弱旧类，D87收益被全部撤销|
|20-15|旧|96.67|90.00|93.33|90.00|D87收益被撤销|
|20-19|旧|93.33|93.33|93.33|93.33|稳定|
|6-15|旧|93.33|73.33|76.67|73.33|遗忘20pp，D87收益被撤销|
|8-20|旧|96.67|93.33|93.33|90.00|较D85额外下降3.33pp|
|1-16|新|—|93.33|93.33|93.33|稳定|
|1-18|新|—|73.33|73.33|73.33|最弱新类|
|18-10|新|—|90.00|86.67|90.00|恢复D87损失|
|14-11|新|—|76.67|73.33|76.67|恢复D87损失|
|8-3|新|—|90.00|90.00|90.00|稳定|

## 机制表现与缺陷

- 15个INT8 fit中仅6个残差激活，9个因无全类共同clean-descent方向而精确回退D62；20步中平均12步为零方向。D87为15/15激活且残差范数均约`1.03..1.26`，D88残差范数均值仅`0.07263`、最大`1.03364`。
- 全15行逐类clean OOF CE相对D62基线均不升；最大类增量仅`2.16147e-11`，位于对应固定数值容差内。sigma目标单调，目标下降范围`−0.02410..0`、均值`−0.001701`。
- 投影开销并不产生足够信息：每fit半空间投影计数`20..9100`、均值4576；OOF support预测0/15变化，outer仅相对D85改变1/15且方向为负。
- 根因不是没有利用地面原型，而是保护代理过强且错位。D88使用D85全部14个domain、84个cell、p90半径、rank13 sigma几何，但逐类CE共同下降锥在11类K8支持集上通常为空或极窄；即使合法support CE不升，也不能保证held query的离散margin不翻转。
- 零残差回退点是D62而非D85；low/fold0因此失去D85/D81的稳健中心收益。这说明不能把D81成功的support可靠性机制替换成head侧安全门。

## 量化与资源

|项目|D88结果|判定|
|---|---:|---|
|INT8/FP32 outer argmax变化|0|通过|
|INT8/FP32 support argmax变化|0|通过|
|margin sign flip|0|通过|
|最大score绝对误差|0.001915|通过|
|Stage2-C新增MAC|307,983,944|总适配的1.222%|
|D88相对D87新增Pareto MAC|22,496,320|仅support侧|
|总适配MAC|25,199,207,914|资源门通过|
|query MAC/额外MAC|6,624/0|单次独立评分|
|持久状态|14,399B|5,816B组件＋8,583B head|
|参数/peak参数|2,159/2,159|≤80k|
|总optimizer steps/Stage2-C|40/20|≤50/20|
|peak CUDA/dense query graph|22,886,912B/0|通过|

D88的query开销、状态量和量化均合格，但新增22.50M support MAC只得到更弱性能，效率Pareto失败。

## D86–D88三轮技术复盘

已重新读取活动目标与`项目.md`，刷新conversation index至1,008条，并复核D80–D88报告、D86–D88完整105行日志与汇总。

|轮次|地面v2用法|正信号|决定性失败|结论|
|---|---|---|---|---|
|D86|p90半径定义反事实中心平移|状态14,399B、无optimizer|15/15预测不变，FP32/INT8出现1次翻转|共享中心方向关闭|
|D87|14个半径sigma点直接优化rank13 head margin|A+2.22pp、F−2.22pp、min-A+6.67pp|N−1.33pp、new floor−3.33pp|地面边界信号真实，但old/new交换|
|D88|所有注册类clean OOF CE共同非增锥|恢复D87的新类损失，量化稳定|9/15零更新；相对D85 A−0.56pp|逐类硬安全门关闭|

三轮共同结论：压缩地面原型最有用的仍是“support样本可靠性”，不是共享target平移、query协方差或head强正则。D80/D83已否定地面协方差直接进入判别度量；D82已否定逐样本Wiener残差；D84–D86已否定共享中心；D87–D88证明head侧边界信号存在但无法用CE硬门同时保护新旧类。没有出现clean/source/query truth/role/quota访问；单物理样本单LEO观测、physical-rank OOF和query独立全类评分均保持。

下一轮D89锁定为`v2 radius-reliability Cauchy support center`：保留D81唯一通过联合开发门的“类内support可靠性加权”框架，但把D81旧v1地面谱替换为D85的5,816B v2重构，并用全部84个p90半径对domain×class残差做固定可靠度加权，再形成类无关谱。target-old/new仍按同一Cauchy公式重估中心，query继续单INT8 affine；不进入D80/D83协方差、不做D82样本残差、不做D87/D88 head优化。单一目标是同时复现D81的`A=82.78,N=84.67`以上联合性能并保留D85的57.66%总状态压缩；若N、任一floor或混淆回退即淘汰，不扫描radius权重。

## 证据与版本

- worktree实现提交：`575a9a16`、`3bfc7a8e`、`0b3bf992`、`a4ab2e09`、`d87f2fe9`；主Git承载面对应`4347165d`、`d18b50f5`、`7e45548f`、`72ebab8c`、`8be8fcc4`。
- core SHA256=`a699133cac451a594e7724ecf8a6e845a7411a5323d6e25d4779108566d89487`；probe SHA256=`11b3b08d57c8fac234c68503d16e805b12879173fb3694f86e3c9328fd97552a`。
- receipt SHA256=`aa5bc86ce53085d243a43af2cbe0e2c47d9849c5e5e42806b09c18e621d9f26c`；training log=`1e3c2565e0e480592bfab22e9a759edf97ae7f32cb43d066c271234ebb47ceb7`；D88 metadata=`bdecdb3ec63b3b6f20fb3c72916c5d19cd3e4ff01ce1dbffe66811370d6a8266`；full summary=`acffc08d68a5589cc2c30c0db35125aefe625a11fdc2549e9bb3c2d346eb9ffd`。
- 根目录报告镜像：`E:\type10-7\automation_reports\CV-SincNet\d88_ground_sigma_pareto_guard_20260720\report.md`。

## 尝试记录

- attempt0于2026-07-20 08:35:42 CST启动，PID=`12868`，在首个目标fit的最终Pareto审计前退出，training row=`0`。stderr显示逐步数值容差累计后超过更小的最终容差；属于验证器/数值闭包失败，不是性能结果。
- 直接修复：每步精确验收统一改为相对初始D62逐类clean OOF CE上界，而不是相对上一步并累计容差；最终审计复用同一个固定容差。未改变数据、地面组件、目标函数、方向投影、步数、trust或实验矩阵。
- retry1于2026-07-20 08:38:53 CST启动，PID=`23780`，133.86秒完成105/105行并生成receipt；最终验证器仍按“相对上一步严格非增”检查旧trace字段，与已锁定的“相对初始D62不升”定义不一致，故D88 metadata未生成。完整105行与日志原样归档，不作为已闭包结果。
- 直接修复：新增每步`clean_ce_max_class_delta_vs_initial`审计字段，验证器逐步检查同一个D62基线上界；不修改任何模型计算、预测、资源或数据路径。retry2须重新跑完整105行，使source closure包含修复后的核心/probe哈希。
- retry2于2026-07-20 08:43:28 CST启动，PID=`25300`，约136秒完成105/105行。30个INT8/FP32目标row中仅4行的审计布尔值为false；实际`delta-tolerance`仅为`5.05e-17`或`1.05e-16`，模型端最终不变量使用`final<=initial+tolerance`已通过，但audit布尔值使用数值上不完全等价的`final-initial<=tolerance`，产生浮点消减差异。完整目录归档。
- 直接修复：audit布尔值改为复用模型端完全相同的比较表达式。模型参数、预测、优化、阈值和协议均不变；retry3用于得到同源闭包artifact。
