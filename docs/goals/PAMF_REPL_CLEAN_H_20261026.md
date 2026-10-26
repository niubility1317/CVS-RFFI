# REPL：仅 clean 与直接星地信道 IQ 的研发

## 当前范围

2026-10-26 用户最新要求：“脱离DURA配合路线，这个方向现在不可取，只使用clean IQ以及其叠加星地信道衍生物进行训练。”这覆盖此前恢复视图、G/H混合、G门槛和依恢复分数选方案的研发计划。DURA不进入新训练、源域筛选或候选/seed扩展决策；不加载其权重，不使用恢复IQ/恢复特征/恢复预测。已有恢复实验记录保留为历史，后续独立配对参考评测也不得反馈研发。

保留原完整 PAMF、一个 canonical SIG_TIME、joint160 默认6类头联合训练。源L/U/V及原F0物理ID不变；U TX隐藏、RX/day可见；L使用全部真TX。已完成监督和普通PL对照直接复用，不重复训练。只继承已核实未使用DURA训练/目标选模的同契约singleSIG_TIME ordinaryPL scratchE20，不能继承已做恢复视图训练的学生；各模型独立+10+20到累计E50/11100（13L+37LU）。

## 立即执行的停止

两个本对话恢复视图run的controller已停止，仍活跃的REPL_G_LU、REPL_GH_U、REPL_GH_U_Pair三学生均按精确进程身份停止并独立核实offGPU。REPL_G_U/G_L已完成部分保留；LU未完成，不作三角色完整统计。GH两行只有真实smoke和部分训练，不能作收敛/目标性能结论。0其他owner动作、0删除。停止证据在对应run的user_stop_evidence.json。旧完整三角色audit脚本不再执行，因为LU被用户取消而未完成。

## 两种关系和L/U权限

同一个clean包经过H_a/H_b仍有相同底层TX，无需猜TX便可建立同包关系；RX相同，因此它不能替代跨RX泛化。不同物理包且不同RX的类别正对：L使用真TX，U只使用原clean高置信伪TX。RX已知不能证明两个低置信U同TX，也不能将它们当作已知不同TX。

L不套用U的.95置信丢弃规则，全部真标签CE保留；L是否采用双随机信道视图作为独立输入消融。U的高置信类别CE/浅中joint跨RX，与低置信同包无硬标签一致性分别计算。此前G-G的低冲突诊断只属恢复域证据，不能直接当作H-H的有效性证明；H-H必须用新真实smoke/源域结果检验。

## 新源域候选计划（尚未启动）

先固定同seed01的小矩阵，实际实现/预算/源门槛在新实验的最小预登记里冻结；不把本计划当成已运行。

| 候选 | L信道视图 | U信道视图 | 低置信U |
| --- | --- | --- | --- |
| REPL_H2_U | 保留原As两次真实CE，As KL0 | 原mixed physical半改为H_a/H_b独立场景/实现，ordinary/raw保留 | 不增加分类或一致性 |
| REPL_H2_U_Pair | 同上 | 同上 | 仅同包H_a/H_b detached互概率一致性，不赋硬TX |
| REPL_H2_LU_Pair | 原As位置也改为H_a/H_b，全部真标签、原CE剂量与As KL0不变 | 同上 | 同上 |

跨RX的原ordinary浅/中/joint对保持；不新增F前向、不另训头/EMA，不先叠加AWGN。H2增加的是独立信道生成次数，其成本需实际报告。同包项不对低置信U制造跨包类别关系，不把一致的共同错误当成正确标签；若应用detached概率项，其系数、全半批归一化和与既有guard的整合在运行前固定。保留原监督As/raw/ordinary骨架，避免把全部训练压力换成困难信道。

## 源域选择与实际目标

新源评测只需要raw和直接未恢复H的固定source practical景；不生成/评分恢复视图。已完成的可比源评测取对应raw/H部分复用，缺少的源预测只补推理，不重训控制。预登记完整参照及明确raw/H正确数门槛后才启动候选；按clean优先、H其次、较低成本作源排序，不能沿用旧G优先条件。源V只作不更新状态的验证；无通过不扩seed/目标。

最终仍需同一方法三个F种子2026092701/02/03，原F0每seed168000物理包（cross_rx126000+cross_both42000按正确数合并；cross_date30000另报），raw clean和六种practical固定预测后独立truth-last。相对于历史及严格单视图监督/普通PL中较高者，raw clean至少+1pp、raw六景>+1.5pp；绝对下限raw clean≥85.98174603174603%、raw六景>72.0961970899471%保持。三变体不是三种子，源诊断不是目标达成。此前DURA配对分数不再作为研发目标筛选条件。

训练默认N607，用户允许全部GPU，全球CUDA+pending每卡最多2且free≥10000MiB，包含其他owner，保护健康任务；新run独占路径和Git代码身份，实际sourceL2/联合backward smoke先行，逐模型训练后自动冻结/源评测，符合门槛后再登记三seed和正式测试。当前性能目标未完成。


## 已实现并启动的新路线

run `20261026-phase1-PAMF-SIG1-REPL-CleanH-source-m3-r01` 已按上述三行实现并启动，执行commit `d860ee49a036c599bcb0d7848feed22cc7de8bd7`。新激活器只使用F/AF/BASE/FOLD及当前practical代码；无恢复器root/权重/IQ/分数。10focused测试、compile、一次P0/P1审查和实际E20联合backward smoke均PASS。三学生GPU0/3/4真实运行，独立核实step增长；当时656条可读记录全部检查通过。

固定源门槛为raw≥26547/27000、H3≥61132/81000，复用既有三参照raw/H部分；通过者clean优先，其次H/较低H成本，无通过不扩seed/目标。每行独立+10+20，累计E50/11100；每行训练后自动源4view评测。三行是三变体单Fseed，不是三seed；当前没有新目标结果，原目标保持未完成。详情与当前证据见[新run报告](../../automation_reports/CV-SincNet/20261026-phase1-PAMF-SIG1-REPL-CleanH-source-m3-r01/report.md)。


## 源探索完成，固定胜出方法进入三种子确认（2026-10-27）

`20261026-phase1-PAMF-SIG1-REPL-CleanH-source-m3-r01` 三行 E50 已完成，独立核算 19980 条新增训练记录、三个实际最终权重和 324000 个源域决策。三项都过原门槛；按 raw 优先规则选中 REPL_H2_LU_Pair，源 raw26554/27000、H362190/81000，较固定参照 raw−0.074074pp、H3+1.606173pp。U-only 的 H3 略高，仍遵守原 clean 优先顺序，不据目标改选。

确认 run `20261027-phase1-PAMF-SIG1-REPL-CleanH-confirm-m3-r01` 复用完成 seed01，只训练 seed02/03 各自同契约单视图普通 PL scratchE20+10+20 到 E50。只有 clean 与直接星地 IQ，无 G 权重/IQ/评分；保持完整 PAMF、单 SIG_TIME、原 jointhead。每 seed 主168000原物理样本 raw clean＋六 practical，cross_date30000另报。复用六个严格监督/普通PL对照的主结果，只补它们缺失的 crossdate 原权重预测，不重训、不重复主预测。矩阵2训练/9预测/9独立评分/1真实源smoke。

目标绝对门槛和配对提升要求不变，方法/seed/预算/测试矩阵在访问目标前固定；目标结果不回流调参或变体选择。当前源通过不是三种子目标成功。精确命令、环境/路径、数据句柄、GPU/owner、停止规则和产物见[确认登记](../../automation_reports/CV-SincNet/20261027-phase1-PAMF-SIG1-REPL-CleanH-confirm-m3-r01/experiment.json)。


## 三种子确认失败；独立 L 锚点源试验继续原计划（2026-10-27）

REPL_H2_LU_Pair 三种子主 168000 raw 七景已经全部固定预测并独立 truth-last 评分，21 job 全部完成。clean 为 84.180754±1.087816%，六景为 71.647817±0.474676%；相对单视图普通 PL 分别 −0.513095±1.025406pp / +1.414583±0.842952pp，相对单视图监督分别 +0.063889pp / +1.479762pp。本版未达到双目标，不晋级最终方法，也不因 seed02 正收益单独选取或扩到全 11 设置。两个新增实际 E50 权重及全部 13320 条新训练记录已闭合核算；所有对照主结果复用，没有重训。

clean 损伤主要在占主样本 75% 的 cross_rx：−0.730688pp；cross_both 为 +0.139683pp，独立 cross_date 为 +0.177778pp。信道均值在这些人口分别 +1.222972 / +1.989418 / +1.521481pp。目标侧混淆变化仅用于如实描述失败，不用于弱类重权、调参、候选重排或选择性重跑。源 V 与训练共享源 RX，高精度源通过无法保证未见 RX 泛化。完整表见 [三种子确认结果](../../automation_reports/CV-SincNet/20261027-phase1-PAMF-SIG1-REPL-CleanH-confirm-m3-r01/report.md)。

在读取该完整目标结果前已独立登记并启动 REPL_H2_LU_LAnchorH 源域小试验，run `20261027-phase1-PAMF-SIG1-REPL-LAnchorH-source-m1-r01`，实际执行 commit `26876bca7a75e14690f8af038863c3c8988209f2`。只新增真实 L raw 停梯度锚点对高置信 U/H 的同 TX 跨 RX 类别约束；低置信 U 不建立跨包 TX 关系。真实 E20 检查 PASS，GPU7 学生实际训练、日志增长；没有额外 F/推理参数、G 或目标读取。保持原配置和 source raw≥26568/27000、H3≥62109/81000 门槛，完整 E50 源评估完成后才决定是否另登记三种子正式确认；不受本次目标结果反馈。见 [源小试验](../../automation_reports/CV-SincNet/20261027-phase1-PAMF-SIG1-REPL-LAnchorH-source-m1-r01/report.md)。

持续目标仍 ACTIVE：raw clean≥85.98174603174603%、六景>72.0961970899471%，且较匹配监督及普通 PL 的 clean≥+1pp、六景>+1.5pp。没有因本批严格基线较低而下调门槛；源码、旧模型及负结果保留，后续仅 clean 与直接星地 IQ，不恢复 DURA 配合路线。


## 补齐低置信项的单因素源消融（2026-10-27）

新增单 seed 源小试验 REPL_H2_LU_NoPair，只关闭已有 H2LU 整配方的低置信同包互概率项，L/U 双 H、高置信类别和原跨 RX 学习保留；来自已完成源消融和共同错误分析，不使用目标弱类调权。两项独立源试验在结果产生前冻结共同门槛 raw≥26568/H3≥62109，均完成后在通过者中 raw/H3 依次优先、完全同分选 NoPair；未通过不扩种子或目标。L-anchor 已在训练的原配置不变，监督/PL 控制不重复训练。原绝对及相对三种子目标保持。见 [新源消融](../../automation_reports/CV-SincNet/20261027-phase1-PAMF-SIG1-REPL-H2LUNoPair-source-m1-r01/report.md) 和其中的源选择计划。


## 源完成记录：REPL_H2_LU_LAnchorH

run `20261027-phase1-PAMF-SIG1-REPL-LAnchorH-source-m1-r01` 单seed源E50完整核算完成：raw26548/27000=98.325926%，H362030/81000=76.580247%，未通过固定26568/62109门槛；6660新增训练记录及108000源决策闭合。相对固定H2LU raw-0.022222pp、H3-0.197531pp。仍按两项共同源选择计划，原三seed目标未完成。见 [完整源结果](../../automation_reports/CV-SincNet/20261027-phase1-PAMF-SIG1-REPL-LAnchorH-source-m1-r01/report.md)。


## 下一研究方向：平衡已有语义监督而非追加特征聚拢

L-anchor源失败后，完成冻结源RX/伪类覆盖诊断：组合相对L先验的最低覆盖约0.6697，低覆盖组合的已接受源代理预测仍正确；单纯RX覆盖差异仅约8%。下一轮拟比较RX与RX×伪类别的有界语义梯度重加权，保持raw95、接受总权重及原cap1，不从V初始化训练统计。它是待实现并预登记的新探索，不能宣称已启动或有效；健康NoPair不热改、不重训控制，原三种子目标及当前两项源选择计划不变。见 [诊断与边界](../../automation_reports/CV-SincNet/20261027-phase1-PAMF-SIG1-REPL-LAnchorH-source-m1-r01/source_receiver_coverage_evidence.json)。


## 源完成记录：REPL_H2_LU_NoPair

run `20261027-phase1-PAMF-SIG1-REPL-H2LUNoPair-source-m1-r01` 单seed源E50完整核算完成：raw26552/27000=98.340741%，H362148/81000=76.725926%，未通过固定26568/62109门槛；6660新增训练记录及108000源决策闭合。相对固定H2LU raw-0.007407pp、H3-0.051852pp。仍按两项共同源选择计划，原三seed目标未完成。见 [完整源结果](../../automation_reports/CV-SincNet/20261027-phase1-PAMF-SIG1-REPL-H2LUNoPair-source-m1-r01/report.md)。


## 两项新源试验的共同选择结果

两项均未通过原源门槛，不扩种子或目标，保留负结果。 目标绝对与相对门槛均保持。完整结果见 [共同源选择](../../automation_reports/CV-SincNet/20261027-phase1-PAMF-SIG1-REPL-H2LUNoPair-source-m1-r01/source_selection_result.json)。


## 接收机与伪类别均衡的两项源试验

登记`20261027-phase1-PAMF-SIG1-REPL-RxBalance-source-m2-r01`：REPL_RB_RX/RXTX两方法、各唯一seed01；只将已有高置信语义CE和RDrop按可见接收机/伪类有界重加权，原L与特征对比不变。L先验和实际U频率，不用V初始化；均保持raw95/weight6/cap1，0.5至2权重及接受总质量守恒。原同契约E20独立+10+20，复用已完成对照，源门槛26568/62109及原三seed目标保持。当前登记而未启动，见[完整方案](../../automation_reports/CV-SincNet/20261027-phase1-PAMF-SIG1-REPL-RxBalance-source-m2-r01/report.md)。


接收机均衡两方案已实际启动，执行commit `a659b861f4f22cc0e43c1c2da6fa7df3fd969b29`；真实E20联合检查PASS，GPU1/0各一学生，早期闭合记录验证通过。自动推进E50/源四景，仍无性能结论。详细启动证据见新run报告。


接收机均衡两学生新增第1轮共444条完整记录已核算：RX权重0.9263至1.1451，RX×伪类0.8798至2，接受总权重保持；分类头冲突率组合均衡44.5946%，RX53.6036%，已完成NoPair同轮52.7027%。这是单seed训练方向描述，不是准确率或因果证明；不调整在训配方、源门槛及目标，也不重复既有分支投影路线。见新run首轮诊断。
