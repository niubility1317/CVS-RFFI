# DURA 原头保护 RX 特征参考的单阶段对照

- run_id：`20261031-phase1-DURA-source-RXcanonical-s2026092701-r01`
- group_id：`phase1-dura-source-head-protected-RX-reference`；类别：`cvs`；阶段：`Phase1`
- 配置与矩阵：[experiment.json](experiment.json)；状态记录：[events.jsonl](events.jsonl)
- 当前登记状态：ANALYZED（实际状态按events.jsonl及独立证据更新）

## 目的与对照

G从零单阶段E20/4940，固定源RX几何.25参考收缩及梯度校准权重.019528750294704578；同冻结F原头，自动全七景truth-last。

## 数据、seed与模型来源

实际数据契约、权限例外、完整seed角色、checkpoint来源和选择规则见experiment.json。
逐行配置通过config_ref/resolved_config_ref定位；待补项必须在对应生命周期补齐。

## 执行与存储

命令、环境、CWD、commit、launch owner、输出和日志路径见experiment.json。
实际PID/GPU、读取时间、remote readback、失败或替代关系在此追加，并用record命令记录证据指针。

## 结果与覆盖

尚无结果。按预登记artifact逐项记录路径和缺项；保留每row与RX/day/TX/scene/K/seed的对应关系。
源域训练完成、预测完成、评分完成及协议有效性分别陈述。不得用总索引或旧状态证明当前运行。

## 交接

记录已完成、当前run/commit、证据路径、阻塞与下一步；恢复先查原run，不重复启动。


## 最小研究说明

固定已通过的源RX方向（aee80ff）和输出梯度校准（c5a6f6c），源特征参考收缩.25，唯一损失权重.019528750294704578。它与旧Fcos.1同时改变参考和系数，不作为孤立参考消融。G保留noVar四项基础目标、原63000 L/U/20缓存/15%clean、从零E20/4940，coarse/action/prior全部联合训练，无恢复器祖先。配对clean IQ和received保持原样，不进行波形相位规范化或频谱增强。

F仍是原F50/defaulthead，权重、状态与头冻结；每次G训练访只一次G输出、一次F restored特征，63000条sourceclean特征缓存一次含成本。源特征经原头零空间的RX方向收缩后作固定参考，不读真实TX、不生成硬伪TX，V/target不参与训练。拟合RX方向的62488源记录排除诊断512，但G仍按原63000全源训练；这不是G留一RX训练或未见RX泛化的既成证明。F上游50轮/11100源监督知识明确继承，非F-free。

首步真实2Ls source模型检查与严格G重载/原F头/参考对应/梯度通过即继续。GPU4/6/7择一加1个本任务worker，准入前最多2个已有computePID、空闲至少12000MiB；记录准入快照并如实记录其他owner后续新增，不停止或修改他人任务。最终E20自动profile、source1800×7、target168000×7；逐角色全七景预测固定后独立truth-last，不以源改善替代正式收益或因源负结果取消测试。完整3pp、clean、R17容忍及实际成本分别验收，默认DURA保持。


## 启动核实

VERIFIED / RUNNING。执行commit `fb795cebeb21d3cbd97aad05aecfa1afff441018`，8项聚焦检查及本候选唯一P0/P1审查PASS，一次N607预检、单归档SHA及远端compile通过。实际controller956189和worker957861/GPU7的PID/CWD/完整argv/CUDA独立读回，源2Ls真实feature到IQ/G梯度、F不变、G严格重载、原头及参考一致smoke全部PASS。

准入快照显示已有2个computePID、剩余22382MiB，满足新增后最多3的准入规则。独立后态当前同卡共4个computePID；其他owner后续任务可能改变共享负载，不将准入快照写成持续并发保证。只有本任务1个worker，其他任务保持。source63000特征缓存耗时3.390798秒，实际worker日志583字节继续增长。固定从零E20后自动全七景，尚无正式成绩，不把启动或源资格当成3pp/clean/效率完成。


## 完整训练与正式七景结果

VERIFIED / ANALYZED。完整20条epoch/4940更新，L126000/U1134000曝光，G从零单阶段、0恢复器祖先/直接TX/V/target训练。固定.25源RX特征参考收缩和source输出梯度校准权重.019528750294704578实际启用，1260000次source参考访问完整。F参数/状态不变、更新0、原头拟合0；sourceclean特征63000缓存一次，restoredF1260000输入继承原F50源监督知识，非F-free。完整20轮固定加权loss逐项复算一致。执行commit `fb795cebeb21d3cbd97aad05aecfa1afff441018`。

source1800×7和target168000×7分别全部预测固定后独立CPU truth-last，再次独立复算126个总体/TX/RX/day、MacroF1及纠错损伤计数一致，完整分组见results.json。完整日志无Traceback/OOM/数值失败，控制器与worker自然退出。该基准此前暴露、单seed，不声称新的独立盲测或多seed稳定收益。

|场景|准确率%|相对noVar pp|相对cos0.1 pp|相对head投影 pp|相对默认DURA pp|距raw clean pp|
|---|---:|---:|---:|---:|---:|---:|
|clean|84.866667|+0.124405|-0.037500|-0.035714|+0.188690|-0.190476|
|practical_high|81.597024|+0.692857|+0.898810|+1.656548|-1.227976|3.079167|
|practical_mid|80.226786|+0.820238|+0.925000|+1.710714|-1.333929|4.449405|
|practical_low_suburban|78.798214|+0.929167|+0.960714|+1.672619|-1.260119|5.877976|
|practical_high_urban|80.604167|+0.605357|+0.873214|+1.469643|-1.141667|4.072024|
|practical_mid_urban|74.482143|+0.089286|+0.736310|+0.872024|-1.004167|10.194048|
|practical_low_urban|68.417857|-0.052976|+0.590476|+0.541071|-0.472024|16.258333|

六景3pp验收=False，clean不低于raw=True，原R17逐景最多下降1pp容忍=True，分别报告，默认DURA保持。reference与lossweight同时不同于旧cos.1，不能把目标差异单独归因RX参考；source几何及输出梯度资格不是身份/正式成功证明，结果不回流子空间/收缩/权重/轮次或选择性重跑。

训练循环与保存1150.663秒，源准备5.377秒（含F特征缓存3.391秒），G完整训练流程1156.041秒/19.267分钟；控制器启动至完整分析1580.748秒/26.346分钟，目标七景预测283.077秒。源RX几何和梯度诊断、原F50训练及原模拟缓存成本另计；共享GPU跨时段耗时不作单一方法因果加速解释。

仍266951参数/2prior/4CG/27FFT，无新增推理分支，部署不读RX参考或目标元数据；同卡G/G+F测量见matched_profile。源V结果仅预定1800条面板，不能扩为整个V。准入已有2PID/free22382MiB，后续其他任务加入后的同卡4PID事实保持，本任务始终1个worker，不修改他人工作。
