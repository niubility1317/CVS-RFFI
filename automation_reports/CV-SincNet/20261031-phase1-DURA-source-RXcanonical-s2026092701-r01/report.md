# DURA 原头保护 RX 特征参考的单阶段对照

- run_id：`20261031-phase1-DURA-source-RXcanonical-s2026092701-r01`
- group_id：`phase1-dura-source-head-protected-RX-reference`；类别：`cvs`；阶段：`Phase1`
- 配置与矩阵：[experiment.json](experiment.json)；状态记录：[events.jsonl](events.jsonl)
- 当前登记状态：RUNNING（实际状态按events.jsonl及独立证据更新）

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
