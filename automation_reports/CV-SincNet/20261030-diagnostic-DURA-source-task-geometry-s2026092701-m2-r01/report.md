# DURA源F判别保持与特征梯度诊断

- run_id：`20261030-diagnostic-DURA-source-task-geometry-s2026092701-m2-r01`
- group_id：`phase1-dura-source-task-geometry`；类别：`diagnostic`；阶段：`Phase1`
- 配置与矩阵：[experiment.json](experiment.json)；状态记录：[events.jsonl](events.jsonl)
- 当前登记状态：ANALYZED（实际状态按events.jsonl及独立证据更新）

## 目的与对照

FrozennoVar/Fguide010sourcefinals;source512x2views;Fclean-choiceagreement/margins/cos/IQ andweightedguide/basegeometry;0updates/TX/V/target

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

## 完整源诊断与预定决策

VERIFIED/ANALYZED_SOURCE_ONLY。CPU2/0GPU约37.791秒，两个source-final模型×同512源位置×cached01/20，64个B32批次全部完成。每模型先实际两Ls G/F checkpoint smoke，全部F/G状态不变、p.grad未积累；0优化器/TX真值/V/target。执行commit fa581bf3bff2e372551d9c1196af8b5695dddaeb，一个独立P0/P1审查PASS，手算teacher选择/margin检查1 PASS；一次N607 preflight、归档两端SHA和compile通过。

首轮回查时该短CPU进程PID4004946已退出；实际launch CWD/argv、两smoke/64批完成日志7230bytes及完整结果/complete已独立回收。未声称直接观测到存活期间PID。原始64批再次核对四组相同512唯一位置、每组L128/U384、范数/点积/余弦/比值共1280个标量，并重新聚合四单元，未重算IQ导数；原始批记录和位置仅存私有artifact，Git只公开汇总。

|源模型|缓存epoch|IQNMSE|F余弦误差|与cleanF选择一致率|cleanF选择margin|加权F/base梯度范数中位比|
|---|---:|---:|---:|---:|---:|---:|
|noVarAux|1|0.02235236|0.01594542|93.359375%|5.928042|0.315966|
|noVarAux|20|0.02112201|0.01140164|96.875000%|6.352800|0.342446|
|Fguide010|1|0.02287066|0.01181898|94.921875%|6.347415|0.245347|
|Fguide010|20|0.02094437|0.00849262|97.070312%|6.707206|0.220445|

Fguide010相对noVar两组源F余弦误差分别约下降25.88%/25.51%，cleanF选择一致率分别提高1.5625/0.1953125pp；IQ误差比为1.023188/0.991590。它们是源teacher保真，不能称真实TX准确率或目标性能。每个缓存epoch重复同物理位置，不构成独立seed。

预定beta0.3资格未通过：当前0.1加权F梯度/base范数中位比0.245347/0.220445，均高于0.2；其他条件虽满足，也不放宽该标准或发布增加系数的训练。模块分解/负cos/null有效数保留results.json，不能只看whole均值。

在读取本轮target正式分数前，源证据已提示：全160维余弦改善未同比改善冻结分类头的判别关系，因此不简单增大同一余弦权重。若后续研究新的判别投影特征约束，应只依据该源现象和固定原头，不按目标TX/场景分数定类权重；先做独立源梯度诊断/固定匹配方案，不能将本结果直接当成新训练或新损失已有效。当前Fguide010全七正式预测按原矩阵完成，不改其权重、预算或测试。目标继续ACTIVE。
