# 冻结模型的身份参照与默认头特征梯度一致性诊断

- run_id：`20261026-diagnostic-PAMF-RXReference-alignment-m1-r01`
- group_id：`phase1-pamf-receiver-reference-20261025`；类别：`cvs`；阶段：`Phase1`
- 配置与矩阵：[experiment.json](experiment.json)；状态记录：[events.jsonl](events.jsonl)
- 当前登记状态：PLANNED（实际状态按events.jsonl及独立证据更新）

## 目的与对照

同seed E30/PLControl及三参照F50全部冻结，在12批可见L/隐藏U上比较参照目标与真实默认头soft/hard身份目标的特征梯度；不训练、不读V/target。

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

## 冻结源诊断发布前范围

参见 [最小预登记](preregistration.md)。五个同seed01 reader全部冻结，只在可见源L和隐藏源U上求特征叶节点梯度，不更新F/G/参照或读取V/目标。三项聚焦检查与compile通过，一次独立P0/P1审查PASS；实际L2默认头与特征梯度smoke由launcher先执行。U与冻结伪标签的一致性不能称为真实准确率，梯度相关性也不是目标收益证明。
