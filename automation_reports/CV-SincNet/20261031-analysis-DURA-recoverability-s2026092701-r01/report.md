# DURA 既有全量 IQ 保真与识别恢复程度分析

- run_id：`20261031-analysis-DURA-recoverability-s2026092701-r01`
- group_id：`phase1-dura-existing-IQ-recoverability`；类别：`analysis`；阶段：`Phase1`
- 配置与矩阵：[experiment.json](experiment.json)；状态记录：[events.jsonl](events.jsonl)
- 当前登记状态：PLANNED（实际状态按events.jsonl及独立证据更新）

## 目的与对照

复用default/noVar/RXcanonical及raw已完整七景输出，CPU独立复算NMSE分布、逐包阈值、尾部和理想复标量校正。

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


复用已完成输出的独立CPU评分分析，不新增预测、模型加载或更新。全七景逐包 IQ 配对只在评分端读取。能量拟合分数、识别保留率与不可认证的真实极限分开，不以 oracle 校正当作部署或识别率上界。
