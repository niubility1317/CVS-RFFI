# 修复后新增14个E200模型：立即执行clean及六星地场景独立测试

- run_id：`20261009-evaluation-reference-repair-completed14-manysig-m14-r01`
- group_id：`cvs-reference-repair-completed-evaluation`；类别：`cvs`；阶段：`evaluation`
- 配置与矩阵：[experiment.json](experiment.json)；状态记录：[events.jsonl](events.jsonl)
- 当前登记状态：PLANNED（实际状态按events.jsonl及独立证据更新）

## 目的与对照

用户要求立即给测试结果；固定2026-10-09 09:27:22已完成的14模型，非按性能筛选，固定完成快照，合并既有5行后按共同seed对照。原32行训练与后续固定矩阵不变。

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

## 固定完成快照补测

只属于32行修复实验。固定2026-10-09 09:27:22核实完成且尚未测试的14行，排除此前已评分5行。新增98组预测全部固定后独立truth-last评分；不按指标筛选，不回流原训练或选模。原32行控制器不修改。与旧5行合并后，共19行测试；主对照严格使用八方法共同seed2026092701/02，seed03的三行单独报告。逐行矩阵见experiment.json。一个串行推理worker，GPU可用内存至少12000MiB。Phase2适应和新类注册N/A。
