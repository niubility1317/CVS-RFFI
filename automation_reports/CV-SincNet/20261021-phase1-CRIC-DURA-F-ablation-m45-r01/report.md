# CRIC机制消融：15变体×3模型seed

- run_id：`20261021-phase1-CRIC-DURA-F-ablation-m45-r01`
- group_id：`phase1-cric-dura-complete-system`；类别：`cvs`；阶段：`Phase1`
- 配置与矩阵：[experiment.json](experiment.json)；状态记录：[events.jsonl](events.jsonl)
- 当前登记状态：PLANNED（实际状态按events.jsonl及独立证据更新）

## 目的与对照

F0原生33L17U固定11100步，15个机制设置，三模型seed；接同一DURA并给raw读出，全部七景预测完成后独立truth-last。

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

## F源队列已核实启动

**VERIFIED / RUNNING**：实际源代码1172e25b；8张GPU各1条本任务，checkpoint/source检查PASS，首轮222步与日志增长已观察。当前首批8条为F机制消融，150主F位于同一队列，三seed均已登记。其他健康任务保留。G35完整谱系、G21学生、自动228行七景预测/评分与独占时延为独立后续阶段，尚未以本次F启动宣称完成。

**方法范围纠正**：本run为误选的旧33L17U配方。只撤下179个未启动行；8健康行和已完成产物保留为历史对照。最终T1-E5013L37U另用不可覆盖run，不把旧结果改标T1。
