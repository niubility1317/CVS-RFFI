# 外部对照 Phase1 七场景同样本补测

- run_id：`20261031-phase1-external-matched-nodura-m40-r02`
- group_id：`phase1-external-group1-matched-seven-views`；类别：`comparison`；阶段：`Phase1`
- 配置与矩阵：[experiment.json](experiment.json)；状态记录：[events.jsonl](events.jsonl)
- 当前登记状态：PLANNED（实际状态按events.jsonl及独立证据更新）

## 目的与对照

复用10种对照各4个本轮源权重，使用PAMF原同一168000样本及固定七场景，只进行原生冻结直接识别、不接DURA；预测后独立truth-last评分，逐接收机及种子汇总。 用户要求立即启动，扩展GPU总并发至每卡4（含其他owner），本run最多1。

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

## 直接启动计划

用户要求补测直接启动。本run替代仅排队的r01，10方法×4seed、PAMF原168000样本及clean＋六practical，冻结原生分类、无DURA，所有视图预测固定后独立truth-last；逐RX/day/TX汇总。切换前核实r01 PID、命令及零活跃/零完成，精确停止本任务controller，保留原记录和健康其他owner。

每GPU本任务至多1个预测进程，所有owner的CUDA与待启动PID计入总数；新进程启动前总数必须小于4、剩余显存不少于12000MiB。Phase2仍运行时须先核实该卡两个已存活Phase2预测worker均在本次GPU快照中，以避免旧快照与Phase2补位交错；CPU评分最多1。其他owner进程不停止、不热改。首次真实源权重无query smoke通过后自动启动，技术失败停止新预测并保留产物。

独立P0/P1审查及一次定点复核指出Phase2补位与GPU快照次序窗口，已要求两个Phase2 worker PID都属于同一快照，并以针对性交错测试验证。
