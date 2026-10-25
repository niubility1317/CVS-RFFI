# 任务强度缩放参照的无更新整网源剂量校准

- run_id：`20261026-diagnostic-PAMF-TaskScaledRef-dose-m1-r01`
- group_id：`phase1-pamf-task-scaled-reference-20261026`；类别：`cvs`；阶段：`Phase1`
- 配置与矩阵：[experiment.json](experiment.json)；状态记录：[events.jsonl](events.jsonl)
- 当前登记状态：PLANNED（实际状态按events.jsonl及独立证据更新）

## 目的与对照

seed01 own E30上保留完整T1 L/rawPL，12源L/U批匹配G普通参照/G任务缩放/raw任务缩放；真实默认头可微分路径，整网物理梯度共同剂量，无optimizer/V/target。

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

## 源校准发布前核实

详见 [最小预登记](preregistration.md)。原L/rawPL主项保持，同输入普通G参照/G任务缩放/raw任务缩放三分支完整12批，依据整网物理梯度给出一个共同剂量；没有新训练或目标访问。两项新检查、基础关系/任务缩放九项检查与compile通过；一次独立P0/P1审查PASS。实际L2默认头/特征图smoke通过后自动继续，不把函数检查当作性能结论。
