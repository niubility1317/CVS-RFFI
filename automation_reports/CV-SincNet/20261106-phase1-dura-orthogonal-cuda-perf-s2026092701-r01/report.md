# DURA 正交恢复源训练 CPU 与 GPU 计时

- run_id：`20261106-phase1-dura-orthogonal-cuda-perf-s2026092701-r01`
- group_id：`phase1-dura-orthogonal-cuda-benchmark`；类别：`diagnostic`；阶段：`Phase1-restoration`
- 配置与矩阵：[experiment.json](experiment.json)；状态记录：[events.jsonl](events.jsonl)
- 当前登记状态：ANALYZED（实际状态按 events.jsonl 及独立证据更新）

## 目的与对照

对相同 512 个源 ID 的三个固定信道视图，使用真实 DURA 进行一轮纯 IQ 训练 smoke；比较 CPU batch8、GPU batch8 和 GPU batch32 的准备、训练及端到端时间。

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

## 已完成计时

CPU 与 GPU 的同源样本实测和数值适用范围见 [analysis.md](analysis.md)。
