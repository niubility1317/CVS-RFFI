# R42 加速调度退出竞态修复与结果复用

- run_id：`20261101-phase2-r42-group1-m4-r05`
- group_id：`phase2-frozen-R42-group1-completion`；类别：`cvs`；阶段：`Phase2-B-C`
- 配置与矩阵：[experiment.json](experiment.json)；状态记录：[events.jsonl](events.jsonl)
- 当前登记状态：PLANNED（实际状态按events.jsonl及独立证据更新）

## 目的与对照

One technical recovery after controller exit race; preserve original predictions, reconcile any live predecessor worker and continue fixed R42 full matrix with DLPack and CUDA acceleration.

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

退出竞态已本地复现并修复，57 项定向测试、一次直接 P0/P1 修复定向复核通过。原 r03 312 个预测完成，交接记录中的9个旧 worker 均已自然闭合；旧 controller 和失败 controller 均已退出，r04 无预测输出。新运行逐项验证旧完成结果及进程，不重算、覆盖或根据目标分数调整。
