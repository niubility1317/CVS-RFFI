# DURA 局部共轭相位修正七景测试

- run_id：`20261107-phase1-dura-circular-ground-test-s2026092701-r01`
- group_id：`phase1-dura-circular`；类别：`cvs`；阶段：`Phase1-restoration`
- 配置与矩阵：[experiment.json](experiment.json)；状态记录：[events.jsonl](events.jsonl)
- 当前状态：七景预测运行中。控制器 PID 347435、预测 worker PID 366089；实时状态以远端进程、日志和产物为准。

## 目的与对照

冻结 DURA，将局部共轭特征引入条件相位等变修正分支。

## 数据、seed与模型来源

实际数据契约、权限例外、完整seed角色、checkpoint来源和选择规则见experiment.json。
逐行配置通过config_ref/resolved_config_ref定位；待补项必须在对应生命周期补齐。

## 执行与存储

命令、环境、CWD、commit、launch owner、输出和日志路径见experiment.json。
实际PID/GPU、读取时间、remote readback、失败或替代关系在此追加，并用record命令记录证据指针。

## 结果与覆盖

源训练固定 final epoch 5 后，112 条源预测 smoke 通过；正式七景预测已启动，尚无完整预测或独立 truth-last 评分，因此尚无识别性能结果。按预登记 artifact 记录路径和缺项；保留每 row 与 RX/day/TX/scene/K/seed 的对应关系。
源域训练完成、预测完成、评分完成及协议有效性分别陈述。不得用总索引或旧状态证明当前运行。

## 交接

记录已完成、当前run/commit、证据路径、阻塞与下一步；恢复先查原run，不重复启动。
