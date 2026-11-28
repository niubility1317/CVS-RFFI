# LoRa 与 WiSig 共同实增益与 R17 末步先验联合训练筛查

- run_id：`20261128-phase1-shared-trainprior-gain-s2026112957-r01`
- group_id：`phase1-crossdata-shared-dura-realgain`；类别：`comparison`；阶段：`Phase1`
- 配置与矩阵：[experiment.json](experiment.json)；状态记录：[events.jsonl](events.jsonl)
- 当前登记状态：PLANNED（实际状态按events.jsonl及独立证据更新）

## 目的与对照

冻结粗恢复、信道作用估计及前两步先验，共同训练末步先验与实增益网络。

## 数据、seed与模型来源

实际数据契约、权限例外、完整seed角色、checkpoint来源和选择规则见experiment.json。
逐行配置通过config_ref/resolved_config_ref定位；待补项必须在对应生命周期补齐。

## 执行与存储

命令、环境、CWD、commit、launch owner、输出和日志路径见experiment.json。
实际PID/GPU、读取时间、remote readback、失败或替代关系在此追加，并用record命令记录证据指针。

远端发布 VERIFIED：提交 `de54a6cfda960530478e0ef296811709b312aa0d`，发布归档本地/远端 SHA 一致，远端编译通过。控制器 PID 4055343，LoRa worker 4055356/GPU 2、WiSig worker 4055357/GPU 3；真实权重源 U smoke 均通过，两个子进程 GPU 占用和日志增长已核实。独立 P0/P1 审查未发现阻断缺陷；本地针对性检查 14 项通过。

## 结果与覆盖

尚无结果。按预登记artifact逐项记录路径和缺项；保留每row与RX/day/TX/scene/K/seed的对应关系。
状态 RUNNING；先完成各自 1000 步源域训练，再自动执行固定源 V 两景预测和独立 CPU truth-last 评分。训练期间不读取源 V 或目标。不得用总索引或旧状态证明当前运行。

## 交接

记录已完成、当前run/commit、证据路径、阻塞与下一步；恢复先查原run，不重复启动。
