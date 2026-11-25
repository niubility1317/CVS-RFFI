# WiSig Main 所有星地 IQ 通过两版 DURA 后识别

- run_id：`20261125-phase1-wisig-dura-force-main-m2-r01`
- group_id：`wisig-forced-dura-main`；类别：`comparison`；阶段：`Phase1`
- 配置与矩阵：[experiment.json](experiment.json)；状态记录：[events.jsonl](events.jsonl)
- 当前登记状态：PLANNED（实际状态按events.jsonl及独立证据更新）

## 目的与对照

两组原样复用 40,000 步冻结权重和半幅读出，六景禁用门控。Clean 保留直通，输入与冻结 PAMF 均沿用原 Main 对照。

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

## 强制恢复和 N607 启动

原两组 40,000 步权重、半幅恢复读出、168,000 Main 物理包、两档信道和三个冻结 PAMF 识别器均复用。全部六种星地视图逐包通过两版 DURA 后识别，clean 使用直通。Raw 和 R69 对照从原实验冻结预测读回并核对同一档、路线及物理 ID 顺序。

执行提交 `5b78d952419d5ac5a2538115c97489ff767a2423`。N607 传输及编译通过。真实源 IQ 冒烟完成并通过，目标读数为 0。已独立核实 GPU 0、5、3、7 的四个默认强度预测进程、命令、工作目录和日志。评分端在八片全部齐备且每个星地视图的恢复激活数等于物理包数、clean 与直通决策一致后才读取真值。
