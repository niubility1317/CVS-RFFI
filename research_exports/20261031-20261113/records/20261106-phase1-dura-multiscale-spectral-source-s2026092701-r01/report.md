# DURA 多尺度时频修正源域 GPU 训练

- run_id：`20261106-phase1-dura-multiscale-spectral-source-s2026092701-r01`
- group_id：`phase1-dura-multiscale-spectral`；类别：`cvs`；阶段：`Phase1-restoration`
- 配置与矩阵：experiment.json；状态记录：events.jsonl
- 当前登记状态：PLANNED（实际状态按events.jsonl及独立证据更新）

## 目的与对照

冻结原 DURA，使用全窗口多尺度编码、频域瓶颈及逐时刻复数滤波修正，沿用源域校准的噪声门控；无 F、TX、V 或目标训练数据。

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

## 源训练结果

训练完成，源 IQ 对照、参数与限制见 analysis.md；正式七景预测正在运行。
