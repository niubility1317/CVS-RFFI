# 有界投影恢复器：第 26 轮技术续训

- run_id：`20261108-phase1-Restore-BoundedProjected-E30-s2026092701-r03`
- group_id：`source-input-bounded-projected-prior-single30`；类别：`cvs`；阶段：`Phase1`
- 配置与矩阵：[experiment.json](experiment.json)；状态记录：[events.jsonl](events.jsonl)
- 当前登记状态：PLANNED（实际状态按events.jsonl及独立证据更新）

## 目的与对照

从资源保护中断前最后完成的第26轮恢复模型、AdamW和随机状态，继续原第27至30轮与完整地面测试。

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

## 技术恢复与固定矩阵

原 r02 在 GPU 2 总计算进程短时超过 3 时仅停止本任务；E26／6422 步模型、AdamW、CPU/GPU 随机状态及源顺序可用，尚无目标预测。固定继承 E26 后继续 E27–E30，累计 30 轮／7410 步；原有训练范围、识别器和两个总体全部七视图保持。GPU 2/4 入场前最多 1 个其他进程、至少 12288 MiB 空闲，持续监测总数最多 3。原任务保留；本续训独占新目录，启动后先执行实际 F0/源样本及技术恢复检查。
