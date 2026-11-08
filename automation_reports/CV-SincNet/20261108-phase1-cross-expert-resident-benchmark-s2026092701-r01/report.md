# 时频恢复器显存驻留训练等价与速度

- run_id：`20261108-phase1-cross-expert-resident-benchmark-s2026092701-r01`
- group_id：`phase1-cross-expert-resident-benchmark`；类别：`diagnostic`；阶段：`Phase1`
- 配置与矩阵：[experiment.json](experiment.json)；状态记录：[events.jsonl](events.jsonl)
- 当前登记状态：PLANNED（实际状态按events.jsonl及独立证据更新）

## 目的与对照

固定同序源信号与模型初始化，对比每批 CPU 到 GPU 传输与 GPU 驻留索引的四步训练耗时和精确状态。

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

## 后台排队

该对照使用 512 个合法源 ID × 3 视图，在相同初始化及批次顺序下比较原 CPU 随机取样后传 GPU 与 GPU 驻留后取样；各路径四步训练，顺序相反的两组配对，另外报告预载耗时。逐项检查损失、模型、优化器及 CPU/GPU/采样 RNG 状态，结果不外推为完整训练或目标预测相同。

代码 commit `a07b956883edfc05723551802f344a69a549bc0c`；五项针对性检查和独立 P0/P1 审查通过，归档传输一致、远端编译通过。N607 GPU0 队列 PID 1134600 已核实，当前等待其他进程释放容量；尚未开始测量或获取加速结果。不停止现有训练或其他 owner 任务。
