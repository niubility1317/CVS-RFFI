# DURA 源 IQ 可信度门控训练

- run_id：`20261108-phase1-dura-signal-confidence-source-s2026092701-r01`
- group_id：`phase1-dura-signal-confidence-source`；类别：`cvs`；阶段：`Phase1-restoration`
- 配置与矩阵：[experiment.json](experiment.json)；状态记录：[events.jsonl](events.jsonl)
- 当前登记状态：PLANNED（实际状态按events.jsonl及独立证据更新）

## 目的与对照

冻结 DURA 与源 final20 修正器，仅用合法源配对 IQ 训练逐样本修正可信度，完成后自动七景测试。

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

## 运行核实

门控以归一化 received IQ、DURA 输出、物理创新、已有修正和估计方差为输入，输出 0 至 1 的逐记录修正权重。固定源域父模型后，使用原 32768 条源 ID × 3 视图、AdamW 0.0003、batch128 训练 20 轮，最终轮固定；训练不读取 F、TX 标签、V 或目标。

代码 commit `9d7e0a271dbe4d1745441719e5081d3b8f8d8e0c`；5 项针对性检查及独立 P0/P1 审查通过。N607 发布归档 SHA 一次比对、远端编译通过。GPU5 启动时空闲 18501 MiB、已有 2 个其他计算进程；队列 PID 979535、控制器 PID 979552、训练 PID 979553。32 条真实源记录的冻结 DURA 检查通过，父模型预计算已超过 16384 条。训练完成后自动执行 112 条源 smoke、七景预测和独立 truth-last 评分；历史测试已观察，不作盲测或在轨声明。
