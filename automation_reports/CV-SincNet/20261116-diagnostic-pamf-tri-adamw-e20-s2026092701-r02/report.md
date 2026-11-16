# Tri-SIG-TIME 源域 AdamW 更新方向诊断

## 固定方案与技术恢复

本 run 替代 r01。r01 在启动前检查阶段因 N607 环境缺少 `pytest` 而失败，未启动诊断；输出与失败日志保留。r02 保持科学矩阵、来源、权限和预算，改为现有 Python 标准库 `unittest` 执行定向检查，不安装软件。

源域诊断比较十类与六源、seed `2026092701` 各自原同契约 scratch E20。每个设置固定前三个 L/U 源批次，按原 continuation epoch 13 日程增强普通 L 前向；同一批次计算原始完整 L、U 和原 guard 梯度，并分别在独立参数容器执行 L-only 与受保护 L+U 的 CENTER_UNITS、物理 clip5、AdamW 首步。记录每组梯度和相对 L-only 的实际增量下降方向。原模型不更新，不读取 V、U 真值或目标；CPU 2 线程，不占 GPU。

共 6 个源批次和 12 次可丢弃影子更新。它们是固定 E20 上的 fresh-moment 一步反事实，既不是独立 seed，也不是后续训练轨迹或目标性能证据。源 checkpoint smoke 作为每个模型首步，传输、编译、定向协议负例及数值检查通过后直接继续；身份、权限、数值或梯度桥接错误终止该诊断。初版增强遗漏已修复，并经过独立 P0/P1 定点复核。

- run_id：`20261116-diagnostic-pamf-tri-adamw-e20-s2026092701-r02`
- group_id：`phase1-pamf-repl-weight-clean`；类别：`diagnostic`；阶段：`Phase1`
- 配置与矩阵：[experiment.json](experiment.json)；状态记录：[events.jsonl](events.jsonl)
- 当前登记状态：PLANNED（实际状态按events.jsonl及独立证据更新）

## 目的与对照

十类和六源各三批次，比较同一 E20 的 L-only 与原保护 L+U 实际首步。

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
