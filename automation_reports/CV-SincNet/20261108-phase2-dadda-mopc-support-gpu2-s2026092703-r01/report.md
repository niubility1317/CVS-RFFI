# Phase2 DADDA→MoPC 支持集数值诊断（GPU2）

- run_id：`20261108-phase2-dadda-mopc-support-gpu2-s2026092703-r01`
- group_id：`phase2-group2-dadda-mopc-nonfinite`；类别：`diagnostic`；阶段：`Phase2-B-C`
- 配置与矩阵：[experiment.json](experiment.json)；状态记录：[events.jsonl](events.jsonl)
- 当前登记状态：PLANNED（实际状态按events.jsonl及独立证据更新）

## 目的与对照

在原 GPU2 对 B r06 任务 5944 的原源权重和 K1/N20 支持集单次诊断首个非有限边界；不生成预测。

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

## 原设备只读诊断范围

单次使用物理 GPU2，在指定支持集和原始源权重上沿原 B/C 配方检查数值边界；不访问 query、真值，不生成预测。既有完整与部分结果保持。此前 CPU 未复现，不能证明 GPU 结果。独立 P0/P1 只读审核通过；启动前检查 GPU2 显存余量及本任务进程，诊断期间其他正式任务不并行。

## 实际启动

控制器 PID 1806700、start_ticks 63816391 的工作目录、命令和物理 GPU2 可见性已独立核实；真实源 checkpoint 的 source-only/no-query smoke PASS。启动前 GPU2 空余 23,548 MiB，其他本任务训练预测进程为零。诊断仅访问源域与原任务支持集，不生成预测；正式任务与已有部分结果保持。

## 诊断结果

原 GPU2 单次支持集诊断已结束，真实源检查通过，DADDA B 100 步及 MoPC C 四阶段 80 步完成；所检查边界与张量未发现非有限值。原故障未复现，根因仍未知；一次正常运行不证明故障已修复。原任务 9 份部分预测原样保留，诊断未读取 query 或真值，也未写预测。
