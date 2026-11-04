# Phase2 第二组：隔离非有限故障并继续其他任务

- run_id：`20261105-phase2-csource-crosscil-quarantine-m4-r07`
- group_id：`phase2-group2-csource-bc`；类别：`comparison`；阶段：`Phase2-B-C`
- 配置与矩阵：[experiment.json](experiment.json)；状态记录：[events.jsonl](events.jsonl)
- 当前登记状态：PLANNED（实际状态按events.jsonl及独立证据更新）

## 目的与对照

Preserve complete and incomplete parent predictions, quarantine exactly the two failed jobs, and resume all unaffected tasks without opening truth-side scoring.

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

## 2026-11-05 隔离恢复启动

控制器 1165491 已运行，真实源模型无 query smoke 21/21 通过。独立核实复用 30572 个完整任务、190832 个硬链接预测文件，已复用的17条先导结果保留评分引用；原父 run 两处故障单元各10条部分预测均保留且隔离。实际23个worker已派发其他合法任务并产生新预测，观察时30585/40572个任务闭合、失败列表为空、2个隔离单元不计完成。两处故障未解决前不会创建全部预测完成标记或启动独立评分；先前针对其中一处的同GPU source/support单次诊断未复现，原因仍未知。

## 2026-11-05 CUDA 初始化失败

任务 38331 在 GPU7 加载源模型时发生 CUDA 驱动初始化失败，未生成预测，调度器停止派发并排空。32632/40572 个任务闭合，继承的两个非有限故障单元及其部分预测保留。随后同一 GPU 的一次简单 CUDA 初始化通过；原因尚未确定，评分和全部预测完成标记均未启动。
