# Phase2 第二组：CUDA 初始化故障恢复

- run_id：`20261105-phase2-csource-crosscil-cudarecovery-m4-r08`
- group_id：`phase2-group2-csource-bc`；类别：`comparison`；阶段：`Phase2-B-C`
- 配置与矩阵：[experiment.json](experiment.json)；状态记录：[events.jsonl](events.jsonl)
- 当前登记状态：PLANNED（实际状态按events.jsonl及独立证据更新）

## 目的与对照

Retain inherited quarantines and resume fixed tasks after one pre-prediction CUDA driver initialization failure.

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

## 2026-11-05 CUDA 初始化恢复运行

控制器 1364637 已运行，source-only/no-query smoke 21/21 通过。独立核实复用 32632 个完整任务和 203317 个硬链接预测文件，18条先导评分引用保留。预测前 CUDA 初始化失败的任务 38331 按原配置在 GPU0 单次恢复，16条预测已闭合；观察时 32661/40572 完成，失败列表为空。两处非有限单元及其20条部分预测保留祖先 run，仍不计完成；这些隔离任务未解决前不打开全部预测完成标记或独立评分。
