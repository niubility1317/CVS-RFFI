# R42 全 KN 张量传输与判决加速接管

- run_id：`20261101-phase2-r42-group1-m4-r04`
- group_id：`phase2-frozen-R42-group1-completion`；类别：`cvs`；阶段：`Phase2-B-C`
- 配置与矩阵：[experiment.json](experiment.json)；状态记录：[events.jsonl](events.jsonl)
- 当前登记状态：PLANNED（实际状态按events.jsonl及独立证据更新）

## 目的与对照

Frozen R42 and original LocalRidge fixed matrix; exact DLPack transfer, FP64 CUDA pair budget 16384, and up to four independent CPU scorers after global prediction closure. Reuse all predecessor predictions and retain live predecessor workers.

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

本次仅优化张量传输、FP64 LocalRidge 判决批次和预测冻结后的最多四进程独立 CPU 评分；不修改 R42、DURA、数据、查询规则或结果口径。53 项定向测试及独立 P0/P1 审查通过；接管前依次对四个真实源 checkpoint 检查基础/八视图/DURA 和 CPU/CUDA 判决一致性，失败恢复旧控制器。共享GPU上的合成基准不代表全任务速度。

2026-11-01 故障核验：FAILED: four real-checkpoint source-only CUDA smokes PASS; after SIGKILL of exact predecessor controller, /proc/321010/environ raised PermissionError during process exit. Both controllers verified stopped; nine predecessor workers subsequently closed; original outputs retained. One targeted r05 recovery prepared.
