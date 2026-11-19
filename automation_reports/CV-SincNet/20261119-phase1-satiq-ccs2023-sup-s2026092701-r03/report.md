# SatIQ CCS 2023 supervised 100 epochs with full-rate temporal bins

最终入模形状为 `[B, 2, 11000]`，全速率路径的 11 个符号各保存 5 个有序时间区间。PAMF 低速路径与全速率路径各输出 160 维，拼成 320 维，PAMF 分类输出 3,103 个波束分数。低速路径内部 256 点覆盖同步字段的 320 μs。[维度表与方法](../../../docs/PAMF_SATIQ_CCS2023_SUPERVISED_20261119.md)

SATIQ、PAMF-SUP 和 PAMF-Metric 均使用 100 个实际数据 epoch。每轮 28,225 个 32 包批次、903,200 条互异样本，逐轮共用同一采样。增强关闭。PAMF-SUP 使用等范数融合、同星竞争辅助 CE、两路各自 CE 和同包一致性。PAMF-Metric 在此添加角距离半难三元组，SATIQ 使用论文重构与三元组。所有行从零初始化。每轮单行预算以配置和实际 `resolved_config.json` 为准。

2026 年 11 月 19 日 19:03（香港时间）核实，N607 调度进程 `1934471` 已启动，等待共同缓存。原版 20,000 步队列只生成 CPU 缓存，100 轮早期架构的队列控制进程已在 GPU 启动前退出。两次用户调整均复用同一数据准备路径。[执行登记](experiment.json) · [状态记录](events.jsonl)

- run_id：`20261119-phase1-satiq-ccs2023-sup-s2026092701-r03`
- group_id：`phase1-satiq-ccs2023-fully-supervised`；类别：`cvs`；阶段：`Phase1`
- 配置与矩阵：[experiment.json](experiment.json)；状态记录：[events.jsonl](events.jsonl)
- 当前登记状态：PLANNED（实际状态按events.jsonl及独立证据更新）

## 目的与对照

Matched SatIQ baseline and PAMF full-rate dual-path supervision, 11,000-point IQ with five ordered bins per raw symbol.

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
