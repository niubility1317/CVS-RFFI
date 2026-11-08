# DURA 可观测自适应增益七景测试

- run_id：`20261108-phase1-dura-large-context-adaptive-ground-test-s2026092701-r01`
- group_id：`phase1-dura-large-context-adaptive`；类别：`cvs`；阶段：`Phase1-restoration`
- 配置与矩阵：[experiment.json](experiment.json)；状态记录：[events.jsonl](events.jsonl)
- 当前登记状态：PLANNED（实际状态按events.jsonl及独立证据更新）

## 目的与对照

归一化大容量 DURA 修正联合学习每条信号的可观测增益，范围 0 至 2；训练不读取目标或 TX，推理不读取 RX。

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

## 预登记补充

源训练第 20 轮固定后接入冻结 Tri-SIG-TIME SSL Main split 默认头，clean 和六种 practical 各 168000 条；全部预测完成后独立 truth-last，报告各景与 DURA 的配对变化、clean 损伤及相对原始 clean 的 3 个百分点和相对 3% 两种门槛。该地面模拟测试集此前已用于探索，不是独立盲测；目标结果不回流训练或选模。

## 七景正式结果（2026-11-08）

七个视图各 168000 条预测固定后独立评分，六个受损场景均低于 DURA 和未加权稳定版，均未达到相对下降 3% 或下降 3 个百分点两种门槛。低城区自适应版 61.7464%，DURA 68.8012%，错变对 8499 条、对变错 20351 条。完整场景和逐 TX/RX/day 见[分析](analysis.md)。该结果为单模型、历史已曝光的地面模拟测试。
