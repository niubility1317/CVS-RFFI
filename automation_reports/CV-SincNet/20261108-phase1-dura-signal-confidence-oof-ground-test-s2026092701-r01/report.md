# 未见源记录门控七景测试

- run_id：`20261108-phase1-dura-signal-confidence-oof-ground-test-s2026092701-r01`
- group_id：`phase1-dura-signal-confidence-oof-ground`；类别：`cvs`；阶段：`Phase1-restoration`
- 配置与矩阵：[experiment.json](experiment.json)；状态记录：[events.jsonl](events.jsonl)
- 当前登记状态：PLANNED（实际状态按events.jsonl及独立证据更新）

## 目的与对照

冻结 source-only 门控 final20 并连接 Tri-SIG-TIME SSL；全部七景预测固定后独立评分。

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

## 固定方案与启动核实

门控训练使用 17632 条 L_s/U_s 源记录：与父修正器的 32768 条拟合记录、固定 672 条源 pilot 不重叠，且排除 RX4/6/8 day3。使用每条三个固定源视图和 batch128 训练 20 轮；父模型与 DURA 均冻结，训练不读取 TX 标签、F 权重/输出、V 或目标。此版源记录数和前版不同，不是等曝光对照。

代码 commit `7840d7592bf68f9b975e2f5e44a479dcdc1688fd`；6 项聚焦测试及一次独立 P0/P1 审查通过。远端发布归档校验及编译通过；GPU5 启动前空闲 18501 MiB、其他计算进程 2 个。队列 PID 1000588、控制器 PID 1000599、训练 PID 1000600；32 条真实源记录 smoke 通过，已进入源数据准备。第20轮固定后自动进行 112 条源 smoke 和完整七景预测，之后独立 truth-last 评分。
