# DURA 多信道与强扰动联合源训练

- run_id：`20261109-phase1-dura-diverse-response10-source-s2026092701-r01`
- group_id：`phase1-dura-diverse-response10`；类别：`cvs`；阶段：`Phase1-restoration`
- 配置与矩阵：[experiment.json](experiment.json)；状态记录：[events.jsonl](events.jsonl)
- 当前登记状态：PLANNED（实际状态按events.jsonl及独立证据更新）

## 目的与对照

30 组源模拟信道轮换与固定 10% 人工扰动响应损失联合训练；部署仅当前接收 IQ。

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

## 联合训练固定方案

相同 32,768 条合法源记录，20 轮中每轮 3 视图，30 个源模拟视图按 10 组轮换两次；每个视图用匹配的模拟 action 构造 10% RMS 人工变化，响应损失权重 0.01。网络、初始化、优化器和更新数与已登记的 30 视图单独版本匹配；此版额外执行扰动输入的 DURA 预计算及恢复器前向，计算成本不同。

每组缓存的接收 IQ、clean IQ 与同记录信道 action 核对，固定当前模拟器状态和原始观测噪声，不重新生成完整信道或 AGC；人工方向尚未证明是实际 TX 特征。训练不读 F 权重/输出、TX、V 或目标，所有七景预测完成后独立 truth-last。

代码 7fe86632acf583762a056372f989502d5be96afc；日程和 10% 幅度、错误 checkpoint 拒绝检查通过，独立 P0/P1 静态审查无阻断。GPU1 逐阶段检查显存与总进程数并保留其他 owner；首步执行真实源训练和无查询识别检查。

N607 控制器 PID 2634386；GPU1 源训练 PID 2634740，已有其他 owner 进程保留。64 条源记录 × 3 视图的训练 smoke 与 112 条冻结识别器源无查询 smoke 均通过；完整训练已启动，七景预测和独立评分等待最终冻结。
