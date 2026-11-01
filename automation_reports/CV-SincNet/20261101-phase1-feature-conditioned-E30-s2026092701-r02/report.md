# Phase1 冻结基座特征条件半监督恢复网络

- run_id：`20261101-phase1-feature-conditioned-E30-s2026092701-r02`
- group_id：`phase1-teacher-feature-conditioned-restoration`；类别：`cvs`；阶段：`Phase1-restoration`
- 配置与矩阵：[experiment.json](experiment.json)；状态记录：[events.jsonl](events.jsonl)
- 当前登记状态：PLANNED（实际状态按events.jsonl及独立证据更新）

## 目的与对照

冻结 Tri-SIG-TIME SSL 和原 DURA，以基座特征条件化受限 IQ 校正；源训练使用合法 L_s 标签和高置信 U_s clean 伪目标。

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

## 重新发布与启动核验

固定源码 `f72f9618859fce30a1983a3b4b10c8c869b2dbba`，远端发布归档 SHA-256 `e56b85dee8c19ac06afc4499df305a759e822abdcaca744806d8f04e60364ae6`，真实权重检查在源数据上进行。CPU 合成中断续训逐位一致，独立复核未发现明确 P1；真实 GPU checkpoint 和续训尚待验证。

真实 F/DURA 的 CPU 2 条和 GPU 128 条合法 L_s 检查通过，GPU 检查中 7 条基座错误，冻结模型未变；正式学生重新初始化，已进入源训练。物化 6300 条 L_s 标签、源档案 L128/U384 IQ，未读 U_s 标签、V、测试 IQ。

控制器 PID 24646；worker PID 24814、GPU 0 已独立核验。 CPU/GPU 小批次检查不是星地性能结果。

## 首轮训练状态核验

GPU 0 上已完成第 1 轮、493 次更新。独立 CPU 读取真实 GPU 写入的 `flow/checkpoints/epoch-01.pt`，核对源码、F、DURA、数据规模和训练预算，模型权重有限且可严格载入，AdamW 状态、NumPy、PyTorch CPU 和 CUDA 随机状态均存在；该文件 SHA-256 为 `250682fdd5923b6dad43c0d4906a65c99f7216f4a985cb3262469c899e011aa7`。这证明首轮状态已落盘，真实 GPU 上的中断续训尚未验证，也不构成性能结果。

## 固定源训练完成

固定 30 轮、14790 次更新完成；独立 CPU 在真实 Phase1 源记录上严格载入最终权重，校验 F/DURA 冻结、输出有限且重复结果一致。最终权重 SHA-256 4b66b90ced7ba4bc31bf7bad1b49483473d9b42b139a4faef2676bbbeaa4b8f9。七视图预测仍在进行，尚无准确率。

## Phase1 完整评分与独立复算

以同一冻结 Tri-SIG-TIME SSL 的 clean 原始准确率 **86.5387%** 为基准，对 168000 条相同物理记录的七视图先固定全部预测再评分。独立脚本复算总准确率、Macro-F1、TX/RX/day 各组；七视图原始判断与原基座缓存逐条一致。此前本集已被用于开发观察，结果为单 seed 非独立盲测。

| 视图 | 恢复准确率 | 相对 clean 下降（百分点） | 相对 clean 下降（%） | 相对 DURA（百分点） |
|---|---:|---:|---:|---:|
| clean | 85.8565% | 0.6821 | 0.7883% | -0.7589 |
| 高仰角 | 84.2958% | 2.2429 | 2.5917% | -0.0476 |
| 中仰角 | 82.4054% | 4.1333 | 4.7763% | -0.6107 |
| 低仰角郊区 | 80.4143% | 6.1244 | 7.0771% | -0.9077 |
| 高仰角城区 | 82.2571% | 4.2815 | 4.9476% | -0.7923 |
| 中仰角城区 | 73.1530% | 13.3857 | 15.4679% | -2.5530 |
| 低仰角城区 | 65.1601% | 21.3786 | 24.7041% | -3.6411 |

六个星地场景仅高仰角低于 3 个百分点和 3% 相对下降；该候选在六景均低于匹配 DURA。clean 下降 0.6821 个百分点，满足本次 clean 不超过 1 个百分点损失的保护条件。中、低仰角城区相较 DURA 的净损失分别为 4289 和 6117 条判断，详见逐条配对复算。
