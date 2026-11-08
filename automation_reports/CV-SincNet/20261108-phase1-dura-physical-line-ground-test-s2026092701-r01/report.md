# DURA 物理步长组合：Phase1 七景测试

- run_id：`20261108-phase1-dura-physical-line-ground-test-s2026092701-r01`
- group_id：`phase1-dura-physical-line-ground-test`；类别：`cvs`；阶段：`Phase1-restoration`
- 配置与矩阵：[experiment.json](experiment.json)；状态记录：[events.jsonl](events.jsonl)
- 当前登记状态：PLANNED（实际状态按events.jsonl及独立证据更新）

## 目的与对照

固定源训练第20轮大容量修正器，利用接收信号与估计信道选择修正幅度，连接冻结 Tri-SIG-TIME SSL 完整七景评分。

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

## 固定方案与运行核实

使用冻结 DURA v1、源域第 20 轮 StableLargeContextDura（源 commit `b873acec23758690e5484bd1c5f82162739c6655`），沿其修正方向在 [0,1] 最小化估计信道下的接收残差。修正幅度只使用 received IQ 与 DURA 估计信道，不使用 clean、F 输出或目标标签；F 为冻结 Tri-SIG-TIME SSL。源域探索理由见 [物理步长诊断](../20261108-phase1-physical-line-source-s2026092701-r01/analysis.md)。历史 Phase1 测试已被观察，本次单 seed 不是新盲测或真实在轨。

代码 commit `2cc7be9a201692268b93d799bfacba0643f51141`，4 项针对性测试通过；独立审查发现评分子进程环境问题，已修复并定点复核通过。远端一次归档 SHA 对比、编译及 112 条真实源记录 smoke 通过。N607 CPU 2 线程 nice15，控制器 PID 954002、预测子进程 PID 954574；已进入七景预测。全部 168000×7 预测固定后自动独立 truth-last，并与 DURA 配对评分。其他健康任务未改动。

## 七景最终评分

七景各 168000 条预测固定后完成独立 truth-last 与 DURA 配对评分。完整逐场景和 TX/RX/day 分组见 [结果分析](analysis.md)。六个 practical 场景均低于 DURA；本结果属于已曝光的 Phase1 地面模拟测试，未作为其他已固定实验的训练或选模反馈。
