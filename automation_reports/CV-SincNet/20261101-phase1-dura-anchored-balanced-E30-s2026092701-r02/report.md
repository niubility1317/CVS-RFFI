# Phase1 旧恢复锚定与梯度平衡生成轨迹

- run_id：`20261101-phase1-dura-anchored-balanced-E30-s2026092701-r02`
- group_id：`phase1-dura-anchor-balanced-flow`；类别：`cvs`；阶段：`Phase1-restoration`
- 配置与矩阵：[experiment.json](experiment.json)；状态记录：[events.jsonl](events.jsonl)
- 当前登记状态：PLANNED（实际状态按events.jsonl及独立证据更新）

## 目的与对照

冻结原 Phase1 DURA 和 Tri-SIG-TIME SSL，使用 DURA 输出为生成轨迹起点，源训练时对波形与语义梯度进行冲突投影和幅度限制。

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

## 已核实启动状态

原 r01 的 CPU smoke 因 train/eval 模式不一致导致逐位比较失败，未开始训练；r02 只修复 smoke 比较模式，正式科学配方不变。源码版本 `f16f514179428d16bd019a23aca56dd82aa93b1b`，远端 release 与本地归档 SHA-256 一致。

真实 Phase1 DURA 和 Tri-SIG-TIME SSL 的 CPU smoke 通过：零场初始输出与 DURA 最大绝对差 `5.9605e-8`，参考与重载使用相同 eval/frozen 模式后严格一致，F/G 参数和状态冻结。DURA checkpoint SHA-256 为 `14200ac0f8eef01f57aa82e8fb5122ff855fc81208983d9234d6c5caef2e28ee`，绑定生成场 checkpoint。

GPU 5 上实际 128 条受扰 L_s 的一次临时 field 更新检查通过，F/G 不变、无 TX/V/test IQ；首次批次 1.2284 秒、峰值分配约 1.24 GB。此次检查使用临时 field 和固定学习率 0.0003，之后丢弃；正式训练重新初始化并从预登记 warmup 开始。

控制器 PID 2241197、GPU worker PID 2241365 已独立核验；与同账号任务共享显卡，启动时显存余量 22815 MB，单卡总进程不超过 2。正式 E30 源训练已启动，尚无该候选测试结果。既有 Phase1 开发测试结果已被观察，后续同集结果不是盲测。

## Phase1 七视图完整结果与独立核验

状态 VERIFIED／ANALYZED，固定 E30 模型，七视图各 168000 个 Phase1 测试物理记录；全部 14 份预测完成后才连接真值。冻结基座为原始 Tri-SIG-TIME SSL，原始 clean 准确率 86.5387%。差距为原始 clean 减恢复准确率，单位为百分点。

| 视图 | 原始准确率 | 恢复准确率 | 与原始 clean 差距 | 恢复减 DURA |
|---|---:|---:|---:|---:|
| clean | 86.5387% | 86.3565% | 0.1821 | -0.2589 |
| practical_high | 80.6155% | 81.2440% | 5.2946 | -3.0994 |
| practical_mid | 78.1077% | 80.0655% | 6.4732 | -2.9506 |
| practical_low_suburban | 75.7446% | 78.6387% | 7.9000 | -2.6833 |
| practical_high_urban | 77.0500% | 80.0643% | 6.4744 | -2.9851 |
| practical_mid_urban | 62.7958% | 73.7179% | 12.8208 | -1.9881 |
| practical_low_urban | 51.6048% | 67.3679% | 19.1708 | -1.4333 |

六种信道均未达到距 clean 不超过 3 个百分点或相对损失不超过 3% 的目标。恢复结果均高于同视图未经恢复的原始输入，但均低于同一冻结基座下的 DURA。clean 满足预登记保护线。独立 CPU 进程按物理 ID 复算总体、TX、RX、day 和 Macro-F1，均与原评分一致；七视图各 168000 条原始预测与基座记录逐项一致。DURA 配对比较核对了同序物理 ID、原始预测与修复／损伤计数。

最终生成场 SHA-256 为 `559cbcda3b4bac259b794b2a28aeaf54f59c6178417c86ee561c6e11da1b5de1`；加载时严格核实 DURA 指纹，源样本推理具有确定性。梯度投影及范数上限只约束优化器输入，不能保证 AdamW 更新后提升性能。单模型 seed；候选设计前已观察 Phase1 开发测试，故结果不构成盲测。
