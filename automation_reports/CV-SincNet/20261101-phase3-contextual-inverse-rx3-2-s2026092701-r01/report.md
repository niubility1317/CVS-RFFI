# 动态逆变换恢复器接入Phase3：同记录冻结PCD对照

- run_id：`20261101-phase3-contextual-inverse-rx3-2-s2026092701-r01`
- group_id：`phase3-contextual-inverse-paired-diagnostic`；类别：`cvs`；阶段：`Phase3-fresh-contextual-G-diagnostic`
- 配置与矩阵：[experiment.json](experiment.json)；状态记录：[events.jsonl](events.jsonl)
- 当前登记状态：PLANNED（实际状态按events.jsonl及独立证据更新）

## 目的与对照

原固定F0/E0/PCD和密度阈值保持，比较新恢复器和DURA在相同源、跨RX clean/星地记录上的身份与未知拒识。

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


## 实际结果与独立复算

VERIFIED/ANALYZED。冻结新 G、原三视图 F0、E0、PCD 和原密度及阈值，源 V27000/novel3150、RX34 同记录 clean 与 G10 星地各 10238 条预测全部固定后独立评分；没有新增训练、目标 U 读取、密度或阈值拟合。另用保存的逐条预测和真值独立复算源、目标总体及六景共 18 组宏指标、逐记录配对及六景 clean 差距，一致。

| 星地条件 | F0 闭集宏准确率 | 已知正确接收 A_K | 已知接收 C_K | 未知拒识 R_U | H-score |
|---|---:|---:|---:|---:|---:|
| 新 G | 37.5707% | 21.2624% | 37.3639% | 57.0797% | 30.9834% |
| DURA 对照 | 41.7211% | 27.9517% | 48.5824% | 49.3077% | 35.6781% |

新 G 在星地已知正确接收方面相对 DURA 增 80 条、损 530 条，净少 450 条；未知正确拒识增 845 条、损 397 条，净多 448 条。高拒识伴随已知接收减少，不能单独解释为综合改进。

同一场景中、同一批已知物理 ID 的 raw clean 与新 G 星地 F0 逐 TX 宏差距如下（百分点）：

| 场景 | raw clean − 新 G 星地 |
|---|---:|
| practical_high | 1.0796 |
| practical_mid | 4.6402 |
| practical_low_suburban | 3.1323 |
| practical_high_urban | -8.6388 |
| practical_mid_urban | 13.0383 |
| practical_low_urban | 23.2369 |

六景有四景超过 3 pp，上述目标未通过；同记录 clean 损伤 0.4467 pp，低于 1 pp。这项按场景分配的 Phase3 配对口径，与历史 F50 对每景使用全体 168000 记录的七景回归不同，不能混成同一表。

该开发 RX3-2 已曝光、单模型 seed，PCD 有既有 target U576 祖先，且密度和阈值是在原 DURA 输入上固定；更换 G 后直接复用只证明固定系统表现，不等于匹配校准的部署性能。G 的源训练为 R07 2.462 GHz，本 Phase3 输入为 G10 2.45 GHz，也不能称为 G10 训练。未知自主发现、可信确权与注册未完成；默认 DURA 和总目标保持。
