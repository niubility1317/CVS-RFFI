# Phase1 observation-conditioned unrolled inverse fixed evaluation

- run_id：`20261102-phase1-observation-unroll-ground-test-s2026092701-r02`
- group_id：`phase1-observation-unroll`；类别：`cvs`；阶段：`Phase1-restoration`
- 配置与矩阵：[experiment.json](experiment.json)；状态记录：[events.jsonl](events.jsonl)
- 当前登记状态：PLANNED（实际状态按events.jsonl及独立证据更新）

## 目的与对照

Frozen observation-conditioned three-step blind inverse; no DURA in candidate, fixed Tri-SIG-TIME SSL.

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


## 观测展开网络：完整 Phase1 地面测试结果

独立观测条件三步展开网络的七景预测与 truth-last 评分已完成；每景 168,000 条，地面基座为固定 Tri-SIG-TIME SSL。全部预测固定后才连接真值，独立审计通过 119 个分组检查，并核对共同物理 ID 和 raw-F 决策与 DURA 基线一致。训练未直接使用 TX 身份标签；模型继承冻结基座指导和以基座伪类别构造的物理先验，属于 F-guided。历史源数据和地面开发测试已被多次探索，结果不是独立盲测或真实卫星链路表现。

| 场景 | DURA | 观测展开 | 相对 DURA（百分点） | 跨 RX | 跨 RX＋day |
|---|---:|---:|---:|---:|---:|
| clean | 86.6155% | 86.4411% | −0.1744 | 88.0683% | 81.5595% |
| 高信号 | 84.3435% | 84.7952% | +0.4518 | 86.3024% | 80.2738% |
| 中信号 | 83.0161% | 83.3548% | +0.3387 | 84.7103% | 79.2881% |
| 低郊区 | 81.3220% | 81.5732% | +0.2512 | 82.7849% | 77.9381% |
| 高城市 | 83.0494% | 82.9976% | −0.0518 | 84.4294% | 78.7024% |
| 中城市 | 75.7060% | 74.2315% | −1.4744 | 75.1127% | 71.5881% |
| 低城市 | 68.8012% | 66.0571% | −2.7440 | 66.7262% | 64.0500% |

总集 clean 基座为 86.5387%，按相对下降 3% 的门槛是 83.9425%，只有高信号受损场景达到；按下降 3 个百分点的门槛 83.5387% 也只有高信号达到。低城市比相对门槛低 17.8854 个百分点。跨 RX 与跨 RX＋day 各自的原始 clean 基座分别为 88.1103%、81.8238%，两列不是以总集 clean 为分母。该模型在三个较轻场景优于 DURA，但三个城市场景均低于 DURA，未达到总体目标；结果不用于调整已固定候选。实验编号：`20261102-phase1-observation-unroll-ground-test-s2026092701-r02`。
