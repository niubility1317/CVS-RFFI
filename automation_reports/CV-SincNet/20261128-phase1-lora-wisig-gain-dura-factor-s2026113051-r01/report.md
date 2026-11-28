# LoRa 与 WiSig 融合的观测项与补充先验消融

- run_id：`20261128-phase1-lora-wisig-gain-dura-factor-s2026113051-r01`
- group_id：`phase1-lora-wisig-gain-dura`；类别：`comparison`；阶段：`Phase1`
- 配置与矩阵：[experiment.json](experiment.json)；状态记录：[events.jsonl](events.jsonl)
- 当前登记状态：PLANNED（实际状态按events.jsonl及独立证据更新）

## 目的与对照

Six frozen paths on a new disjoint same-receiver source screen. Add observation-only and companion-only arms with the exact previous projection and cap; keep old combined as control.

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

## 完成与真实数据范围

两套冻结权重沿用首轮，训练增量为零。LoRa 固定 b200_1，WiSig 固定 RX1。每套 160 个源 V 包，已逐物理 ID 核对与首轮 192 个包无交集。LoRa 另排除历史 300 包面板，WiSig 仍来自历史已评的 1,800 包源 V。两套 Clean 和低城区的六条同包预测固定后，独立 CPU 评分。目标读取为零。

N607 控制器及工作目录和真实检查点检查已核实，双冻结预测和独立评分闭合。

| 数据集与场景 | 直通 | DURA | 实增益 | 原融合 | 仅观测更新 | 仅补充先验 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| LoRa Clean | 159/160（99.38%） | 159/160（99.38%） | 160/160（100.00%） | 160/160（100.00%） | 160/160（100.00%） | 160/160（100.00%） |
| LoRa 低城区 | 69/160（43.12%） | 45/160（28.12%） | 92/160（57.50%） | 93/160（58.13%） | 93/160（58.13%） | 93/160（58.13%） |
| WiSig Clean | 160/160（100.00%） | 160/160（100.00%） | 160/160（100.00%） | 160/160（100.00%） | 160/160（100.00%） | 160/160（100.00%） |
| WiSig 低城区 | 91/160（56.88%） | 139/160（86.88%） | 96/160（60.00%） | 140/160（87.50%） | 144/160（90.00%） | 135/160（84.38%） |

## 低城区同包作用

| 数据集 | 固定比较 | 改对 | 改错 | 净变化 | 95% 配对区间 |
| --- | --- | ---: | ---: | ---: | ---: |
| LoRa | 原融合 / 实增益 | 9 | 8 | +1 | -7.0 至 +9.0 |
| LoRa | 仅观测更新 / 实增益 | 6 | 5 | +1 | -6.0 至 +7.0 |
| LoRa | 仅补充先验 / 实增益 | 9 | 8 | +1 | -7.0 至 +9.0 |
| LoRa | 原融合 / 仅观测更新 | 7 | 7 | +0 | -7.0 至 +8.0 |
| LoRa | 原融合 / 仅补充先验 | 1 | 1 | +0 | -3.0 至 +3.0 |
| LoRa | 仅观测更新 / 仅补充先验 | 8 | 8 | +0 | -8.0 至 +8.0 |
| WiSig | 原融合 / DURA | 7 | 6 | +1 | -6.0 至 +8.0 |
| WiSig | 仅观测更新 / DURA | 7 | 2 | +5 | +0.0 至 +11.0 |
| WiSig | 仅补充先验 / DURA | 6 | 10 | -4 | -12.0 至 +4.0 |
| WiSig | 原融合 / 仅观测更新 | 3 | 7 | -4 | -11.0 至 +2.0 |
| WiSig | 原融合 / 仅补充先验 | 8 | 3 | +5 | -1.0 至 +12.0 |
| WiSig | 仅观测更新 / 仅补充先验 | 13 | 4 | +9 | +1.0 至 +17.0 |

WiSig 只用观测项比仅补充先验净增 9 包，配对区间 +1 至 +17。从仅观测项加回补充先验后净减 4 包，区间跨零。此面板支持分开看两项。仅观测项比原 R69 多判对 5 包，仍低于预登记的 +8 包门槛，区间下界为零。LoRa 三条候选均比实增益多 1 包，但同包交换不同，区间都跨零。两套 Clean 都达到最强对照，正式矩阵未启动。

融合相对原对照的参数复用和原融合 B1 性能仍见首轮报告。当前因素试验只改解算项，单项新时延待与后续候选一同测量。逐 TX、日期、其他接收机和六景均值待完整正式评估。
- [独立评分与每条对照的置信区间](/E:/type10-7/local_artifacts/20261128-phase1-lora-wisig-gain-dura-factor-s2026113051-r01/readback/scores.json)
- [冻结预测与真实检查点检查](/E:/type10-7/local_artifacts/20261128-phase1-lora-wisig-gain-dura-factor-s2026113051-r01/readback)
- [第一轮融合记录](../20261128-phase1-lora-wisig-gain-dura-screen-s2026113047-r01/report.md)
