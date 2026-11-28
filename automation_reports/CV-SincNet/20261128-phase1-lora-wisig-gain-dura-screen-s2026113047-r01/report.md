# LoRa 与 WiSig DURA／实增益融合首筛

- run_id：`20261128-phase1-lora-wisig-gain-dura-screen-s2026113047-r01`
- group_id：`phase1-lora-wisig-gain-dura`；类别：`comparison`；阶段：`Phase1`
- 配置与矩阵：[experiment.json](experiment.json)；状态记录：[events.jsonl](events.jsonl)
- 当前登记状态：PLANNED（实际状态按events.jsonl及独立证据更新）

## 目的与对照

Shared projection-constrained two-step MAP with data-specific anchor and a companion prior, evaluated on one fixed source validation receiver and two views per dataset.

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

## 已核实的对照与数据契约

| 数据集 | 冻结识别器 | DURA 对照 | 实增益对照 |
| --- | --- | --- | --- |
| LoRa | PAMF W24 E104，种子 2026092702，原 8×2×432 预处理和默认头 | 原生 LoRa R02/R07/R17 的固定 R17，有合法同契约权重。固定 R69 v1.0 的 WiSig 权重和 256 点输入与本合同不同，匹配 LoRa 双阶段对照待补 | 原生 8,192 点实增益版，30,000 包正式 Full 82.2967%／62.1167%／95.5467% |
| WiSig | Main split 三视图 Tri-SIG-TIME SSL E50，种子 2026092701，默认头 | 原 DURA v1.0/R69 E20、4,940 步、266,951 参数、FP32 Graph-RR，168,000 包同种子正式 Full 已完成 | 本轮从零训练，同口径首筛已完成，正式 Full 尚未评估 |

WiSig 单种子正式归档中，R69 六景正确率 79.3768%，低城区 115,599/168,000（68.8089%），Clean 145,514/168,000（86.6155%）。三个种子的归档另有汇总。
- [LoRa 实增益正式记录](/E:/type10-7/automation_reports/CV-SincNet/20261128-phase1-lora-raw8192-frequency-quadrature-s2026113037-r01/report.md)
- [WiSig R69 正式对照读回](/E:/type10-7/local_artifacts/wisig_r69_paired_main_20261123/verified_paired_readback.json)

## 单接收机筛查

LoRa 固定源接收机 b200_1，从合法 V 排除原 300 包源评面板后盲抽 192 包。WiSig 固定源接收机 1，在原 1,800 包源 Full 验证面板的 360 个同接收机包中盲抽 192 包。WiSig 面板曾用于历史源评。两景共享同一物理包和链路实现，目标 IQ 和目标真值均为零读取。

两数据集共用观测一致性求解器。主输出取 LoRa 实增益或 WiSig R69，将配对恢复输出与主输出的差值作为补充先验。DURA 从实收包估计作用和方差，求解器只更新主修正复数张成空间之外的方向，两次共轭梯度更新，补充编辑范数上限为主修正的 25%。

WiSig 实增益使用 WiSig 本身的 52,604 个合法 U_s 训练索引，从零训练 8,000 步，batch 24，通用回波与加性噪声。拟合隔离 Tri-SIG 权重、输出、源 V 和所有 TX 真值。包内用 20 点重复前导估计频偏，8×32 输入包级时频网络。共享逐频点实增益模块。训练 9.18 分钟，峰值分配显存 0.094 GiB。LoRa 与融合求解器均无新训练。

| 数据集与场景 | 直通 | DURA | 实增益 | 融合 |
| --- | ---: | ---: | ---: | ---: |
| LoRa Clean | 191/192（99.48%） | 190/192（98.96%） | 191/192（99.48%） | 191/192（99.48%） |
| LoRa 低城区 | 84/192（43.75%） | 44/192（22.92%） | 113/192（58.85%） | 108/192（56.25%） |
| WiSig Clean | 192/192（100.00%） | 192/192（100.00%） | 192/192（100.00%） | 192/192（100.00%） |
| WiSig 低城区 | 105/192（54.69%） | 167/192（86.98%） | 103/192（53.65%） | 164/192（85.42%） |

## 低城区同包配对

| 比较 | 改对 | 改错 | 净变化 | 95% 同包区间 |
| --- | ---: | ---: | ---: | ---: |
| LoRa 融合相对直通 | 41 | 17 | +24 | +9.0 至 +38.0 |
| LoRa 融合相对DURA | 78 | 14 | +64 | +48.0 至 +80.0 |
| LoRa 融合相对实增益 | 11 | 16 | -5 | -15.0 至 +5.0 |
| WiSig 融合相对直通 | 62 | 3 | +59 | +46.0 至 +73.0 |
| WiSig 融合相对DURA | 6 | 9 | -3 | -10.0 至 +4.0 |
| WiSig 融合相对实增益 | 64 | 3 | +61 | +47.0 至 +74.0 |

LoRa 融合相对原生 R17 的 Clean 净增 1 包，与直通和实增益逐包持平。WiSig 四条 Clean 路径均正确 192 包。两套均通过 Clean 检查。低城区晋级要求相对每个对照至少净增 8 包，且 95% 同包区间下界大于零。融合相对 LoRa 实增益净减 5 包，相对 WiSig R69 净减 3 包，两套均未晋级，本候选未启动完整目标矩阵。单接收机筛查不提供其他接收机、六景、逐 TX 或日期的正式结论。

融合对两套较强路径都带来少量改对和更多改错。现有四臂只识别出净损失，尚未区分来自补充先验还是观测一致性更新。

## 参数、实测时延与产物

共享 RTX 3090，物理 GPU2，用同一个合法源包交错测量各条固定路径。预热 8 次，计时 28 次。端到端包含固定识别器及实际预处理。

| 数据集 | 路径 | 恢复器参数 | B1 中位数 / P95（ms） |
| --- | --- | ---: | ---: |
| LoRa | 直通 | 0 | 30.88 / 36.73 |
| LoRa | DURA | 4,665,073 | 110.37 / 126.05 |
| LoRa | 实增益 | 325,029 | 44.73 / 53.50 |
| LoRa | 融合 | 4,990,102 | 116.89 / 146.24 |
| WiSig | 直通 | 0 | 16.07 / 21.90 |
| WiSig | DURA | 266,951 | 21.37 / 26.15 |
| WiSig | 实增益 | 317,093 | 24.93 / 34.68 |
| WiSig | 融合 | 584,044 | 60.42 / 81.32 |

r01 训练和双盲预测完成，独立评分阶段因归档 Python 命名空间冲突失败。CPU-only r02 使用两份原预测完成了 truth-last 评分，权重和预测均未重跑。

- [WiSig 实增益冻结权重](/E:/type10-7/local_artifacts/20261128-phase1-lora-wisig-gain-dura-screen-s2026113047-r01/readback/wisig/real_gain.pt)
- [两个数据集的冻结预测与训练日志](/E:/type10-7/local_artifacts/20261128-phase1-lora-wisig-gain-dura-screen-s2026113047-r01/readback)
- [独立双数据集评分和配对区间](/E:/type10-7/local_artifacts/20261128-phase1-lora-wisig-gain-dura-screen-s2026113047-r02/readback/scores.json)
