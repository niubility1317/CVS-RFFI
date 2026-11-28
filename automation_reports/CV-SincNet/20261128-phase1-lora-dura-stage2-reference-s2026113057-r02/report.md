# LoRa 原生 8192 点双阶段 DURA 结构对照（冻结评估技术恢复）

- run_id：`20261128-phase1-lora-dura-stage2-reference-s2026113057-r02`
- group_id：`phase1-lora-wisig-gain-dura`；类别：`comparison`；阶段：`Phase1`
- 配置与矩阵：[experiment.json](experiment.json)；状态记录：[events.jsonl](events.jsonl)
- 当前登记状态：ANALYZED（源域屏查完成）

## 目的与对照

原 LoRa E104 加载阶段因模块导入冲突失败。保持原冻结恢复器、识别器、物理包、场景和方法规则。只恢复 LoRa 预测及随后独立评分。

## 数据、seed与模型来源

实际数据契约、权限例外、完整seed角色、checkpoint来源和选择规则见experiment.json。
逐行配置通过config_ref/resolved_config_ref定位；待补项必须在对应生命周期补齐。

## 执行与存储

命令、环境、CWD、commit、launch owner、输出和日志路径见experiment.json。
实际PID/GPU、读取时间、remote readback、失败或替代关系在此追加，并用record命令记录证据指针。

## 独立评分结果

LoRa 双阶段结构对照在 b200_1 的预登记 160 个源 V 包评分完成。表中 DURA 是历史三阶段 LoRa 原生 R17。新对照用 LoRa U_s 原生 IQ 与同包通用模拟配对，从零训练两阶段 R69 结构。训练未读取 W24、TX 真值、V 或目标数据。冻结选择固定在第 18,000 步。最终阶段损失为末阶段 IQ NMSE 与对数功率谱误差，加 0.15 倍前一阶段 IQ NMSE。

| 数据集与场景 | 直通 | DURA | 实增益 | 双阶段 DURA |
|---|---:|---:|---:|---:|
| LoRa Clean | 159/160（99.38%） | 159/160（99.38%） | 160/160（100.00%） | 159/160（99.38%） |
| LoRa 低城区 | 69/160（43.12%） | 45/160（28.12%） | 92/160（57.50%） | 38/160（23.75%） |

| 数据集 | 场景 | 对照 | 改对 | 改错 | 净变化 | 95% 配对区间 |
|---|---|---|---:|---:|---:|---:|
| LoRa | Clean | 直通 | 0 | 0 | +0 | +0.00 至 +0.00 |
| LoRa | 低城区 | 直通 | 18 | 49 | -31 | -47.00 至 -15.00 |
| LoRa | Clean | DURA | 0 | 0 | +0 | +0.00 至 +0.00 |
| LoRa | 低城区 | DURA | 14 | 21 | -7 | -19.00 至 +4.00 |
| LoRa | Clean | 实增益 | 0 | 1 | -1 | -3.00 至 +0.00 |
| LoRa | 低城区 | 实增益 | 12 | 66 | -54 | -70.00 至 -38.00 |

新增模型 266,951 个参数，共耗时 44.62 分钟，峰值 CUDA 分配 0.75 GiB。RTX 3090、FP32、B1 原生完整 8192 点恢复中位时延 61.08 ms，恢复接 E104 中位时延 107.42 ms。这是原生 eager 路径在源包上的观测，包含 W24 主机端预处理的恢复加识别时延单列。训练完成后严格载入冻结模型接同一 E104，用两个 U_s 包完成无 query 冒烟。本任务只补齐源域匹配对照。正式六景、样本日期分组和完整接收机比较待对应正式矩阵。

[独立评分与冻结证据](/E:/type10-7/local_artifacts/20261128-phase1-lora-dura-stage2-reference-s2026113057-r02/readback) · [原 r01 冻结权重](/E:/type10-7/local_artifacts/20261128-phase1-lora-dura-stage2-reference-s2026113057-r01/readback/dura_stage2.pt)
