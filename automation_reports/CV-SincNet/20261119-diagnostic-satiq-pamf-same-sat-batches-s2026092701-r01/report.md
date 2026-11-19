# SatIQ PAMF same-satellite hard-negative batching diagnostic

- run_id：`20261119-diagnostic-satiq-pamf-same-sat-batches-s2026092701-r01`
- group_id：`phase1-satiq-ccs2023-fully-supervised`；类别：`diagnostic`；阶段：`Phase1`
- 配置与矩阵：[experiment.json](experiment.json)；状态记录：[events.jsonl](events.jsonl)
- 当前登记状态：ANALYZED（实际状态按events.jsonl及独立证据更新）

## 目的与对照

Match PAMF-Metric architecture and objective for one full source epoch. Group half-batches by satellite so in-batch triplets encounter wrong beams of the same satellite.

## 数据、seed与模型来源

实际数据契约、权限例外、完整seed角色、checkpoint来源和选择规则见experiment.json。
逐行配置通过config_ref/resolved_config_ref定位；待补项必须在对应生命周期补齐。

## 执行与存储

命令、环境、CWD、commit、launch owner、输出和日志路径见experiment.json。
实际PID/GPU、读取时间、remote readback、失败或替代关系在此追加，并用record命令记录证据指针。

## 结果与覆盖

第一轮真实训练清单已逐批核对。原方案只有 11.04% 的训练锚点在批次内遇到同星错波束，平均 0.116 个候选。新版达到 93.43% 和 2.638 个。两者的跨 UTC 日期正对比例均约为 93.42%。两者各遍历 903,200 条互异样本，首轮共享 903,172 条。首 5,000 步由于调换批次，仅共享 28,356 条，因此以一整轮结果为主要配对对照。

该诊断保留 PAMF-Metric 的 11,000 点输入、模型、所有损失、训练随机初值和验证面板。正式三组百轮训练继续运行。[正式实验](../20261119-phase1-satiq-ccs2023-sup-s2026092701-r03/report.md)

用正式 PAMF-Metric 的冻结第 1 轮权重，分别重放两种采样的前 96 批，各 9,216 个有序训练正对。半难负例存在率 96.94% 与 96.84%。损失选中同星错波束的比例从 1.61% 升至 36.88%。这项读回只确认组批实际改变了训练信号，模型性能仍按冻结源验证点比较。

## 交接

记录已完成、当前run/commit、证据路径、阻塞与下一步；恢复先查原run，不重复启动。

N607 GPU 0 received the new job with no incumbent workers. PID 1967567 passed the real-packet smoke and produced a growing step log, using one source epoch and no augmentation.

## 第 5,000 步源验证

| 采样 | 同星 AUC K1／K16 | K1／K16 Top-1 | 分类头 CE |
|---|---:|---:|---:|
| 原 PAMF-Metric 随机组批 | 0.5012／0.4908 | 0.094%／0.063% | 8.0375 |
| 同星组批 | 0.5001／0.4916 | 0.117%／0.078% | 8.0798 |

两种采样的初期同星 AUC 均接近 0.50。在第 5,000 步，组批已改变早期物理样本顺序和子集，因此首轮全部数据的结果用于主要判断。诊断进程已继续训练，独立留出测试尚未使用。

## 第 20,000 步源验证

| 采样 | 同星 AUC K1／K16 | K1／K16 Top-1 | 分类头 CE |
|---|---:|---:|---:|
| 原 PAMF-Metric 随机组批 | 0.5170／0.5121 | 0.094%／0.055% | 7.8161 |
| 同星组批 | 0.5015／0.4992 | 0.094%／0.141% | 7.9483 |

同星组批提高了负例覆盖，但本检查点的同星 AUC 低于随机组批。主判断仍使用跑完首轮的结果。

## 第 1 个完整数据 epoch

| 采样 | 同星 AUC K1／K16 | K1／K16 Top-1 | 分类头 CE |
|---|---:|---:|---:|
| 原 PAMF-Metric 随机组批 | 0.5252／0.5255 | 0.188%／0.188% | 7.7548 |
| 同星组批 | 0.5080／0.5073 | 0.156%／0.078% | 7.9372 |

同星组批的 28,225 步已完成，`complete.json` 和第 1 轮检查点已核实。这一组批方式虽改变了损失选中的负例，固定预算内的同星识别仍较差，未晋级到 100 轮。
