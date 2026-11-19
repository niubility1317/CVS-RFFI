# SatIQ CCS 2023 supervised 100 epochs with full-rate temporal bins

最终入模形状为 `[B, 2, 11000]`，全速率路径的 11 个符号各保存 5 个有序时间区间。PAMF 低速路径与全速率路径各输出 160 维，拼成 320 维，PAMF 分类输出 3,103 个波束分数。低速路径内部 256 点覆盖同步字段的 320 μs。[维度表与方法](../../../docs/PAMF_SATIQ_CCS2023_SUPERVISED_20261119.md)

SATIQ、PAMF-SUP 和 PAMF-Metric 均使用 100 个实际数据 epoch。每轮 28,225 个 32 包批次、903,200 条互异样本，逐轮共用同一采样。增强关闭。PAMF-SUP 使用等范数融合、同星竞争辅助 CE、两路各自 CE 和同包一致性。PAMF-Metric 在此添加角距离半难三元组，SATIQ 使用论文重构与三元组。所有行从零初始化。每轮单行预算以配置和实际 `resolved_config.json` 为准。

2026 年 11 月 19 日 19:03（香港时间）核实，N607 调度进程 `1934471` 已启动，等待共同缓存。原版 20,000 步队列只生成 CPU 缓存，100 轮早期架构的队列控制进程已在 GPU 启动前退出。两次用户调整均复用同一数据准备路径。[执行登记](experiment.json) · [状态记录](events.jsonl)

- run_id：`20261119-phase1-satiq-ccs2023-sup-s2026092701-r03`
- group_id：`phase1-satiq-ccs2023-fully-supervised`；类别：`cvs`；阶段：`Phase1`
- 配置与矩阵：[experiment.json](experiment.json)；状态记录：[events.jsonl](events.jsonl)
- 当前登记状态：RUNNING（实际状态按events.jsonl及独立证据更新）

## 目的与对照

Matched SatIQ baseline and PAMF full-rate dual-path supervision, 11,000-point IQ with five ordered bins per raw symbol.

## 数据、seed与模型来源

实际数据契约、权限例外、完整seed角色、checkpoint来源和选择规则见experiment.json。
逐行配置通过config_ref/resolved_config_ref定位；待补项必须在对应生命周期补齐。

## 执行与存储

命令、环境、CWD、commit、launch owner、输出和日志路径见experiment.json。
实际PID/GPU、读取时间、remote readback、失败或替代关系在此追加，并用record命令记录证据指针。

## 结果与覆盖

已核实源验证阶段结果和各行检查点。100 轮训练与随后留出的独立预测、评分仍待完成。

## 交接

记录已完成、当前run/commit、证据路径、阻塞与下一步；恢复先查原run，不重复启动。

## N607 live readback (Hong Kong time, 2026-11-19)

21 original archives (83.28 GB) matched the official MD5. Source and scorer caches completed. SATIQ and the adapted PAMF graph both passed the real-packet training and validation smoke. Active rows independently verified via process inventory, resolved configurations, and loss files.

| Row | Physical GPU | PID | Input | Identity embedding |
|---|---:|---:|---|---:|
| SATIQ | 7 | 1941032 | B x 2 x 11000 | 512 |
| PAMF-SUP | 2 | 1944076 | B x 2 x 11000 | 320 |
| PAMF-Metric | 3 | 1944077 | B x 2 x 11000 | 320 |

The original controller exited after dispatching SATIQ, which retained its PID and continued training. The direct dispatcher (1944065) started only the two pending rows under the same immutable training release. Each new GPU had two incumbent processes and over 14 GiB free under the previously authorized SatIQ concurrency override. The controller log and owner file are linked by path in [experiment.json](experiment.json).

## 第 5,000 步源验证

三组共用 1,375 个波束和 12,785 条查询。K1 和 K16 分别用每类 1 条和 16 条有标签参考。均匀随机检索的 Top-1 为 0.0727%。

| 方法 | K1 Top-1 | K16 Top-1 | 同星 AUC K1／K16 | 跨星 AUC K1／K16 | 分类头 CE |
|---|---:|---:|---:|---:|---:|
| SatIQ | 0.117% | 0.196% | 0.5132／0.5258 | 0.5130／0.5236 | — |
| PAMF-SUP | 0.070% | 0.078% | 0.5018／0.4913 | 0.5039／0.4924 | 8.0230 |
| PAMF-Metric | 0.094% | 0.063% | 0.5012／0.4908 | 0.5033／0.4921 | 8.0375 |

对源验证集确定性抽取 16 颗卫星、每星 4 个波束、每波束 8 包，共 512 包。冻结抽查显示，PAMF-SUP 的不同样本融合向量平均余弦相似度由初始化的 0.931940 升至 0.999819，公共同步路为 0.999945，全速率路为 0.999693。PAMF-Metric 融合为 0.999778。SatIQ 对照在第 5,000 步为 0.048287。PAMF 的 CE 已接近 3,103 类均匀预测的 8.040，向量坍塌提供了 CE 下降而检索仍接近随机的直接证据。5,000 步约占 0.18 轮。三条正式训练继续按原配置推进；仅源域样本用于此诊断。

用实际每类采样频率给所有输入输出相同的预测分布，已知类验证 CE 为 7.8352，固定输出源端最高频类别的验证 Top-1 为 0.0903%。两个 PAMF 分类头的第 5,000 步验证 Top-1 分别为 0.0783% 和 0.0823%，低于这个频率基线。[分支归一化诊断](../20261119-diagnostic-satiq-pamf-branch-bn-s2026092701-r01/report.md)

## 第 20,000 步源验证

| 方法 | K1 Top-1 | K16 Top-1 | 同星 AUC K1／K16 | 跨星 AUC K1／K16 | 分类头 Top-1 | 分类头 CE |
|---|---:|---:|---:|---:|---:|---:|
| SatIQ | 0.102% | 0.219% | 0.5144／0.5235 | 0.5127／0.5257 | — | — |
| PAMF-SUP | 0.117% | 0.149% | 0.5099／0.5104 | 0.5103／0.5095 | 0.090% | 7.9060 |
| PAMF-Metric | 0.094% | 0.055% | 0.5170／0.5121 | 0.5212／0.5125 | 0.126% | 7.8161 |

PAMF-Metric 的分类头 CE 首次在已登记检查点上低于源标签频率基线 7.8352。两个 PAMF 模型的同星波束区分仍很弱，K16 AUC 低于 SatIQ 对照。三组使用相同验证面板，均已继续训练。20,000 步约占 0.71 个数据 epoch，留出测试尚未评分。

相同 512 包抽查中，PAMF-SUP 原生路、全速率路、融合向量的不同包余弦均值分别为 0.999984、0.998977、0.999481。PAMF-Metric 分别为 0.949570、0.999893、0.974732，早期改善主要来自原生路，全速率路仍高度集中。这一区别支持分开记录两路表征，而非由 320 维输出本身推断有效容量。

## 第 1 个完整数据 epoch

| 方法 | K1 Top-1 | K16 Top-1 | 同星 AUC K1／K16 | 跨星 AUC K1／K16 | 分类头 CE |
|---|---:|---:|---:|---:|---:|
| SatIQ | 0.133% | 0.188% | 0.5320／0.5652 | 0.5304／0.5652 | — |
| PAMF-SUP | 0.125% | 0.180% | 0.5161／0.5186 | 0.5153／0.5191 | 7.8864 |
| PAMF-Metric | 0.188% | 0.188% | 0.5252／0.5255 | 0.5244／0.5158 | 7.7548 |

第 28,225 步，SatIQ 同星 K16 AUC 明显高于两组 PAMF。PAMF-Metric 的分类头 CE 继续低于按源类频率预测的 7.8352。三组继续按 100 轮配置推进。[同星负例组批诊断](../20261119-diagnostic-satiq-pamf-same-sat-batches-s2026092701-r01/report.md)

原 PAMF-SUP 的第 5 轮（141,125 步）同星 K1／K16 AUC 为 0.5308／0.5629，分类头 CE 7.5173、Top-1 0.3433%。原型版同一套源验证在较早的第 1 轮达到 0.5517／0.5646，另登记等预算结果。[PAMF-SUP 分支原型对照](../20261119-diagnostic-satiq-pamf-sup-branch-prototypes-s2026092701-r02/report.md)

## PAMF-SUP 训练诊断

完整解析该行截至第 83,000 步的 832 条训练记录和 stdout，与同期 SatIQ、PAMF-Metric 及三个相关诊断交叉核对。训练任务仍在运行，步号及时间单调增长，各分项损失有限。按实际窗口步数加权如下。

| 步数范围 | 融合 CE | 同星 CE | 同步路 CE | 全速率路 CE |
|---|---:|---:|---:|---:|
| 1 至 5,000 | 8.4177 | 4.2579 | 8.7516 | 9.0176 |
| 5,001 至 20,000 | 7.9620 | 3.8235 | 8.0084 | 8.0317 |
| 第 2 个完整 epoch | 7.8646 | 3.7443 | 7.9081 | 7.9770 |
| 81,001 至 83,000 | 7.6967 | 3.6141 | 7.8984 | 7.9270 |

用训练采样频率给所有已知类验证包输出同一个分布，CE 为 7.8352，固定预测最高频波束的 Top-1 为 0.0903%。首轮 PAMF-SUP 的验证 CE 为 7.8864，Top-1 为 0.1164%，同星 K16 AUC 为 0.5186。训练 CE 在第 3 轮末已低于训练频率基线 7.8338。固定验证日程在第 1 轮之后直接到第 5 轮，因此当前还没有第 3 轮的同口径验证指标。

在同一批 4,096 个已知类源验证包上对三个冻结检查点读回显式 head 分数。第 5,000 步，将每个包的融合分数替换成整批同一个平均向量，CE 只从 8.02346 变为 8.02377。首轮结束时，该替换从 7.89003 变为 7.91424。平均融合分数与训练对数频率的 Pearson 相关由 0.9206 升为 0.9575。首轮结束时 4,096 个包的融合 head 只给出 12 个不同 Top-1 类，出现最多的标签占 34.2%。单独的公共同步路 99.5% 的包预测成同一个类别。这个源诊断中的批均值仅用于离线归因，不进入训练或预测。

首轮冻结检查点的公共同步路、全速率路及融合向量在不同包之间的平均余弦分别为 0.9980、0.9948、0.9964，以中心化协方差特征值计算的有效维度分别为 1.08、1.19 和 1.94。当前最直接的瓶颈是两路波束表征低秩且共享方向很强；均值分数主要随标签频率变化。两路有波形差异，但强度尚不足以形成可靠的身份间隔。基于 8,192 条 L 波形的源端部分白化将首轮 K16 同星 AUC 提至 0.5344，K1 没有提高，说明匹配阶段还有可恢复的小信号，无法代替 SUP 训练优化。[完整读出对照](../20261119-diagnostic-satiq-source-calibrated-readout-s2026111907-r01/report.md)

## 三组第 5 轮的同口径读出

第 141,125 步的随机同星 K16 原始 AUC 分别为 SatIQ 0.5925、PAMF-SUP 0.5629、PAMF-Metric 0.6009。使用相同 L 包分别拟合部分白化后，跨日同星 AUC 分别为 0.6161、0.6049、0.6594。三组仍继续原登记百轮预算。[完整同口径对照](../20261119-diagnostic-satiq-paper-original-e5-readout-s2026111907-r01/report.md)
