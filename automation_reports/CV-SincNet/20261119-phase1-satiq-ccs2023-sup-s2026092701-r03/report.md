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

尚无结果。按预登记artifact逐项记录路径和缺项；保留每row与RX/day/TX/scene/K/seed的对应关系。
源域训练完成、预测完成、评分完成及协议有效性分别陈述。不得用总索引或旧状态证明当前运行。

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
