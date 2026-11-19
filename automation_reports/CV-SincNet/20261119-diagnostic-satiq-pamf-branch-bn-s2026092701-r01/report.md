# SatIQ PAMF branch batch-normalization diagnostic

- run_id：`20261119-diagnostic-satiq-pamf-branch-bn-s2026092701-r01`
- group_id：`phase1-satiq-ccs2023-fully-supervised`；类别：`diagnostic`；阶段：`Phase1`
- 配置与矩阵：[experiment.json](experiment.json)；状态记录：[events.jsonl](events.jsonl)
- 当前登记状态：ANALYZED（实际状态按events.jsonl及独立证据更新）

## 目的与对照

Matched 5,000-update source diagnostic of separate affine-free branch normalization before L2.

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

## N607 readback

N607 GPU 5, PID 1951173. The run passed measured-waveform forward/backward and source smoke evaluation before starting the fixed 5,000-update comparison. Source training reads the existing manifest and cache. Independent test remains unused.

## 第 5,000 步诊断

5,000 步已完成，检查点和源验证结果已核实。与正式 PAMF-SUP 对照使用完全相同的样本、步数和参考名单。每条路径 160 维，融合后 320 维，分类输出 3,103 类。

| 方法 | K1 Top-1 | K16 Top-1 | 同星 AUC K1／K16 | 分类头 Top-1 | 分类头 CE | 不同包融合余弦均值 |
|---|---:|---:|---:|---:|---:|---:|
| PAMF-SUP | 0.070% | 0.078% | 0.5018／0.4913 | 0.078% | 8.0230 | 0.999819 |
| 两路分支归一化 | 0.125% | 0.047% | 0.5034／0.4977 | 0.092% | 8.4601 | 0.451353 |

完整源验证的同星 AUC 仍接近 0.50。固定 512 包抽查中，归一化大幅降低不同包的平均余弦，但以中心化协方差特征值计算的有效维度为 1.65，原版为 1.74。归一化版全速率路的第一主成分占中心化方差 84.5%，与实测同步模板相干幅度的 Pearson 相关绝对值为 0.65。这些统计支持后续用同步字段控制信号幅度因素，因果与跨日身份仍需独立验证。此诊断保留为消融，三条百轮训练基线继续运行。[正式对照](../20261119-phase1-satiq-ccs2023-sup-s2026092701-r03/report.md)
