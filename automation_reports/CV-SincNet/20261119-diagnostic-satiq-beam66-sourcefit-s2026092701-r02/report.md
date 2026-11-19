# SatIQ Beam66 frozen source fit audit

对四个停止时最后完整落盘的检查点，逐条重评分原训练集 7,920 条和后续三日验证集 2,376 条。四组评分复现原检查点记录的 V 准确率，V NLL 与原值差小于 0.00001。只做源数据推理。[逐组评分](audit.jsonl) · [原运行和结论](../20261119-phase1-satiq-beam66-wide-s2026092701-r01/report.md)

| 组别 | 检查点 | L Top-1 | V Top-1 | 告知真实卫星号时的 V 波束 Top-1 |
|---|---:|---:|---:|---:|
| B0 | E75 | 38.59% | 1.89% | 33.42% |
| L0 | E75 | 40.96% | 4.46% | 48.95% |
| W0 | E125 | 64.05% | 3.83% | 46.63% |
| W1 | E50 | 30.29% | 4.25% | 46.59% |

最后一列只是在原 logits 上保留真实卫星的 3 个波束，随机基线是 33.33%。全 66 类随机基线是 1.52%。每个模型在 V 上均预测过全部 66 类。

- run_id：`20261119-diagnostic-satiq-beam66-sourcefit-s2026092701-r02`
- group_id：`phase1-satiq-oxford-beam66-sourcefit`；类别：`diagnostic`；阶段：`Phase1`
- 配置与矩阵：[experiment.json](experiment.json)；状态记录：[events.jsonl](events.jsonl)
- 当前登记状态：PLANNED（实际状态按events.jsonl及独立证据更新）

## 目的与对照

Score frozen stopped supervised checkpoints on complete clean L and V from the original source bundle. Diagnostic metrics only.

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
