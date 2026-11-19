# SatIQ source waveform length and calibration diagnostic

- run_id：`20261119-diagnostic-satiq-waveform-span-s2026111907-r02`
- group_id：`phase1-satiq-ccs2023-fully-supervised`；类别：`diagnostic`；阶段：`Phase1`
- 配置与矩阵：[experiment.json](experiment.json)；状态记录：[events.jsonl](events.jsonl)
- 当前登记状态：ANALYZED（实际状态按events.jsonl及独立证据更新）

## 目的与对照

Compare 256/8000/11000 points and deterministic affine UW calibration on identical frozen source reference/query panels, with a UTC time-blocked check.

## 数据、seed与模型来源

实际数据契约、权限例外、完整seed角色、checkpoint来源和选择规则见experiment.json。
逐行配置通过config_ref/resolved_config_ref定位；待补项必须在对应生命周期补齐。

## 执行与存储

命令、环境、CWD、commit、launch owner、输出和日志路径见experiment.json。
实际PID/GPU、读取时间、remote readback、失败或替代关系在此追加，并用record命令记录证据指针。

## 结果与覆盖

共读取 34,785 条互异源验证包。论文式面板为 1,375 类、12,785 条查询。跨日期面板在 2023 年 4 月 3 日 00:00 UTC 分隔参考和查询，含 309 类、4,865 条查询。两套面板分别在所有输入视图之间共享物理记录、参考顺序和竞争波束。下表为同星波束 AUC。

| 波形视图 | 论文式 K1 | 论文式 K16 | 跨日期 K1 | 跨日期 K16 |
|---|---:|---:|---:|---:|
| 原始短窗 256 | 0.4966 | 0.4958 | 0.5047 | 0.5012 |
| 原始同步字段 8,000 | 0.4981 | 0.4941 | 0.4983 | 0.5027 |
| 原始报头 11,000 | 0.4985 | 0.4940 | 0.4980 | 0.5023 |
| 同步字段复增益／DC 对齐 8,000 | 0.4983 | 0.4943 | 0.4984 | 0.5032 |
| 报头复增益／DC 对齐 11,000 | 0.4986 | 0.4941 | 0.4979 | 0.5028 |
| 同步模板残差 8,000 | 0.4986 | 0.4946 | 0.4995 | 0.5021 |
| 同步字段差分相位 8,000 | 0.4990 | 0.4966 | 0.4982 | 0.5033 |
| 报头差分相位 11,000 | 0.4994 | 0.4965 | 0.4982 | 0.5032 |

八种免训练读出均接近随机。波形加长和单包标量校准对这套固定池化与余弦读出没有实质收益。这个结果只覆盖免训练读出；神经网络输入长度的对照仍需使用一致训练配方。跨日期面板是源侧参考与查询时间隔离，独立留出测试未参与评分。

## 交接

N607 的 PID 1964732 已结束，完整评分 JSON 已读回。r01 的 NumPy/PyTorch 接口故障由 r02 的 DLPack 边界修复。100 轮训练继续独立运行。[正式实验记录](../20261119-phase1-satiq-ccs2023-sup-s2026092701-r03/report.md)
