# SatIQ source-only embedding centering and whitening

- run_id：`20261119-diagnostic-satiq-source-calibrated-readout-s2026111907-r01`
- group_id：`phase1-satiq-ccs2023-fully-supervised`；类别：`diagnostic`；阶段：`Phase1`
- 配置与矩阵：[experiment.json](experiment.json)；状态记录：[events.jsonl](events.jsonl)
- 当前登记状态：ANALYZED（实际状态按events.jsonl及独立证据更新）

## 目的与对照

Compare untreated, mean-centered, and partially whitened readouts for three matched first-epoch source checkpoints on the full random and chronological panels.

## 数据、seed与模型来源

实际数据契约、权限例外、完整seed角色、checkpoint来源和选择规则见experiment.json。
逐行配置通过config_ref/resolved_config_ref定位；待补项必须在对应生命周期补齐。

## 执行与存储

命令、环境、CWD、commit、launch owner、输出和日志路径见experiment.json。
实际PID/GPU、读取时间、remote readback、失败或替代关系在此追加，并用record命令记录证据指针。

## 结果与覆盖

在 8,192 条互异训练包上估计固定中心和协方差。PAMF 两路各自计算，SatIQ 对 512 维嵌入计算。34,785 个互异源验证包覆盖两套固定面板，所有行使用各自第 1 轮的冻结权重。

| 方法 | 读出 | 论文式同星 AUC K1／K16 | 跨日期同星 AUC K1／K16 | K16 Top-1 论文式／跨日期 |
|---|---|---:|---:|---:|
| SatIQ | 原读出 | 0.5320／0.5652 | 0.5245／0.5524 | 0.188%／0.843% |
| SatIQ | 中心化 | 0.5313／0.5659 | 0.5234／0.5507 | 0.211%／0.843% |
| SatIQ | 部分白化 | 0.5341／0.5755 | 0.5253／0.5608 | 0.219%／0.843% |
| PAMF-SUP | 原读出 | 0.5161／0.5186 | 0.5106／0.5140 | 0.180%／0.452% |
| PAMF-SUP | 中心化 | 0.5135／0.5327 | 0.5056／0.5233 | 0.110%／0.740% |
| PAMF-SUP | 部分白化 | 0.5124／0.5344 | 0.5058／0.5269 | 0.180%／0.678% |
| PAMF-Metric | 原读出 | 0.5252／0.5255 | 0.5304／0.5394 | 0.188%／0.596% |
| PAMF-Metric | 中心化 | 0.5256／0.5591 | 0.5277／0.5585 | 0.227%／0.802% |
| PAMF-Metric | 部分白化 | 0.5247／0.5702 | 0.5241／0.5670 | 0.305%／0.946% |

论文式面板 1,375 类、12,785 条查询。跨日期面板以 2023 年 4 月 3 日 00:00 UTC 隔离参考和查询，覆盖 309 类、4,865 条查询。后者测试的是参考与查询的日期隔离，训练仍使用论文式源 L，故不等同于模型训练之外的新日期。

三组未校正的论文式指标与训练时登记的第 1 轮值一致。部分白化让 PAMF-Metric 的 K16 同星 AUC 分别提升 0.0447 与 0.0275。与同样白化的 SatIQ 相比，论文式 K16 相差 −0.0053，跨日期相差 +0.0062。K1 与绝对 Top-1 仍弱。一轮源侧结果验证了 K16 读出收益，最终独立测试与训练后期检验仍待完成。

## 交接

物理 GPU 4，PID 1981336，已验证退出及完整 JSON。三组比较均使用已登记的冻结权重和源数据角色。[正式训练](../20261119-phase1-satiq-ccs2023-sup-s2026092701-r03/report.md)
