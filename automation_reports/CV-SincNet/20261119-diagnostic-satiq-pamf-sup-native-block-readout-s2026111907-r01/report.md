# PAMF-SUP native and expanded coordinate ablation

- run_id：`20261119-diagnostic-satiq-pamf-sup-native-block-readout-s2026111907-r01`
- group_id：`phase1-satiq-ccs2023-fully-supervised`；类别：`diagnostic`；阶段：`Phase1`
- 配置与矩阵：[experiment.json](experiment.json)；状态记录：[events.jsonl](events.jsonl)
- 当前登记状态：ANALYZED（实际状态按events.jsonl及独立证据更新）

## 目的与对照

Score the original 160 block, extension alone, and block-calibrated fusion on locked source panels for the four width variants at step 5000.

## 数据、seed与模型来源

实际数据契约、权限例外、完整seed角色、checkpoint来源和选择规则见experiment.json。
逐行配置通过config_ref/resolved_config_ref定位；待补项必须在对应生命周期补齐。

## 执行与存储

命令、环境、CWD、commit、launch owner、输出和日志路径见experiment.json。
实际PID/GPU、读取时间、remote readback、失败或替代关系在此追加，并用record命令记录证据指针。

## 结果与覆盖

四个第 5,000 步模型已完成冻结读出。相同 8,192 个源训练包拟合白化，相同 34,785 个物理源验证包组成两个固定面板。下表为原论文式面板的同星 K16 AUC。独立白化融合沿用训练时原块与扩展块 80:20 的比分比例。

| 组 | 全向量原读出 | 整体白化 | 原 160 白化 | 扩展块白化 | 分块白化融合 | 扩展块有效维度 |
|---|---:|---:|---:|---:|---:|---:|
| 原生 160 | 0.5295 | 0.5781 | 0.5781 | — | — | — |
| 融合 256 | 0.5266 | 0.5796 | 0.5792 | 0.5736 | 0.5786 | 1.29 |
| 融合 320 | 0.5299 | 0.5829 | 0.5824 | 0.5739 | 0.5811 | 1.30 |
| 分支 320 | 0.5268 | 0.5788 | 0.5782 | 0.5765 | 0.5780 | 1.67 |

跨日期面板整体白化后的同星 K16 AUC 依次为 0.5692、0.5672、0.5700、0.5671。扩展坐标的不同包平均余弦约为 0.99964 至 0.99987。三种扩展在此训练阶段都主要跟随原 160 块，分块白化和融合没有显示独立互补收益。这里读取的是冻结的源特征，留出 T 未动。

## 交接

N607 GPU 1、PID 2029153 已核实运行后退出。四组的完整 JSON 独立读回。[原生宽度训练](../20261119-diagnostic-satiq-pamf-sup-native-widths-s2026092701-r01/report.md)
