# Source-trained weighting of native and full-rate PAMF-SUP embeddings

- run_id：`20261119-diagnostic-satiq-pamf-sup-weighted-fusion-s2026111907-r01`
- group_id：`phase1-satiq-ccs2023-fully-supervised`；类别：`diagnostic`；阶段：`Phase1`
- 配置与矩阵：[experiment.json](experiment.json)；状态记录：[events.jsonl](events.jsonl)
- 当前登记状态：ANALYZED（实际状态按events.jsonl及独立证据更新）

## 目的与对照

Estimate a fixed 320-dimensional two-block cosine mixture from source L, and measure both branches and fusion on known source V.

## 数据、seed与模型来源

实际数据契约、权限例外、完整seed角色、checkpoint来源和选择规则见experiment.json。
逐行配置通过config_ref/resolved_config_ref定位；待补项必须在对应生命周期补齐。

## 执行与存储

命令、环境、CWD、commit、launch owner、输出和日志路径见experiment.json。
实际PID/GPU、读取时间、remote readback、失败或替代关系在此追加，并用record命令记录证据指针。

## 结果与覆盖

在原有 L 登记和 8,192 条 L 校正记录之外，另抽 8,192 条互异有标签训练包拟合融合。候选是全速率余弦比分的 0.10、0.25、0.50、0.75、0.90。按这部分 L 的同星真假对 AUC 选择。固定将两块的向量分别乘以平方根权重，保持 160+160 维，点积等于两个余弦分数的加权和。在源 V 上固定使用已选权重，和相同分数定义的等权模型比较。

检查点为绝对和相对 I/Q 两个第 5,000 步权重，以及绝对 I/Q 原型版首轮权重。N607 GPU 5 的 PID 2014961 已核实。留出 T 未读取。[此前等权拆分](../20261119-diagnostic-satiq-pamf-sup-enrolled-head-s2026111907-r02/report.md)

## 源验证结果

| 检查点 | 原生 160 AUC | 新增 160 AUC | 等权分数 AUC | L 选权后 AUC |
|---|---:|---:|---:|---:|
| 绝对 I/Q，5,000 步 | 0.5882 | 0.5174 | 0.5737 | 0.5872 |
| 相对 I/Q，5,000 步 | 0.5894 | 0.5184 | 0.5718 | 0.5882 |
| 绝对 I/Q，28,225 步 | 0.6844 | 0.5275 | 0.6320 | 0.6816 |

三组均在 L 上选择 10% 的全速率比分权重。第 1 轮权重的 49,814 包验证 Top-1 从等权的 0.2108% 升到 0.2228%，原生单路为 0.2369%。减轻弱支路的权重可恢复部分融合损失，新增支路仍未提供稳定的净收益。本项作为此前新增支路的归因记录。新一轮扩展按用户要求只使用原生五分支。[原生宽度方案](../20261119-diagnostic-satiq-pamf-sup-native-widths-s2026092701-r01/report.md)

## 交接

记录已完成、当前run/commit、证据路径、阻塞与下一步；恢复先查原run，不重复启动。
