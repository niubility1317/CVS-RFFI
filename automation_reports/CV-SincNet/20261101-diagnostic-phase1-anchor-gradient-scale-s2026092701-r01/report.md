# Phase1 锚定生成轨迹的源梯度尺度核查

- run_id：`20261101-diagnostic-phase1-anchor-gradient-scale-s2026092701-r01`
- group_id：`phase1-anchor-source-gradient-scaling`；类别：`cvs`；阶段：`Phase1-source-diagnostic`
- 配置与矩阵：[experiment.json](experiment.json)；状态记录：[events.jsonl](events.jsonl)
- 当前登记状态：PLANNED（实际状态按events.jsonl及独立证据更新）

## 目的与对照

固定 Phase1 源 clean、LOS 和 BLOCKED 样本，各 32 条，测量波形、基座语义与特征损失对零初始化生成场的梯度尺度。

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

## 固定源样本梯度核查

VERIFIED／ANALYZED_SOURCE_ONLY。读取 Phase1 L_s/U_s 的 512 条源 clean/native IQ，按原缓存顺序从 clean、LOS、BLOCKED 三个互斥组各取前 32 条计算；两个冻结基座和生成场均无更新，不读取 TX、V 或测试 IQ。以下为零输出生成场末层权重的未加权梯度范数：

| 源状态 | IQ 梯度范数 | 分类 KL 梯度范数 | 两梯度余弦 |
|---|---:|---:|---:|
| clean | 0.005205 | 0.339064 | -0.4409 |
| LOS | 0.040654 | 0.621850 | 0.0388 |
| BLOCKED | 0.075373 | 42.328598 | 0.3605 |

按事先登记的 LOS／BLOCKED 梯度范数比中位数并限制在 [1e-5, 1e-2]，建议 KL 权重为 0.01。即使按该权重，BLOCKED 组的 KL 梯度范数仍约为 IQ 项的 5.6 倍；clean 组两项梯度方向相反。该规则的局限应保留，不能把 0.01 称为所有条件下的平衡值。

结果提示可以研究源训练中对不同目标梯度进行约束或均衡；这不是测试性能、独立泛化或优化器更新后性能的证明。源分组为固定小样本，LOS 组本次全部来自 L_s；先前 Phase1 开发测试已被观察。
