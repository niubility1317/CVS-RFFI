# PAMF-SUP relative full-rate 160-dimensional branch

- run_id：`20261119-diagnostic-satiq-pamf-sup-relative-full-rate-s2026092701-r01`
- group_id：`phase1-satiq-ccs2023-fully-supervised`；类别：`diagnostic`；阶段：`Phase1`
- 配置与矩阵：[experiment.json](experiment.json)；状态记录：[events.jsonl](events.jsonl)
- 当前登记状态：ANALYZED（实际状态按events.jsonl及独立证据更新）

## 目的与对照

On branch-prototype PAMF-SUP, use packet-gain-invariant ordered five-channel relative IQ for the added 160 dimensions, preserving native 160 and all supervision.

## 数据、seed与模型来源

实际数据契约、权限例外、完整seed角色、checkpoint来源和选择规则见experiment.json。
逐行配置通过config_ref/resolved_config_ref定位；待补项必须在对应生命周期补齐。

## 执行与存储

命令、环境、CWD、commit、launch owner、输出和日志路径见experiment.json。
实际PID/GPU、读取时间、remote readback、失败或替代关系在此追加，并用record命令记录证据指针。

## 结果与覆盖

全速率支路仍读取 11,000 点 I/Q，保留 11 个符号各 5 个时间区间。五维卷积输入换为对数相对功率和延迟 1 点、25 点的复相关实虚部。先去除包级复直流偏置，再按包能量归一化。在 25 MS/s 下，延迟 25 点对应 1 μs。原生 160 维、融合方式、参数尺寸、原型权重 0.5、采样顺序均与原型版一致。

真实训练包核对通过。两版共享状态逐张量相等，初始原生 160 维输出一致，新增 160 维对整体复增益和偏置保持稳定，并保留尾部符号变化。N607 物理 GPU 1、PID 2006780 已运行实包检查和训练。[原型版对照](../20261119-diagnostic-satiq-pamf-sup-branch-prototypes-s2026092701-r02/report.md)

## 第 5,000 步

| 方法 | 同星 K1 AUC | 同星 K16 AUC | 分类头 CE |
|---|---:|---:|---:|
| 原型版的绝对 I/Q 支路 | 0.5211 | 0.5244 | 8.0800 |
| 原型版的相对 I/Q 支路 | 0.5260 | 0.5303 | 8.0442 |

相对 I/Q 版本在整套验证面板上略有改善。单独检查新增支路时，4,096 个验证包的平均余弦仍为 0.999964，有效维度约 2.77，独立分类只输出了一个波束。这些观察表明目前无法把联合提升归因为新增 160 维已修复，模型继续训练。

## 固定预算结束

| 步数 | 绝对 I/Q K1／K16 | 相对 I/Q K1／K16 | 绝对／相对 I/Q 分类头 CE |
|---|---:|---:|---:|
| 20,000 | 0.5457／0.5546 | 0.5476／0.5575 | 7.6674／7.6704 |
| 28,225 | 0.5517／0.5646 | 0.5500／0.5632 | 7.5440／7.5705 |

计划的整轮已验证完成，早期收益没有延续到同预算终点。该结果作为以前的额外支路对照保留。[原生五分支扩维方案](../20261119-diagnostic-satiq-pamf-sup-native-widths-s2026092701-r01/report.md)

## 交接

记录已完成、当前run/commit、证据路径、阻塞与下一步；恢复先查原run，不重复启动。
