# PAMF-SUP source enrolled classification by path

- run_id：`20261119-diagnostic-satiq-pamf-sup-enrolled-head-s2026111907-r01`
- group_id：`phase1-satiq-ccs2023-fully-supervised`；类别：`diagnostic`；阶段：`Phase1`
- 配置与矩阵：[experiment.json](experiment.json)；状态记录：[events.jsonl](events.jsonl)
- 当前登记状态：FAILED（实际状态按events.jsonl及独立证据更新）

## 目的与对照

Decompose native 160, extra full-rate 160 and fused 320 using fixed labelled L prototypes on known source V. Four matched checkpoints at 5000 and 20000 steps.

## 数据、seed与模型来源

实际数据契约、权限例外、完整seed角色、checkpoint来源和选择规则见experiment.json。
逐行配置通过config_ref/resolved_config_ref定位；待补项必须在对应生命周期补齐。

## 执行与存储

命令、环境、CWD、commit、launch owner、输出和日志路径见experiment.json。
实际PID/GPU、读取时间、remote readback、失败或替代关系在此追加，并用record命令记录证据指针。

## 结果与覆盖

使用 3,103 个已知波束的源 L 样本登记，每类最多 16 个包。分别比较原生 160 维、全速率 160 维和两路融合 320 维的余弦原型分类，并保留训练得到的融合分类头作对照。原始、中心化和部分白化读出使用相同的标签和验证包。校正统计只拟合在 8,192 条源 L 记录上。

第 5,000 和第 20,000 步的四个冻结检查点已进入 N607 GPU 5 的同一评分进程，PID 为 2007351。独立留出测试保持隔离。[原 PAMF-SUP](../20261119-phase1-satiq-ccs2023-sup-s2026092701-r03/report.md) · [分支原型训练](../20261119-diagnostic-satiq-pamf-sup-branch-prototypes-s2026092701-r02/report.md)

## 交接

记录已完成、当前run/commit、证据路径、阻塞与下一步；恢复先查原run，不重复启动。

首版读出因 N607 的 NumPy 整数数组直接索引 Torch 张量失败，未产生评分文件。新版本对两个索引显式使用 DLPack 转换。四个源检查点、样本及评分口径一致。
