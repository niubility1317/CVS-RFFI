# SatIQ PAMF-SUP class-frequency correction diagnostic

- run_id：`20261119-diagnostic-satiq-pamf-sup-prior-adjusted-s2026092701-r01`
- group_id：`phase1-satiq-ccs2023-fully-supervised`；类别：`diagnostic`；阶段：`Phase1`
- 配置与矩阵：[experiment.json](experiment.json)；状态记录：[events.jsonl](events.jsonl)
- 当前登记状态：RUNNING（实际状态按events.jsonl及独立证据更新）

## 目的与对照

Matched pure-supervision comparison isolating fixed source training-frequency logit offsets in PAMF-SUP.

## 数据、seed与模型来源

实际数据契约、权限例外、完整seed角色、checkpoint来源和选择规则见experiment.json。
逐行配置通过config_ref/resolved_config_ref定位；待补项必须在对应生命周期补齐。

## 执行与存储

命令、环境、CWD、commit、launch owner、输出和日志路径见experiment.json。
实际PID/GPU、读取时间、remote readback、失败或替代关系在此追加，并用record命令记录证据指针。

## 结果与覆盖

原版的首轮融合平均分数与训练标签对数频率相关达到 0.958，独立采样的 4,096 条验证包只预测出 12 个不同波束。本诊断把训练类频率作为固定对数偏置，应用于融合、同星、两路和一致性损失计算。模型和初值、采样顺序、30 倍余弦尺度、输入及损失权重均沿用 PAMF-SUP。没有三元组、合成增强或伪标签。

数据来自同一套源 L/V。校正频率取每类四包组数，早期及完整一轮的训练样本顺序与正式 PAMF-SUP 一致。固定 V 面板上分别记录原始 CE、加频率后的 CE、原始预测覆盖及集中度，以及去掉逐包变化后只用平均分数的 CE。同星 K1/K16 使用同样的原始身份嵌入。

## 交接

物理 GPU 5 的 PID 1989109 已通过实包检查并进入训练。计划 28,225 步，固定验证位于 5,000、20,000 和 28,225 步。正式百轮对照独立运行，留出测试隔离。[原 PAMF-SUP 分析](../20261119-phase1-satiq-ccs2023-sup-s2026092701-r03/report.md)
