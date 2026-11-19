# SatIQ PAMF-SUP six-path leave-one-out prototypes

- run_id：`20261119-diagnostic-satiq-pamf-sup-branch-prototypes-s2026092701-r02`
- group_id：`phase1-satiq-ccs2023-fully-supervised`；类别：`diagnostic`；阶段：`Phase1`
- 配置与矩阵：[experiment.json](experiment.json)；状态记录：[events.jsonl](events.jsonl)
- 当前登记状态：RUNNING（实际状态按events.jsonl及独立证据更新）

## 目的与对照

Compare original PAMF-SUP with its own centered, bounded, supervised leave-one-out branch prototypes under the same L/V waveform and model seed.

## 数据、seed与模型来源

实际数据契约、权限例外、完整seed角色、checkpoint来源和选择规则见experiment.json。
逐行配置通过config_ref/resolved_config_ref定位；待补项必须在对应生命周期补齐。

## 执行与存储

命令、环境、CWD、commit、launch owner、输出和日志路径见experiment.json。
实际PID/GPU、读取时间、remote readback、失败或替代关系在此追加，并用record命令记录证据指针。

## 结果与覆盖

保留 PAMF-SUP 的融合分类、同星波束 CE、两路辅助 CE 和同包 KL。只增加原生五分支及全速率支路的留一原型 CE，权重 0.5。每批仍使用 8 个真实波束、每类 4 包。每个锚点用该波束的另外 3 个包作为正原型。其它波束的原型来自该波束当前批次的 4 个包。分类头和后续推理接口不变。

原型只在训练批内计算。每条支路先去掉当前 L 批次的均值，再以残差平方范数加 0.1² 的平方根作为归一化分母。用 3 批真实 L 包，在初始化及两个冻结 PAMF-SUP 检查点上测量梯度。权重 0.5 时，全速率投影在初始化的辅助梯度为原目标的 12.2% 至 18.8%。在第 5,000 步冻结检查点，时域投影的抽查峰值为原目标的约 1.01 倍，直接按残差范数归一的未加权峰值为 773.7 倍。梯度校准未使用 V。[首版 preflight](../20261119-diagnostic-satiq-pamf-sup-branch-prototypes-s2026092701-r01/report.md)

## 交接

N607 物理 GPU 4 的 PID 1996981 已通过真实 IQ 检查并写出各训练 CE。第一轮 28,225 步，与原始 PAMF-SUP 使用相同训练包顺序和源验证点。SatIQ 作为同数据评分对照，留出测试保持隔离。[原 PAMF-SUP](../20261119-phase1-satiq-ccs2023-sup-s2026092701-r03/report.md)
