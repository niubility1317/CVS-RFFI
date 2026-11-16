# Tri 有标签 CE 与完整监督目标的源域梯度诊断

本固定源域诊断检查现有实际更新保护的一个边界：保护锚点是完整监督目标，但用户特别关心有标签 CE。十类和六源各使用原论文同数据角色、同 seed、固定重排的 source-only scratch E20，各取预固定的前三个 L/U 批次；L 标签可见、U 真值隐藏，仅 CPU FP32 两线程，模型副本不执行训练更新，不读 V、目标或 U 真值。

复用原有普通有标签 CE 张量与完整 L/U 梯度，记录 CE 与完整 L、原 U、原保护后 U 的梯度关系；另外对相同参数用空 AdamW 动量、物理单位和 clip5 的可丢弃参数影子比较 L-only 与原保护 L+U 的首步增量，并记录该增量相对 CE 的方向。6 批各新增一次 CE 梯度，总计 24 次包含原诊断的影子优化器更新；影子和源模型均不保存为训练权重。真实 E20 无 query 检查和首个固定批次直接作为 smoke，成功后不间断继续。

该诊断只支持固定 E20 上六个相关批次的首步解释，不能推断有历史动量的后续训练、U 真值正确率或目标成绩；seed 先前目标曝光。身份、协议、非有限、梯度一致性或 guard backward 不匹配时停止，不改动任何健康训练。执行 commit、环境、命令、来源、输出和预期记录见 experiment.json。

独立 P0/P1 审查发现 CE 捕获范围早于最终 hard-PL 外层 hook：已先用嵌套测试复现，再移动捕获至最终 hook 并保留原输入能力标志；定点复核通过。发布、真实数据 smoke、远端日志和结果仍需独立核验。

# Tri 有标签 CE 与完整监督锚点源域诊断

- run_id：`20261116-diagnostic-pamf-tri-ce-anchor-e20-s2026092701-r01`
- group_id：`phase1-pamf-repl-weight-clean`；类别：`diagnostic`；阶段：`Phase1`
- 配置与矩阵：[experiment.json](experiment.json)；状态记录：[events.jsonl](events.jsonl)
- 当前登记状态：PLANNED（实际状态按events.jsonl及独立证据更新）

## 目的与对照

十类和六源原始 E20 各三批，测量 CE 与完整 L、U 梯度及 fresh-moment AdamW 实际增量关系。

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
