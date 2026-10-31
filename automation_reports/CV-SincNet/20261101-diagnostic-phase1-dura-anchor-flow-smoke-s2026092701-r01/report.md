# Phase1 冻结恢复器锚定生成轨迹真实权重检查

- run_id：`20261101-diagnostic-phase1-dura-anchor-flow-smoke-s2026092701-r01`
- group_id：`phase1-frozen-dura-anchor-flow-smoke`；类别：`cvs`；阶段：`Phase1-source-diagnostic`
- 配置与矩阵：[experiment.json](experiment.json)；状态记录：[events.jsonl](events.jsonl)
- 当前登记状态：PLANNED（实际状态按events.jsonl及独立证据更新）

## 目的与对照

真实冻结 Phase1 DURA 与 Tri-SIG-TIME SSL，使用两条 L_s 源训练 IQ 验证锚定轨迹初始恒等、梯度和严格重载。

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

## 实际 Phase1 权重检查

VERIFIED／ANALYZED_SOURCE_ONLY。N607 CPU 使用原始 Tri-SIG-TIME SSL 固定权重、原始 Phase1 DURA stage2 固定权重及两条受扰 L_s 源样本；只取这两条源 IQ，没有读取 U_s IQ、TX 标签、V 或测试 IQ。

零输出速度场下，两条反向噪声轨迹均值相对 DURA 起点的最大绝对误差为 `5.9605e-8`；IQ 损失和未加权分类分布 KL 对生成场输出层的梯度范数分别为 `0.2016` 和 `153.0847`。完成一次检查性质的生成层优化后，严格保存／重载及实际 F 推断通过，两套原模型参数和状态保持冻结。

语义项的梯度量级在这两条样本上明显偏大，不能直接假定 0.1 权重适合锚定模型。接下来应只用预固定的更多源样本核查不同条件下的分项梯度，再确定训练权重；两条记录不足以估计泛化性能。此前开发测试结果已被观察，后续使用同一测试集需披露非盲测。
