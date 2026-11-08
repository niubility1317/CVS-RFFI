# 联合恢复与条件扩散的同源波形诊断

- run_id：`20261108-phase1-joint-diffusion-source-pilot-s2026092701-r01`
- group_id：`phase1-joint-diffusion-source-pilot`；类别：`diagnostic`；阶段：`Phase1`
- 配置与矩阵：[experiment.json](experiment.json)；状态记录：[events.jsonl](events.jsonl)
- 当前登记状态：PLANNED（实际状态按events.jsonl及独立证据更新）

## 目的与对照

比较两条固定源训练模型与同一冻结 DURA 在合法源 pilot 的波形误差；无 F 或目标读取，不影响既定七景测试。

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

## 固定比较及边界

固定 DURA 和两个已完成源训练的 checkpoint：联合波形／增益模型最终 20 轮、条件残差扩散模型最终 20 轮。在同一条源 received 上计算 DURA 上下文一次，并分别输出两个候选 IQ；用合法源 clean 计算归一化波形误差，统计各分组的均值和每记录相对 DURA 改善／变差数。源 pilot 共 672 条，RX4/6 与 RX8 各七景、每组 48 条；与两个候选的源训练 fit ID 均互斥，但历史已探索，DURA 可能见过这些记录。

此诊断不读取 F 权重或输出、TX 标签、V、目标 IQ 或真值，不更新模型，不根据源结果取消或更改两条预登记正式测试；波形误差不等于身份准确率。GPU2 至少 8192 MiB 空闲、总计算进程少于 3 且本任务同卡无其他 worker；真实 DURA 和两个 checkpoint 的两条源记录 smoke 为同一进程首步。非有限值、来源不匹配、输出冲突或系统技术故障停止本诊断并保留产物。

固定诊断代码 commit ab70a7122815ca33faff3cce2ddd37126bc84623；源模型 commit 分别为 0cddb5ff260381205a5445d04507ed211fac9989 与 c48f5bebc921b4e2fd97341cb8eebcf6a627dbe2。三项聚焦检查和一次独立 P0/P1 审查通过；精确命令、资源及输出见 experiment.json。

## 已完成源域诊断

真实源 IQ 和两个固定权重 smoke 通过，672 条源记录比较完成。两个候选均呈现 RX8 受损场景平均波形误差下降、RX4/6 六景均值上升及 clean 波形误差上升；逐场景和记录数见[分析](analysis.md)。这不是目标识别评分，既定正式测试继续。
