# 恢复性能受控实验：同MSAT结构与预算，增加逐维源F特征保真：GPU7设备故障技术恢复r02

- run_id：`20261101-phase1-Restore-MSAT-VarF-E30-s2026092701-r02`
- group_id：`restoration-source-Fvariance-controlled-single30`；类别：`cvs`；阶段：`Phase1`
- 配置与矩阵：[experiment.json](experiment.json)；状态记录：[events.jsonl](events.jsonl)
- 当前登记状态：PLANNED（实际状态按events.jsonl及独立证据更新）

## 目的与对照

根据原源特征方差与判决偏离关联，只改变F0特征目标：.1(cos+逐维源方差标准化MSE)，保留.5IQ梯度cap；scratch单阶段30、正确原Main七景自动测试。 科学配方不变；r01第30轮GPU设备故障，无已训练checkpoint，本次独立从零30，逐轮保存恢复态。

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

## r01设备故障的同配方恢复

仅解决设备故障与训练态丢失，VarF科学配方保持。r01末轮GPU7/驱动不可用、无已训练checkpoint；不从E0冒充续训，也不加载旧MSAT。新root从零单阶段30、相同源数据/初值/顺序/损失/优化器，固定final30后完整Main两总体七景与独立truth-last。

只查询已授权GPU4/6，按同显存/总进程上限准入，并用UUID避免设备枚举变化；不重置GPU7或更改其他owner。每完整epoch原子保存模型/优化器/RNG到独占PRIVATE文件，只供技术恢复，不能选模。9聚焦检查及原故障定点P0P1复审PASS。若同指纹再次出现，保留停止，不盲目再发。

## 恢复尝试仍因平台故障停止

FAILED / NO_PERFORMANCE_RESULT：CUDA初始化在source smoke前失败，0epoch/0update/0正式预测。独立fresh-process设备微探针证明GPU4编号/UUID、GPU6编号均无法初始化CUDA；没有模型/真实IQ输入。已停止进一步发布，未重置设备或干预其他owner。全部失败证据保留，原性能目标未达到。详见analysis.md/failure_summary.json。
