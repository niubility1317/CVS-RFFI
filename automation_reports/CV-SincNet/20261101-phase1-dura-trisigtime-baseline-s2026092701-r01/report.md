# Phase1 既有恢复网络的 Tri-SIG-TIME 同基座对照

- run_id：`20261101-phase1-dura-trisigtime-baseline-s2026092701-r01`
- group_id：`phase1-dura-trisigtime-matched-baseline`；类别：`cvs`；阶段：`Phase1-baseline-evaluation`
- 配置与矩阵：[experiment.json](experiment.json)；状态记录：[events.jsonl](events.jsonl)
- 当前登记状态：PLANNED（实际状态按events.jsonl及独立证据更新）

## 目的与对照

将已有 Phase1 DURA 恢复 IQ 经冻结 Tri-SIG-TIME SSL 重新识别，与新候选的同输入原始预测按同物理 ID 对照。

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

## 已核实状态（2026-11-01）

已有 Phase1 DURA 的 `native_map_prior/prediction` 包含 clean 加六景恢复 IQ；其 168000 个物理 ID 与本次 Phase1 正式参考逐项同序。后续只取恢复 IQ，用原始 Phase1 三视图 Tri-SIG-TIME SSL 固定权重重新分类；旧识别器 logits、features 和分数均不作为本对照。原始信号决策取新候选已完整固定的同一 Tri-SIG-TIME 预测。

代码版本 `5dc285d0bbd44462178f3715f495bf32dc9ac918` 已发布，归档两端 SHA-256 一致。真实 Tri-SIG-TIME 权重与两条源训练记录的 CPU smoke 已通过；未训练或更新模型，未读测试 IQ 或 TX 标签。

控制器 PID 2197520 已独立读回，正等待第一条新候选控制器与 GPU worker 结束。七视图对照预测尚未开始；全部预测固定后才独立评分。当前没有同基座 DURA 新分数。
