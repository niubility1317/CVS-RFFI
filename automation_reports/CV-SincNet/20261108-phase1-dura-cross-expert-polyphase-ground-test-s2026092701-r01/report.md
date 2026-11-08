# 多相时频跨恢复器七景测试

- run_id：`20261108-phase1-dura-cross-expert-polyphase-ground-test-s2026092701-r01`
- group_id：`phase1-dura-cross-expert-polyphase-ground`；类别：`cvs`；阶段：`Phase1-restoration`
- 配置与矩阵：[experiment.json](experiment.json)；状态记录：[events.jsonl](events.jsonl)
- 当前登记状态：PLANNED（实际状态按events.jsonl及独立证据更新）

## 目的与对照

独立多相卷积实现训练与评测时频跨恢复器；不改动原版，自动完成七景预测及独立评分。

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

## 固定执行与证据边界

代码 b3356e8dbbc15f2934dadaf19cfbc0b96e9e78b8。独立多相实现用于训练和正式推理；同源 17632 个 ID、三视图、固定 20 轮，最终 checkpoint 冻结后自动进行 clean 和六个 practical 场景的 168000 条预测，全部完成后独立 truth-last 评分。模型基座为冻结 Tri-SIG-TIME SSL；DURA 和 F-free 直接恢复器仅按登记权限使用。源域 no-query checkpoint smoke 是训练首步；已有 12 项针对性检查及一次 P0/P1 评审和针对修复的定点复核。

短程多相训练与原始实现不逐位等价；该独立版本不得冒充原版或继承原版测试结果。GPU2 至少 8192 MiB 可用、总计算进程少于 3、本任务同卡最多一 worker；保护其他任务。仅按已登记系统技术失败停止规则操作，冻结后不使用目标分数调参、选择或取消测试。此前目标结果已观察，结果不称独立盲测或真实星地性能。
