# 多相时频跨恢复器七景测试

- run_id：`20261108-phase1-dura-cross-expert-polyphase-ground-test-s2026092701-r02`
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

## 固定执行

代码 b3356e8dbbc15f2934dadaf19cfbc0b96e9e78b8。r01 因发布包缺失源数据模块而在访问数据前失败；本运行只修正发布包依赖，保留相同训练、测试矩阵和权限，独立目录输出。先运行真实源 checkpoint smoke，再进行固定 20 轮训练；冻结后 clean 和六个 practical 场景全部预测完成后独立 truth-last 评分。训练和推理均使用多相实现，不能继承原版结果。GPU2 需至少 8192 MiB 可用、少于 3 个计算进程且本任务同卡最多一 worker。
