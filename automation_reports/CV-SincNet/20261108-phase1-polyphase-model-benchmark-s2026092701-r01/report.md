# 多相分解完整恢复网络短程 GPU 训练对照

- run_id：`20261108-phase1-polyphase-model-benchmark-s2026092701-r01`
- group_id：`phase1-polyphase-model-benchmark`；类别：`diagnostic`；阶段：`Phase1`
- 配置与矩阵：[experiment.json](experiment.json)；状态记录：[events.jsonl](events.jsonl)
- 当前登记状态：PLANNED（实际状态按events.jsonl及独立证据更新）

## 目的与对照

固定合法源样本与初始状态，对比原始和多相分解完整网络的短程训练耗时、状态与源 IQ 输出差异。

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

## 整网短程对照排队

固定代码 commit `747b57d2999d2e3764cf922092e4375c14a64f28`。11 项针对性检查、独立 P0/P1 审查及修复定点复核通过；发布包两端一致、远端编译通过。N607 队列 PID 1321275 和实际工作目录已核实，GPU2 当前有三个其他 owner 计算进程，队列等待小于 3 个且显存充足后执行；不改变现有模型和其他进程。

两种完整时频网络以相同 seed、模型状态、源 IQ 批次及 AdamW 初始化，分别热身一步、按两种执行顺序各训练四步；比较真实源 IQ 输出、损失、模型和优化器状态，以及 CPU/GPU/采样随机状态。只用合法源缓存和冻结 DURA/direct，真实 checkpoint 的 8 条源样本检查是首步；不读 F/TX/V/目标、不保存网络。单算子结果已发现浮点差异，本对照不承诺位级一致，也不能代替完整训练或七景身份测试。
