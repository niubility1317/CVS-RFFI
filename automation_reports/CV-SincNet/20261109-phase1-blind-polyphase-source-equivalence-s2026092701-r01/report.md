# 盲恢复多相执行源域一致性检查

- run_id：`20261109-phase1-blind-polyphase-source-equivalence-s2026092701-r01`
- group_id：`phase1-blind-polyphase-source-equivalence`；类别：`diagnostic`；阶段：`Phase1`
- 配置与矩阵：[experiment.json](experiment.json)；状态记录：[events.jsonl](events.jsonl)
- 当前登记状态：PLANNED（实际状态按events.jsonl及独立证据更新）

## 目的与对照

用同一源预检权重比较原始和多相计算的源 IQ 与冻结识别器决策，并比较两者各自六步预检权重。

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

## 固定比较矩阵

对已探索源域 pilot 中 RX4/6 与 RX8 各七视图各 8 条，共 112 条记录；不读取 TX 真值、V、目标 IQ 或目标真值。固定原版 64 物理 ID × 3 视图六步 smoke 权重，同时加载原始卷积和多相卷积，比较逐样本 IQ 最大绝对差、冻结 Tri-SIG-TIME SSL logits 最大绝对差及决策不一致数；另比较两版各自独立六步预检所得权重。识别器仅在此次源域无查询比较读取，未用于恢复网络训练。

GPU 前向计时包含同步、不包含识别器与数据传输，固定执行顺序且无热身，只能描述本次运行。112 条源域记录不能证明完整目标测试预测一致或 20 轮训练轨迹完全相同；两版完整训练与七景评测继续独立执行。

GPU7 需不少于 8192 MiB 空闲、少于 3 个 GPU 进程且本任务同卡最多一个 worker，保护其他任务。代码 eda2f5964cc0c76641787cb97c02b2ae47489dd1；具体命令、输入、停止条件与输出见 experiment.json。局部两项检查及独立 P0/P1 静态审查通过，远端执行待核验。

## 服务器执行与结果

发布归档两端 SHA-256 均为 ed898056357029d1ed94195154dae3da6f8ca3f25a89cce2029f016f2bde3e3a，服务器编译和导入通过。GPU7 启动时空闲 24110 MiB、无计算进程；112 条源记录全部完成，日志和 result.json 已核实。任务较短，结束前未捕获独立存活 PID/GPU 记录。结果详见 [源域比较](analysis.md)，不外推到完整训练或目标预测。
