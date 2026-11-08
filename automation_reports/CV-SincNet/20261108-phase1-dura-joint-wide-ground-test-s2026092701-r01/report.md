# DURA 联合恢复大容量七景测试

- run_id：`20261108-phase1-dura-joint-wide-ground-test-s2026092701-r01`
- group_id：`phase1-dura-joint-wide`；类别：`cvs`；阶段：`Phase1-restoration`
- 配置与矩阵：[experiment.json](experiment.json)；状态记录：[events.jsonl](events.jsonl)
- 当前登记状态：PLANNED（实际状态按events.jsonl及独立证据更新）

## 目的与对照

与固定三视图联合恢复模型保持数据与预算一致，将修正网络宽度从 192 增至 384；最终轮冻结后七景测试。

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

## 固定方案与对照

独立初始化联合恢复修正网络，宽度 384、可训练参数 7120514，另使用冻结 DURA v1.0 的 266951 参数。它保持原联合模型的三次波形／复数增益更新、训练一随机候选及推理四个固定反向配对候选平均；这是容量对照，不保证识别效果。与宽度 192 版使用相同源记录、三个固定源视图、seed、损失、AdamW 和每轮更新数；宽度改变导致实际计算量及训练轨迹不同。

合法 32768 个 L_s/U_s 物理 ID × 三视图 × 20 轮，batch32，最终轮冻结。恢复器训练不读 TX 标签、F 权重／输出、V、目标 IQ 或目标真值。控制器先执行真实源数据／冻结 DURA 权重六步检查及独立 112 条源记录 Tri-SIG-TIME SSL 无查询预测检查；训练后自动预测 clean 与六个 practical 各 168000 条，全部固定后独立 truth-last，与 DURA 配对报告 TX/RX/day、净纠错和 raw clean 相对 3%／绝对 3 个百分点差距。既有测试历史已曝光，不称新盲测或真实卫星结果。

N607 GPU1 每阶段至少 8192 MiB 可用、总计算进程少于 3、本任务同卡不新增其他 worker；不修改或停止其他 owner。源权限、非有限数、错误权重、输出冲突、smoke 或系统技术失败时停止本 run 并保留产物。代码 commit 24f144c5c59e9beb07a49b1526b1e23b95a91d91，具体命令和产物见 experiment.json。

本机四项聚焦检查通过，包括训练／评测宽度一致、初始化回到 DURA、宽度不匹配权重拒绝及 GPU 有限非零梯度；独立 P0/P1 审查未发现阻断。正式服务器源数据检查和训练尚待启动。
