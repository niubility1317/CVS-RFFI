# DURA 联合恢复：源信道视图轮换训练

- run_id：`20261108-phase1-dura-joint-diverse-source-s2026092701-r01`
- group_id：`phase1-dura-joint-diverse`；类别：`cvs`；阶段：`Phase1-restoration`
- 配置与矩阵：[experiment.json](experiment.json)；状态记录：[events.jsonl](events.jsonl)
- 当前登记状态：PLANNED（实际状态按events.jsonl及独立证据更新）

## 目的与对照

与固定三视图联合恢复保持模型与优化预算一致，训练时轮换 30 组合法源信道视图；最终轮冻结后七景测试。

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

## 预登记与比较口径

独立源训练从零初始化联合修正网络，冻结 DURA v1.0；每个训练 epoch 使用同一批 32768 个合法 L_s/U_s 源物理 ID 的三个预生成模拟视图。固定 20 轮，依次轮换视图 01–30，30 种视图各使用两次。与固定三视图联合模型匹配相同网络、物理 ID、模型／顺序／噪声种子、AdamW、学习率、batch32 和更新次数；具体输入不同，不能将输出或模型权重视为相同。训练不读取 TX 标签、F 权重或输出、V、目标 IQ 或目标真值。训练模拟视图不是新增独立物理记录。

独立真实权重启动检查使用 64 条合法源记录和一次 112 条源记录的冻结 Tri-SIG-TIME SSL 无查询预测检查。之后固定最终第 20 轮；自动预测 clean 与六个 practical 视图各 168000 条，全部预测固定后独立 truth-last，报告逐景准确率、DURA 配对净纠错、TX/RX/day 和距 raw clean 的相对 3% 与绝对 3 个百分点差距。历史测试曾曝光，不称新独立盲测或真实卫星结果。

源训练及恢复推理使用 N607 GPU6；每阶段至少 8192 MiB 可用、总计算进程少于 3，本任务同卡最多一个训练／预测 worker，不修改或停止其他 owner。启动前执行真实 source-only 权重检查，数据权限、非有限数、输出冲突、接口失败或系统技术故障停止本 run 并保留证据。代码 commit 0a3179a1cc71eec8b72dd420a0ff9e50462d5fc4；精确路径、命令、固定种子和输出见 experiment.json。

本机合成 GPU 六步训练流程、视图日程及来源约束等 6 项聚焦检查通过；一次独立 P0/P1 审查未发现阻断。服务器真实权重 smoke 及正式实验尚待启动，合成检查不是正式训练或性能结果。
