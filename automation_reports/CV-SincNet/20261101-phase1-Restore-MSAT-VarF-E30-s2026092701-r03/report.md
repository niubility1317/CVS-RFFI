# MSAT 标准化源 F 特征指导：N607 CUDA 恢复后同配方单阶段30轮 r03

- run_id：`20261101-phase1-Restore-MSAT-VarF-E30-s2026092701-r03`
- group_id：`restoration-source-Fvariance-controlled-single30`；类别：`cvs`；阶段：`Phase1`
- 配置与矩阵：[experiment.json](experiment.json)；状态记录：[events.jsonl](events.jsonl)
- 当前登记状态：PLANNED（实际状态按events.jsonl及独立证据更新）

## 目的与对照

原MSAT_VarF科学配方不变。N607 GPU4在原Torch环境实际CUDA分配与计算恢复PASS，GPU7设备查询恢复；独立新root从零单阶段30。两次失败root保留；每轮技术恢复快照，无学习态继承。

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

## 平台恢复后继续原性能实验

2026-11-01 05:11 UTC，N607 GPU 4 在原 Torch 2.1.0+cu121/CUDA 12.1 中实际分配、同步和计算 PASS；GPU 7 设备查询也已恢复。设备恢复来自外部状态变化，本任务未重置设备、改驱动或干预其他 owner。旧 r01/r02 的失败证据完整保留，均无正式性能结果。

本次仅将已审查实现的不可覆盖 run ID 改为 r03，科学配方不变，9 项聚焦检查 PASS。沿用此前候选唯一 P0/P1 审查及设备恢复代码的定点复审，不对未变化代码新增审查。原 L_s 6300/U_s 56700、同初值/源顺序/损失/优化器，从零单阶段 30 轮、7410 更新；每轮原子保存模型/优化器/RNG 仅作技术恢复，固定 final30。

性能主线仍是原地面完整 Tri-SIG-TIME SSL、原默认联合头、原 Main split 168000 物理样本。raw clean 145385/168000=86.538690%，每个 practical 至少 140345 条正确（距 raw clean ≤3 个百分点），并保持 clean。冻结后自动两总体全部七景预测，再独立 truth-last 评分和逐场景分析。源域好坏不取消测试，不据目标反馈改配方；默认 DURA 不变。训练效率只作成本报告。

本次仅选已授权 GPU 4/6，每卡新增一个、计入其他任务后总数不超过三个，实际余量不少于 12000 MiB；绑定准入 UUID。GPU 7 即使恢复也不纳入本版选择器。若技术故障复现，保留独占 root/逐轮快照，停止自动重试，不进行设备管理操作。

## 启动后独立读回

VERIFIED / RUNNING。已核对 controller/worker 的实际命令、CWD、CUDA UUID、GPU 计算进程和日志从零增长；真实源2条 checkpoint/输入梯度检查及 E0 严格重载 PASS。至少完成首轮247步，E01 技术恢复文件经独立 CPU 加载核实，模型/优化器及 CPU/CUDA RNG 均存在且状态有限。尚无正式性能结果，不把启动或源日志当作验收完成。

## 全量训练、预测与独立评分完成

VERIFIED / ANALYZED。新 VarF 从零单阶段 30 轮、7410 更新完成；原 Main split 两总体共 168000 物理样本的七景全部预测固定后，独立 CPU truth-last 评分。新 run 四个 GPU/CPU 阶段均正常退出，本任务 controller/worker 已退出。与冻结原 MSAT 的 84 单元复算通过：2604 评分组、1736 raw 配对、1302 额外配对，最大指标差 1.24e-14；raw/DURA 复用判决差 0；初值和源顺序一致，全部 30 份恢复文件存在且 E30 模型等于 final30。

VarF 六景达标 0/6，clean 86.585119%（保持）。完整性能目标仍未达成，保持 ACTIVE；本版不能替换默认 DURA。全部场景、分 TX/RX/day、配对修复/损伤、训练成本和环境限制见 analysis.md/results.json 及配套表图。此前失败尝试不作为性能负结果；已知 29 轮耗时和本次成功成本同时报告，不隐去故障重试成本。

分总体补充：合并 clean 多对78条，cross_rx多对79条，cross_both少对1条；不宣称所有子总体clean均保持。已知失败加本次训练累计至少45.499分钟，部分失败轮次/启动等另计。详见analysis.md和results.json的scope/cost字段。
