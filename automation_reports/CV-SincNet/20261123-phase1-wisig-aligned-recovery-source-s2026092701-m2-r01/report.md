# WiSig 接收端时标对齐与联合细化源域消融

- run_id：`20261123-phase1-wisig-aligned-recovery-source-s2026092701-m2-r01`
- group_id：`wisig-observable-satellite-recovery`；类别：`diagnostic`；阶段：`Phase1`
- 配置与矩阵：[experiment.json](experiment.json)；状态记录：[events.jsonl](events.jsonl)
- 当前登记状态：PLANNED（实际状态按events.jsonl及独立证据更新）

## 目的与对照

两行各从零训练一次，合法源 U、预算、弱复谱损失及通用扰动全部匹配。唯一差别是其中一行在训练和推理中用当前包的短长前导过渡估计并对齐。固定两路线的恢复权重、门限及推理规则。

两行从同一源 U 合法波形出发，使用同一宽范围通用扰动和公开短训练周期定义的位移，允许在正负 40 个采样点内变化。匹配弱差异对共用复数扰动和位移，另一份同包观测独立。实际合成位移仅用于生成输入。对齐组训练和推理都运行 [短长前导定时估计代码](../20261123-phase1-wisig-timing-engineering-source-s2026092701-r01/report.md)，共用该轮 U 域固定门限与协议后缀填充。两行共享相同架构、参数量、训练随机流和 1,200 步优化。对照组增加了同样的通用时移，学习时保持观测坐标。各自的粗恢复、时域细化及置信度观测约束在同一个损失中联合训练。

每组冻结后输出粗、联合、源校准门三项。对齐组固定门在时标更正通过或周期残差超过 U 域门限时启用恢复，其余输入直接通过。两路都用同一模型和门限。评估复用当前 F50 与同面板原始直通和 R69 决策。短长前导工程对照的同包结果可用于核查联合学习相对单独时标校正的变化。受保护 clean、逐接收机方向一致和两路线的 R69 保持是本轮源选择依据，未晋级的全部结果保留。

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
