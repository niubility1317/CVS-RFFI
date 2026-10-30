# DURA 从零单阶段 40 轮收敛对照

- run_id：`20261030-phase1-DURA-direct-noVar-E40-s2026092701-r01`
- group_id：`phase1-dura-single-stage-convergence`；类别：`cvs`；阶段：`Phase1`
- 配置与矩阵：[experiment.json](experiment.json)；状态记录：[events.jsonl](events.jsonl)
- 当前登记状态：RUNNING（实际状态按events.jsonl及独立证据更新）

## 目的与对照

完整源20轮曲线仍下降；固定noVar损失，单阶段从零40轮/9880更新，无祖先/F训练；自动源与目标全七景，披露不同训练预算。

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


## 单阶段预算与比较边界

本次固定从零 40 轮、9880 更新，全部模块同时训练，不加载 E20 父权重，也不加入扰动或 F 指导。已完整检查 noVar 的 20 条源训练日志：最后 5 轮平均 IQNMSE 为 0.02556429，前 5 轮为 0.02889302，下降约 11.52%；这支持检验更充分收敛，不能保证目标迁移收益。分段汇总见 source_convergence.json。

原 20 份源模拟缓存循环两遍，每轮重新打乱；累计 L252000/U2268000 曝光，来源和标签权限保持。最终固定 E40 后直接完成同一 F 的源 1800×7、目标 168000×7 和独立 truth-last，不选中间轮次。E40、原 E20 与默认 DURA 的训练预算不同，分别报告实际完整成本，不按轮数声称等质量加速。


## 已核实启动

VERIFIED / RUNNING。执行 commit `d0ac66573e7e5f775da76de01a2e4ac0a18ce1bd`，唯一独立 P0/P1 审查 PASS，14 项聚焦新旧测试 PASS，N607 preflight、单归档两端 SHA 和远端 compile 通过。控制器 PID295188，GPU7 worker PID296374；独立核实实际进程 CWD、完整命令、GPU7绑定和训练日志增长。真实 source2Ls 随机图完整梯度、checkpoint重载、同一F默认头无query smoke PASS，随即开始训练。首轮 E1/247 完成，用时36.317秒；尚无本版正式分数。

启动时只新增本任务一个进程，未停止或修改其他 owner。原 E20 与 F 指导候选产物保留，DURA 默认不变。继续固定40/9880完整预算，随后自动全七景、独立 truth-last 和逐场景比较。
