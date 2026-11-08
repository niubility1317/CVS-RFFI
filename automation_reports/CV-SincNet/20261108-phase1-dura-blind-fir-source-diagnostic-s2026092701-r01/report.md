# DURA 源域多抽头盲信道拟合方向诊断

- run_id：`20261108-phase1-dura-blind-fir-source-diagnostic-s2026092701-r01`
- group_id：`phase1-dura-blind-fir-source`；类别：`diagnostic`；阶段：`Phase1`
- 配置与矩阵：[experiment.json](experiment.json)；状态记录：[events.jsonl](events.jsonl)
- 当前登记状态：PLANNED（实际状态按events.jsonl及独立证据更新）

## 目的与对照

只读合法源 IQ，比较 DURA 原物理方向与 8、32、64 抽头正则拟合后的方向；源 clean 理想步长只作不可部署诊断。

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

## 固定诊断方案

固定 DURA v1.0 和 672 条合法源 pilot，对同一观测和原 DURA 波形分别拟合 8、32、64 个复数信道抽头，实／虚部分按 DURA 的共轭分量联合构造实线性设计矩阵。正则系数 0.1，信道更新范数不超过原估计信道范数的 25%；每条记录单独拟合，仅使用当前观测及当前波形。比较拟合前后的观测残差和更新物理方向；用源 clean 计算受限正向／双向理想步长时，该步长只作不可部署诊断。14 个 RX／场景组各 48 条；不读取 F、TX 标签、V、目标 IQ 或真值，不更新网络。源样本已探索，DURA 可能见过这些记录，波形误差不代表身份识别。

N607 GPU0 需至少 8192 MiB 空闲、总计算进程少于 3 且本任务同卡无其他 worker。真实 DURA 和源 IQ 的 2 条预检是同一进程首步；非法来源、非有限数、算子不匹配、残差显著上升或输出冲突时停止并保留产物。固定代码 commit 0bba752228926de9d95a1381b4b61c166ddcb201；精确命令与路径见 experiment.json。

4 项合成数学检查通过；在 N607 使用真实 DURA action 实现的合成张量，三个抽头数均通过拟合残差和算子一致性检查，最大浮点差约 8.6e-6。独立 P0/P1 审查无阻断；真实源 IQ smoke 和完整诊断尚待执行。
