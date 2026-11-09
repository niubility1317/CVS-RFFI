# 源模拟信道延迟成分分解诊断

- run_id：`20261109-phase1-source-delay-components-s2026092701-r01`
- group_id：`phase1-source-delay-components`；类别：`diagnostic`；阶段：`Phase1`
- 配置与矩阵：[experiment.json](experiment.json)；状态记录：[events.jsonl](events.jsonl)
- 当前登记状态：PLANNED（实际状态按events.jsonl及独立证据更新）

## 目的与对照

将源模拟 FIR 64 抽头固定分为四个连续窗口，检查其波形分量能量、相干性与重建和；不读取身份标签、F 或目标。

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

## 固定分析与边界

对 672 条已探索的 Phase1 源 L_s/U_s 观测，按 RX4/6、RX8 与七视图组成 14 个每组 48 条的单元；按同一物理 ID 读取 epoch19 的 clean、received 与匹配的模拟 kernel、复数作用系数。固定将 64 个 FIR 抽头切分为 0–15、16–31、32–47、48–63 四段，对源 clean 分别施加，逐批验证四段之和等于未分割作用。记录各段相对完整输出能量、段间复数相关绝对值、每条主导段和观测残差；残差包含加入噪声和模型差异，不直接称为噪声或 SNR。

四个区间是 FIR 抽头分组，并不等同真实可分辨传播路径。模拟器与源 clean 仅用于源端诊断，目标部署不可读取；不读取 TX 标签、F 权重／输出、V 或目标，不更新模型。首批两条使用真实冻结 DURA 做源无查询 smoke；CPU 两线程后台运行，不占用 GPU。源码、命令、输出、停止规则见 experiment.json。

本诊断受 [Fadul 等多解码头恢复论文](https://arxiv.org/abs/2304.00648)启发，但该论文联合训练使用真实类别标签和分类器，当前恢复器约束不允许直接照搬；亦不能继承其识别效果。代码 commit 发布前回填。

代码 commit：`59036abf3cb85f56b35fbae482e3ea82e0a85c5b`。两项针对性检查已通过，包括真实复数作用的分段可加性、零分量、非法输入；独立 P0/P1 静态审查无阻断。首批实际源记录与冻结 DURA 的 CPU smoke 尚由正式执行首步完成。

固定发布归档本地和远端 SHA-256 均为 `471961e7ae93cdfc4dc2f2902c595c5e65e0338ae369f729abe25ce82596e8ae`；N607 CPU 环境远端编译和 2 项针对性检查通过。真实源 checkpoint smoke 作为执行首步。
