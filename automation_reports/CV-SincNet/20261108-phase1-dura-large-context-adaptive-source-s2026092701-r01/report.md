# DURA 可观测自适应增益训练

- run_id：`20261108-phase1-dura-large-context-adaptive-source-s2026092701-r01`
- group_id：`phase1-dura-large-context-adaptive`；类别：`cvs`；阶段：`Phase1-restoration`
- 配置与矩阵：[experiment.json](experiment.json)；状态记录：[events.jsonl](events.jsonl)
- 当前登记状态：PLANNED（实际状态按events.jsonl及独立证据更新）

## 目的与对照

归一化大容量 DURA 修正联合学习每条信号的可观测增益，范围 0 至 2；训练不读取目标或 TX，推理不读取 RX。

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

## 预登记补充

源 IQ 诊断提示现有修正方向的合适幅度因记录而异，单个可观测量相关性较弱。新候选从零联合训练大容量时序修正和多输入非线性增益；增益只读取单记录噪声、物理伴随残差、候选修正、received 与 DURA 的相对能量，输出范围 0 至 2、初始为 1。训练不使用理想 alpha、源 pilot、F、TX 标签、V 或目标；推理不读 clean 或 RX。

固定 DURA 266951 参数，新网络 14794243 参数。32768 合法源 ID × 3 视图、batch 32、20 轮，最终轮冻结；与稳定基线曝光和更新次数匹配。8 项针对性测试、真实原 DURA 及源 IQ 的 6 步 GPU 检查、一次独立 P0/P1 审查通过。N607 GPU7 的后台队列在总计算进程少于 3 且空闲显存至少 4096 MiB 时启动，不改变其他进程。

## 后台队列核实（2026-11-08）

代码已推送并核对远端版本；发布包比对和远端编译通过。N607 队列 PID 747670，已核对进程、工作目录和完整命令。当前 `GPU_QUEUE_WAITING 7`：GPU7 有 3 个计算进程，新方案尚未开始训练；达到容量条件后自动运行启动 smoke 和完整训练。

字段说明：checkpoint 中 `target_reads=0`、`gain_target_reads=0` 记录训练期未读取目标数据；正式推理会合法读取每条待恢复 received IQ，并由此计算该条的增益。推理不读取目标 clean、TX 真值、RX 标识或跨记录统计。

## 后台训练启动（2026-11-08）

GPU7 容量条件满足后，控制器 PID 775125 和训练进程 PID 775887 已启动；64 条 × 三视图的真实源训练 smoke 通过，正式源训练进行中，尚无完整成绩。

正式源训练第 1/20 轮完成，训练目标 0.022542、训练阶段累计 229.4 秒；尚无识别准确率。

## 训练与七景完成

固定 20 轮源训练已完成；摘要记录 32768 条源 ID，TX、F、V 和目标读取均为 0，DURA 冻结。冻结后的七景预测与独立评分已完成：六个受损场景均低于 DURA，详情见[七景分析](../20261108-phase1-dura-large-context-adaptive-ground-test-s2026092701-r01/analysis.md)。源 pilot 增益饱和只能说明所测源样本上的行为，不能单独证明目标结果成因。
