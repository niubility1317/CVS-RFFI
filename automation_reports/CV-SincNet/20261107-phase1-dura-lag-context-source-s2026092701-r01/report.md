# DURA 长距离相关恢复 GPU 训练

- run_id：`20261107-phase1-dura-lag-context-source-s2026092701-r01`
- group_id：`phase1-dura-lag-context`；类别：`cvs`；阶段：`Phase1-restoration`
- 配置与矩阵：[experiment.json](experiment.json)；状态记录：[events.jsonl](events.jsonl)
- 当前登记状态：RUNNING；N607 GPU 0 源训练运行中。

## 目的与对照

冻结 DURA，使用非循环 20/40 延迟的相位对齐差异作为源训练修正网络输入。

## 数据、seed与模型来源

实际数据契约、权限例外、完整seed角色、checkpoint来源和选择规则见experiment.json。
逐行配置通过config_ref/resolved_config_ref定位；待补项必须在对应生命周期补齐。

## 执行与存储

命令、环境、CWD、commit、launch owner、输出和日志路径见experiment.json。
实际PID/GPU、读取时间、remote readback、失败或替代关系在此追加，并用record命令记录证据指针。

## 结果与覆盖

预登记：固定源域 pilot 显示时延 20、40 的复相关较高，以此提出非循环双向时延上下文；分支使用同一记录的 received、冻结 DURA 输出、物理残差和可观测方差，按 DURA 输出估计对齐相位。无效边界补零，不把源 clean 最优步长用于部署。候选从零初始化、输出初始等于 DURA，新增 33408 参数；使用 32768 个合法源 ID × 三视图、5 轮、AdamW 0.0003、batch 32，固定最终轮；不读 TX 标签、F、V 或目标训练。源数据、缓存和 DURA 来源见 experiment.json。

GPU 0 预检空闲 18501 MiB，已有两个其他计算进程；队列须继续核查容量，保留其他 owner。训练结束后自动进行 Tri-SIG-TIME SSL Main split 固定默认头的七景预测，每景 168000 条，全部预测固定后独立 truth-last；与固定 DURA 逐记录配对，并记录 TX/RX/day。预测或评分不得反馈到训练和选模。训练或预测发生系统性错误、非有限值、协议错误或输出冲突时立即停止本 run 并保留证据；不因模型分数选择性停止。

11 项聚焦检查、真实源 IQ 与 DURA checkpoint 的 6 步 GPU 检查通过；一次独立 P0/P1 审查无阻断问题。N607 真实 checkpoint 与源 IQ 的 64 ID × 三视图 smoke 已通过，记录 TX/F/V/目标读取均为 0 且 DURA 冻结；controller PID 494858、trainer PID 495033 已核实。源相关和波形误差不是身份识别结果，尚无正式性能结论。

## 交接

记录已完成、当前run/commit、证据路径、阻塞与下一步；恢复先查原run，不重复启动。
