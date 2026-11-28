# LoRa R17 与实增益主干源域联合微调

- run_id：`20261128-phase1-lora-joint-r17-realgain-s2026112873-r01`
- group_id：`phase1-lora-joint-r17-real-gain`；类别：`comparison`；阶段：`Phase1`
- 配置与矩阵：[experiment.json](experiment.json)；状态记录：[events.jsonl](events.jsonl)
- 当前状态：SOURCE_SCREEN_SCORED；未达到预登记晋级条件。

## 目的与对照

冻结 R02/R07/R17 和 E104，联合微调实增益主干及 R17 物理残差双方向头，源域无泄漏筛查。

## 数据、seed与模型来源

实际数据契约、权限例外、完整seed角色、checkpoint来源和选择规则见experiment.json。
逐行配置通过config_ref/resolved_config_ref定位；待补项必须在对应生命周期补齐。

## 执行与存储

命令、环境、CWD、commit、launch owner、输出和日志路径见experiment.json。
归档传输校验和远端编译通过。控制器 PID 3586351，训练进程 PID 3586422；已独立核对 CWD、命令、GPU 与日志增长。真实 checkpoint 首步检查了零增益恒等、双方向增益与实增益 core 梯度及 tracker 冻结。代码提交为 `6ffe09217d1021ac9d3d7129cbc97008804ef8c0`。

## 结果与覆盖

训练完成 2000 步，源 L／U 各曝光 8000 次，接受 7379 个 U 伪标签，耗时 990.95 秒，峰值 CUDA 分配 1189277696 字节；可训练实增益 core 有 325029 参数，新增增益头 11394 参数。训练未读取 U 真值、V 或目标数据。

源 RX b200_1 的 300 个此前未筛查物理包的 clean 与 low urban 预测固定后，由 CPU 独立连接真值。low urban 中直通、LoRa R17、实增益、联合融合正确数分别为 131、74、185、175；clean 分别为 299、299、299、296。相对实增益，融合改对 8 包、改错 18 包，净减 10 包，两个半区分别净减 2、8 包，配对 bootstrap 95% 区间 [−20, 0] 包；clean 净减 3 包，也超过预登记允许损失 2 包的门槛，故不晋级。

历史源 V 已参与 E104 选模，历史目标基准也已曝光；同契约 DURA v1 LoRa 对照及完整目标七景待完成。

## 交接

保留权重及负结果；后续候选另行预登记并排除已筛查物理包。
