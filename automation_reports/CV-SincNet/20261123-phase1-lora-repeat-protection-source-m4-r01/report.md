# LoRa 同量复数修正与重复成分保护

- run_id：`20261123-phase1-lora-repeat-protection-source-m4-r01`
- group_id：`lora-observable-satellite-recovery`；类别：`diagnostic`；阶段：`Phase1`
- 配置与矩阵：[experiment.json](experiment.json)；状态记录：[events.jsonl](events.jsonl)
- 当前登记状态：PLANNED（实际状态按events.jsonl及独立证据更新）

## 目的与对照

粗恢复和细化改动的逐频点范数匹配，检验限制复数共享成分更新能否提高双路线冻结设备识别。

两组沿用 LoRa 分块遮挡粗先验和同参细化网络，均从零训练。对当前实收六条去扫频复谱，每频点用跨 chirp 平均和偏差功率估计重复成分。若独立扰动近似成立，偏差功率除以五给出平均观测中的噪声功率近似，从共享复均值功率中扣出后可以形成共享置信度。相关干扰、发端细节跨 chirp 不同和追踪失配均会干扰这个估计。所有估计随当前包计算，训练与部署用同一公式，无模拟通道参数。

保护臂只衰减网络改动在跨 chirp 共享复方向上的投影，独有方向完整通过。标量臂对相同提议乘统一实数，使逐频点修正的跨 chirp 复数 L2 范数与保护算子一致，并保持原提议方向。该代数匹配针对同一提议，两臂训练后可以学到不同提议。两组在粗恢复和细化都使用各自约束，细化读取已约束的粗频谱。载波系数与原时标在回写时保留。

两臂各 34,693 参数、1,800 步，匹配源 U IQ、通用扰动、重建和同包一致性、弱复谱损失权重及训练批次。使用原 2,500 个 clean 源 U 的门限定义。冻结后双路线共用各组权重和阈值，同包 900 条源 V 用 Tri-set 单种子做独立真值评分。研究思路与经典阵列保护所需响应的线性约束相关。阵列方法处理空间传感器权重，本实验对单接收链的包内重复约束复数 IQ 更新，并使用同范数方向对照。 [Frost 原论文](https://doi.org/10.1109/PROC.1972.8817) · [LoRa 分块遮挡实验](../20261123-phase1-lora-spectral-block-source-m4-r01/report.md) · [LoRa 协方差 Wiener 对照](../20261123-phase1-lora-local-wiener-source-m2-r01/report.md)

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
