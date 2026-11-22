# LoRa 原 Tri 单独监督预热与载波适配

冻结源诊断显示，历史 E140 联合模型直接拿掉附加分支后，原 Tri 分支在源 L 只判对 352/2800（12.57%），源 V 为 1318/12000（10.98%）。短程 LoRa REPL 回合前两轮没有置信度达到 0.95 的 U。需要给完整原 Tri 路线独立且充分的有标签训练，再检验半监督。

固定种子 2026092702，两个从零训练的原三视图 SIG-TIME。原 Tri 默认头与固定局部重排一致。第一行在原 LoRa IQ 上训练。第二行将观测 chirp 去载波后标准化非载波残差送入原时域、频谱与 PA 骨干，保留 raw IQ 给既有参考响应，融合仍走原头。这项确定性输入适配不增加可训练分支。每行前 70 轮仅用 L_s 的原配对监督目标与合法同包 practical 的 As 分支，联合 28 轮才加载盲 U。每轮固定 88 步，两行各 8,624 步。联合期保留 U=3、.95 当前学生硬伪标签、六次 U 前向、同伪类跨接收机共享样式计划、联合特征对比、原分类头外推与分组 L 锚梯度保护。记录每轮 L、固定轮次源 V 七景、切换前源性能、实际 U 接受率和独立离线伪标签正确率。

两行在原 15 台未见 RX、同 30,000 包、clean 和六种 practical 上各固定 105 份盲预测。由独立评分比较每个方法各自的 clean 差距，同时报告绝对 clean 和每景准确率，与同种子普通 PL 和候选集合 Tri 对照。此前 E140 残余分支实验保留完整固定设计。

使用 `python -m experiments.pamf_lora_main.full_adaptive_publish --warmup-submit ... --native-warm`，Git commit 定义代码。普通 N607 账户，GPU0–7 容量调度。输出 `CV-SincNet/runs/20261122-phase1-lora-native-tri-warm-m2-r01`，CWD `CV-SincNet/releases/20261122-phase1-lora-native-tri-warm-m2-r01-release`。实际源数据两层和六前向检查通过即继续。出现协议、梯度、预算或盲矩阵技术错误才停止。冻结预测后自动独立评分。

[原 Tri 分支直接续训预登记](../20261122-phase1-lora-native-tri-repl-m2-r02/report.md)
