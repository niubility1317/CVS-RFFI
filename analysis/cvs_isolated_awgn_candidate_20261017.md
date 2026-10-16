# 下一单一候选：raw 上孤立的轻度白噪声观察

2026-10-17。源检验已完整执行并独立复算，PASS_SOURCE_IDENTITY_ONLY；三匹配 E20 已完成并独立复算，FAIL_FIXED_NATIVE_AWGN_20_36_FULL_COVERAGE_TRAINING_CLOSED；不改变任何健康或完成任务。基线继续 F＋跨 RX 同 TX／原默认头联合训练／E20。

前一普通信道源审计恢复了部分错误，但联合移除仍比 raw 低约6个百分点，已按固定条件关闭，单项移除不会晋级。下一问题不再修改完整 ordinary 以挽救该配方，而是在保留公共原 ordinary 的前提下，独立检验额外 raw＋noise 观察。没有 multipath、PA、量化、剪裁、时移、相位／CFO 或形状混合；不修改 raw、As、U 或测试 IQ。

固定待测算子为原生 proper complex AWGN，仅用原 raw 包功率决定方差，SNR 沿用仓库既有 20 至36 dB 范围，不从目标或本轮分数选择新范围。先生成一次噪声 IQ，再供同 IQ RDrop 两次前向。名义 SNR 有界，高斯样本幅度不具严格有限界；需要报告实际噪声能量、IQ 变化和尾部，不能靠删包／重抽／clip 满足轻度条件。新增噪声 SNR 不等于原录音总 SNR，也不能称作真实 RX 替换。

文献仅作为候选依据：[OpenRFI（AAAI 2025）](https://ojs.aaai.org/index.php/AAAI/article/download/32003/34158) 在预训练中使用轻度 AWGN，并与片段重排和其他训练机制组合；其消融重点为损失和阈值，不能据其总收益证明本协议下 AWGN 的独立跨 RX 效果。本次不采用重排、额外原型／分类头或其损失。

最小真实源检验：原两个合法 E20 默认头／fitL5040，raw与四个固定命名噪声观察；仅有标签 L，不读 U/V/heldRX1/target。先检验身份保持，不要求噪声能修复近乎满分的 raw；两模型均须 accuracy／macro-F1 相对raw≥−0.5pp、raw正确保留≥99%、每RX/TX/day accuracy≥−1pp，逐draw同样保护，全部物理样本和观察保留。实际相对IQ RMS变化中位数≥0.01、最大≤0.15，超过则关闭此固定20至36 dB/全覆盖方案，不抬SNR、clip、删尾或换seed救跑。精确种子、NN和技术停止规则必须在新run登记中固定，当前未作源数据访问。

源条件通过后，才实施三组 scratch E20 匹配训练：公共 raw.75＋原 ordinary.25／原 SupCon、U和默认头流程保持；extra.25分别raw、同份原ordinary、孤立noise(raw)，均8L前向、总权重1.25、同IQ RDrop。noise须同时超过两个控制的 source-heldRX clean≥+1pp、macro-F1不降、每practical／seenV≥−0.5pp，再独立seed确认。仅通过源身份条件不能称A_g有效；E20不能同时优于两个匹配控制就关闭，不按目标分数调整噪声范围／loss／seed重跑。

此路线与另一对话的 practical 内部正负噪声采样不同：本候选只研究额外 raw 观测的轻度噪声，不重做完整星地传播，不修改As的接收状态／AGC／补偿依赖或测试信道。最终仍需完整契约 A_g／A_s 四格及交互项、跨seed与正式clean＋六practical，不能把各实验增益相加。
