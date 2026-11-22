# 保留 LoRa 输入适配的原 Tri REPL

同合同 E140 纯监督起点的 LoRa 对齐分支保留，原三视图 SIG-TIME 骨干、固定序列重排和已有两个联合读出均保持。拆出原骨干的对照源 L/V 明显欠拟合且 U 零接受，本次比较完整模型上的原 REPL 配方。

固定种子 2026092702，两条 28 轮、2,464 联合更新，U 64 包、阈值 .95、权重 3。两条都使用当前学生 clean 伪标签、原监督 ordinary/raw KL 和 As 双 CE、六次 U 学生前向、同伪类跨 RX 共享 style plan、联合对比与 L 锚分组梯度保护。特征外推同时应用到真实原头和原 LoRa 对齐分支各自的现有 160 维输入，然后按原方式相加，不额外引入分类头或特征前向。源实践 U 行仅对同物理包、相干度达到 .40 的 ordinary 半槽使用 practical，另一行使用加噪 clean ordinary。两条 U 的 As 都仍使用 raw clean，硬伪标签都从 clean 生成。实际 U 接受、样式配对与 practical 曝光记录，以冻结后的独立 U 评分给出正确率。

正式对照同种子纯监督、普通 PL、原适配 Tri 和候选集合 Tri，15 个未见 RX、同样的 30,000 包、clean 与全部六种 practical。每条候选完整 105 份盲预测，共 14 个单元，再独立评分。以绝对 clean、各 practical 和相对自身 clean 的差距一起判断，避免靠降低 clean 缩小差距。

使用 `python -m experiments.pamf_lora_main.full_adaptive_publish --warmup-submit ... --native-adapt`。N607 普通账户、GPU0–7，依据容量选择。Git commit 固定实现，运行根 `CV-SincNet/runs/20261122-phase1-lora-tri-repl-compatible-m2-r01`，CWD `CV-SincNet/releases/20261122-phase1-lora-tri-repl-compatible-m2-r01-release`。用对应实际完整 E140 和原始源 IQ 验证三视图梯度、保护后的优化步骤与六视图，训练冻结后自动盲预测和独立评分。合同、非有限输出、训练步数或盲矩阵异常为技术停止条件。
