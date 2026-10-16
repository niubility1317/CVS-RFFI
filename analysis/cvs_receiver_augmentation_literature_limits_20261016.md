# 接收机泛化 IQ 增强：本次原始文献核查的适用边界

2026-10-16。当前 F＋跨 RX 同 TX、固定 E20 的 additive As 三组正在运行。本说明只记录后续 A_g 的证据边界，不改变该实验、晋级条件、预算或目标权限，不是新增发布门槛。

1. Ma 等的 [Towards Channel-Robust and Receiver-Independent Radio Frequency Fingerprint Identification](https://arxiv.org/abs/2512.12070) 采用谱图、对比预训练和 Siamese 分类训练，并在 LoRa 数据上评估。它支持分别考虑信道和接收机差异的研究动机；其模型、表示和训练阶段不同于当前 CVS，论文成绩不能作为本项目 A_g 或 practical 收益，也不能与现有 RDrop/SupCon 的增益相加。

2. Shen 等的 [Towards Receiver-Agnostic and Collaborative Radio Frequency Fingerprint Identification](https://pureadmin.qub.ac.uk/ws/portalfiles/portal/535502020/TMC_2023_Receiver_Agnostic_RFFI.pdf) 使用真实多接收机 LoRa 数据、对抗训练及协同推理，并另有微调。其接收机变化证据涉及真实硬件；不能由此将已接收 IQ 的数字变换描述为替换真实 RX。本项目不引入论文的附加判别器、目标微调或多接收机部署推理。

3. [Domain Generalization for Cross-Receiver Radio Frequency Fingerprint Identification](https://arxiv.org/abs/2411.03636) 的摘要以可分离条件和发射机／接收机表示分解为依据。该条件不是“任意 IQ 扰动都身份保持”的结论。本次仅核对摘要，没有把它作为某个具体增强算子的有效性证据。

4. [Towards Distance-Invariant Radio Frequency Fingerprinting via Augmented Unsupervised Learning](https://ojs.aaai.org/index.php/AAAI/article/download/37005/40967) 第 3 页列出缩放、片段重排、窗口裁剪及时间／幅度／窗口扭曲，且包含对比／原型对齐及后续无标签目标适应。以下是本项目判断：文中的传播动机不能直接证明这些变换对固定参考窗口和当前身份特征安全。片段重排会改变时序连接，窗口伸缩会改变参考定位；本项目不据此新增上述变换、重开已关闭的缩放／时移配方或使用目标数据适应。

本次核查范围有限，未取得能直接支持全通相位滤波、采样时钟扭曲等候选在当前数据上身份保持且跨 RX 有效的证据。它们尚未实现、登记或启动；物理动机不是实证结论，也不是宣称不存在相关文献。后续若提出新 A_g，仍需先有适用于本数据的独立依据，再固定一项最小源域反证；不把健康、输入有变化或论文命名作为识别收益。

当前唯一在检验的假设仍是：保留完整原始 raw/ordinary 绝对监督系数后，追加完整 practical(raw) 是否同时保护 clean、改善 urban，并超过等尺度重复 raw 监督。现有三组固定 source RX1-held/seed01/scratchE20/8 L 前向，最终源读出前不改变方案。完整联合目标、A_g 证明、B00/B10/B01/B11 交互项、跨 seed 和正式 target clean＋六 practical 仍未完成。
