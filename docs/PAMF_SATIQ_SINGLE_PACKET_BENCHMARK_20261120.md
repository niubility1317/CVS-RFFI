# SatIQ 单包波束识别与公开结果的评估口径

## 本项目的主要指标

主 CE 的目标标签是每条包的 `(ra_sat, ra_cell)`。它最小化真实波束概率的负对数。单包全类 Top-1 直接统计 3,103 类分类头的首选波束是否正确。融合 256 维第 10 轮源 V 的主 CE 为 7.0043，Top-1 为 0.9315%（464 / 49,814）。同轮训练 L 为 1.1328%，说明单包难点在训练集上也存在。[全类诊断](../automation_reports/CV-SincNet/20261120-diagnostic-satiq-fused256-head-e5-e10-s2026111907-r01/report.md)

K 指每个候选波束的历史注册包数。当前实现先固定 K16 面板，要求每类至少 16 条注册包和 1 条独立查询。跨日面板的 16 条注册包还要处于时间界线之前。因此跨日面板有 309 类、4,865 条查询。同一面板的 K1 和 K16 使用同样的候选和查询，当时两种读出均每次处理一条查询。融合 256 部分白化后的跨日 K1 原型 Top-1 为 1.2744%（62 / 4,865），同星验证 AUC 0.5767。K16 为 2.5077%（122 / 4,865），AUC 0.7093。模型训练使用原论文式消息划分，跨日修饰的是注册和查询的时间隔离。[面板实现](../experiments/pamf_satiq66/paper_eval.py) · [配对冻结评估](../automation_reports/CV-SincNet/20261120-diagnostic-satiq-e10-center-whiten-s2026111907-r01/report.md)

## 检索到的主要公开结果

截至 2026-11-20，本轮检索未找到直接报告 Oxford Zenodo 8220494、冻结相同划分、同样 3,103 个波束、单个公共同步字段查询的全类 Top-1。公开结果按问题与数据来源解读。

| 论文 | 标签、输入与协议 | 可核实的指标 |
|---|---|---|
| SatIQ，ACM CCS 2023 | Oxford Zenodo 8220494，发射端以卫星与波束为身份。25 MS/s 报头，重构和三元组联合训练。验证已声明的发射身份，取单条查询与参考包的平均距离。 | 真实发射端 K1 AUC 0.627、EER 40.5%，K16 AUC 0.698、EER 35.0%。0.946 AUC、12.0% EER 属于 K32 重放检测。 [原文 §§4.3–4.4、5.1](https://www.cs.ox.ac.uk/files/14284/2305.06947.pdf) |
| Att-Sat，IEEE Access 2025 | 同一公开来源取 500,000 条，80/10/10 划分，注意力与度量学习。 | K1 AUC 0.600、EER 43.5%，本论文同划分 SatIQ 对照 0.580、44.6%。 [论文全文](https://www.researchgate.net/publication/392349504_Secure_Satellite_Downlink_Communication_with_Attention-enhanced_Transmitter_Fingerprinting) · [DOI](https://doi.org/10.1109/ACCESS.2025.3575409) |
| SatIQ 扩展，ACM TOPS 2026 | 扩展后的 Oxford 等接收地点数据，1,029 万条总采集量。同源波束验证表现优于跨站点。 | 全英国训练与英国测试基础 AUC 0.703、EER 35.1%，K32 AUC 0.795、EER 27.7%。0.960 AUC 来自重放检测。 [原文 §4.5、表 4、§6](https://www.cs.ox.ac.uk/files/14805/main.pdf) |
| OrbID，SpaceSec 2025 | 在原 SatIQ Iridium 数据验证替代模型。另有独立 Orbcomm 任务。 | Iridium 验证 AUC 0.576、EER 44.8%。 [原文表 I 与 §V-D](https://www.cs.ox.ac.uk/files/15143/orbid.pdf) |
| IriSig-Spoof / MACNN，2026 预印本 | 自采 66 颗卫星，2,000 个 IQ 点为一个查询，前 7 天训练和验证，后 25 天做固定模型测试。 | 平均跨日卫星分类 **97.75%**，仅标准预处理，无额外增强。原始 MACNN 为 90.81%。 [原文 §§III-B、IV、V-A](https://arxiv.org/html/2608.18642v1) |
| Zhu 等 3D-CNN，IEEE GLOBECOM 2023 | 自采 66 颗卫星，只选论文所述 beam 0。每个输入由单条 IRA 中 4,200 个 IQ 点按时间分组。70/30 划分。 | 卫星分类 **92.3824%**，F1 92.3475%。 [原文 §§III–V、表 I](https://web.xidian.edu.cn/ylshen/files/6797c14dd7fc3.pdf) |
| IridiumRF-MR，Sensors 2026 | 自采 66 颗卫星，同一副天线分给六台 SDR。单个预处理后 3,000 点 QPSK 段。 | Mamba 六接收机内结果平均准确率 **85.67%**，ResNet34 跨接收机 **81.93% 至 84.06%**。 [原文表 5、表 8](https://www.mdpi.com/1424-8220/26/20/6418) |
| PAST-AI，IEEE TIFS 2023 | 自采 66 颗卫星，每张输入图聚合 10,000 个 IQ 点。 | 66 类约 **82%**。论文指出，仅用单包样本构成图像的验证率明显较低。 [原文 §V-B–V-C](https://pure.tue.nl/ws/portalfiles/portal/261348390/PAST_AI_Physical_Layer_Authentication_of_Satellite_Transmitters_via_Deep_Learning.pdf) |

## 下一步优化次序

1. 以固定 3,103 类、单个查询的源 V Top-1 和主 CE 作为模型优化的主判据。K1 和 K16 用于解释注册验证性能。SatIQ 的闭集诊断可用源 L 注册原型，保留 3,103 个实际训练类，并让同一 V 查询与全部候选比较。若需要严格跨日训练泛化，对训练与验证日单独划分，报告类别和样本覆盖。
2. 保留 PAMF 五分支、融合 256、输入与波束标签，先做一次仅改变训练损失的全源配对诊断，比较现有完整损失和只用主 CE。在相同 1,024 包、1,600 步原生 160 诊断中，只用主 CE 的训练 Top-1 为 12.40%，完整损失为 7.03%。缩小原型系数的全源首轮对照反而变差，因此先确认整套辅助监督是否影响真实验证，再决定主 CE 预热后何时接回原型损失。[小样本拟合](../automation_reports/CV-SincNet/20261120-diagnostic-satiq-native-fit-subset-s2026092701-r01/report.md) · [全源原型减权对照](../automation_reports/CV-SincNet/20261120-diagnostic-satiq-native160-proto25-source-e1-s2026092701-r01/report.md)
3. 如果两组都低，再做一次只改变前端时序分辨率的配对测试。现有频带功率诊断没有显示加回 250 kHz 以上成分的收益，结论仅覆盖频谱功率形状。指纹信息是否能从更细的时序中恢复，需要继续用相同波束和同一批包验证。[截频对照](../automation_reports/CV-SincNet/20261120-diagnostic-satiq-native-stopband-source-s2026111907-r01/report.md)

8,000 点模型的完整首轮复评已完成。完整损失的 L/V 单包 Top-1 为 0.2322% / 0.2509%，仅主 CE 为 0.1124% / 0.0964%。两组的源训练与验证指标都支持在当前正式配置中保留完整损失。现在对齐 11,000 点实验的原生同步模板时间轴，验证同一输入上的单包 Top-1。[仅主 CE 对照](../automation_reports/CV-SincNet/20261120-diagnostic-satiq-fused256-main-ce-source-e1-s2026092701-r01/report.md) · [模板时间对齐](../automation_reports/CV-SincNet/20261120-diagnostic-satiq-full-header-uw-time-source-e1-s2026092701-r01/report.md)
