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

## 同数据训练与 PAMF 优化

当前正式训练使用同一份 910,464 条训练消息、50,000 条验证和 50,000 条独立测试的重建清单。PAMF 和参考 SatIQ 读取同一个 25 MS/s 的 11,000 点报头，共用 3,103 个合格波束及逐轮批次，分别按各自训练目标运行 200 轮。比较两种表示时，使用相同的源训练注册包及全类余弦原型。PAMF 另给出直接分类头，独立测试在两组固定末轮预测闭合后接真值。两组首轮同口径源 V 的原型 Top-1 为 SatIQ 0.0562%、PAMF 0.2088%，PAMF 直接分类头为 0.2007%。[同数据训练与首轮结果](../automation_reports/CV-SincNet/20261120-phase1-satiq-ccs2023-header-matched-sup200-s2026092701-r01/report.md)

8,000 点的完整首轮复评支持保留 PAMF 的完整损失。完整损失的 L/V 单包 Top-1 为 0.2322% / 0.2509%，只用主 CE 为 0.1124% / 0.0964%。后续前端实验使用已对齐原生同步模板的 11,000 点配置，保留五分支和 256 维身份表示。[损失对照](../automation_reports/CV-SincNet/20261120-diagnostic-satiq-fused256-main-ce-source-e1-s2026092701-r01/report.md) · [模板时间对齐](../automation_reports/CV-SincNet/20261120-diagnostic-satiq-full-header-uw-time-source-e1-s2026092701-r01/report.md)

相同的 49,814 个源 V 包上，512 点内部网格在完整首轮命中 131 包，256 点对照命中 109 包。两组完成了同预算第 5 轮，512 点命中 339 条，256 点命中 363 条，首轮差异已反转。镜像边界在完整首轮命中 122 条，对照命中 109 条；第 20,000 步镜像组曾落后。暂用 256 点原边界与已对齐模板作为新配对基准。[网格续训](../automation_reports/CV-SincNet/20261120-diagnostic-satiq-full-header-grid256-grid512-source-e5-s2026092701-r01/report.md) · [边界处理](../automation_reports/CV-SincNet/20261120-diagnostic-satiq-full-header-reflect-source-e1-s2026092701-r01/report.md)

首轮逐包误差复核确认两组共同命中 65 条，512 点独有 66 条，256 点独有 44 条。按真实波束抽样的 95% 净差区间跨过 0。 [首轮错误配对](../automation_reports/CV-SincNet/20261120-diagnostic-satiq-grid-paired-errors-e1-s2026111907-r01/report.md)

同数据正式第 5 轮，SatIQ 和 PAMF 用共用源注册原型命中 66 条与 198 条，PAMF 自身分类头命中 357 条。PAMF 第 20 轮单包 Top-1 为 1.3550%（675 条）。第 5 轮 PAMF 首选波束所属卫星正确率为 2.99%，真卫星内的离线诊断 Top-1 为 18.44%。已启动的卫星概率监督对照重新分配现有 0.20 辅助权重，推理仍每包面对 3,103 类。[同数据第 5 轮](../automation_reports/CV-SincNet/20261120-diagnostic-satiq-matched-source-e5-s2026111907-r01/report.md) · [卫星概率监督](../automation_reports/CV-SincNet/20261120-diagnostic-satiq-full-header-satellite-mass-source-e5-s2026092701-r01/report.md)
