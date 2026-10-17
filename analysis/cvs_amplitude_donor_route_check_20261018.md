# 256 点 IQ：同 TX 跨 RX 幅谱借用的路线核查

状态：研究结论已保存；未注册或启动新训练、源预测或真实 IQ 批处理。联合目标仍 ACTIVE。本轮新增 IQ 读取、checkpoint 加载、神经网络前向、教师前向、更新和 N607 作业均为 0。

最新用户约束继续生效：F＋跨 RX 同 TX 对比学习、累计 E30、原默认 cosine 头正常联合训练、全部源 RX1/3/4/6/8、L6300/U56700（TX 隐藏）/V27000。用户明确没有原始录制或长 IQ 的可靠映射，继续使用现有 256 点 IQ；不新增训练 RX 留出。

## 本轮唯一核查的问题与结论

假设草案是：保持 anchor 的 Fourier 相位，只借用同可见 TX、不同 RX 的 donor 幅谱，能否构成有独立依据的身份保持观测变化。没有固定幅度、核、donor 规则或训练矩阵，也没有生成这种 IQ。

结论：本轮不将它作为新候选。它仍属于同身份双观测插值，不能仅凭相位保留宣称与已关闭的同 TX IQ 插值构成独立方向；真实 RF 的幅度、相位都可能携带身份及接收条件。原训练报告已明确后续避开同身份插值类。本次没有修改旧停止条件或据目标结果重选幅度、seed、RX、父权重、预算。

## 文献实际支持的范围

[FACT 原论文，CVPR2021](https://openaccess.thecvf.com/content/CVPR2021/papers/Xu_A_Fourier-Based_Framework_for_Domain_Generalization_CVPR_2021_paper.pdf)对图像作幅谱插值并保留相位，研究图像域泛化。该图像语义假设不能直接证明 256 点 RF 的 TX 身份保持，本项目不据此启动 RF 训练。

[Receiver-Agnostic RFFI 作者公开仓库](https://github.com/Edith-xx/Receiver-agnostic-RFFI-CL-)及其 [README 原文](https://raw.githubusercontent.com/Edith-xx/Receiver-agnostic-RFFI-CL-/main/README.md)介绍 Fourier 相位教师蒸馏和跨 RX 特征均值/协方差对齐，使用 equalized 256 点 WiSig。已读作者方法说明，未取得该论文完整出版正文；不把 README 说成已读全文。该方法没有给出本草案的 donor 幅谱 IQ 身份保护证据，也不作为新增教师、对齐损失或部署分支的依据。

FreqMix（DOI 10.1007/978-981-92-3438-7_50）只找到题录，出版页和 PDF 未能读取；没有获得可核实的操作公式、标签处理、数据协议或对照。题名和二手检索摘要不作为方法依据，不引用其效果支持本草案。

## 与项目既有负结果的关系

| 已有证据 | 本轮判断 |
|---|---|
| [同 TX IQ 插值完整源训练报告](../automation_reports/CV-SincNet/20261016-phase1-F-crossRX-sameTX-IQmix-sourceRX1-e20-s2026092701-m3-r01/report.md) | 固定配方失败并关闭；当时为旧四 RX/留出 RX1 合约，不能当作当前五 RX 正式结果，也不能以改变混合域为由自动救跑。 |
| [PSD 正仿射源模型](../automation_reports/CV-SincNet/20261017-diagnostic-F-crossRX-PSDaffine-fitL-aggregate-m1-r01/report.md) | 六个源 RX 对均失败，没有可借用的已验证 RX 幅谱映射。它否定固定统计模型，不证明所有频域变化均无效。 |
| [轻度 FIR 首次训练](../automation_reports/CV-SincNet/20261017-phase1-F-crossRX-spectralIQ-firsttrain-e20plus10-s2026092701-m2-r01/report.md) | 当前五 RX 合约下 B10 clean 及 B11 相对 B01 的 clean/mid/low 增量为负，七视图交互项均为负。不能以保能量或核较轻代替学习收益。 |
| [有限窗酉相位源检查](../automation_reports/CV-SincNet/20261018-diagnostic-F-unitaryDispersion-sourceV-s2026092701-m1-r01/report.md) | 数学保护通过而固定默认头身份条件失败；不缩幅或换相位基救跑。 |

这些证据不足以支持本草案，尤其不能把数字处理已接收 IQ 称为替换真实接收机。same-TX donor 并不隔离传播、噪声、采集状态或非线性 TX/RX 混合；有限 256 点 FFT 还需面对窗口边界、弱谱 bin 和归一化问题。此处是可识别性及操作解释的限制，不是新增审批或固定审计层。

## 最近完成的真实实验与剩余目标

[批次组成 E30 三行报告](../automation_reports/CV-SincNet/20261018-phase1-F-crossRX-batchStratify-e20plus10-s2026092701-m3-r01/report.md)为最近完成的训练及七视图独立 truth-last 评分。固定重组行 clean82.954167%、mid urban64.629762%、low urban53.631548%；相对原顺序为＋.201786/＋.247024/＋.190476 pp，相对匹配随机重组为＋.176190/＋.157738/＋.188690 pp。七视图 accuracy/F1 均有单种子小增量，但 low 未达预定＋.5 pp，固定方案关闭。没有新 IQ A_g。

三个 suburban 距自身 clean 分别为1.013095/3.330357/5.754762 pp，low urban 仍低于60%。同一隔离 loader 流下缺严格 no-As B00，故不推断 clean 净收益或 A_g×A_s 交互。不能将小增量、源健康或数学性质写成联合成功。

关联对话“开发 CVS Phase1 星地增强训练”已重新 read_thread；其最新可见工作是 U 星地训练及物理噪声代理分析，仍有健康训练进行。本路只读其上下文，没有发送指令、改动或重启其任务，不把对话进度当成本路成绩。

## 最小后续实验和放弃规则

本草案的最小决策已经完成：先查原文和关闭范围；结论是不启动 source guard 或 E30，不增加一次缺乏独立依据的训练。当前没有可据实宣布的下一条已支持 IQ A_g 假设；保留这个缺口，不用任意新扰动填补。

后续若出现不同且有依据的观测机制，仍需固定完整五 RX 合约与 E30，在额外视图/前向和原始监督份额匹配下分别检验 B00/B10/B01/B11，报告七视图、配对差和交互项。组合 A_s(A_g(x)) 另行消融。单种子低于预定识别增量或 clean 保护条件时关闭固定候选，不依据 target 重调或选择性补跑；不额外添加审批、身份诊断头或健康检查替代正式识别。
