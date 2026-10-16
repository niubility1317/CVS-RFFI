# 混合增强复盘与 F 下一步

截至本次诊断启动，clean 与 practical 双提升仍未证明，low urban 超过 60% 及跨 seed 复现仍未完成。后续基座唯一采用用户固定 F（三路 SIG_TIME＋统一 Sinc 修正），scratch 总 E20、原默认 cosine 头联合训练，原 U/旧门控/EMA 保持。下表区分旧基座正式测试、源域训练与冻结诊断，不能相加预测新模型收益。

| 已执行检验 | 实际证据 | 结论与限制 |
|---|---|---|
| 旧 mixedIQ 因子，B01−B00 | clean −2.8488 pp，low urban +20.1708 pp | practical 适应有效，但有 clean 代价；不是当前 F 的结果 |
| 同矩阵 B11−B01 | clean +0.0131 pp，low urban +0.0393 pp | bounded G 未提供有意义的独立增量；不能称联合突破 |
| 旧 ordinary 五个源 RX 留出 | 均值 +2.0296 pp；仅 3/5 正，RX8 −2.5926 pp | 首 RX 的大收益不能代表跨 RX 稳定泛化；五折不是五 seed |
| 冻结模型去 PA 输入干预 | 保持率均值 +1.1266 pp，accuracy +0.6002 pp | 只是条件输入效应 |
| 随后 NoPA 真实 E20 | 比完整 ordinary −3.3519 pp，macro F1 −3.5120 pp | 冻结输入改善不能推出删除该训练算子有效 |
| 当前 F raw/ordinary 配对 | A2−A0 +4.0370 pp；A2−A1 −0.6296 pp，F1 −0.5317 pp | 原始监督占一半仍未通过预定保持标准；不改比例救援或换主行 |
| 当前 F WHITE30 | W30−A0 +0.0185 pp（净 1/5400），F1 −0.1736 pp；修复 80、破坏 79 | 轻度独立噪声 Ag 失败；不扫强度/概率或扩 fold/seed |
| HQ 独立噪声 | 冻结 EMA 默认头源 L_dev720：low urban 64.8611%→70.9722%，0 更新 | 属于源诊断，不是目标测试或训练收益；不能将 6.11 pp 直接加入训练预期 |
| 原 SIG_TIME RDrop E20 | 三 seed clean 相对 ORIGINAL +0.8272 pp，2/3 正 | 随机子网络正则有 clean 证据，不等于 IQ Ag 或星地收益 |

普通增强旧 R_LEGACY 的 2 次 L 前向与 mixedIQ 四因子 6 次前向不同，不能用它们的差额给 ordinary 归因。课程、间隔、单分支/全部/joint 对齐、身份排序和 HQ/bridge 的历史负结果继续保留，不再堆入主线。当前 F raw-anchor 和 WHITE30 均匹配四次 L 前向、同原始物理顺序与 U 规则；其失效主要表现为 TX 间修复和新错误抵消，不能由训练 loss 降低宣称身份泛化增强。

证据见 [F raw-anchor 完整报告](../automation_reports/CV-SincNet/20261016-phase1-F-raw-anchor-ordinary-RX1-e20-s2026092701-m3-r01/report.md)、[WHITE30 完整报告](../automation_reports/CV-SincNet/20261016-phase1-F-white-observation-RX1-e20-s2026092701-m1-r01/report.md) 及 [先前混合研究与五折结论](cvs_mixed_augmentation_next_design_20261016.md)。失败候选已关闭，不根据新诊断重新晋级。

下一项唯一问题是：**固定 F 默认头在真实 practical residual 下，哪些源域身份错误能够通过移除某类损伤恢复；接收状态切换与该损伤的直接作用各占什么位置。** 不预设噪声、多径、相位或门控为主因。用两个在 W30 终点前已固定的 final20 学生，仅读取其未训练的原 source RX1 L1260；同一真实 IQ、同一命名信道流，clean＋六 practical×11 控制。分别冻结 full 接收状态/补偿/AGC与重算状态，检查白输入质量代理、波形实际功率、边界及 FIR 数值延迟。所有反事实仅解释源域，正式测试条件不变。

已实现并启动的 [具体诊断方案](cvs_F_practical_source_reaudit_20261016.md) 包含真实 time/frequency/PA/reference/三 SIG 路径、实际输入门控和默认头 /30 间隔。原 14 个参数张量的 CE 导数复用已有图，0 optimizer/EMA/head 更新，参数及 buffer 逐值保持。它们是解释量，不能替代识别收益。两个模型共预计 168844 个网络包前向、2680 次求导，1260 个独立物理包；每个父模型既有 E20/4440 更新另计。

诊断后先固定一个与风险证据相符的 IQ 训练处理，独立检验它对 A_s 或 A_g 的价值，不同时改变强度、损失、U、网络或头。后续真实 A_s 保留完整原 residual、EQoff G=1 与接收质量依赖；所有有标签 satellite 包参与 CE，clean 份额单独归一化，输入从同一原 IQ 独立构造。首次组合仍需无 Ag/无 As、仅 Ag、仅 As、联合四因子与匹配前向控制；As(Ag(x)) 另做消融，不能假定算子可交换。尚未固定新训练系数、登记或运行该下一训练，不在这里编造其结果。

放弃依据必须是新处理在匹配源训练中没有独立身份收益、clean 损伤超过预定范围，或收益集中于少数 TX/RX/day 的错误迁移。冻结反事实无分类修复时，不沿该损伤做主增强；有修复也只说明可恢复性，仍须真实训练验证。正式模型冻结后自动 clean＋六 practical、prediction 完成后独立 truth-last。目标成绩只作最终报告，不能回流改变候选或选择性重跑。

## 本轮诊断已完成

新诊断0训练、0target，168844网络包/2680导数完整运行并独立分析。A0/W30源clean81.1905/80.0000%，lowurban29.4444/28.7302%，未显示噪声Ag收益。A0单独去噪、flat_H、carrier/IQ移除均未恢复clean；公共核相位与24点观察原点仍未隔离。下一唯一假设及最小2×2方案见[设计](cvs_F_phase_origin_next_20261016.md)，尚未实现/运行；正式混合训练、lowurban>60%和跨seed目标仍未完成。
