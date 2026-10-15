# 幅度谱候选结束后的唯一源问题

固定幅度谱方向 EMP 的独立 E20 源资格 FAIL，按原规则结束，不继续调幅、联合或扩种子。正式成绩只用于报告，不用于选择 ROT、改变阈值或挑更有利 seed。旧33tapPHYS/ROT也没有建立实测方向优势；不能把这一路线当作已获得泛化增强。

当前尚未检验的新问题是：**合法源接收观测中，排除整体复增益、CFO及时移后，是否仍有跨 TX/day 可重复的频率相位关联？** 当前候选只研究幅度谱，不能由它的负结果推断相位可用或不可用。

依据是既存源材料，不是本轮目标成绩。20261002已知激励诊断对L6300/V27000、IQ256/25MHz/equalized1/RMS、固定80:160窗口执行过包含CFO和320时移的模板匹配；RX3/8相关仍较低，而RX关联远大于TX关联。此事实说明不能直接归为简单窗口错位，也不能区分幅度、相位、传播、均衡、硬件或噪声。L-STF的共轭/时移/复增益歧义仍须保留，不能输出唯一RX硬件或TX PA参数。

历史20261012有限实际操作研究已经用±1.5dB FIR、±0.25sample插值进行五个源RX完全留出E20比较。ACTION_CE较replay留出clean均值仅+0.3408pp；FINITE_MARGIN未过预定规则。因此不重复“增加有限间隔或微小时移就能改善”的路线，也不另训诊断头。

最小下一步先检索并复用既存复数统计。若缺少能回答该问题的数组，只用合法L6300、固定主窗口及跨TX/day排除折做零模型统计；V/U/目标不参与拟合，不加载checkpoint，不作新的性能训练。先明确packet内相位规范、时移/CFO歧义、低能量tone的可用性和主判定，再读取新统计；不把低质量噪声相位当成硬件变化。

如果剩余相位关联不能跨TX/day重现，或主要来自规范/边界/低能量不确定性，就放弃相位增强。即使可重复，也只支持另一个有限IQ、当前默认头的源验证，不能跳过身份保留检验或宣称clean/urban有效。任何后续模型仍固定CVS/SIG_TIME+RDrop、默认头正常联合训练、总E20和原测试IQ；不追加泛化损失或部署分支。

当前只固定研究问题，未实现、登记、启动新的相位实验，也没有相位收益或跨seed结果。整体目标保持原clean/low urban/suburban要求，不缩减成统计诊断。

证据：[已知激励全量源诊断](E:/type10-7/automation_reports/CV-SincNet/20261002-diagnostic-rff-known-excitation-source-manysig-m1-r01/report.md)、[真实有限操作留出结果](E:/type10-7/automation_reports/CV-SincNet/20261012-phase1-finite-margin-source-manysig-s2026092701-r01/final_analysis.md)、[当前独立G源停止决定](cvs_source_response_source_decision_20261016.md)。
