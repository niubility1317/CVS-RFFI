# PAMF 半监督探索方法及测试排名（2026-10-27）

本报告汇总本对话 PAMF 半监督路线的所有已完成主配方、结构／损失消融及仅源域探索。主配方 42 条历史记录、12 条当前单视图记录（均含对照），另列 37 行结构／损失消融；重复模型引用只保留一份，消融中的 REPL／without_extrapolation／ordinary_PL／supervised 与历史配方重复，不再当成独立实验。所有结果均复用已完成预测和评分，本次新增训练／预测为 0。另一个对话的研发候选不纳入本对话排名。

**当前要求下尚无候选全面超过普通 PL。** 单视图候选中 clean 最好为 RXTrust-A：84.6107%，仍低于 PL 的 84.6938%；六景最好为 RXTrust-PM：70.5692%，较 PL +0.3359 个百分点，但 clean −0.1510 个百分点。历史完整网络主配方 clean 最好为 REPL：85.7268%；包括结构控制时，REPL_SHUFFLE_TIME 为 86.7107%，但其六景仅 70.7634%。历史 raw 六景最好为全 U RDrop：72.7124%，其 clean 为 84.9292%。这些是不同方法，不能合并为一次达标结果。

## 统计口径

各表按 clean 均值降序，同列给出六景均值。正式主口径固定 F0 原 168000 包／seed＝cross_rx 126000＋cross_both 42000，按正确数合并，不能将两人群等权平均。六景为 practical_high、practical_mid、practical_low_suburban、practical_high_urban、practical_mid_urban、practical_low_urban，先在每个 seed 内等权平均，再求三个模型的均值与样本标准差。三个模型 seed 固定为 2026092701／02／03。

主表均为 E50／11100 步。历史不同方法存在 E30＋20、E20＋30或从零训练及不同 U 前向／损失剂量，故排序是最终性能描述，不是等计算预算的单机制因果比较。历史 X3 与当前单视图 SIG_TIME 不混作匹配消融；严格 CE 的 L 损失和 fullT1 L 也不同。

数据角色为 L6300／U56700（TX 隐藏、RX 已知）／源 V27000，原物理划分保持。正式分数均来自冻结默认头预测后的独立 truth-last；本次不重读逐包目标真值。目标基准已在连续研发中暴露，三 seed SD 不能单独证明显著性或新的独立盲测。原联合绝对要求仍为 clean ≥85.981746%、六景 >72.096197%，并保留逐景／分层要求；尚未达到。

## 一、当前完整 PAMF＋单视图 SIG_TIME：三 seed 正式排名

保留原 fullT1 L、普通 PL 0.95／权重 1、原默认头联合训练；候选只使用 clean IQ 与同包星地派生物及有界接收响应作用。此组训练／测试不接 DURA。12 行是 10 个新候选＋2 个已完成匹配对照。

| 排名 | 方法 | 机制 | clean（%） | 六景均值（%） | 较 PL：clean／六景（pp） | 证据 |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | SIG1-PL | 普通 PL（阈值 0.95，匹配对照） | 84.6938 ± 0.6925 | 70.2332 ± 1.0433 | +0.0000／+0.0000 | [summary.csv](E:/type10-7/automation_reports/CV-SincNet/20261026-phase1-PAMF-SIG1-matched-baselines-m6-r01/summary.csv) |
| 2 | RXTrust-A | 可信 raw 硬标签锚；拒绝 U 保持软一致性 | 84.6107 ± 0.6609 | 70.5004 ± 1.0191 | -0.0831／+0.2671 | [analysis.json](E:/type10-7/automation_reports/CV-SincNet/20261027-phase1-PAMF-RXTrustA-SIG1-threeSeed-m3-r01/results/analysis.json) |
| 3 | CovSel-U | 仅 U 的选择性通道相关性一致性 | 84.5784 ± 0.7641 | 70.2862 ± 1.0519 | -0.1155／+0.0530 | [analysis.json](E:/type10-7/automation_reports/CV-SincNet/20261027-phase1-PAMF-Covariance-SIG1-threeSeed-m12-r01/results/analysis.json) |
| 4 | CovRandom-U | 仅 U 的随机同尺寸相关项一致性 | 84.5706 ± 0.8272 | 70.1429 ± 1.2969 | -0.1232／-0.0903 | [analysis.json](E:/type10-7/automation_reports/CV-SincNet/20261027-phase1-PAMF-Covariance-SIG1-threeSeed-m12-r01/results/analysis.json) |
| 5 | RXMean-MaxU | 逐包选择 KL 最大的有界 RX 响应风格 | 84.5446 ± 0.6595 | 70.2621 ± 1.1848 | -0.1492／+0.0288 | [analysis.json](E:/type10-7/automation_reports/CV-SincNet/20261027-phase1-PAMF-RXTrustP-SIG1-threeSeed-m3-r01/results/analysis.json) |
| 6 | RXTrust-PM | 候选集合监督＋raw/H 分模式源梯度定剂量 | 84.5429 ± 0.6447 | 70.5692 ± 1.0977 | -0.1510／+0.3359 | [analysis.json](E:/type10-7/automation_reports/CV-SincNet/20261027-phase1-PAMF-RXTrustA-SIG1-threeSeed-m3-r01/results/analysis.json) |
| 7 | CovFull-U | 仅 U 的全部通道相关项一致性 | 84.5329 ± 0.8533 | 70.3334 ± 1.1480 | -0.1609／+0.1001 | [analysis.json](E:/type10-7/automation_reports/CV-SincNet/20261027-phase1-PAMF-Covariance-SIG1-threeSeed-m12-r01/results/analysis.json) |
| 8 | RXMean-U | 仅 U 的随机 RX 均值差风格一致性 | 84.5323 ± 0.5589 | 70.3646 ± 1.0737 | -0.1615／+0.1314 | [analysis.json](E:/type10-7/automation_reports/CV-SincNet/20261027-phase1-PAMF-RXMeanMax-SIG1-threeSeed-m3-r01/results/analysis.json) |
| 9 | CovSel-LU | L/U 共同承受选择性相关项一致性 | 84.5272 ± 0.7119 | 70.2478 ± 1.0627 | -0.1667／+0.0146 | [analysis.json](E:/type10-7/automation_reports/CV-SincNet/20261027-phase1-PAMF-Covariance-SIG1-threeSeed-m12-r01/results/analysis.json) |
| 10 | RXMean-LU | L/U 共同承受随机 RX 风格一致性 | 84.5256 ± 0.6291 | 70.3581 ± 1.0975 | -0.1683／+0.1249 | [analysis.json](E:/type10-7/automation_reports/CV-SincNet/20261027-phase1-PAMF-RXMean-SIG1-threeSeed-m6-r02/results/analysis.json) |
| 11 | RXTrust-P | raw/H 候选集合监督，单标签或双标签集合 | 84.3679 ± 1.0836 | 69.7168 ± 2.2955 | -0.3260／-0.5164 | [analysis.json](E:/type10-7/automation_reports/CV-SincNet/20261027-phase1-PAMF-RXTrustPM-SIG1-threeSeed-m3-r01/results/analysis.json) |
| 12 | SIG1-Sup | 匹配监督对照 | 84.1169 ± 0.8876 | 70.1681 ± 1.0352 | -0.5770／-0.0652 | [summary.csv](E:/type10-7/automation_reports/CV-SincNet/20261026-phase1-PAMF-SIG1-matched-baselines-m6-r01/summary.csv) |

CovSel-LU 相对 CovSel-U 的 clean／六景分别再低 0.0512／0.0384 pp；RXMean-LU 相对 RXMean-U 分别低 0.0067／0.0065 pp。目前没有证据支持让 L 无条件复制 U 的新增一致性约束。AL 正在另外检验 L 的真实标签 CE，尚无最终测试结论，不能据这些结果推断 L 永远不应参与。

## 二、历史 SIG_TIME_X3 主配方：三 seed 正式排名

包括早期物理匹配、完整 U 监督组、梯度保护、共识和视图重分配。raw 为本表排序依据。历史 G 列仅保存当时固定 DURA-LU 的独立读出结果，不是当前研发路线，也不用于替代 raw 的验收。F20/F21/PLControl 是对照，部分候选含自身 EMA 或旧门控，与当前普通 PL 叠加路线分开解释。

| 排名 | 方法 | 机制 | clean（%） | 六景均值（%） | 较 PL：clean／六景（pp） | 历史 G：clean／六景（%） | 证据 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | HPL_E20_JOINT_RXE6G | PAMF-REPL：接收机特征外推伪标签学习 | 85.7268 ± 0.6415 | 70.0855 ± 0.4689 | +0.7450／-0.2738 | 85.7907／77.8977 | [20261024-phase1-PAMF-HardPL-ReceiverExtrap-m3-r01](E:/type10-7/automation_reports/CV-SincNet/20261024-phase1-PAMF-HardPL-ReceiverExtrap-m3-r01) |
| 2 | HPL_E20_NAMS_JOINT6G | 增加 joint160 跨 RX 对比 | 85.6417 ± 0.7136 | 70.1712 ± 0.4019 | +0.6599／-0.1881 | 85.6855／78.0808 | [20261024-phase1-PAMF-HardPL-Joint160-m3-r01](E:/type10-7/automation_reports/CV-SincNet/20261024-phase1-PAMF-HardPL-Joint160-m3-r01) |
| 3 | HPL_E20_NAMS6G | 无 U As＋MixStyle＋权重 6＋梯度保护 | 85.5008 ± 0.5032 | 70.2834 ± 0.3976 | +0.5190／-0.0759 | 85.5435／77.9146 | [20261024-phase1-PAMF-HardPL-GradientGuard-m6-r01](E:/type10-7/automation_reports/CV-SincNet/20261024-phase1-PAMF-HardPL-GradientGuard-m6-r01) |
| 4 | HPL_E20_JOINT_RAS6G | joint160＋可靠 As 使用，否则 raw 回退 | 85.4488 ± 0.6240 | 70.7015 ± 0.3803 | +0.4671／+0.3421 | 85.5187／77.8642 | [20261024-phase1-PAMF-HardPL-ReliableAs-m3-r01](E:/type10-7/automation_reports/CV-SincNet/20261024-phase1-PAMF-HardPL-ReliableAs-m3-r01) |
| 5 | HPL_NO_UAS | 完整 U 监督组中删除 U 的受损 As 输入 | 85.4230 ± 0.3124 | 70.6572 ± 0.4853 | +0.4413／+0.2979 | 85.4938／77.9856 | [20261024-phase1-PAMF-HardPL-noUAs-m3-r01](E:/type10-7/automation_reports/CV-SincNet/20261024-phase1-PAMF-HardPL-noUAs-m3-r01) |
| 6 | HPL_NAMS2 | 无 U As＋跨 RX MixStyle | 85.3111 ± 0.1199 | 70.5730 ± 0.6791 | +0.3294／+0.2136 | 85.4000／77.8250 | [20261024-phase1-PAMF-HardPL-NoAsMixStyle-m3-r01](E:/type10-7/automation_reports/CV-SincNet/20261024-phase1-PAMF-HardPL-NoAsMixStyle-m3-r01) |
| 7 | HPL_E20_NAMS2 | E20 起启用无 U As＋MixStyle | 85.2462 ± 0.4201 | 70.7302 ± 0.6168 | +0.2645／+0.3709 | 85.3319／77.7059 | [20261024-phase1-PAMF-HardPL-E20plus30-m6-r01](E:/type10-7/automation_reports/CV-SincNet/20261024-phase1-PAMF-HardPL-E20plus30-m6-r01) |
| 8 | HPL_E20_LS6G | 平滑硬 PL 权重 6＋L 梯度保护 | 85.1016 ± 0.0868 | 69.9360 ± 0.6865 | +0.1198／-0.4233 | 85.0966／78.2380 | [20261024-phase1-PAMF-HardPL-GradientGuard-m6-r01](E:/type10-7/automation_reports/CV-SincNet/20261024-phase1-PAMF-HardPL-GradientGuard-m6-r01) |
| 9 | HPL2 | 普通硬 PL 增权至 2 | 85.0756 ± 0.0757 | 70.1875 ± 0.9468 | +0.0938／-0.1718 | 85.1038／78.0787 | [20261024-phase1-PAMF-HardPL-T1U-m6-r01](E:/type10-7/automation_reports/CV-SincNet/20261024-phase1-PAMF-HardPL-T1U-m6-r01) |
| 10 | HPL_LS2 | 硬 PL 权重 2＋标签平滑 0.01 | 85.0700 ± 0.1786 | 70.3003 ± 0.7857 | +0.0883／-0.0590 | 85.1004／78.1316 | [20261024-phase1-PAMF-HardPL-LS-m3-r01](E:/type10-7/automation_reports/CV-SincNet/20261024-phase1-PAMF-HardPL-LS-m3-r01) |
| 11 | HPL_E20_LS2 | E20 起启用硬 PL＋平滑 | 85.0528 ± 0.1318 | 70.2430 ± 0.8401 | +0.0710／-0.1164 | 85.0839／78.0898 | [20261024-phase1-PAMF-HardPL-E20plus30-m6-r01](E:/type10-7/automation_reports/CV-SincNet/20261024-phase1-PAMF-HardPL-E20plus30-m6-r01) |
| 12 | HPL_MS2 | 硬 PL＋平滑＋浅中层跨 RX MixStyle | 84.9964 ± 0.2497 | 70.3283 ± 0.6652 | +0.0147／-0.0310 | 85.0433／78.0759 | [20261024-phase1-PAMF-HardPL-MixStyle-m3-r02](E:/type10-7/automation_reports/CV-SincNet/20261024-phase1-PAMF-HardPL-MixStyle-m3-r02) |
| 13 | F21 | 普通 PL 对照 | 84.9817 ± 0.1893 | 70.3594 ± 0.9576 | +0.0000／+0.0000 | 85.0296／78.0746 | [20261024-phase1-PAMF-HardPL-T1U-m6-r01](E:/type10-7/automation_reports/CV-SincNet/20261024-phase1-PAMF-HardPL-T1U-m6-r01) |
| 14 | PLControl | 恢复软标签实验的匹配普通 PL 对照 | 84.9671 ± 0.2248 | 70.3349 ± 0.9566 | -0.0147／-0.0245 | 85.0222／78.0616 | [20261025-phase1-PAMF-RestoredPL-m9-r01](E:/type10-7/automation_reports/CV-SincNet/20261025-phase1-PAMF-RestoredPL-m9-r01) |
| 15 | HPL_E20_JOINT_COMP6G | 完整融合／无签名路径互补 CE | 84.9589 ± 0.0503 | 70.2528 ± 0.8477 | -0.0228／-0.1066 | 85.0401／76.4464 | [20261024-phase1-PAMF-HardPL-Complement-m3-r01](E:/type10-7/automation_reports/CV-SincNet/20261024-phase1-PAMF-HardPL-Complement-m3-r01) |
| 16 | HPL_E20_AS_ALLRD | 原生 U As＋被 PL 拒绝 U 的 RDrop | 84.9292 ± 0.4897 | 72.7124 ± 0.2128 | -0.0526／+2.3530 | 85.0454／77.3179 | [20261024-phase1-PAMF-HardPL-AllURDrop-m3-r01](E:/type10-7/automation_reports/CV-SincNet/20261024-phase1-PAMF-HardPL-AllURDrop-m3-r01) |
| 17 | ChannelPL | 可信受损视图的 CE 剂量重分配 | 84.8790 ± 0.4067 | 70.5894 ± 0.9856 | -0.1028／+0.2301 | 84.9514／77.9922 | [20261025-phase1-PAMF-ChannelPL-m33-r01](E:/type10-7/automation_reports/CV-SincNet/20261025-phase1-PAMF-ChannelPL-m33-r01) |
| 18 | HPL_E20_AS_BCE6G | 原生 U As＋平衡伪标签 CE | 84.8786 ± 0.1199 | 72.3230 ± 0.4979 | -0.1032／+1.9637 | 84.9861／77.4406 | [20261024-phase1-PAMF-HardPL-BalancedCE-m3-r01](E:/type10-7/automation_reports/CV-SincNet/20261024-phase1-PAMF-HardPL-BalancedCE-m3-r01) |
| 19 | RawSoft | 直接 raw 软标签 CE | 84.8728 ± 0.2547 | 70.2627 ± 0.8326 | -0.1089／-0.0966 | 84.9270／78.0118 | [20261025-phase1-PAMF-RestoredPL-m9-r01](E:/type10-7/automation_reports/CV-SincNet/20261025-phase1-PAMF-RestoredPL-m9-r01) |
| 20 | HPLT1U | 给可信 U 施加完整 T1 监督组机制 | 84.8671 ± 0.1338 | 71.9747 ± 0.7905 | -0.1147／+1.6153 | 84.9494／77.5089 | [20261024-phase1-PAMF-HardPL-T1U-m6-r01](E:/type10-7/automation_reports/CV-SincNet/20261024-phase1-PAMF-HardPL-T1U-m6-r01) |
| 21 | HPL_E20_JOINT_AS6G | 恢复原生 U As 的完整监督组 | 84.8319 ± 0.0956 | 72.2644 ± 0.5159 | -0.1498／+1.9050 | 84.9444／77.4284 | [20261024-phase1-PAMF-HardPL-NativeAs-m3-r01](E:/type10-7/automation_reports/CV-SincNet/20261024-phase1-PAMF-HardPL-NativeAs-m3-r01) |
| 22 | HPL_E20_PATH_RAS6G | 路径互补监督＋可靠 As | 84.8210 ± 0.2409 | 70.7584 ± 0.6571 | -0.1607／+0.3990 | 84.9321／76.3863 | [20261024-phase1-PAMF-HardPL-PathReliableAs-m3-r01](E:/type10-7/automation_reports/CV-SincNet/20261024-phase1-PAMF-HardPL-PathReliableAs-m3-r01) |
| 23 | HPL_E20_JOINT_PATH6G | 完整／无签名／core 三路径互补 CE | 84.8187 ± 0.1712 | 70.3020 ± 0.5835 | -0.1631／-0.0574 | 84.9028／76.2970 | [20261024-phase1-PAMF-HardPL-PathComplement-m3-r01](E:/type10-7/automation_reports/CV-SincNet/20261024-phase1-PAMF-HardPL-PathComplement-m3-r01) |
| 24 | PeerT1U2 | 共识标签＋完整 U 监督组，剂量 2 | 84.8050 ± 0.2109 | 72.2796 ± 0.7886 | -0.1768／+1.9203 | 84.8980／77.4413 | [20261025-phase1-PAMF-PeerSupervised-m9-r02](E:/type10-7/automation_reports/CV-SincNet/20261025-phase1-PAMF-PeerSupervised-m9-r02) |
| 25 | PeerPL2 | 三教师共识伪标签＋PL 权重 2 | 84.8048 ± 0.0400 | 70.3234 ± 0.9622 | -0.1770／-0.0360 | 84.8444／77.9449 | [20261025-phase1-PAMF-PeerSupervised-m9-r02](E:/type10-7/automation_reports/CV-SincNet/20261025-phase1-PAMF-PeerSupervised-m9-r02) |
| 26 | PeerT1U1 | 共识标签＋完整 U 监督组，剂量 1 | 84.7893 ± 0.2301 | 71.8086 ± 0.8106 | -0.1925／+1.4493 | 84.8764／77.6909 | [20261025-phase1-PAMF-PeerSupervised-m9-r02](E:/type10-7/automation_reports/CV-SincNet/20261025-phase1-PAMF-PeerSupervised-m9-r02) |
| 27 | F20 | PAMF 匹配监督对照 | 84.7853 ± 0.8236 | 70.5962 ± 0.9519 | -0.1964／+0.2368 | 84.8575／77.8549 | [20261024-phase1-PAMF-HardPL-T1U-m6-r01](E:/type10-7/automation_reports/CV-SincNet/20261024-phase1-PAMF-HardPL-T1U-m6-r01) |
| 28 | UVB | raw／受损视图梯度分别投影并平衡 | 84.6117 ± 0.2265 | 71.6754 ± 0.8876 | -0.3700／+1.3161 | 84.6369／77.5453 | [20261023-phase1-PAMF-U-ViewBalance-m3-r01](E:/type10-7/automation_reports/CV-SincNet/20261023-phase1-PAMF-U-ViewBalance-m3-r01) |
| 29 | PAIREDU | raw／星地受损 U 联合训练 | 84.5438 ± 0.2914 | 71.9783 ± 0.8816 | -0.4379／+1.6189 | 84.6204／77.5076 | [20261023-phase1-PAMF-paired-U-m3-r01](E:/type10-7/automation_reports/CV-SincNet/20261023-phase1-PAMF-paired-U-m3-r01) |
| 30 | HPL_E20_AS_BG2 | 原生 U As＋分支梯度保护 | 84.5270 ± 0.1650 | 72.4817 ± 0.4239 | -0.4548／+2.1224 | 84.6298／77.0329 | [20261024-phase1-PAMF-HardPL-BranchGuard-m3-r01](E:/type10-7/automation_reports/CV-SincNet/20261024-phase1-PAMF-HardPL-BranchGuard-m3-r01) |
| 31 | HPL_E20_AS_SG2 | 原生 U As＋实际优化步保护 | 84.4857 ± 0.2427 | 72.4150 ± 0.4424 | -0.4960／+2.0557 | 84.5645／77.0139 | [20261024-phase1-PAMF-HardPL-StepGuard-m3-r01](E:/type10-7/automation_reports/CV-SincNet/20261024-phase1-PAMF-HardPL-StepGuard-m3-r01) |
| 32 | HPL_E20_AS_CAP2 | 原生 U As＋辅助梯度上限 2 | 84.4611 ± 0.2075 | 72.4163 ± 0.4438 | -0.5206／+2.0569 | 84.5512／77.0643 | [20261024-phase1-PAMF-HardPL-NativeAsCap2-m3-r01](E:/type10-7/automation_reports/CV-SincNet/20261024-phase1-PAMF-HardPL-NativeAsCap2-m3-r01) |
| 33 | VIEWANCHOR | RXPSM＋同包物理视图特征锚 | 84.3796 ± 0.2666 | 71.9470 ± 0.9653 | -0.6022／+1.5876 | 84.4482／77.5218 | [20261023-phase1-PAMF-SSL-DURA-reference-m9-r01](E:/type10-7/automation_reports/CV-SincNet/20261023-phase1-PAMF-SSL-DURA-reference-m9-r01) |
| 34 | UMIX | U 浅中层同伪类跨 RX MixStyle | 84.3629 ± 0.3971 | 72.0009 ± 1.0123 | -0.6188／+1.6415 | 84.4331／77.5541 | [20261023-phase1-PAMF-U-MixStyle-m3-r01](E:/type10-7/automation_reports/CV-SincNet/20261023-phase1-PAMF-U-MixStyle-m3-r01) |
| 35 | HPL_E20_AS_SGNC | 实际优化步保护，去除范数上限 | 84.3478 ± 0.2274 | 72.4886 ± 0.3551 | -0.6339／+2.1293 | 84.4516／76.8797 | [20261024-phase1-PAMF-HardPL-StepNoCap-m3-r01](E:/type10-7/automation_reports/CV-SincNet/20261024-phase1-PAMF-HardPL-StepNoCap-m3-r01) |
| 36 | UCMR | U 同伪类跨 RX 均值／标准差匹配 | 84.3365 ± 0.0411 | 71.9441 ± 0.9802 | -0.6452／+1.5848 | 84.4552／77.1752 | [20261023-phase1-PAMF-U-CMR-m3-r01](E:/type10-7/automation_reports/CV-SincNet/20261023-phase1-PAMF-U-CMR-m3-r01) |
| 37 | UGP | 对 U 梯度作 L 锚定单侧投影与范数限制 | 84.2954 ± 0.3609 | 71.8183 ± 1.0184 | -0.6863／+1.4589 | 84.3665／77.4200 | [20261023-phase1-PAMF-U-GradProtect-m3-r01](E:/type10-7/automation_reports/CV-SincNet/20261023-phase1-PAMF-U-GradProtect-m3-r01) |
| 38 | UCRIC | 扩展 U 浅中层同伪类跨 RX 对比 | 84.2907 ± 0.5713 | 71.8662 ± 1.1517 | -0.6911／+1.5068 | 84.4002／77.1860 | [20261023-phase1-PAMF-U-CRIC-m3-r01](E:/type10-7/automation_reports/CV-SincNet/20261023-phase1-PAMF-U-CRIC-m3-r01) |
| 39 | RXPSM | 物理软匹配＋其他 RX 的有标签原型 | 84.2724 ± 0.3597 | 71.8536 ± 1.0301 | -0.7093／+1.4942 | 84.3486／77.4202 | [20261023-phase1-PAMF-SSL-DURA-reference-m9-r01](E:/type10-7/automation_reports/CV-SincNet/20261023-phase1-PAMF-SSL-DURA-reference-m9-r01) |
| 40 | UDAP | 完整 L 与 raw-L CE 双梯度锚 | 84.2548 ± 0.3764 | 71.8053 ± 1.0538 | -0.7270／+1.4460 | 84.3363／77.4297 | [20261023-phase1-PAMF-U-DualAnchor-m3-r01](E:/type10-7/automation_reports/CV-SincNet/20261023-phase1-PAMF-U-DualAnchor-m3-r01) |
| 41 | PSM | 物理视图软匹配／置信加权 | 84.2226 ± 0.3409 | 71.8973 ± 1.1559 | -0.7591／+1.5379 | 84.2911／77.3528 | [20261023-phase1-PAMF-SSL-DURA-reference-m9-r01](E:/type10-7/automation_reports/CV-SincNet/20261023-phase1-PAMF-SSL-DURA-reference-m9-r01) |
| 42 | DURASoft | 直接恢复 IQ 软标签 CE（已退出路线） | 83.6845 ± 0.3091 | 70.0562 ± 0.8162 | -1.2972／-0.3032 | 83.7510／76.4049 | [20261025-phase1-PAMF-RestoredPL-m9-r01](E:/type10-7/automation_reports/CV-SincNet/20261025-phase1-PAMF-RestoredPL-m9-r01) |

## 三、完整结构与 L／U 损失消融：三 seed 正式排名

此表从原 m201 的 F0 raw 结果按 126000／42000 加权。37 行完整保留，包括严格 CE 和原 REPL 单视图。为避免把分群 SD 直接加权成整体 SD，本表只列整体均值；原 cross_rx／cross_both 的三 seed SD 保留在来源表。本表改变网络或 L 配方，不能把更高 clean 自动解释为半监督单项收益。

| 排名 | 方法 | 机制 | clean（%） | 六景均值（%） | 证据 |
| --- | --- | --- | --- | --- | --- |
| 1 | REPL_SHUFFLE_TIME | REPL 打乱时间顺序 | 86.7107 | 70.7634 | [summary_overall.csv](E:/type10-7/exports/unified_figures/20261025-PAMF-REPL-complete-paper-r01/source_data/summary_overall.csv) |
| 2 | L_without_raw_CE | L 删除 raw CE | 85.9351 | 70.5162 | [summary_overall.csv](E:/type10-7/exports/unified_figures/20261025-PAMF-REPL-complete-paper-r01/source_data/summary_overall.csv) |
| 3 | without_style | REPL 删除 U 风格混合 | 85.8476 | 70.4419 | [summary_overall.csv](E:/type10-7/exports/unified_figures/20261025-PAMF-REPL-complete-paper-r01/source_data/summary_overall.csv) |
| 4 | L_single_forward | L 改为单次前向 | 85.8312 | 70.6403 | [summary_overall.csv](E:/type10-7/exports/unified_figures/20261025-PAMF-REPL-complete-paper-r01/source_data/summary_overall.csv) |
| 5 | REPL_COV_X3 | REPL 以三视图协方差描述替代签名 | 85.7873 | 70.4464 | [summary_overall.csv](E:/type10-7/exports/unified_figures/20261025-PAMF-REPL-complete-paper-r01/source_data/summary_overall.csv) |
| 6 | REPL | 完整 REPL，等同历史接收机外推配方 | 85.7268 | 70.0855 | [summary_overall.csv](E:/type10-7/exports/unified_figures/20261025-PAMF-REPL-complete-paper-r01/source_data/summary_overall.csv) |
| 7 | L_As_KL_on | L 开启独立 As KL | 85.7083 | 70.3491 | [summary_overall.csv](E:/type10-7/exports/unified_figures/20261025-PAMF-REPL-complete-paper-r01/source_data/summary_overall.csv) |
| 8 | ABC011 | 核心开关 011（A=L跨RX对比，B=L配对KL，C=完整REPL U） | 85.6762 | 69.8976 | [summary_overall.csv](E:/type10-7/exports/unified_figures/20261025-PAMF-REPL-complete-paper-r01/source_data/summary_overall.csv) |
| 9 | L_joint_only | L 跨 RX 对比仅 joint 层 | 85.6450 | 69.5589 | [summary_overall.csv](E:/type10-7/exports/unified_figures/20261025-PAMF-REPL-complete-paper-r01/source_data/summary_overall.csv) |
| 10 | without_extrapolation | REPL 删除接收机外推 | 85.6417 | 70.1712 | [summary_overall.csv](E:/type10-7/exports/unified_figures/20261025-PAMF-REPL-complete-paper-r01/source_data/summary_overall.csv) |
| 11 | L_middle_only | L 跨 RX 对比仅中层 | 85.6367 | 70.0348 | [summary_overall.csv](E:/type10-7/exports/unified_figures/20261025-PAMF-REPL-complete-paper-r01/source_data/summary_overall.csv) |
| 12 | ABC001 | 核心开关 001（A=L跨RX对比，B=L配对KL，C=完整REPL U） | 85.6341 | 69.7911 | [summary_overall.csv](E:/type10-7/exports/unified_figures/20261025-PAMF-REPL-complete-paper-r01/source_data/summary_overall.csv) |
| 13 | without_guard | REPL 删除 U 梯度保护 | 85.6155 | 69.9377 | [summary_overall.csv](E:/type10-7/exports/unified_figures/20261025-PAMF-REPL-complete-paper-r01/source_data/summary_overall.csv) |
| 14 | L_shallow_only | L 跨 RX 对比仅浅层 | 85.5998 | 70.2860 | [summary_overall.csv](E:/type10-7/exports/unified_figures/20261025-PAMF-REPL-complete-paper-r01/source_data/summary_overall.csv) |
| 15 | without_joint | REPL 删除 joint160 对比 | 85.5246 | 70.1768 | [summary_overall.csv](E:/type10-7/exports/unified_figures/20261025-PAMF-REPL-complete-paper-r01/source_data/summary_overall.csv) |
| 16 | ABC101 | 核心开关 101（A=L跨RX对比，B=L配对KL，C=完整REPL U） | 85.4893 | 69.9363 | [summary_overall.csv](E:/type10-7/exports/unified_figures/20261025-PAMF-REPL-complete-paper-r01/source_data/summary_overall.csv) |
| 17 | L_without_As | L 删除受损 As 监督 | 85.3069 | 41.9298 | [summary_overall.csv](E:/type10-7/exports/unified_figures/20261025-PAMF-REPL-complete-paper-r01/source_data/summary_overall.csv) |
| 18 | REPL_WRONG_REFERENCE | REPL 使用错误参考控制 | 85.2460 | 70.1170 | [summary_overall.csv](E:/type10-7/exports/unified_figures/20261025-PAMF-REPL-complete-paper-r01/source_data/summary_overall.csv) |
| 19 | REPL_NO_TIME | REPL 去时间坐标 | 85.1877 | 70.1394 | [summary_overall.csv](E:/type10-7/exports/unified_figures/20261025-PAMF-REPL-complete-paper-r01/source_data/summary_overall.csv) |
| 20 | ordinary_PL | 完整 L 机制＋普通 PL | 84.9817 | 70.3594 | [summary_overall.csv](E:/type10-7/exports/unified_figures/20261025-PAMF-REPL-complete-paper-r01/source_data/summary_overall.csv) |
| 21 | REPL_SIG_TIME | REPL 改为单视图 SIG_TIME（原 U 配方） | 84.8133 | 69.4062 | [summary_overall.csv](E:/type10-7/exports/unified_figures/20261025-PAMF-REPL-complete-paper-r01/source_data/summary_overall.csv) |
| 22 | supervised | 完整 L 机制的监督对照 | 84.7853 | 70.5962 | [summary_overall.csv](E:/type10-7/exports/unified_figures/20261025-PAMF-REPL-complete-paper-r01/source_data/summary_overall.csv) |
| 23 | ABC010 | 核心开关 010（A=L跨RX对比，B=L配对KL，C=完整REPL U） | 84.6391 | 70.9695 | [summary_overall.csv](E:/type10-7/exports/unified_figures/20261025-PAMF-REPL-complete-paper-r01/source_data/summary_overall.csv) |
| 24 | REPL_NO_REFERENCE | REPL 删除参考响应分支 | 84.4058 | 69.7844 | [summary_overall.csv](E:/type10-7/exports/unified_figures/20261025-PAMF-REPL-complete-paper-r01/source_data/summary_overall.csv) |
| 25 | CE_SHUFFLE_TIME | 严格 CE 打乱时间顺序 | 83.5798 | 71.3980 | [summary_overall.csv](E:/type10-7/exports/unified_figures/20261025-PAMF-REPL-complete-paper-r01/source_data/summary_overall.csv) |
| 26 | ABC100 | 核心开关 100（A=L跨RX对比，B=L配对KL，C=完整REPL U） | 83.2593 | 70.4584 | [summary_overall.csv](E:/type10-7/exports/unified_figures/20261025-PAMF-REPL-complete-paper-r01/source_data/summary_overall.csv) |
| 27 | CE_SIG_TIME | 严格 CE＋单视图 SIG_TIME | 83.1405 | 70.0831 | [summary_overall.csv](E:/type10-7/exports/unified_figures/20261025-PAMF-REPL-complete-paper-r01/source_data/summary_overall.csv) |
| 28 | CE_NO_SIGNATURE | 严格 CE 删除整个签名描述分支 | 82.7238 | 70.4615 | [summary_overall.csv](E:/type10-7/exports/unified_figures/20261025-PAMF-REPL-complete-paper-r01/source_data/summary_overall.csv) |
| 29 | CE_full | 严格 CE 的完整网络参照 | 82.6653 | 70.9105 | [summary_overall.csv](E:/type10-7/exports/unified_figures/20261025-PAMF-REPL-complete-paper-r01/source_data/summary_overall.csv) |
| 30 | REPL_NO_PA | REPL 删除 PA 分支 | 82.6367 | 62.2767 | [summary_overall.csv](E:/type10-7/exports/unified_figures/20261025-PAMF-REPL-complete-paper-r01/source_data/summary_overall.csv) |
| 31 | ABC000 | 核心开关 000（A=L跨RX对比，B=L配对KL，C=完整REPL U） | 82.6024 | 70.4453 | [summary_overall.csv](E:/type10-7/exports/unified_figures/20261025-PAMF-REPL-complete-paper-r01/source_data/summary_overall.csv) |
| 32 | CE_COV_X3 | 严格 CE＋三视图协方差描述 | 82.4940 | 71.2655 | [summary_overall.csv](E:/type10-7/exports/unified_figures/20261025-PAMF-REPL-complete-paper-r01/source_data/summary_overall.csv) |
| 33 | CE_NO_TIME | 严格 CE 去时间坐标 | 82.4877 | 70.3568 | [summary_overall.csv](E:/type10-7/exports/unified_figures/20261025-PAMF-REPL-complete-paper-r01/source_data/summary_overall.csv) |
| 34 | CE_WRONG_REFERENCE | 严格 CE 使用错误参考控制 | 81.9222 | 70.6671 | [summary_overall.csv](E:/type10-7/exports/unified_figures/20261025-PAMF-REPL-complete-paper-r01/source_data/summary_overall.csv) |
| 35 | CE_NO_REFERENCE | 严格 CE 删除参考响应分支 | 81.6337 | 70.3974 | [summary_overall.csv](E:/type10-7/exports/unified_figures/20261025-PAMF-REPL-complete-paper-r01/source_data/summary_overall.csv) |
| 36 | REPL_NO_SIGNATURE | REPL 删除整个签名描述分支 | 81.2940 | 67.1671 | [summary_overall.csv](E:/type10-7/exports/unified_figures/20261025-PAMF-REPL-complete-paper-r01/source_data/summary_overall.csv) |
| 37 | CE_NO_PA | 严格 CE 删除 PA 分支 | 78.5502 | 64.0948 | [summary_overall.csv](E:/type10-7/exports/unified_figures/20261025-PAMF-REPL-complete-paper-r01/source_data/summary_overall.csv) |

## 四、单 seed 正式确认与仅源域探索

这些记录不进入上述三 seed 正式排名。历史单 seed 是当时已完成的研究记录，后续三 seed 要求保持，不将一个 seed 的结果冒充稳定改善。

| 方法 | 实际证据 | 结果与限制 |
| --- | --- | --- |
| PL-History：稳定负类＋历史软目标 | 单 seed 正式 raw 七景，168000 包 | clean 85.101190%；六景 70.737599%。按 cross_rx／cross_both 的 3:1 比例合并；未扩成三 seed，未达完整目标。见 [target_summary.csv](E:/type10-7/automation_reports/CV-SincNet/20261026-phase1-PAMF-SIG1-UHistory-confirm-s2026092701-r01/target_summary.csv) |
| PL-Negative | 单 seed 源 V | 源 clean 98.370370%，raw 三景 74.711111%；较旧 X3 PL −0.055556／−0.416049 pp，无正式目标测试。 |
| PL-Temporal | 单 seed 源 V | 源 clean 98.311111%，raw 三景 74.704938%；较旧 X3 PL −0.114815／−0.422222 pp，无正式目标测试。 |
| PL-History 源阶段 | 单 seed 源 V | 源 clean 98.322222%，raw 三景 74.683951%；较旧 X3 PL −0.103704／−0.443210 pp。三行源证据见 [report.md](E:/type10-7/automation_reports/CV-SincNet/20261026-phase1-PAMF-SIG1-UHistory-source-m3-r01/report.md)。 |
| PL-VAT：全 U 虚拟对抗局部平滑 | 单 seed 完整 E50＋源 V | 源 clean 较旧 X3 PL −0.040741 pp，raw 三景 −0.217284 pp；未过源必要条件，无正式目标测试。见 [report.md](E:/type10-7/automation_reports/CV-SincNet/20261026-phase1-PAMF-SIG1-AllUVAT-source-s2026092701-r01/report.md)。 |
| 有界接收响应直接校正 | 三份固定 E20＋源 V；0 新训练 | clean／low suburban／mid urban／low urban 配对均差 −0.0284／−0.0333／−0.0543／−0.0432 pp；三 seed 均未过源规则。见 [results.md](E:/type10-7/automation_reports/CV-SincNet/20261027-diagnostic-PAMF-BoundedResponse-SIG1-source-m3-r01/results.md)。 |
| FastTrust 借鉴：困难 U 候选集合／负类风险校准 | 三 seed 冻结源诊断，0 新训练 | 单一概率质量评分未得到满足风险条件的阈值，新增路由为空；不能说已完成 FastTrust 训练或已提升 PAMF。见 [report.md](E:/type10-7/automation_reports/CV-SincNet/20261026-analysis-PAMF-FastTrust-source-r01/report.md)。 |
| GuardVAT、RXTrust-U／LU 恢复关系原型、NoSIG-RGPL | 设计／数值原型；未发布正式训练 | 无正式目标分数；恢复关系和 NoSIG 的旧方案已被当前完整 PAMF／SIG1／无 DURA 范围覆盖。 |

### 已退出的恢复视图源留出路线：全部 30 个已训练候选

下表只排序源留出 RX 的 G 后三景增益，**不是目标测试性能排名**。所有候选均只有一个学生 seed，全部未过预登记 +0.3 pp 源门槛，未追加正式目标测试。所谓留出 RX 只从新增分支排除，原 L／PL／父模型见过它，不能称真正未见接收机验证。分数与方法名完整复用历史复盘，原恢复器未重训。

| 方法 | 家族 run | 源 held-G 三景差（pp） | 源 clean 差（pp） |
| --- | --- | ---: | ---: |
| GTaskScaled | [report.md](E:/type10-7/automation_reports/CV-SincNet/20261026-phase1-PAMF-TaskScaledRef-sourceHold-m3-r01/report.md) | +0.222222 | +0.011111 |
| GTask2x | [report.md](E:/type10-7/automation_reports/CV-SincNet/20261026-phase1-PAMF-TaskRefDose-sourceHold-m4-r01/report.md) | +0.209877 | +0.003704 |
| GGCLMulti | [report.md](E:/type10-7/automation_reports/CV-SincNet/20261026-phase1-PAMF-URealizations-sourceHold-m4-r01/report.md) | +0.209877 | -0.011111 |
| BridgeGG | [report.md](E:/type10-7/automation_reports/CV-SincNet/20261026-phase1-PAMF-RestorationBridge-sourceHold-m4-r01/report.md) | +0.203704 | -0.003704 |
| HGCLMulti | [report.md](E:/type10-7/automation_reports/CV-SincNet/20261026-phase1-PAMF-URealizations-sourceHold-m4-r01/report.md) | +0.203704 | +0.003704 |
| GJoint | [report.md](E:/type10-7/automation_reports/CV-SincNet/20261026-phase1-PAMF-AsymmetricHG-sourceHold-m4-r01/report.md) | +0.179012 | -0.025926 |
| BridgeHG | [report.md](E:/type10-7/automation_reports/CV-SincNet/20261026-phase1-PAMF-RestorationBridge-sourceHold-m4-r01/report.md) | +0.179012 | +0.000000 |
| GTask3x | [report.md](E:/type10-7/automation_reports/CV-SincNet/20261026-phase1-PAMF-TaskRefDose-sourceHold-m4-r01/report.md) | +0.172840 | +0.000000 |
| GJointMulti | [report.md](E:/type10-7/automation_reports/CV-SincNet/20261026-phase1-PAMF-URealizations-sourceHold-m4-r01/report.md) | +0.172840 | -0.029630 |
| BothJoint | [report.md](E:/type10-7/automation_reports/CV-SincNet/20261026-phase1-PAMF-AsymmetricHG-sourceHold-m4-r01/report.md) | +0.166667 | -0.022222 |
| BridgeRG | [report.md](E:/type10-7/automation_reports/CV-SincNet/20261026-phase1-PAMF-RestorationBridge-sourceHold-m4-r01/report.md) | +0.166667 | +0.000000 |
| HJoint | [report.md](E:/type10-7/automation_reports/CV-SincNet/20261026-phase1-PAMF-AsymmetricHG-sourceHold-m4-r01/report.md) | +0.160494 | -0.007407 |
| GTaskMulti | [report.md](E:/type10-7/automation_reports/CV-SincNet/20261026-phase1-PAMF-URealizations-sourceHold-m4-r01/report.md) | +0.148148 | -0.018519 |
| GFeatureHard | [report.md](E:/type10-7/automation_reports/CV-SincNet/20261026-phase1-PAMF-FeatureTask-sourceHold-m4-r01/report.md) | +0.141975 | -0.014815 |
| HAnchor | [report.md](E:/type10-7/automation_reports/CV-SincNet/20261026-phase1-PAMF-AsymmetricHG-sourceHold-m4-r01/report.md) | +0.135802 | -0.011111 |
| MixBase | [report.md](E:/type10-7/automation_reports/CV-SincNet/20261026-phase1-PAMF-ResidualMix-sourceHold-m4-r01/report.md) | +0.129630 | -0.018519 |
| PairBase | [report.md](E:/type10-7/automation_reports/CV-SincNet/20261026-phase1-PAMF-ResidualMix-sourceHold-m4-r01/report.md) | +0.129630 | -0.003704 |
| RawTaskScaled | [report.md](E:/type10-7/automation_reports/CV-SincNet/20261026-phase1-PAMF-TaskScaledRef-sourceHold-m3-r01/report.md) | +0.123457 | +0.000000 |
| GJointHard | [report.md](E:/type10-7/automation_reports/CV-SincNet/20261026-phase1-PAMF-FeatureTask-sourceHold-m4-r01/report.md) | +0.123457 | -0.018519 |
| RestoredReference | [report.md](E:/type10-7/automation_reports/CV-SincNet/20261026-phase1-PAMF-RXReference-sourceHold-m3-r01/report.md) | +0.117284 | +0.014815 |
| SeparateRG | [report.md](E:/type10-7/automation_reports/CV-SincNet/20261026-phase1-PAMF-RestorationBridge-sourceHold-m4-r01/report.md) | +0.117284 | +0.000000 |
| HGFeatureHard | [report.md](E:/type10-7/automation_reports/CV-SincNet/20261026-phase1-PAMF-FeatureTask-sourceHold-m4-r01/report.md) | +0.111111 | -0.014815 |
| GUniform3x | [report.md](E:/type10-7/automation_reports/CV-SincNet/20261026-phase1-PAMF-TaskRefDose-sourceHold-m4-r01/report.md) | +0.104938 | +0.018519 |
| PairDose | [report.md](E:/type10-7/automation_reports/CV-SincNet/20261026-phase1-PAMF-ResidualMix-sourceHold-m4-r01/report.md) | +0.104938 | -0.022222 |
| PairedReference | [report.md](E:/type10-7/automation_reports/CV-SincNet/20261026-phase1-PAMF-RXReference-sourceHold-m3-r01/report.md) | +0.098765 | +0.011111 |
| GUniform | [report.md](E:/type10-7/automation_reports/CV-SincNet/20261026-phase1-PAMF-TaskScaledRef-sourceHold-m3-r01/report.md) | +0.092593 | +0.029630 |
| HFeatureHard | [report.md](E:/type10-7/automation_reports/CV-SincNet/20261026-phase1-PAMF-FeatureTask-sourceHold-m4-r01/report.md) | +0.086420 | -0.007407 |
| MixDose | [report.md](E:/type10-7/automation_reports/CV-SincNet/20261026-phase1-PAMF-ResidualMix-sourceHold-m4-r01/report.md) | +0.080247 | -0.014815 |
| RawReference | [report.md](E:/type10-7/automation_reports/CV-SincNet/20261026-phase1-PAMF-RXReference-sourceHold-m3-r01/report.md) | +0.067901 | +0.022222 |
| RawTask3x | [report.md](E:/type10-7/automation_reports/CV-SincNet/20261026-phase1-PAMF-TaskRefDose-sourceHold-m4-r01/report.md) | +0.049383 | +0.007407 |

这 30 行覆盖固定跨 RX 参照、任务缩放与剂量、默认头／特征 CE、IQ 残差插值、跨视图／跨 RX 关系、非对称 H/G 联合、四信道实现。最新四实现 GTaskMulti／GGCLMulti／HGCLMulti／GJointMulti 相对对应 fixed 控制，仅多 +1／+1／+4／−1 条净正确（16200 次源留出判断），没有证据支持仅增加恢复视图数量便能解决迁移问题。

## 五、尚未完成及失败记录的处理

RXTrust-AL 三个 seed 仍健康训练，已通过各自 E20 的真实 L/U 梯度校准与 checkpoint 检查；它在 A 目标分数读取前固定，新增 L 真实标签 CE，原 U 目标保持，尚无正式结果。本报告不将运行快照当最终结果。

原 VIEWANCHOR r01、物理软匹配首个技术发布、MixStyle r01、Covariance 确认 r01/r02、RXMean r01 等为被替代或技术失败的 run，保留原记录；成功替代 run 的模型在正式表中只计一次。协方差初始单 seed 确认已被三 seed 表复用，不重复算新实验。缓存、掩码校准、源置信诊断和队列控制不作为新增方法。

ChannelPL 原 11 设置中已完成 F0／F1／F2／ManyRx 四设置，各三个 seed；其他 21 个未启动学生保持原状态。本报告主排名固定 F0；其他设置见 [interim_four_contract_endpoints.csv](E:/type10-7/automation_reports/CV-SincNet/20261025-phase1-PAMF-ChannelPL-m33-r01/interim_four_contract_endpoints.csv)。原 REPL 的 11 设置和外部网络属于泛化范围验证，不拿不同数据契约分数与 F0 混排。

## 数据文件与复算

- [current_SIG1_three_seed_raw_ranking.csv](E:/type10-7/analysis_reports/PAMF_SSL_METHOD_RANKING_20261027/current_SIG1_three_seed_raw_ranking.csv)：当前单视图三 seed 全排名。
- [historical_three_seed_raw_ranking.csv](E:/type10-7/analysis_reports/PAMF_SSL_METHOD_RANKING_20261027/historical_three_seed_raw_ranking.csv)：历史 42 行，含实际可得的 seed 计数／均值／SD／PL 差值和历史 G 端点。
- [structural_and_loss_ablations_F0_raw_ranking.csv](E:/type10-7/analysis_reports/PAMF_SSL_METHOD_RANKING_20261027/structural_and_loss_ablations_F0_raw_ranking.csv)：完整 37 行结构／损失消融。
- [ranking_data.json](E:/type10-7/analysis_reports/PAMF_SSL_METHOD_RANKING_20261027/ranking_data.json)：未舍入机器可读汇总。

核算：当前表按方法去重，并逐项核对重复对照数值；历史原 summary 的 35 行由三个 seed 每景整数正确数重算均值与样本 SD，其余 7 行复用原同口径三 seed 端点 CSV；重算均值与历史复盘端点相差均 <1e-9。结构表与历史 REPL／PL／监督／去外推四个重复参照数值相符。未对未知整体 SD 作推算，未把源 98% 分数当作目标成绩。
