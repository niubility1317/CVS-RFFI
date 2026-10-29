# 源端预测类别条件风险阈值

run `20261029-phase3-star-class-risk-s2026092701-r01`。本轮是完整Phase3目标内的新独立source-only校准研究，目标保持ACTIVE，沿用户最新v3双90%/原F0同输入同path闭集到开放损失2pp/宏A_K-R_U-H主报。上一paired1000主法的A_K90.3593%、R_U43.7143%、H58.9227%，宏AUROC94.2525%，global阈值受源RX6/low urban已知保护约束；同预算额外P0未提高未知运行点。新假设是按模型预测类别分配已知风险预算，而非继续增加网络或预算。

考虑过继续扩大读出、改变身份表征、改变源已知阈值分配。选择第三项，严格冻结parent `20261029-phase3-star-paired-evidence-s2026092701-r01`（code `bcc873cd9f9273133d74b5bc591c947b37ce16e1`）的paired1000原始分数与P1身份；不挑parent控制，不加载其它模型。global_control复算并须完全复现该parent原globalτ；唯一主方法predicted_class使用六个条件阈值。过去结果与判定不追改。

输入仅parent已完成source knownV27000/novel3150分数与公共预测metadata。原F0 E50/11100、L6300/U56700/V27000、OE6134训练祖先保持；本run无L/OE/raw/clean/IQ读取、F/G前向、神经/geometry/normalizer训练或新独立物理记录。全部父IQ均为G10 MIX01 P0/P1，身份P1、拒识原两路pair200源模型。source RX1/3/4/6/8和三源日，novel19-15/19-16/7-1各1050、同源RX/day且历史开发暴露；它们不是未见RX或新TX盲测。target/dev/formal U/Q全部0。

阶段首步为实际parent readouts.npz/json权重的无query checkpoint smoke：两条合成200D feature直接验证冻结读出、无IQ/F0/G加载，无optimizer/update，PASS立即继续。它明确是readout feature-space smoke，不冒称新真实星地/F0/G测试。parent模型/数据契约从原complete及聚合继承，复用已完成数据，不重验raw或加hash/receipt。

之后仅打开parent known预测，固定取paired1000列。对每个冻结head预测类别c∈0..5，把knownV raw score按完整ties分块，前缀数P_c(j)，该组总数n_c。共同尾率候选为所有P_c(j)/n_c的Fraction并集，含0和1；对q，在每组选择满足P_c(j)≤floor(q*n_c)的最大完整前缀，τ_c取下一未拒score。以42组（overall/5RX/6scene/30RXscene）的真实已知TX宏C_K≥90%、原F0P1 closed−open≤2pp精确Fraction约束选择最大q。模型predclass缺失、必需known类缺失或零拒绝本身不可行时，全接受fallback且科学false。global控制沿既有v3最大合法scalar算法，仍仅knownV，不用未知分数。

这是source已知阈值校准，V不拟合神经权重、geometry或score normalizer。不称conformal或分布外覆盖保证；同一V的经验保护不能替代独立target验证。部署仅保存6个scalar阈值，不保留CDF或逐行source bank。每样本使用自己的模型预测c取τ_c，转换knownness margin=s−τ_c；global margin=s−τ_global，两者接受margin≥0，完整ties。未知真实TX、角色、类数、source/target配额、query真值和真实场景都不参与部署阈值路由。

六阈值/global控制全部冻结并写calibration marker后，才第一次打开parent novel原始分数；转换成两列margin、固定P1身份与opaque ID并完成所有预测。只有独立scorer随后连接private source novel truth，计算真实TX宏A_K/C_K/R_U/H、原F0保护、宏AUROC/OSCR、42组和逐TX。class-specific平移会改变分数排序，曲线以部署margin计算，不以原raw AUROC冒充新曲线。

唯一predicted_class必须42known保护和unknown overall/5RX/6scene≥90%，unknown RXscene仅诊断；缺必需类N/A，不设H硬门槛，不让global救主。报告两方法配对接受/正确接收/未知拒对净数，量化预算分配是否改善而非只报最高值。source通过最多获得下一独立冻结小规模跨RX资格；本run无DA/发现/未来注册，A_N/H_OW为N/A。

默认N607 CPU-only、CUDA_VISIBLE_DEVICES为空，GPU新增进程0，不受其他用户GPU占用影响也不干预任何工作。单不可覆盖run/output、唯一owner，actual code/command/environment/CWD/paths/stoprule/artifacts在同run登记。非法继承/输入/非有限/输出碰撞/无法合法预测评分则保留partial，不盲重发；科学低分仍完成全部预测/独立评分、不调未知阈值或扩seed。一次独立下一实跑P0/P1审查、一份archive单次local/remote SHA与compile、独立启动/完成读回和Git提交push。REJECTED_EXTRA_GATE：无独立设计审批、重复数据验证、额外hash/receipt或报告审查门槛。
