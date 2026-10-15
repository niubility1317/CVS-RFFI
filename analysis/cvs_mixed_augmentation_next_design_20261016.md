# 下一步：先补齐普通增强的匹配源 RX 留出证据

当前状态为设计，尚未新登记或启动模型。不把以下计划写成已完成实现或增益。

两轮小幅度响应与本轮源相位曲率均未支持新 Ag。旧 ordinary L 增强与 practical As 之间仍有一个实际对照缺口：R_LEGACY 保留普通增强且每样本2前向；mixedIQ 的 B00/B10/B01/B11 关闭 ordinary L 增强且6前向。因此不能把 R_LEGACY−B00 当作 ordinary 增强的因果收益，也不能依据该目标差额选择幅度或损失权重。此问题来自已审计的调用差异。

下一步唯一假设：**固定原 SIG_TIME＋RDrop 时，旧普通增强整体是否具有真实的源 RX 留出身份收益，并能成为与 As 独立生成的 clean 支路。** 这属于补齐必要基准，不宣称新物理方法。旧 PA/AWGN/多径等不因已有开关就获得身份保持证明；不新增或放大它们，不据目标表现挑子开关。

最小第一步固定 model seed2026092701、允许源 RX 中顺序第一项 RX1 作为留出；不依据最新目标或源分数换 RX。两个 scratch 总E20=13L+7U：N0关闭 ordinary L 增强，N1保留真实原 ordinary L 配方。两组每个有标签物理输入均生成一次IQ、两次随机子网络前向执行既有RDrop，默认头与全网正常联合更新；没有 As、新G、额外一致性、拟合探针或U增强。oldgate、cos、ownEMA及原U标签权限完全相同。

这是明确的源伪域留出契约，不能冒充fullL6300：只训练RX3/4/6/8的L5040和U45360，RX1的L1260/U11340不训练；源V仍只验证，其中RX1 V5400为未参与训练的首要终点，其他RX V21600为健康证据。所有物理ID沿用已合法的原角色，不把V或U真值转给训练。不得加载full-source或历史fold权重，两行从零。实际步数、native batch/optimizer和启用开关必须由可执行配置在发布前核对，不能只继承上述数量推算。

第一步规则预定为：N1相对N0，留出RX1 clean accuracy至少+1pp、macroF1不下降，已见RX V均值下降不超过.5pp，全部更新有限且TX×训练RX×day覆盖完整。任一失败则结束该 ordinary整体作为Ag的候选，不换heldRX救援、不按目标选择增强。单fold通过只允许固定其余4fold的源确认，不构成泛化证明；五fold均值至少+1pp、至少4fold正差且最差fold下降<=1pp，才考虑正式full-source组合。具体launcher/manifest将另行登记，尚无新run/模型/成绩。

若建立独立Ag证据，后续full-source组合才比较匹配预算的无Ag/有Ag × 无As/有As。四组共享相同初始化、前向、原始监督权重及U规则；clean支路只施加一次候选Ag，satellite支路从原IQ独立生成原practical residual，避免普通AWGN/多径再次叠在As之前。组合视图As(Ag(x))保留为独立后续消融，首轮不堆额外对齐损失。每个正式训练行仍scratch总E20，并自动冻结、原clean+6practical预测和独立truth-last。

原测试IQ、信道参数、CVS/SIG_TIME/defaultcos联合训练及low urban>60%目标不变。首轮留出实验明确source-only，故不访问正式target；源规则失败就不扩展组合或seed。已有目标结果仅作历史报告，不能输入新配置或晋级。

已完成证据：[混合IQ完整负结果](../automation_reports/CV-SincNet/20261015-phase1-sigtime-rdrop-mixedIQ-e20-manysig-s2026092701-m5-r01/report.md)、[源响应独立E20](../automation_reports/CV-SincNet/20261016-phase1-source-response-e20-manysig-s2026092701-m2-r01/report.md)、[源相位筛查](../automation_reports/CV-SincNet/20261016-diagnostic-source-phase-crossfit-manysig-m1-r01/report.md)。

## 已落实最小两行实现

20261016已实现并登记 `20261016-phase1-ordinary-augmentation-RX1-e20-s2026092701-m2-r01`，尚未发布。实际原native/fold轨迹保持222更新/epoch、4440/row，而不是178×20；四RX U45360通过原循环形成每pseudo epoch56624观察。两行当前RAW_HZ/LS.01、两次RDrop，同初态；旧CENTER25K/LS0源折只复用代码逻辑，不载权重。HeldRX1 V5400仅FINAL，未用L1260。单一源V按RX作评价分层，不生成校准/选模新角色。普通增强与loader私有随机流、实际L/U全序列比较、计数和默认头梯度已接入。一次限定P0/P1 PASS，本地实际图检查PASS；正式数据/测试信道未访问或修改。

20261016两行已实际发布并启动VERIFIED：训练代码96365a0aa，GPU0/1，own fresh CP noquery smokePASS，首个增长快照E2/E2。实际defaulthead/SIG_TIME与源角色核实，尚无最终heldV资格或target结果。后续严格按本规则处理，原release不热改。
