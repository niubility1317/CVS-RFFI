# Phase3 固定特征增量与拒识读出的组合

2026-10-29。候选 `20261029-phase3-source-embedding-readout-s2026092701-r01`。上一222步原生增量中，OE行MSP源未知拒识59.9683%，监督对照47.8095%，两行旧类保护均通过但未知95%未达。本候选检验原生特征增量与此前有效的跨分支读出是否互补，不继续更新编码器或改变原MSP候选结论。

固定真实r02的supervised_only与uniform_oe两套E51 checkpoint，分别保留其原训练config／父契约和历史11100＋222步。两套均加载自己的权重，冻结完整六类F和原身份头；本轮新encoder/head更新0。编码器训练中的源OE21祖先仍如实记录，读出阶段则匹配原consensus的四设备OE4200，使用额外源成员监督，不给辅助TX身份。原L6300／V27000／novel3150角色不变。

每个编码器先自有checkpoint两条合法L IQ smoke，继而重新导出四role的8原始分支及完整6类logits，共40652新F前向。不能把旧编码器的features或几何冒充新编码器结果。导出本身全程eval/无梯度，novel只读表示不进入拟合；fit只读取新L与OE4200，重新由L形成源几何，再构造同`head_aligned_branch_summary_v1`100维描述符。linear101和MLP1633各固定200次CPU AdamW全批量更新，seed2026092701、lr.001／wd.0001、loss两role各.5，normalizer仍为原L6300＋OE4200普通均值／总体标准差。它们的15域天然均衡，不采用21设备版域配重。

新冻结状态标记encoder_state_identity、descriptor schema与`source_unweighted_membership_v1`。默认新F六类head仍决定接受样本身份，新增二分类器仅决定拒识。predict只由Vknown标签／分数以原4%宏FRR／1.5pp自正确拒绝损失冻结每方法一个全局最大阈值，总体及各RX均保护；此后才读取novel特征。旧原F V／novel缓存仅取ID和logits作baseline，不充当新表示，30150×7预测全部固定后独立scorer连接未知TX。

scorer独立核对已知baseline为原F6同IQ argmax，要求新encoder闭集宏损伤≤.5pp、总开放宏损伤≤2pp、FRR≤5%、未知宏拒识≥95%，整体及每个源RX均满足。只允许两种encoder上的MLP16作为本独立候选晋级；linear和五静态分数全部诊断。完整两行后，若两MLP均过，优先supervised_only以减少编码器OE训练成本；仅一行过就选该行。此规则事前固定，不追改上一r02只OE::MSP可推进的NO_PROMOTION。

原E50＋consensus的85.7460%结果只作为已完成参考，直接复用原产物，不重训或再次评分。所有比较是同source开发记录的描述性／配对比较；source novel3已多次开发评估，known是同批经验风险，不称独立新确认。即使source通过，也未证明未见RX／practical三景、无标签适应、发现和匿名持续注册；正式RX24 U/Q不读。

两行可并行exportGPU→fitCPU→predictCPU→scoreCPU，每行至多一条新增GPU进程，全任务每卡至多2，不停止任何其他用户进程。一行技术失败保留另一健康行，整体PARTIAL不晋级；科学未达仍完整评分归档。新root／实际code-config提交、一次本候选P0/P1、一次N607 preflight／归档比较／编译和真实PID/CWD/argv/GPU/log证据留在同名最小登记。当前实现中，尚无组合实测性能。
