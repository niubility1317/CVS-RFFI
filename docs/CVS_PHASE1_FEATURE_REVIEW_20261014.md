# Phase1 特征机制复盘与当前推进

新目标限定 Phase1：增强实际 joint160 特征，原头三预定义 seed clean均值84%与可迁移特征证据分别验收。可以用冻结骨干、丢弃类别W后仅L_s拟合同形式ordinarycosine作为源特征诊断，成本单列；主训练仍默认cos头正常联合学习。Phase2适配/新类注册/未知拒识/support调参不在本轮范围。

## 当前事实与配方边界

两分支已只读核对HEAD和远端：本分支1f8814c046a8c68eba8a334e66a0aceb76e491e0，另分支48325acd1f7781411cba214269fdf5353a2d6e9e，均一致。四个既有run远端completion为EXPERIMENT_COMPLETE，queue无active/pending/failure，原worker全部自然退出。没有重新启动、合并或覆盖另一owner工作。

本对话唯一后续对照是32label＋18pseudo的LEGACY50三seed固定final50；来源为原frequency-context run及其独立evaluation recovery。旧多尺度和归一化恢复路线为33＋17，保留各自历史baseline，不与32＋18合并。RC换头为10轮AdamW.001/wd.0001/seed+43019；本对话EPI/CF为20轮Adam.01/wd0/seed+60013，均scale30/160维/L_s，但不同预算和算法，不能直接合并。

所有数据继续ManySig原L6300/U56700/V27000，训练池标注率10%（全source池7%），split392005/原RX/day/TX/物理ID不变；U真值隐藏，V不更新模型。NoExplicitCFO及内部逐包补偿保留。无新增星地训练，当前clean不声称部署。目标结果只作历史复盘，不用于新结构/超参数/父权重/停止选择。

|实验|完整新训练|原头均值与匹配对照|当前判断|
|---|---|---|---|
|多尺度33＋17|12scratch50/133200updates；3seed|OLD80.5268%，TIME81.6905%，PA79.0032%，JOINT81.2895%|TIME＋1.1637pp仅2/3正；普通换头与同容量单尺度控制缺失，不能归因为多尺度|
|临时任务32＋18|6scratch50/66600updates；基线3复用|LEGACY80.6454%，SAME80.6204%，CROSS80.7494%|CROSS普通cos＋.2478pp仅1/3正；原型＋1.2863pp为3/3正，不证明通用迁移|
|恢复/竞争33＋17|12scratch50/133200updates；基线3复用|BASE80.5268%，ORD80.3075%，RESTORE80.2532%，COMP80.5663%，组合80.1298%|完整15行已结束；COMP仅＋.0395pp，恢复未胜过同容量ORD，缺因果证据|
|跨频关系32＋18|6scratch50/66600updates；基线3复用|LEGACY80.6454%，POWER76.8591%，COMPLEX79.3058%|COMPLEX胜POWER＋2.4466pp但低于baseline−1.3397pp，不宣称有效|
|当前PA三阶32＋18|6scratch50已启动；基线3已评分|新模型无完整结果|已发布方案固定，新增576参数，不在运行中改配方或追加读出|

完整台账、逐seed、配对、成本、原始报告指针与证据层级见 [机器可读台账](research_data/CVS_PHASE1_FEATURE_REVIEW_20261014.json)。MS完整日志/原预测审计是另一owner已完成证据，本轮只读复盘及独立终态读回，没有声称重新审计全部raw日志。EPI/CF为本owner既有全日志/原预测审计。RC已重计完整默认CM/epoch；历史REFIT完整汇总尚未回收到本主报告。必要关联负结果保留，不将部分源几何写成完整三seed机制分析。

## 源证据与待解决问题

源几何已有实测7/15行：三seedBASE＋首seedORD/RESTORE/COMP/组合，每行L_s6300、实际unit-normalized joint160、模型更新0/targetreadfalse。首seedBASE有效秩7.648，ORD7.802，RESTORE7.573，COMP7.499，组合7.559；BASE三seed7.648/7.831/8.185。有效秩不能证明塌缩、RX方差不能直接当作捷径；当前不默认新增编码率、白化或强对齐损失。

多尺度PA下降或组合不优于TIME不能凭target均值归因于RX捷径/梯度冲突。临时任务使原型方向更一致，但ordinarycos不稳定；原头使用L/U、重训头只L且不同预算，不能把差距全解释为旧头过拟合。RESTORE与ORD必须同容量比较；COMP的L-only遮蔽与teacher/U/full inference路径按实际代码核对，不改已结束方案。跨频已完成同容量比较，尚缺合法源跨RX增量稳定性。

下一项实际源诊断：只读原初始与final50 checkpoint，比较32＋18 LEGACY3、33＋17 OLD3/TIME3的真实Sinc cutoff参数、有效频带与实际滤波核，共9模型。先核对源角色、scratch祖先和固定终轮，使用各自initial_smoke.pt；不加载目标数据，不训练，不做人工梯度替代实训证据。诊断代码experiments/cvs_feature_diagnostics/sinc_change.py，算术一致/已知变化/4negative检查PASS；真实测量仍待执行，暂不宣称频率冻结或覆盖缺陷。

## 三条可执行方向

1. 时序归因：先补OLD/TIME的同配方源冻结ordinarycos，再在相同source配方下做同容量单尺度残差控制，分清残差、容量与尺度。修改t1/2/3，保留实际joint160和默认头；从零，不取目标最高权重当父模型。新训练≤50，尚不启动额外矩阵。
2. Sinc单因素：以真实9模型参数/核运动与coverage测量作为依据。只有合法源证据支持更新尺度问题，才比较原参数坐标和归一化频率坐标，保证相同初始滤波器、不增加分支/头；若已正常改变，不以静态代码数量级认定瓶颈。
3. PA独立交互：回收新目标到达前已发布的576参数CONV/CROSS固定矩阵，完成数值更新→joint160改变→默认预测，再做冻结源端撤除新增残差与简单ordinarycos诊断。旧EnvelopeGate保留，不能把成功或失败直接解释成PA硬件恢复。

本轮最多两个动作：完成源Sinc及既有REFIT/表征证据回收；继续已健康运行PA到训练/默认预测/独立评分闭环。所有未来机制以代码或合法源证据选择，不按现有目标分数扫描到84%。目标仍未达到；尚无真实新TX能力声明。
