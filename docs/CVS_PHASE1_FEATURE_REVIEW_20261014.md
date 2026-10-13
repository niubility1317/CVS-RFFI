# Phase1 特征机制复盘与当前推进

新目标限定 Phase1：增强实际 joint160 特征，原头三预定义 seed clean均值84%与可迁移特征证据分别验收。最新时序设计再次明确本轮只用默认分类头，不单独换头、拟合分类器或原型。历史换头结果仅作已完成记录保留。Phase2适配/新类注册/未知拒识/support调参不在本轮范围。

## 当前事实与配方边界

此前历史发布点，两分支已只读核对当时HEAD和远端：本分支1f8814c046a8c68eba8a334e66a0aceb76e491e0，另分支48325acd1f7781411cba214269fdf5353a2d6e9e，均一致。四个既有run远端completion为EXPERIMENT_COMPLETE，queue无active/pending/failure，原worker全部自然退出。没有重新启动、合并或覆盖另一owner工作。

已结束本对话 PA/CF/EPI 对照是32label＋18pseudo的LEGACY50三seed固定final50；来源为原frequency-context run及其独立evaluation recovery。旧多尺度和归一化恢复路线为33＋17，保留各自历史baseline，不与32＋18合并。RC换头为10轮AdamW.001/wd.0001/seed+43019；本对话EPI/CF为20轮Adam.01/wd0/seed+60013，均scale30/160维/L_s，但不同预算和算法，不能直接合并。

所有数据继续ManySig原L6300/U56700/V27000，训练池标注率10%（全source池7%），split392005/原RX/day/TX/物理ID不变；U真值隐藏，V不更新模型。NoExplicitCFO及内部逐包补偿保留。无新增星地训练，当前clean不声称部署。目标结果只作历史复盘，不用于新结构/超参数/父权重/停止选择。

|实验|完整新训练|原头均值与匹配对照|当前判断|
|---|---|---|---|
|多尺度33＋17|12scratch50/133200updates；3seed|OLD80.5268%，TIME81.6905%，PA79.0032%，JOINT81.2895%|TIME＋1.1637pp仅2/3正；同容量覆盖/非线性控制缺失，不能归因为多尺度|
|临时任务32＋18|6scratch50/66600updates；基线3复用|LEGACY80.6454%，SAME80.6204%，CROSS80.7494%|CROSS普通cos＋.2478pp仅1/3正；原型＋1.2863pp为3/3正，不证明通用迁移|
|恢复/竞争33＋17|12scratch50/133200updates；基线3复用|BASE80.5268%，ORD80.3075%，RESTORE80.2532%，COMP80.5663%，组合80.1298%|完整15行已结束；COMP仅＋.0395pp，恢复未胜过同容量ORD，缺因果证据|
|跨频关系32＋18|6scratch50/66600updates；基线3复用|LEGACY80.6454%，POWER76.8591%，COMPLEX79.3058%|COMPLEX胜POWER＋2.4466pp但低于baseline−1.3397pp，不宣称有效|
|PA三阶32＋18|6scratch50/66600updates；基线3复用；全9行完成|LEGACY80.6454%，CONV80.7738%，CROSS80.8133%|CROSS仅胜同容量CONV＋.0395pp/2seed正；不确认三阶机制收益，0独立head更新|

完整台账、逐seed、配对、成本、原始报告指针与证据层级见 [机器可读台账](research_data/CVS_PHASE1_FEATURE_REVIEW_20261014.json)。MS完整日志/原预测审计是另一owner已完成证据，本轮只读复盘及独立终态读回，没有声称重新审计全部raw日志。EPI/CF为本owner既有全日志/原预测审计。RC本轮已回收全部15行两读出，并独立重计540个已保存CM/accuracy/F1；没有重新预测、另训头或重新连接truth。必要关联负结果保留，不将部分源几何写成完整三seed机制分析。

## 源证据与待解决问题

源几何已有实测7/15行：三seedBASE＋首seedORD/RESTORE/COMP/组合，每行L_s6300、实际unit-normalized joint160、模型更新0/targetreadfalse。首seedBASE有效秩7.648，ORD7.802，RESTORE7.573，COMP7.499，组合7.559；BASE三seed7.648/7.831/8.185。有效秩不能证明塌缩、RX方差不能直接当作捷径；当前不默认新增编码率、白化或强对齐损失。

多尺度PA下降或组合不优于TIME不能凭target均值归因于RX捷径/梯度冲突。临时任务使原型方向更一致，但ordinarycos不稳定；原头使用L/U、重训头只L且不同预算，不能把差距全解释为旧头过拟合。RESTORE与ORD必须同容量比较；COMP的L-only遮蔽与teacher/U/full inference路径按实际代码核对，不改已结束方案。跨频已完成同容量比较，尚缺合法源跨RX增量稳定性。

本轮实际源诊断：只读原初始与final50 checkpoint，比较32＋18 LEGACY3、33＋17 OLD3/TIME3的真实Sinc cutoff参数、有效频带与实际滤波核，共9模型。先核对源角色、scratch祖先和固定终轮，使用各自initial_smoke.pt；不加载目标数据，不训练，不做人工梯度替代实训证据。诊断代码experiments/cvs_feature_diagnostics/sinc_change.py，诊断代码commit f9e962b375f52cd5e8e7f1975986a170e8f15f81。已完成9个真实initial/final50比较，每个模型23/24截止参数与24/24带宽参数逐位未变；唯一变化截止参数的最大变动为7.6294e−6至6.5422e−4Hz。实际滤波核relativeL2为0至2.4071e−8，23或24个kernel完全未变，频率覆盖约80Hz至12.499999MHz基本维持初值。全部源角色/祖先/终轮/actualresolved配方一致，诊断IQ/truth/optimizer更新均0。9份nativeargs的freeze_backbone=false、from_scratch=true；原优化器收集所有requires_grad参数。接下来进一步核对Sinc的实际optimizer/gradient/EMA路径；这一证据支持测试坐标尺度修复，不证明固定滤波器无用或是泛化失败的唯一原因。

## 当前三个方向与最多两个动作

1. TIME 归因优先：按最新用户设计，t1/2/3 四组同容量：（3,1）/（5,1）/（5,1）紧凑 vs 第三支（5,2）长覆盖，各配线性或每分支 SiLU。PA/frequency/reference/GN/外部 ReLU/pool/dropout/默认头及旧 SSL 保持。固定33＋17、50轮；历史原TIME只有完整执行/RNG路径匹配才复用，否则新训匹配对照并单列历史参考。
2. Sinc 单因素排队：实际近静止权重证据保留。后续可核对优化器/EMA及参数坐标，不根据目标分数扫描，不在本TIME矩阵混入。
3. PA 当前完成：所有负结果与小收益保留，本轮不追加组合、换头或新的PA训练矩阵。

当前最多两个动作：现有真实 TIME 权重的等效卷积核、偏移能量、每输入通道核秩及固定源L分支撤除诊断；随后实现上述 TIME 2×2。源诊断固定每个 TX/RX/day 单元按物理 ID 排序取前8个L包，共720/seed，使用原默认头与真实joint160，零参数/缓冲/分类头更新，不读取U或target。branch撤除是推理诊断，不冒充独立训练消融；核秩不等于特征秩，GN使整块不等于线性卷积。

## 完整历史普通cos诊断回收

RC历史10轮AdamW普通cos的三seed均值：BASE78.5657%，ORD78.3613%，RESTORE78.5956%，COMP78.5095%，组合78.1139%。RESTORE相对BASE仅＋0.0300pp（2/3正），相对同容量ORD＋0.2343pp（2/3正）；组合相对BASE−0.4518pp，仅1/3正，且三个seed都低于RESTORE。未建立原头和ordinarycos同时稳定改善的证据。完整逐seed/样本SD、14条配对、最差RX/day和合法源probe指标已在主台账回收，不把此10轮诊断与本对话20轮Adam直接合并。

## TIME 只读诊断已返回与下一固定矩阵

PA 完整结果与 TIME 诊断实现已上传 f38bf6233aaa0f75857e6bba2bf206bf735b853a，remote OID一致。真实TIME3seed源角色/祖先/固定final50严格重建后，各t1/t2/t3的preGN等效9-tap核最大rank4，doublefold误差≤3.34e−15；外±4偏移能量合计3.4%至4.8%。固定L720包/seed的9次branch撤除全部改变joint160和默认margin，影响随seed/layer变化；零θ/buffer/head更新，不把branch撤除视为独立训练消融。完整源统计见新run source_time_diagnostic.json与主台账。源L选样规则已在诊断执行前固定，无U/target IQ。

新TIME矩阵 20261014-phase1-time-factorial-oldgate-cos-e50-clean-manysig-m15-r01：四组same237697identity参数/初始state，compact/long×linear/branchSiLU，各scratch50、33＋17，OLD_GATE_COS50三seed外部基线复用。真实builder检查发现历史TIME主RNG与私有CPU初始化的新矩阵不同，LONG_LINEAR因此重新训练。旧PA/frequency/reference/outputnorm/defaulthead保持；独立headfit0。首seed4组源L末222stepCE≤.5后才放行其他2seed；全部训练行自动clean/truth-last，无target选择或热修改健康任务。额外源V33/34健康54000包与逐epoch层运动、U预测类别/RX/day接收统计成本单列。代码/config与最小report将一起绑定实际发布GitHEAD。Sinc坐标修复排队，不混入此轮。

TIME 已实际发布，执行代码351e35cd0d298a6832c0865628bba28ab5a14cbe。唯一P0/P1审查、archive传输、remotecompile、真实默认checkpoint smoke PASS；首seed4个PID/CWD/argv/GPU/resolvedcommit一致，epoch0→4且loggrowth VERIFIED。4份真实初始checkpoint的200个参数/buffer tensors逐项一致，单独headfit0。3历史baseline原默认决定复用已评分，8个后续模型等待预定sourceL门槛，没有新target收益声明。完整发布证据在同run launch_verification.json。旧健康任务未热改/重启。
