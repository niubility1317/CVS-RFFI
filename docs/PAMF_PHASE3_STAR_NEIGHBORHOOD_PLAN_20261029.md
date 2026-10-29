# G10 全无标签池邻域适应与匿名注册

Run：`20261029-phase3-star-neighborhood-s2026092701-r01`。本轮是新的联合适应/发现/注册可证伪实验，唯一主状态为 `DA1_REG1`，固定比较 `DA0_REG0/DA1_REG0/DA0_REG1/DA1_REG1`。不将源端 class-risk 的失败拒识器单独晋级为已成功的目标拒识器；先前源端未知拒识59.65%保持负结果。当前候选与晋级规则在本轮任何目标模型访问前固定。

原流程只聚类被逐条拒识的 U，容易使少量候选无法达到注册数量。本轮改为全 U 构造邻域，在没有 TX/known-unknown 角色、真实 K 或配额的情况下，用冻结源证据给邻域估计新颖性。它检验无标签上下文是否有助于拒识及未来匿名识别，不以建簇样本回测替代未来 Q，也不保证聚类数等于设备数。

## 数据和合法状态

原 F0 固定为三视图 Tri-SIG-TIME SSL 的原始 Main split E50/11100/seed2026092701，232993 参数；默认六类头不变。恢复器固定 DURA v1.0 R69 E20/4940，266951 参数，FP32/TF32off、reuse/graph开启、native关闭。所有新模型输入为已固定 G10 星地观测；P0 是 received，P1 是同 received 经 DURA 的结果，身份输出采用 P1。

源先验复用 paired1000 的 source L/OE 二分类读出（3233参数，P1/P0各100维描述）和 class-risk 的六个预测旧类条件阈值，保留其全部源端监督祖先：原 Phase1 L6300/U56700/V27000，读出 L6300/OE6134 的成员关系监督，阈值 V27000 的已知标签。不能把本轮零目标标签写成整个系统零标签。本轮不重训读出或阈值，只重新消费已有 G10 cache 的源 L6300，每记录共享 P0/P1；按已知旧 TX、五源 RX、三日期形成90组 embedding/margin均值。仅聚合均值进入部署，无逐记录源 bank。源V、OE、source novel和原U_s均不新增读取。加载祖先与契约按已完成 parent marker/manifest 检查，源42组保护必须已通过。

目标复用 `20261029-phase3-star-g10-pool-rx24-1-r01` 的 RX24-1：U384为2021-03-08，独立future Q864为2021-03-15。source RX为1-19/18-2/19-2/2-19/3-19，与24-1不交。U的六旧类288条、到达19-15/19-16各48条，Q六旧类576条、到达两类共192条、后续7-1共96条；这些配额仅builder/scorer和本报告可见，预测配置不传入。原物理记录1248条，不把两路径或重复实验计为新增物理样本。G10 MIX01六场景面积概率、receiver2027、scene/channel seeds2026103001/2026104001保持；不是六景等量压力集。既有开发RX/设备/物理记录已暴露，不能称全新盲测。正式RX24-18不访问。

## 固定算法

1. P1八分支生成既有五支等权标准化/白化单位 embedding，共688维。对全部U作Euclidean average-linkage，K候选2至floor(sqrt(N))，只保留每簇至少8条的划分。用全U平均silhouette最大者，精确并列取较小K；无合格划分则返回无簇。silhouette是无需真值的内部簇紧密度/分离度指标，定义见[scikit-learn官方文档](https://scikit-learn.org/stable/modules/clustering.html#silhouette-coefficient)；它不证明设备纯度，也不等价于未知类数估计正确。本轮不实现或声称FINCH原算法。
2. 每簇冻结embedding、P1六logits和原始knownness margin的均值，半径为逐样本leave-one-out到其余成员均值距离的.95分位，floor1e-6。opaque ID排序确保输入行重排不改变结果。
3. DA0保持逐条冻结先验。DA1只在Q距最近U簇中心不超过该簇半径时，将自身logits/margin与簇均值各半融合；范围外回退原先验。它是无梯度邻域输出适应，默认身份头权重及结构不变。Q既不决定簇数，也不更新均值或半径。
4. 每簇至少16个独立ID，完整mean margin<0且按opaque ID交错两半各自mean margin<0时，才能成为匿名类。无需每条样本先被拒识，匿名数自由包含0。每匿名类用90个source L聚合anchor确定独立保护惩罚：.05+max(0,max_anchor(1-distance/radius-tanh(anchor_margin)))。
5. 六旧类margin=(logits−maxlogit)/30+tanh(rawmargin)，每匿名margin=1−distance/radius−penalty。旧类和全部匿名类共同竞争，最大margin<0才拒识；并列优先旧类。REG0不计算匿名列。该跨类型margin为预登记项目启发式，不宣称概率校准或覆盖保证。

所有U状态写完并固定后才打开Q IQ。各batch逐行推理，不使用Q均值/近邻/BN/类别数/标签/分数分位，不回收历史query。完整四态预测固定后由独立CPU scorer首次连接U/Q truth；只用U做匿名ID的一对一Hungarian置换映射，映射冻结后评Q。未发现到达设备保留未来识别分母，未匹配匿名接受算错误而非拒识。

## 验收和执行

主报逐TX宏平均A_K、R_U及H=2A_KR_U/(A_K+R_U)，并列C_K、原F0同P1闭集准确率、closed−open百分点差。REG0未知为全部非旧设备；REG1主R_U为预留后续未注册设备，A_N另报；旧新harmonic与H_OW明确另名。要求C_K≥90%、R_U≥90%、原F0闭集损失≤2pp，适应须同Q旧A_K正收益；未来匿名A_N≥80%、较同DA注册前旧类损失≤2pp、全部计划注册类C≥90%、后续R≥90%。总体与六主景分别检查；缺必要TX覆盖为N/A，不判通过。逐TX/RX/day、簇污染/合并/拆分、配对四态差异和缺项如实报告。没有新的H硬门槛。

A_N≥80%沿用总体硬门槛，各场景A_N分别报告，不新增每景80%门槛。每景全部计划注册类接收率仍需要旧类与到达类覆盖；没有必需到达类时该项N/A，不能认证。

本轮是固定单seed单开发RX小矩阵：source anchors→U/futureQ四态→独立truth-last全部执行，不因科学失败选择性取消或重跑。指标未全过不扩展种子或正式RX，不降低最终目标。旧source unknown90科学门槛属于旧单独拒识候选的晋级规则，不冒充本轮联合方法已通过；本轮合法进入开发falsification依据是源先验、方法和晋级规则预冻结及known42保护，不是formal性能确认。

唯一launch owner为当前聊天root，N607单GPUworker串行；每卡最多新增2条、本轮同时仅1条，actualfree≥4096MB，外部任务不停止/修改/重启。最小预登记补入actual code commit、环境/CWD、精确命令、路径和artifact。一次源真实F0无query星地smoke、目标真实F0/G synthetic无query smoke后立即运行，聚焦协议负测、一次独立P0/P1审查，一份release archive一次SHA比较与compile；启动后独立PID/CWD/cmdline/GPU/log增长核验。越界、祖先不匹配、碰撞、非有限或无法合法预测属于技术停止，保留partial不原地重启；低分、0匿名簇和缺景是科学结果。完整目标保持ACTIVE。
