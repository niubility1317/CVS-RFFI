# ordinary／As 跨视图 dropout 相关性：固定 E30 比较

状态：PREREGISTERED／LOCAL_CHECKS_PASS／INDEPENDENT_P0P1_PASS，普通账户连接预检通过，尚未发布或训练。完整联合目标 ACTIVE，未证明新 A_g 或身份识别收益。

唯一假设：保留两对独立 RDrop 的边际目标和原 practical 输入，只改变 ordinary 对与 As 对之间的随机子网络相关性，可能改善联合优化后的真实识别。此项属于训练组织；[原 R-Drop 论文](https://proceedings.neurips.cc/paper_files/paper/2021/file/5a66b9200f29ac3fa0ae244cc2a51b39-Paper.pdf)只支持原两次随机子网络一致性的定义，不证明本项 coupling 有效。旧 AsPairCE、噪声 antithetic、RX 会话覆盖的负结果保持关闭，不改变其参数或判定。

## 固定比较

两行 INDEPENDENT／ANTITHETIC，model seed2026092701。都从原全源 ordinaryAs CONCAT-s2026092701 的 scratch fixed-final20 学生及自身 EMA 出发，独立重置同 AdamW lr0.0002／wd0.0001、cos10；追加10轮，每行2220新更新、累计E30／6660更新。不是 scratch30 或精确 optimizer resume，父成本 E20／4440 单列。父来源与全部源物理角色沿用既有合规核对，不继承失败变体或按目标排名换父。

F 三路 SIG_TIME、统一 Sinc CENTER_25K／CENTER_UNITS、joint160、默认 cosine W6×160 与骨干正常联合训练；不新增网络、部署分支、头拟合或损失。NoExplicitCFO 与必要内部补偿保留。全部5源 RX1/3/4/6/8、day1/2/3，L6300／U56700 隐藏 TX／V27000，split392005；没有训练 RX 留出，也不改为5580/720。用户确认无可靠长上下文映射，仅使用现有256点IQ。

原损失仍为 .5 ordinaryRD＋.5 AsRD＋.1共有ordinary浅中层跨RX同TX SupCon；每个L128物理样本共4次身份前向，原生CE correction不变。每对复用同一个已生成IQ；As从raw-before-ordinary走原 complete residual/post_sync/NoEQ，全部合法有标签 As 监督。传播、噪声、接收质量、补偿、采样先验和 formal received IQ 不改。U、ownEMA、旧门控和半监督规则保持，V仅健康监测，fixed-final10冻结。

按 O1、O2、S1、S2 顺序，仅操作四次共有的身份 encoder 中既有 nn.Dropout。INDEPENDENT 四掩码独立；ANTITHETIC 令S1/S2分别使用O1/O2离散随机数的反射值，O1/O2及S1/S2对内仍独立。p不调，有限均匀支持 M=2^24，keep threshold=round((1−p)M)、原1/(1−p)缩放；相对原生理想p的离散误差≤1/(2M)，两行相同。私有seed20261017561只依赖step/module/occurrence，不读IQ、TX、RX或day。两行都抽4份同大小随机张量并计算反射，匹配额外抽样与反射工作；As前向不重抽IQ。原生首次前向中身份网络之外的dropout不改；U／teacher／eval范围外使用原生 dropout。新增可训练参数、NN前向、教师、loss和U规则均0；掩码缓存／抽样成本按实际记录。

## 预定判定与停止

全部七视图 fixed received IQ：clean、practical_high、practical_mid、practical_low_suburban、practical_high_urban、practical_mid_urban、practical_low_urban。每视图168000相同物理样本；逐row全部 prediction/logp 固定后独立CPU truth-last评分，报告accuracy、macroF1、worstRX及全部RX/TX/day。最终默认头直接预测，不另训诊断头。

固定反证条件：ANTITHETIC相对新 INDEPENDENT control 的low urban至少+0.5pp、clean及mid urban不降、每个suburban不低于−0.25pp；任一不满足即关闭固定配方，不改mask key／p／父／seed／预算救跑，不以目标做晋级或自动扩种子。以上只作本次固定比较的报告判定，不证明完整目标。仍单列clean相对无新增增强同预算参照的净收益、low>60%、三个suburban绝对值与距自身clean的差，2pp为首轮探索标准，不能靠clean下降缩差；修复／新增错误及clean和low共同正确用于检查错误重分配。本次无新Ag／四因子或交互项证明。

技术停止：父/数据/权限/默认头/预算/掩码轨迹不符、非有限loss/grad、已有输出冲突或所属row技术失败，保留产物，不自动重试或改动健康旧任务。每GPU最多3个全owner实验进程，并遵守原共享容量选择。两行固定矩阵互不使用先完成row的目标成绩。

## 发布与证据

逐行完整矩阵、配置和存储以[experiment.json](experiment.json)为唯一登记入口；配置[固定矩阵](../../../configs/cvs_dropout_coupling_e30.json)。执行commit将由release_commit.txt记录；发布命令为CVS-RFFI/bin/python -u experiments/cvs_dropout_coupling30/runner.py，CWD为PROJECT/releases/cvs_F_dropoutCoupling_E30_20261017_r01，输出根PROJECT/runs/20261017-phase1-F-crossRX-dropoutCoupling-e20plus10-s2026092701-m2-r01。普通N607账户／固定现有Python和F依赖／全FP32／CPU2／共享GPU，owner codex/root/F-dropoutCoupling-E30。

本地26项权限/状态负例与分布/回退/deepcopy检查通过。两行实际F合成四视图、默认头联合梯度、无state_dict变化通过；每前向13个活动dropout、每步52随机张量／13565952随机元素。实现期3次合成探针失败保留在[验证说明](verification.json)，合计1920包模型前向、384次合成As处理、0真实IQ／CP／优化更新；这些不是性能证据。后续按项目最小流程进行唯一一次P0/P1审查、真实CP无query smoke、单archive传输比较、远端编译和PID/CWD/argv/GPU/loggrowth验证，随后自动训练与正式评分。
