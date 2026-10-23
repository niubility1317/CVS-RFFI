# PAMF 无标签类别条件接收机矩匹配：固定源域假设

状态：VERIFIED_RUNNING。93 项检查、一次 P0/P1 审查及真实 E30 source-only 梯度检查 PASS；三 seed 按原固定配方续训，原目标仍 ACTIVE。

源依据：原 E30 三 seed 只读 CPU 特征诊断中，source U／V 的浅层、中层同类 RX 质心差分别对应约 34% 至 42% 的类内散布；均有 30 个类别×源 RX 单元。固定 source V 使用合法真值，U 仅用原 EMA＋L-only 原型的伪类与连续置信权重，不读取 U TX 真值。具体规范、分母和局限见九模型 reference 的 source_receiver_geometry.md／e30_source_receiver_geometry.json；比例是描述性分解，不是纯 RX 因果效应或目标收益证据。

从原 RXPSM 同 seed E30 学生、自身 EMA／高斯统计／L-only 原型独立追加 20 轮。原 T1 六次 L 前向、默认分类头联合训练，以及原 U 半普通噪声／半 Practical、软 CE、原型最终特征损失、ownEMA0.996、高斯0.999、置信低限0.7、锐化0.5及各系数保持。只增加一次原 raw U 学生前向，在 time_down 和 t2.pool 采集时间均值／population 标准差描述子；临时 hooks 只覆盖这一次 raw U 前向，教师、L 和原 strong U 不改变。分类头仍参与原 L／U 联合损失，新增矩损失不拟合新头。

每层 h 按当前 raw U 全批逐维均值／population 标准差作标准化，二者均停止梯度，标准差下限1e-4；不作 L2 归一化。分组仅用原已停止梯度的软目标 argmax 和原始源 RX，现有置信权重>=0.5 才参与。每个伪类×RX单元计算置信加权 population 均值与标准差（variance加1e-6后开方），有效样本数 ESS=(sum w)^2/sum(w^2)>=4 才有效。只比较相同伪类、不同源 RX 的有效单元，损失为逐维平均 (mean_r-mean_s)^2+(std_r-std_s)^2，先平均同类内所有无序 RX 对，再平均至少一对的类别，最后平均两个固定层位；整项系数固定0.1，复用原 L CRIC 的量级，不搜索系数。层位没有有效对时该层返回可微零，仍计入两层平均并记录零覆盖；不重采样 U 或按目标挑批次。没有 EMA 矩库、额外教师、U真值、其他新机制或 G 训练来源。

这是条件一维边缘矩匹配，不做完整协方差对齐。相同伪类保护、已有置信权重与原监督分类损失用于限制身份损伤，但不能保证不抹去有用类内变化或实现目标增益。参考 [Deep CORAL](https://arxiv.org/abs/1607.01719) 的层激活统计对齐思想；本版只在合法源 RX 之间训练，使用伪类条件的均值／标准差，不是该论文源目标适配或完整复现，不使用目标无标签输入。

每 seed 累计 E50／11100、13L＋37LU；teacher U2097900，student U3231900，physical U1048950，新增 raw U1134000、CMR4440批，唯一样本仍56700。记录实际可用记录、每层有效单元／类／RX对、原学生／教师更新和额外前向成本；与两个既有50轮基线及同父 RXPSM 对照比较时不称同 FLOP。固定最终学生、source V仅诊断，无源 V 微调选模或科学早停。

GPU5／6／7 固定 seed2026092701／02／03。等待同 seed UMIX 的 raw＋DURA 全14项GPU预测结束、UGP已派发后，占空出的第二名额；每卡本任务最多两个 worker，含外部最多三个计算进程，派发 free>=10000MiB。单 CPU scorer 等 UGP 终态后接续。不可覆盖 root `PROJECT/runs/20261023-phase1-PAMF-U-CMR-m3-r01`；相同技术失败两次停止新派发、保留健康任务和全部产物；source smoke PASS立即正式续行，不按源或目标微小涨跌停止。

每行训练冻结后自动完成 raw PAMF 和用户固定 DURA v1.0 同一 F，各 clean＋六 Practical，每景168000；全14项决策固定后独立 truth-last。沿用同 readout 的 F20监督／F21普通PL三seed对照，逐seed／景／RX／TX／day、SD、配对纠错／损伤、曝光和实际成本完整报告。原 raw clean 至少+1pp、六景严格超过+1.5pp 对两个基线同时成立才满足 goal；DURA 结果并列参考。该假设在 UMIX／UGP 目标成绩读取前固定，不由后续分数修改参数、父来源、矩阵或选择性重跑，不声称新独立盲测。


93 项检查覆盖条件均值／标准差的解析值、原始权重ESS、置信排除／常量／缺对、按类平均而非大类加权、只传学生梯度、额外一次前向确为raw U、临时hooks正常／异常清理、隐藏 U 真值拒绝、原教师／曝光状态往返、无新增网络参数、固定父配置、GPU双owner顺序、独立CPU评分和raw＋同一DURA方法名，以及已有方案回归。一次独立P0/P1审查PASS，无发现。真实E30 no-query smoke将检查两层矩损失非零、真实梯度有限非零、extra raw U256和原L768／teacher权限；probe丢弃，正式3seed重新加载原E30。尚不把local检查当作真实训练或目标收益。


实际不可变执行 commit `a3371f1339f8687ee6593afb71b8e0a79574a9cd`，一次发布包 SHA 本地／远端比较、远端编译 PASS。独立读回 CPU 控制器 PID／CWD／完整命令及存活状态，三 seed 均 QUEUED，无新 GPU 子进程，source smoke 尚未开始；证据见 queue_verification.json。当前 UMIX 正在完成双七景预测，UGP 健康训练保留。UMIX 同 seed 全14项 GPU 预测结束且 UGP 已派发后，UCMR 才占第二名额；首个真实 E30 source-only smoke PASS 即续行。没有将排队或本地检查当成训练／目标完成。


## 已核实实际 raw U 矩约束训练

实际执行 commit `a3371f1339f8687ee6593afb71b8e0a79574a9cd`；一次发布包 SHA 本地／远端对照及远端编译 PASS。独立核实控制器和三个 worker 的 PID／CWD／完整命令、实际 GPU 与原生 step 日志增长。真实 E30 probe 保留 L768／原U256，新增raw U256，246条可用记录，两层各56个同伪类跨RX对；浅、中层真实梯度有限非零、临时hooks已清理、默认头梯度有限且教师无梯度，EMA仅一次。probe丢弃，正式三seed重新载原E30。已逐条核对全部刷盘的完整step前缀，U／extra raw曝光、固定0.1系数／两层平均、可用记录和各层pair累计计数准确，新增损失有限。gzip尚未关闭，只是前缀证据，不宣称完整训练或收益。详见source_activation.json。

UMIX三seed双七景GPU预测均完成后，CMR在GPU5／6／7各接续原槽位，与原UGP每卡各一条本任务worker，含其他owner总计算进程最多三个。没有停止或热改旧任务；后续最终E50学生自动完成raw／同一DURA各七景，CPU评分器等UGP终态后接续。
