# 原完整 practical 的 L 批次组成：三行 E30 检验

状态：IMPLEMENTED／LOCAL_CHECKS_PASS／P0P1_PASS。真实训练、正式预测和评分尚未启动。这是训练组织研究，新增 IQ A_g=false，不能替代 A_g×A_s 四因子或完整联合目标。

## 依据与唯一假设

只回放原完整源 CONCAT/s2026092701 的4440步采样日志及原source契约：568320次合法L曝光、6300个不同物理L，5源RX1/3/4/6/8与day1/2/3全保留。未打开IQ、checkpoint、U真值、target或目标评分。按冻结数据层的TX/RX/day/eq/sig规范和源索引顺序连接可见L，所有记录字段及opaqueL成员一致；不解析U TX。初次部分物理键因6个合法L候选而拒绝，补全实际冻结索引后完成，旧失败输出保留，无模型或训练成本。

默认跨RX对比项先用当前batch均值/标准差（detach）标准化描述，再逐包L2和对比，因此批次组成会影响对比计算。原启用3330步的所有128锚点均有效，完整metadata4440步也无无效锚点；不能把问题归因于缺正例。

|metadata回放|TX×RX计数平方偏差均值|每批空单元均值|最少正例数的跨批均值|
|---|---:|---:|---:|
|原顺序|4.029354|0.364640|10.164189|
|随机重组|4.054264|0.376802|10.188514|
|TX×RX重组|0.772237|0|13.993018|

偏差相对每单元128/30计算，不是梯度方差或识别指标。740个6×128池内，每个物理出现次数精确保持（含原重复），每单元在6输出批计数差≤1。[完整回放](metadata_replay.json)、[分析程序](../../../analysis/cvs_batch_repartition_metadata_20261018.py)。103个合成池及真实metadata回放通过，IQ/CP/NN/更新均0。

唯一待检验假设：减少L批次TX×RX组成波动，能改善原practical与跨RX对比的联合训练，在同一个默认模型中保护clean并改善low/mid urban。元数据变化尚未证明识别收益。不同于已关闭的整批六步排序、ordinary按RX分组、传播分组和dropout耦合；这些旧路线不重开。

## 固定矩阵和预算

3行 ORIGINAL、RANDOM、TX_RX_BALANCED，全部严格加载同一个原完整源 ordinaryAs scratch-final20 学生＋自身EMA，来自 20261017-phase1-F-crossRX-fullsource-concat-e20-s2026092701-m3-r02/CONCAT-s2026092701；固定预算冻结、非target排名、无其他祖先。原L6300/U56700隐藏TX/V27000、split392005、全部5源RX不留出。每行继续10轮，累计E30/6660更新，重置AdamW/cosine10，SupCon沿E21..30绝对权重.1；旧ordinary为父实际E13日程。这不是scratch30，不与关联对话的R25/教师U路线合并。

三行同样缓冲6×128，分别原顺序／随机重组／TX×RX均匀分配；每轮222步=37完整池，10轮370池，无末池丢弃。Lloader seed20261018131、Uloader20261018132、repartition20261018121+poolindex。共同隔离CPU loader RNG，保证同输入池不受模型/增强RNG消耗影响；U仅隔离loader RNG，保留原shuffle/邻近批次、weak/strong、旧gate、ownEMA及loss，不按U标签分组。

F三SIG_TIME、统一Sinc CENTER_25K/CENTER_UNITS、joint160、默认cos6×160/scale30、NoExplicitCFO及必要内部补偿、骨干与头正常联合训练不变。每L原.5RD(ordinary)+.5RD(完整practical(raw))，每对共享同一生成IQ；原SupCon仅ordinary两次，所有合法有标签As监督，无urban置信筛除。0新IQ操作/PA/DAC/量化/剪裁/损失/teacher/部署参数/额外NN。批重组会改变旧batch级增强门和信道namespace对应的物理包，不冒称逐包IQ完全配对。原传播/噪声/质量/补偿/15session硬件依赖保持，一包仅一次完整As。

父E20/4440共享一次；新增3×10=30 model-epoch/6660更新；各模型累计30，逻辑累计90与独立训练成本50（父20＋新30）分列。每row预计物理L/ordinary/As各284160，L学生1136640NN，含原U学生训练1703640，教师eval567000；source eval/smoke、正式预测、CPU/内存/池重组和wall实测单列。额外NN=0不等于组织操作零成本。父既有训练NN3661082，不计本轮新训练。

## 固定反证与自动收尾

全部3行固定final10/cumulative30，V仅health，不挑checkpoint或row。每row完成即自动clean＋六practical的168000×7预测，全部固定后独立CPU truth-last评分；不等整批结束，不改变received IQ/SNR/信道/困难样本。

TX_RX_BALANCED分别相对ORIGINAL和RANDOM：clean不降、low urban至少+.5百分点、mid urban不降、每suburban最多−.25百分点。任一失败关闭固定6批/30单元重组，不换池大小/seed/父/层位/系数/预算救跑。所有预登记终态均报告；target不用于扩种子、晋级、重排或选择性重跑。最终clean净收益、low>60%、各suburban距自身clean探索≤2百分点仍照实报告，不能降低clean缩差；这3行不能证明新增A_g或完整联合目标。

唯一owner codex/root/F-batchStratify-E30；普通N607账户，CPU2，共享容量每GPU最多3个全owner进程且≥12GB空闲，不停/热改旧任务。不可覆盖根 PROJECT/runs/20261018-phase1-F-crossRX-batchStratify-e20plus10-s2026092701-m3-r01；CWD PROJECT/releases/cvs_F_batchStratify_E30_20261018_r01；命令 CVS-RFFI/bin/python -u experiments/cvs_batch_stratify30/runner.py。[精确登记](experiment.json)、[配置](../../../configs/cvs_batch_stratify_e30.json)。

17聚焦负例、6个完整epoch边界/1332合成输出批/222池/对齐及RNG异常恢复PASS，0真实IQ/CP/NN；一次独立P0/P1 PASS。首步真实CP无query smoke后继续；一个Git代码配置commit、一个archive/一次传输比较/远端compile、一次PID/CWD/argv/GPU/log增长核对。仅输入/父/角色/非有限/输出冲突/不能合法预测等系统技术失败停止pending并保留产物，不自动重试，不影响健康旧任务。预期完整step/epoch/reference/Sincunit/LU序列、370池原metadata+permutation、默认头CP、7视图prediction/score、RX/TX/day配对、成本曲线及同row分析/Git交付；原始数据/权重/日志/预测保持私有。
