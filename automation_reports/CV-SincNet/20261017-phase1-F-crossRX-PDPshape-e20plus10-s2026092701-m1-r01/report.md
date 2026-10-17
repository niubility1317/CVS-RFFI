# RMS 匹配的三径时延形状：单项 A_s E30 检验

状态 **RUNNING／VERIFIED，尚无完整识别结果**。唯一矩阵及命令见 [experiment.json](experiment.json)。本轮属于有界的既有传播算子修改；采用已读 CVS 实验与登记流程。另行设计审批属于 REJECTED_EXTRA_GATE／NONBLOCKING，不新增发布层。

假设来自实际代码：原 ChannelStream 的三条 diffuse 路径始终使用归一化时延 [0,1,3]，仅系数与状态刷新。当前训练已有连续仰角和原始三场景采样，不把新增仰角或改变旧 session 键当新路线。旧 OJ24、RX session、共享传播、calibration/finite-quality 和时钟配方负结果保持关闭。

唯一待检验假设：保持各状态总集合 RMS 时延扩展与原径功率后，加入第二种时延形状能改善完整 practical 的身份适应，同时保留 clean。新形状 [0,1,2] 是等间隔三径工程对照，与原 [0,1,3] 等概率抽取；不是按目标分数或源分数拟合的分布。两者都用原 diffuse/平均 LOS 功率加权中心二阶矩，将时延缩放到原20/50/100 ns。它不保持每一瞬时传播核能量、均值时延或最大时延相同，这些实际差异会记录；不对 H 或输出做振幅重归一化以掩盖 SNR 变化。

[3GPP TR 38.901 v14.2.0，第7.7.3节](https://www.etsi.org/deliver/etsi_TR/138900_138999/138901/14.02.00_60/tr_138901v140200p.pdf)给出归一化时延按期望 RMS 缩放的定义。这只支持尺度区分，不证明本三径工程形状是标准 NTN 模型或该增强有效。未使用论文或其他实验的目标成绩预测收益。

## 唯一比较与固定契约

新行 PDP_SHAPE-s2026092701 从原完整源 CONCAT source-final20 的同一学生和 own EMA 严格加载，追加10轮／2220更新，累计 E30／6660更新；重置 AdamW/cos10（lr .0002、wd .0001），不能冒充 scratch E30 或精确续接优化轨迹。复用既有严格同父、同 E20＋10 配方的 NOMINAL_AS E30 全部冻结产物，不重复训练、预测或评分旧控制，不使用失败候选父模型。来源已由既有 source-final20 全源契约核对，加载前仍走实际父证明。

完整5源 RX1/3/4/6/8、day1/2/3、split392005、L6300/U56700隐藏TX/V27000保持；不留出训练 RX、不切内部 L_dev、不扩大 L。F＋跨RX同TX、三SIG_TIME、统一Sinc、joint160、原cos30默认头与骨干联合训练、NoExplicitCFO/必要内部补偿、原U/oldgate/ownEMA不变。

每步原 ordinary 的 RDrop 对和 full practical(raw) 的 RDrop 对，分别权重 .5/.5；原浅/中层 SupCon 仅作用 ordinary 对，累计E21至30权重.1。每个RDrop对复用同一已生成IQ。4L身份前向／128物理包，0新增损失/网络参数/部署分支/U视图/教师；不把RDrop或本 A_s 当成 A_g。原raw在各组的监督组织相同，不因新增视图稀释。

原训练场景仍为 low_suburban、mid_urban、low_urban 等概率，不改变它们的状态、遮挡、几何、shadow、LOS、ray系数/角度、noise、RX/session及校准的原分布。每包只生成一次完整 residual：fs25MHz/fc2.462GHz/post_sync/EQoff/G=1/reference30/reflect/共同数值延迟24。只在该次处理前换时延；H改变后质量、锁定、同步、tracking和AGC按原规则重算，保留依赖。全部合法有标签 As 参与CE，无高置信筛选、难包删除或连续两次完整信道。数字模拟不等于真实替换RX；未知原TX硬件不反演。

## 固定判定与预算

沿用此前已固定的同预算普通 As 参照判据：low urban 至少 +0.5 pp，clean、mid urban不降，每个suburban不低于−0.25 pp。另报告相对同预算B00的clean净收益、low>60%、三个suburban绝对值及距自身clean的差（2 pp探索目标）；降低clean不能缩小差距冒充成功。本次没有新Ag或完整四因子，不能从单行宣称联合完成。

任一识别条件失败，关闭本固定双形状／等比例配方；不改形状/概率/系数/seed/父/预算救跑，不自动扩种子，也不按目标成绩晋级。固定final10后立即执行全部7视图预测，prediction/logp完整后由独立CPU进程truth-last评分；原正式receivedIQ、信道、SNR及样本均不改变。

新增训练10modelE／2220更新／1136640 L NN，原U及教师按真实计数单列；As284160次，每次一个完整residual，额外完整信道预跑/物理滤波/NN为0，仅增加三状态时延代数。预测1176000 NN＋预测smoke2，发布真实CP smoke2；训练smoke/源V/教师/墙钟由已有实际计数记录。原父20轮/4440更新和旧参照30轮成本分列，旧控制缺失的瞬态eval计数不伪造。

## 已完成验证及执行边界

本地39个协议反例、108组同latent三路径（324次residual）、30个实际训练适配器调用通过；另外先前纯物理检查324次，因此当前合成residual累计678次。原形状波形/元数据逐值相同，三种状态全覆盖，RMS误差最大2.8422e−14 ns，原几何/径功率/RX/noiseprior不变，质量从变化H重新计算，无振幅重归一化。实际F/defaulthead/joint160接口2合成NN、900722参数通过；0真实IQ/CP/训练更新。已有本地Torch/NumPy警告保留，未使用其NumPy桥接或改环境。输入变化、RMS约束和健康均不等于识别收益。

一次独立 P0/P1 检查发现标准库 profile 命名遮蔽，已重命名 pdp_profile.py 并更新两个相对导入；直接脚本路径下 cProfile/空状态 AdamW 导入与编译通过，0 IQ/CP/NN/更新，唯一问题定点复核 PASS。第一次修后观察因既有 Torch/NumPy 警告超出输出缓冲而中断，保留；仅扩大输出缓冲后核实成功，没有改环境或科学方案。审查仅限会使下一真实运行错误、越权、不可启动或不能合法预测的缺陷。一次实际代码/config Git提交后，一份新不可变归档、一次普通N607 preflight、单archive传输比较及remote compile、真实同契约CP无query smoke通过后立即训练，独立核对PID/CWD/argv/GPU/log增长。每GPU计所有owner/preCUDA最多3进程；不使用管理员、不修改任何健康旧任务。

唯一owner codex/root/F-PDPshape-E30。CWD PROJECT/releases/cvs_F_PDPshape_E30_20261017_r01；命令 CVS-RFFI/bin/python -u experiments/cvs_pdp_shape30/runner.py；不可变输出 PROJECT/runs/20261017-phase1-F-crossRX-PDPshape-e20plus10-s2026092701-m1-r01。实际路径、GPU、日志、stop rule与预期产物见登记。仅系统技术失败保留产物并停新派发，不自动重复启动，不因低分重跑；代码commit由发布器填入release_commit.txt及远端登记。

以上为发布前登记；实际运行状态见下文。仍无新Ag、As或跨seed收益，完整目标 ACTIVE。


## 实际发布与当前进度

执行commit `476f350914ca72c753365c1b25d9deebf4920701` 已推送且远端OID独立匹配。一次普通N607预检、单archive传输比较、remotecompile、远端39反例及2合成F NN检查、真实匹配父CP2合成NN/noquery smoke全部通过。dispatcher PID513370、原worker PID513399的CWD/argv/CUDA/GPU与更新日志增长已独立核实；日志260805→326500字节，已读到149条完整实际更新记录。此为运行证据，不是epoch完成或准确率收益；新科学结论仍待固定final10与7view。

本地合成物理调用678＋远端354＝1032；本地/远端随机初始化检查共4合成NN，真实父发布smoke另2NN，训练初始smoke包含在实际源eval计数，后续训练/教师/预测单列。初次读取活跃gzip末尾不完整JSON时只影响本地观察，现只读取完整行，不修改或重启worker。当前无failure、无正式target反馈，健康原进程继续。

后续同一原PID只读核实：已完整追加E1（累计E21／4662更新），113664 L NN／28416完整As，实际全部5源RX、无留出、U真值和target访问均false；当前完整更新记录已到298，尚未完成追加10轮或任何正式七视图读出。stdout6785→11965、step日志持续增长，无failure。
