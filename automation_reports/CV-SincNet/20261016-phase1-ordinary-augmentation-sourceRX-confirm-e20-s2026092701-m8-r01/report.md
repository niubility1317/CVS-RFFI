# 旧普通增强：其余四个源 RX 的固定确认

状态REGISTERED_NOT_RUN。RX1首轮已通过原源门槛：当前RAW_HZ/LS.01/defaulthead、scratchE20的N1−N0 accuracy+6.4074pp，macroF1+7.3596pp，seenV−.3241pp；TX0−11.8889pp/TX1−1.3333pp等代价保留。本轮只执行此前固定的另外RX3/4/6/8×N0/N1，8个新scratchE20，不用任何旧模型或其他fold权重。RX1旧预测/结果只读复用，不重训/重测。

唯一假设：普通增强整体的源RX1净收益能在其他合法源接收机留出中重复，满足五fold均值/最差fold规则，而非单RX偶然收益。固定同一modelseed2026092701，不宣称五fold等于五个seed。数据是已有ManySig合法源carrier392005，模型、默认cos/joint160、NoExplicitCFO与必要内部补偿、25MHz/IQ256/equalized1/nativeRMS均固定。

每行只训练该heldRX补集4RX、day1/2/3和6TX：L5040/U45360，原V按RX评价分层，seen21600与held5400。该heldRX的L1260/U11340不使用；所有train/probe/calibration/EMA构建前排除heldRX，U只按RX元数据过滤，TX隐藏。V不参与梯度或持久模型更新。各fold不共享模型/teacher/特征缓存，完整L6300历史模型不加载。

N0关闭ordinaryL，N1保留原完整ordinaryL系数及schedule。两行都是13L+7U、LS.01、RDrop1、同IQ两前向，AdamW lr.0002/wd.0001/cos与oldgate/ownEMA不改。私有augmentation stream20261016051、role loader20261016053及offset不改；每对最终逐条核对实际全部L/U物理样本序列。没有As、新G、新loss/branch/head/U机制，原测试IQ/信道不接触。

每行保留原222step/epoch=4440更新；四RX U loader178batch循环至222，因此每pseudo epoch56624重复观察。每行L1136640网络前向，U/EMA/sourceeval/held5400/smoke2按实际计数。新8×20=160model-epoch/35520更新，RX1旧pair40model-epoch/8880另外列账；总10个独立E20模型不是E200单模型，不加载E50/E200或续训权重。

全局源规则：将旧RX1和新4fold全部纳入，N1−N0 accuracy均值>=+1pp、至少4fold严格正差、最差fold>=−1pp；同时平均macroF1不下降，各fold seenV均值下降<=.5pp，预算/有限值/覆盖/样本顺序完整。前三项在RX1前的设计中已固定；F1均值和逐fold健康限制在本轮任何新观察前显式固定。单fold低分不触发停止、重排或删除；全局FAIL结束本route，不换heldRX、强度或seed救援。PASS仅提供源证据，仍需独立full-source Ag/As矩阵和正式truth-last，当前不自动target/joint/扩seed。

单owner/不可覆盖root，八卡都可用，shared receiver_residual_capacity4锁内按所有owner与preCUDA进程<=3/free>=12000MiB选择，CPU2/workers0。唯一本run技术异常保存产物、停止待启动行，健康已启动行继续；不热改、不自动重发、不干预其他任务。精确路径、CWD、命令、六类seed和矩阵见experiment.json。

共用已验证训练循环，只把heldRX/fitRX从单折常量变为每个配置的值。所有五个RX在五条train/probe/U/val/calibration路径的metadata-only fixture通过，0IQ/0U-TX读取。当前真实图1024个合成网络前向包含原RX1完整回归和新RX3两行；198个同初态state张量、默认W/SIG梯度、EMA绑定、全部8row nativeargs通过，14负例含最差fold/正fold数量/均值/F1/健康拒绝。真实scratchCP noquery smoke仍是每worker首个模型步骤，PASS立即继续。

原ordinary含PA等未经逐操作认证部分；首折输入峰值204.893/累计MSE.181549并不支持将整个配方称轻度有界，正负类别效果将完整报告。本轮保持原配方作确认，不新增或放大PA/TXIQ/量化/剪裁，也不把数字操作等同真实RX替换。预算耗尽不是收敛；当前没有新增source成绩或正式urban结果。

唯一独立P0/P1增量审查PASS，未发现直接缺陷；再执行1024合成前向/128增强包/14负例。本轮本地已知总2048合成前向/256增强包，0真实IQ/CP/更新。登记字段、common+new本地编译、内嵌remote发布编译通过；普通账户N607只读preflight通过。尚未发布。
