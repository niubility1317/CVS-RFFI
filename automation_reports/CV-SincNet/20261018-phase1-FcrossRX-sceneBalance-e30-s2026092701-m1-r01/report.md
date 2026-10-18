# Practical 批内三场景配额：E30 最小匹配实验

状态：PLANNED。20261018-phase1-FcrossRX-sceneBalance-e30-s2026092701-m1-r01 已登记；实现中，尚未发布或训练，没有新性能结果。唯一 owner codex/root/practical-sceneBalance。代码提交将在发布前写入 experiment.json；矩阵以该文件唯一维护。

当前最值得保留的是原完整 practical 与普通拼接监督。已关闭的 weighted、MIXED128、stepALT、每物理包六场 sceneCycle 等配方保持关闭。现有普通 As 本来逐包独立选择三场景，本项不把“批内多场景”称为新机制。历史 sceneCycle 是 held-RX1/L5040/8L 下同 PID 跨六次访问六场轮转；本项不改变物理包覆盖规则，不复活该失败路线。

唯一假设：保持完整 practical 的三场景长期各1/3和每step As64，减少每一步场景计数引起的损失混合权重波动，可能改善 low/mid urban 并控制 clean 代价。每批配额21/21/22，extra scene=(cumulative_step−1)%3，4441步先low_suburban22、随后midurban22和lowurban22；用独立命名的私有RNG将场景打乱分配到原64个位置。三step每场64、222step每epoch每场4736、尾10每场47360。只改场景分配，不能按TX、U真值、置信度、状态或uniquePID重采样；原sampler重复如存在保留。

场景确定后，原cfg、stable_seed(train_seed,namespace,scene,pid)、rx/day/eq会话硬件、归一化、ResidualChannel.process及传播—遮挡—噪声—接收质量—锁定—补偿依赖保持。原scene RNG与channel RNG已独立命名，因此新assignment不改变同pid/step/scene的物理实现。不得独立重抽锁定状态、丢困难样本、改测试SNR、信道或receivedIQ。不增加损失、门、分支、网络参数、As包数或前向；private assignment不推进原global NumPy/Torch/dropout/U流。

矩阵只有一个新行B01-s2026092701，严格从原P43 ownFINAL20 student＋ownEMA追加10轮，总E30／6660更新。直接原ORD128 SupCon、time_down/t2.pool、tau.2、lambda.1；.5RDordinary128＋.5RDmixed128＋.25RDraw128，总L尺度1.25、6L／768每step，mixed内ordinary64与As(raw64)各.25，RDrop两次复用同一已生成IQ。oldU／ownEMA／原有默认cosine头随骨干联合训练不变。ordinary始终原20args epoch13，SupCon+20，channel累计21至30／4441至6660；resetAdamW lr.0002／WD.0001／cos10／min1e−6，与原尾段匹配。

原NoAs30和As30两条完整冻结读出及已有审计复用，形成NOAS30、AS30、BAL_AS30三逻辑条件。新operator只作用于As、对NoAs恒等且不消费global随机流或额外NN，因此不为该控制重复训练。相同学生总epoch／初始化规则／数据／头／U／六L及累计native NN比较；不能把旧参照作为免费训练，父20及旧E30成本另列。直接父来源是原P43 run 20261018-phase1-FcrossRX-impulse-factorial-e30-s2026092701-m6-r01 B01/stage20；从零20／4440更新／ownEMA／source-only FINAL20来源已核实，当前同句柄在一次preflight核对，不能加载weighted或MIX候选。

保留F＋CROSS_RX、SIG_TIME三路／统一Sinc／真实joint160／默认cosine6×160 scale30、NoExplicitCFO和必要逐包内部补偿。全部源RX[1,3,4,6,8]／days[1,2,3]，L6300／U56700真TX隐藏／V27000，split392005、IQ256、Fs25MHz、dataset eq1不变；不留训练RX或Ldev。practical residual/post_sync、接收EQ=false、reference_snr30、reflect和物理依赖原样。训练及正式／source面板各自已固定的载频和观察保留，不把dataset eq1当接收均衡开关。

源规则仅两项，提前固定：BAL fullV clean−原NOAS30≥−.1个绝对百分点；V1800两个固定draw的mid/low等权均值，BAL−原AS30≥+.5pp。任一失败关闭本固定配方，不扩E50、扫描配额/相位/幅度或换种子救跑；不加逐cell或预测保持门槛。通过只表示未来E50资格，root在首次读取新formal前保存源决定并另作预算内推进记录，不自动启动。科学负结果仍完成all7正式。

每新row固定FINAL30后先源读出，再clean＋六practical全预测完成后独立truth-last评分。固定正式RX[0,2,5,7,9,10,11]／days[0,1,2,3]／168000每view，receivedIQ与困难包不变。三条件×七视图pred及独立score齐备后，0NN分析才连接truth，预计378CM／63分区合并。报告全部accuracy、macroF1、最差RX、urban3mean、RX/day/TX、修复/新增错误、suburban绝对值和自身clean差距、原As总体效果及BAL增量；不把原As已有收益归给配额。正式阶段BAL−匹配NoAs clean≥−.1pp，最终仍需可重复clean正收益；n1／不足.1pp小涨都不算稳定双提升。正式结果不得回流任何方案、参数、晋级、seed或重跑。

执行环境：普通N607既有CVS-RFFI/PyTorch fullFP32，CPU2，无admin或安装。CWD=PROJECT/releases/cvs_FcrossRX_sceneBalance_E30_20261018_r01；精确命令CVS-RFFI/python -u experiments/cvs_practical_scene_balance30/runner.py，worker追加--mode worker --row B01-s2026092701。不可变输出PROJECT/runs/20261018-phase1-FcrossRX-sceneBalance-e30-s2026092701-m1-r01。每GPU所有owner compute总进程≤3、free≥12000MB，动态共享capacity分配，不停止、重启或热改健康任务。

发布只用一个代码/配置提交、聚焦协议负例、一次独立P0/P1、最小预登记、一次普通preflight、单archive比较和remotecompile。launcher首步一个真实own20无query CP2NN＋一组6L768NN，共770NN，PASS直接训练；随后独立核对PID/CWD/argv/GPU/log增长。不新增审批、签名、member哈希或重复数据验证。

停止规则仅run-owned协议/父来源/默认头/数值/输出冲突或无法合法预测等系统技术失败，按准确PID/CWD/cmdline/runroot处理所属故障、保留产物，不自动补跑。source科学失败或正式不达标不停止健康训练和既定评价。

预期新增成本（尚非实际结果）：10model-epochs／2220更新，native3162962NN（student train2271960、student eval324002、teacher eval567000），source12600、formal1176002、共享smoke770，合4352334样本前向；L1704960、As142080、extra机制NN0。原父20 native4797722和两旧E30／读出另列，不重计新增；source wall若未独立记录须如实披露。完整reference_steps、shape_steps、14实际native指标、epoch/counter/曝光与scene counts将用于实际成本和执行一致性检查，不作为识别收益替代品。

另一对话《开发 CVS Phase1 星地增强训练》正在验证U教师来源，已读取其最新活动。本项保持U和teacher规则不变，只改L的场景计数组织，不重复或干预其任务。


本地代码已冻结：19py／47协议负例、全部2220步配额和3step/epoch/尾段总量、private RNG、重复PID、真实下游apply接口mock及2段发布AST PASS；analysis4AST／9synthetic检查PASS，0NN/CP/IQ/更新/remote。唯一独立P0/P1正在进行；真实770NN烟测仍待launcher首步，当前未发布或训练。

归因限制：保持的是nominal三场景边际1/3，不与原IID逐包、逐ID或完整尾段的实际场景次数一一匹配。增量只能先归于该固定分层分配，不能声称已经隔离纯梯度方差；不为此回扫旧stream或加新门槛。


唯一合并独立P0/P1完成：0／0，无阻塞缺陷；未重复旧helper审查或本地测试。当前LOCAL_VERIFIED_P0P1_PASS，真实770NN首步smoke、发布和PID/GPU验证仍待执行，不把代码就绪写成已训练。
