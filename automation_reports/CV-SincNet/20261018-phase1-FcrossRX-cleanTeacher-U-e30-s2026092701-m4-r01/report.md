# 完整 practical U 教学：固定同龄 clean 教师来源

状态：已登记并通过本地检查，尚未发布。以下为最小预登记；矩阵以experiment.json唯一维护。

当前保留完整 nominal practical、明确 clean监督份额和原默认头。P64温度与CE减半已完成负结果，固定路线关闭。另一对话已完成Asweight E50三种子，目前在GPU0/1健康运行mixed128对比学习，本轮不重复、不修改。归一化running-stat假设已由P43真实模型审计否定：10 GroupNorm＋4 LayerNorm，无BN/running buffers，未为此启动任何实验。

唯一新假设：在同龄固定教师之间，NoAs训练来源能否比As训练来源更好地监督同一U的完整practical学生。它可能避免教师与受损学生同步漂移，也可能因E20陈旧或高置信错误而失败，不预设其标签更准。来源都覆盖原全部源RX，没有leaveRX或新教师训练。

新增四行：ZERO_KL；EVOLVING_AS_EMA；FROZEN_AS20_EMA；FROZEN_NOAS20_EMA。四个学生同P43B01As20 student＋ownEMA独立追加10轮，重置AdamW/cos10/cache，学生累计E30/6660更新。原6L768/1.25L/ordinarySupCon.1/RDrop对称KL/nativeUownEMA规则保持，全部有标签As参与CE。

每Ustep额外取原native批次前min(128,n)，full native256/tail124完全保留。额外subset不看TX/q，不padding。四行都执行同一个完整AsU学生及两个冻结rawU教师前向，包括ZERO。每row新增AsU284120、冻结teacher各284120，共extra852360NN。只新增同一种KL：q=clip((teacher pmax-.95)/.05,0,1)，T1，subset batchmean，新增前5epoch ramp至.16；ZERO0。教师的内容与连续q覆盖共同变化，不能宣称同权重下标签更准。原U硬mask不改变，不要求urban学生先高置信；不沿用P39全局多proxy禁用门。U真值始终隐藏。

FROZEN_NOAS20−FROZEN_AS20检验固定同龄训练来源；FROZEN_AS20−EVOLVING检验固定；各KL−ZERO检验额外教学。原P39输入clean不代表无As训练来源，它用evolvingownAsEMA；其源mid/low两draw pooled +.986111pp仅说明方向值得研究，旧严格逐格C4FAIL保留，不能用旧4L对照冒充当前6L精确归因。

**预算口径：**每个学生30E；系统复用As20和NoAs20两项历史前提40E/8880更新，加new10为50 unique prerequisite model epochs/row，四行相同。没有新教师训练，教师原defaulthead未另训。历史来源费用不得当免费，也不在四行重复计算。原As30/NoAs30只匹配学生epoch/seed/data，extraNN与系统前提成本不同；新增机制归因以本四行匹配对比为准，不能把对历史NoAs全部增量归为教师收益。两frozen训练reference各900722参数完全eval且stopgrad，外部holder不进student/optimizer/nativeEMA/deploy。部署仍900722参数/206statekeys/cos30。

源判据：仅FC主候选；fullVclean−NoAs30≥−.1pp；Vpanel midlow−As30≥+.5pp、low−As30≥0；midlow−FA≥+.5pp、midlow−ZERO≥+.5pp、low−FA≥0。其他三行是控制不晋级。任一FAIL关闭FC固定路线，不扫描q/teacherage/系数，不自动E50/seed。formal结果不回流，所有已登记row无论sourceFAIL都完成clean＋六种practical。

数据L6300/U56700隐藏/V27000，RX1/3/4/6/8、day1/2/3、sixTX、split392005，无heldRX/Ldev。现有VALIDATED_ONCE源/固定正式IQ复用，不重验数据。25MHz/IQ256/RMS center=false，完整residual/post_sync、EQfalse⇒G1、质量适配/相位跟踪/IQ补偿开启、ref30dB、reflect；训练fc2.462GHz、正式2.45GHz保持。训练lowSub/midUrban/lowUrban各1/3，L/U流独立，RX/day硬件session稳定且不由TX定义。

同一普通N607环境Python3.10.19/Torch2.1.0+cu121/NumPy2.2.5，fullFP32/CENTER_UNITS/CPU2；CWD=PROJECT/releases/cvs_FcrossRX_clean_teacher_U_E30_20261018_r01；命令CVS-RFFI/python -u experiments/cvs_clean_teacher65/runner.py。代码 commit：8bfc5f53883933a6e069526dc05793340f805370。owner codex/root/clean-teacher65，独占PROJECT/runs/20261018-phase1-FcrossRX-cleanTeacher-U-e30-s2026092701-m4-r01。总GPU进程上限2计入所有owner，free≥12000MB；资源不足等待，不触碰其他任务。

停止规则仅本run协议/来源/数值/无法运行/输出冲突等技术失败，保留产物，不自动补跑。source科学FAIL不会停止评价。每row全部七预测完整后独立CPU scorer读truth；最后6logicalrows×7在truthjoin前验证。原As30/NoAs30不重新训练、预测或评分。

预计新成本（不是结果）：4×10=40模型E/8880更新；每rownative3162962＋extra852360=4015322NN；全部新remote=20815702NN（4row16061288＋source50400＋formal4704008＋一次realCPsmoke6）；本地actualF6656NN单列。历史2refs60E/13320更新＋新40=unique100E/22200更新，两个parent各计一次；wall在运行后实报。

已执行59协议负例、8教学toy、4流程检查、一次四arm真实F6656NN通过。synthetic随机未训练教师q全0，因此actualF额外KL梯度0符合公式；阳性q的非零教学梯度由toy证明，未补跑actualF。原L/nativeU默认头与Sinc梯度有限，原U256mask/extra128/私有RNG与外部teacher冻结边界通过。分析12toy执行证据见local_artifacts命令/stdout；唯一独立P0/P1发现新教师双来源未被旧分析元数据接受，已修复；新增1项元数据toy通过，并已完成一次定向复核PASS。原12项及actualF未重跑。尚未训练。

交付完整epoch/step曲线、各rowq/coverage/KL/teacher disagreement、source及formal acc/F1/worstRX/repair/newerrors、固定NoAs30clean正确集合保持率及margin/30、U无TX覆盖、actual native/extra/历史成本、5组真实PNG/PDF、全部负结果和缺项。U真值不开放，不把q/同意率当标签准确率；source状态/addednoise不是总SNR或物理因果。未达到lowUrban>60%或重复双提升时goal保持active。

最小流程：一个code/configcommit、必要负例和一次realCPnoquerysmoke、唯一P0/P1及直接问题一次定向复核、一个archive比较/compile/独立poststate、truthlast。额外设计审批/哈希链/重复review为REJECTED_EXTRA_GATE，不作为阻塞。

实际发布结果：code 8bfc5f53883933a6e069526dc05793340f805370，一个archive本地/remote比较PASS，23py compile PASS，真实2CP/6NN no-query smoke PASS。四行均在首次优化更新前因frozen teacher嵌套model builder递归退出，training_sequence均0，无finalCP/源判定/正式prediction/truth评分。五个本run PID已独立核实全部自然退出，未终止其他任务。状态FAILED技术失败，不能判断新方法性能。失败产物保留，不自动重跑；本地正在做最小修复。实际新训练0E/0更新/0训练NN，remote smoke6NN，local actualF6656NN。训练wall无已完成记录，不编造数值。

最新预算授权：用户允许对表现良好、源域证据支持的方案延长至总E50，使用匹配同seed对照；不是追加50轮，不按正式测试晋级。本run因技术初始化失败，没有可供E50晋级的科学证据。

递归修复已完成并通过1项0NN构造回归。当前研发任务的既有技术恢复流程允许诊断修复后使用新root完成同矩阵；准备r02一次技术替代。原noautomaticretry保留为禁止盲目原地/同指纹重启，不解释为新增用户审批。原r01继续FAILED且不改写其实际代码身份或结果。
