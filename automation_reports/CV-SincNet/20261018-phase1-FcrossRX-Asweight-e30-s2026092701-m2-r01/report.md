# 完整 practical 的监督份额：固定 E30 因子对照

状态：PLANNED，实现中；尚未启动，无新性能结果。Run 20261018-phase1-FcrossRX-Asweight-e30-s2026092701-m2-r01。固定 seed2026092701、split392005、全部源RX[1,3,4,6,8]／day[1,2,3]，L6300／U56700（TX隐藏）／V27000，IQ256，无RX留出或Ldev。F／三路SIG_TIME／统一Sinc／joint160／默认cos头联合训练，NoExplicitCFO与内部补偿保持。

唯一假设：当前完整As配方只给现有As64平均0.25的L监督份额；在不增加物理损伤、样本、视图或前向的条件下，将mixed槽内0.125重复ordinary份额转给已有As，可能改善practical识别。它是监督组织消融，不是新IQ物理增强、Ag／DG有效性证明。历史cleanretain37的4L配方源urban弱于其50%As监督对照，但输入／前向／阶段也不同，不能证明权重因果；本次才固定其余因素独立检验。该历史“50%As”不指E50预算。已关闭stepALT不重做。另一对话正做普通PA旁路，本次普通PA／CFO／相位／噪声均保持。

|L组成|原配方|新配方|
|---|---:|---:|
|ordinary128 RDrop对|0.5|0.5|
|mixed ordinary64 RDrop对|0.25|0.125|
|mixed selected64 RDrop对|0.25|0.375|
|raw128 RDrop对|0.25|0.25|
|合计|1.25|1.25|

As行selected64为原真实As(raw64)；实际NoAs.apply(raw[mask])返回raw.clone，所以NoAs行selected64为raw64。原配置文案与执行不一致，本次保留真实旧路径，因此新增同权重NoAs控制。CE和原对称KL同时按组mean归一化，两次RDrop使用同一已生成IQ。原 .1 ORD128 CROSS_RX仍只用实际ordinary128 maps，primary .5和raw .25稳定；全部有标签As受监督，没有置信度筛选。As行有效ordinary/raw合计1.0→0.875，As0.25→0.375，此监督转移明确披露，不能称clean总份额完全不变。NoAs行原ordinary0.75／raw0.5，新ordinary0.625／raw0.625；所比较的As因子是selected的raw→As(raw)，不能称NoAs mixed等同ordinary128。

两条新训练行：B00／B01分别从P43原对应NoAs20／As20 student与ownEMA严格恢复，追加10轮。原run 20261018-phase1-FcrossRX-impulse-factorial-e30-s2026092701-m6-r01，固定release提交19aedc3604fed9c193237dd92366164178af0983；原NoAs30／As30冻结产物复用。四逻辑条件NOAS30_ORIGINAL／AS30_ORIGINAL／WEIGHT_NOAS30／WEIGHT_AS30，不能把两个新row当两个模型种子。

每条own20／4440更新＋新10／2220＝累计30／6660；共新增20 model-epochs／4440更新。每步始终6L、网络batch128、768LNN，每新行1704960LNN；As行64As/step合142080，NoAs0As。原选择mask、每个生成IQ、网络调用顺序、RNG、三训练场景均匀1/3、物理质量／补偿／归一化、旧U规则均不变。普通增强沿原epoch13冻结，SupCon／channel累计e21..30、step4441..6660。resetAdamW／lr2e-4／WD1e-4／cosine10／min1e-6，与原尾段匹配。实际teacher／eval／读出成本分别记录，原prefix及旧控制不重计新增。

**本轮源推进条件在训练前固定**：fullV27000 newAs clean−originalAs30 ≥−0.1绝对pp，且V1800两个固定draw的mid／low等权均值newAs−originalAs30 ≥＋0.5pp。它只决定后续是否有依据研究总E50，不阻止本E30训练和正式收尾。该规则衡量新增权重的增量与轻量clean保持；加权NoAs源变化、As净收益及交互完整报告，不增加其他gate。规则不追溯改变旧配方的关闭决定。若任一失败，关闭此固定0.375配方，不扫描权重／seed／epoch救跑；若通过，也不声称已完成最终目标。

正式阶段仍为WEIGHT_AS30−WEIGHT_NOAS30 clean≥−0.1绝对pp，并另报相对原NoAs30。新机制增量分别比较newAs−originalAs、newNoAs−originalNoAs；交互为(newAs−originalAs)−(newNoAs−originalNoAs)。全部clean＋六practical准确率／macroF1／最差RX／RXdayTX、urban均值、suburban-clean差距、修复与新错、种子数和成本报告。源fullV只有原生指标时不补造全V逐包F1／错误对；Vpanel两draw不是两个模型种子。

固定FINAL30后每row自动source→七视图固定预测→独立truth-last评分，科学负结果仍完整收尾。完整四条件分析先保存source判定，再在全部4×7预测／独立评分齐备后连接truth。正式received IQ／SNR／信道／全部168000包每视图保持，不按目标成绩选权重、晋级、seed或选择性重跑。已暴露历史基准背景如实保留。

发布命令：既有N607普通账户／项目Python，在唯一PROJECT/releases/cvs_FcrossRX_AsWeight_E30_20261018_r01下运行 experiments/cvs_practical_as_weight30/runner.py。唯一输出根PROJECT/runs/20261018-phase1-FcrossRX-Asweight-e30-s2026092701-m2-r01，owner codex/root/practical-Asweight。GPU动态共享锁、最多3个全部owner进程／free≥12000MB，CPU2；不改健康旧任务。只因本run非法输入／父来源／头、非有限值、成本契约、输出碰撞或系统技术故障停止，保留产物、不自动重试。

启动器首步：两真实父CP无query各2NN，加两行实际6L各768NN，共1540NN、0更新；验证defaulthead／Sinc／backbone联合梯度与原ORD128公式，通过即训练。聚焦检查覆盖权重CE＋KL、0.25/0.25还原原公式及gradient、As0vs64／head／role／source／预算负例、实际hook telemetry。一次限定P0/P1、代码配置Git提交、一次ordinary账户preflight／归档SHA比较／远端compile／PID-CWD-argv-GPU-loggrowth依既有最小流程执行，不新增签名或审批层。

预期产物：实际配置／模块／channel／严格双方parent与EMA／全4440step＋20epoch／分组CE-KL和1.25总尺度／物理序列与普通MSE及As曝光／各自固定FINAL30／全部源及4×7正式与独立score／实际成本分项／配对交互报告／registry与Git。当前code.commit为规划父提交，发布前固定实际实现提交。

发布前调用路径纠正：NoAs selected实际raw64已在报告、spec及分析语义修正，模型仍保留既有NoAs.apply(raw)和原RNG／IQ。不存在对旧健康任务、原NoAs代码或历史结果的热修改；本轮仍是只改组权重的独立因子。

本地实现已冻结并验证：66协议／标签／头负例、分组CE/KL标量及gradient mock、.25/.25还原原公式、实际hook21指标与schema、源结果原生JSON类型均PASS；19py及两段远端AST PASS。本候选唯一独立P0/P1审查PASS（P0=0，P1=0）。四条件只读分析8项synthetic与21字段接口检查PASS，不执行模型、不读正式分数；真实双方CP＋6L smoke仍由launcher首步实际执行，通过立即训练。
