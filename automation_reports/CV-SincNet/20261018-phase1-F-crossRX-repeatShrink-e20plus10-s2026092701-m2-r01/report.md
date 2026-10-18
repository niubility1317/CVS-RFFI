# 完整源 E30：重复残差 IQ 增强的独立价值

状态：两行已完成训练、七视图预测和独立评分；源判据失败关闭，正式数值尚待本轮独立复算。唯一配置与逐行矩阵在 [experiment.json](experiment.json)。

源前置检查 20261018-diagnostic-F-repeatShrink-sourceV-s2026092701-m1-r01 已通过全部36必要条件，848混淆矩阵独立复算误差0。它只证明固定轻度算子在两个原默认头下通过源保护，不能证明已提取域无关身份。本轮只检验 Ag 的独立训练价值，避免先把尚无收益证据的Ag叠进As。本对话与“开发 CVS Phase1 星地增强训练”的As扩种子方向不同，不合并不同实验增益或父权重。

|行|.5权重primary|.25权重原始锚点|.25权重新槽|有标签前向/包|
|---|---|---|---|---:|
|B00|同次ordinary的RDrop对|raw的RDrop对|raw的RDrop对|6|
|B10|同次ordinary的RDrop对|raw的RDrop对|Ag(raw)的RDrop对|6|

各RDrop对内部复用同一个已生成IQ，alpha1/LS.01不变；按物理包128平均，三项权重 .5+.25+.25=1，原primary SupCon及U目标另按原规则相加。两行都有原ordinary .5和raw .25稳定监督，不稀释为不断加视图后重新求平均。B00是额外前向/随机子网络方差匹配控制，不能与旧4前向基线冒称相同成本；旧普通增强实际E13参数/RNG/顺序在两行保持，完整记录，旧PA/CFO开关不视为新标签保持证明。

Ag仅作用新有标签槽的现有256点raw：固定[80,160)、四个20点周期，自己的均值P、每周期相同sin²相位权重D且两端0，x'=x−lambda D(I−P)x。每包lambda独立Uniform[0,.1]，私有seed20261018104及累计epoch/step生成；B00消耗同一随机数但施加强度0。无TX/RX/质量/置信依赖，无额外RMS、CFO估计、PA/量化/剪裁/新噪声/外部donor/长上下文。周期残差可能含TX证据，不能称纯RX噪声；数字改变已接收IQ不等于换了真实RX。As训练和组合视图均关闭，正式测试仍用原完整practical。

两行严格加载同一个原fullsource REPEAT source-final20学生及其ownEMA。该父从匹配完整契约scratch训练，按固定E20冻结，未用目标排名；不使用已失败的E30子模型。所有继承、原source_frozen/actualargs/physicalrole权限在加载前复用既有证明并核对。重置AdamW/weight_decay.0001/clip5/物理SincCENTER_UNITS与cos10（peak.0002/min.000001），各新增10轮×222=2220更新，累计E30/6660，固定final10；不是scratchE30。共享父成本E20/4440单列，新增2×10=20model-epoch/4440更新，唯一累计计算40model-epoch，不把每行30相加隐藏共享父或额外视图成本。

F＋跨RX同TX、三SIG_TIME、统一Sinc、joint160、默认6×160 cosine/scale30与骨干共同训练；NoExplicitCFO与原内部补偿保持。不新增参数、部署分支、头、拟合诊断器、VJP或额外教师。原U56700真TX始终隐藏、oldgate/ownEMA/原半监督目标和前向保持，不向U施加本Ag。原L6300/V27000、RX[1,3,4,6,8]、day[1,2,3]、6TX、split392005及所有物理角色不变，无held训练RX或L5580内部拆分；源V仅读出。

唯一源判据：复用每行原生末轮完整V27000默认头读出（0额外NN），B10−B00整体clean至少+.5pp，且每个RX至少−.25pp。任一不满足即关闭本固定窗/周期/taper/Uniform0至.1/父/seed/E30，不用target救幅度或换seed。源决定保存于source_matrix_decision.json后，主分析才读正式数值；两行训练冻结后仍各自自动预测clean+六固定practical，全部7视图/每视图168000决策完成后独立CPU truth-last。源失败不停止健康旧任务或跳过已登记正式测试。

正式分析报告Aclean(B10)−Aclean(B00)、六practical绝对准确率/macroF1、各RX/day/TX、修复/新增错以及suburban对自身clean差；≤2pp仅探索目标，不能靠降低clean缩差。当前没有B01/B11，也没有交互项或跨seed结论。有效Ag的必要证据必须包括真实跨RX clean收益，源健康检查不能代替。

实际有标签预算每行2220×128×6=1704960 NN，比旧4前向增加568320；两行完全匹配。原U/teacher/validation及正式1176000预测/行另计，CPU私有uniform、数值变换、CPU/GPU字节桥和日志时间记录，不能称免费。sourceguard先前14404NN/0更新另列。初始dispatcher一次真父CP/2合成NN smoke，PASS立即继续；每行原训练自身初始检查/原native验证也计入真实前向计数。

本地20项来源/预算/role/源gain反例/truth-last负例PASS，0真实IQ/CP/NN/更新/target；远端同次发布检查将执行真实hook合成6前向、同IQ成对、固定raw锚点、正确loss尺度和两行同lambda抽样验证。只对新增训练集成做一次限定P0/P1；已通过源guard的DSP不重复全量审查，不增加设计审批、签名、额外数据校验或报告门槛。

普通N607，唯一owner codex/root/F-repeat-shrink-E30；CWD PROJECT/releases/cvs_F_repeat_shrink_E30_20261018_r01，命令CVS-RFFI/bin/python -u experiments/cvs_repeat_shrink30/runner.py，本地入口F:/App/miniconda3/python.exe -X utf8 -m experiments.cvs_repeat_shrink30.publish。每GPU最多3个全owner实验进程且至少12GB空闲，CPU2；不热改/停止/重启健康旧任务。唯一不可覆盖根PROJECT/runs/20261018-phase1-F-crossRX-repeatShrink-e20plus10-s2026092701-m2-r01，Git提交为代码配置身份，一次普通预检/一个archive单次SHA传输比对/一次compile，启动后一次精确PID/CWD/argv/GPU/日志增长读回。代码执行commit写入release_commit与actualresolved。

技术停止仅实际角色/父来源/非有限/输出冲突/无法执行或产生合法完整预测等系统失败，保存所有原产物，无自动重试；准确率低是科学结果。预期实际args/module路径/角色/父和EMA证明、全部2220步与10轮、L/U物理序列、新槽lambda/数值/时间、source frozen/defaultCP、完整源判据、7view prediction、独立score、终态/同row分析/登记和Git交付；私有IQ/物理ID/日志/权重/predictions不入Git。

## 实际启动

执行commit a0461e0de33484561ab28db440c046510b02a0e6已提交、push并独立核对远端OID。普通账户预检、单archive传输比对、远端compile与20项协议负例通过；真实hook合成检验通过6前向/同IQ对/raw锚点/1.0损失尺度/同lambda抽样，1536合成前向包单位及2次合成反向单列，0真实数据/CP。dispatcher真父1CP/2合成NN noquery检查PASS后直接训练。

两行实际worker已独立核对PID/CWD/argv/GPU，日志自首次读回均增长。最近核实均完成新增第1轮/累计第21轮、222新更新/4662累计更新；每行L前向170496，恰为222×128×6。B00新增Ag0包，B10为28416包。无failure；健康旧任务未干预。精确运行证据见[启动核实](launch_evidence.json)。当前不能报告Ag训练收益、完成的正式指标或联合成功；继续至固定E30后自动七视图与独立评分，不延长预算。

## 冻结源判定

完整源V两行均26532/27000（98.266667%），B10−B00为0pp，未达固定+.5pp；各RX保护通过。因此关闭固定[80,160)/period20/taper/Uniform0至.1/父seed/E30，不换强度、窗口、seed或父权重救跑，不因稍后target值重排。全部source2220步/10轮、L6300/U56700、857实际args、680字段nativeCSV/JSONL、284160私有lambda及L/U序列/NN成本完全复算；只有candidate_id/output_dir不同。见[source_decision.md](source_decision.md)和[source_analysis.json](source_analysis.json)。本结论在本轮正式target数值读取前保存，不以源保护或普通增强健康检查替代泛化成功。
