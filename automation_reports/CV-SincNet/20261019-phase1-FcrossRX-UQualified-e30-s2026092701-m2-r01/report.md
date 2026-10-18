# 已有教师资格的 U practical 路由：E30 最小实验

本轮唯一假设：把完整 practical 放在已有 raw ownEMA 教师取得合法身份监督资格的 U 槽位上，可以获得原完整 L As 以外的星地增量，并以较小 clean 代价保留收益。先前固定四分之一独立路由已源FAIL关闭，本轮不调该比例或门槛。新增的关键区别是身份资格与受损学生输入的对应关系；加入同数量规则随机控制，避免将减少增强量全部归因于资格关系。

## 固定矩阵与控制

- R01（URANDOM30）：先完成原整批 Gaussian .015，再私有均匀选择 k=本行当步原教师资格数的槽位，覆盖为完整 practical(rawU)。
- Q01（UQUALIFIED30）：原整批 Gaussian 后，仅把全部原教师资格槽覆盖为完整 practical(rawU)，未资格保持 Gaussian。
- N/C：复用 20261019-phase1-FcrossRX-UPractical-e30-s2026092701-m3-r02 的 N01 MATCHED_NOAS30／C01 UTEACHER30 完整冻结训练审计、源读出、七视图预测及独立评分。实际代码 fa9b272aa400ed3403e8f550c50a6957ec0f8588。N 为对应原NoAs20+10，C为对应原As20+10，不称N/C同父。

R/Q均从原 P43 B01 ownAs20学生与同源EMA独立追加E10，totalE30/6660累计更新；新段每行2220更新，resetAdamW lr2e-4、WD1e-4、cos10 min1e-6。seed2026092701/split392005固定。原父release执行提交19aedc3604fed9c193237dd92366164178af0983；不继承已关闭U候选或由目标挑出的权重。已有E50普通/加权三seed已完成，本轮不重复预算实验。

两行共用当前teacher-only资格公式：raw U single eval/no-grad ownEMA→原RX/day threshold quantile.86 clip[.92,.97]→原batch_neighbor(window2,minconf.8)。domain与学生strong-agreement均False，selectionhook不存在。原函数返回对象/内容原样保留。它与旧P43 strongagreement=True不同，当前冻结N/C控制也已共同采用本规则。不得要求urban学生先正确/高置信。

本轮仅用学生视图生成前已经存在的 conf_mask & temporal_mask。R/Q分别依赖自身后续EMA轨迹，k与未来资格可能不同；同数量规则不等于未来数量逐步相等。实际k、每步选位、CE/entropy资格与物理模拟成本均记录，Q-R仅作有限归因，不声称纯物理因果。

## 数据、模型和实际目标

保持 fullL6300/U56700（TX真值隐藏）/V27000、原全部source RX1/3/4/6/8、day1/2/3、全部6TX、eq1/IQ256/Fs25MHz、split392005。不拆Ldev、不留出训练RX、不扩大标签权限。F/SIG_TIME_X3/unifiedSinc/joint160、既有默认6×160 cosine head/scale30正常联合训练，无新分支/参数/拟合诊断头。

L仍是6前向/768包每步：.5RD(ordinary128)+.5RD(mixed128)+.25RD(raw128)+.1CROSS_RX ORD128；mixed含原ordinary64和raw→完整As64。总L名义尺度1.25，As CE .25，sameIQ RDrop与原LS.01、父epoch13 ordinary实际配置不变。不将普通噪声再叠在practical前。NoExplicitCFO/既有内部逐包补偿保持。

每U原一次raw教师与一次学生前向，原CE=.16×全部教师资格槽均值（空为可微0），entropy=.01×全U槽均值，包括未资格及R中的未资格As。Q中全部As进入原CE，熵项输入也变化；不能称只改变CE。hook只记录原损失并返回(None,{})，无新KL、附加CE、额外教师或U标签。

practical从raw生成，保留现有完整residual/post_sync、EQfalse/G=1、referenceSNR30、reflect、质量状态/相位跟踪/IQ补偿/AGC内部依赖；Fs25MHz/fc2.462GHz。与原L相同三训练场景，不连续二次加完整信道。U_R/Q共享固定新命名随机流U_QV，RX/day/eq稳定接收硬件，非TX定义；选择私有seed20261019731。

## 固定源决定及正式收尾

固定FINAL30，不按中间波动选checkpoint。仅两项源资格：Q−C在既有V1800×2draw的mid/low等场景准确率均值≥+0.5绝对pp，且fullV27000 clean Q−N≥−0.1pp。sourcepanel载频2.45GHz与训练/正式2.462GHz差异沿用并披露，两draw不是模型seed。Q-R、质量/锁定/各RX/day只作实际分层说明，不加额外晋级门槛。任一FAIL关闭固定资格路由、不E50/比例/阈值/父/seed救跑；R不作为备用候选晋级。通过也只提供未来预算/seed研究资格，不等于阶段或最终目标完成。

新row冻结后立即source读出→全部7固定视图prediction→separate exec truth-last scorer，失败源判定不取消正式评分。owner先保存并读取source决定，再显式launch final分析、读取新正式结果；不由目标分数选参数/种子/模型/重跑。分析采用4逻辑模型、6配对、504CM/84分区独立复计。主报告Q-C、Q-R、Q-N，全部accuracy/macroF1/最差RX/urbanmean/RXdayTX/修复新增/三个suburban绝对值及自身clean差。

正式每视图168000物理样本，RX0/2/5/7/9/10/11、day0..3。固定received IQ、SNR、信道及困难样本不改，clean+practical_high/practical_mid/practical_low_suburban/practical_high_urban/practical_mid_urban/practical_low_urban。正式阶段要求Q−匹配N clean≥−0.1绝对pp；最终仍需可重复clean正收益及星地提升，单seed/微差不作稳定证明。

## 执行、成本和停止规则

唯一代码/config Git提交待实现检查和一次P0/P1后记录，以release_commit.txt中的实际提交为准；不要求报告/config/sourceplan额外SHA。命令 CVS-RFFI/python -u experiments/cvs_u_qualified30/runner.py，worker --mode worker --row R01-s2026092701或Q01-s2026092701。CWD PROJECT/releases/cvs_FcrossRX_UQualified_E30_20261019_r01，普通N607既有Python/Torch fullFP32，CPU2；每GPU全部owner computePIDs≤3、free≥12000MB，不干预健康旧任务。

不可变输出 PROJECT/runs/20261019-phase1-FcrossRX-UQualified-e30-s2026092701-m2-r01，每row独占source/source_qualification/prediction/scoring/logs。首步真实As20父2NN+两syntheticL1536+两Uteacher/student32=1570NN smokePASS直接训练，0真实IQ/query/optimizer更新。边界检查覆盖E21第一步至E30最后一步、0/全资格和末batch124；无额外smoke授权文件或重复数据验证。一次最小普通preflight、一次archive传输SHA比较、一次remotecompile、一次PID/CWD/cmdline/GPU/log增长核实。

预计新两尾段20modelE/4440更新；每行studenttrain2271960/studenteval324002/teachereval567000，共native3162962；source12600/formal1176002，每行4351564。新训练/读出8703128NN+共享smoke1570=8704698NN，以实际counter为准。As U包数和非空调用数由实际资格动态决定；CPU选择/模拟/copy/logging/hostwall分别记录，inclusive route不重复相加。父As20复用去重，旧N/C训练及读出不再次收费，旧Urun技术失败770不向新run重复计。

只对精确run-owned的协议/query越界、错误数据/父契约、默认头/非数值、输出碰撞、无法合法prediction闭合等系统技术失败执行停止；先核实PID/CWD/argv并保留全部产物，不盲目重启。科学负结果不停止中途训练，不热改其他健康任务。不增加审批、seal/receipt/hash链或未来阶段机制。


实现已就绪：路由61聚焦负例/39正例、实际冻结native边界、20py/2emitted AST、analysis7fixtures/完整4440步及动态成本检查PASS。唯一独立P0/P1检查发现并修复archive依赖包bootstrap路径问题，唯一该点定向复核PASS，剩余P0/P1=0；不重复全审。真实1570NN烟测和训练仍待发布，不把本地检查当作模型成绩。


只读预检技术失败保留：初始提交93f402572的comparator config带Windows分隔符，远端未找到对应元数据路径而退出。独立poststate确认release/archive/run均不存在，0NN/CP/IQ/更新/worker，尚未发生传输。仅统一路径为POSIX，6项定点边界PASS；科学配置、源判据和预算未变。修复后才首次传输/启动，实际执行以新release_commit.txt为准。


实际发布：训练与分析代码 fe734e2b571767bae00b86fd0a19fdc784dadcdf，singlearchive compare/compile VERIFIED；1570NN actual真实父＋L/U smokePASS后直接训练。dispatcher1712988，R1713028/GPU2、Q1713110/GPU3，CWD/cmdline/CUDA/nvidia/loggrowth独立核实；actualnativeArgs全L6300 hiddenU56700 V27000、strictownAs20EMA/commonteacher-only符合。R首epoch真实As863、Uteacher/student各56700、extraNN0，Q仍健康运行。Sourceanalysis1714389仅phase source，WAIT_SOURCE、trutharg不存在、NN0；formalphase未启动，尚无识别结果或E50晋级结论。launch_verification.json为本次观察时刻记录，后续以完整训练/评分产物更新。


首epoch双行真实核实：R/Q均累计E21、222新增更新，LAs14208包、Uteacher/student各56700，UAs分别863/864，extraU NN0；无Utruth/target读取。数量差异来自各自teacher轨迹，不将其伪称精确同步剂量。


最终执行状态：ANALYZED。两行新增E10/累计E30、共4440新增更新，源与七视图独立评分完整，504CM/84分区及终态独立核实VERIFIED、missing=[]。源两条件FAIL在正式读分前已固定，资格路由关闭、R不备用晋级，不E50。正式Q-N clean−0.475pp阶段FAIL；Q-C low＋0.499405/mid＋0.276190pp，同时clean−0.475595pp。全部宏F1/最差RX/分层/纠错/预算见[最终分析](final_analysis.md)。运行时状态语句保留当时记录，以最终产物为准。
