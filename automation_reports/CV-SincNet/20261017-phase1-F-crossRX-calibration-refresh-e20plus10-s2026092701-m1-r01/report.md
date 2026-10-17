# 原分布接收校准刷新 A_s：同父 E20＋10

状态：**RUNNING／VERIFIED**。执行commit f375e6cc47cfb7d5e7b069b0bd912d5bf8819a0e。一次归档传输、remote compile、同引擎检查、真实CP无query2包smoke均PASS；原worker PID/CWD/argv/GPU和源step日志228585→293911字节增长已独立核实。严格同父学生／ownEMA／空优化器初始化通过，现已观察新增E1／累计E21、222／累计4662更新、L113664 NN／As28416；原进程健康，尚无final30或正式识别结果。唯一假设：保持原 practical 传播、噪声、接收质量／锁定及真实注入硬件参数不变，仅将原模拟器校准估计误差按源 epoch／RX-day-eq 刷新，能减少对固定15校准状态的依赖并改善星地身份识别。代码证据为原 receiver_for_session 由固定 seed＋RX/day/eq生成同一个true-plus-error估计，现有训练15状态全程复用；它不是已证明的性能瓶颈。

唯一模型 CAL_REFRESH-s2026092701 从完整普通 As 的固定source-final20及自身EMA严格追加10轮，累计E30／6660更新；对照是现成同父nominal E30 FULL_AS，不重训、重测或从失败quality/clock模型继承。父只由严格算子隔离确定，非目标分数排名。承认此前基准已暴露，不声明新的独立盲测；目标分数不用于校准幅度、分布、seed、父、选模、选择性重跑或后续扩种子。

F＋跨RX同TX、三路SIG_TIME、统一Sinc、joint160、原默认cos头与骨干联合训练、NoExplicitCFO及必要补偿保持。全部原RX1/3/4/6/8、day1/2/3，L6300／U56700 TX隐藏／V27000，split392005；无源训练域留出、无5580/720替代。AdamW／cos10复位lr.0002、wd.0001、原U/ownEMA、segment/cache复位、ordinary冻结父实际E13、SupCon累计E21至30为.1，与nominal相同。不是scratch30或精确optimizerresume。

原 true amplitude／quadrature／CFO 永远保持15个session。仅estimated amplitude=true＋N(0,.05 dB)，estimated quadrature=true＋N(0,.2°)；尺度直接继承既有Config，不拟合源/目标。新私有seed20261017541按RX/day/eq和累计epoch21..30产生150组共享估计，label不参与；同epoch同session重复调用一致。原latent seed、h、噪声方差、几何、state、quality／lock／phase／frequency／AGC规律保持，同一原IQ只过一次完整residual。无新物理滤波、完整信道预跑、A_g、loss、U机制、teacher、NN前向或部署参数。数字变换已接收IQ不等于真实RX替换；不修改或校正源TX身份非线性/IQ特征，不把校准多样性命名为已证域泛化。

原ordinary／As RDrop两对各.5，物理L128归一化、4L前向，各对同一已生成IQ，clean监督份额保持；As从raw-before-ordinary生成，全部合法有标签样本监督，无urban置信筛选。预期新增2220更新、L1136640包NN、As284160、原U567000；校准估计实际draw150，估计应用284160；教师／源eval／预测／smoke／墙钟实际单列，父20成本复用另列，0超过E30预算。

固定final10／累计30后自动clean＋六practical，每视图168000，七视图prediction与logp全部固定后独立CPU truth-last。测试receivedIQ、SNR、信道与困难样本不变。全部RX/TX/day、accuracy／macroF1／worstRX、各suburban绝对值及自身clean差、修复／新增错误与clean场景同包联合正确均报告。

固定反证：相对nominal低urban增量不足.5pp，或clean／midurban下降，或任一suburban下降>.25pp，关闭该固定识别配方，不调整误差幅度／刷新间隔／seed／父／预算救跑、不自动扩seed。还须报告完整目标：clean对同预算B00净正、low>60%、各suburban自身clean差≤2pp首轮探索条件，不能降低clean缩差；此单As隔离不构成新四因子或联合成功。

32项协议负例、240同latent合成配对（另240原参考＋240重复验证调用）PASS，覆盖LOS/PARTIAL/BLOCKED；只变估计参数，h／quality／lock／freq／原噪声方差相同，输入及全局NumPy RNG不变；合成IQ最大相对变化.00760068。本地合成检查0真实IQ／CP／NN／更新；发布smoke另有1真实CP／2合成NN／0查询。这是实现证据，不能替代真实标签保持／识别收益。唯一独立P0/P1审查PASS（0 P0／0 P1）。旧自身final loader修复原样保留。

owner codex/root/F-calibration-refresh-E30；不可覆盖root PROJECT/runs/20261017-phase1-F-crossRX-calibration-refresh-e20plus10-s2026092701-m1-r01，CWD PROJECT/releases/cvs_F_calibration_refresh_E30_20261017_r01，命令 CVS-RFFI/bin/python -u experiments/cvs_calibration_refresh30/runner.py，N607普通账户／CPU2／每GPU最多3总实验进程含其他owner。实际代码commit由release_commit与resolved写入，一次N607preflight、一个归档传输比较、remotecompile、首步真实匹配CP无query2包smoke PASS即继续，再一次PID/CWD/argv/GPU/loggrowth核实。仅数据／来源／非有限／预测不全／输出冲突／系统技术失败停止run-owned流程并保留，不自动重试，不修改旧健康任务。

REJECTED_EXTRA_GATE：单独设计审批／额外hash或报告完备审批在用户明确最小实验工作流下NONBLOCKING；无新增审批层。
