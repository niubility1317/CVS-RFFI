# 有限窗接收质量 A_s：同父 E20＋10，累计 E30

状态：已实现、已登记，尚未发布。唯一假设：用实际原始 IQ 的有限窗 H*x 功率决定 practical 接收质量，保留传播、噪声和补偿依赖并改善识别。依据是完整原 L6300／5 RX 的18900同信道物理配对：1010（5.34%）改变有效处理、355改变lock；不是身份识别收益。

只新增 QUALITY_POWER 一行，从完整源普通 FULL_AS 的 source-final20 学生和自身 EMA 严格追加10轮，累计 E30／6660更新。复用已完成同父 FULL_AS E30 对照，不复训或重测对照，不继承失败时钟 Ag。父是为唯一算子匹配固定的 source-final20，原五行预算与父在新 E20 目标读取前固定，不按目标排名选父。两行同父、同初始化规则、AdamW／cos10复位（lr0.0002/wd0.0001）、原 U／EMA规则与分段计数／temporal cache复位、ordinary冻结父实际E13、SupCon累计E21至30权重0.1、同场景／seed／IQ随机流；不是 scratch30 或精确 optimizer resume。

全部源 RX1/3/4/6/8、day1/2/3、L6300／U56700 TX隐藏／V27000、split392005；无训练域留出或5580/720替代。F＋CROSS_RX、三路SIG_TIME、统一Sinc、joint160、默认cos头及骨干正常联合训练、NoExplicitCFO／内部逐包补偿保持。无新Ag、loss、U机制、诊断头、部署分支或参数。

原传播核 h 保持。每 As 用同一 raw、原 reflect 边界做一次 H*x 功率预滤波，除以不变的参考噪声方差，再传给原 residual.process；完整残差引擎只处理一次，不先生成另一份完整信道或叠加信道。保留参考噪声30dB、原几何／状态／硬件／校准抽样及依质量联动的相位／频率处理和AGC。采集噪声未知，此口径仍不是真实总 SNR。正式 received IQ／测试 SNR／信道／样本完全不改。

As走 raw-before-ordinary，不重复旧 AWGN／multipath／PA；全部合法有标签As监督，不筛urban置信。ordinary和As的 RDrop 对各权重0.5、按128物理样本归一化；每对同一生成IQ、4次L前向，clean份额与完整普通拼接一致。每 As 额外一次纯物理功率预滤波，0额外网络前向；预滤波计数和实际CPU／墙钟单列，不称零成本。

固定 final10／累计30，自动clean和六practical，每视图168000物理样本。七视图 prediction/logp 全固定后 exec 独立CUDA为空CPU scorer连接truth。报告全部RX/day/TX、宏F1、最差RX、suburban绝对值／自身clean差、同包修复和新增错误。此单As隔离不是新四因子，不把旧Ag增益与本候选相加预测联合成绩。

预定放弃条件：相对同父nominal As，low urban增量不足0.5个百分点，或clean／mid urban下降，或任一suburban下降超过0.25个百分点，即关闭该固定配方；不据目标成绩调窗口／噪声／系数／seed救跑或自动扩种子。另报告完整联合目标：clean相对同预算B00净正、low>60%、各suburban与自身clean差≤2pp探索条件，不以降低clean缩小差距。单seed局部正差不完成总目标。

本地21项协议负例＋48单次物理实现例（另48参考调用），覆盖LOS/PARTIAL/BLOCKED，PASS；0真实IQ/CP/NN/更新，compile PASS。限定独立P0/P1审查一次PASS。旧native收尾加载自身final误拒绝，只在本新副本修复：缓存声明父或原loader读取本run自身final，其余拒绝；正负路径已直接验证，旧任务不改。发布首步执行一次真实CP无查询2包smoke，默认头及严格父数据证明通过即继续。

owner codex/root/F-practical-power-E30。不可覆盖root PROJECT/runs/20261017-phase1-F-crossRX-practical-power-e20plus10-s2026092701-m1-r01；CWD PROJECT/releases/cvs_F_practical_power_E30_20261017_r01。命令 CVS-RFFI/bin/python -u experiments/cvs_practical_power30/runner.py。N607普通账户、CPU2、每GPU最多3个总实验进程含其他owner。一次N607资源检查实际回读已完成。执行commit在release_commit／resolved记录；一个发布归档和一次传输核对、remote compile、首步smoke及PID/CWD/argv/GPU／日志增长依既有最小流程。停止仅数据／来源／非有限／预测不全／输出冲突／系统技术失败，保留产物，无自动重试，不动其他健康任务。

新增训练10 model-epoch／2220更新；父20／4440单列，累计30／6660。复用nominal对照的原20＋10另记，不计为新训练。预期新增L网络包1136640、As284160、功率预滤波284160、原U567000观测；教师、学生源评估、smoke和实际墙钟分别记录。无超过 E30 的预算。
