# 完整源四因子与普通拼接：E20＋10，累计E30

状态：PARTIAL／五行训练预算完成、原生收尾评估失败。每行追加10轮／2220更新、累计30轮；自身final_ssdg.pth安全保留，所属进程全部退出。cached_parent误拒绝自身最终CP评估读取，原生状态HELDOUT_EVAL_INCOMPLETE／7保留；正式7视图尚未执行，另建零训练评分恢复run。

用户2026-10-17将预算提升E30。本修订在当前B10/B01/B11尚无正式结果时固定全部续训：B00继承完整源参照r02的REPEAT，B10/B01/B11继承clock-E20-r01对应行，FULL_AS继承完整源参照r02的CONCAT。5行都用固定source-final20，各追加10轮／2220更新，累计30轮／6660更新；不是从零单一E30，也不是原优化轨迹resume。父100modelE、新增50modelE／11100更新、累计150modelE分别记录，不按E20目标成绩筛行或选权重；健康旧E20不热改／覆盖。

F＋CROSS_RX、三路SIG_TIME、统一Sinc/CENTER_25K+CENTER_UNITS、joint160、默认cos头随骨干联合训练，NoExplicitCFO与内部必要补偿保持。全部RX1/3/4/6/8、day1/2/3、L6300/U56700隐藏TX/V27000、split392005；不留出训练域或L_dev。父数据角色和完整继承按原proof核对，学生／自身EMA严格加载。

追加10轮均L＋U，保留原oldgate／EMA规则，无新U视图或额外教师。统一重置AdamW（lr.0002、wd.0001）及原生cos10；不恢复优化器／RNG轨迹，段内更新计数与时序缓存重置并披露。普通增强固定真实父E13强度（父E13–20相同）；SupCon只在共有ordinary对，以累计E21–30权重.1继续。clock／practical命名流偏移20轮／4440步。

唯一假设仍是轻度±80ppm包内采样速率变化是否改善身份泛化并帮助practical，幅度／算子／损失均不变。B00/B10/B01/B11共有ordinary RDrop对.5，额外mixed对.5；G64/S64，缺失因子回填同一ordinary，所有组4次L前向，两次RDrop复用同一已生成IQ。Ag处理已有ordinary，As对ordinary之前原IQ施加一次完整residual/post_sync/noeq，所有卫星L监督。无As(Ag)组合、新损失、分支或头。FULL_AS保持完整ordinary128/practical128等权对，且同样累计30轮。B01/B11为半As剂量，不能冒充全As；B11必须同时直接比较FULL_AS。

每行新增L物理284160／L前向1136640，累计852480／3409920；启用G／半As新增142080、累计426240，FULL_AS新增284160、累计852480。学生／教师／smoke／源评估／正式预测与实际时间单列。固定seed2026092701，无扩种子。

自身父source-final20冻结且资源允许即训练，不等待父目标分数。固定追加阶段final10（累计30）；每行完成后自动原clean＋六practical（每视图168000），全部7视图及logp固定后exec独立CUDA为空CPU scorer才连接truth。receivedIQ、测试SNR／信道／样本不变。报告准确率／宏F1／最差RX、RX/day/TX分层、成对修复／新增错误、suburban绝对值与自身clean差，交互A11−A10−A01＋A00。

弃用规则不变：B10无clean正差或宏F1下降，或B11未保持相对B00的clean正差、lowurban>60%且超过B01和FULL_AS、三个suburban各差≤2pp，则关闭固定配方；单种子不完成稳定联合目标。不据测试改幅度、系数、seed、RX、场景概率、选父或选择性重跑。所有固定行自然完成，不按表现中途停止；技术停止仅协议／角色、非有限更新、输出冲突、不能启动或不能合法预测，保留产物与其他健康行，无自动重试。

Run 20261017-phase1-F-crossRX-clock-factorial-e20plus10-s2026092701-m5-r03；owner codex/root/F-clock-E30-20261017。实际Git commit发布时写入release_commit及resolved config。命令CVS-RFFI/bin/python -u experiments/cvs_clock_factorial30/runner.py，CWD PROJECT/releases/cvs_F_clock_factorial_e20plus10_20261017_r03；不可覆盖root PROJECT/runs/20261017-phase1-F-crossRX-clock-factorial-e20plus10-s2026092701-m5-r03。N607普通账号、CPU2/worker，共享每GPU最多3总进程（计入其他owner）。产物：父来源、优化器／EMA初态、实际args与源角色、全日志、最终CP、7视图prediction、独立scores、覆盖／成本／交互和报告；逐行字段见experiment.json。

成本口径：五条E30谱系父训练100modelE＋新增50modelE=150modelE；相关完整源矩阵另有旧NATIVE的20modelE（未继承），因此这些完整源矩阵实际合计170modelE／37740更新，另列测试／合成／教师前向。源阶段完成文件与training_result均就绪才启动对应续训，不等待任何目标分数。

技术恢复：显式native a1_scratch_only=False、from_scratch=False及对应baseline_ckpt；父仍严格验证／加载。旧机制晋级要求属于白名单外额外gate，记录REJECTED_EXTRA_GATE，只保持其在原scratch配方中关闭的实际语义；协议、tail、export、finalblocked、缺CP、heldout／endpoint失败均保留。实际原生scratch验证正反例与终态失败负例纳入合成检查。r01成本另计：5次合成更新／2580合成NN＋2包CP无查询smoke／1次worker父CP加载，0训练IQ／0真实更新／0正式预测。

失败r01/r02均0训练IQ／0真实更新／0正式预测，根目录与证据保留；其合成检查各5次更新／2580NN与CP无查询各2包、worker父CP各1次单列为技术成本。r03不重复或停止健康E20。

实际发布：单归档比较／远端编译VERIFIED；实际F合成136项负例PASS，2580合成NN／256合成practical包／128采样时钟包／5合成更新，0真实IQ／CP／target。一次真实CP无查询smoke PASS（2合成包／0查询／更新）。相隔63.05s独立核实5worker的PID/CWD/argv/GPU与各自step日志增长；5行严格父学生、严格自身EMA、空AdamW初态lr.0002／wd.0001均已读回。B00/B10/B01/B11/FULL_AS全部实际运行，源RX未留出，当前无技术失败；正式E30预测／评分尚未开始，检查通过不等于收益。见execution_evidence.json。

当前独立恢复：`20261017-phase1-F-crossRX-clock-e30-score-recovery-s2026092701-m5-r01`，只读取这五个已固定E30学生默认头，无新训练轮次或更新；不修改本run的原错误文件／权重。恢复将核对完整原生step／epoch／optimizer／角色／父证据，再在新root预测7视图与独立truth-last。瞬时reference／unit／clockMSE／scene细节未落盘明确缺项，不补造。
