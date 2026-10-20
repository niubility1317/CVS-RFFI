# 从零学习恢复映射时加入冻结原 F 指导

状态：PLANNED。用户目标为目标low urban≥70%，其余五practical接近原clean85.113889%。“接近”暂按≤2个百分点，是Agent操作解释。当前最好66.406548%，完整目标未实现。本次假设和两行预算在teacheragreement的新目标分数读取前固定；代码依据为原R03先以波形目标训练coarse20+correction20，近期方案再做E10后置F约束。检验从原逆映射学习开始加入身份保真是否不同，不根据目标成绩选起点、调参或选择性重跑。

|固定行|恢复器初始化|训练目标|训练F开销|
|---|---|---|---|
|scratch_wave|相同新ComplexTFUNet随机权重|IQNMSE+.1logpower|0教师缓存/前向|
|scratch_F_guided|相同新ComplexTFUNet随机权重|同波形+三原F finalcos+.3T²KL(T2)，per_frame_cap|三F参考缓存/训练前向|

两行均1305958可训练参数、相同原始ComplexTFUNet架构，零temporal输出初始严格等于当前receivedIQ。无任何恢复器初始化/resume；原R03 b01b534f只作已经核实的数据契约一致的源/目标读出参照，不载入新模型。禁止晋级任何失败或按测试挑选权重。三原F seeds2026092701/02/03/method4df7b94d/defaulthead整体冻结，无新身份/RX分类头或单独头拟合，L/U TX真值均不读取。

原source L6300/U56700、R02缓存epochs1..20按1..20顺序重复一次，随机shuffle由dataseed2026101951持续生成40次，原.15clean/.60residual/.25full路由不变。observed+samephysicalclean，batch256物理→512IQ，末批按实际数归一化。modelseed2026092701，E40/9880，L252000/U2268000优化曝光、5040000IQ。两臂AdamW2e-4/warm247/cos2e-5、WD1e-4/clip1、FP32/noTF32。guided从首步计算mean3F finalcos+.3T²KL；逐IQ wave/identity sum-loss输出梯度，经既有per_frame_cap后除实际IQ数一次。wave臂不建训练教师缓存、不执行教师前向，输出梯度严格等于波形mean-loss梯度。损失/实际cap/路由/LU曝光/三F训练调用和IQ、成本均保留。输出梯度限制不保证AdamW有限步或最终身份收益。

两新行的40轮从零训练是匹配比较，旧R03复合架构/40+10继承训练不冒充相同训练成本或结构。原F50训练已固定，不更新，三原F独立读出不是三个恢复器seed。恢复推理只输入当前receivedIQ，不调用F、clean、缓存、TX/RX/环境标签、其他帧或投票，不更新模型/运行状态。

source V1800/固定F1/七视图仅诊断，固定finalE40，无源资格/早停/排序/状态更新。每行训练冻结后自动168000物理目标clean+六practical，三原头各自argmax。每行七视图IQ/ID/预测全部固定后独立无CUDA CPU连接目标truth和同物理clean，报告逐F/TX/RX/day、准确率/MacroF1、修复/损坏/净纠错、距原clean差距及full256/prefix232/tail24 NMSE/logpower。波形、源分数或实现检查不替代目标结果；已暴露基准不称新独立盲测、Phase2或在轨性能。

31项聚焦检查PASS，覆盖scratch/state/参数量/零输出/批独立/checkpoint与权限预算、wave梯度等价、guided原头到IQ和冻结F、缓存轮次/LR、source行负例及truth-last包。唯一一次独立P0/P1审查PASS，审查者独立复跑31项通过，未发现确定P0/P1。launcher首步真实原F/四Ls no-query smoke：相同随机初态、原头到IQ与新head梯度、F完整状态/新checkpoint往返，0TX/V/target/optimizer；PASS立即进入资源队列。

新run/root/release不可覆盖。一次N607普通账号预检核实合法原F/缓存/基线、空间、现有owner和GPU4/5；CPUcontroller先等agreement根终态且其owned source/target真实PID均退出，再创建两worker，不能只看终态marker，放行不读分数。spatial GPU6/7全部保留，总最多4个本owner GPUworker，每卡一个本owner，source≥18GB/target≥12GB、allowner<2。CPU2线程/本owner scorer≤1，新source scorer等spatial根终态，再共用本run score.lock；source实际退出才同卡target。权限/代码/冲突、非有限/OOM、预算/梯度/F冻结或预测失败则保留产物、失败该行、不自动重试；健康行继续。

代码配置一个Gitcommit；一次归档本地/远端SHA比较、远端compile；launch后实际PID/CWD/完整命令/GPU/日志增长独立核实。精确命令、CWD、原始来源、路径、矩阵和stop rule见experiment.json。IQ/权重/逐样本预测/日志和私有机器状态不提交。未向另一用户对话发送消息。

一次N607普通账号预检VERIFIED：新root/release不存在，合法原R03只用作baseline；agreement已终态，当前其他owner与spatial健康任务完整保留，GPU4/5可用且启动队列仍核实旧owned handles真实退出，磁盘空间充足。无额外签名、数据重验证或审批流程。正式训练尚未启动。
