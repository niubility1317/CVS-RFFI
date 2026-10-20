# F跨RX：均值梯度、标准差停止的固定三种子消融

此补充组在任何新目标成绩读取前固定，原STAT_GRAD/NORM_ONLY六行及其不可变release/PID/配方保持。完整统计导数同时改变均值与标准差的梯度，原范数控制不能单独说明二者作用；CENTER_GRAD只保留均值导数，当前标准差继续按原方式停止梯度。描述子辅助梯度为g_native−mean_batch(g_native)，原尺度方向响应保留；这不是完整重算std前向的总导数，也不是梯度范数匹配。数学性质不等于RX因素被消除或目标收益。

固定CENTER_GRAD2026092701/02/03，共3行，均从零50＝33L＋17U/11100更新。原F三路SIG_TIME、统一Sinc、旧门控/ownEMA/cos、RDrop1、LS.01和普通增强不变；默认分类头正常联合训练，无另训头/投影/参数/额外网络前向。两次实际RDrop共用原同一次增强IQ；新SC仅消费可见TX的L128，U真值隐藏。保留原跨RX同TX正对/异TX负对、温度0.2、warm5/ramp10/max0.1。匹配BASE三种子复用，不重复训练。

原主比较为CENTER−同seedBASE：两弱TX各均差>0且各至少2/3正差、弱均差3/3正、整体均差≥0、其他各TX均差≥−0.25pp、最差RX均差≥−0.5pp。无源扩种子gate，全部3行自动final50→clean预测→独立CPU truth-last评分。保留每seed配对差/均值/样本SD/正差数、MacroF1、最差RX/day及成本、负结果。预先声明CENTER−STAT_GRAD与CENTER−NORM_ONLY为次级配对，其他条件完整3种子齐才计算，否则PENDING不凑部分均值；不据目标择优重排/调参。CENTER−STAT同时改变梯度大小与方差导数，不能只归因于尺度方向修正。

数据复用原READY/EXACT_MATCH/VALIDATED_ONCE L6300/U56700TX隐藏/V27000与目标clean168000，source/targetRX、日期和物理ID角色不变。源V仅默认头及无拟合几何诊断，不反传。无星地增强，仅测clean；不声称新的独立盲测、Phase2或部署性能。

CPU/CUDA f32/f64偏导参考/同输入前向损失/梯度均值0/保留尺度响应/常量触底/RNG检查PASS，14项配置权限负例PASS；真实合法BASE01 checkpoint合成no-query128双前向、Sinc辅助梯度/默认头主梯度/EMA不触发辅助项/导出边界PASS，加载权重丢弃，不被正式模型继承。一次直接P0/P1审查另存review.json。

普通本机账户，native Python3.10/PyTorch2.10cu128，原FP32/noTF32/benchmarkfalse/deterministicfalse，CPU2/workers0。GPU0源训练/预测总worker上限3，计入其他owner；派发前系统空闲内存≥4.5GiB；CPU评分最多1个。资源许可时可作为第三个实际GPUworker，不停止或热改原任务。Windowsvenv launcher与真实子进程不同，旧计数可能保守降低并行；不伪造计数或绕过容量。

一次新archive复制比较、compile、实际PID/CWD/cmdline/GPU/log增长核验；run/output不可覆盖，代码/config以实际Git commit唯一标识。命令/路径别名/CWD/逐行矩阵/stop/artifacts见experiment.json。技术停止仅来源/权限/配置/metadata/输出冲突、非有限或预算/worker错误，停止新派发并保留健康任务和产物，不按性能停止或自动重跑。尚无本轮目标性能结果，目标ACTIVE/UNMET。
