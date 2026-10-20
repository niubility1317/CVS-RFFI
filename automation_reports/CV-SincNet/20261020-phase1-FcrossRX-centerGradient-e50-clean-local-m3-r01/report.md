# F跨RX：均值梯度、标准差停止的固定三种子消融

此补充组在任何新目标成绩读取前固定，原STAT_GRAD/NORM_ONLY六行及其不可变release/PID/配方保持。完整统计导数同时改变均值与标准差的梯度，原范数控制不能单独说明二者作用；CENTER_GRAD只保留均值导数，当前标准差继续按原方式停止梯度。描述子辅助梯度为g_native−mean_batch(g_native)，原尺度方向响应保留；这不是完整重算std前向的总导数，也不是梯度范数匹配。数学性质不等于RX因素被消除或目标收益。

固定CENTER_GRAD2026092701/02/03，共3行，均从零50＝33L＋17U/11100更新。原F三路SIG_TIME、统一Sinc、旧门控/ownEMA/cos、RDrop1、LS.01和普通增强不变；默认分类头正常联合训练，无另训头/投影/参数/额外网络前向。两次实际RDrop共用原同一次增强IQ；新SC仅消费可见TX的L128，U真值隐藏。保留原跨RX同TX正对/异TX负对、温度0.2、warm5/ramp10/max0.1。匹配BASE三种子复用，不重复训练。

原主比较为CENTER−同seedBASE：两弱TX各均差>0且各至少2/3正差、弱均差3/3正、整体均差≥0、其他各TX均差≥−0.25pp、最差RX均差≥−0.5pp。无源扩种子gate，全部3行自动final50→clean预测→独立CPU truth-last评分。保留每seed配对差/均值/样本SD/正差数、MacroF1、最差RX/day及成本、负结果。预先声明CENTER−STAT_GRAD与CENTER−NORM_ONLY为次级配对，其他条件完整3种子齐才计算，否则PENDING不凑部分均值；不据目标择优重排/调参。CENTER−STAT同时改变梯度大小与方差导数，不能只归因于尺度方向修正。

数据复用原READY/EXACT_MATCH/VALIDATED_ONCE L6300/U56700TX隐藏/V27000与目标clean168000，source/targetRX、日期和物理ID角色不变。源V仅默认头及无拟合几何诊断，不反传。无星地增强，仅测clean；不声称新的独立盲测、Phase2或部署性能。

CPU/CUDA f32/f64偏导参考/同输入前向损失/梯度均值0/保留尺度响应/常量触底/RNG检查PASS，14项配置权限负例PASS；真实合法BASE01 checkpoint合成no-query128双前向、Sinc辅助梯度/默认头主梯度/EMA不触发辅助项/导出边界PASS，加载权重丢弃，不被正式模型继承。一次直接P0/P1审查另存review.json。

普通本机账户，native Python3.10/PyTorch2.10cu128，原FP32/noTF32/benchmarkfalse/deterministicfalse，CPU2/workers0。GPU0源训练/预测总worker上限3，计入其他owner；派发前系统空闲内存≥4.5GiB；CPU评分最多1个。资源许可时可作为第三个实际GPUworker，不停止或热改原任务。Windowsvenv launcher与真实子进程不同，旧计数可能保守降低并行；不伪造计数或绕过容量。

一次新archive复制比较、compile、实际PID/CWD/cmdline/GPU/log增长核验；run/output不可覆盖，代码/config以实际Git commit唯一标识。命令/路径别名/CWD/逐行矩阵/stop/artifacts见experiment.json。技术停止仅来源/权限/配置/metadata/输出冲突、非有限或预算/worker错误，停止新派发并保留健康任务和产物，不按性能停止或自动重跑。尚无本轮目标性能结果，目标ACTIVE/UNMET。

发布状态：实现提交 `67afd61dad3f313c0919d5c08349e2ad4a1ff0f6` 已与远端分支OID核实一致。不可变release复制比较/compile通过，真实训练子进程PID/CWD/cmdline/GPU0及逐步日志增长核实；全局实际3个GPUworker，原六行保持健康。CENTER三种子固定，首行已发生实际更新，另两行排队。训练后逐行自动clean预测/独立truth-last评分；尚无新目标性能结果。

完成核对准备：新增只读CPU工具 `experiments.cvs_f_centergrad50.audit`，在三行完成且worker退出后核对实际checkpoint/默认头/特征更新、预测与独立评分时序、完整日志汇总及三种子配对算术。非对称混淆矩阵/MacroF1、样本标准差与错误结果拒绝检查通过，真实CLI在当前未完成矩阵上返回等待且未加载权重、IQ或真值。完整真实完成路径尚未执行，不能据此宣称实验或目标已完成；训练release与配方不变。

真实机制启用：CENTER首种子原生日志完整前缀1400次更新，其中第1111步（第6轮）起290次辅助更新；SC实际权重0.01至0.02，与warm5/ramp10日程一致。该前缀的训练/辅助损失及总梯度范数均有限，非有限跳过0、额外网络前向0、默认头直接辅助梯度0。完整39960次逐层反向观测待source结束后统一写出核对；此项不是完整三种子训练、导数审计或目标性能结论。其他健康行和配置不变。

首种子完整源训练核对：CENTER_GRAD-s2026092701完成50轮/11100次更新、9990次辅助更新和39960次逐层反向。完整日志无非有限跳过、U真值或星地活跃步骤；U曝光963900，接受15266（1.5838%）。实际默认头联合更新，部署特征54个参数张量变化且有限。源训练5606.01秒，峰值显存1972.05MiB。均值导数策略实际交付梯度范数比均值0.99718，尺度移除量为0，与保留停止标准差的设计一致；这不证明性能收益。clean预测依原队列自动进行，所有三种子仍完整执行，不依据部分成绩更改。证据见first_source_row_verified.json。
