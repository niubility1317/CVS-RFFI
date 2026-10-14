# 固定 SIG_TIME 的签名排序训练：首种子结果

原始签名排序83.6363%，比同路径SIG_TIME的83.5815%高0.0548个百分点；辅助去常量排序83.4476%，比SIG_TIME低0.1339个百分点。仅预定义seed2026092701，不能据此声称稳定增益或达到完整三seed84%。两个候选均按原源资格扩展，不依据本轮目标差异删候选、改权重、改epoch或重跑。

运行 20261014-phase1-signature-order-oldgate-cos-e50-clean-manysig-s01-m4-r01，实现 f6dfdb587c505b5720469d4a4a101b8df8b321f4，执行 2658815a2fe5ef931f24558a6c6c61146433fd9c。源资格决定 d56f9b79c9fea83629c525e362d473ff0fcafb1e 已提交推送、固定02/03扩展已派发后才回收新目标数字。source资格：三终轮LCE 0.247153857/0.248473601/0.248924990；RAW/DYNAMIC−ZERO L720zL2 1.310938478/1.795316935；真实t1/t2/SIG投影在17/33/50有效非零、预算/export/W独立PASS。资格验证接线与源健康，不证明DG有效，也不按源几何排行。

|方法|clean准确率|Macro-F1|最差RX|最差day|
|---|---:|---:|---:|---:|
|OLD_GATE_COS50|80.6685|80.1734|70.6167|75.4524|
|SIG_TIME|83.5815|83.1524|71.4875|78.9452|
|SIG_RAW_ORDER|83.6363|83.2076|71.2000|79.2214|
|SIG_DYNAMIC_ORDER|83.4476|83.0075|70.4417|78.9786|

单位均为%。固定clean168000/row、旧6TX×7目标RX×4days、modelseed2026092701、split392005。全部72个同row整体/RX/day/TX指标在results.json。暴露研究基准，不称新的独立盲测。源数据L6300/U56700 hidden/V27000与physicalIDs、RX1/3/4/6/8、days1/2/3保持。

|配对|差值（百分点）|修正错误|引入错误|净正确变化/168000|
|---|---:|---:|---:|---:|
|SIG_DYNAMIC_ORDER − SIG_RAW_ORDER|-0.1887|405|722|-317|
|SIG_DYNAMIC_ORDER − SIG_TIME|-0.1339|573|798|-225|
|SIG_DYNAMIC_ORDER − OLD_GATE_COS50|2.7792|10252|5583|4669|
|SIG_RAW_ORDER − SIG_TIME|0.0548|588|496|92|
|SIG_TIME − OLD_GATE_COS50|2.9131|10229|5335|4894|

RAW−ZERO净+92，DYNAMIC−ZERO净−225；DYNAMIC−RAW净−317。原网络旧gatecos仅为历史锚点，SIG候选相对其大约2.8至3个百分点的总差不能全部归给本轮辅助损失。n=1不计算样本标准差。

## 训练机制与实际证据

遵循用户最新要求，网络固定单路SIG_TIME112→joint160，fθ195041参数。三新臂相同网络、同初始body/W和完整参数、初始eval函数同。RAW约束实际描述符112；DYNAMIC仅在辅助距离去4固定时间增量成108，保留所有时间面积，部署112/结构不变。普通默认cosine6×160分类头与骨干按原CE+U流程联合训练，无任何单独头拟合/重训/替换或RX/day诊断分类头。

三新均scratch50、33label+17pseudo、11100更新、原oldgate batch_neighbor/ownEMA/cos，无新父权重或卫星增强。辅助使用真实L标签同日跨RX身份三元组，m=.1/coeff=.1/5warmup+10ramp预先固定；每4native步额外L90 eval且保留梯度，恢复mode/RNG，三臂含ZERO均2750次/247500额外源包。无SupCon/PAIR/层级/难度权重叠加。

实际训练epoch1/17/33/50各55extra保存并复算mainLCE、RAW、DYNAMIC、effective三参数梯度。epoch1有效辅助0，17/33/50两个候选三对象均非零，所有辅助W依赖检查为None。E50有效辅助/mainLCE范数：

|候选|t1PW|t2PW|实际SIG投影|
|---|---:|---:|---:|
|SIG_RAW_ORDER|2.4305%|2.3190%|3.5857%|
|SIG_DYNAMIC_ORDER|2.7049%|2.5594%|3.8620%|

这些比率来自mainL128与另一个extraL90批，非同包梯度或累计AdamW贡献。全部1980层cohort记录/7920保存向量、8250extra/742500额外源包已独立重读完整产物复算；200源epoch、33300新更新＋11100历史更新、672000固定defaultdecisions及72组指标亦复算VERIFIED。新增模型150model-epochs，历史50另列。

三个实际部署fθ数值更新，5个SIG pool张量均有更新，导出最大差0，W置零不改变z、导出删除旧W并保留特征归一化/增益。两对实际源特征差与初态相同均验证，所有额外头拟合0。该边界和参数变化不证明新TX迁移。

当前旧随机门控U接收：SIG_TIME15452、RAW15445、DYNAMIC15428，分母均963900（约1.60%）。门控、lambda和U日程保持；acceptedmean损失不能乘接受比例解释。此结果不能与另一独立SIG_TIME/PAIR试验拼成高覆盖U已有效的结论。

辅助去常量在源几何和梯度上有变化，首seed目标却更低；更大梯度、较低未满足率不作为泛化代理。GPU非确定性、两个辅助目标归一化尺度不同均保留，不从小单seed差异作单一因果归因。固定最终epoch结果不等于已证明收敛；完整50源曲线和四期梯度见results.png/json。

## 后续固定工作

原样02/03扩展运行 20261014-phase1-signature-order-oldgate-cos-e50-clean-manysig-s0203-m8-r01，执行 d56f9b79c9fea83629c525e362d473ff0fcafb1e；六新scratch50已真实启动，首seed不重训。完整3seed尚待完成，报告两候选和所有负结果，不以首seed挑选赢家。另固定SIG_TIME半监督RANDOM/PAIR参照 20261014-phase1-sigtime-pair-oldgate-cos-e50-clean-manysig-s01-m3-r01 按原独立配置运行，用户研发重点为训练/DG/SSL机制，停止网络结构扩展。84%目标未达成，无Phase2/卫星/新TX能力声明。
