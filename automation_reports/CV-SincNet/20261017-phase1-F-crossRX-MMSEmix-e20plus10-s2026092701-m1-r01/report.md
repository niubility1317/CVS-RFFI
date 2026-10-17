# NoEQ／MMSE24 接收处理混合：完整源、同父模型累计 E30

**本轮完成训练、七视图预测、独立 truth-last 评分和全日志分析；固定条件未通过，关闭这一固定方案。**

执行代码 d17844d18e0795d8eac9a3465aa2613a20ab6b86。同普通拼接式 source-final20 与自身 EMA 追加 10 轮，累计 E30／6660 更新；既有同父 NoEQ E30 控制直接复用。未留出训练 RX。不是从零 E30，也不是原优化器轨迹恢复。

clean 83.185119%，low urban 53.184524%。同预算 B00 clean 85.268452%，clean 净差 -2.083333 个百分点。单种子结果不能证明 A_g 有效或联合目标完成。

## 七种正式视图

|视图|MMSE 准确率 %|普通准确率 %|差值 pp|MMSE 宏 F1 %|普通宏 F1 %|MMSE 最差 RX %|普通最差 RX %|
|---|---:|---:|---:|---:|---:|---:|---:|
|clean|83.185119|82.963690|+0.221429|82.759958|82.531385|73.641667|73.012500|
|suburban high|81.470833|81.620238|-0.149405|81.039770|81.193932|72.891667|73.000000|
|suburban mid|79.092857|79.329762|-0.236905|78.675192|78.901063|71.325000|71.595833|
|suburban low|76.707738|76.977381|-0.269643|76.315406|76.570904|69.445833|69.720833|
|urban high|77.986310|78.189286|-0.202976|77.525825|77.733848|70.425000|70.570833|
|urban mid|64.039286|64.371429|-0.332143|63.576590|63.909243|58.195833|58.650000|
|urban low|53.184524|53.447024|-0.262500|52.641215|52.899938|48.800000|48.954167|

每视图均为原 168000 个物理样本。全部七视图 prediction/logp 固定后，独立 CPU scorer 才读取 truth。未提高测试 SNR、改变 received IQ／信道、删除困难样本或单独拟合头。完整 RX／TX／day 见 [分层结果](stratified_results.md)。

## 固定条件与联合目标

|预登记条件|实际结果|
|---|---|
|low_urban_gain_at_least_0_5pp|未通过|
|clean_not_lower|通过|
|mid_urban_not_lower|未通过|
|each_suburban_no_more_than_0_25pp_lower|未通过|

条件在训练和目标访问前固定：low urban 至少 +0.5 pp、clean 与 mid urban 不下降、各 suburban 不低于控制超过 0.25 pp。结果不回流混合概率、NMSE、延迟、SNR、系数、父模型、seed 或预算；不自动扩种子。

|suburban|混合与自身 clean 差距 pp|普通与自身 clean 差距 pp|
|---|---:|---:|
|suburban high|1.714286|1.343452|
|suburban mid|4.092262|3.633929|
|suburban low|6.477381|5.986310|

2 pp 是本轮事前采用的探索目标。low urban >60%：False；全部 suburban 差距≤2 pp：False；clean 相对 B00 净正：False。降低 clean 所缩小的差距不构成保护成功。

## 配对错误变化

|视图|修复旧错误|新增错误|准确率净差 pp|同时 clean 与场景正确净差 pp|共同 clean 正确子集内修复／新增|
|---|---:|---:|---:|---:|---:|
|clean|850|478|+0.221429|+0.221429|0／0|
|suburban high|1032|1283|-0.149405|+0.008929|625／988|
|suburban mid|1255|1653|-0.236905|-0.133333|797／1357|
|suburban low|1361|1814|-0.269643|-0.135714|919／1489|
|urban high|1184|1525|-0.202976|-0.055952|761／1213|
|urban mid|1608|2166|-0.332143|-0.190476|1194／1770|
|urban low|1898|2339|-0.262500|-0.134524|1525／1940|

## 全日志执行与成本

复算本轮全部 2220 个新 step，损失恒等式最大误差 1.0728836064e-07；完整 L／U 采样序列与控制相同，68 个实际普通增强／预算／U／父权重参数相同。独立重放 284160 次接收处理选择，覆盖 6300 个 L 与原 15 个 RX/day session；硬件及标定状态逐值相同。

|成本／执行项|实际值|
|---|---:|
|status|FULL_SOURCE_E30_TRAINED|
|actual_steps|2220|
|parent_steps|4440|
|cumulative_steps|6660|
|parent_epochs|20|
|new_epochs|10|
|cumulative_epochs|30|
|L_forward_packets|1136640|
|student_train_forward_packets|1703640|
|student_eval_forward_packets|324002|
|teacher_train_forward_packets|0|
|teacher_eval_forward_packets|567000|
|SupCon_applied|2220|
|SupCon_layer_forward_terms|8880|
|As_training_observations|284160|
|Ag_training_observations|0|
|source_RX_holdout|None|
|head_fit_updates|0|
|U_truth_read|False|
|target_access|False|
|elapsed_seconds|1326.587725162506|
|optimizer_reset|True|
|own_parent_EMA_restored|True|
|receiver_calls|284160|
|receiver_counts|{'0': 142289, '1': 141871}|
|equalizer_applied|109303|
|session_count|15|
|channel_wall_seconds|564.2015551759396|
|physical_channel_prepasses|0|
|amplitude_renormalization|False|
|true_hardware_and_calibration_unchanged|True|
|propagation_prior_unchanged|True|
|noise_prior_unchanged|True|
|unlocked_EQ_skipped|True|

新增预测 NN 前向 1176002，墙钟 43.220295 s；独立评分 CPU 墙钟 3.396524 s。真实 checkpoint smoke 为 1 CP／2 个合成 NN 前向／0 IQ／0 更新。分析阶段 0 IQ／CP／原始 truth／NN／更新。相同 NN 前向与更新数不等于相同 CPU 或墙钟成本。父 20 轮／4440 更新及源诊断成本单列，未重复控制训练或预测。

训练曲线读取父模型全部 20 epoch／4440 step，以及本轮与控制各 10 epoch／2220 step；未使用历史静态数据。旧控制技术清理缺失的瞬态计数保持缺失，不推算为实测。

## 结论与边界

源域 MMSE 输入干预收益未足以转化为本轮预登记要求的正式训练收益。关闭这一固定方案，不用源健康、特征变化或单个视图涨分代替成功。

本轮未提出或验证新 A_g，也未重做四因子矩阵。既有 E30 四因子负结果保留，不与本行成绩相加预测联合收益。当前没有被本轮证据证实的后继候选；下一步须由独立合法源证据提出新的身份保持 IQ 假设，不能从本轮目标错误反推增强参数。

启动验证、唯一一次有界 P0/P1 审查、真实 checkpoint smoke 见 [verification.json](verification.json)。方法／命令／路径／停止规则见 [experiment.json](experiment.json)，完成证据见 [execution_evidence.json](execution_evidence.json) 与 [result_summary.json](result_summary.json)。

[完整训练曲线与七视图结果（PNG）](../../../local_artifacts/F_MMSE_mix_E30_20261017/figures/MMSE_mix_matched_E30.png) · [PDF](../../../local_artifacts/F_MMSE_mix_E30_20261017/figures/MMSE_mix_matched_E30.pdf)

## 原预登记与启动验证

## 唯一训练行与匹配参照

新行 MMSE_MIX-s2026092701：对应普通星地 CONCAT source-final20 及自身 EMA 追加10轮，累计 E30／6660更新，新增2220更新。与既有 NOMINAL_AS E30 共用同一父模型、初始化／EMA恢复规则、空 AdamW＋原基准学习率重置 cosine10、原 E13 普通增强冻结日程和累计 E21至30 的 SupCon 权重0.1。不是从零E30或恢复原optimizer轨迹，不继承源诊断或PDP变体权重。

模型仍 F＋跨RX同TX，三SIG_TIME、统一Sinc CENTER_25K／CENTER_UNITS、joint160、默认cos6×160 scale30与骨干联合更新。NoExplicitCFO和必要逐包补偿保留，0新参数／部署分支／头／损失／新U规则；RDrop是既有随机子网络正则，不称为IQ A_g。

全部5源RX1/3/4/6/8、day1/2/3、L6300/U56700/V27000、split392005保持。不存在留出训练RX或内部L5580/L_dev720。U TX真值隐藏，原旧门控／自身EMA／batch_neighbor半监督不变，不增加U星地或teacher视图。V仅原健康观测，不反向更新。

## 实际 IQ 与损失

每个L先保留原普通增强IQ的两次RDrop前向；原始L独立生成一次完整practical IQ，再复用它做两次RDrop。总监督仍0.5 RD(ordinary)＋0.5 RD(As)＋0.1 crossRX SupCon，按物理L和每对视图原规则归一化；原监督份额与总损失尺度不变。全部有标签As参加监督，不看困难学生置信度。

唯一改变：私有、与TX标签无关的逐物理样本／累计step抽样，以0.5／0.5先验选择NoEQ或现有MMSE24。传播scene选择、H、遮挡／几何／阴影／散射、原噪声N0、RX/day session硬件及标定、CFO／相位／IQ补偿和质量依赖沿用。MMSE既有FIR长度129、增益上限20dB、H加误差NMSE−25dB代理及质量scale不变；目标延迟24点在源诊断前由NoEQ共同数值延迟固定，64点不作为候选。选择MMSE但unlocked时仍按原规则跳过EQ；G同时作用信号和噪声，AGC按处理后的实际输出重新计算。不增SNR，不裁剪／删除困难包，不连续叠加两次完整星地信道。

实际接收处理发生在新的L-only IQ adapter。native未消费的practical_EQ参数仍为false，不能据该参数误判实际关闭；actual_channel、每step选择／生效数和各RX/day/scene/state/lock覆盖记录实际执行。共同PA/DAC/class-signature/CFO等旧普通增强仍按真实parent args原样记录，不擅自更改或称其已验证身份保持。

## 预算、自动测试和放弃条件

新增每模型10轮／2220更新，1136640个L NN前向、284160个完整As观测；原U／EMA／验证、真实checkpoint smoke和测试分别计数。MMSE额外CPU处理、实际EQ生效数、信道墙钟及总墙钟单列，不声称与NoEQ浮点计算成本完全相同。父20轮／4440更新单列；既有NoEQ E30控制不重训或重复预测。

固定final10／累计30冻结后自动完成原168000个物理样本×clean和六种practical的全部prediction/logp，再exec为独立CPU scorer连接truth。测试received IQ、SNR、信道、困难样本和分类头均不改。精确命令、环境／CWD、独占输出root、GPU容量选择和工件见[experiment.json](experiment.json)。每GPU跨owner总进程≤3、空闲≥12000MiB，实际GPU启动后读回；不停止或热改任何健康任务。

预先固定的否证比较是MMSE_MIX−同父NoEQ E30：low urban至少+0.5个百分点，clean和mid urban不下降，每个suburban不下降超过0.25个百分点。联合目标另要求相对同4L B00的clean净正、low>60%、每suburban与自身clean差距探索目标≤2个百分点，不能靠降低clean缩差。任何正式结果不回流混合概率、NMSE、SNR、延迟、loss、父模型、seed或重跑；不自动扩种子。本行只检验 A_s 的一个接收处理变化，不冒充新的A_g＋A_s四因子完成。

## 已执行验证与当前边界

本地33个协议／预算／来源／数据反例通过；实际训练adapter30个合成样本，15个逐值重放（6个NoEQ、9个MMSE24）与原完整core最大误差0。输入不变，批顺序重排一致，Torch／NumPy全局RNG不变；真实F结构900722参数、joint160和默认6类头的2个合成前向通过。新增检查0真实IQ／checkpoint／训练更新。旧Torch与NumPy2的ABI警告保留，已验证路径使用值拷贝／frombuffer，检查exit0且PASS，不修改环境。

发布前仅对新训练接线做一次有界P0/P1检查，不重复审查已通过的源候选或原物理core。单份archive一次传输SHA比较、普通N607预检和remote compile，launcher真实同契约checkpoint no-query smoke PASS立即继续。源角色／CP继承、非有限、预算、输出冲突或系统性技术失败保留本run产物并停止新派发，不自动重发；观察超时先核实原进程和产物。

该段保留原预登记与启动验证；完成状态以本文开头的实测结果为准。

新训练接线的唯一独立P0/P1检查PASS：18文件语法和登记一致、2220更新／284160 As预算复算、私有抽样／单residual／累计偏移／receiver_calls／同父ownEMA／七视图后独立评分均核对；审查者未读取真实IQ／CP／target或执行远端操作。
