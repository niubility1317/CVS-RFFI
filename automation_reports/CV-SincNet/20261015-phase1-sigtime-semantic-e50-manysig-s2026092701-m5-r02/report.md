# 固定 SIG_TIME：语义身份关系 E50 技术恢复

本次修复原 r01 首步的 labels CUDA 与 RX/day CPU 设备不一致，并纠正 D0 对照的通用增强差异。三维 triplet mask 前将 RX/day 移至 labels.device 并统一 long，数学公式及监督不变。r01 D0 曾关闭 native generic IQ augmentor，而真实 A0/A-J 在 concat 之前执行它，因此 r01 D0 不作为严格的“仅关闭 practical”对照。该健康行仍不停止、不改动，完整保留为独立诊断；恢复版另训练匹配的 D0。本次不是新增损失候选或放宽科学失败门槛。

## 矩阵、来源与实际预算

|条件|本次处理|训练与评价身份|
|---|---|---|
|D0 无 practical|新建严格匹配控制，保留 native 通用增强|scratch E50；只关闭 practical，原 U／EMA 保持|
|D1 普通 practical CE|复用八位置 A0|原 final50 及完整预测／评分；本次 0 更新／0 重测|
|D2 旧正配对方向|复用八位置 A-J|按方法定义复用，未按 target 选择；本次 0 更新／0 重测|
|D3 单向 urban→clean 相对身份|修复后从零 E50|新独占 root、原 seed01、33L＋17 原 U、11100 更新|
|D4 双向相对身份|修复后从零 E50|同上；两方向半和并独立固定源强度|

新训练 150 epoch／33300 更新；两个复用控制共 100 历史 epoch、22200 更新单列。原 r01 无通用增强 D0 的 50 epoch／11100 更新及失败 D3/D4 首步开销均作为技术开销另记，不冒充有效匹配矩阵。新模型无任意父权重／optimizer／teacher 初始化；EMA 只来自各自 scratch 轨迹。三个新模型均为相同5580/720内部折；D1/D2来自实际训练 commit152d8921 对照。默认头、SIG_TIME、内部门控和参考响应全部保留联合更新，不拟合诊断头。

实际数据仍为 ManySig、源 RX1/3/4/6/8、day1/2/3、全部6TX；外层 L/U/V=6300/56700/27000，内部 L_train5580/L_dev720（seed2026101503），split/data392005，U TX隐藏，V仅健康。新训练流、source dev、校准与正式测试流保持分离；逐row六类seed、原控制路径、命令、环境／CWD、GPU、输出和owner唯一记录在 [experiment.json](experiment.json)。

## 不变的目标与校准

训练仍是 clean:practical=1:1 的原 concat CE，普通 low suburban/mid urban/low urban 采样。关系目标使用真实 L 的 joint160：同TX正参照／异TX负参照来自同RX/day，参照RX与anchor不同；hinge margin.1，满足即0，按anchor后全批平均。keys停止梯度；D3只更新urban anchor，D4两方向半和。物理可用性q只衰减额外关系项，全部合法satellite有CE，不加新U目标、bridge、课程或其他一致性。

直接复用 r01 的既定 A0 epoch5 源强度统计：720L_dev、2160原前向、0更新、模型和.grad未变，D3=.686371983770465、D4=1.1577435580713538。恢复版校验来源/角色/预算后原样复制，新增校准前向为0，不重算、不调强度、不按target反馈。原统计commit仍为e8f51edab，不改写成恢复发布commit。

新row保持6190200主前向预算，原D0为4769400；额外L clean EMA forward、原U门控与33＋17日程保留。默认cosine头6×160、scale30，完整863730／身份195041参数。IQ256、25MHz、native RMS后simulator input RMS，无output RMS；residual/post_sync、EQ关闭G=1、referenceSNR30、质量适配／相位跟踪／IQ补偿、reflect和24点数值延迟保持。正式clean＋六practical原VALIDATED_ONCE IQ保持，不改噪声或测试样本。

## 运行和完成规则

首个真实 A0/A-J 默认头无query smoke PASS立即继续；三条新训练在共享容量锁内按8卡实际所有owner／preCUDA占用分散，每卡最多3总实验进程、free≥12000MB。CPU2线程、workers0；原D0诊断GPU0健康进程保持。源校准复用与独立scorer使用空CUDA_VISIBLE_DEVICES。精确命令为发布CWD下 `python -u -m experiments.cvs_urban_semantic50.dispatch`；唯一run/output和N607 Python路径见登记。

每条新行 final50 冻结后自动预测全部七视图，预测固定后独立exec scorer连接truth。完成后先写final50 L_dev source决定，再读取target scores汇总。原r01 D0诊断不参与source决定或严格性能差值；其自动测试仍由原队列负责，不重复启动。它的训练成本仅以完成元数据单列。

原 native 路径在 concat 激活时先做通用增强，再创建两视图；无 concat 时在 preparation 后做通用增强。新版 D0 保留原 use_aug=True，只把 capture 改成绑定其下一次真实 L 模型输入，并验证形状／设备／dtype；完成这次 L 前向立即移除所有 hook，再执行任何 U／EMA 前向。实际 CE 和额外 EMA 预算使用同一真实 L 输入。D3/D4仍使用原严格 concat 对象捕获，不改变原增强算子或调用次数。

源推进条件不变：任一D3/D4，clean−D0≥0.5个百分点、low−D1≥1个百分点、urban−D1≥0、最差urban RX−D1≥−1个百分点。clean与low正式差值分别报告，不能相抵。单seed源资格通过也不自动扩seed；跨seed需预定独立确认。正式low≥60%及原探索幅度目标仍不是保证。

技术停止仅作用于本次owned输入／来源／图／非有限／预算／输出／launch失败，停止未启动行、保留全部产物、健康行自然完成，不操作无关进程，不自动重试。一次Git实际代码/config提交、一次archive比较／remote compile、一次独立P0/P1恢复差异审查和独立启动读回遵守原最小流程，不新增审批。

## 当前证据与缺项

本地8项检查PASS（6.717s）：关系合法性、单双向梯度、margin/U负例、默认头与gain边界、两个复用行禁止重训／重测、D0/D3/D4真实native前后两种通用增强位置／输入／CE／EMA图、CPU元数据跨设备mask、禁止旧D0作为匹配来源和源校准复用负例。修复前混合设备错误已用meta设备复现，修复后回归通过。一次有界恢复差异审查已通过；新增发现的 D0 通用增强差异已修复，原问题定向复核PASS，尚未发布r02。原D0已独立读回运行至11/50；不把该历史快照当实时完成。

当前没有D3/D4新训练收益、正式clean代价或跨seed重现证据。HQ的64.86%→70.97%是冻结源域降低新增噪声的受控效应，不是本次训练或目标成绩。Bridge-CE/S仍按最新附件留待语义源证据后独立比较；跨RX教师仍是独立未发布草稿。

## 实际发布与独立启动读回

VERIFIED：实际训练代码/config commit `63f6bd461834b4def6e3d7cfe4d8e8f00389b13e`，本地HEAD与远端branch OID独立匹配。一次archive传输比较VERIFIED、remote compile PASS，真实A0/A-J默认头无query smoke PASS（4前向）。原720L_dev强度统计直接复用，新增校准前向/optimizer更新均0；原统计commit e8f51edab保持。

dispatcher PID3070243；三row实际CWD/argv/PID、GPU及日志增长独立匹配：

|row|PID|GPU|已完成epoch快照|日志字节增长|
|---|---:|---:|---:|---:|
|D0-s2026092701|3070268|1|4|6444→27026|
|D3-s2026092701|3070345|2|2|6459→16746|
|D4-s2026092701|3070426|3|2|6459→16745|

实际resolved确认：三row均use_aug=True、原U/EMA=True、label33/pseudo17、strong_noise_std=.015、batch_neighbor、label_smoothing=.01、clip5；D0 sat disabled，D3/D4 concat_full。模型各863730参数。原r01诊断D0实际use_aug=False，保持运行，未纳入主比较。各worker约3.3至3.6GiB，CPU2线程、loader0；历史同类A0 E50为4593.410037s，仅作为资源估计，不承诺本轮耗时。其他owner随后使用剩余容量，独立快照所有卡总进程数仍在≤3内，没有停止/迁移任何健康行。

实际leo_practical导入来自本release的 `experiments/adv3b02_xuc/code/leo_practical/channel.py` 与 `residual.py`；EQ_off_G_equals_one=True、formal_test_changed=False。D3/D4首步同为14670合法triplet、128个anchor有参考；两方向开关与原定一致。预热期额外梯度为0符合计划，非零关系梯度和完整更新仍待epoch6及训练产物，不能将首步通过当作有效学习结论。

证据位于本地artifact `delivery_r02/`、`independent_initial/readback.json`、`independent_growth/readback.json`、`execution_evidence.json`、`launch_summary.json`、`independent_launch_evidence.json`。当前状态RUNNING，无新的固定七视图成绩、clean代价或跨seed结果；最终训练/预测/评分/机制/曲线仍待完成。

## 预热后接线与待冻结分析

独立只读核验三PID仍活跃、CWD/argv不变、日志继续增长，无traceback。以下是单次训练梯度快照，不能作为全程均值、因果收益或目标成绩：

|row|完成epoch|satellite CE范数|额外梯度范数|比值|两者余弦|W/gain额外梯度|
|---|---:|---:|---:|---:|---:|---|
|D0-s2026092701|28|0.000000|0.000000|0.000000|N/A|all0|
|D3-s2026092701|14|0.948449|0.087072|0.091804|0.370065|all0|
|D4-s2026092701|13|0.914069|0.077629|0.084927|0.202973|all0|

D3/D4在epoch6后存在非零额外特征梯度，原W/gain额外为0；静态源强度只将校准中位比定为.1，不要求每个训练批次恒定。原CE/U联合更新继续，完整clip/AdamW及参数更新归因仍待全日志。

冻结源分析已实现并登记为 `20261015-phase1-sigtime-semantic-source-manysig-s2026092701-m5-r01`，代码提交4ed88cf189c1486c1ff069a57d2831bbf8c90e01已push且独立OID匹配。5项检查PASS0.831s、compile PASS、一次P0/P1及原信道类修复定向复核PASS。分析将只在五个实际final50冻结后单独CPU运行，预算36722前向/0更新；尚未发布、尚无分析图或结果。D4 clean anchor在同次配对图中保留，假设性D0 satellite CE梯度明确区别于实际训练。

当前正式提升/clean代价/跨seed结论仍缺失；最近进度为28/14/13，不宣称预算已结束或已收敛。证据：local artifact monitor_03.json；独立源分析预登记报告见对应run。
