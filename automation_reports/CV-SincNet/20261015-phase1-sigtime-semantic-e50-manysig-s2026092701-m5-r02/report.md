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
