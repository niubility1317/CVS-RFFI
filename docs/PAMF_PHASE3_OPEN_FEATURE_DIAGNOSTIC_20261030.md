# PAMF-OW 未知身份结构冻结诊断

Run `20261030-phase3-star-open-feature-diagnostic-s2026092701-r01`。上一轮 M 已完成 source/target 独立评分，源 novel R_U45.396825%、AUROC82.919322%，跨 RX Q 的 A_K60.590278%、R_U64.583333%。本次区分开放特征是否保留未知设备身份结构，以及关系评分的失败是否伴随结构损伤。完整开放世界目标不缩减。

固定比较七种表征：原 F0 fused160，原 `star-open-episodic r02` 的 S0/S1/M 各 fused160 与 open64。原 F0 是 Tri-SIG-TIME SSL E50/11100；三臂来自同一原生祖先，各已固定 source-only2220步、累计13320，执行代码1029bc4f4ab86022c75ae0f3332b5dfafc1fe909。所有权重和默认头冻结，M仍为主身份，不根据本诊断换主、选择权重或调阈值。

只新读既有 G10P1 source novel3150条，19-15/19-16/7-1各1050，source RX1/3/4/6/8、2021-03-08/15/23。设备未参与F/OE拟合，但这些设备和源RX已被开发评价；不是严格未见RX或新设备盲测。L/V/OE、原Us、target U/Q、正式RX24-18都不新读，G不运行；不打开旧clean IQ。

所有输入为同一固定G10 MIX01星地received经原冻结DURA的缓存P1。复用`20261029-phase3-star-representation-s2026092701-r01/cache/novel`及其既有数据句柄，不重建/重验数据或重新抽信道。原缓存的F0/DURA祖先必须与三个固定parent一致。

实际checkpoint的两条合成G10观测首步smoke覆盖全部四个模型，通过后才打开源novel cache。每条同次前向导出unit fused160；新模型同次导出unit open64；确实为零的向量保留零，作为可能结构退化的诊断，不将其替换为随机向量。导出只含opaque ID、RX/day和特征，不含真实TX、K或标签。

全部四个NPZ、七种表征固定后，独立CPU评分器先为所有表征固定近邻索引，再打开唯一source novel真值。距离选择只用特征、opaque稳定次序、RX/day，不用TX/K。相同距离按opaque ID稳定次序处理，自身始终排除。

固定四个1NN检索条件：除自身外任意记录、不同RX、不同day、同时不同RX且不同day。逐真实TX等权宏命中率及逐RX/day/TX全部报告；没有合法候选保持N/A。不把这种使用同池参考记录的事后检索写成无标签分类或未来新类A_N。

同时报告cross-RX-and-day下同TX/异TX余弦均值差、中心化协方差的谱/有效秩/参与秩、TX/RX/day各单因素组间方差比例。TX只在独立评分端用于判断同类和描述统计；比例不当作可加的物理解耦或因果证明。常量向量的秩为0，方差比例无定义则N/A。

本轮无聚类K输入、无拟合/优化/新分类头/阈值校准、无目标预测/评分、无匿名映射。七项全部报告，不按返回结果取消控制。A_N/H_OW和完整Phase3成功均不由这组诊断认定。源几何好不能替代目标跨RX拒识、真正未知发现或未来匿名注册。

新增实际feature-forward行为4×3150=12600条（原F0一次、三臂各一次），合成烟测8条；G前向0、模型/原头/optimizer/阈值更新0。样本仅3150个既有物理记录，不因多表征导出增加。特征bank仅私有诊断产物，不部署、不进入Git。

N607单个GPU导出worker，允许与其他用户共卡，每次启动free≥4096MB，每卡最多新增2个本任务进程；本run仅新增1个。source export完成后自动接独立CUDAempty CPU scorer，BLAS线程1。不停止、修改或重启其他用户任务。固定新run/output root，技术失败保留而不自动重跑；未知提交状态先核实原PID/CWD/argv与输出。

本轮沿最小流程：实际代码commit、聚焦negative/接口检查、一次直接P0/P1 review、最小预登记、一次N607preflight/单归档比较/compile、实际checkpoint首步smoke、独立启动后进程/GPU/log增长、全部特征和邻居固定后truth-last。没有额外hash、签章、批准或报告门槛。具体commit、精确命令、环境/CWD、路径、资源与产物在本run登记维护。
