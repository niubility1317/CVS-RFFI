# MILKR 冻结函数先验与同记录query融合先导

用户已明确允许query同记录raw与DURA输出融合，并保留仅DURA输出对照；support仍严格S0 raw／S1 G，不混入另一处理视图或增加物理K。科学协议已先加入本轮新run例外，旧实验/其他owner定义保持。新版Q0=raw、QG（机器Q1）=仅G、QF（机器Q2）=同记录融合。最终逐格目标仍为两S Q0较匹配原版Q0≥+5个百分点，并按用户新口径要求QF>Q0；QG完整单列，QF与原QG差注明路径改变，不冒充同输入配对。

R26 保持完整Tri-SIG-TIME SSL、DURA v1.0、原分类头冻结，C4moments/sourceCCA64固定。新raw读出直接使用冻结head的函数先验b0(x)：old列为0.5·(meanC4 softmax旧head−1/oldcount)，new列0，MILKR拟合当前SUP残差Y−b0并使用不惩罚的残差截距。不是R25的概率几何：核仅用原RF signature几何，head知识直接保留到输出函数。新神经训练/改头为0。源old4/new2先去掉heldnew两head列再softmax，F曾见源6的边界披露；target用sourcecontract.classes与合法registeredold表映射head顺序，实际六旧类顺序已核实相同。

融合规则事前固定：同一冻结SUP状态分别算raw/G全注册类分数、到SUP的最近RF距离及top2分数间隔。只有G距离严格更小且G间隔严格更大时，QF为两路分数等权均值；否则为raw。没有query真值、真实old/new角色、批统计、权重/阈值更新或类配额。权重只取0/0.5，raw始终保留。QF计算需要同记录8次F和一次G（两路各C4），Q0和QG对照也保留；G只对原received执行一次，无重抽信道/第二次观测。

source合法L6300物理记录、既有ep91/92共12600配对；五RX/TXheldCCA均排除heldRX及两TX全部RX/两epoch。源K1规则120个按Q0/QF四主路径H、更大ridge/固定序选择，QG完整记录；ref6 lambda随当前注册类数/6缩放。K>1真实三物理折OOF选120个RF核/正则配置，各折只用训练SUP估尺度/系数，再全SUP重拟合。F仍见源六TX，源old4/new2/三景不等于目标old6/new20/六景，不称新盲测。最终checkpoint只普通CCA权重，无source样本/特征库/类条件统计上传。

source合法/数值/实际checkpoint无query smoke后，首RX1-1/P01/MIX01/seed2026092701四角点K1/20×N2/20。B12+C24=36行，所有SUP状态在读取query前固定；每个state只读一种SUP路径。raw每行分数/距离临时缓存，仅与同物理ID、同case的G分数融合，不改变模型；全部36预测固定后独立truth-last。原版Q0/QG同case历史复用。每corner两S raw≥原版+5点、QF>raw、raw旧新均值guard≥原版−1点才补剩余12KN，最终all16后才六RX；均值不能替代逐格。

sourceGPU4/targetGPU0，启动前核实实际显存、最多每卡两个本owner新训练/预测，不动其他owner。完整commit/命令/环境/CWD、来源/data句柄、路径和停止条件见experiment.json。最小独立P0/P1审查、真实checkpointsmoke、一次archiveSHA/remotecompile、实际PID/GPU/log增长和独立评分保持；技术失败保留，不盲重试。

唯一独立P0/P1审查PASS；函数prior残差正规方程/截距、类别表映射、融合无标签/逐条独立、真实物理OOF、mockworker三path行对齐/严格SUP/缓存清空及既有teacher/orbit/非法输入共18项相关检查通过。真实F/G/head checkpoint无query smoke为launcher第一步，PASS后直接源校准及36行先导。
