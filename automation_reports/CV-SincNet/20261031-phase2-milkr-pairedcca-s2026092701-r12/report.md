# MILKR 源配对 CCA 共同特征空间

预登记。完整Tri-SIG-TIME SSL基座与DURA v1.0冻结；方法、命令、环境/CWD、数据/模型来源、路径、owner、GPU和stop rule见experiment.json。

R14仅在合法source L6300物理记录×既有模拟ep91/92的12600配对观察学习参数。八个原特征块分别采用正则CCA：边缘协方差加0.01 trace/d，inverse-root白化，交叉协方差SVD，两路投到相同rank，再补零到原维度、逐块unit。raw和DURA各自有冻结线性编码器。严格S0/Q0只raw编码器；S1/Q1只真实G后使用restored编码器；不引入raw/G混合输入。R12当前K标签nested物理支持留出head不变。目标只加载冻结参数，不携带源样本/特征缓存。

源RX留出选择rank16/32/64/96。每次投影以及R13源控制的bridge训练均完全排除heldRX的两个模拟epoch。heldRX内每类选52个物理记录，support前K≤20，query固定位置20至51共32/class，无physical overlap。五RX×两episode×K1/5/10/20×四路。source旧/新各三类，F已见所有六类，不能称真新类泛化。

固定源门槛：四路平均H较sourceR13至少+0.5pp，两个S的Q0均不低于R13超过0.5pp，两个S的Q1−Q0均至少+0.5pp。满足者取源四路均H最大，平局小rank；没有rank通过即source-only科学停止，目标0行。通过才用全部sourceL拟合选定参数checkpoint，启动首RX1-1 P01/MIX01 seed2026092701的全部16KN/四路单arm80行。R13和原版同case各80行复用。全部预测固定后独立truth-last。

目标验收仍为两个S各自Q0较原版≥+5pp、Q1高于Q0，Q0旧/新macro≥原版−1pp，逐KN完整列出。通过仅允许固定方法其余6RX确认，不能以singleRX先导宣布goal完成。失败不扩展、原root不覆盖、不据query调参。源GPU4、目标GPU0一worker，实际余量检查，其他owner不动。

本候选唯一独立P0/P1审查PASS。配对旋转合成held数据对齐、逐query独立、保存重载逐位一致及错误path/rank拒绝测试PASS；原nested与协议非法输入拒绝检查PASS。真实checkpoint source-only smoke为launcher第一步，PASS即继续源留出筛选。
