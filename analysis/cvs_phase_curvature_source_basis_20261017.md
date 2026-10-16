# 下一唯一假设：输入相位曲率是否存在可复现的 RX 相关分量

2026-10-17。仅提出一个 source-only 依据检验；已实现登记，尚未启动，不是已有效 A_g，不授权目标访问或新增训练预算。完整联合目标 ACTIVE，F＋跨 RX 同 TX／默认头联合训练／E20 保持。

## 为什么检验这一点

孤立 AWGN 三匹配 E20 已关闭：noise 相对 raw clean＋0.952381pp，但相对 ordinary clean−0.476190pp、F1−0.356864pp；clean 总净修复12中 TX2贡献19，其他TX合计−7。六practical准确率均正增量，不能代替clean独立收益或证明跨RX共同不变性。固定20至36dB／全覆盖方案不抬SNR、改范围／权重、clip／删尾／重抽、换seed/RX救跑，也不事后晋级ordinary控制或TX2子群。

唯一待检验假设：在去除常相位和线性CFO影响的观测统计中，存在同一RX、跨TX和day可复现的非恒定相位变化。统计若主要依赖TX、day或零散包，就没有足够依据把相位扰动作为接收／观测变化。即使PASS，也只支持进一步身份保护检验，不能认定已分离RX振荡器或新Ag/DG有效。

## 与旧路线的边界

旧commonphase20是每包常相位uniform[−.37,.37]rad，已E20失败，不改角度重跑。已有phase_curvature_identity是在六个复杂FIR后加入可训练延迟曲率残差，属于旧结构路线；本次不新增／重开该模块。已有频偏、周期20 covariance、谱形、DC、IQ circularity等固定失败保持关闭，不改旧窗口/lag/seed或把旧统计换名晋级。这里只问原IQ的常相位／线性CFO不变的非周期20输入统计是否具有独立source证据，不在失败的cycle模型上拟合新的修补参数。

## 最小源依据检验

只读原VALIDATED_ONCE source cache中的精确fitL5040（RX3/4/6/8×day1/2/3×6TX、72格各70包），RX1在统计前排除。L6300缓存及未用V/历史特征会反序列化，U仅opaque角色ID、无IQ/TX标签。0新IQ视图／0checkpoint／0NN／0更新／0target，不改变接收IQ、NoExplicitCFO或必要内部逐包补偿。

固定使用原参考范围80:160中的全部相邻三点，78个位置；不做循环回绕、重采样或lag20重复假设。对原复数IQ z，计算相位曲率观测 c[n]=z[n+1]^2·conj(z[n])·conj(z[n+2])，其相位抵消公共常相位和线性CFO。为保留全部点，用 c/(abs(c)+1e-12（Complex128计算）) 的实虚两路作为观测，逐包／逐cell均值和离散程度全部报告；稳定项1e-12与Complex128精度在任何新源统计前固定，不依据源或目标效果调节，不屏蔽低能量点／包。这个观测同时包含已接收波形、滤波和TX/RX作用，不把它直接称为RX phase-noise参数。

只用源L标签组织TX×RX×day均值，不拟合任何身份／RX分类器。对每个held TX/day，用该组四RX均值的中心化位移作为实际RX相对观测；用其余5TX×2day估计四RX位移，预测held组。与零RX位移基准比较完整72个cell向量的平方误差、每RX和每heldTX/day正负变化；保留全部包／位置／cell。固定探索条件：pooled误差至少减少10%，每RX均不恶化，18个TX/day组至少15个正收益。任一FAIL即关闭该相位依据，不换统计、lag、窗口、TX/day/RX、阈值或seed救跑；不会自动启动增强或E20。

若PASS，仍需另行明确物理合理且轻度的IQ算子与范围，并在原默认头下验证身份保持；不能从一个统计向量恢复完整接收机、加入新loss或直接进入target。范围／算子目前未定，文献不会替代本源证据。

## 文献核对与不采用的路线

[WiSig官方采集与处理说明](https://cores.ee.ucla.edu/wisig/overview/)描述了256点preamble及非同步RX采集；因此本计划使用分组统计，不假定各RX相同sig_i是逐点配对的同一波形。官方说明只支持数据来源和限制，不证明相位曲率一定是RX变化。

[Huang等，AAAI2026](https://ojs.aaai.org/index.php/AAAI/article/view/37005/40967)将多种时域变换、对齐与目标无标签适应放在同一系统，不能把总成绩当作孤立增强的DG证据；本次不采用片段重排、window/time warping、prototype损失或target适应，也不把其“模拟多径”的文字等同真实practical。

当前不是实验预登记或发布门槛新增层；实现前须按既有最小实验流程登记唯一source-only统计矩阵，之后直接运行，不另设审批。
