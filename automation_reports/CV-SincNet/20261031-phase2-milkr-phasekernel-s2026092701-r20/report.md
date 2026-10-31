# MILKR 完整相位分布核角点验证

R21 固定完整 Tri-SIG-TIME SSL 和 DURA v1.0。R20 四角点失败产物保留，不扩展、不选择其其他源配置重跑。新方法检验均值/标准差压缩是否丢失轨道分布形状，C4 和 rank64 保持预定预算。

同一 received 记录做四个确定性相位旋转；恢复路径先对原包执行一次 G，raw 路径不执行 G。每相位提取真实 F z/t/f/pa 和三路112 signature，FFT只在原包或实际G输出算一次。source raw/G同phase配对训练标准 PairedCCA64，共6300物理记录、两epoch12600配对记录、50400计算型phasepairs；无TX监督metric。source五RX/TX-held折排除两TX全部RX/epoch/phase，仅source标签选择120个K1核规则，最终全合法L拟合一次。只有普通冻结encoder权重进入target，无source样本/特征库/类条件统计。F已见源六TX，不把encoder-held TX说成完整基座未知类。

每个物理record保持四个编码特征，两个record的核为全部16个phasepair RBF/linear核值均值；不先平均特征再RBF。保留轨道在核空间的集中程度，不做self-diagonal归一化或跨query统计。每条记录仍为一个support核行、一个标签，K不变。带宽只用当前old support的平均phase距离；K1源规则和ref6类数缩放，K>1当前物理support条件LOO选择120核/正则项，无R20 Fisher/outer选择。明确带宽/几何仍在完整support上定义，非无偏nested CV。

源相对收益不设门槛，合法性/数值和真实F/G无query smoke必须通过。first RX1-1/P01/MIX01/seed2026092701 K1/20×N2/20、四S/Q，共B8+C16=24行，全query补集。S0/S1严格分开、共享对应support状态的Q0/Q1，所有注册类竞争。全部24预测固定后独立truth-last。每角点两S Q0≥原版+5个百分点且Q1>Q0、旧新均值≥原版−1点才同模型补12KN；最终16KN全部通过才其余六RX。角点均值不与历史16格均值混比。

源GPU4/targetGPU0，启动前核实容量、最多每卡两个本owner训练/预测，不动其他owner。完整commit/命令/环境/CWD、checkpoint/data契约、路径/停止条件见experiment.json；技术失败保留产物，不盲重试。

唯一独立 P0/P1 审查 PASS。新核 PSD/相位顺序不变/逐query独立/K1物理标签数、phase encoder配对保存重载、K2正则方程一致及旧orbit、K1/源隔离、非法输入共15项相关检查通过。真实checkpoint无query smoke为发布第一步，PASS后继续。
