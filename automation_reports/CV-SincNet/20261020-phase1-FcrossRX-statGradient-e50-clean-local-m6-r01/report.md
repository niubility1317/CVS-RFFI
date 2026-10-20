# F跨RX：标准化统计量梯度的固定三种子实验

原SupCon在实际浅层time_down/中层t2.pool的均值、标准差描述子上，用当前L批次统计量标准化，但反向时停止统计量梯度。此次只改该导数；相同输入下描述子、对比掩码、温度0.2、正样本数常数、损失值和默认头输出与原基底完全相同。STAT_GRAD保留真实均值/标准差导数，NORM_ONLY保持原梯度方向、将其描述子梯度范数匹配到相同特征下STAT_GRAD的范数。后者不代表整网参数范数、实际AdamW位移或不同训练轨迹范数相同。

动机来自代码中的统计停止梯度和已记录的源弱TX混淆，不依据新目标结果。完整导数会去掉辅助描述子梯度的共同平移分量，以及标准差未触底通道的共同缩放分量。此数学性质不证明接收机因素被消除或泛化必然提升。新机制全类等权，不加分类器、投影、额外网络前向、训练记忆或部署运行统计。

固定矩阵为STAT_GRAD、NORM_ONLY各2026092701/02/03，共六行；匹配原BASE三种子复用。各行从零50轮＝33L＋17U、11100更新，保留三路SIG_TIME/统一Sinc/旧门控/ownEMA/cos/RDrop1/LS.01/普通增强，默认分类头联合训练。两次RDrop复用同一次实际增强IQ。新SC只用于有TX标签L，不读取U真值；源V仅默认头和无拟合几何诊断。无星地增强，逐行固定final50后立即clean预测、独立CPU truth-last评分。所有三种子已固定，不套用源域扩种子门槛。

固定主判断：两个弱TX各自三种子平均配对差>0且各至少2/3为正；弱两类均差三种子全正；整体均差≥0；其余每TX均差≥−0.25个百分点；最差RX均差≥−0.5个百分点。梯度方向的额外归因单独比较STAT_GRAD−NORM_ONLY：弱均差>0且至少2/3为正。两组全部结果均报告，不按目标择优或调参；低性能不停止健康任务。均值、样本SD、正差数、Macro-F1、逐TX、最差RX/day和成本完整保存。

数据角色/物理ID复用既有READY/EXACT_MATCH/VALIDATED_ONCE：L6300/U56700TX隐藏/V27000；clean目标168000，RX/day/TX及seed与原BASE一致。正式模型没有checkpoint继承，真实原BASE01CP仅用于合成no-query检查后丢弃。CPU/CUDA float32/64数学检查、gradcheck、恒定/触底通道、前向完全相同、14项配置权限负例、实际原两次L前向/Sinc辅助梯度/默认头主梯度/EMA无辅助调用/导出边界均PASS。唯一一次直接P0/P1审查另存review.json。

本机普通账户、native Python3.10/PyTorch2.10cu128、FP32/noTF32、线程2/workers0。GPU0总训练/预测worker上限3，计入其他owner；派发前RAM空闲≥4.5GiB；独立CPU评分上限1。不可覆盖新release/run；实际代码/config以Git commit唯一标识，一次archive复制比对、compile和PID/CWD/参数/GPU/log增长核对。精确命令/CWD/路径别名、逐行矩阵、GPU、stop/artifacts见experiment.json。停止规则：来源、角色、配置、metadata或输出冲突、非有限及预算/工作进程技术失败时停止新派发，保留健康进程和全部产物，不自动重跑。

预期逐行source冻结/标准化梯度全步观测/clean预测/独立评分，以及六行完整日志汇总、配对分析和报告。当前仅检查通过，尚无本轮目标性能结果。研究目标保持ACTIVE/UNMET。

实际执行提交：`7b5532e9ff3845c23c7c229385ff5763254d31c3`，已推送并核对远端。发布归档一次复制比较VERIFIED，新release编译PASS；固定六行、controller真实PID/参数/CWD和持续更新的progress均已核实。当前GPU训练尚未派发：可用系统内存低于既定4.5GiB门槛，controller健康等待资源，其他进程保持。不能把队列启动当成训练完成或性能提升；后续沿同一PID/root核实，不重复启动。

资源允许后，首种子的STAT_GRAD与NORM_ONLY均已开始GPU训练。已独立分辨Windows虚拟环境launcher与实际Python子进程，核对两真实worker的PID/命令/CWD/GPU0及逐步日志增长，实际从零50＝33L＋17U/默认头联合/无星地配方一致；其余四行仍按原固定队列。source_launch_verified.json保存聚合证据。实际初始206个参数/buffer张量完全相同；已读取全部当时可用前两轮源记录，存在微小预热轨迹差异，原生参数仅candidate_id/output_dir不同。新辅助项此时权重0，不能把该差异归因于机制；来源尚未定位，原样保留，不据此改配方或重跑。未读取新目标结果。
