# 高置信伪标签的类别与接收机 CE 贡献均衡

LOCAL_VERIFIED_NOT_LAUNCHED：23 项目标函数及数学检查、119 项受影响流程检查、一次独立 P0/P1 检查全部通过；首个匹配释放卡位上的真实原 E20 检查 PENDING，通过即丢弃探针、重新加载原父权重继续。

三个 seed 为 2026092701/02/03，各从原同 seed F21 E20/4440 独立执行 native10/2220＋20/4440，累计 E50/11100＝13L＋37LU。仅复用已核实契约的原 E20，不加载候选权重。L6300/U56700/V27000 与 F0 未见 RX 测试契约不变；U TX 真值隐藏，V 仅诊断，无梯度或模型、归一化、EMA、原型、头更新。N607 普通账户、CVS-RFFI、FP32/TF32off、每 worker CPU2。命令、CWD、不可覆盖路径、GPU、停止规则见 experiment.json，执行 commit 在发布后核实。

源设计 e7c9eb60 已冻结，完整原 E20 U56700×3 的贡献 CV 下降约 22% 至 24%，权重边界和接受质量检查全部通过。已有完整 V 的最坏加权精度下界在 99.80% 以上，它是数学界，不能当作新实测加权精度或实际 U 正确率。完整源依据及固定公式见 source_balanced_pseudo_ce_summary.json 和 balanced_ce_followup_design.json。此前完整目标曝光保留；本设计在 RAS、组合、RXE、NativeAs 新目标分数读取前固定，不依据测试结果挑父权重、seed、checkpoint 或改变顺序。

在完整 NativeAs 配方上只改变接受样本的 CE 贡献：按当前 raw 接受集合的伪类与实际源 RX 计数，计算逆频数几何均值，接受样本平均归一化，再作有界凸混合。每批权重为 0.5 至 1.5，接受样本平均权重为 1，全部原接受样本保留；没有新标签、门控、历史计数或可训练权重。原 raw argmax/0.95、U 权重 6、六学生、原生 As 半批、LS0.01、ordinary/raw RDrop1、As KL0、浅中层跨 RX0.1、joint160 跨 RX0.05、共享 style、总 CE 分母 1.25、原 L/defaulthead 联合、完整 L guard 一次及 nativeAdamW/clip5/decay 均保持。

重用六个现有 logits，CE 配对系数为 ordinary0.5、mixed0.5、raw0.25，再除以 1.25；mixed 的两半分别沿用原 0.25×半批均值。仅把加权 CE 与原 CE 的差额加入完整 U 目标，KL/对比项不加权。每个接受样本在三组视图中的总 CE 系数为 1/N，接受平均权重 1 因而保持总贡献质量，但不保持损失数值或梯度方向。完整新 U 梯度只进入一次 L guard。临时 logits、权重上下文每次退出清除。

每模型新增 6660 步、U 曝光 1701000、学生 U 10206000，累计教师 2097900、学生 U10602900；As 生成 850500、物理学生视图 1701000，实际有效 As CE 为 2×rawacceptedhalf。无 raw 回退、额外 As 教师、骨干、既有头前向、新头或推理参数；增加六次 CE 算术/步，共 39960 次 CE 求值。单列源校准 170100 次 CPU 前向/118.410 s，新增 V 前向为 0，不称等 FLOP 或独占 GPU 小时。

GPU5/6/7 固定对应三个 seed，每卡最多两个本任务、计外部 compute 进程小于 3 且可用显存至少 10000 MiB。等待同 seed RXE 全部 GPU 预测结束且 NativeAs 已派发，预留两侧 pending CUDA PID。真实原 E20 检查核对原 NativeAs 的 L768/U1536、rawteacher256、As128/物理学生256/Practical192，以及权重边界、接受质量、原 mask、六 logits 重用、完整 L 显式参数梯度和临时 hook 清除；通过后丢弃探针并重载原权重。唯一 CPU scorer 接 NativeAs 终态。两个同类系统性技术错误只停止新派发，保留健康任务和全部产物；不按目标分数停止、调参或选择性重跑。

每 row 最终默认学生冻结后自动预测 raw 及固定 DURA 各 clean＋六 practical，168000/view，三 seed 共 42 cell；整 row 所有预测固定后独立 truth-last。报告逐 seed/景、SD、MacroF1、TX/RX/day/RX×day、净纠错及实际时间，保留 NativeAs 匹配比较和 F20/F21 两个控制。原 raw clean 对两个控制均至少高 1 个百分点、六景均严格高于 1.5 个百分点才达到目标，DURA 单列。DURA 固定 v1.0 R69、266951 参数、FP32/TF32off、reuse=True/graph=True/native=False，不用于新 F 训练，不套用旧 F50 时延或替换默认部署。目标保持 ACTIVE。


发布后独立核实：VERIFIED，执行 commit `0276faf2f9da7845be71e34cfce4a2d92e4e905c` 的控制器存活，CWD 和完整命令匹配；三行实际 QUEUED，新增 GPU worker 为 0，真实原 E20 检查 PENDING。等待对应 seed RXE 全部 GPU 预测结束且 NativeAs 已派发；独立核实两侧控制器及现有运行行的 PID/CWD/命令/GPU/日志增长。保留全部健康任务；排队不代表训练成功或达到目标。


真实原 E20 检查及首批启动已独立核实：VERIFIED，执行 commit `0276faf2f9da7845be71e34cfce4a2d92e4e905c`，检查 seed 2026092702，L768/U1536/rawteacher256/As128/物理学生256/Practical192，完整 L 显式梯度、joint 签名梯度和临时 hook 清除通过。探针丢弃，正式重新加载原 E20。目前实际 1 个正式 GPU worker 已有正源步骤，并核实 PID/CWD/命令/GPU/日志增长；其余仍排队。完整事实见 first_source_start.json；首个启动不能当作三种子全部启动或测试成功。


原生As真实检查：固定半批生成128，两个原mixed学生实际使用256条物理视图，rawAs回退0；原rawteacher256，无增强视图evalteacher或额外学生前向，原接受mask和CE1.25保持。原生As输入及非物理ordinary输入匹配，生成成本与有效As CE视图2×rawacceptedhalf分别记录；没有用排队或预期计数代替实际训练证据。


## 真实原 E20 检查通过并实际启动三种子

VERIFIED：发布commit `0276faf2f9da7845be71e34cfce4a2d92e4e905c` 的真实原E20 GPU source smoke PASS。实际joint160两次捕获512条已有Upacket，跨RX有效anchors=478；实际签名sequence_pool辅助梯度范数=0.370686236且有限非零，原L768／U1536／rawAs0／U As生成128／物理学生256／LPractical64＋U128、共享style计划、完整L guard参数梯度与显式期望相同及临时wrapper退出均符合。探针丢弃，正式行重新加载原E20。

三个seed已按GPU5/6/7启动并实际产生原生joint／guard源步骤；独立核实进程／CWD／完整命令／GPU及日志增长，逐行现有U复用计数见source_activation.json。只按原矩阵接替释放卡位；当前是训练前缀，不代表全部源训练结束或目标收益。训练、raw＋固定DURA七景及truth-last继续自动执行，goal ACTIVE。


## 完整源训练

VERIFIED_SOURCE_FINAL：每个模型严格从原同 seed E20 开始，原生10＋20段各2220／4440步，总计新增30轮／6660步，累计E50／11100；全部19980条闭合记录逐条核对。门控0.95、权重6、U LS0.01、原L／默认头联合训练保持，新增U原生As生成850500／物理学生1701000，无EMA、原型、U真值或G训练。NAMS累计学生U10602900，每model仅在已有U中joint捕获13320批／3402000条、6660次对比。新增原生As850500／物理学生1701000／rawAs0、style39960次／6660共享计划、RDrop／AsKL0／浅中层跨RX及CE归一化1.25均符合预登记。每模型实际6660次完整L anchor的U投影／范数帽1及逐组冲突／帽计数全部核实。父E20／新增30轮／累计经过时间分别保存，不把共享GPU墙钟视为独占小时。

固定最终学生自动完成raw与固定DURA各七景、独立truth-last，源证据不替代目标收益；已经完成的整组三seed先行报告不改变冻结的其余训练，本次源检查不读取目标分数。

原生As版实际有效物理CE学生视图为2×rawacceptedhalf，完整三seed计数在actual_adapter_SSL/U_native_As，生成和学生成本仍包含原raw拒绝样本；无增强视图门控/回退/额外As教师或新增损失，CE1.25、原raw接受集合保持。

贡献均衡版逐模型核实6660次目标/39960次额外CE求值，六原logits重用，权重0.5至1.5且每批接受质量守恒，全部原接受样本保留；NativeAs物理/教师/学生预算不变，无额外模型或头前向。实际边界、质量和最大剂量误差见U_balanced_CE。


## HPL_E20_AS_BCE6G 冻结 E50 的伪标签源域健康检查

三个固定最终学生各覆盖完整源 V27000 条，原默认头 eval 硬 argmax；无 GPU worker、参数／教师／原型／归一化更新、U 真值或目标读取，不改变健康训练或门控。此项只报告该方法的源域健康，不把源精度当作 U 实测正确率、完整本批测试成绩或目标达成。

父模型诊断复用原始固定 E20 的完整记录：`automation_reports/CV-SincNet/20261024-phase1-PAMF-HardPL-E20plus30-m6-r01/source_pseudo_quality.json`。本次没有重新计算父模型诊断。

| seed | E20 接受精度（%） | E50 接受精度（%） | E20 覆盖（%） | E50 覆盖（%） | E50 接受错误数 |
|---|---:|---:|---:|---:|---:|
| 2026092701 | 99.900216 | 99.907128 | 92.792593 | 95.711111 | 24 |
| 2026092702 | 99.868683 | 99.922252 | 93.074074 | 95.274074 | 20 |
| 2026092703 | 99.916428 | 99.949573 | 93.066667 | 95.481481 | 13 |

上表门控均为 0.95；0.98 对照及完整 TX／RX／day 分层保留在 source_pseudo_quality_E50_HPL_E20_AS_BCE6G.json。不依据微小源域变化选 checkpoint 或修改当前配方，完整预登记 raw／DURA 目标矩阵继续。
