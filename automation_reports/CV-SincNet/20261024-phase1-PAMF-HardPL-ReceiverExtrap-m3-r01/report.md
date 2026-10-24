# 完整特征接收机外推

LOCAL_VERIFIED_NOT_LAUNCHED：110 项相关检查和一次独立 P0/P1 PASS；真实原 E20 检查尚待释放的匹配卡位，通过即丢弃探针、重载原权重继续。

三 seed 2026092701/02/03，各从原同 seed F21 E20/4440 独立执行原生10/2220＋20/4440，累计 E50/11100＝13L＋37LU。保留已验证 F0 物理数据契约 L6300/U56700/V27000，核对父来源/实际 cfg/payload；U TX 真值隐藏，V 只诊断、不更新模型/归一化/EMA/原型/头。N607 普通账户、CVS-RFFI 环境、FP32/TF32off、每 worker CPU2。命令、CWD、不可覆盖路径、GPU、停止规则和预期产物见 experiment.json。

源域固定网格 alpha=0/.25/.5/1，预先规定每 seed V 真值准确率≥99%且有效≥600，U 原伪类一致率≥99%且有效≥800、hardCE≥原1.25倍；仅 .25 通过。V 有效样本661/665/665，准确率100/99.699248/99.849624%；U898/902/903，原伪类一致率均100%，CE增至2.527858/2.852442/2.483095倍。U 真值正确率仍未知。较高损失仅说明这些固定源特征视图更难，不能证明真实信道保真或目标增益。已知 Joint/Complement 和更早目标结果；Path/ReliableAs/组合目标分数未读，本候选源依据和矩阵已固定。

以原 Joint6G 为配方，从原 E20 而非候选权重起步。前两次普通 U 已有 joint160 特征复用 sharedstyle 的 samepseudo/rawaccepted/differentRX 伙伴；有效位置 h'=h+0.25(h-hpartner)，保留完整签名/参考/主干。普通 U CE0.5 拆为 full0.25＋外推0.25，总 CE 分母1.25、LS0.01；无效伙伴保留原 CE 且差额精确为0，保留全部原接受样本。共享特征仍依参与项正常求梯度，不以 V 或 U 真值选伙伴。

保持 rawgate0.95、U权重6、六学生前向、ordinary/raw RDrop1、AsKL0、浅中层跨RX0.1/joint跨RX0.05、共享style、原L/defaulthead联合、完整L保护分组投影及范数帽1、nativeAdamW/clip5/decay；完整新增U梯度只进入guard一次，不保证AdamW步或目标性能单调。新增每步2次已有头调用，每模型6660步/13320调用/3402000条已有特征，实际有效≤2×accepted；无额外骨干/teacher/物理As/新头/推理参数。累计rawteacher2097900、studentU10602900、rawAs850500，逐seed分别报告父/新增/累计时间和实际有效计数，不称相同FLOP或独占GPU小时。

GPU5/6/7固定对应三seed，每卡本任务最多2条、计外部compute进程<3且free≥10000MiB。等待同seed ReliableAs全部GPU预测结束且PathReliableAs已派发，预留两侧pendingCUDA PID；CPU唯一scorer接组合终态。首可用匹配卡真实E20 smoke验证新CE对签名/参考/主干/头梯度均非零、2头512条、L768/U1536/rawteacher256/physicalU0/rawAs128/LPractical64、CE1.25/有效上限/完整L显式梯度；探针丢弃，正式重载原E20。两个同类系统性技术错误只停止新派发并保留健康任务；不按目标结果调参、重排或选择性重跑。

每row冻结最终E50学生后自动完成raw和固定DURA各clean＋六practical，168000/view、三seed42cell；整row预测固定后独立truth-last。报告全部逐seed/景/SD/MacroF1/TX/RX/day/RX×day/净纠错/成本，F20/F21同读出及预先固定Joint对照。raw clean相对两个控制均≥+1个百分点、六景均严格>+1.5个百分点才满足原目标，DURA单列。恢复器固定DURA v1.0 R69/266951、FP32/TF32off/reuse=True/graph=True/native=False，不用于新F训练，不沿用旧F50时延或替换默认部署。目标保持 ACTIVE。


## 发布后独立核实

VERIFIED：执行 commit `02979123053aa2b2d8d6f044ba3a497a3d484cd0` 的控制器存活，CWD及完整命令匹配。三行实际QUEUED，GPU训练worker为0，真实原E20检查仍PENDING，等待同seed ReliableAs全部GPU预测完成且组合已派发。独立核实当前六个ReliableAs/组合健康worker的PID/CWD/命令/GPU及日志增长。每卡最多两个本任务，现有任务保持；排队不代表训练或目标成功。


完整特征外推真实检查：新CE对signature/response/core/head的梯度均有限非零，数值见source_activation.json；有效外推仅作用于原接受合法伙伴，实际2次既有头调用512条已有特征，CE分母1.25、alpha0.25、全部原接受样本保留。无额外骨干/teacher/物理U/新头。三seed均有实际RXE计数，不能把排队或预期计数当作已运行。


## 真实原 E20 检查通过并实际启动三种子

VERIFIED：发布commit `02979123053aa2b2d8d6f044ba3a497a3d484cd0` 的真实原E20 GPU source smoke PASS。实际joint160两次捕获512条已有Upacket，跨RX有效anchors=478；实际签名sequence_pool辅助梯度范数=0.370686235且有限非零，原L768／U1536／rawAs128／Uphysical0／LPractical64、共享style计划、完整L guard参数梯度与显式期望相同及临时wrapper退出均符合。探针丢弃，正式行重新加载原E20。

三个seed已按GPU5/6/7启动并实际产生原生joint／guard源步骤；独立核实进程／CWD／完整命令／GPU及日志增长，逐行现有U复用计数见source_activation.json。只按原矩阵接替释放卡位；当前是训练前缀，不代表全部源训练结束或目标收益。训练、raw＋固定DURA七景及truth-last继续自动执行，goal ACTIVE。


## 完整源训练

VERIFIED_SOURCE_FINAL：每个模型严格从原同 seed E20 开始，原生10＋20段各2220／4440步，总计新增30轮／6660步，累计E50／11100；全部19980条闭合记录逐条核对。门控0.95、权重6、U LS0.01、原L／默认头联合训练保持，无U物理生成、EMA、原型、U真值或G训练。NAMS累计学生U10602900，每model仅在已有U中joint捕获13320批／3402000条、6660次对比。新增rawAs850500、style39960次／6660共享计划、RDrop／AsKL0／浅中层跨RX及CE归一化1.25均符合预登记。每模型实际6660次完整L anchor的U投影／范数帽1及逐组冲突／帽计数全部核实。父E20／新增30轮／累计经过时间分别保存，不把共享GPU墙钟视为独占小时。

固定最终学生自动完成raw与固定DURA各七景、独立truth-last，源证据不替代目标收益；已经完成的整组三seed先行报告不改变冻结的其余训练，本次源检查不读取目标分数。

完整特征外推每模型6660次目标、13320次既有头调用／3402000条已有特征，实际有效<=2×accepted，alpha0.25、fullCE0.25＋外推CE0.25、分母1.25，无效伙伴差额0，完整签名/参考保留。无额外骨干/teacher/物理U/新头。


## HPL_E20_JOINT_RXE6G 冻结 E50 的伪标签源域健康检查

三个固定最终学生各覆盖完整源 V27000 条，原默认头 eval 硬 argmax；无 GPU worker、参数／教师／原型／归一化更新、U 真值或目标读取，不改变健康训练或门控。此项只报告该方法的源域健康，不把源精度当作 U 实测正确率、完整本批测试成绩或目标达成。

| seed | E20 接受精度（%） | E50 接受精度（%） | E20 覆盖（%） | E50 覆盖（%） | E50 接受错误数 |
|---|---:|---:|---:|---:|---:|
| 2026092701 | 99.900216 | 99.938530 | 92.792593 | 96.403704 | 16 |
| 2026092702 | 99.868683 | 99.926564 | 93.074074 | 95.825926 | 19 |
| 2026092703 | 99.916428 | 99.949869 | 93.066667 | 96.044444 | 13 |

上表门控均为 0.95；0.98 对照及完整 TX／RX／day 分层保留在 source_pseudo_quality_E50_HPL_E20_JOINT_RXE6G.json。不依据微小源域变化选 checkpoint 或修改当前配方，完整预登记 raw／DURA 目标矩阵继续。


## 完整目标测试结果

ANALYZED：各方法三个seed，共3模型、42项raw／固定DURA七景决策均完成独立truth-last评分，六个匹配监督／普通PL对照模型完整复用。逐景／seed／SD／Macro-F1／RX／TX／day、修复损伤及曝光／成本见target_results.md和附属完整表。原raw双门槛尚未满足，研发目标仍ACTIVE。

HPL_E20_JOINT_RXE6G / raw clean：85.726785714% ± 0.641460398%（三个模型 seed 样本 SD）。
HPL_E20_JOINT_RXE6G / DURA-LU clean：85.790674603% ± 0.660973333%（三个模型 seed 样本 SD）。
HPL_E20_JOINT_RXE6G / raw six_mean：70.085515873% ± 0.468886486%（三个模型 seed 样本 SD）。
HPL_E20_JOINT_RXE6G / DURA-LU six_mean：77.897718254% ± 0.942587591%（三个模型 seed 样本 SD）。

不依据本结果热改健康旧任务或替换父来源。


HPL_E20_JOINT_RXE6G 固定匹配比较完成，见 HPL_E20_JOINT_RXE6G_matched_comparison.md；比较不替代原主目标或全六行完成状态，无目标反馈改配方。
