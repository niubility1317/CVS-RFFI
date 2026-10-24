# 高权重伪标签与监督分支同时保护

LOCAL_VERIFIED_NOT_LAUNCHED：40 项分支／投影检查及 133 项受影响流程检查通过。独立训练 P0/P1：PASS。真实原 E20 GPU 检查等待首个匹配释放的 BalancedCE 卡位；检查通过后丢弃探针、重载原父权重直接继续。

源设计 2743f9aa 先于 NativeAs 目标读取：原 E20 三 seed 各一个 L128/U256 源步，在同一导数上同时约束完整 L、非物理 L 和物理 L，可保留至少99.744189%的完整L cap2整体U范数并移除9个实质分支冲突。该旧 Gram 算术结果只支持源域几何假设，不冒充新 QR 训练或目标收益；新数值实现待真实 GPU 检查。实现中近线性相关的监督头边界测试触发旧法失败，现用薄 QR 和小矩阵 SVD 枚举至多8个 active set，实际参数dtype检查通过后选最小改动候选，避免正规方程平方条件数。pinv截断仍1e-10但作用于线性矩阵；约束、实际余弦1e-6、cap2、tiny-L平方范数1e-12及原源规则保持。没有任意归零回退或结果后网格调参，技术差异完整记录在 numerical_implementation_note.json。

相对固定 NativeAsCap2，只改变 U 梯度方向算子。完整 L 不变，复用原L hook四对输出：非物理L为0.5ordinary＋0.25nonphysicalmixed＋0.25raw（含原RDrop），物理L为0.25physicalmixed（AsKL0）。计算完整L/U及两个L分支共4次导数，比原cap2多2次/步、每模型13320次。用原显式参数梯度桥接接回 native backward；保持rawargmax/0.95/U权重6/六学生/原生As/raw回退0/LS0.01/ordinary与raw RDrop1/AsKL0/浅中层跨RX0.1/joint160跨RX0.05/共享style/CE1.25/原L/defaulthead联合/nativeAdamW/CENTER_UNITS/clip5/decay。无额外教师、学生或头前向，无新推理参数。非负梯度点积不保证AdamW步、各项损失或目标性能单调。

只加载原同seed F21 E20/4440，实际角色、物理ID、配置、payload及完整从零继承来源核实后，各自独立native10/2220＋20/4440，累计E50/11100＝13L＋37LU；不用候选权重或原E30。seed2026092701/02/03，原L6300/U56700/V27000及F0未见RX物理集不变。U真值隐藏，实际U正确率UNKNOWN；V仅诊断，无模型/归一化/EMA/原型/分类头更新，无目标或G参与新F训练。每模型累计rawteacher2097900、学生U10602900，新增As850500、物理学生1701000、有效接受As视图2×rawacceptedhalf；全部原接受样本保留，生成/前向成本包含拒绝样本。额外导数与共享GPU经过时间单列，不称相同计算成本。

N607普通账户、CVS-RFFI、FP32/TF32off、每worker CPU2，CWD为PROJECT/releases/20261024-phase1-PAMF-HardPL-BranchGuard-m3-r01。完整命令、路径、资源和停止规则见experiment.json。GPU5/6/7固定三个seed，每卡最多两个本任务训练/预测，计外部compute进程小于3且空闲至少10000MiB才派发。等待同seed BalancedCE所有GPU预测完成且Cap2已派发，预留两侧pendingCUDA PID；唯一CPU scorer接Cap2终态，不停止或热改现有健康任务。首卡真实原E20检查验证L768/U1536/rawteacher256/As128/物理学生256/rawAs0/Practical192、完整L显式梯度、4个Lpair、4次梯度加1次joint检查导数、三个非微小锚点、U≤2L且至少一组U>L、所有临时context退出。两个相同系统性技术错误只停止新派发并保留健康任务和产物；不按目标性能停训、调参、重排或选择性重跑。

每row最终E50默认学生/默认头冻结后，自动预测raw及同固定DURA v1.0各clean＋六practical、168000/view，三seed共42cells；整row的全部14项决策固定后独立truth-last。保留逐seed/view、均值/SD/MacroF1、TX/RX/day/RX×day、净纠错和实际时间。主匹配固定NativeAsCap2检验新增分支约束，次匹配NativeAs cap1同时含范数帽变化；不混作单因素。主控制F20/F21：raw clean对两者均至少+1pp，六景平均对两者均严格>+1.5pp，分别至少433348/504000和2180190/3024000正确才达目标；DURA参考单列。DURA固定R69/266951/FP32TF32off/reuse=True graph=True native=False，G原训练sourceLU/TXhidden/Ffree；不用于新F训练、不套用旧F50时延、不换部署默认。

NativeAs正式目标84.831944/72.264385与DURA84.944444/77.428439已知；本轮按此前固定源路线实现，数值修复来自合成边界失败。BalancedCE/Cap2目标未读。早先目标暴露如实保留，不作新的独立盲测声明，不从目标挑父模型、参数、seed或checkpoint。当前没有本候选训练或目标结果，goal ACTIVE。


发布后独立核实：VERIFIED，执行 commit `61b942a4d922d41ff235707534169074293ec565` 的控制器存活，CWD 和完整命令匹配；三行实际 QUEUED，新增 GPU worker 为 0，真实原 E20 检查 PENDING。等待对应 seed BalancedCE 全部 GPU 预测结束且 Cap2 已派发；独立核实两侧控制器及现有运行行的 PID/CWD/命令/GPU/日志增长。保留全部健康任务；排队不代表训练成功或达到目标。


真实原 E20 检查及首批启动已独立核实：VERIFIED，执行 commit `61b942a4d922d41ff235707534169074293ec565`，检查 seed 2026092702，L768/U1536/rawteacher256/As128/物理学生256/Practical192，完整 L 显式梯度、joint 签名梯度和临时 hook 清除通过。探针丢弃，正式重新加载原 E20。目前实际 1 个正式 GPU worker 已有正源步骤，并核实 PID/CWD/命令/GPU/日志增长；其余仍排队。完整事实见 first_source_start.json；首个启动不能当作三种子全部启动或测试成功。

真实分支保护检查同时核对完整L/非物理L/物理L锚点、四Lpair、四导数加joint检查1次及实际dtype约束；每步额外两次L导数，无额外前向，完整L只应用一次，临时context退出。


分支同时保护真实检查：逐组U范数≤2L、负点积投影、一次完整L显式梯度均符合，至少一组真实U>L，每步额外两次L分支导数，无额外前向。固定半批生成128，两个原mixed学生实际使用256条物理视图，rawAs回退0；原rawteacher256，无增强视图evalteacher或额外学生前向，原接受mask和CE1.25保持。原生As输入及非物理ordinary输入匹配，生成成本与有效As CE视图2×rawacceptedhalf分别记录；没有用排队或预期计数代替实际训练证据。


## 真实原 E20 检查通过并实际启动三种子

VERIFIED：发布commit `61b942a4d922d41ff235707534169074293ec565` 的真实原E20 GPU source smoke PASS。实际joint160两次捕获512条已有Upacket，跨RX有效anchors=478；实际签名sequence_pool辅助梯度范数=0.370686237且有限非零，原L768／U1536／rawAs0／U As生成128／物理学生256／LPractical64＋U128、共享style计划、完整L guard参数梯度与显式期望相同及临时wrapper退出均符合。探针丢弃，正式行重新加载原E20。

三个seed已按GPU5/6/7启动并实际产生原生joint／guard源步骤；独立核实进程／CWD／完整命令／GPU及日志增长，逐行现有U复用计数见source_activation.json。只按原矩阵接替释放卡位；当前是训练前缀，不代表全部源训练结束或目标收益。训练、raw＋固定DURA七景及truth-last继续自动执行，goal ACTIVE。


## 完整源训练

VERIFIED_SOURCE_FINAL：每个模型严格从原同 seed E20 开始，原生10＋20段各2220／4440步，总计新增30轮／6660步，累计E50／11100；全部19980条闭合记录逐条核对。门控0.95、权重6、U LS0.01、原L／默认头联合训练保持，新增U原生As生成850500／物理学生1701000，无EMA、原型、U真值或G训练。NAMS累计学生U10602900，每model仅在已有U中joint捕获13320批／3402000条、6660次对比。新增原生As850500／物理学生1701000／rawAs0、style39960次／6660共享计划、RDrop／AsKL0／浅中层跨RX及CE归一化1.25均符合预登记。每模型实际6660次完整L anchor的U投影／范数帽2及逐组冲突／帽计数全部核实。父E20／新增30轮／累计经过时间分别保存，不把共享GPU墙钟视为独占小时。

固定最终学生自动完成raw与固定DURA各七景、独立truth-last，源证据不替代目标收益；已经完成的整组三seed先行报告不改变冻结的其余训练，本次源检查不读取目标分数。

原生As版实际有效物理CE学生视图为2×rawacceptedhalf，完整三seed计数在actual_adapter_SSL/U_native_As，生成和学生成本仍包含原raw拒绝样本；无增强视图门控/回退/额外As教师或新增损失，CE1.25、原raw接受集合保持。

分支保护版逐条核实四个原Lpair/四导数及每步额外两导数，累计每模型26640完整导数/13320额外导数，完整L与非物理/物理L锚点实际余弦均在固定容差内。U范数帽2，L/接受集合/原生As及前向预算不变。非负点积不保证AdamW或目标单调。


## HPL_E20_AS_BG2 冻结 E50 的伪标签源域健康检查

三个固定最终学生各覆盖完整源 V27000 条，原默认头 eval 硬 argmax；无 GPU worker、参数／教师／原型／归一化更新、U 真值或目标读取，不改变健康训练或门控。此项只报告该方法的源域健康，不把源精度当作 U 实测正确率、完整本批测试成绩或目标达成。

父模型诊断复用原始固定 E20 的完整记录：`automation_reports/CV-SincNet/20261024-phase1-PAMF-HardPL-E20plus30-m6-r01/source_pseudo_quality.json`。本次没有重新计算父模型诊断。

| seed | E20 接受精度（%） | E50 接受精度（%） | E20 覆盖（%） | E50 覆盖（%） | E50 接受错误数 |
|---|---:|---:|---:|---:|---:|
| 2026092701 | 99.900216 | 99.907074 | 92.792593 | 95.655556 | 24 |
| 2026092702 | 99.868683 | 99.922255 | 93.074074 | 95.277778 | 20 |
| 2026092703 | 99.916428 | 99.941813 | 93.066667 | 95.477778 | 15 |

上表门控均为 0.95；0.98 对照及完整 TX／RX／day 分层保留在 source_pseudo_quality_E50_HPL_E20_AS_BG2.json。不依据微小源域变化选 checkpoint 或修改当前配方，完整预登记 raw／DURA 目标矩阵继续。


## 完整目标测试结果

ANALYZED：各方法三个seed，共3模型、42项raw／固定DURA七景决策均完成独立truth-last评分，六个匹配监督／普通PL对照模型完整复用。逐景／seed／SD／Macro-F1／RX／TX／day、修复损伤及曝光／成本见target_results.md和附属完整表。原raw双门槛尚未满足，研发目标仍ACTIVE。

HPL_E20_AS_BG2 / raw clean：84.526984127% ± 0.164983350%（三个模型 seed 样本 SD）。
HPL_E20_AS_BG2 / DURA-LU clean：84.629761905% ± 0.165988202%（三个模型 seed 样本 SD）。
HPL_E20_AS_BG2 / raw six_mean：72.481746032% ± 0.423863687%（三个模型 seed 样本 SD）。
HPL_E20_AS_BG2 / DURA-LU six_mean：77.032936508% ± 0.652258630%（三个模型 seed 样本 SD）。

不依据本结果热改健康旧任务或替换父来源。


HPL_E20_AS_BG2 固定匹配比较完成，见 HPL_E20_AS_BG2_matched_comparison.md；比较不替代原主目标或全六行完成状态，无目标反馈改配方。


HPL_E20_AS_BG2 固定匹配比较完成，见 HPL_E20_AS_BG2_vs_HPL_E20_JOINT_AS6G_matched_comparison.md；比较不替代原主目标或全六行完成状态，无目标反馈改配方。
