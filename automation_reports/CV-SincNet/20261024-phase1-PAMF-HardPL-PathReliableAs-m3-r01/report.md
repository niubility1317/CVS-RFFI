# 主干互补与可靠信道视图组合消融

LOCAL_VERIFIED_NOT_LAUNCHED：86 项相关检查、一次独立 P0/P1 PASS；首个释放且匹配 seed/card 的真实原 E20 检查尚 PENDING，通过即丢弃探针并重载原权重继续。

固定 HPL_E20_PATH_RAS6G，三 seed2026092701/02/03。每个模型从原同 seed F21 E20/4440 独立 native10/2220＋20/4440，累计 E50/11100＝13L＋37LU；不加载原 E30 或候选权重。数据复用已有 VALIDATED_ONCE 的 F0 物理契约，L6300/U56700/V27000；严格核实父状态角色、物理ID、scratch继承和实际cfg/payload。U TX真值隐藏；V只诊断，无模型/归一化/EMA/原型/头更新。N607普通账户/CVS-RFFI、FP32/TF32off、每worker CPU2，CWD PROJECT/releases/20261024-phase1-PAMF-HardPL-PathReliableAs-m3-r01。命令、不可覆盖路径、GPU、停止规则和预期产物见experiment.json。一个实际代码commit、一个archive一次本地远端比较、一次remotecompile；提交后独立核实PID/CWD/完整命令/GPU/日志增长，健康旧任务保留。

源依据分为两项：原 E20 U960/V720 原生As视图诊断显示 raw 高纯伪类不等于增强视图容易识别，固定 .95＋同类门控采用As，其余 exactraw回退以保留原样本；源诊断D既有E30的主干特征交原默认头时，固定V有效子集准确率从Joint90.5073%提高到Complement97.5892%和Path98.5434%，完整融合均99.8493%。这些是源域机制证据，不是U真值正确率或目标成功。已知Joint和此前目标结果，当前Complement/Path/ReliableAs目标分数尚未读取；本组合在其目标结果前固定，不改父、门槛或矩阵。

组合完全保留两个独立固定配方：Path普通U CE系数full0.25＋noSig0.125＋core0.125，总CE分母1.25；noSig从同次joint扣实际sequence_pool，core再扣response_projection乘response_gain.tanh；已有头增加4次调用，完整签名/参考仍参与部署和原完整视图训练。ReliableAs只改变原mixed半批输入：原生As生成一次，currentstudent eval/noGrad得到q，raw已接受且q置信度≥.95且argmaxq==raw伪类才用As，其余精确同包raw回退。raw接受集合、硬伪类、所有六学生前向保持。额外As teacher先于临时style/joint/path hook，恢复训练模式，不更新归一化。

其余保持 .95rawPL、权重6、LS.01、ordinary/raw RDrop1、AsKL0、浅中层跨RX0.1/joint160跨RX0.05、六前向共享accepted同伪类跨RXstyle，原L/defaulthead原位联合、完整L anchor分组投影/范数帽1、原nativeCENTER_UNITS AdamW/clip5/decay。多继承共同祖先只执行一次Joint原六U loss，完整L guard只消费一次组合U梯度；不重复CE/teacher/学生损失。不新增头、参数、额外学生骨干、EMA、原型、Utruth、target或G训练；不保证AdamW步、源L损失或目标单调。

每模型新增6660步、U1701000、学生U10206000；As生成850500及额外As teacher850500，Path既有头26640次/6804000条已算特征。累计raw teacher2097900＋As850500＝2948400，学生U10602900；实际used＋fallback＝850500，物理学生视图2×used。生成/额外eval即使回退仍计成本，逐seed记录实际采用率/回退/父与增量及累计墙钟，不称相同FLOP或独占GPU小时。

真实原E20检查同时验证两个机制：L六前向768、U六前向1536、rawteacher256＋As128、源Practical原L64＋U128＝192；As精确谓词/回退/原接受集合、eval归一化和模式恢复、Path四既有头调用1024条已有特征、主干CE不走exclusive签名/参考而原头/主干梯度非零、完整joint签名辅助梯度、原CE总剂量、完整L guard显式梯度相等与临时hook退出。探针丢弃，正式行重载原E20。

GPU5/6/7固定对应三个seed，每卡最多两个本任务且计外部最多3 compute进程；派发前free≥10000MiB。只接同seed Path全部GPU预测结束的卡位，且ReliableAs已分发；Path和ReliableAs两侧pendingCUDA PID预留。首个符合条件的匹配卡做真实检查，smoke临时首row对齐，正式三seed/card分配保持。唯一CPU scorer接ReliableAs终态，两个相同系统性技术错误只停止新增派发并保留健康任务/产物；不按目标分数停训、调参、重排或选择性重跑。

最终E50学生逐row自动测试raw及固定DURA各clean＋六practical，168000/view，共三seed42cell，整row全部决策固定后独立truth-last。报告全部逐seed/景/SD/MacroF1/TX/RX/day/RX×day、纠错损伤、接受/视图/成本，并复用F20/F21同读出完整对照。固定源设计的Joint/Path/ReliableAs三项邻近比较全部保留。原raw clean对两控制均≥+1pp、六景对两控制均严格>+1.5pp，DURA不能替代主目标。G固定用户DURA v1.0 R69/266951、FP32/TF32off/reuse=True/graph=True/native=False，G不进入新F训练，不套用原F50时延；默认部署定义保持，goal ACTIVE。


## 发布后独立核实

VERIFIED：执行 commit `2ffbfcb8d04c2abd94ffc24f7f00fac0bebee529` 的控制器存活，CWD 与完整命令匹配。三条 PathReliableAs 组合行实际 QUEUED、训练 worker 0，原 E20 真实权重检查尚 PENDING；等待对应 seed Path 全部 GPU 预测完成且 ReliableAs 已分发后释放第二卡位，固定 GPU5/6/7 对应三个 seed。已核实前序 Complement／Path 共 6 个健康进程的身份、GPU 和日志增长。排队不代表检查通过、训练启动或目标收益；全部 raw 与固定 DURA 七景测试和独立 truth-last 自动衔接。


固定四项配方报告工具已通过6项整数差、交互量、完整矩阵前置检查与覆盖保护测试。只有Joint/Path/As/组合全部完成三seed raw+DURA七景且元数据/物理契约一致后才读取四份summary，输出三个邻近配对差及combined−Path−As＋Joint的实际配方交互量；不代替主目标。比较固定因素、命令和产物见factorial_reporting_plan.json；未完成时只WAIT，不阻塞主目标报告或读取部分新结果。


## 真实 E20 权重检查与三 seed 实际启动

VERIFIED：执行 commit `2ffbfcb8d04c2abd94ffc24f7f00fac0bebee529`，首释放 seed 2026092702／GPU6 原 E20 source smoke PASS。原 raw 接受集合保持；128 条实际 As 生成及额外 eval 中 used=45、精确 raw 回退=83；实际学生物理视图为2×used。教师 raw256＋As128，学生 U1536、L768、实际源 Practical192；门控谓词、eval 归一化状态及训练模式恢复、完整 L anchor 显式梯度、六次学生前向、CE/KL 剂量和临时 wrapper/hook 退出均核实。探针更新丢弃，正式重新加载原 E20。

三个 seed 已在 GPU5/6/7 实际产生原生 Path CE、ReliableAs、joint、style 和 guard 步骤，PID/CWD/完整命令/GPU/日志增长独立核实。这里只确认源训练启动，完整 E50、raw／DURA 七景与独立 truth-last 尚未完成；目标仍 ACTIVE。

同时核实 Path core/head 梯度及 exclusive 签名/参考旁路，四次既有头调用/1024条特征、CE分母1.25保持；源训练逐步Path和As计数共存。


## 完整源训练

VERIFIED_SOURCE_FINAL：每个模型严格从原同 seed E20 开始，原生10＋20段各2220／4440步，总计新增30轮／6660步，累计E50／11100；全部19980条闭合记录逐条核对。门控0.95、权重6、U LS0.01、原L／默认头联合训练保持，新增U物理生成850500／额外eval850500，实际学生物理视图为2×used，raw回退为fallback，逐seed动态计数已核对；无EMA、原型、U真值或G训练。NAMS累计学生U10602900，每model仅在已有U中joint捕获13320批／3402000条、6660次对比。rawAs为实际fallback、累计teacher2948400（raw2097900＋As850500）、style39960次／6660共享计划、RDrop／AsKL0／浅中层跨RX及CE归一化1.25均符合预登记。每模型实际6660次完整L anchor的U投影／范数帽1及逐组冲突／帽计数全部核实。父E20／新增30轮／累计经过时间分别保存，不把共享GPU墙钟视为独占小时。

固定最终学生自动完成raw与固定DURA各七景、独立truth-last，源证据不替代目标收益；已经完成的整组三seed先行报告不改变冻结的其余训练，本次源检查不读取目标分数。

主干互补CE每模型26640次既有头调用／6804000条已有特征，普通0.5分为full0.25＋noSig0.125＋core0.125，CE分母1.25不变；完整签名／参考部署保持，无额外骨干／新头／推理参数。


## HPL_E20_PATH_RAS6G 冻结 E50 的伪标签源域健康检查

三个固定最终学生各覆盖完整源 V27000 条，原默认头 eval 硬 argmax；无 GPU worker、参数／教师／原型／归一化更新、U 真值或目标读取，不改变健康训练或门控。此项只报告该方法的源域健康，不把源精度当作 U 实测正确率、完整本批测试成绩或目标达成。

| seed | E20 接受精度（%） | E50 接受精度（%） | E20 覆盖（%） | E50 覆盖（%） | E50 接受错误数 |
|---|---:|---:|---:|---:|---:|
| 2026092701 | 99.900216 | 99.865348 | 92.792593 | 96.270370 | 35 |
| 2026092702 | 99.868683 | 99.858020 | 93.074074 | 96.518519 | 37 |
| 2026092703 | 99.916428 | 99.919249 | 93.066667 | 96.318519 | 21 |

上表门控均为 0.95；0.98 对照及完整 TX／RX／day 分层保留在 source_pseudo_quality_E50_HPL_E20_PATH_RAS6G.json。不依据微小源域变化选 checkpoint 或修改当前配方，完整预登记 raw／DURA 目标矩阵继续。
