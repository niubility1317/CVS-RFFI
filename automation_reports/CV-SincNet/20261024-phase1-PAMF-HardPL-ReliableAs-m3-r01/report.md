# 可靠信道视图与原始输入回退

LOCAL_VERIFIED_NOT_LAUNCHED。77项相关检查和一次独立P0/P1PASS；真实原E20 source检查是首个释放且匹配seed/card的启动步骤，通过即重载原父权重继续。

固定HPL_E20_JOINT_RAS6G，三seed2026092701/02/03，原同seed F21E20独立native10+20到E50/11100、13L37LU。沿用F0 VALIDATED_ONCE L6300/U56700/V27000，严格核实实际source角色/物理ID契约、scratch继承、cfg/payload及原默认头；不加载原E30或前一候选。U TX隐藏，V只做无参数/归一化/EMA/原型/头更新的诊断，不使用target训练/选模。N607普通账户CVS-RFFI，FP32/TF32off、CPU2/worker，CWD PROJECT/releases/20261024-phase1-PAMF-HardPL-ReliableAs-m3-r01；完整命令、路径、GPU、stop rule、产物见experiment.json。一个实际代码commit、一个archive一次传输比较、远端compile、独立PID/CWD/cmdline/GPU/loggrowth核实，原健康任务保留。

方案在读取Joint/Complement/Path新target前由commit346b3fc2938e7b6fd52d803fca4db2e0267201a5固定，原设计及源view摘要附后。原E20 fixed U960/V720、18条source视图行、两namespace原生As三景混合显示：raw伪标签源V高纯，但扰动后模型同标签一致率约74%-77%、hardCE.61-.66；这不是U真实准确率下降。高置信同类As只保留全部U约43%-46%，因此回退保留原始样本，不能仅丢弃难视图。历史target暴露（含当前已读Joint结果）如实保留；当前Complement/Path未读target，不据结果改父、阈值、日程或矩阵。

保留固定Joint6G全部.95 rawPL/weight6/LS.01、六U共享同伪类跨RXstyle、ordinary/raw RDrop1/AsKL0、浅中层CRIC.1/joint160CRIC.05、CE分母1.25、原L/defaulthead原位联合、完整L梯度projection/normcap1、原nativeCENTER_UNITS AdamW/clip5/decay。无额外一致性或CE/KL权重。唯一变化是mixed半批：原生Practical生成As一次；同currentstudent在eval/noGrad下对该As求q，只有rawaccepted且maxq>=.95且argmaxq==rawpseudo时采用As，其余精确回退同条rawIQ。伪标签和accepted原集合不变，原acceptedU全部参与原六前向/全部原损失。hard_pl_no_uas=False真实声明UAs生成；旧Joint验证代理仅为核对其他字段恢复True，不隐瞒实际执行。

混合视图生成在现有普通2＋mixed2＋raw2学生顺序之前，额外teacher不带临时style/joint hook，无梯度、无EMA、恢复原trainmode，源smoke核实归一化buffer不变。两次mixed学生复用同一门控后的IQ；其他原始/普通/对比全部保留，完整签名/参考仍部署，无新头/参数。只用source物理metadata模拟，不读Utruth或Vtruth门控、原型或G训练。完整L guard仍只消费一次U梯度；显式参数梯度相等不保证Adam步、L损失或target单调。

每模型6660新增步/原U1701000，原生As生成850500、额外As eval teacher850500、学生U10206000。累计rawteacher2097900＋Asteacher850500＝2948400，学生U10602900；实际As used＋rawfallback＝generated850500，真正物理学生视图2*used，rawfallback单独计。生成与eval即使回退也有成本，父/增量/累计经过时间、实际采用/回退率、共享卡位成本均报告，不称同FLOP或免费增强。

首个真实原E20 smoke固定L128/六L768、U256/六U1536、UAs128生成/eval、原LAs64合计Practical192；teacherraw256+As128=384，精确门控/回退、原accepted保留、模式/归一化恢复、joint签名辅助梯度/浅中层/style、原CE/KL及完整L保护后参数梯度等于显式期望。探针丢弃、正式行重新加载原E20。

GPU5/6/7固定seed2701/02/03，每卡最多2本任务、计入外部最多3进程，派发前free>=10000MiB。只接替同seed Complement完成全部GPU预测的卡位，同时保留健康Path；两侧pendingCUDA PID都预留。首释放且容量足够的匹配seed/card做其原E20 smoke，临时smoke_runtime首row对齐、正式三row映射不变。CPU唯一scorer接Path终态后。两个相同技术失败停止新增派发，保留健康任务/产物；不按target停训/调参/重排/选择性重跑。

固定最终E50学生逐row自动raw及固定DURA各clean+六practical，168000/view，三seed42cell；整row全部prediction固定后独立truth-last。报告逐seed/景/TX/RX/day/RX×day、SD/MacroF1、净纠错/损伤、U实际采用/回退和teacher/学生曝光/成本。复用同读出完整F20/F21；原raw目标clean同时>=+1pp、六景同时严格>+1.5pp，DURA不能替代。G固定R69/266951、FP32/TF32off/reuse+graph/nativeFalse；原G的LU无TX标签/F-free约束保持，不参与新F训练，不套旧F50时延。goal ACTIVE。


## 发布后独立核实

VERIFIED：执行 commit `0ffe0ba43a6b2ead84db96ff5b98494f29550594` 的控制器存活，CWD 与完整命令匹配。三条 ReliableAs 行实际 QUEUED、训练 worker 0，原 E20 真实权重检查尚 PENDING；等待对应 seed Complement 全部 GPU 预测完成且 Path 已分发后释放第二卡位，固定 GPU5/6/7 对应三个 seed。已核实前序 Complement／Path 共 6 个健康进程的身份、GPU 和日志增长。排队不代表检查通过、训练启动或目标收益；全部 raw 与固定 DURA 七景测试和独立 truth-last 自动衔接。


## 首个真实权重检查及正式源训练

VERIFIED：首释放 seed 2026092702／GPU6 原E20检查PASS、探针丢弃且正式重载，完整L显式梯度、As谓词与精确raw回退、原raw接受集合、eval归一化与模式恢复全部核实。检查128条As实际采用45、回退83；额外teacher128、真实源Practical192、原学生U1536/L768。正式worker已产生53条原生As/joint/style/guard记录，PID/CWD/命令/GPU与日志独立核实。当前实际1worker、其余2row仍QUEUED；本记录是训练前缀和单批检查，不是完整采用率、完整训练或目标收益。源完整和目标测试均继续自动完成，goal ACTIVE。


## 真实 E20 权重检查与三 seed 实际启动

VERIFIED：执行 commit `0ffe0ba43a6b2ead84db96ff5b98494f29550594`，首释放 seed 2026092702／GPU6 原 E20 source smoke PASS。原 raw 接受集合保持；128 条实际 As 生成及额外 eval 中 used=45、精确 raw 回退=83；实际学生物理视图为2×used。教师 raw256＋As128，学生 U1536、L768、实际源 Practical192；门控谓词、eval 归一化状态及训练模式恢复、完整 L anchor 显式梯度、六次学生前向、CE/KL 剂量和临时 wrapper/hook 退出均核实。探针更新丢弃，正式重新加载原 E20。

三个 seed 已在 GPU5/6/7 实际产生原生 ReliableAs、joint、style 和 guard 步骤，PID/CWD/完整命令/GPU/日志增长独立核实。这里只确认源训练启动，完整 E50、raw／DURA 七景与独立 truth-last 尚未完成；目标仍 ACTIVE。


## 完整源训练

VERIFIED_SOURCE_FINAL：每个模型严格从原同 seed E20 开始，原生10＋20段各2220／4440步，总计新增30轮／6660步，累计E50／11100；全部19980条闭合记录逐条核对。门控0.95、权重6、U LS0.01、原L／默认头联合训练保持，新增U物理生成850500／额外eval850500，实际学生物理视图为2×used，raw回退为fallback，逐seed动态计数已核对；无EMA、原型、U真值或G训练。NAMS累计学生U10602900，每model仅在已有U中joint捕获13320批／3402000条、6660次对比。rawAs为实际fallback、累计teacher2948400（raw2097900＋As850500）、style39960次／6660共享计划、RDrop／AsKL0／浅中层跨RX及CE归一化1.25均符合预登记。每模型实际6660次完整L anchor的U投影／范数帽1及逐组冲突／帽计数全部核实。父E20／新增30轮／累计经过时间分别保存，不把共享GPU墙钟视为独占小时。

固定最终学生自动完成raw与固定DURA各七景、独立truth-last，源证据不替代目标收益；已经完成的整组三seed先行报告不改变冻结的其余训练，本次源检查不读取目标分数。


## HPL_E20_JOINT_RAS6G 冻结 E50 的伪标签源域健康检查

三个固定最终学生各覆盖完整源 V27000 条，原默认头 eval 硬 argmax；无 GPU worker、参数／教师／原型／归一化更新、U 真值或目标读取，不改变健康训练或门控。此项只报告该方法的源域健康，不把源精度当作 U 实测正确率、完整本批测试成绩或目标达成。

| seed | E20 接受精度（%） | E50 接受精度（%） | E20 覆盖（%） | E50 覆盖（%） | E50 接受错误数 |
|---|---:|---:|---:|---:|---:|
| 2026092701 | 99.900216 | 99.938343 | 92.792593 | 96.111111 | 16 |
| 2026092702 | 99.868683 | 99.914639 | 93.074074 | 95.455556 | 22 |
| 2026092703 | 99.916428 | 99.949731 | 93.066667 | 95.781481 | 13 |

上表门控均为 0.95；0.98 对照及完整 TX／RX／day 分层保留在 source_pseudo_quality_E50_HPL_E20_JOINT_RAS6G.json。不依据微小源域变化选 checkpoint 或修改当前配方，完整预登记 raw／DURA 目标矩阵继续。
