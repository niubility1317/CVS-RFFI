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
