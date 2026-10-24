# PAMF 全 U RDrop

LOCAL_VERIFIED，尚未启动训练。固定源设计3904aac和完整源结果248b3e0在本版生产实现前保存；55项相关检查与唯一一次生产P0/P1审查PASS。发布Git commit是实际代码身份，完整命令、环境/CWD、不可覆盖路径、GPU、停止规则和预期产物见experiment.json；固定矩阵和机制见hypothesis.json。

相对StepNoCap只让原拒绝U参与已有ordinary／mixed非物理半批／raw预测对的对称KL，沿用监督剂量0.5／0.25／0.25再除1.25及外部权重6。原0.95硬伪标签接受、argmax、接受样本全部CE/LS0.01/浅中层0.1和joint160跨RX0.05保持；被拒样本不增加硬CE或伪类对比，As独立KL保持0。六个既有学生logits复用，新增三次KL求值/step，6660步共19980次；无新教师/学生/既有头前向、新头或推理参数。额外KL并入完整pending U一次，再统一四导数与参数保护，原L及默认头联合训练不变。先保持三向原始梯度约束、无U/L幅度上限、原始U范数不扩张、CENTER_UNITS/clip5/nativeAdamW和当前moments L-only参考/实际FP32额外位移投影；每步四导数、相对NativeAs额外2L导数、每模型6660次参数参考。所有成本、显式量化回退和逐组保留单列，原参数坐标范数不是样本使用率，计时窗口含参考、真实joint优化及投影，不冒充纯新增开销。

源依据仅来自原完整V保存记录和固定L/U小批次。E20三seed81000源V中0.95接受精度99.895103%，拒绝5688中4270原预测正确（75.070323%）；不把这一估计冒充U真实准确率，也不降低硬标签门控。三原E20每seed一批L128/U256，原拒绝U数21/17/18；复用6U前向，21导数/7686包视图/9个丢弃真实physicalAdamW参数副本，CPU2耗时33.388986s。三seed均至少一组新增下降与拒绝KL梯度正对齐，全部实际三锚点和不扩张检查通过，最小非零组增量保留90.645862%，零回退。原模型/梯度/持久状态更新0，无V/Utruth/target/G读取。这是fresh-first-step源可行性，不是带历史moments的整轨迹、实测损失单调或目标结果；组和锚点相关，完整所有组均保留。

只加载匹配实际数据/物理ID/配置/payload/从零继承链的原同seed F21 E20/4440，各自own native10+20至E50/11100＝13L＋37LU；不加载StepNoCap/其他候选或原E30。复用VALIDATED_ONCE L6300/U56700/V27000，U TX隐藏，V仅诊断无模型/归一化/EMA/原型/头更新，无目标或G参与新F训练。

N607普通账户、CPU2/FP32TF32off，物理GPU5/6/7固定seed2701/02/03，每卡最多2条本任务train/predict，外部及pendingCUDA进程在内总数<3才启动、空闲>=10000MiB。逐seed等待StepGuard全部GPU预测完成且StepNoCap已派发，预留两前任RUNNING/pendingPID，保留所有健康旧任务。首个可用匹配卡做实际原E20 smoke，核实原L/U、固定硬接受、正的拒绝RDrop、四导数和真实FP32每非零组保留>=50%；通过后丢弃探针、重载原父立即继续。独立唯一CPU scorer在StepNoCap终态后运行。

每row固定最终E50默认学生/默认头后自动预测raw和固定DURA v1.0各clean＋六practical，168000/view，共14项决策固定后才独立truth-last。全三seed42cells保留逐view/seed、均值/SD/MacroF1/RX/TX/day/RXday/修复损伤/时间/缺项；匹配StepNoCap是唯一单因素参考，F20/F21双主控制固定。raw clean相对两者均至少+1pp，六景均值相对两者均严格>+1.5pp，整数门槛433348/504000、2180190/3024000；DURA单列参考，固定R69/266951/FP32TF32off/reuse=True graph=True native=False及原sourceLU/TXhidden/F-free语义，不套用旧F50时延，也不更改已指定部署版本。

历史目标暴露如实保留，包含已完成BranchGuard；设计固定时StepGuard和StepNoCap目标分数未读。不是新的独立盲测；不按目标调参、选父/seed/checkpoint、重排或选择性重跑。当前无本版GPU训练/目标结果，goal ACTIVE。源通过没有自动推出目标成功。


独立发布后核验：VERIFIED_RELEASED_CONTROLLER_SOURCE_SMOKE_PENDING，执行commit `e74a1e49ae39521e5c0cf87761a1a3f8c9705035`，实际正式GPU worker 0。源检查 PENDING；详见 queue_observation.json。验证以实际PID/CWD/完整命令/GPU/日志及参数更新记录为据，排队、探针或源前缀均不代表目标达成。保留健康任务，自动训练、raw与固定DURA七景、独立truth-last继续按原矩阵执行。


两前任方法最早完整E21–E30的三seed源执行观察已保存，见source_matched_E21_E30_observation.md。共13320条完整真实更新记录及对应已刷出参数记录，参数gzip在全部E50训练结束时统一关闭；本观察不替代最终日志关闭检查。原始梯度帽None三组计数均0，固定门控/六学生/As/四导数/实际参数保护保持；曝光接受率和参数回退不冒充U正确率或目标性能。没有新训练、目标访问、选模门槛或健康任务变化。


独立发布后核验：VERIFIED_REAL_E20_ALL_U_RDROP_SMOKE_AND_PARTIAL_SOURCE_RUNNING，执行commit `e74a1e49ae39521e5c0cf87761a1a3f8c9705035`，实际正式GPU worker 1。源检查 PASS；详见 first_source_start.json。验证以实际PID/CWD/完整命令/GPU/日志及参数更新记录为据，排队、探针或源前缀均不代表目标达成。保留健康任务，自动训练、raw与固定DURA七景、独立truth-last继续按原矩阵执行。


新增一次源域动量历史反证，见source_moment_history_plan.json。只用原三E20各一批L128/U256，比较相同当前参数／固定L梯度下两种AdamW历史；当前U置零，检查复制joint动量的参考是否看不到历史U影响。15个参数副本步骤全部丢弃，原模型和持久状态零更新；不读V、U真值或目标。仅当固定三seed源规则支持时考虑已固定的独立监督动量参考；当前训练和raw／DURA矩阵保持。该对照不是完整训练轨迹或目标性能证据。


原三E20动量历史对照已完成，固定源规则pass=True，科学状态SOURCE_HISTORY_LIMITATION_SUPPORTED，详见source_moment_history.md。27导数／7686源packet／15个丢弃参数副本步骤，原模型及持久状态更新0、V/U真值/target/G读取0。相同当前参数与固定梯度隔离出的历史差异不是实际训练轨迹、损失上升或目标性能证据；健康训练不变，未自动启动新GPU模型。


独立发布后核验：VERIFIED_REAL_E20_ALL_U_RDROP_SMOKE_AND_ALL3_SOURCE_RUNNING，执行commit `e74a1e49ae39521e5c0cf87761a1a3f8c9705035`，实际正式GPU worker 3。源检查 PASS；详见 source_activation.json。验证以实际PID/CWD/完整命令/GPU/日志及参数更新记录为据，排队、探针或源前缀均不代表目标达成。保留健康任务，自动训练、raw与固定DURA七景、独立truth-last继续按原矩阵执行。


## 完整源训练

VERIFIED_SOURCE_FINAL：每个模型严格从原同 seed E20 开始，原生10＋20段各2220／4440步，总计新增30轮／6660步，累计E50／11100；全部19980条闭合记录逐条核对。门控0.95、权重6、U LS0.01、原L／默认头联合训练保持，新增U原生As生成850500／物理学生1701000，无EMA、原型、U真值或G训练。NAMS累计学生U10602900，每model仅在已有U中joint捕获13320批／3402000条、6660次对比。新增原生As850500／物理学生1701000／rawAs0、style39960次／6660共享计划、RDrop／AsKL0／浅中层跨RX及CE归一化1.25均符合预登记。每模型实际6660次完整L anchor的U投影／范数帽1及逐组冲突／帽计数全部核实。父E20／新增30轮／累计经过时间分别保存，不把共享GPU墙钟视为独占小时。

固定最终学生自动完成raw与固定DURA各七景、独立truth-last，源证据不替代目标收益；已经完成的整组三seed先行报告不改变冻结的其余训练，本次源检查不读取目标分数。

原生As版实际有效物理CE学生视图为2×rawacceptedhalf，完整三seed计数在actual_adapter_SSL/U_native_As，生成和学生成本仍包含原raw拒绝样本；无增强视图门控/回退/额外As教师或新增损失，CE1.25、原raw接受集合保持。

分支保护版逐条核实四个原Lpair/四导数及每步额外两导数，累计每模型26640完整导数/13320额外导数，完整L与非物理/物理L锚点实际余弦均在固定容差内。原始U/L幅度上限已取消，保留原始U范数不扩张；L/接受集合/原生As及前向预算不变。非负点积不保证AdamW或目标单调。

全U非物理RDrop逐条核实6660次目标／19980次额外KL求值、原拒绝集合及mixed非物理角色，所有CE／伪类对比仍按原0.95接受mask；新增一致性对数／重复使用预测视图见actual_adapter_SSL/U_all_RDrop。无新模型／头前向，As独立KL0、全部原L与默认头保持。


## HPL_E20_AS_ALLRD 冻结 E50 的伪标签源域健康检查

三个固定最终学生各覆盖完整源 V27000 条，原默认头 eval 硬 argmax；无 GPU worker、参数／教师／原型／归一化更新、U 真值或目标读取，不改变健康训练或门控。此项只报告该方法的源域健康，不把源精度当作 U 实测正确率、完整本批测试成绩或目标达成。

父模型诊断复用原始固定 E20 的完整记录：`automation_reports/CV-SincNet/20261024-phase1-PAMF-HardPL-E20plus30-m6-r01/source_pseudo_quality.json`。本次没有重新计算父模型诊断。

| seed | E20 接受精度（%） | E50 接受精度（%） | E20 覆盖（%） | E50 覆盖（%） | E50 接受错误数 |
|---|---:|---:|---:|---:|---:|
| 2026092701 | 99.900216 | 99.945523 | 92.792593 | 95.181481 | 14 |
| 2026092702 | 99.868683 | 99.933346 | 93.074074 | 94.462963 | 17 |
| 2026092703 | 99.916428 | 99.953105 | 93.066667 | 94.774074 | 12 |

上表门控均为 0.95；0.98 对照及完整 TX／RX／day 分层保留在 source_pseudo_quality_E50_HPL_E20_AS_ALLRD.json。不依据微小源域变化选 checkpoint 或修改当前配方，完整预登记 raw／DURA 目标矩阵继续。


## 完整目标测试结果

ANALYZED：各方法三个seed，共3模型、42项raw／固定DURA七景决策均完成独立truth-last评分，六个匹配监督／普通PL对照模型完整复用。逐景／seed／SD／Macro-F1／RX／TX／day、修复损伤及曝光／成本见target_results.md和附属完整表。原raw双门槛尚未满足，研发目标仍ACTIVE。

HPL_E20_AS_ALLRD / raw clean：84.929166667% ± 0.489677676%（三个模型 seed 样本 SD）。
HPL_E20_AS_ALLRD / DURA-LU clean：85.045436508% ± 0.485486033%（三个模型 seed 样本 SD）。
HPL_E20_AS_ALLRD / raw six_mean：72.712400794% ± 0.212755666%（三个模型 seed 样本 SD）。
HPL_E20_AS_ALLRD / DURA-LU six_mean：77.317923280% ± 0.900048034%（三个模型 seed 样本 SD）。

不依据本结果热改健康旧任务或替换父来源。


HPL_E20_AS_ALLRD 固定匹配比较完成，见 HPL_E20_AS_ALLRD_matched_comparison.md；比较不替代原主目标或全六行完成状态，无目标反馈改配方。
