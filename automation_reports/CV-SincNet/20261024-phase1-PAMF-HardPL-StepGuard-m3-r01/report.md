# 高权重伪标签与实际参数更新保护

REGISTERED，尚未启动。源域规则已经通过，生产路径87项检查通过，等待一次独立训练P0/P1和首个真实原E20 GPU检查。完整命令、环境、路径、资源、矩阵、停止规则见experiment.json；参数见hypothesis.json及hard_pl_step_guard.py。代码身份由发布时Git commit和runtime.json固定。

普通raw PL在完整源V的0.95门控后精度约99.9%，U实际正确率仍UNKNOWN。本版沿用户指定高权重＋监督机制路线，只对AdamW后的实际额外U参数更新增加保护。原始梯度非冲突并不能保证经过AdamW后仍非冲突；固定3个E20源批次的12个副本更新显示该问题。新的预登记源算子消除27处实质冲突，最小完整增量保留率99.991374%，未量化回退，原模型更新0。它只验证三个新建优化器首步，不是完整续训或目标提升证据。

相对BranchGuard，四次导数、六个U学生、所有L/U损失、0.95门控、权重6、原生As、LS0.01、RDrop1、AsKL0、浅中层跨RX0.1、joint跨RX0.05、共享style、CE/1.25和cap2均保持。每步在原生physical clip5之前复制当前模型参数与AdamW动量，执行一次仅完整L梯度的参考更新；真实joint更新及动量只执行一次，然后只对相对L参考的实际FP32增量施加三L锚点、逐组范数不扩张约束。固定4次量化尝试不满足时精确回退该组L参考并计数，不掩盖更新量损失。保留真实joint动量，因此只约束当前一步相对参考的一阶增量；不保证监督损失、全轨迹或目标准确率单调。

只加载原同seed F21 E20/4440并核实实际数据权限与从零继承链，own native10＋20到E50/11100＝13L＋37LU，不加载候选或原E30。三个固定seed在GPU5/6/7，每卡最多两个本任务train/predict，计其他owner及pendingCUDA，至少10000MiB空闲。保留健康任务，唯一CPU scorer在BranchGuard终态后。每模型额外6660次仅参数优化器参考，无额外前向或导数/新头/推理参数；已有BranchGuard额外两次L导数成本仍保留，报告实际时间、完整更新保留率和回退数。

既有VALIDATED_ONCE源L6300/U56700/V27000；U TX隐藏，V仅诊断，无模型/归一化/EMA/原型/头更新，无target/G用于新F训练。每row默认E50学生/默认头冻结后raw＋固定DURA v1各clean＋六practical、168000/view，全部14项预测固定后独立truth-last，三seed共42cells。报告逐seed/view、均值/SD/MacroF1、RX/TX/day/RXday、净纠错和时间。主匹配BranchGuard、次匹配Cap2（同时包含两种保护因素），F20/F21双主控制。原raw目标分别至少433348/504000和2180190/3024000正确；DURA单列，不代替raw目标。G固定R69/266951/FP32TF32off/reuse=True graph=True native=False，原LU/TX隐藏/F-free权限保留，不套用旧F50时延、不改变部署默认。

源算子及其门槛在Cap2目标值读取前固定；登记时Cap2及此前历史测试已知，全部曝光如实保留，不作新独立盲测声明。本版按固定源证据实现，不按目标修改配方、选父/seed/checkpoint、重排或选择性重跑。当前没有本版正式结果，goal ACTIVE。


生产实现 87 项相关检查 PASS，唯一一次独立训练 P0/P1审查 PASS。真实原E20 GPU检查仍是launcher第一步，通过后立即丢弃探针并重载原父训练；当前尚未发布或启动。


独立发布后核验：VERIFIED_REAL_E20_STEP_GUARD_SMOKE_AND_ALL3_SOURCE_RUNNING，执行commit `6459bc989343622564e232bbe0f67cd7479946d4`，实际正式GPU worker 3。源检查 PASS；详见 source_activation.json。验证以实际PID/CWD/完整命令/GPU/日志及参数更新记录为据，排队、探针或源前缀均不代表目标达成。保留健康任务，自动训练、raw与固定DURA七景、独立truth-last继续按原矩阵执行。


首轮真实参数更新已核实：三个固定 seed 均至少完成新增阶段的首个完整源 epoch（累计≥E21）。逐条检查当前已刷出的完整 native 损失记录及实际FP32参数更新记录，三组非微小监督锚点约束和不扩张检查均通过。组级量化回退按固定规则记录，没有热改配方、门槛或接受mask。完整逐组回退／零增量／保留率见 source_first_epoch_observation.json；整体范数用原参数坐标，不能代表各组重要性或物理量保留。

计时澄清：参数日志的seconds以及早期source_activation的reference_and_projection_seconds，窗口包含L参考、真实joint物理裁剪/AdamW及投影，截止日志序列化之前，不是纯额外开销，也不包含模型前向/反向。完整训练实际耗时待三seed源训练结束后按匹配记录报告。当前只是源前缀，不代表训练结束、监督损失单调或目标达标。


已预登记 source_uncapped_step_plan.json：仅在原E20三固定源批次验证去掉原U/L幅度帽、保留三梯度方向约束与实际参数更新保护。完整phaseC源记录显示cap2仍频繁截断U梯度，不能直接推断实际训练收益。每seed五个可丢弃optimizer副本，总15步，复用27次导数/7686包视图，无原模型更新或新前向/目标读取。固定源门槛：每个非零组实际增量至少保留50%、每seed至少一组保护后增量较cap2SG增加5%、至少移除1个实质冲突、全部实际约束与不扩张检查。失败如实保留，不调门槛或网格。当前GPU训练不变，无新候选自动启动。


无幅度帽源算子43项检查及唯一一次源P0/P1审查PASS，八模块传输与15个可丢弃副本更新预算已核对。按e9ddffd8预登记规则执行原E20小源矩阵，真实checkpoint无query检查后立即继续；不改变现有GPU训练，不自动启动新候选。
