# 高权重伪标签与无幅度帽更新保护

REGISTERED，尚未启动。固定源设计d1b15e5f之后实现，75项本地相关检查PASS，等待唯一一次生产P0/P1与首GPU真实原E20检查。完整命令、环境/CWD、不可覆盖路径、资源、停止规则、矩阵与产物见experiment.json；固定机制与原始源证据见hypothesis.json。代码身份由发布Git commit/runtime.json固定。

相对StepGuard只取消raw U/L幅度帽2，明确None，仍按原三锚点最近方向锥投影U，完整L不变，并保留raw U范数不扩张及tinyL处理。实际参数保护继续用同一步前参数/当前AdamW动量的L-only参考、同CENTER_UNITS/physicalclip5/真实joint状态单次更新及FP32三锚点增量保护。默认头/全部L/U损失、rawargmax/0.95/权重6/六学生/原生As/rawfallback0/LS0.01/RDrop1/AsKL0/浅中层跨RX0.1/joint160跨RX0.05/sharedstyle/CE1.25不变。四导数/每步额外2L导数/一个L-only参数参考更新的预算与SG相同，无额外教师/学生/头前向或推理参数；SG相比原NativeAs的额外训练成本仍存在。实际总耗时、逐组保留率及显式回退单列，参数窗口包含真实joint优化器，不冒充纯额外开销或独占GPU小时。

固定原E20三个源批次验证采用27导数/7686包视图/15个可丢弃参数副本optimizer更新，每非零组最小实际增量保留90.946575%，各seed至少一组较带帽SG保护后增量增加5%，移除27个明显冲突、零量化回退。原模型持久状态更新0，无V/U真值/目标/G读取。该源单步证据不是完整轨迹、监督损失单调或目标性能提升。原幅度帽频率不是样本拒绝率；原0.95接受集合保持。

只加载实际数据权限/物理ID/配置/payload/从零继承链已核实的原同seed F21 E20/4440，再各自own native10+20，到E50/11100＝13L＋37LU，不加载SG/BG或原E30。既有VALIDATED_ONCE L6300/U56700/V27000，U TX隐藏，V只诊断无模型/归一化/EMA/原型/头更新，无target或G用于新F训练。

N607普通账户/CPU2/FP32TF32off，GPU5/6/7固定seed2701/02/03，每卡最多2个本任务train/predict，计外部及pendingCUDA进程小于3、空闲至少10000MiB。每seed等待BG全部GPU预测完成且SG已派发，预留两侧pendingPID；首个匹配可用卡做真实原E20 source smoke，核对原L/U/defaulthead/四导数/实际None/raw不扩张/三raw锚点及实际FP32每非零组保留≥50%，通过即丢弃探针、重载原父直接继续。唯一CPU scorer排在SG终态后，不停止或热改健康旧任务。

每row最终E50默认学生/默认头冻结，自动预测raw和固定DURA v1.0各clean＋六practical、168000/view，全14项决策固定后独立truth-last；3seed共42cells，保留逐seed/view、均值/SD/MacroF1、RX/TX/day/RXday、净纠错、缺项与时间。主匹配SG仅raw幅度帽变化；次匹配BG同时含实际参数步保护，不作单因素。F20/F21双主控制：raw clean均至少+1pp、六景均值均严格>+1.5pp，整数门槛433348/504000、2180190/3024000；DURA单列参考。G固定R69/266951/FP32TF32off/reuse=True graph=True native=False，sourceLU/TXhidden/F-free权限保留，不套用旧F50时延、不改变部署默认。

历史测试曝光如实保留。设计冻结时BG一行已评分但指标未读，完整矩阵未结束，SG无目标结果；本变体由完整源范数帽统计和固定源实验支持，不按目标成绩改配方、挑父/seed/checkpoint、重排或选择性重跑。不是新的独立盲测，当前没有本版训练或目标结果，goal ACTIVE。


生产实现75项相关检查PASS，唯一一次生产P0/P1审查PASS。None贯通实际原U投影、有限日志指标、smoke和两段累计，保留旧cap2默认。发布后首GPU真实原E20检查通过即丢弃探针并重载原父继续；当前尚未启动。


独立发布后核验：VERIFIED_REAL_E20_STEP_NO_CAP_SMOKE_AND_ALL3_SOURCE_RUNNING，执行commit `147875ade725c293c1415001f13c15ae3fd929fc`，实际正式GPU worker 3。源检查 PASS；详见 source_activation.json。验证以实际PID/CWD/完整命令/GPU/日志及参数更新记录为据，排队、探针或源前缀均不代表目标达成。保留健康任务，自动训练、raw与固定DURA七景、独立truth-last继续按原矩阵执行。


首轮真实参数更新已核实：三个固定 seed 均至少完成新增阶段的首个完整源 epoch（累计≥E21）。逐条检查当前已刷出的完整 native 损失记录及实际FP32参数更新记录，三组非微小监督锚点约束和不扩张检查均通过。组级量化回退按固定规则记录，没有热改配方、门槛或接受mask。完整逐组回退／零增量／保留率见 source_first_epoch_observation.json；整体范数用原参数坐标，不能代表各组重要性或物理量保留。

计时澄清：参数日志的seconds及source_activation的optimizer_and_projection_window_seconds，窗口包含L参考、真实joint物理裁剪/AdamW及投影，截止日志序列化之前，不是纯额外开销，也不包含模型前向/反向。完整训练实际耗时待三seed源训练结束后按匹配记录报告。当前只是源前缀，不代表训练结束、监督损失单调或目标达标。


补充源覆盖诊断与固定反证矩阵已登记，见 source_pseudo_coverage.md 和 source_rejected_rdrop_plan.json。E20门控0.95仍保持，原拒绝集合只提出无硬标签RDrop候选；当前六条GPU训练不变。先用三原E20各一批L128/U256、CPU2、原模型零更新和9个丢弃参数影子核实实际FP32新增下降，固定规则不依据目标返回调整。条件后续配方和raw／DURA三seed42cell也先固定；本登记不是训练启动或目标成功。
