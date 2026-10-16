# F＋跨 RX：孤立轻度 AWGN(raw) 源身份检验

PLANNED，尚未启动。逐 row 唯一矩阵与模型来源见 [experiment.json](experiment.json)，候选、文献限制和未来匹配 E20 计划见 [研究说明](../../../analysis/cvs_isolated_awgn_candidate_20261017.md)。原 F＋跨 RX 同 TX、三 SIG_TIME／统一 Sinc／joint160／NoExplicitCFO／cosine30 默认头保持。

唯一算子是原 native proper complex AWGN(raw)：每包 SNR uniform[20,36] dB，方差按该 raw 包功率，两路独立零均值高斯噪声。沿用既有 ordinary 的 SNR 范围，未从目标或源曲线选择新范围。没有任何 PA／MP／CFO／时移／相位／剪裁／量化／RMS 恢复或完整 practical，不读 TX／RX／day／置信度决定噪声。名义 SNR 有界，高斯样本幅度无严格有限界；实际能量与尾部全部保存，不删除、重抽、clip或归一化。新增噪声 SNR 不等于原录音总 SNR，数字噪声不是实际 RX 替换。

两份原合法 source-only scratch E20 final20 B00 默认头，seed2026092701/02，仅作冻结源检查，不作为未来训练初始化。原训练 L5040/U45360/V21600、RX1held、final-only/4440 更新、无未知祖先／target／另训头来源核对后，每模型在原不可变后端加载一次。U 在原训练中仍遵守隐藏真值规则，本轮不训练。

只用 fitL5040，RX3/4/6/8×day1/2/3×6TX，72格各70包。原缓存 L6300、未用源V／历史特征反序列化，RX1原L1260在推断前排除；U仅角色ID，无IQ或TX标签。不作heldRX1／U／V／target前向，不混同完整L6300或L_train5580/L_dev720；匹配既有VALIDATED_ONCE句柄，不新增数据验证。

增强流202610172101至04，独立shuffle generators202610172110至13，每draw全5040一次，两个模型相同命名观察，不挑抽样。每模型raw5040＋四noise5040=25200源NN，总50400源NN＋首模型真实CP合成2包no-query smoke=50402NN；source noise生成40320，另canonical native noise128（每模型原生32＋算子32），0NN。旧ordinary／PA／As生成均0，0新epoch/优化器/教师/头拟合/target。参照历史40model-E/8880更新/9373356NN单列。

固定源条件：两个模型 pooled 及每draw均要求accuracy与macro-F1相对raw≥−0.5pp、raw正确保留≥99%、每RX/TX/day accuracy≥−1pp。全部20160 noise的相对IQ RMS变化median≥0.01、max≤0.15；不要求噪声修复接近满分的raw。全部包/分层/尾部保留。工程变化与非空不能代替身份条件。科学FAIL仍完成双模型全部观察，然后关闭固定20至36dB/全覆盖方案，不抬SNR/clip/删尾/重抽/按PA或RX条件/换draw或seed救跑。

PASS仅源身份保持依据，不是Ag/DG/As或联合收益。未来必须另登记三组 scratch E20：公共raw.75＋原ordinary.25/SupCon、原U/默认头联合规则保持，extra.25分别raw、同一份ordinary、独立noise(raw)，均8L前向/总权重1.25/sameIQ RDrop。候选须同时超过两个控制 source-heldRX clean≥+1pp、macro-F1不降、每practical/seenV≥−0.5pp，再独立seed确认；这三组尚未实现登记启动，不是此次source launcher的自动训练。

唯一owner codex/root/F-isolated-AWGN-source-20261017，普通N607现有CVS-RFFI Python，CPU/OMP/MKL/OpenBLAS2，CUDA_VISIBLE_DEVICES为空、GPU none，两worker顺序，不干预旧健康任务。CWD PROJECT/releases/cvs_F_isolated_AWGN_fitL_20261017_r01；命令 CVS-RFFI/bin/python -u -m experiments.cvs_isolated_awgn17.dispatch，逐row runner命令在experiment.json。真实私有账户与Python路径只存runtime，不入Git。

发布入口 F:/App/miniconda3/python.exe -X utf8 -m experiments.cvs_isolated_awgn17.publish，CWD当前managed工作树。Git commit是代码身份，实际值写release_commit.txt/resolved_config。唯一不可变输出 PROJECT/runs/20261017-diagnostic-F-crossRX-isolatedAWGN-fitL-manysig-m2-r01；各seed.stdout.log及release/source_isolated_AWGN.stdout.log。一个预检、一份archive与一次传输SHA比较、一个remote compile，launch后PID/CWD/argv/CUDA/日志增长或自然终态独立读回。首worker真实CP smoke PASS立即继续，无token/receipt/签章或重复审查层。

错来源/角色/native配置、非有限或零能量、输出碰撞或worker异常保存failure/partial停止，无自动重试或旧进程变更。本地14输入/配置负例＋9科学反例、1152合成native noise PASS，0源IQ/CP/NN/更新；native输出/RNG、正增益等变、I/Q properness和lag统计、原输入/PythonRNG保持已核实。旧本地NumPy桥警告未调用、不装包。一次独立P0/P1审查PASS，未发现直接阻断缺陷。

产物包含root request/active/launch/result/completion；每模型actual native args/callable路径/原baseline配置和来源；raw/四draw完整logits/硬判决/72格；320 noise batch全包名义/实际added-noise SNR、RMS、IQ变化、功率交叉项和原始物理ID私有保存；完整日志、smoke、canonical和模型状态/真实计数；独立全50400判决/CE/F1/分层/每draw/保留率/能量统计/成本复算、图、同row报告/登记。IQ、权重、逐包预测/ID/运行态不提交Git，仅聚合元数据。

完整联合目标ACTIVE；现有固定失败保持关闭，本方向区别另一对话practical内正负噪声配对，不修改其任务或继承其模型。
