# 纯 CE 续训起点：优化器状态与实际特征余量

当前状态 **ANALYZED / VERIFIED，固定RESET gate0/4**。以下保留预登记方案。上一跨RX类间差方向诊断有103至129/300活跃、实际浅中层变化，但FINAL20 held关系身份间隔仅微弱正增益，固定gate0/8；该变体关闭。本轮不复活或改损失系数，而是先核对未来E20+20拟使用的训练阶段状态：原E20末态savedmoments/lr1e-6，不等于清空矩并开启新20轮日程的起点。

同一个已核实自有BASE_CE第一seed2026092701 final20、同一原clean CE梯度，比较END_SAVED（保存矩、原lr1e-6）、RESTART_KEEP（保存矩、新lr2e-4）、RESTART_RESET（空矩、新lr2e-4）。END/KEEP只隔离学习率，KEEP/RESET只隔离矩及step计数；所有原AdamWgroup参数/顺序一致，不reset模型权重。2e-4是原配方新阶段起点，运行前固定，不按目标分数或本次源结果扫描。此项是baseline状态诊断，不能包装成已确认创新机制。

原CE与默认cosine头正常联合梯度保留，无新loss/EMA/teacher/拟合头。每个临时optimizer只取得同一个CE梯度，simulate纯CE/纯CE两支须逐位相同；None-unused跳过语义保持，不给unused参数补zero。数据复用原7/63/30角色，L6300；四current L180的90cell ordinal0/1、17/18、35/36、52/53，共720包，独立评价L180 ordinal68/69，与current及彼此物理ID不交。heldL已被父训练见过，不是盲测/未见RX。U/V/target IQ不读，U真值隐藏。

本次真实f_theta含Sinc/时频/PA/reference/原归一化以及三个after-step融合gain，共71tensor/195041coord；具体classW不属于f_theta。after完整188参数复制到eval临时模型，单次IQ前向得到真实四项joint160。ACTUAL用after默认W；PRE_W只用beforeW分析同一z，after三个gain仍保留。两路同FP64 scale30/eps1e-4，无新head或fit。只改W不应改变z/PRE_W；各gain改变应进入z。旧辅助诊断固定全部gain与这里的测量边界不同，不能直接用绝对数值推翻或拼接旧结论。

固定source gate仅看RESET：至少2/4单元实际F位移norm>1e-6、heldRMS>1e-5、至少1个浅中层tensor变化、PRE_W heldmeanCE减少>1e-6、raw特征关系身份间隔增益>1e-5。身份间隔为同TX跨RX同day余弦减不同TX同RX同day余弦，区分于classlogitmargin。其他regime只作机制对照，不能替代RESET资格。规则只识别拟定阶段的源feature信号，不是DG/epoch训练收益；失败不改LR/清矩比例/样本顺序或复活关闭候选。

预算900uniqueL/6300metadata/3062packet前向/24临时CPUAdamW（固定state的一步，非24个实际更新）；0实际训练/头拟合/EMA/U/V/target。真实CP无querysmokePASS后立即继续固定12单元。来源/角色/ID/状态不一致、非有限/不支持optimizer/技术异常保存产物并停止此run；不影响其他owner或做不变重试。

run `20261015-phase1-SIGTIME-pureCE-restart-state-sourceL-s2026092701-r01`，owner `codex/root/restart-state-source-20261015`。实际GitHEAD写release_commit.txt，父E20执行6693e4480ef990bc23c5021ff1440b547fe5fcaf。普通N607账户/CVS-RFFI Python/CPU2线程/GPU0，CWD `/home/szu2070436088/2510044040/CV-SincNet/releases/cvs_SIGTIME_pureCE_restart_state_sourceL_20261015_r01`，命令 `/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python -u /home/szu2070436088/2510044040/CV-SincNet/releases/cvs_SIGTIME_pureCE_restart_state_sourceL_20261015_r01/experiments/cvs_restart_state20/runner.py /home/szu2070436088/2510044040/CV-SincNet/releases/cvs_SIGTIME_pureCE_restart_state_sourceL_20261015_r01/resolved_request.json`，唯一输出 `/home/szu2070436088/2510044040/CV-SincNet/runs/20261015-phase1-SIGTIME-pureCE-restart-state-sourceL-s2026092701-r01/source_diagnostic`，日志 `/home/szu2070436088/2510044040/CV-SincNet/releases/cvs_SIGTIME_pureCE_restart_state_sourceL_20261015_r01/source_probe.stdout.log`。配置/矩阵/环境/stop/artifacts见[experiment.json](experiment.json)。

只有源规则通过才进一步准备拟定reset阶段的E20+20公共纯CE控制：父20＋增量20累计40，原默认头/合法角色保持，正常训练结束逐row自动clean预测再独立truth-last。这里尚未发布该真实训练。SIG_TIME/CENTER25K/COMPACT_LINEAR/LONG_SILU历史正收益分别保留，84%默认头clean三seed目标仍active、未达成。


## 完整真实源结果

执行 `be3062a0941b1a197305ba73270d6a0f72f483b8` 已推送且remote OID一致，归档传输SHA、远端编译、实际CP无querysmoke、PID/CWD/命令/CPU/GPU/log增长和独立自然终态核实；无failure。全部12单元/4paired比较、same原CE梯度/原state不变，source进程约6.98 s。

| ordinal | 状态 | lr | 实际F位移 | held RMS | 固定preW CE减少 | 实际W CE减少 | 关系身份间隔增益 |
|---|---|---:|---:|---:|---:|---:|---:|
| 0 | END_SAVED | 1.0e-06 | 0.000081 | 0.000070 | +2.114e-06 | +2.143e-06 | -9.730e-07 |
| 0 | RESTART_KEEP | 2.0e-04 | 0.016224 | 0.014211 | +5.687e-05 | +5.118e-05 | -3.268e-04 |
| 0 | RESTART_RESET | 2.0e-04 | 0.083060 | 0.070088 | -2.201e-03 | -2.258e-03 | +8.757e-03 |
| 17 | END_SAVED | 1.0e-06 | 0.000073 | 0.000059 | +1.644e-06 | +1.667e-06 | -1.351e-06 |
| 17 | RESTART_KEEP | 2.0e-04 | 0.014506 | 0.011827 | +1.028e-04 | +1.001e-04 | -3.628e-04 |
| 17 | RESTART_RESET | 2.0e-04 | 0.082900 | 0.070949 | -9.139e-03 | -9.460e-03 | -1.577e-03 |
| 35 | END_SAVED | 1.0e-06 | 0.000086 | 0.000077 | +2.581e-06 | +2.611e-06 | -3.160e-06 |
| 35 | RESTART_KEEP | 2.0e-04 | 0.017233 | 0.015540 | +4.018e-05 | +3.265e-05 | -7.834e-04 |
| 35 | RESTART_RESET | 2.0e-04 | 0.082873 | 0.076261 | -3.472e-03 | -3.606e-03 | -3.058e-03 |
| 52 | END_SAVED | 1.0e-06 | 0.000086 | 0.000078 | +2.511e-06 | +2.527e-06 | -1.761e-06 |
| 52 | RESTART_KEEP | 2.0e-04 | 0.017192 | 0.015689 | +3.883e-06 | -7.327e-06 | -5.215e-04 |
| 52 | RESTART_RESET | 2.0e-04 | 0.082898 | 0.078500 | -5.959e-03 | -6.295e-03 | -4.813e-03 |

RESET四单元0/4达标，直接原因是固定preW held CE全升0.002201至0.009139，实际afterW CE也全升0.002258至0.009460。更新前held meanCE是0.004644，准确率100%；CE是置信度敏感量，不能把CE变化当准确率百分点。RESET关系身份间隔只有ordinal0提高0.008757，其他三组下降0.001577至0.004813。单个大几何正差也不能代替稳定可读性或预登记资格。

RESET实际F位移0.08287至0.08306、heldRMS0.07009至0.07850，全部18个浅中层tensor变化、24个Sinc核变化。三个gain的位移norm约0.002525至0.002530，接近每个160向量的 `2e-4*sqrt(160)`。这与首次AdamW较强的归一化步幅一致，但未测逐坐标符号，不把近似解释当全网或全训练期因果证明。新增学习能力不是零，问题在更新方向/尺度与独立源身份保持。

KEEP使用相同lr2e-4而保留矩：F位移0.01451至0.01723，约END的200倍。固定preW CE四组都减少3.88e-6至1.03e-4，实际W CE三降一升（ordinal52升7.33e-6），说明头和特征必须区分。raw关系身份间隔四组均下降3.27e-4至7.83e-4，不能只按CE降低宣称泛化收益。按预登记KEEP不能替代RESET资格，也不因为KEEP这项正迹象直接派发新的阶段训练。

END保留末态lr1e-6和矩：固定preW CE四组下降1.64e-6至2.58e-6，raw关系间隔四组小降9.73e-7至3.16e-6。原末态并非所有量都无变化；只用增大LR解释源泛化瓶颈并不充分。KEEP/RESET是同LR纯状态对照，源证据确认了尺度历史及方向历史的重要性，但尚不能识别哪个历史分量是RX因素。

实际900uniqueL/6300metadata/3062packet网络前向/24临时CPUAdamW/60条RX双读出记录，0实际训练/头拟合/EMA/U/V/target。原model/buffer/optimizer/CPpayload/grad/RNG保持，3gain是after真实feature参数，PRE_W只替W且无拟合。该测量边界与旧固定全部gain的辅助诊断不同，不拼接绝对数值。heldL已被父训练见过，single eval CPU一步不等于dropout/多epoch/未见RX行为；没有新clean预测或评分。

固定reset阶段资格失败，不把KEEP/END作fallback，不扫LR/矩比例/样本顺序，不复活已关闭CROSS_DIFF。下一独立机制只写出设计：保留二阶尺度及原当前梯度，借真实L的RX梯度对比只过滤历史feature一阶动量贡献，默认W原状态继续联合学习。这是与整体清矩和旧完整位移风险投影不同的结构化假设，尚未实现/发布/证明创新或效果；不把设计等同晋级。

SIG_TIME/CENTER25K/COMPACT_LINEAR/LONG_SILU历史正路线独立保留，不相加或按目标最高分选父。84%默认头clean三seed目标仍active、未达成。完整[12单元](source_cells.csv)、[60条RX记录](source_RX_metrics.csv)、[4组状态差](paired_regimes.csv)、[标量证据](result_evidence.json)、[运行核验](launch_evidence.json)和[下一机制设计](../../../analysis/phase1_receiver_contrast_momentum_20261015.md)已保存，私有原输出/日志/权重保留。
