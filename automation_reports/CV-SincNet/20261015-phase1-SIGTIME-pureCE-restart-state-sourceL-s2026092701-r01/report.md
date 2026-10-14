# 纯 CE 续训起点：优化器状态与实际特征余量

状态 PLANNED。上一跨RX类间差方向诊断有103至129/300活跃、实际浅中层变化，但FINAL20 held关系身份间隔仅微弱正增益，固定gate0/8；该变体关闭。本轮不复活或改损失系数，而是先核对未来E20+20拟使用的训练阶段状态：原E20末态savedmoments/lr1e-6，不等于清空矩并开启新20轮日程的起点。

同一个已核实自有BASE_CE第一seed2026092701 final20、同一原clean CE梯度，比较END_SAVED（保存矩、原lr1e-6）、RESTART_KEEP（保存矩、新lr2e-4）、RESTART_RESET（空矩、新lr2e-4）。END/KEEP只隔离学习率，KEEP/RESET只隔离矩及step计数；所有原AdamWgroup参数/顺序一致，不reset模型权重。2e-4是原配方新阶段起点，运行前固定，不按目标分数或本次源结果扫描。此项是baseline状态诊断，不能包装成已确认创新机制。

原CE与默认cosine头正常联合梯度保留，无新loss/EMA/teacher/拟合头。每个临时optimizer只取得同一个CE梯度，simulate纯CE/纯CE两支须逐位相同；None-unused跳过语义保持，不给unused参数补zero。数据复用原7/63/30角色，L6300；四current L180的90cell ordinal0/1、17/18、35/36、52/53，共720包，独立评价L180 ordinal68/69，与current及彼此物理ID不交。heldL已被父训练见过，不是盲测/未见RX。U/V/target IQ不读，U真值隐藏。

本次真实f_theta含Sinc/时频/PA/reference/原归一化以及三个after-step融合gain，共71tensor/195041coord；具体classW不属于f_theta。after完整188参数复制到eval临时模型，单次IQ前向得到真实四项joint160。ACTUAL用after默认W；PRE_W只用beforeW分析同一z，after三个gain仍保留。两路同FP64 scale30/eps1e-4，无新head或fit。只改W不应改变z/PRE_W；各gain改变应进入z。旧辅助诊断固定全部gain与这里的测量边界不同，不能直接用绝对数值推翻或拼接旧结论。

固定source gate仅看RESET：至少2/4单元实际F位移norm>1e-6、heldRMS>1e-5、至少1个浅中层tensor变化、PRE_W heldmeanCE减少>1e-6、raw特征关系身份间隔增益>1e-5。身份间隔为同TX跨RX同day余弦减不同TX同RX同day余弦，区分于classlogitmargin。其他regime只作机制对照，不能替代RESET资格。规则只识别拟定阶段的源feature信号，不是DG/epoch训练收益；失败不改LR/清矩比例/样本顺序或复活关闭候选。

预算900uniqueL/6300metadata/3062packet前向/24临时CPUAdamW（固定state的一步，非24个实际更新）；0实际训练/头拟合/EMA/U/V/target。真实CP无querysmokePASS后立即继续固定12单元。来源/角色/ID/状态不一致、非有限/不支持optimizer/技术异常保存产物并停止此run；不影响其他owner或做不变重试。

run `20261015-phase1-SIGTIME-pureCE-restart-state-sourceL-s2026092701-r01`，owner `codex/root/restart-state-source-20261015`。实际GitHEAD写release_commit.txt，父E20执行6693e4480ef990bc23c5021ff1440b547fe5fcaf。普通N607账户/CVS-RFFI Python/CPU2线程/GPU0，CWD `/home/szu2070436088/2510044040/CV-SincNet/releases/cvs_SIGTIME_pureCE_restart_state_sourceL_20261015_r01`，命令 `/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python -u /home/szu2070436088/2510044040/CV-SincNet/releases/cvs_SIGTIME_pureCE_restart_state_sourceL_20261015_r01/experiments/cvs_restart_state20/runner.py /home/szu2070436088/2510044040/CV-SincNet/releases/cvs_SIGTIME_pureCE_restart_state_sourceL_20261015_r01/resolved_request.json`，唯一输出 `/home/szu2070436088/2510044040/CV-SincNet/runs/20261015-phase1-SIGTIME-pureCE-restart-state-sourceL-s2026092701-r01/source_diagnostic`，日志 `/home/szu2070436088/2510044040/CV-SincNet/releases/cvs_SIGTIME_pureCE_restart_state_sourceL_20261015_r01/source_probe.stdout.log`。配置/矩阵/环境/stop/artifacts见[experiment.json](experiment.json)。

只有源规则通过才进一步准备拟定reset阶段的E20+20公共纯CE控制：父20＋增量20累计40，原默认头/合法角色保持，正常训练结束逐row自动clean预测再独立truth-last。这里尚未发布该真实训练。SIG_TIME/CENTER25K/COMPACT_LINEAR/LONG_SILU历史正收益分别保留，84%默认头clean三seed目标仍active、未达成。
