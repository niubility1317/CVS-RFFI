# SIG_TIME 同分支位置对齐 E50 八组小试

状态：RUNNING，A0已执行，其他七组等待共同源定标后自动从零起跑。落实用户最新附件：A0、A-T、A-F、A-P、A-R、A-S、A-J、A-ALL，单 seed 2026092701 先检验实现/学习信号，正式三 seed 为 2026092701/02/03。首 seed 不能宣布最佳分支，F/P/R 不按替换诊断或目标成绩剔除。旧四行 E50 和两行 KL/DIR 的 immutable release 不变。

八组均从零 E50=33 labelled+17 native old-gated pseudo，222 更新/轮、11100 更新，128 原 L 样本+128真实 practical residual 普通拼接；各半 CE 均值等价于拼接 CE 均值。原 AdamW lr .0002/wd .0001、cosine/minlr1e-6、smoothing .01、own EMA 与原 U 训练不变。无 margin、KL、课程或新 U urban 目标。所有原网络与默认 cosine6×160/scale30 联合训练；不增加部署模块/拟合分类头。33/17 是 E50 原基线配方，额外方向损失仅 epoch6至50 启用，前5轮所有位置规则一致。A0 全程使用相同 clean EMA L 前向和实际分支捕获，只不加对齐。

数据：复用外层 source contract split392005，L_s/U_s/V=6300/56700/27000、6TX、源RX1/3/4/6/8、day1/2/3，IQ256、25MHz、equalized=1、center/nativeRMS→simulatorRMS，输出不RMS。原L按TX×RX×day共90单元各70条，以fold seed2026101503的物理ID随机顺序各留8条，共L_dev720；L_train5580。开发样本不进入CE/U/EMA/normalization更新。V27000只健康诊断，不拆分；外层测试IQ/信道/物理集合不变。内部分折先已写入PROJECT_PROTOCOL，实际IDs写inner_source_contract.json并进入checkpoint split_info。历史完整L模型见过开发样本，不能作为新组初始化、教师或严格匹配基线，不能称开发折全历史未见盲测。

结构：真实core为experiments/adv3b02_xuc/code/model.py。T/F读实际fuse321输入，F包含启用freq_stats_proj；P读pa_norm的实际完整增量；R读response_projection；S读output_norm gain前160，反传到local_projection/t2。学生图保留，教师eval/stopgrad，日志detach；同一单位方向损失eps1e-4，L资格统一为clean ownEMA默认头正确×urban场景，无urban学生置信门槛，所有合法L satellite始终参与CE。J按原数值重组，仅aux gain stopgrad；所有aux不更新默认W/gain，CE正常更新。训练时在原concat前向前临时捕获，前向结束即移除hook，之后原生U和EMA不被捕获。

强度：A0固定epoch5保存student/ownEMA和内部契约，使用同一L_dev720、mid/low urban独立固定源流，计算实际joint祖先195041参数/71张量的全局梯度，不含W，无梯度项按0。lambda_b=.1*median||grad(.5 CEsat)||/(median||grad L_b||+1e-12)，不是旧逐组/逐batch比值的中位数。小于1e-8、非有限或系数超[1e-4,10]停止待发组并保留产物，不无限放大。ALL先用单分支尺度定标，omega=.2且和1，再校准组合总系数。定标3600前向、0优化更新，只做一次，后续位置和seed共享系数；源定标/面板保存并恢复CPU/CUDA RNG和所有模型状态/待积累梯度。其余位置等待源recipe存在后从零起跑，不继承A0权重。A0校准额外成本单列。

源开发：固定14个epoch观察（1/5/6/10/15/20/25/30/33/34/35/40/45/50），全六源practical+clean，70560前向/组，不更新模型。仅final50参与推进。扩整个八组家族的预登记条件：至少一个新增约束对A0的源low≥+1pp、clean≥−0.5pp、urban≥0pp、最差urban RX≥−1pp；全部八组保留，随后额外两个预定义seed，绝不按target挑组。正式三seed后，b1/b2必须各在3/3源low提高、均值≥+0.5pp、均值clean≥−0.5pp、urban≥0pp、最差RX≥−1pp；按low、urban、最差RX顺序，0.2pp内才比较约束/成本。至少两个合格单项才做b1+b2、b1+J、b1+b2+J，重新定标组合总强度。此后续组合尚未发布，不预设T+S。若源家族条件不满足，保留负结果，不用target修正门槛。

训练后每row固定final50，立即自动对原clean+六practical的168000物理记录/视图预测，共1176000预测/组；全部七视图预测固定后exec独立truth-last评分。当前正式IQ直接复用VALIDATED_ONCE，不重新生成/调SNR/删难样本。主指标low/mid/urban/最差RX/Macro-F1，保持clean/suburban。测试不回流参数/选模/推进。源分支方差、gain/贡献、梯度比例/夹角、原生实际clipping、总AdamW更新与覆盖保留，不能把aux loss下降或gain关闭称为鲁棒性改善。

预算：每组父0/新增50/累计50，11100更新；主前向6190200=3×1420800个L clean/sat/EMAclean+2×963900个原U教师/强视图。另列70560源开发、A0一次3600校准、原生V各epoch和最终健康验证、smoke和1176000测试；墙钟以实际completion/telemetry为准。8组总88800更新。尚无本轮训练结果或收益声明。

发布命令：`E:/type10-7/local_envs/ssr-gpu/python.exe -X utf8 -m experiments.cvs_urban_branch50.publish --output E:/type10-7/local_artifacts/cvs_branch_alignment_20261015/pilot_delivery_r01`；同参数加`--inspect`独立回读。远端命令`/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python -u -m experiments.cvs_urban_branch50.dispatch`，CWD PROJECT/releases/cvs_sigtime_branch_align_e50_20261015_r01；唯一run根PROJECT/runs/20261015-phase1-sigtime-branch-align-e50-manysig-s2026092701-m8-r01，owner codex/root/branch-align-e50-20261015。普通N607账户，CPU2线程/每GPU最多3总实验进程（计其他owner），共享容量锁/free≥12000MB，不终止或热改健康任务。

最小发布：一次实际代码/config commit、11项有针对性的本地检查、首步真实冻结B0默认头无query smoke（仅操作检查，绝不作内部折初始化）、一次P0/P1审查、一次archive传输SHA比较、远端编译/预检、一次PID/CWD/argv/GPU与日志增长独立回读。输入/内部角色/来源、非有限、近零或越界校准、aux泄漏、预算、输出冲突或启动失败停止待发组，活动健康行自行完成，产物保留、不自动重训。后续真实产物以experiment.json列出的路径为准，代码/config身份以Git commit与release_commit.txt为准。

本地：`experiments.cvs_urban_branch50.checks`5项PASS，`training_checks`6项PASS，compileall/diff-check PASS；完整F/P、单项替换保留urban其他分支/rho、native实际concat学生图、teacher/U前hook移除、S梯度与主CE W、aux W/gain边界、全局中位数比与退化拒绝、L开发折互斥/均衡/顺序稳定、完整8配置和源家族规则均已检验。一次独立P0/P1审查PASS；发现的recipe部分写入竞态已按同目录临时完整写入后rename修复，并完成一次定点核实，其余指定范围未发现P0/P1。完整正式三seed与三种组合、同信道中间视图和局部SIG_TIME分组教学仍待后续证据，不把此小试等同全部目标完成。

发布代码commit `152d8921c903fc6d5f782c199b075d0837840ece` 已push并独立核对远端OID。一次归档传输/远端编译/普通账户预检VERIFIED；dispatcher2937430，A0 worker2937455，GPU3，实际CWD/argv/默认checkpoint首步smoke核实，epoch1日志增长5152字节。实际inner_source_contract为L_train5580/L_dev720/U56700/V27000，物理ID数互斥，真实总参数863730，默认头未改，max_grad_norm=5.0，A0新增aux梯度全0。独立回读`local_artifacts/cvs_branch_alignment_20261015/pilot_delivery_r01/readback.json`及`monitor/latest.json`；唯一release不变，健康旧KL/DIR继续原E50。尚无八组target结果或正式三seed结论。


## 一次源全局定标与八组放行

A0固定epoch5源定标已SOURCE_BRANCH_CALIBRATED，6.332468秒、3600前向、0优化/拟合头更新。实际特征195041参数/71张量、W排除、未接收梯度按零；student/EMA state与已有训练grad逐值保持，CPU/CUDA随机流恢复。源only、query/U truth均未读取。所有参数系数在预设范围且未裁切。

| 位置 | median全局aux梯度 | 冻结系数 | 定标后median范数比 |
|---|---:|---:|---:|
| T | 0.492487101 | 0.516316511 | 0.100000 |
| F | 0.172256637 | 1.476165016 | 0.100000 |
| P | 0.216721104 | 1.173301617 | 0.100000 |
| R | 2.252207133 | 0.112902236 | 0.100000 |
| S | 0.811468640 | 0.313356807 | 0.100000 |
| J | 0.392445790 | 0.647934640 | 0.100000 |
| ALL | 0.128869134 | 1.973158456 | 0.100000 |

共同分母median||grad(.5 CEsat)||=2.542792220。ALL表中系数是先各分支定标、omega=.2平均后再乘的总系数，不是直接把五个单项强度相加。定标后比0.1指同一源静态检查点的全局范数中位数比，不保证各层或后续AdamW更新完全相同。

八组均实际启动（旧release仍健康），每组独立scratch并按同规则先预热5轮，epoch6才开唯一aux；它们并未加载A0模型权重。一次放行后的独立PID/CWD/argv/GPU回读VERIFIED：

| row | PID | GPU | 已完成epoch（读取时） |
|---|---:|---:|---:|
| A0-s2026092701 | 2937455 | 3 | 6 |
| A-T-s2026092701 | 2941955 | 4 | 1 |
| A-F-s2026092701 | 2942032 | 5 | 1 |
| A-P-s2026092701 | 2942113 | 6 | 1 |
| A-R-s2026092701 | 2942194 | 7 | 1 |
| A-S-s2026092701 | 2942275 | 7 | 0 |
| A-J-s2026092701 | 2942757 | 6 | 0 |
| A-ALL-s2026092701 | 2942835 | 4 | 0 |

源系数完整原始记录在local_artifacts/cvs_branch_alignment_20261015/branch_calibration.json；不将物理/训练原始数据入Git。训练完成及真实aux梯度/全部七视图评分仍待执行，本状态不等于八组完成。
