# 真实 L 跨 RX 双边身份匹配：最小源原型

状态 **PLANNED**。上一RX历史过滤源资格0/4，CE四组改善而raw跨RX关系四组下降；删除RX差异方向没有身份依据。旧教师全关系、可信集合/补偿、CROSS_DIFF、episodic/ridge及原始周期pretext的限制保留。本轮只开一条不同的假设：用**真实L身份对应**约束实际四项joint160的跨RX匹配；不要求把接收机公共偏移从z删除，不新增头/prototype或teacher判定链。

每侧同RX/day包含六个不同TX，各一包，两侧物理ID不交。Dij=||a_i-b_j||²/160，B=D-rowmean-colmean+grandmean，C=B/max(RMS(B),1e-8)，RMS保持可微。令J=I-11ᵀ/6，则B=JDJ=-2(Ja)(Jb)ᵀ/160；任意两侧公共feature平移被消去，正全局幅度在floor未活跃时不改变C。可微分母阻止纯全局norm膨胀降低损失；只在损失成本表去除这些项，部署z仍是原actualjoint。它不保证真实RX失真只是加性，不消除任意非线性/频率相关失真，不宣称全部因素可辨识。

BI从exp(-C/.1)在log域做固定80次行列均匀边际缩放（目标1/6），报告实际残差和approximate标记，不默称精确OT最优。由于有限迭代未必同时完全平衡，正确TX质量分别由logP行/列条件归一化得到；只有精确均匀边际极限才等于6P正确元素。辅助loss=.1*(mean relu(log.9-logqrow)+mean relu(log.9-logqcol))/2，达到.9即停压。ROW对照使用同C的独立行列softmax，无共享列容量。W不进辅助图，主CE仍正常联合训练原默认cos头；三个gain与所有实际F71/195041coord保持原可微路径，没有分支删除或辅助embedding替代。

四条件ZERO/SAME_BI/CROSS_ROW/CROSS_BI。原L180为5RX×3day×2side×6trueTX；CROSS在side0的RXr与side1的RXt（r!=t）同day，共20有序RX对×3day=60个6×6矩阵，2160成本格/360row+360col真实身份关系；SAME同RX/day15unique重复4以匹配60成本表及每packet4端点曝光。四组都计算全部目标并保留sameforward预算，ZERO只加graphconnected0。已知类均衡只用于合法L监督与源特征分析；U不能按TX配对或用真类数，部署/target不运行全局匹配或quota。

固定own BASE_CE第一seed2026092701的原INITIAL及final20，父执行6693e4480ef990bc23c5021ff1440b547fe5fcaf，不按目标最高分选父。source_proof先核对完整roles/IDs/scratch来源/fixedfinal20后才load两个产物，strictmodel与actualargs检查。INITIAL用原groupoptions的空moments，FINAL保留实际m/v/step；两者临时新stageLR2e-4，无整体清矩或state过滤。固定四current L180 ordinal0/1、17/18、35/36、52/53，共720包；held180 ordinal68/69。900native5ID与globalbaseindex不交，held已被父训练见过，不能冒充盲测或未见RX。

源判据每state至少2/4 cohort的CROSS_BI同时优于ZERO、SAME_BI和CROSS_ROW：actualF差norm>1e-6、heldjointRMS差>1e-5、至少1浅中层tensor变化、held真实匹配条件质量均值增益>1e-4、held720双向匹配错数不增；ACTUAL与PRE_W heldCE相对三对照恶化不超过INITIAL1e-4/FINAL1e-6。当前CROSS_BI有活跃身份关系及实际aux梯度。这里直接评价新机制所要保留的身份对应，rawcos关系另报但不是资格代理，因为允许公共RXoffset而不强行几何全对齐；这不是修改旧RX过滤判据救活旧候选。INITIAL通过支持早期scratch20研究，若只有FINAL通过支持独立E20+20；没有在source assay中启动真实训练。所有阈值/τ/π/权重/iterations先于真实结果固定，失败不扫描救活或替换对照晋级。

实际after全188参数进入临时eval模型，after三个gain参与真实joint160；ACTUAL afterW与PRE_W仅beforeW分析同一z，用同FP64默认cos scale30/eps1e-4，无fit。报告各RX CE、classmargin、rawidentitygeometry、matching质量/错数、有限80边际残差、RMSfloor、aux/CE梯度方向、Sinc核与浅中层实际差。匹配损失下降或源proxy通过都不等于未见RX性能；before/after样本及配对关系曝光完整披露。

预算900uniqueL/6300metadata/7564packet网络前向（两state各2零IQ烟测）/64临时CPUAdamW、0实际训练/头拟合/EMA/U/V/target。两个own CP无query smoke PASS立即继续固定32单元。技术不一致/非有限异常保存并仅停本run，不影响其他owner。

run `20261015-phase1-SIGTIME-identity-transport-sourceL-s2026092701-r01`，owner codex/root/identity-transport-source-20261015，actualGitHEAD写release_commit.txt。普通N607/CVS-RFFI Python/CPU2线程/GPU0，CWD `/home/szu2070436088/2510044040/CV-SincNet/releases/cvs_SIGTIME_identity_transport_sourceL_20261015_r01`，命令 `/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python -u /home/szu2070436088/2510044040/CV-SincNet/releases/cvs_SIGTIME_identity_transport_sourceL_20261015_r01/experiments/cvs_identity_transport20/runner.py /home/szu2070436088/2510044040/CV-SincNet/releases/cvs_SIGTIME_identity_transport_sourceL_20261015_r01/resolved_request.json`，唯一输出 `/home/szu2070436088/2510044040/CV-SincNet/runs/20261015-phase1-SIGTIME-identity-transport-sourceL-s2026092701-r01/source_diagnostic`、日志 `/home/szu2070436088/2510044040/CV-SincNet/releases/cvs_SIGTIME_identity_transport_sourceL_20261015_r01/source_probe.stdout.log`。一个归档SHA比较/remotecompile/所属PID-CWD-cmd-GPU-loggrowth核验和独立终态读回；没有新增seal/receipt/report-onlygate。详细matrix/stop/artifacts见[experiment.json](experiment.json)。

双边匹配、熵正则OT和assignment表征学习都有先例：[Cuturi](https://arxiv.org/abs/1306.0895)、[SwAV](https://arxiv.org/abs/2006.09882)。候选待检验区别是CVS完整部署特征的**真实跨RX身份对应、损失中公共cost项与幅度中性化、有限双边容量和达到身份质量即停压**，不是Sinkhorn本身新发明。目前没有完整文献首创性或泛化收益证明。SIG_TIME/CENTER25K/COMPACT_LINEAR/LONG_SILU历史正路线按各配方保留，84%默认头clean三seed目标active、未完成。正式训练仍只能scratch20或同父E20+20，与正常defaulthead/原hiddenU配方匹配，并在冻结后逐row自动clean预测和独立truth-last。
