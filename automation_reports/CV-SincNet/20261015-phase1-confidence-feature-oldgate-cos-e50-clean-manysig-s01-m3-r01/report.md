# 固定 SIG_TIME 置信度门控特征一致性：首种子最小预登记

状态 LOCAL_VERIFIED / NOT_LAUNCHED。实际代码/config 325d113dff08206cb2f174e74d33e48f3e0b2ad5（主体 cadd843d569e85b81b5e9e8bc14c73b0ab4e00ae，修复发布依赖不改科学配方）；唯一独立 P0/P1 修复后 PASS。真实 checkpoint 无 query smoke 为发布首步，PASS 即继续。

源依据：20261015-phase1-source-teacher-confidence-manysig-m6-r01 已完成六模型全 L/V 源证伪、7200向量独立复算，固定 FP32 teacherweak maxprob>=studentstrong maxprob（ties保留，差阈值0）在6/6模型V非空有益代理组保留率严格高于有害代理组。源资格/完整结果先于本候选实现提交b1233fc4b，无新目标成绩反馈。L全局导数仍混合，故只提出单一可检验训练假设，不宣称解决冲突、U准确率或DG增益。

候选与位置：网络固定SIG_TIME/f_theta195041/defaultcos6x160 scale30，默认头按原CE联合训练；新损失只作用实际部署 joint160。复用已经计算的 ownEMAweak/studentstrong logits/features，置信度判定 detach，原 unitjoint160 half squaredL2 乘gate后 sum/fullbatchN，绝不除保留数；原 .1 系数ramp为第34轮.02、38轮起.1。所有原U CE .16、entropy .01、旧batch_neighbor门控/strongagreement/EMA.999、L smoothing.01、cos日程/优化器/增强与采样不变。新gate不改原硬伪标签损失，不增加teacher/student前向、模块、参数、头或Fishr/PAIR机制。

矩阵 20261015-phase1-confidence-feature-oldgate-cos-e50-clean-manysig-s01-m3-r01：seed2026092701，OLD_GATE_COS50复用核实历史原默认头/同clean预测，SIG_TIME ZERO和CONF_FEATURE_MT两个新模型从零各50轮=33label+17pseudo、222step/轮、11100更新；相同初始化/RNG/native RANDOM U顺序，新增hook默认None且恢复后清空，不热改已有release。每模型U17x56700=963900包，U真值隐藏；teacher仅同scratch学生EMA。禁止加载目标最优/污染checkpoint、续训超预算或单独训头。原L6300/U56700/V27000、sourceRX1/3/4/6/8、day1/2/3、TX/物理ID、split392005、NoExplicitCFO/内部补偿均不变，复用现有VALIDATED_ONCE数据，V无梯度，不新增信道增强。

固定源扩展规则：两新模型实际11100更新/17x56700、默认头/特征导出W独立/数值特征变化>=1e-6、final LCE<=.5；候选t1/t2/SIG实际effective梯度在34/50轮非零。最后5轮候选/对照保留样本feature均值比<=.95，同时全U未门控feature均值比<=1.0且候选保留率>=.5，以避免仅缩小覆盖伪装优化。该规则只决定是否固定02/03扩展，不改变首种子已登记测试，不按目标分数停跑/调参/排名/选择性重跑。任何新增数值目标结果读回前，先记录完整source资格与02/03扩展决定、commit+push。源PASS仅优化/接线资格，不等于DG成功；若FAIL保留完整负结果，不补扫阈值。

逐row自动链：训练→固定final50→默认ORIGINAL_HEAD clean168000逐包预测→prediction全部固定→独立truth-last scorer→同row报告登记，3row共504000决策/54条预定分组指标。只clean，不追加卫星/practical；不另训头、不源换头、新TX/Phase2。报告原头accuracy、macroF1、最差RX/day、配对差/修正与引入错误；n=1不替代3seed84%目标。完整3seed扩展前只报告实际seed数与缺项。

执行普通N607既有CVS-RFFI/fullFP32、Torch/OMP/MKL/OpenBLAS2线程，CWD PROJECT/releases/cvs_confidence_feature50_20261015_r01，immutable run PROJECT/runs/20261015-phase1-confidence-feature-oldgate-cos-e50-clean-manysig-s01-m3-r01；入口 python -u -m experiments.cvs_confidence_feature50.dispatch，本地 experiments.cvs_confidence_feature50.publish --output local_artifacts/confidence_feature50_delivery_r01。GPU由既有capacity lock动态分配，所有owner每GPU最多3个实验进程、allocationfree>=12000MiB；当前健康Fishr任务保持原GPU/配方，不停止/重启/迁移。实际命令/路径/角色/owner/完整row configs见experiment.json。

成本：100新增骨干epoch/22200更新 + 共享历史baseline50/11100；每模型参数与原SIG完全相同，新loss无额外modelforward。保留原33/34/50各222batch的真实L/U导数观测，并保存门控/全U/保留样本loss、保留数、ties及源曲线；各模型5994新增feature诊断向量（不冒充训练优化器累计更新）、actual f_theta参数/特征变化和W独立export。性能/速度据实测，不以轮次等同收敛。

本地检查PASS：相同初始模型、原U对象和头梯度不变、teacherstopgrad、实际t1/t2/SIG及默认头正常更新、全批分母/全保留/空保留/ties、50轮/换头/target来源等10负例、export精确/W独立、nativeRANDOM顺序/RNG、hook恢复和生成REMOTE/INSPECT compile。独立P0/P1发现缺失FMS seeds发布依赖，已最小修复并定点确认PASS，无真实数据/模型/远端访问。发布只有一次普通账户preflight、单archive本地/远端SHA比对、remotecompile、真实CPnoquery首步及PID/CWD/argv/GPU/loggrowth核实，无额外签名/收据/重复审批。

技术停止规则：仅run-owned系统技术失败（权限/契约/query/head/非有限或跳过更新/预算/输出冲突/可执行或合法预测/评分失败），保留所有partial并写failure，不自动重试；只核实准确owner，不触碰健康任务。科学分数失败保留，不停其他row。预期artifact包括原final_ssdg/source completion/epoch curve、feature witness和ssl+feature梯度/门控步日志、默认头predictions/score/completion，以及全runsource_features/source_qualification/analysis/completion。原始IQ/checkpoint/PT/日志/预测留不可变目录，Git交实现/config/报告/必要聚合JSON/图。

## 发布后状态

RUNNING / VERIFIED：执行commit 704cf37536d3cb6b8764af2131547db4a9939480，单archiveSHA/remotecompile/真实CPnoquery smoke PASS。SIG worker2588400/GPU3、CONF worker2588549/GPU6的PID/CWD/argv/GPU/loggrowth核实，两个scratch50模型均已完成第1轮（新feature机制按计划从第34轮启用），历史baseline自动clean链已执行；当次failure0。新目标数值摘要尚未读取，源资格与扩种子决定仍先行commit/push。完整过程见release_launch_evidence.json；训练/测试/性能结论尚未完成。

## 实际初始化与前段训练核验

另进程只读两模型initial_smoke.pt，198个参数/buffer张量、864924个状态元素逐值相同，scratch/无父权重/数据角色一致；无新前向或模型更新。第3轮源CE分别0.403988/0.402676、V95.1741%/95.2741%，此时feature系数/loss/U前向均0，因此不能计作新机制收益。两模型cudnn_deterministic=false，轨迹未保证逐步数值完全一致；实际初始化匹配与未启用阶段的训练波动分别记录于actual_initial_and_pre_stage_comparison.json，不改变健康训练或配方。
