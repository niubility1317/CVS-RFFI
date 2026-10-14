# 固定 SIG_TIME：900L 骨干方差形状梯度最小预登记

当前状态 ANALYZED / VERIFIED，源资格PASS，完整结果见[final_analysis.md](final_analysis.md)。以下保留预登记历史。原预登记状态 LOCAL_VERIFIED / NOT_LAUNCHED。代码/config 6907495c3c35147a08a5daa5c30f85fce9060ba8；独立新增包P0/P1 PASS，本地11负例/可微分/尺度不变/W无依赖与入口依赖/REMOTE/INSPECT compile PASS。真实checkpoint无query smoke为发布首步，PASS立即继续，不设新审批。

源依据是已完成BF的936保存矩阵拆分（结果b6014d5ae）：原归一化目标的终轮径向项占85.92%至96.54%，径向项12/12下降，加权角向项12/12上升；纯形状cos未全面改善。新问题是直接RAW方差向量角度目标是否传递真实t1/t2梯度，且与同批身份监督兼容。既有目标负结果只作已知上下文，不用来选父权重、阈值或参数。

矩阵仅一个冻结模型：20261014-phase1-backbone-fishr-oldgate-cos-e50-clean-manysig-s01-m4-r01/SIG_TIME-s2026092701/final50，原实际执行600afeeefe84e0d9f50f07c38ae0767bf1cabf07，原scratch50=33label+17pseudo/11100更新、旧batch_neighbor+ownEMA+cos、匹配ZERO额外L180对照、既有默认头正常联合训练。来源/物理角色由现有source_proof核对，不用targetbest/续训权重。模型固定eval，所有参数与buffers不更新；默认W仅在辅助per-packet CE图中常量，绝不拟合或替换头。当前研究不开发Phase2/newTX/卫星。

输入仅原L_s6300中既有确定BF分组的前5个L180 cohort，共900不同物理样本（每TX/day/RX2个）；实际数据seed2026210712、原modelseed2026092701、split392005。sourceRX1/3/4/6/8、day1/2/3、TX/physicalID、NoExplicitCFO/内部补偿保持，复用VALIDATED_ONCE。共享source builder返回V后立即丢弃，不做V前向/梯度；U无tensor或TX真值，query无访问。此为事前固定小源证伪，不冒充全L6300或多个modelseed。

目标：用原默认CE smoothing.01对真实t1PW3456、t2PW6912求逐包梯度G，维持原RX内无偏方差（36包、分母35）。对5个非零方差向量作L2归一化，逐层平均10对1-cos，再平均t1/t2；固定识别floor为范数1e-15，退化方向技术失败，不扫epsilon。不做TX/day中心化、不加其他损失或投影器。比较原归一化RAW目标的实际二阶导数。身份CE参考为同一L180真实逐包G.mean，两个目标均使用不乘新系数的导数，报告normratio与cos；没有optimizer或训练配方。

固定源规则：5个cohort全部完整，t1/t2 ANGLE导数5/5非零（数值界1e-12），汇总同批CE和ANGLE向量均非零、全局cos在两层都>=0，所有参数/buffer未变、无累积grad/头更新/非法数据访问。任一层不满足为科学FAIL，不扩展此固定RAW-angle公式或训练它；PASS也仅允许另外登记02/03同公式源诊断，不自动发布训练。零点/对象/批数/种子规则均在任何本run真实输入读回前固定。

预算：原checkpoint50轮历史成本；新增骨干训练0、optimizer0、头拟合0。实际900包eval前向、5个梯度批次、10个真实参数梯度矩阵、10层导数记录/30向量；另2个零IQ真实checkpoint smoke输入单记。原calculate也返回centered统计作为复用计算开销，不用其选新目标；RAW与ANGLE只有两个预定比较。保存actual_source_gradients.pt后，独立第二进程重算原/新loss、同批CE向量、900物理覆盖、层统计和源资格；所有PT/IQ/日志留不可变run根，不入Git。

普通N607既有CVS-RFFI/fullFP32，CUDA_VISIBLE_DEVICES空，OMP/MKL/OpenBLAS2线程，CPU-only不分配GPU、不占用/更改健康GPU任务；每GPU全owner最多3进程规则保持。CWD PROJECT/releases/cvs_source_variance_angle_20261015_r01，入口 python -u -m experiments.cvs_source_variance_angle.audit；不可变根 PROJECT/runs/20261015-phase1-source-backbone-variance-angle-manysig-s2026092701-r01，单owner codex/root/source-variance-angle-20261015。本地verifiedPython经原runner调用 experiments.cvs_source_variance_angle.publish --output local_artifacts/source_variance_angle_delivery_r01。完整路径/config/seed/来源见experiment.json。

只一次普通N607 preflight、单archive本地/远端SHA比较、remotecompile和真实CPnoquery smoke，发布后独立PID/CWD/argv/loggrowth核实。技术停止限run-owned角色/query/来源错误、非有限或不可识别方向、状态更新、预算/产物/独立复算错误、输出冲突/不可启动；保留partial/failure、不自动重试、不碰其他任务。源资格FAIL保留负结果，不改规则重跑。预期preflight/submit、row resolved_config/actual_source_gradients/analysis/completion及顶层source_qualification/independent_artifact_recount/analysis/completion；本run明确只做源诊断，不训练、不产生目标预测或truth评分。


源诊断已完成：真实900L/10矩阵/30向量独立复算VERIFIED，两层ANGLE对同批CE全局cos0.646206/0.593616、均5/5非零，源资格PASS。当前无新训练或目标性能结果；固定02/03同公式源扩展决定见source_extension_decision.json，尚未实现/发布。
