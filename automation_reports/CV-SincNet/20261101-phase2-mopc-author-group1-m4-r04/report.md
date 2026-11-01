# MoPC原生Group1 CUDA故障后复用恢复

GPU环境已恢复，8卡可识别且独立CUDA allocation/synchronization通过。本次为技术故障后的独立恢复，不更改原科学方法、矩阵、种子、数据权限或预算；已完成结果优先复用，不覆盖原root。

User2026-11-01 continue originalGroup1 remaining comparisons after CUDA restore; reuse four completedMoPCsourceA,398predictions,397scores; no re-training or method/budget changes

Author xmuLdz/MoPC-HR ae6554316ad1a2175920e330133a2f103408bf78; preserve 10 attention residualblocks and author adaptivepool; input onlyI/Q/amplitude eachrecord256, noresampling/multirecordmerge.

999339parameters sixclassF, 256embedding; sourcecheckpointmustexactarchitecture recipe, rejectoldgeneric MoPC encoder.

NoB10/oldtargetlabels; A frozen6x256cleanL_sprototypes fixed before target, nativeCE+protoaugmentation+squaredHR+momentum prototypecorrection, paper semantics.

Freshsourceaugmentationservice/cache ownroot; currentexternal/r42owners untouched. NoISSLrestart, noF/Gtraining. This modelisexternalMoPC, notPAMFbase replacement.

r01 failed before completed training due omitted cache import; r02 adds exact committed source augmentation dependencies only. Same method/matrix/seeds/budgets and fresh scratch; no failed-source weights loaded.

r02 actual activate check stopped before launch due missing compat caused by partial dependency package shadowing; r03 includes full committed practical_full_2x2 package, same scientific recipe and fresh roots.

CUDAoutage recovery only: current8GPUs listed and freshCUDAallocation/synchronizationPASS. No scientific method, seed, target split, matrix, budget or precision change. Every closed prediction/score reused, incomplete partials retained without overwrite. No admin action or unrelatedowner modification.

Reuse fourfinalE200/10000step sourceA models,398completepredictiondirs and397score dirs by actualmarkers; oldsource marker retained separately, newmarker neveroverwrites hardlinkedparentJSON.

精确命令、环境、逐行/分组矩阵、数据/源模型来源、输出、GPU与停止规则见experiment.json。实际运行先执行当前阶段所需smoke，再自动补齐预测与独立评分，最后分析/资源表/登记与Git交付。当前RUNNING，独立读回已验证；不宣称性能或完成。

启动后独立核实（2026-11-01T13:21:44.188663+08:00）：VERIFIED RUNNING: fourinheritedactualsource checkpointsmokesPASS, sourceupdates0;398predictiondirs/1172rows and397scoredirs/1170rows reused; newtargetpredictionandindependentscore processes/loggrowth observed; originalnativeC20epochs/5classsession unchanged.

控制器实际PID/CWD/argv与子进程、GPU、日志/预测/拟合产物增长已核实。源代码/config commit b4903b6928d88f755b6609a6160a0dde0acf22b0，双仓库push和remoteOID VERIFIED；单一归档传输SHA比对、远端compile通过。25项定向测试和每个恢复入口唯一一次P0/P1均通过，无额外审查/签名/成员hash。全部矩阵仍在执行，宏H与+5pp逐格结论待最终独立评分。
