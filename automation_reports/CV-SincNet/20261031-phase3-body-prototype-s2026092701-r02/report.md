# PAMF 原型骨干加载技术恢复

VERIFIED / ANALYZED_DEVELOPMENT：r01完整训练精确复用，r02新增优化0，全部预测和独立评分完成；科学NO_PROMOTION，完整目标ACTIVE。

Run `20261031-phase3-body-prototype-s2026092701-r02` / evaluationcode `4273e88392abe0717c1bd94fca1370d19c67cbb3`，训练继续为r01/a336、SOURCE/TARGET各444原final.pt。严格run/code/root身份分离，唯一PCD_predicted_ratio/六列三DURA/原36τ规则和科学门槛保持；先checkpoint smoke，再sourcefits/V冻结/allpred，再CPU三truth。完整公式/数据/祖先/停止规则见`docs/PAMF_PHASE3_BODY_PROTOTYPE_RECOVERY_20261031.md`和原计划，六类seed与矩阵在experiment.json。

Environment `/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python`；CWD `/home/szu2070436088/2510044040/CV-SincNet/releases/20261031-phase3-body-prototype-s2026092701-r02-release`；owner `codex-phase3-body-prototype-recovery-20261031`；command `/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python -B -u -m experiments.pamf_phase3_open_world.star_body_prototype_launch start --runtime /home/szu2070436088/2510044040/CV-SincNet/releases/20261031-phase3-body-prototype-s2026092701-r02-release/runtime.json`。新root `/home/szu2070436088/2510044040/CV-SincNet/runs/20261031-phase3-body-prototype-s2026092701-r02`，只predict→score，拒绝新训练命令。原失败未读Q/truth，目标真值/K/配额不进入算法。

## 完整结果与判定

source_DURA: 89.740741/96.729630/65.523810/75.743693%; F0loss1.240741pp；clean_DURA: 73.090278/89.756944/99.305556/84.204711%; F0loss6.944444pp；star_DURA: 65.277778/87.152778/78.125000/71.125908%; F0loss7.638889pp。主star对E0pred正确净+7/接收+9/未知拒对+3，对SOURCEpred正确+5/接收+4/未知+8；AK同时高于两对照且RU均不降，存在开发池相对改善。clean正确+13、未知拒对−2；源正确−1、未知+113（均对E0）。sourceknown12通过，但CK/星地RU/同DURA原F0损2pp及coverage仍失败，不晋级、不扩seed、不改原主基线，AN/HOW NA。

原r01失败和一次修复保留，r02实际checkpoint/noquery smoke及strictreload成功，无重复故障。训练仍a336/r01，eval为4273/r02；r02无trainjob/optimizer0，继承原两444 body/projection更新，Sinc/F0head/G0。总真实signal557826、G2112、F01728，含训练父12818/学生454656及eval90352，原权重和数据不转存Git。

独立driver1415050/predict1415080 live/CWD/fullargv/CUDA1，GPU304/392MiB离散非峰值，predictlog509→1597B；CPU1415842实际live CUDAempty/CWD/fullargv，终态owned空/foreign0。r02过程秒数{"predict": 38.864556, "score": 12.976472, "pipeline": 51.929297}，不是B1时延。718files/data0单archiveSHA/compile/runtime VERIFIED，初审及唯一原问题定点复审PASS。完整888步、18点、配对、场景和成本见`docs/PAMF_PHASE3_BODY_PROTOTYPE_RESULTS_20261031.md`；future3-2仅metadata容量候选、未建池/未用于本轮。
