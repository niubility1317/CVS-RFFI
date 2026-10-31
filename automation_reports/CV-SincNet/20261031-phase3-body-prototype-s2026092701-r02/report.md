# PAMF 原型骨干加载技术恢复

PLANNED：r01完整训练精确复用，r02新神经更新0，尚无性能结论，完整目标ACTIVE。

Run `20261031-phase3-body-prototype-s2026092701-r02` / evaluationcode `4273e88392abe0717c1bd94fca1370d19c67cbb3`，训练继续为r01/a336、SOURCE/TARGET各444原final.pt。严格run/code/root身份分离，唯一PCD_predicted_ratio/六列三DURA/原36τ规则和科学门槛保持；先checkpoint smoke，再sourcefits/V冻结/allpred，再CPU三truth。完整公式/数据/祖先/停止规则见`docs/PAMF_PHASE3_BODY_PROTOTYPE_RECOVERY_20261031.md`和原计划，六类seed与矩阵在experiment.json。

Environment `/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python`；CWD `/home/szu2070436088/2510044040/CV-SincNet/releases/20261031-phase3-body-prototype-s2026092701-r02-release`；owner `codex-phase3-body-prototype-recovery-20261031`；command `/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python -B -u -m experiments.pamf_phase3_open_world.star_body_prototype_launch start --runtime /home/szu2070436088/2510044040/CV-SincNet/releases/20261031-phase3-body-prototype-s2026092701-r02-release/runtime.json`。新root `/home/szu2070436088/2510044040/CV-SincNet/runs/20261031-phase3-body-prototype-s2026092701-r02`，只predict→score，拒绝新训练命令。原失败未读Q/truth，目标真值/K/配额不进入算法。
