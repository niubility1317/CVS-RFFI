# PAMF 多模态共同方差密度比

PLANNED，无新性能。完整 Phase3 目标 ACTIVE。

Run `20261031-phase3-tied-density-s2026092701-r01`，actual code `9a80e2bef7eee4d8c2ec4d4725e0ca552552443b`。方法、四列×三DURA矩阵、数据角色、成本和停止规则完整记录于 `docs/PAMF_PHASE3_TIED_DENSITY_PLAN_20261031.md`；逐row/六类seed在experiment.json唯一维护。

N607 CWD `/home/szu2070436088/2510044040/CV-SincNet/releases/20261031-phase3-tied-density-s2026092701-r01-release`，解释器 `/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python`，唯一owner `codex-phase3-tied-density-20261031`。命令 `/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python -B -u -m experiments.pamf_phase3_open_world.star_tied_density_launch start --runtime /home/szu2070436088/2510044040/CV-SincNet/releases/20261031-phase3-tied-density-s2026092701-r01-release/runtime.json`。不可覆盖run root `/home/szu2070436088/2510044040/CV-SincNet/runs/20261031-phase3-tied-density-s2026092701-r01`。

Uniqueprimary tied_predicted_ratio，原selected predicted_ratio/pooled_ratio精确控制。仅共享原pooled_L_variance评分，不新拟合方差/密度/NN，不读取L/OE/Us IQ或目标U；原source聚合、F0/G/r03冻结，V27000仅12newtau。实际3CPsynthetic2首步后继续；全4x3prediction后独立CPU3truth。source失败仍测Q；所有历史曝光和high缺未知类NA保持。聚焦验证/一次P0P1结果见private verification/review，原基线不替换、不得按Q调参。
