# PAMF 非平衡传输混合先验适应

PLANNED，无新性能。完整目标 ACTIVE。

Run `20261031-phase3-uot-prior-s2026092701-r01`，actualcode `c0b7cc09a808e832d4d64a18080239e72e910250`。唯一主UOT_predicted_ratio，四列×三DURA、公式和固定参数/数据/成本/停止规则见 `docs/PAMF_PHASE3_UOT_PRIOR_PLAN_20261031.md`，逐row/六类seed在experiment.json唯一维护。

N607解释器 `/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python`，CWD `/home/szu2070436088/2510044040/CV-SincNet/releases/20261031-phase3-uot-prior-s2026092701-r01-release`，唯一owner `codex-phase3-uot-prior-20261031`，不可覆盖root `/home/szu2070436088/2510044040/CV-SincNet/runs/20261031-phase3-uot-prior-s2026092701-r01`。命令 `/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python -B -u -m experiments.pamf_phase3_open_world.star_uot_prior_launch start --runtime /home/szu2070436088/2510044040/CV-SincNet/releases/20261031-phase3-uot-prior-s2026092701-r01-release/runtime.json`。

F0/defaulthead/r03/G/原density11和两组六τ不改。actual3CPsynthetic2首步→sourcefreeze→缓存starU384执行唯一200步UOT→47先验权重冻结→source/clean/star全四列→独立CPU3truth/Utruth0。source RX原先验，newRX两Q共用starU权重。新NN/源GMM/τ/真实GFsignal/源IQ0，目标先验fit1次/47数值权重45自由度；派生adapter不是source-only，不称完整UniOT复现。sourceknown12/clean95star90/CK90/sameG闭集损2pp/coverage及starAKstrictgain/RU不退步固定；source科学失败仍Q。原所选基线保持，历史曝光/单RXseed/high缺19-15宏RU/HNA/正式RX未访问；无发现REG/ANHOW为NA。
