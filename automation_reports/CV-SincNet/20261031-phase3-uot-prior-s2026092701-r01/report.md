# PAMF 非平衡传输混合先验适应

VERIFIED / ANALYZED，四列×三DURA及独立三truth评分全部完成。科学结论 NO_PROMOTION，完整目标 ACTIVE。

Run `20261031-phase3-uot-prior-s2026092701-r01`，actualcode `c0b7cc09a808e832d4d64a18080239e72e910250`。唯一主UOT_predicted_ratio，四列×三DURA、公式和固定参数/数据/成本/停止规则见 `docs/PAMF_PHASE3_UOT_PRIOR_PLAN_20261031.md`，逐row/六类seed在experiment.json唯一维护。

N607解释器 `/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python`，CWD `/home/szu2070436088/2510044040/CV-SincNet/releases/20261031-phase3-uot-prior-s2026092701-r01-release`，唯一owner `codex-phase3-uot-prior-20261031`，不可覆盖root `/home/szu2070436088/2510044040/CV-SincNet/runs/20261031-phase3-uot-prior-s2026092701-r01`。命令 `/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python -B -u -m experiments.pamf_phase3_open_world.star_uot_prior_launch start --runtime /home/szu2070436088/2510044040/CV-SincNet/releases/20261031-phase3-uot-prior-s2026092701-r01-release/runtime.json`。

F0/defaulthead/r03/G/原density11和两组六τ不改。actual3CPsynthetic2首步→sourcefreeze→缓存starU384执行唯一200步UOT→47先验权重冻结→source/clean/star全四列→独立CPU3truth/Utruth0。source RX原先验，newRX两Q共用starU权重。新NN/源GMM/τ/真实GFsignal/源IQ0，目标先验fit1次/47数值权重45自由度；派生adapter不是source-only，不称完整UniOT复现。sourceknown12/clean95star90/CK90/sameG闭集损2pp/coverage及starAKstrictgain/RU不退步固定；source科学失败仍Q。原所选基线保持，历史曝光/单RXseed/high缺19-15宏RU/HNA/正式RX未访问；无发现REG/ANHOW为NA。

## 已完成结果

唯一主UOT_predicted_ratio宏AK/CK/RU/H：source: 89.744444/96.759259/61.936508/73.291437%；clean_DURA: 70.659722/87.152778/100.000000/82.807731%；star_DURA: 64.062500/85.590278/77.430556/70.115031%。原F0同G损失source/clean/star为1.237037/9.375000/8.854167个百分点。

对所选predicted_ratio同ID：source无任何变化；clean正确接收净−1、未知拒对0；star已知接收和正确接收不变，未知拒对1增0失/net+1（RU+0.347222pp/H+0.142706pp）。未取得预登记starAK严格收益，绝对CK/RU/2pp/coverage仍失败，不以1样本收益换主。原selected baseline与pooled对照保持，不能从本候选外推完整UniOT无效。

原F0/defaulthead/r03/G/11arrays/two6tau不改，sourceRX不用target先验；U384只做一次200步UOT，新47权重/45自由度，整体target派生非source-only。源几何GMM与阈值新拟合0不代表没有目标密度先验学习。KKT2.1826436214e−9/scaling2.2930457533e−9，已知transport fraction0.6356480为模型统计非真实known比例。新NN/realGFsignal/sourceIQ0，三CPsynthetic2各PASS，Utruth0。

697files/data0单archiveSHA/compile/runtime VERIFIED；driver1341003/predict1341026 live/CWD/fullargv/CUDA1和UUIDa2d039ff的406MiB两离散样本（非peak）、log472→497→599B，CPU未独立live；终态owned空foreign0。11unique33sub/workerAST6+launchimport/一次P0P1通过。详细数据、12点、分组和费用见`docs/PAMF_PHASE3_UOT_PRIOR_RESULTS_20261031.md`及private score/results.json/observations.jsonl，原历史曝光/high缺19-15NA/正式RX未读边界不变。
