# PAMF Phase3 默认头开放暴露与相对结构保持

## 预登记（2026-10-30）

Run `20261030-phase3-star-relation-oe-s2026092701-r01`，code `19691a2cad35dcef6dd1ae42973bfba0e14ea705`。唯一主relation_oe::entropy_global，控制relation_control。完整固定机制/数据/主指标/停止条件见`docs/PAMF_PHASE3_STAR_RELATION_OE_PLAN_20261030.md`、`configs/pamf_phase3_star_relation_oe_20261030.json`，矩阵唯一维护在experiment.json。

两臂分别从原Tri-SIG-TIME SSL F0 E50/11100/default6head/232993独立开始，nativeSinc骨干与默认头联合2220次更新、累计13320（非自然10epoch），fresh CENTER_UNITS AdamW lr1e-4/继承parent wd/clip5，eval BN/dropout/固定buffer。L128 IIDseed2026092701、OE128域均匀seed2026102907，concat256单前向。两臂CE＋T2logitsKD×4＋OE offdiag Gram MSE权重1；唯一主加默认头KL(uniform6||p)权重1，控制该项0。无逐向量保真/aux/VIC/增广/U/PL/EMA/G更新，不扫权重或另换分数。

Teacher仅读旧jointOVA r02/5aa041122396d48c49b8cb3d5912649bdc3217e1的原F0 OE单位特征，非训练后joint模型；L teacherlogits来自同P1缓存。teacher原导出和G缓存成本保留。L6300/OE6134（membership额外源监督）/V27000/sourceNV3150，五已见RX1/3/4/6/8、三日、G10MIX01 P1。旧六标签合法，OE TX/K、Us隐藏真值未解封；NV三个历史设备各1050，仅冻结后独立评分，非新TX盲测。

两臂源训练固定后，knownV仅标量阈值：默认头1−H(p)/log6，单一全局最大合法观测值/完整ties，overall+六主景7组宏CK90及原同P1F0closed−AK≤2pp，其他35/42组诊断，不再加共同tail。闭集损伤已超限则finiteacceptall+known失败，仍预测sourceNV和开发RX24-1 Q864。sourceRU90不前置，两臂不按返回score变化。全部本row源/目标prediction固定后独立CPUscorer重建公有logits/argmax/entropy/margin/known7校准，再连接两truth。

目标RX24-1（源RX互斥）、G10现有futureQ864/2021-03-15，Q-only前向、无U/Q拟合、无未知TX/K/配额输入、正式RX24-18未接触。Q真实old576/unknown288和三未知角色仅scorer。既有U384保持未读；本轮只有DA0_REG0，AN/HOW=N/A/完整goalACTIVE。主报跨RX宏AK/RU/H、CK、原F0损、candidateclosed、AUROC/OSCR、六scene/逐类/CI及两臂同ID配对；七目标组同时CK90/RU90/F0loss2pp才为下一独立UDA资格，控制不替主，不宣称完整目标或新盲测。

### 执行、存储与资源

- N607 Python `/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python`；CWD `/home/szu2070436088/2510044040/CV-SincNet/releases/20261030-phase3-star-relation-oe-s2026092701-r01-release`；唯一owner `codex-phase3-star-relation-oe-20261030`。
- 命令 `/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python -B -u -m experiments.pamf_phase3_open_world.star_relation_oe_launch start --runtime /home/szu2070436088/2510044040/CV-SincNet/releases/20261030-phase3-star-relation-oe-s2026092701-r01-release/runtime.json`。
- 独占 `/home/szu2070436088/2510044040/CV-SincNet/runs/20261030-phase3-star-relation-oe-s2026092701-r01`；每row train→完整source+target eval→独立CPUscore，GPU最多2row分别不同卡，实际free≥4096MB、每卡≤2新增own，foreign不停止/修改/重启。evalconfig13字段无truth/U/K/roles；两truth只runtime/controller和CPUscorerargv，producer不接收。
- 原sourcecache/L/OE/teacher只用于训练；冻结推理loader不回读逐样本源features。单模型参数F232993+G266951=499944、无aux；完整native训练wrapper900722另有不用的域分支，不将其误计当前识别器部署结构。singleτ8B，训练checkpoint含optimizer不当精简结构收益。
- 预期train final/resolved/summary/2220steps/smoke/complete；predict known/NV/targetNPZ、threshold.npy/calibration/smoke/targetsmoke/complete；score results/complete；pipeline/rowstate/logs/observations。新IQ/weights/pred/rawlog只private、不进Git。
- 每臂L/OE各284160曝光、训练F568320真实行；evalcandidate31014真实+4合成、originalF0额外4014真实，train烟测2合成。两臂完整F预算1206708行（真实1206696+合成12）；G新增1732（真实1728+合成4），缓存与教师祖先另计。新消费同42584源+864target=43448既有物理记录，不计额外独立仿真样本，不预报B1时延。
- 系统技术失败只保留failedrow、healthyrow继续；科学失败不重试/早停/重采样/重选阈值，UNKNOWN先核对同PID/root再决定，不重复launch或触碰foreign。

验证：26 passed, 232 warnings, 41 subtests passed in 7.57s，10AST/实际runtime PASS；唯一独立直接P0/P1 PASS，未重复主suite或接触真实IQ/权重/真值。一次N607 preflight核对原F0/teacher/source句柄与目标VALIDATED_ONCE/capsule/split，未读IQ或truth值；8卡实际free14797–20268MB，无foreign干预。单codecommit、单archive比较/compile、actualcheckpoint合成G10首步梯度smoke与launch后PID/CWD/argv/GPU/log增长、prediction全完后truthlast；不另加审批、封印或重复数据验证。
