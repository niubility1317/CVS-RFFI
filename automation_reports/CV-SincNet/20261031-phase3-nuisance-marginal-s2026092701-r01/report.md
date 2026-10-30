# PAMF 源 RX／日期扰动积分

PLANNED，当前无新性能。Run `20261031-phase3-nuisance-marginal-s2026092701-r01`，实际代码 `93fc96a25b929c534e9b67570d4f2b38bc054c3e`，完整 Phase3 目标 ACTIVE。

唯一主 `marginal_predicted_ratio`，另列新 marginal_pooled 与原 pooled/predicted 两精确对照，共四方法×源星地DURA/跨RX cleanDURA/跨RX starDURA。方法、公式和固定验收见 `docs/PAMF_PHASE3_NUISANCE_MARGINAL_PLAN_20261031.md`；逐row矩阵、六类seed与命令唯一维护在 experiment.json。

只用原 r03 density160 的90已知均值，先去类均值、六类平均，估计15个RX/day offsets，C=offsets.T@offsets/15/allpositiveSVD≤14；每known/OE Gaussian积分新增δ∼N(0,C)，cov=vI+C，原mean/v/weight/whitener保持。不搜索rank/scale，不拟合目标偏移，不硬删除域方向。F0/default6、r03开放分支、DURA均冻结，来源与split392005/VALIDATED_ONCE句柄和源starP1祖先不变。

原密度祖先 L6300旧六各1050/每TXRXday70、OE6134仅outside-six/publicRXday；sourceRX1/3/4/6/8=1-19/18-2/19-2/2-19/3-19，2021-03-08/15/23。原Us56700未新读，L占L+Us10%，V为额外known校准。实际3CP合成2条smoke→源聚合prior冻→V27000各旧类4500拟合2新6τ/2旧exact→全部阈值冻→NV3150及两cachedQ→全部预测后独立CPU3truth。NV19-15/19-16/7-1各1050，同源已见RX，真值仅scorer。

两Q均RX24-1/index26/2021-03-15，同864物理记录：576known=六类×96、288unknown=三类×96；clean/star均原同DURA输出、原F0闭集保护基线，特征/身份逐值复用。U384/03-08、formalRX24-18不读；无目标标签/trueK/quota/Qfit/clean学习/DA/发现/匿名REG。sourceNV/devQ历史曝光、单seed/RX/日期重合，high未知0/4/2缺19-15保持NA，不重抽或认证完整六景。

主宏AK/RU/H，并列CK、同DURA原F0closed−AK、AUROC/OSCR、全TX/RX/day/scene/42groups与配对增失。gate源known12、cleanRU95/starRU90/CK90/损≤2pp/必需场景完整，并要求主starAK/RU均不低于旧predicted。sourceRU仅诊断，源失败不取消Q，不换主或回流调参。

新增NN/真实G/F0/signal前向0，实际合成各2；新sourceprior1/新V阈值12标量，原density/normalizer0refit，旧两τ精确。原NN/cache/星地生成成本继承；单部署834665NN参数，prior/全部τ字节按真实数组报告，不冒称零总成本或B1时延。

命令：`/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python -B -u -m experiments.pamf_phase3_open_world.star_nuisance_marginal_launch start --runtime /home/szu2070436088/2510044040/CV-SincNet/releases/20261031-phase3-nuisance-marginal-s2026092701-r01-release/runtime.json`。N607解释器 `/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python`，CWD `/home/szu2070436088/2510044040/CV-SincNet/releases/20261031-phase3-nuisance-marginal-s2026092701-r01-release`，唯一owner `codex-phase3-nuisance-marginal-20261031`，不可覆盖run `/home/szu2070436088/2510044040/CV-SincNet/runs/20261031-phase3-nuisance-marginal-s2026092701-r01`，logs同root/logs。GPU仅smoke/producer，free≥4096MiB且每卡新增本任务≤2，scorer独立CUDAempty；不停止/修改/重启其他owner。

本地聚焦检验详情见 verification.json，唯一直接P0/P1审查PASS；实际CP smoke是首步并在PASS后立即继续。单preflight/archive SHA比较/remotecompile/启动PID-CWD-argv-GPU-log核实。非法角色/祖先、非有限、输出碰撞或不能合法prediction保留并停止本run，不partial truth评分/覆盖/自动重启；UNKNOWN核原PID/root，科学负结果不停止健康任务。

预期launch/pipeline/configs/logs、predict smoke/complete、models density/prior/complete、源V calibration/thresholds、known/novel/两个Q的完整预测、独立score结果及observations。IQ、权重、预测、原日志和私有runtime不进Git。
