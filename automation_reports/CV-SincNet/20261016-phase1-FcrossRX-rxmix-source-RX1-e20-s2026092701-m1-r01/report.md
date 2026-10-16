# 有界接收RXMIX：完整源域结果

固定源规则 **FAIL**。一个新scratchE20，一个modelseed，held sourceRX1，0target。

|scene|C0 accuracy|RXMIX accuracy|Delta pp|C0 macroF1|RXMIX macroF1|
|---|---:|---:|---:|---:|---:|
|clean|83.444444%|84.148148%|+0.703704|0.832275|0.839713|
|practical_high|38.425926%|36.537037%|-1.888889|0.301789|0.289872|
|practical_mid|36.166667%|33.870370%|-2.296296|0.290926|0.277693|
|practical_low_suburban|34.037037%|32.333333%|-1.703704|0.280013|0.271361|
|practical_high_urban|37.000000%|34.907407%|-2.092593|0.290694|0.280101|
|practical_mid_urban|30.166667%|29.148148%|-1.018519|0.263720|0.258467|
|practical_low_urban|25.685185%|24.611111%|-1.074074|0.231182|0.223886|

固定clean>=+1pp/F1非负/seenV损失<=.5pp及完整健康判据：{'status': 'FAIL', 'checks': {'clean': False, 'F1': True, 'seen': True, 'complete_health': True}, 'clean_gain_pp': 0.7037037037037064, 'macroF1_gain_pp': 0.7438923750769377, 'seen_drop_pp': -0.04166666666667318, 'source_only': True, 'target_access': False, 'model_seeds': 1, 'automatic_extension': False, 'goal_completed': False}。不调bound/ratio/coordinate/RX/seed救援。

新增20模型轮/4440更新/3579018网络包，墙钟1171.774131s。C0原20轮独立列，不继承权重。全部5994primary L/U记录逐条相同。

独立复放568320次有界RXMIX，5040物理包、240个epoch/RX/day session；私有源profile只用于幅度验证、不提交Git。已知响应逆与逐点能量界为数值证据，不证明身份保持或硬件因果。

固定F+CROSS_RX/默认头联合训练、第一对SupCon、原U/ownEMA/旧门控/cos不改。0.5RD ordinary+0.5RD RXMIX(sameordinary)，4L前向，一次ordinary，无新增损失/部署头/As。

完整七源视图、20轮/4440步日志、全部TX/day/state/quality、修复/新错、固定C0clean正确集合保持率、同scale30间隔及曲线保留；未改正式IQ/channel/SNR。单seed源证据不是正式target结果或跨seed复现；真正Ag/As四格、low>60%及suburban接近自身clean仍未证明。

实际终止状态：**VERIFIED／ANALYZED，FAIL_SOURCE_RX1_ONLY**。唯一worker和dispatcher自然完成，0技术failure／重试／热改；实际训练代码 `772046d5758bbecf0a98d12542350f9862852eb6`。

数据：fit RX3/4/6/8 的 L5040／隐藏U45360／seenV21600，heldRX1 L/U不训练、heldV5400仅FINAL20。与原全source L6300/U56700/V27000或内部5580/720分别说明。scratch E20=13L＋7U／4440更新，父权重／EMA／optimizer继承0；C0既有同合同own scratch FINAL20只复用结果／日志／物理顺序。未加载E50或E200；源V曾用于多个候选，保留探索背景，不称独立盲测。

F三路SIG_TIME_X3／CENTER_25K／CENTER_UNITS、25MHz／IQ256／equalized1／原nativeRMS center=false／NoExplicitCFO／joint160／defaultcos30联合训练不改。原L-only time_down/t2.pool跨RX同TX SupCon温度.2、最大.1、warm5/ramp10，只读第一对原RDrop；原U／ownEMA／旧门控／cos及strongnoise .015不改。每对同IQ两dropout、LS .01／RDrop alpha1，.5RD(原ordinary)+.5RD(RXMIX同ordinary)，四L前向／一次ordinary／原物理样本和总损失尺度匹配C0。所有合法L视图均监督，无新损失、分类头或部署模块。

新增算子仅 `y=(x+b*conj(x))/sqrt(1+abs(b)**2)`。实际六源幅度项来自已登记fit4RX椭圆性profile的两两绝对差/2再cap .02，私有profile及bank不提交Git。参数按physicalRX/day虚拟session跨TX稳定、每epoch刷新；20×4×3共240 session、568320次IQ变换和5040物理包、全部TX/RX/day/amplitude覆盖及5994原L/U批次独立重放。幅度最大 0.01642912927936461，实际已知逆最大误差 1.52587890625e-05、逐点能量界浮点误差 5.7220458984375e-06。私有种子20261016921不读TX/loss/quality/预测、不推进全局RNG，不反演原录音未知TX、不变边界、采样率或逐包RMS。数值可逆与覆盖完整只证明执行，不证明身份保持或RX硬件因果。

源统计的先验限制继续保留：12/18正折、聚合风险降低27.486642%、RX差p95 .032791289171；TX index1/3全部6折为负。默认头实际识别才决定本候选是否有效，不以统计PASS、辅助损失、梯度非零或门控接受量替代性能。

同C0的实际增量：clean +0.703704 pp、clean宏F1 +0.743892 pp；urban三场景均值 -1.395062 pp、mid -1.018519 pp、low -1.074074 pp。此处仅一个源heldRX1／一个modelseed，不是正式目标结果或最差五RX结论。

|arm|clean|high距clean|mid距clean|low suburban距clean|low urban|
|---|---:|---:|---:|---:|---:|
|C0|83.444444%|45.018519 pp|47.277778 pp|49.407407 pp|25.685185%|
|RXMIX|84.148148%|47.611111 pp|50.277778 pp|51.814815 pp|24.611111%|

|TX index|clean修复|clean新错|clean Δ pp|low urban Δ pp|
|---|---:|---:|---:|---:|
|0|28|6|+2.444444|-0.666667|
|1|1|3|-0.222222|+0.111111|
|2|28|7|+2.333333|-8.444444|
|3|1|7|-0.666667|-1.000000|
|4|0|0|+0.000000|+1.888889|
|5|3|0|+0.333333|+1.666667|

全部TX/day/state/lock/quality／有效观察相关统计、固定C0 clean正确集合的satellite保持率、同cosine30间隔、门控／融合轨迹、默认头梯度及完整20轮／4440步原生CSV/stdout/SupCon/Fclip均保存并独立核实。单个子集修复不能替代总正确数；无单因素物理因果或强不可恢复性声明。新增noise SNR不是总SNR。

|arm|新增或复用|模型轮|更新|网络包|训练/源V 秒|六practical 秒|RXMIX构造 秒|
|---|---|---:|---:|---:|---:|---:|---:|
|C0|复用结果|20|4440|3579018|1123.572077|64.008927|0.000000|
|RXMIX|新增|20|4440|3579018|1096.708680|69.166357|8.124450|

新增20模型轮／4440更新／3579018网络包，矩阵墙钟1171.774131s；C0原E20单列。RXMIX构造含算子、数值核对、metadata与trace，不含额外网络；所有真实前向费用按保存计数重算，预算耗尽不等于收敛。先验椭圆性CPU统计1.737601s另列，0网络／更新；关闭的DC统计1.421083s另列，不能把其失败隐去。离线分析0新IQ／checkpoint／网络，CPU分析总墙钟未统一计时。

本地合成验证1024网络包／2更新／16协议负例、206初态／原默认头SIG3Sinc梯度／4L／SupCon首对／同primary IQ／私有session与能量界通过；5120仅指独立session回放算子包，另128直接公式包及256集成L算子包，总5504算子包，不与网络前向数混算。0真实IQ／checkpoint。隔离真实参数与C0／源统计metadata、污染拒绝PASS；唯一限定P0/P1只读审查PASS。N607普通账户preflight／单archive比较／remotecompile、真实own scratchCP no-query smoke与PID/CWD/argv/GPU/log增长VERIFIED。离线读取器16元数据负例／384合成trace包，0网络／IQ／CP。

评价器实际导入__file__：

- channel: `/home/szu2070436088/2510044040/CV-SincNet/releases/cvs_SIGTIME_evidence_responsibility_train20_20261015_r02/experiments/adv3b02_xuc/code/leo_practical/channel.py`
- residual: `/home/szu2070436088/2510044040/CV-SincNet/releases/cvs_SIGTIME_evidence_responsibility_train20_20261015_r02/experiments/adv3b02_xuc/code/leo_practical/residual.py`

全部六实际asdict：fs25MHz／residual／post_sync／EQoff即G1／reference30／quality_adaptation=true、quality_reference20／phase_tracking=true／IQcomp=true／reflect，与C0每个物理ID和physics元数据一致。没有As训练，未修改正式IQ／SNR／信道／困难样本，0target／正式truth-last。

固定源判据：{"status": "FAIL", "checks": {"clean": false, "F1": true, "seen": true, "complete_health": true}, "clean_gain_pp": 0.7037037037037064, "macroF1_gain_pp": 0.7438923750769377, "seen_drop_pp": -0.04166666666667318, "source_only": true, "target_access": false, "model_seeds": 1, "automatic_extension": false, "goal_completed": false}。关闭固定RXMIX，不调幅度／概率／比例／坐标／RX／seed或E20预算救援。 一个modelseed不构成跨seed复现；联合四格交互、正式七视图、lowurban>60%与suburban接近自身clean仍未证明，目标保持active。

精确row／command／environment／CWD／paths／GPU／owner／stop见experiment.json和preregistration.md。实际图表（已查看3幅图）：

- [complete_training_curves.png](E:/type10-7/local_artifacts/cvs_FcrossRX_rxmix20_20261016/final/complete_training_curves.png)
- [source_scene_results.png](E:/type10-7/local_artifacts/cvs_FcrossRX_rxmix20_20261016/final/source_scene_results.png)
- [low_urban_strata.png](E:/type10-7/local_artifacts/cvs_FcrossRX_rxmix20_20261016/final/low_urban_strata.png)
- [scene_results.csv](E:/type10-7/local_artifacts/cvs_FcrossRX_rxmix20_20261016/final/scene_results.csv)
- [source_strata.csv](E:/type10-7/local_artifacts/cvs_FcrossRX_rxmix20_20261016/final/source_strata.csv)
- [paired_strata.csv](E:/type10-7/local_artifacts/cvs_FcrossRX_rxmix20_20261016/final/paired_strata.csv)
- [training_curves.csv](E:/type10-7/local_artifacts/cvs_FcrossRX_rxmix20_20261016/final/training_curves.csv)
- [fusion_gain_curves.csv](E:/type10-7/local_artifacts/cvs_FcrossRX_rxmix20_20261016/final/fusion_gain_curves.csv)
- [measured_costs.csv](E:/type10-7/local_artifacts/cvs_FcrossRX_rxmix20_20261016/final/measured_costs.csv)
