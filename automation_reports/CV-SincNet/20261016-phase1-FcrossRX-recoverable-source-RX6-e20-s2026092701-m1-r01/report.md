# 可恢复状态课程：完整源域结果

固定源规则 **FAIL**。一个新scratchE20，一个modelseed，held sourceRX6，0target。

|scene|C0 accuracy|C1 accuracy|C2 accuracy|C2-C0 pp|C2-C1 pp|
|---|---:|---:|---:|---:|---:|
|clean|81.129630%|77.481481%|77.203704%|-3.925926|-0.277778|
|practical_high|40.351852%|76.685185%|76.907407%|+36.555556|+0.222222|
|practical_mid|38.481481%|73.055556%|73.203704%|+34.722222|+0.148148|
|practical_low_suburban|36.574074%|69.500000%|69.703704%|+33.129630|+0.203704|
|practical_high_urban|38.055556%|72.833333%|73.166667%|+35.111111|+0.333333|
|practical_mid_urban|32.444444%|57.388889%|58.388889%|+25.944444|+1.000000|
|practical_low_urban|27.592593%|47.796296%|47.851852%|+20.259259|+0.055556|

固定clean/seen保留且urban/low/F1不低于C1、实际激活、全健康判据：{'status': 'FAIL', 'checks': {'clean': False, 'seen': True, 'urban': True, 'low': True, 'urban_F1': True, 'activation': True, 'complete_health': True}, 'clean_drop_vs_C0_pp': 3.9259259259259216, 'seen_drop_vs_C0_pp': 0.24537037037036802, 'urban_gain_vs_C1_pp': 0.462962962962965, 'low_gain_vs_C1_pp': 0.05555555555555314, 'urban_F1_gain_vs_C1_pp': 0.44514251428244833, 'source_only': True, 'target_access': False, 'model_seed_count': 1, 'source_fold_count': 1, 'automatic_extension': False, 'goal_completed': False}。FAIL关闭固定课程，不扫描eta/范围/warm/proposals/比例/RX/seed救援。

新增20模型轮/4440更新/3579018网络包，矩阵墙钟2892.487447s；原C0/C1共40模型轮只复用结果，无父权重。

全部5994条primary L/U序列逐条相同，568320次实测课程视图与有限提案独立复放。固定F+CROSS_RX/defaulthead、4L前向、原U/EMA/旧门控/cos不变；课程只有fitL进展、无新增损失或部署模块。

一个已知失败源RX6/一个modelseed，不能证明其他源RX或跨seed；无正式target评分，不宣称low>60%或联合Ag+As目标已完成。新增噪声比不是总SNR，质量代理与原状态计算不改。分层关联不是物理因果。

实际终止状态：**VERIFIED／ANALYZED，FAIL_SOURCE_RX6_ONLY**。唯一worker和dispatcher自然结束，0技术failure／重试／热改；实际训练代码 `a47c7302eefc485e75703a78c749f24144e5b77b`。独立读取器和发布记录的后续提交不冒充训练代码。

数据：fit RX1/3/4/8 的 L5040／隐藏U45360／seenV21600，heldRX6 L/U不训练、heldV5400只在FINAL20使用。与全source L6300/U56700/V27000或旧内部5580/720区分；scratch E20=13L＋7U／4440更新，父权重／EMA／optimizer继承0。原同合同C0/C1各E20只复用结果、完整日志、顺序与physics，不加载权重。无E50/E200；RX6是先前已暴露的源失败折，本轮不是独立盲测或五源确认。

固定F三路SIG_TIME_X3／CENTER_25K／CENTER_UNITS、fs25MHz／IQ256／equalized1／nativeRMS center=false／NoExplicitCFO／joint160／defaultcos30联合训练。L-only time_down/t2.pool跨RX同TX SupCon温度.2／最大.1／warm5/ramp10，只读第一对原RDrop；原U／ownEMA／旧门控／cos／strongnoise .015不变。.5RD(原ordinary)+.5RD(full practical(raw))，每对同IQ两dropout、LS.01／RDrop alpha1，4L前向／一次ordinary；不叠ordinary PA/AWGN/multipath到As前。无新损失、分类头、部署模块或Ag。

本轮是训练状态分层课程的独立检验：p=(1-eta)nominal+etaQ4，eta≤.5。保留至少.5 nominal，nominal仍均匀low_suburban/mid_urban/low_urban；hard仅mid/low urban、locked/degraded且新增噪声比[3,15]dB，新增噪声比不等于总SNR。每包先生成C1原nominal，hard分支若不合格最多追加4个独立完整urban提案；全失败回到最初nominal。Q4是有限提案的实际分布，不冒称无限条件采样。每物理L仍有一个直接有标签satCE，无学生置信度过滤／样本删除，不调弱物理参数。硬件按RX/day/eq跨TX稳定，训练stream与验证分离。

前5轮eta0。随后每轮只用上轮fitL已有primary默认头logits的72个TX/RX/day单元准确率、margin/30、CE及原native梯度决定是否+.1：accuracy≥.7、p10accuracy≥.6、p10margin≥.02、CE≤1、梯度≤5，且相对前3轮中位数CE≤1.05倍、margin下降≤.01、梯度≤2倍；否则保持。无heldV、seenV或Utruth参与课程，无额外网络前向。

实际课程独立复放：{"status": "VERIFIED", "steps": 4440, "physical_IDs": 5040, "packets": 568320, "proposal_packets": 1046334, "hard_requested": 184198, "hard_accepted": 116502, "hard_fallback": 67696, "nominal_selected": 475718, "eta_used": [0.0, 0.0, 0.0, 0.0, 0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5], "eta_final": 0.5, "source_cells_per_epoch": 72, "all_labeled_physical_samples_retained": true, "analysis_IQ_checkpoint_network_access": 0}。完整20轮／4440步、5994条原L/U序列、所有568320选择／提案计数、72单元进展及eta公式均核对；全部TX×RX×scene和LOS/PARTIAL/BLOCKED支持保留。被拒的是模拟提案，不是物理训练样本，0丢弃。执行正确及梯度非零不代替识别收益。

实际替换原nominal的视图为92602/568320（16.293989%）；hard分支请求184198，接受116502，有限提案失败后回原nominal 67696。这说明课程已实际执行；不据此追加提案数、改变质量区间或宣称增加采样比例就会有效。

实际默认头净增量：相对C0 clean -3.925926 pp、clean宏F1 -2.887054 pp、urban均值 +27.104938 pp、mid +25.944444 pp、low +20.259259 pp；相对C1 clean -0.277778 pp、urban +0.462963 pp、low +0.055556 pp。

|arm|clean|high距clean|mid距clean|low suburban距clean|low urban|
|---|---:|---:|---:|---:|---:|
|C0|81.129630%|40.777778 pp|42.648148 pp|44.555556 pp|27.592593%|
|C1|77.481481%|0.796296 pp|4.425926 pp|7.981481 pp|47.796296%|
|C2|77.203704%|0.296296 pp|4.000000 pp|7.500000 pp|47.851852%|

C2的三个suburban距clean约.296296／4.000000／7.500000 pp，后两项不满足探索目标2pp。部分差距缩小来自clean下降，不能称为clean与星地共同改善。

|low urban分层|样本数|C0准确率|C1准确率|C2准确率|C2-C1 pp|
|---|---:|---:|---:|---:|---:|
|LOS|1674|39.068100%|77.120669%|77.120669%|+0.000000|
|PARTIAL|766|30.287206%|63.315927%|63.054830%|-0.261097|
|BLOCKED|2960|20.405405%|27.195946%|27.364865%|+0.168919|
|locked|2111|37.612506%|75.177641%|74.988157%|-0.189484|
|degraded|1061|23.751178%|41.658812%|42.601320%|+0.942507|
|unlocked|2228|19.928187%|24.775583%|24.640934%|-0.134650|

相对C0，原paired-As收益主要出现在LOS/PARTIAL/locked；C2相对C1没有新增LOS收益，PARTIAL和locked略降，degraded略升，BLOCKED/unlocked仍低。mid urban增量+1pp与low仅净增3/5400必须分开报告。warm期已有浮点轨迹分歧，单seed小差不能唯一归因于课程；没有跨seed重复、物理单因素因果或联合突破声明。

|TX index|C2-C0 clean修复|clean新错|clean Δ pp|C2-C1 clean Δ pp|C2-C1 low Δ pp|
|---|---:|---:|---:|---:|---:|
|0|11|6|+0.555556|-0.333333|-0.555556|
|1|13|76|-7.000000|-0.222222|-0.777778|
|2|7|289|-31.333333|-1.777778|-0.333333|
|3|1|0|+0.111111|+0.000000|+0.222222|
|4|6|9|-0.333333|+0.000000|+1.555556|
|5|133|3|+14.444444|+0.666667|+0.222222|

全部TX/day/state/lock/quality、AGC／边界／残余核／频偏相关统计，固定C0 clean正确集合sat保持率、cosine30间隔、融合gain轨迹及默认头梯度均保留。状态差异是机制关联，不是独立物理原因；没有额外拟合诊断头或全特征／门控强制对齐。

同一物理包的clean错误流向（不参与课程和选模）：

|对照|clean分组|包数|low修复|low新错|low净正确变化|
|---|---|---:|---:|---:|---:|
|C2-C0|clean_both_correct|3998|1223|276|947|
|C2-C0|clean_lost|383|73|48|25|
|C2-C0|clean_repaired|171|57|6|51|
|C2-C0|clean_both_incorrect|848|124|53|71|
|C2-C1|clean_both_correct|4156|75|77|-2|
|C2-C1|clean_lost|28|1|0|1|
|C2-C1|clean_repaired|13|0|0|0|
|C2-C1|clean_both_incorrect|1203|20|16|4|

分组按各对照与C2的clean是否正确固定，再统计同一批包的practical预测；各组净正确变化之和等于整体变化。不能把clean新错组里局部urban修复直接称为双收益。统计程序16种布尔正确性组合穷举通过，0IQ／CP／NN。

|arm|新增或复用|模型轮|更新|网络包|训练及sourceV秒|六practical秒|物理proposal数|物理生成秒|
|---|---|---:|---:|---:|---:|---:|---:|---:|
|C0|复用结果|20|4440|3579018|1110.199878|64.152623|0|0.000000|
|C1|复用结果|20|4440|3579018|2082.930084|64.007136|568320|0.000000|
|C2|新增|20|4440|3579018|2816.237325|69.217922|1046334|1672.718023|

新增20模型轮／4440更新／3579018网络包，矩阵墙钟2892.487447s，复用C0/C1原40模型轮分别列。C1旧物理生成秒未单独记录，表中0不表示免费。课程physics_seconds含全生成／CPU转换／metadata与trace，无额外神经网络；总预算用尽不等于收敛。离线分析0新IQ／checkpoint／网络，CPU分析未统一计时。

本地真实F合成1024网络包／2更新／14协议负例，206初态、eta0逐值等于C1、finite fallback、原头SIG3Sinc梯度／4L／首对SupCon／ownEMA／全局RNG通过；528模拟选中观察／656完整模拟提案，0真实IQ／CP。隔离配置与原源RX6参考metadata污染拒绝通过；唯一限定P0/P1只读审查PASS。独立结果读取器24元数据负例／512合成draw／20轮history PASS，0IQ／CP／NN；N607普通账户单archive比较／remotecompile及真实ownCP no-query smoke、PID/CWD/argv/GPU/log增长VERIFIED。

实际warm轨迹限制：前5轮eta0，原nominal state/lock覆盖、硬件session、F优化日程及SupCon记录完全一致；训练权重轨迹不逐位相同。首步主／次CE及总loss相同，首步gradient差1.544621e-8，gradient在step4、主CE在step5已超过1e-6差异。两次均cudnn_deterministic=false、benchmark=false、TF32=false；证据与浮点非确定训练相容，但未单独证明唯一原因。E5头weight移动norm .726220对.721270、融合gain最大差.008736、Sinc坐标移动norm差最大1.457787；不是模型权重逐点差或准确率。不得声称eta0实际训练完全相同、把小差当因果或视为独立跨seed重现。本轮不热改确定性设置或重跑既有C1；全部记录见warm_comparison.json。

实际评价__file__：

- channel: `/home/szu2070436088/2510044040/CV-SincNet/releases/cvs_SIGTIME_evidence_responsibility_train20_20261015_r02/experiments/adv3b02_xuc/code/leo_practical/channel.py`
- residual: `/home/szu2070436088/2510044040/CV-SincNet/releases/cvs_SIGTIME_evidence_responsibility_train20_20261015_r02/experiments/adv3b02_xuc/code/leo_practical/residual.py`

六实际asdict固定：25MHz／residual／post_sync／EQoff即G1／reference30／quality_adaptation=true、quality_reference20／phase_tracking=true／IQcomp=true／reflect。每个评价物理ID与physics都与C0/C1相同；原正式IQ、信道、SNR及困难样本不变。

预登记判据与实际决定：{"status": "FAIL", "checks": {"clean": false, "seen": true, "urban": true, "low": true, "urban_F1": true, "activation": true, "complete_health": true}, "clean_drop_vs_C0_pp": 3.9259259259259216, "seen_drop_vs_C0_pp": 0.24537037037036802, "urban_gain_vs_C1_pp": 0.462962962962965, "low_gain_vs_C1_pp": 0.05555555555555314, "urban_F1_gain_vs_C1_pp": 0.44514251428244833, "source_only": true, "target_access": false, "model_seed_count": 1, "source_fold_count": 1, "automatic_extension": false, "goal_completed": false}。关闭该固定课程，不调eta／范围／warm／提案数／比例／RX／seed／阈值救援。本轮一个modelseed／一个已暴露失败源RX，不能声称跨RX稳定、正式部署或跨seed重现。真正Ag/As四格和交互、low>60%、clean净收益与suburban接近clean仍未证明，目标active。

精确row／command／环境／CWD／paths／GPU／owner／stop见experiment.json及preregistration.md。实际图表已查看4幅：

- [complete_training_curves.png](E:/type10-7/local_artifacts/cvs_FcrossRX_recoverable20_20261016/final/complete_training_curves.png)
- [source_scene_results.png](E:/type10-7/local_artifacts/cvs_FcrossRX_recoverable20_20261016/final/source_scene_results.png)
- [low_urban_strata.png](E:/type10-7/local_artifacts/cvs_FcrossRX_recoverable20_20261016/final/low_urban_strata.png)
- [course_progress.png](E:/type10-7/local_artifacts/cvs_FcrossRX_recoverable20_20261016/final/course_progress.png)
- [scene_results.csv](E:/type10-7/local_artifacts/cvs_FcrossRX_recoverable20_20261016/final/scene_results.csv)
- [source_strata.csv](E:/type10-7/local_artifacts/cvs_FcrossRX_recoverable20_20261016/final/source_strata.csv)
- [paired_strata.csv](E:/type10-7/local_artifacts/cvs_FcrossRX_recoverable20_20261016/final/paired_strata.csv)
- [training_curves.csv](E:/type10-7/local_artifacts/cvs_FcrossRX_recoverable20_20261016/final/training_curves.csv)
- [fusion_gain_curves.csv](E:/type10-7/local_artifacts/cvs_FcrossRX_recoverable20_20261016/final/fusion_gain_curves.csv)
- [measured_costs.csv](E:/type10-7/local_artifacts/cvs_FcrossRX_recoverable20_20261016/final/measured_costs.csv)
- [error_transport.csv](E:/type10-7/local_artifacts/cvs_FcrossRX_recoverable20_20261016/final/error_transport.csv)
