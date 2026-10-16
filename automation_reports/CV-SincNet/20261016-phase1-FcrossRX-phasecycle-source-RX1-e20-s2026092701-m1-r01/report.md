# 精确 C4 相位轮转：源域完整结果

固定源规则 **FAIL**。一个新scratch E20、一个modelseed、held sourceRX1；0target。

|scene|C0 accuracy|C4 accuracy|Δ pp|C0 macroF1|C4 macroF1|
|---|---:|---:|---:|---:|---:|
|clean|83.4444%|85.6296%|+2.1852|0.832275|0.855910|
|practical_high|38.4259%|34.2222%|-4.2037|0.301789|0.282649|
|practical_mid|36.1667%|32.6111%|-3.5556|0.290926|0.270895|
|practical_low_suburban|34.0370%|30.6296%|-3.4074|0.280013|0.255751|
|practical_high_urban|37.0000%|32.3333%|-4.6667|0.290694|0.267110|
|practical_mid_urban|30.1667%|27.2778%|-2.8889|0.263720|0.236611|
|practical_low_urban|25.6852%|23.6481%|-2.0370|0.231182|0.210562|

原clean≥+1pp/F1不降/seenV损失≤.5pp及完整健康规则：{'status': 'FAIL', 'checks': {'clean': True, 'F1': True, 'seen': False, 'complete_health': True}, 'clean_gain_pp': 2.18518518518519, 'macroF1_gain_pp': 2.3635491743851267, 'seen_drop_pp': 0.6712962962962865, 'source_only': True, 'target_access': False, 'model_seeds': 1, 'automatic_extension': False, 'goal_completed': False}。不调角度、概率、比例、RX/seed或预算救援。

完整20新模型轮／4440更新／3579018网络包；C0既有20轮成本单列，父权重成本0。完整5994条primary L/U与C0相同；568320次C4轮转、5040包各三元素、15120物理phase cell独立重放核实，逐点功率／逆恢复执行值为0误差。矩阵墙钟1198.810524s；构造墙钟含旋转、核对及trace，见measured_costs.csv。

原F＋跨RX SupCon/默认头联合训练/原U/ownEMA不改，无新部署模块或分类头。0.5RD同ordinary＋0.5RD(C4同IQ)，四L前向、一次ordinary；不做跨视图一致性。新增输入变化可逆不等于训练有效，也不等于替换RX硬件或拟合相位bank。

全部七源视图默认头预测独立复算、同物理标签和原practical流；全部20轮/4440步/nativeCSV/stdout/F clip/SupCon解析。所有TX/day/state/quality、修复/新错、固定C0 clean正确集合保持率、间隔与曲线保留；不修改正式IQ/信道/SNR。没有As训练、正式target或跨modelseed复现；真正四格混合、lowurban>60%及suburban接近clean仍未证明。预算耗尽不等于收敛。

实际状态：**VERIFIED／ANALYZED，FAIL_SOURCE_RX1_ONLY**。唯一worker和dispatcher自然结束，无技术failure、重试或热改。训练代码b03d183c5672bbf7d7e1e84c30841203b105e9b5；离线读取在独立工具中复算，不修改运行release。

数据角色与来源：fit RX3/4/6/8的原L5040／隐藏U45360／seenV21600，held RX1的L/U全部不训练，heldV5400仅FINAL20。一个新scratch E20＝13L＋7U、4440更新，父权重／EMA／optimizer／统计继承成本0；不加载E50/E200或C0 checkpoint。C0只复用已核实同合同own scratch FINAL20的结果／原日志／5994物理批次trace。完整source L6300和内部5580/720与本折分别说明，U TX真值始终隐藏。

保持F三路SIG_TIME/CENTER_25K/CENTER_UNITS、25MHz／IQ256／equalized1／native RMS／NoExplicitCFO、原joint160与default cosine30联合训练，未冻结骨干重训头。time_down/t2.pool的原L-only跨RX同TX SupCon温度.2、最大.1、warm5/ramp10，只读第一对两次已有前向；没有新部署参数、头或跨视图对齐损失。原U/ownEMA/旧门控/cos、strong_noise_std=.015不改。原默认头、SIG三路、Sinc均正常获得梯度和更新，全部4440步F物理clip／lr映射核实。

新候选仅0.5RD(共同旧ordinary)+0.5RD(C4(同一actual ordinary IQ))，每RD内同IQ两次dropout，LS=.01／RDrop alpha1不变。四L前向、一次ordinary生成、独立原L物理曝光数及原U完全匹配C0；C4只做精确符号与I/Q交换，不增加噪声、CFO、相位噪声、多径、时延、硬件畸变、边界或RMS。它是接收观察的数字整包常数相位，不是更换RX硬件、未知TX反演或失败频率相关相位bank。

C4私有种子20261016811只在物理包首次出现时取随机offset，之后按访问次数循环90°／180°／270°；选角度不读TX、quality、loss或预测，不推进全局RNG。实际568320次轮转、5040包的15120个physical×phase cell及TX×fitRX×day×k覆盖完整，逐点功率／精确逆恢复误差执行值均0，每包三旋转曝光次数差≤1。独立离线工具按原L trace和命名私有流完整重放4440步，全部5994个原L/U批次逐条匹配C0。可逆输入不保证训练有效，记录实际识别分数作判定。

实际旧ordinary normal路径保留：anti-shortcut→general channel→可选PA，class-signature=false、DAC=0。原batch-wide相位触发的执行概率已从C0全部20轮mechanism记录核实：前三轮.045，E13为.11277015027073835，E14至E20保持该值；其范围仍[0,2π)。本候选固定配对C4循环，不扫描原概率、强度或角度。原ordinary其他损伤仅共同旧条件，未预称每个算子身份安全。

source clean＋六practical每视图5400包，共75600个保存默认头预测独立复算。实际practical来自原release的experiments/adv3b02_xuc/code/leo_practical；保持fs25MHz/residual/post_sync/EQoff即G1/reference30/reflect、原接收质量／相位跟踪／RX IQ补偿／AGC和传播依赖，与C0每个物理ID和physics元数据一致。训练As调用0；未改正式received IQ、SNR、信道或困难样本，0target／正式truth-last。新增noise SNR不等于总SNR。

读取完整20轮／4440步nativeCSV／stdout／所有CE/KL/SupCon及梯度指标，非有限、跳步与U真值暴露检查均通过；默认头间隔统一除cosine scale30。固定保持集合为C0 clean正确集合，不随C4重选。全部TX/day/state/lock/quality的修复／新错、负结果与原融合参数轨迹保留，参数变化或辅助loss下降不充当身份收益或单因素因果证据。

费用实测：

|arm|新增/复用|模型轮|更新|网络包|训练/冻结源V 秒|六practical 秒|C4构造 秒|
|---|---|---:|---:|---:|---:|---:|---:|
|C0|复用结果|20|4440|3579018|1123.572077|64.008927|0.000000|
|C4|新增|20|4440|3579018|1127.588725|63.383062|7.043161|

新增20模型轮／4440更新／3579018网络包，矩阵墙钟1198.810524 s；C0既有20轮／4440步／3579018网络包单列，父权重0。每行L2273280／studentU396368／student sourceeval475202（含2包ownCP smoke）／EMA396368／heldclean5400／六practical32400。新增C4构造计符号交换、功率/逆核对、metadata/trace，不含骨干前向；无新ordinary draw或新L标签。预算耗尽不等于收敛。离线读取/分层/绘图0网络/IQ/CP，CPU分析总墙钟未统一计时。

本地合成正测1024网络包／2更新／15协议负例、206初态／primary IQ逐值匹配／原头SIG3Sinc梯度／ownEMA／四前向／C4代数与RNG均PASS；先前两个测试脚本错序尝试另1152合成网络包／1更新，0真实IQ/CP，累计2176包／3更新。隔离发布实际E20参数及真实C0元数据核对／污染拒绝PASS。唯一P0/P1只读审查PASS无直接问题，一次N607普通账户preflight、单archive本远比较／remotecompile、worker首步真实ownCP无query smoke及PID/CWD/argv/GPU/log增长VERIFIED。离线读取器13元数据负例与384合成phase metadata包，0网络/IQ/CP；合成读数非性能。

固定主判据实际值：held clean Δ 2.18518518518519 pp、宏F1 Δ 2.3635491743851267 pp、seen V损失 0.6712962962962865 pp。本固定C4候选关闭，不扫描角度/概率/比例、不换源RX/seed、不延长E20或改主参照救援。 用户指定F＋跨RX基线保持；一个model seed／一个held源折不能声明跨seed复现。旧paired-As五源clean保护失败和realpair失败不改写。真正独立Ag/As及四格交互、正式七视图、low>60%和suburban接近自身clean仍未完成，目标保持active。

本候选的正负收益进一步分开：clean83.444444→85.629630%，但六practical均下降；urban三场景均值30.950617→27.753086%（-3.197531 pp），low25.685185→23.648148%（-2.037037 pp）。固定C0 clean正确4506包在low的保持率25.943187→26.165113%，仅多保留10包；C0 clean错误894包中的low正确数218→98，整体low净少110包。这不支持把clean涨分称为跨信道突破，也不把某个子集保持率上升代替全体识别。

clean修复302／新增错误184；其中TX index2从573/900到789/900（+24 pp），TX index5从890/900到774/900（-12.888889 pp）。day1下降0.555556 pp、day2提高5.777778 pp、day3提高1.333333 pp；完整TX/day负结果保留。TX是原0..5网络标签索引，不另猜物理编号。cosine默认头的clean平均分类间隔/30反而0.131897→0.094635；同尺度下准确率上涨不意味着所有判别关系更强。表明当前配方存在明显错误重分配，不能据此宣称成功或单一因果机制。

精确command/environment/CWD/paths/GPU/owner/stop及预期artifact见experiment.json和preregistration.md。离线复算（不重测或开IQ/CP）：`F:/App/miniconda3/python.exe -X utf8 -m experiments.cvs_f_phasecycle20.harvest --output E:/type10-7/local_artifacts/cvs_FcrossRX_phasecycle20_20261016/final --state E:/type10-7/local_artifacts/cvs_FcrossRX_phasecycle20_20261016/terminal_readback.json --reference-root E:/type10-7/local_artifacts/cvs_FcrossRX_practical20_20261016/final --cached`。

实际表图（三图已逐张检查）：

- [complete_training_curves.png](E:/type10-7/local_artifacts/cvs_FcrossRX_phasecycle20_20261016/final/complete_training_curves.png)
- [source_scene_results.png](E:/type10-7/local_artifacts/cvs_FcrossRX_phasecycle20_20261016/final/source_scene_results.png)
- [low_urban_strata.png](E:/type10-7/local_artifacts/cvs_FcrossRX_phasecycle20_20261016/final/low_urban_strata.png)
- [scene_results.csv](E:/type10-7/local_artifacts/cvs_FcrossRX_phasecycle20_20261016/final/scene_results.csv)
- [source_strata.csv](E:/type10-7/local_artifacts/cvs_FcrossRX_phasecycle20_20261016/final/source_strata.csv)
- [paired_strata.csv](E:/type10-7/local_artifacts/cvs_FcrossRX_phasecycle20_20261016/final/paired_strata.csv)
- [training_curves.csv](E:/type10-7/local_artifacts/cvs_FcrossRX_phasecycle20_20261016/final/training_curves.csv)
- [fusion_gain_curves.csv](E:/type10-7/local_artifacts/cvs_FcrossRX_phasecycle20_20261016/final/fusion_gain_curves.csv)
- [measured_costs.csv](E:/type10-7/local_artifacts/cvs_FcrossRX_phasecycle20_20261016/final/measured_costs.csv)

运行实际导入的__file__（由最终source评价器保存，非默认路径推测）：

- channel: `/home/szu2070436088/2510044040/CV-SincNet/releases/cvs_SIGTIME_evidence_responsibility_train20_20261015_r02/experiments/adv3b02_xuc/code/leo_practical/channel.py`
- residual: `/home/szu2070436088/2510044040/CV-SincNet/releases/cvs_SIGTIME_evidence_responsibility_train20_20261015_r02/experiments/adv3b02_xuc/code/leo_practical/residual.py`

全部六配置的实际asdict核实fs25MHz／residual/post_sync／EQoff G1／reference30／quality_adaptation=true、quality_reference20、phase_tracking=true、IQcomp=true、reflect，与原C0同流核对通过。完整值见practical_runtime_audit.json；0新增IQ/CP/网络。
