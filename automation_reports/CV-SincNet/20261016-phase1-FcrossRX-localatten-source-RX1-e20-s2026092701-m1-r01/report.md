# 有界局部观察衰减：完整源域结果

固定源规则 **FAIL**。一个scratch E20,一个modelseed,held sourceRX1;0As训练/0target。

|scene|C0 accuracy|LA16 accuracy|Δ pp|C0 macroF1|LA16 macroF1|
|---|---:|---:|---:|---:|---:|
|clean|83.444444%|86.000000%|+2.555556|83.227457%|85.863159%|
|practical_high|38.425926%|36.092593%|-2.333333|30.178906%|29.278985%|
|practical_mid|36.166667%|33.370370%|-2.796296|29.092638%|27.838089%|
|practical_low_suburban|34.037037%|32.500000%|-1.537037|28.001335%|28.298846%|
|practical_high_urban|37.000000%|34.203704%|-2.796296|29.069414%|27.844953%|
|practical_mid_urban|30.166667%|28.574074%|-1.592593|26.372042%|25.614097%|
|practical_low_urban|25.685185%|24.314815%|-1.370370|23.118247%|22.652121%|

原clean>=+1pp/F1不降/seenV及逐项practical下降<=.5pp和完整健康条件：{'status': 'FAIL', 'checks': {'clean': True, 'F1': True, 'seen': True, 'practical': False, 'complete_health': True}, 'clean_gain_pp': 2.5555555555555554, 'macroF1_gain_pp': 2.635701558899206, 'seen_drop_pp': -0.04166666666667318, 'practical_gain_pp': {'practical_high': -2.3333333333333317, 'practical_mid': -2.7962962962962967, 'practical_low_suburban': -1.5370370370370368, 'practical_high_urban': -2.7962962962962967, 'practical_mid_urban': -1.5925925925925954, 'practical_low_urban': -1.370370370370369}, 'source_only': True, 'target_access': False, 'model_seeds': 1, 'automatic_extension': False, 'goal_completed': False}。失败不调width/depth/L2bound/probability/ratio/RX/seed/预算。

新增20模型轮/4440更新/3579018网络包；C0既有E20成本单列，0父权重继承。全5994主L/U序列与C0一致；568320窗口/5040原L/241起点独立复放，保存的边界/深度/L2/功率变化及全部聚合核对。未重新访问IQ、模型或CP，不把保存统计复算写成原始IQ独立重算。矩阵墙钟1225.787516s；局部构造包含bound计算、CPU同步与trace、不含NN。

F+CROSS_RX/defaultcos/NoExplicitCFO/原U/ownEMA不变。0.5RD原ordinary+0.5RD局部衰减同ordinary，四L前向、一次ordinary、无新一致性/头/参数。所有合法有标签包CE，0置信度过滤和物理样本丢弃。

完整七源视图预测独立复算，同一物理标签和原practical流；全部20轮/4440步/CSV/stdout/Fclip/SupCon解析。保存全部TX/day/state/lock/quality、间隔、repair-break、固定C0 clean正确集合保持率和成本。单seed源结果不等于正式target、多seed或联合突破。lowurban>60%、suburban接近clean、真正Ag/As四格交互仍未证明。

实际状态：**VERIFIED／ANALYZED，FAIL_SOURCE_RX1_ONE_SEED**。唯一dispatcher与worker自然结束，0训练技术failure／重启／热改／选择性重跑；训练代码3bba9d406fe1dee68b7f76e403f114c11954b862，后续报告／离线读取器提交不冒充训练代码。

本轮证实的是该源RX1/seed上的clean局部收益：83.444444%→86.000000%，默认头+2.555556pp、macroF1+2.635702pp；修复159包、新错21包。它同时否定该固定独立Ag配方天然保持practical的假设：六场景全降，urban均值Δ-1.919753pp、mid-1.592593pp、low-1.370370pp。不能把clean上涨写成星地上涨，也不能由此推断所有Ag或未运行的联合训练都无效。原规则必须全部同时通过，practical项失败，所以固定候选关闭，不改width/depth/L2bound/probability/ratio/RX/seed/epoch/阈值救援，不自动扩确认、target或Ag+As。

|arm|clean|high距自身clean|mid距自身clean|low suburban距自身clean|
|---|---:|---:|---:|---:|
|C0|83.444444%|45.018519pp|47.277778pp|49.407407pp|
|LA16|86.000000%|49.907407pp|52.629630pp|53.500000pp|

LA16没有把自身clean降低来缩差；三个suburban绝对准确率下降，距clean差距反而扩大，都远未达到2pp探索目标。lowurban=24.314815%，未达60%。

|low urban分层|N|C0准确率|LA16准确率|差pp|
|---|---:|---:|---:|---:|
|LOS|1674|37.933094%|34.109916%|-3.823178|
|PARTIAL|710|30.281690%|27.746479%|-2.535211|
|BLOCKED|3016|17.805040%|18.070292%|+0.265252|
|locked|2081|36.617011%|33.109082%|-3.507929|
|degraded|965|23.937824%|21.968912%|-1.968912|
|unlocked|2354|16.737468%|17.502124%|+0.764656|

BLOCKED仅净增8/3016，unlocked净增18/2354；LOS净损64、PARTIAL净损18，locked净损73、degraded净损19。小分层正差不足以抵消可靠观察的损失，不声称urban退化总体缓解或物理因果。

|TX index|clean修复|clean新错|clean Δpp|low urban Δpp|
|---|---:|---:|---:|---:|
|0|50|2|+5.333333|+2.888889|
|1|3|4|-0.111111|+0.555556|
|2|100|4|+10.666667|-14.000000|
|3|3|9|-0.666667|+2.111111|
|4|0|0|+0.000000|-2.888889|
|5|3|2|+0.111111|+3.111111|

TX2的lowurban从352/900降至226/900（−14pp）是主要负项之一；其他TX的局部正差保留，不能挑出正类作晋级。

固定C0 clean正确集合4506包的low urban保持率：25.943187%→25.011096%。分类间隔均用原cosine30尺度：clean均值0.131897402→0.140269322；low urban均值-0.045724936→-0.044931254。六practical的平均间隔都略升，准确率却都下降，直接说明平均间隔／梯度健康不能替代识别收益；没有通过改logit尺度造间隔。

|clean正确性分组|N|原low正确|新low正确|low修复|low新错|净正确变化|
|---|---:|---:|---:|---:|---:|---:|
|clean_both_correct|4485|1166|1124|178|220|-42|
|clean_lost|21|3|3|1|1|0|
|clean_repaired|159|42|28|4|18|-14|
|clean_both_incorrect|735|176|158|34|52|-18|

新修复clean的159包，其lowurban正确数反而42→28；原两模型clean都正确4485包的low正确1166→1124。各组与全局净−74严格守恒，16种布尔组合及实际七场景/28分组核验通过，0新IQ/CP/NN。此为错误流向描述，不反推唯一骨干/门控或噪声原因。

全568320窗口、4440step、5040合法L、241起点和1152个TX×RX×day×16位置区间独立回放/核对，5994条主L/U序列逐条等于C0。最大/平均相对L2=0.099999820192/0.075074089160，平均/最大功率移除=2.542508%/17.550989%；深度最大.5，窗外240点差0，实际零功率包0，无物理样本过滤。位置流独立复放；波形变化统计来自保存trace归约/界限检查，未重新读取IQ，不能冒称原IQ独立重算。

实际起点均匀，样本覆盖在包两端较低；重建图明确保留边界差异。累积均值/最大值曲线不是逐轮独立分布，E6/E18最大功率移除跳变不代表策略或强度ramp。算子同时缩放录音中信号与既有噪声、改变包络及频谱，不是选择性去噪或SNR提升，也不等于纯RX硬件变换。

完整20轮/4440步/nativeCSV/stdout/Fclip/SupCon全部解析，0critical nonfinite、0非有限跳步、0stdout训练错误；3330非零SupCon调用/4440总调用、原warm5/ramp10及13L+7U保持。sourceU真值可用计数最大0。默认头、三路SIG与Sinc正常联合更新，原融合gain曲线保留；未额外拟合分类头或强制全部特征/门控相等。

|arm|新训/结果复用|模型轮|更新|L网络包|总网络包|训练及源V秒|六practical秒|局部构造秒|
|---|---|---:|---:|---:|---:|---:|---:|---:|
|C0|复用|20|4440|2273280|3579018|1123.572077|64.008927|0.000000|
|LA16|新增|20|4440|2273280|3579018|1152.867336|65.782955|11.209853|

增量成本20模型轮/4440更新/3579018 NN，矩阵墙钟1225.787516s（20.430min）；C0历史E20成本另列，父权重训练/继承0，不把结果复用写成免费训练。每row网络包分解L2273280/studentU396368/student sourceeval475202(含2zero-smoke)/EMA396368/heldclean5400/practical32400。局部构造11.209853s含math/GPU同步/trace，不含NN，不把所有墙钟差都归因为算子。离线SCP+完整分析约24.4s、绘图及后处理另计，CPU工作不计成模型前向；预算用尽不等于收敛。

局部测试18协议负例/735直接变换包/384采样包/256管线变换包/1024合成NN和2更新PASS，206初态与匹配C0逐tensor一致、所有241起点人工公式/零包保留/窗外不变/范数界/primary输入精确对照/原2forward SupCon/defaulthead-SIG3-Sinc梯度PASS。独立reader18元数据负例、128模拟trace、37800原C0源决策及2边界oracle通过；本次共75600实际保存源决策独立复算。已有NumPy ABI警告没有阻止纯Torch检查，未安装依赖。

一次限定独立P0/P1 PASS；普通账户/单归档比较/remotecompile/真实own scratchCP noquery smoke/独立PID-CWD-argv-GPU及日志增长VERIFIED。观察脚本曾因重复字段、以及进程自然退出时/proc消失报错，均只修复只读观察层；没有据观察错误重启训练。公共预测helper未校验correct计数字段的缺口只在本候选新reader补齐，未改历史代码或结果。

实际leo_practical __file__：

- channel：/home/szu2070436088/2510044040/CV-SincNet/releases/cvs_SIGTIME_evidence_responsibility_train20_20261015_r02/experiments/adv3b02_xuc/code/leo_practical/channel.py
- residual：/home/szu2070436088/2510044040/CV-SincNet/releases/cvs_SIGTIME_evidence_responsibility_train20_20261015_r02/experiments/adv3b02_xuc/code/leo_practical/residual.py

六视图真实25MHz/residual/post_sync/EQoff即G1/reference30/qualityadapt=true、qualityreference20/phase tracking=true/IQcomp=true/reflect，逐物理ID和全部physics与C0相同。正式测试IQ、信道、SNR和困难样本均未改变。

只使用当前fit RX3/4/6/8 L5040/U45360/seenV21600，heldRX1 L/U不训练、heldV5400 FINAL-only；不是全源L6300或旧内部5580/720。此held源RX1是已暴露研发源折，本轮不是新的盲测。一个seed2026092701，未做其他未见RX或独立modelseed确认；没有正式目标truth-last、B00/B10/B01/B11交互或共同low>60%证据。不能宣称跨seed重现、已改善实测urban或任务完成。

精确实现/配置/command/environment/CWD/paths/GPU/stop见experiment.json和preregistration.md；原模型/defaulthead/F/SupCon/U来源保持，训练commit与离线reader版本分开。实际产物：

- [complete_training_curves.png](E:/type10-7/local_artifacts/cvs_FcrossRX_localatten20_20261016/final/complete_training_curves.png)
- [source_scene_results.png](E:/type10-7/local_artifacts/cvs_FcrossRX_localatten20_20261016/final/source_scene_results.png)
- [low_urban_strata.png](E:/type10-7/local_artifacts/cvs_FcrossRX_localatten20_20261016/final/low_urban_strata.png)
- [local_observation_coverage_cumulative.png](E:/type10-7/local_artifacts/cvs_FcrossRX_localatten20_20261016/final/local_observation_coverage_cumulative.png)
- [local_observation_coverage_cumulative.pdf](E:/type10-7/local_artifacts/cvs_FcrossRX_localatten20_20261016/final/local_observation_coverage_cumulative.pdf)
- [scene_results.csv](E:/type10-7/local_artifacts/cvs_FcrossRX_localatten20_20261016/final/scene_results.csv)
- [source_strata.csv](E:/type10-7/local_artifacts/cvs_FcrossRX_localatten20_20261016/final/source_strata.csv)
- [paired_strata.csv](E:/type10-7/local_artifacts/cvs_FcrossRX_localatten20_20261016/final/paired_strata.csv)
- [training_curves.csv](E:/type10-7/local_artifacts/cvs_FcrossRX_localatten20_20261016/final/training_curves.csv)
- [fusion_gain_curves.csv](E:/type10-7/local_artifacts/cvs_FcrossRX_localatten20_20261016/final/fusion_gain_curves.csv)
- [local_mask_exposure.csv](E:/type10-7/local_artifacts/cvs_FcrossRX_localatten20_20261016/final/local_mask_exposure.csv)
- [measured_costs.csv](E:/type10-7/local_artifacts/cvs_FcrossRX_localatten20_20261016/final/measured_costs.csv)
- [error_transport.csv](E:/type10-7/local_artifacts/cvs_FcrossRX_localatten20_20261016/final/error_transport.csv)
