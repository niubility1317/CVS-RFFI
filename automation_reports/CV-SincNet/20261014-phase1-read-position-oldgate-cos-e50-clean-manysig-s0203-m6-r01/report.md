# SIG_TIME 取点：条件式剩余两种子

固定seed2026092702/03 × OLD_GATE_COS50、SIG_TIME post、SIG_TIME_PRE pre，共6row：2历史baseline只读复用、4独立scratch50/11100。仅在原首seed 20261014-phase1-read-position-oldgate-cos-e50-clean-manysig-s01-m3-r01 完成3row默认头闭环且固定源条件PASS后发布；当前未启动。首seed不重训、不按目标筛除或改配方。

机制原样沿用报告路线三：单6D SIG_TIME、4×8窗、112→160/LN/gain .05/.2、pool18976/fθ195041/W960。只改descriptor读t2或t2.pool，主干Dropout.1不变；默认cosine6×160scale30与骨干联合训练，所有额外分类器拟合关闭。旧batch_neighbor＋自身EMA.999＋cos33label+17pseudo/AdamW2e−4→1e−6/wd1e−4/lambdaU.16/entropy.01/smoothing.01保持。无COV/AREA/层级/Sinc/SiLU/PAIR/U难度权重堆叠。

L6300/U56700/V27000、trainingpool标注10%、split392005和原RX/day/TX/physicalID/NoExplicitCFO及内部补偿不变；U TX真值隐藏，V健康不训练/选模，无新星地训练。4×50=200新增model-epochs/44400更新，历史100单列复用；合并首seed为6新模型300轮/66600更新、3历史150复用。全部从零，无父权重/teachercheckpoint继承。

唯一发布条件是原首seed执行 b4d55dba50c8601017604a2a0eb0d037f8d835e8 的既有source verdict：两finalLCE≤.5，实际新增参数更新、feature-only导出/W独立、真实初始全参数及L720 eval function相同、PRE−POST最终L720L2≥1e−6，source_only且无目标分数读取/头拟合。必须原首seed3row完整504000默认decisions/truthlast，但资格只读completion元数据和source witnesses，不读target analysis/metrics。FAIL保留产物不派发；4个新模型不会由目标分数选择性启动/改参/重跑。

实现 ea3228433751d21751cf0f275326dceb71b3c7d5。4真实native模型上下文检查PASS，first3配置与已发布b4d55dba50c8601017604a2a0eb0d037f8d835e8完全相同，seedpair初始native/body/W/pool参数同、eval函数同、实际读t2/t2.pool、两paired初始eval检查和195041/buildcallback正确。source gate负例FAIL/targetread/headfit/query/缺行/wrongexecution/非有限LCE/空初态witness均拒绝且无target metrics访问；3seed汇总和remote6row/4new/1008000/108统计compile PASS。候选核心复用首seed一次直接P0/P1 PASS，新增执行只做相关上下文验证，不重复独立审查同一候选。

入口python -u -m experiments.cvs_read_position_seeds.dispatch；普通N607既有CVS-RFFI fullFP32/2threads；CWD /home/szu2070436088/2510044040/CV-SincNet/releases/cvs_read_position_seeds_20261014_r01；不可变root /home/szu2070436088/2510044040/CV-SincNet/runs/20261014-phase1-read-position-oldgate-cos-e50-clean-manysig-s0203-m6-r01。共享capacity锁计全部owner每GPU≤3/free≥12000MiB，保护健康旧任务。发布一个archive/一次localremoteSHA比较/remotecompile，首步source资格及旧seed02真实CP无查询smoke，PASS继续。复用VALIDATED_ONCE/现有data handles，无额外签名/hash/token/审批或数据重验。

Stop：本run角色/来源/预算/非有限/默认头/featureexport/ID/prediction/truthlast系统技术失败停止新增派发，产物及健康进程保留，无自动重试；不以目标停止/调参/换seed。每row final50即自动clean168000默认头预测、固定prediction后独立truthlast；本run6row1008000决策/108指标，合并首seed9row1512000/162。报告完整3预定义seed均值/样本SD/配对正差数、MacroF1、最差RX/day、错误修正/引入、全部源50epoch曲线/U接受/实测时间及实际零fit/更新/初态一致证据。只有SIG_TIME_PRE完整3seed均值≥84才达目标。

预期release资格/preflight/submit；run launch/queue/newrows initial-final-frozen/resolved/args/roles/50epoch/11100steps/sourcewitness/defaultprediction/scores/completion；source_features/analysis/combined_three_seed_analysis/completion。数据/weights/rawlogs/prediction留既定远端，代码/config/报告/必要聚合Git交付并核remoteOID。当前只准备和登记，无完整首seedsource verdict，不宣称启动或性能提升。
