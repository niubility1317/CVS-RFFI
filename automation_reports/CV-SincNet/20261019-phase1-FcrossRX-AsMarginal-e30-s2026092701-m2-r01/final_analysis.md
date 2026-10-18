# P73 最终结果：两随机子网概率均值 CE 未形成星地增量

状态：ANALYZED，missing=[]。训练、两行固定七环境预测、独立truth-last评分和完整分析均完成。固定候选关闭，无E50或参数/seed补救。代码/config提交cb5d8abdd7159ca5a3dad7bbf62f6b37a4c9d6fd；起点99e0030044cebc88568446a7418eb24302640ad1，运行登记提交7e628f954f0975c67fd6553af7830e5d8af3c793。

主比较P−C未缓解任何已证实的urban退化：clean −0.023214、mid urban −0.064286、low urban −0.066071、urban均值 −0.075397 个百分点；七环境准确率、宏F1、最差RX均下降。low修复491次、新错602次，净少111次正确；mid修复396次、新错504次，净少108次正确。

相对同预算、同seed、同数据契约NoAs30，P的clean +0.044643、mid +27.613095、low +21.072619、urban均值 +28.314087 个百分点。这个单seed正式结果满足clean下降≤0.1绝对百分点，但大幅practical收益来自已有完整As，不能归于概率聚合。clean小于0.1pp的上涨不构成稳定双提升。low=51.212500%仍低于60%。已有原As配方保留，本轮新聚合不保留。


两行均从原P43 As20 student/ownEMA独立追加10轮，重置AdamW并使用相同cos10。唯一差异是As64的两次CE均值，或先平均两次softmax概率再按原LS 0.01标签计算CE。原As CE与KL的绝对系数均为0.25；KL、6L、原生Dropout、ordinary/raw、ORD128 SupCon、native U和默认cosine30分类头保持。没有新增物理信息、教师、部署结构或损失栈。

| 环境 | NoAs30 % | 原As30 % | 两CE均值 % | 概率平均CE % | 概率平均−两CE均值 pp |
|---|---:|---:|---:|---:|---:|
| clean | 85.247619 | 85.341071 | 85.315476 | 85.292262 | -0.023214 |
| practical_high | 42.685119 | 81.405357 | 81.422619 | 81.292857 | -0.129762 |
| practical_mid | 40.995238 | 78.661310 | 78.681548 | 78.572024 | -0.109524 |
| practical_low_suburban | 39.472619 | 75.861905 | 75.887500 | 75.776190 | -0.111310 |
| practical_high_urban | 41.439881 | 77.767262 | 77.792262 | 77.696429 | -0.095833 |
| practical_mid_urban | 35.132738 | 62.806548 | 62.810119 | 62.745833 | -0.064286 |
| practical_low_urban | 30.139881 | 51.244048 | 51.278571 | 51.212500 | -0.066071 |

主要比较概率平均CE−两CE均值：clean -0.023214 pp，mid urban -0.064286 pp，low urban -0.066071 pp，urban均值 -0.075397 pp。
阶段比较概率平均CE−NoAs30：clean +0.044643 pp，urban均值 +28.314087 pp；clean下降≤0.1绝对百分点：True。
仅一个模型seed；两个源信道draw不是模型复现。单seed或不足0.1个百分点的正变化不能证明可重复双提升。四行不是2×2因子矩阵。

source仅两项：概率平均CE对NoAs30 fullV clean≥−0.1pp；pooled mid/low概率平均CE对两CE均值≥+0.5pp。判定先于正式读取固定；任一FAIL关闭该配方且不扩E50，全部七个正式环境仍完成，不按正式成绩重新晋级。

全部accuracy、Macro-F1、最差RX、RX/day/TX、修复/新增错误、margin/30和固定NoAs30 clean正确集合保持率见CSV。精确原P43 As20 student源clean正确集合另列；状态/lock/质量分层只在源域可用，不作物理因果归因。新增噪声比不是总SNR。Jensen gap和训练view概率聚合纠错是同logits目标差异，不是身份收益或额外晋级条件。

新增20 model-epochs／4440更新，复用两父及两个旧尾段后四行独有训练总计80 model-epochs。新增native NN=6325924，source=25200，formal=2352004，smoke=4，远端总计=8703132；本地和分析NN均0。新增训练行墙钟之和=1865.454秒。历史池生成／本次缓存读取和loss算子成本另列，预算耗尽不等于收敛。


## 宏F1、最差RX与配对纠错

| 环境 | 概率均值CE 宏F1 % | 最差RX % | 最差RX标识 | 修复／新增错误（P−C） |
|---|---:|---:|---|---:|
| clean | 84.991732 | 78.325000 | 1-1 | 148／187 |
| practical_high | 80.971477 | 71.720833 | 14-7 | 203／421 |
| practical_mid | 78.269841 | 70.137500 | 14-7 | 241／425 |
| practical_low_suburban | 75.487462 | 67.629167 | 14-7 | 288／475 |
| practical_high_urban | 77.355451 | 68.950000 | 14-7 | 237／398 |
| practical_mid_urban | 62.402977 | 56.379167 | 14-7 | 396／504 |
| practical_low_urban | 50.867154 | 46.350000 | 14-7 | 491／602 |

每个正式环境168000包。C1为AS_MEAN_CE，P1为AS_PROBMEAN_CE，均为seed2026092701、totalE30/6660更新，直接使用最终checkpoint的model学生默认头，不是EMA推理或新拟合分类器。urban均值C=63.960317%、P=63.884921%。三种suburban与P clean的差距依次3.999405／6.720238／9.516071个百分点；未达到历史建议的2pp探索目标，不能降低clean人为缩小差距。

相对原As30，P的clean −0.048810、mid −0.060714、low −0.031548、urban均值 −0.054365pp。新C1与旧As30也有小幅差异，故机制归因始终使用新P−新C，不将旧参照差异全归于聚合机制。P的所有环境平均margin÷30均有小幅上升（约0.000599至0.001971），准确率却均下降；不能以平均间隔代替识别收益。头尺度固定30。

## 源证据、机制与关闭条件

预登记只有两个源推进条件：P的fullV27000 clean−NoAs≥−0.1pp；固定1800样本×2draw的mid/low合并P−C≥+0.5pp。实际clean −0.096296pp通过；mid/low合并0pp，未通过。两个draw并非两个模型seed。源low净+4/3600（+0.111111pp），mid净−4/3600（−0.111111pp），没有合并身份增量。source_matrix_decision.json和owner关闭记录均先于owner读取正式分数固定，不因正式结果修改判定。

每行实际2220步、142080 As观测，按原顺序各用一次，原物理key/y/domain/origin校验及全RX/原场景state支持通过；无额外信道模拟、样本删除或质量权重。Jensen gap在C为0.011232535，在P为0.011751589；P的平均总loss修正−0.002937897，严格为−.25×gap。两mask训练view不一致率C=13.426943%、P=13.638091%。这证明目标聚合确实改变了优化；loss数值降低不是新增身份信息，训练view概率均值的纠错也不是部署集成收益。质量<3、3–8、8–15、≥15dB的P gap依次0.016184710、0.018670857、0.019036990、0.007838133，并非只集中于最弱质量组。新增噪声比不是总SNR。

完整源state/lock/质量、固定原P43 As20 student的1765个clean正确样本保持与分层margin见mechanism_readout.md及原CSV。原NoAs20自身父集合及正式父20集合未额外读取，不伪称已完备；正式received IQ没有逐包state/lock标签，不能将源分层关联套作正式物理因果归因。完整loss/梯度曲线无数值异常或缺轮；预算用完不等于收敛。

## 实际代码与配置审计

本轮新增包experiments/cvs_as_marginal73/，关键文件model.py、objective.py、objective_checks.py、ordered_pool.py、observer.py、design.py；另新增analysis/cvs_as_marginal73_analysis_20261019.py及对应checks。网络F＋CROSS_RX、SIG_TIME_X3/joint160、900722参数/206state、默认cosine30头与正常联合训练不变，无部署分支、额外头或额外一致性。新机制是完整As之上的监督聚合，不冒称新IQ泛化增强。

两行同一原P43合法scratchAs20学生/自身EMA，严格加载同原L6300/U56700hidden/V27000、RX1/3/4/6/8、day1/2/3、split392005的父来源，fixedfinal20；新尾段独立RESET AdamW lr2e-4/wd1e-4/cos10/CENTER_UNITS。未用已测试P70/71/72 final作父，未加载E200，未留出源RX/Ldev。原NoAs30来自自身NoAs20而不是同As20，作为匹配NoAs训练路线；四行不是2×2因子。

真实模块__file__已保存并核对：PROJECT/releases/cvs_SIGTIME_evidence_responsibility_train20_20261015_r02/experiments/adv3b02_xuc/code/leo_practical/{channel,residual}.py及SSDG/train_ssdg.py，不是根code副本。25MHz、float32 B×2×256、原逐包RMS center=false后从raw生成As；residual/post_sync、EQfalse G=1、reference_snr_db30、reflect，接收质量适配/相位跟踪/IQ补偿/AGC原值保持。训练fc2.462GHz、正式固定fc2.45GHz。正式IQ、信道、SNR、困难样本均未改；RX硬件session按RX/day/eq稳定，不由TX定义，随机流分离。

原六L与CE系数1.25保持；As CE=.25、同IQ KL=.25，ordinary/raw/U及ORD128 SupCon不变，原生Dropout调用和两次独立mask保持。仅P行在原目标加 .25×[CE((softmax(z0)+softmax(z1))/2,q)−.5(CE(z0,q)+CE(z1,q))]，q仍LS=.01。稳定logsumexp，非平均logits；只复用实际mixed两次已有As64 logits，未增加网络前向、头拟合或教师。全部合法带标签As均参与CE。

## 成本与实际验证

两个新E10尾段共4440更新，每行native3162962NN、source12600、formal1176002，共享真实父无query smoke4，总新远端8703132样本前向；分析/本地实际CVS NN/IQ/CP=0，额外学生/教师NN=0。各新模型累计E30/6660更新；历史As20父4797722NN/1810.249311秒复用不重训。每新模型累计native7960684NN；去重共享As20后的父＋两尾段native11123646NN，父＋所有新NN13500854。四逻辑行含两个不同历史父和四尾段，唯一模型epochs80；本轮新增只有20。

实际训练墙钟C=945.671424秒、P=919.782436秒，进程时间和1865.453860秒，不能当并行经过时间。提交至完整分析1054.796420秒；CPU分析39.824285秒。source C=1.931932/P=2.147104秒；formal C=50.555812/P=51.595488秒；池加载C=.054497/P=.039059秒。池历史生成142080次/287.656秒，新模拟0；无私有mask抽样。没有隔离loss算子墙钟，不能由不同GPU墙钟宣称新机制加速。

已执行35聚焦检查：20协议负例、6概率聚合CPU、1复用实际P70hook的六前向toy、8分析CPU；21新py AST PASS。一次独立P0/P1 PASS；一次归档本地/远端比较与24py remote compile VERIFIED；真实父student/ownEMA严格无query smoke4NN PASS。启动后PID/CWD/argv/GPU/log增长、实际模式/角色/系数核对VERIFIED；完整GPU训练每行2220更新，模型与头joint正常更新。终态独立回读ANALYZED/missing=[]，driver与两worker均自然退出，无failure，未干扰其他健康实验。

CPU分析先验证全4逻辑row×7预测再truth连接，独立复计504混淆矩阵、84分区汇总。4PNG均实际目视检查，4PDF非空且头有效；完整日志用于曲线分析，不以抽样冒充完整。

## 命令与交付

普通非特权N607 CVS-RFFI Python3.10.19/Torch2.1.0+cu121/NumPy2.2.5、FP32/CPU2；CWD=PROJECT/releases/cvs_FcrossRX_AsMarginal_E30_20261019_r01。实际入口：

```text
CVS-RFFI/python -u experiments/cvs_as_marginal73/runner.py
```

唯一owner=codex/root/as-marginal73，immutable run=20261019-phase1-FcrossRX-AsMarginal-e30-s2026092701-m2-r01，C1/GPU0、P1/GPU1，遵守每GPU最多2总实验进程和free≥12000MB。第一步smoke通过即继续，训练冻结后逐row自动all7与独立scorer。技术停止及唯一矩阵保留于experiment.json/[预登记](report.md)，无新增审批/hash/receipt链。

- [全部逐row/逐seed指标](E:/type10-7/local_artifacts/cvs_FcrossRX_AsMarginal_E30_20261019_r01/retrieved_r01/analysis_r01/formal_metrics.csv)、[完整配对变化](E:/type10-7/local_artifacts/cvs_FcrossRX_AsMarginal_E30_20261019_r01/retrieved_r01/analysis_r01/formal_pair_summary.csv)、[分层净纠错](E:/type10-7/local_artifacts/cvs_FcrossRX_AsMarginal_E30_20261019_r01/retrieved_r01/analysis_r01/formal_paired_effects.csv)。
- [源分层](E:/type10-7/local_artifacts/cvs_FcrossRX_AsMarginal_E30_20261019_r01/retrieved_r01/analysis_r01/source_strata.csv)、[源配对纠错](E:/type10-7/local_artifacts/cvs_FcrossRX_AsMarginal_E30_20261019_r01/retrieved_r01/analysis_r01/source_paired_effects.csv)、[原父正确集合](E:/type10-7/local_artifacts/cvs_FcrossRX_AsMarginal_E30_20261019_r01/retrieved_r01/analysis_r01/P43_As20_clean_cohort.csv)。
- [训练曲线数据](E:/type10-7/local_artifacts/cvs_FcrossRX_AsMarginal_E30_20261019_r01/retrieved_r01/analysis_r01/training_curves.csv)、[实际覆盖](E:/type10-7/local_artifacts/cvs_FcrossRX_AsMarginal_E30_20261019_r01/retrieved_r01/analysis_r01/training_As_coverage.csv)、[逐步CE聚合](E:/type10-7/local_artifacts/cvs_FcrossRX_AsMarginal_E30_20261019_r01/retrieved_r01/analysis_r01/marginal_steps.csv)、[质量组聚合](E:/type10-7/local_artifacts/cvs_FcrossRX_AsMarginal_E30_20261019_r01/retrieved_r01/analysis_r01/marginal_quality_groups.csv)、[成本](E:/type10-7/local_artifacts/cvs_FcrossRX_AsMarginal_E30_20261019_r01/retrieved_r01/analysis_r01/costs.json)。
- [训练曲线PNG](E:/type10-7/local_artifacts/cvs_FcrossRX_AsMarginal_E30_20261019_r01/retrieved_r01/analysis_r01/full_training.png)／[PDF](E:/type10-7/local_artifacts/cvs_FcrossRX_AsMarginal_E30_20261019_r01/retrieved_r01/analysis_r01/full_training.pdf)，[源分层PNG](E:/type10-7/local_artifacts/cvs_FcrossRX_AsMarginal_E30_20261019_r01/retrieved_r01/analysis_r01/source_state_quality_lock_noise.png)／[PDF](E:/type10-7/local_artifacts/cvs_FcrossRX_AsMarginal_E30_20261019_r01/retrieved_r01/analysis_r01/source_state_quality_lock_noise.pdf)。
- [正式配对PNG](E:/type10-7/local_artifacts/cvs_FcrossRX_AsMarginal_E30_20261019_r01/retrieved_r01/analysis_r01/formal_paired.png)／[PDF](E:/type10-7/local_artifacts/cvs_FcrossRX_AsMarginal_E30_20261019_r01/retrieved_r01/analysis_r01/formal_paired.pdf)，[实际CE聚合PNG](E:/type10-7/local_artifacts/cvs_FcrossRX_AsMarginal_E30_20261019_r01/retrieved_r01/analysis_r01/As_CE_aggregation_readout.png)／[PDF](E:/type10-7/local_artifacts/cvs_FcrossRX_AsMarginal_E30_20261019_r01/retrieved_r01/analysis_r01/As_CE_aggregation_readout.pdf)。

模型seed仅1，未提供新机制跨seed复现。阶段单seedclean≤0.1pp满足；可重复clean正收益、lowurban>60%、suburban接近clean仍未完成。sourceFAIL关闭固定候选，不以正式结果翻案、调参、选seed或重跑；当前goal继续ACTIVE。
