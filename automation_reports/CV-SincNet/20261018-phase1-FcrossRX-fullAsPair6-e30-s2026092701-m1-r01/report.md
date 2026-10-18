# P68：固定六次前向内的独立完整 practical 配对 CE

状态：ANALYZED。全部七环境和同 row 分析已完成，missing=[]。源域 SOURCE_FAIL，不延长 E50；整体目标 ACTIVE。最终结论见 [final_analysis.md](final_analysis.md)，以下设计和判据保留为运行前记录。

最值得保留的是完整 practical 真实标签监督及稳定 clean 份额。原普通配方已显示星地适应收益，但新增损失／更高 U 接受覆盖等不能替代识别增量。P67 已完成全部评分，新增时间确认没有提高 low urban；另一对话的 LRaw、tailNoU 已实际训练，不重开。E50 是符合源证据的好方案的总预算，不是追加50轮。

本轮唯一假设：在原 6L／768 样本前向预算不变时，第二次 satellite 前向换成同一真实 L IQ 的独立完整 practical 实现，是否比重复同一实现的 CE 更有效。不是串联两次完整信道，不增加前向或分类头，不要求不同信道的 logits／特征相等。

## 已有证据与本次关键改变

旧 practicalMC2 已真正训练两个独立完整 As；它采用10L、RX1留出及每实现双前向，源结论为负。本轮不把独立采样称为新算法，也不凭改名重开同一实验：关键变化是把独立实现放进已有6L的两个As位置，不在不同观测之间施加KL。原SAT18 AS_CE30已经提供同IQ／As-only KL-off的控制，直接复用，不另训。

SAT18 AS_CE20 从scratch20就关闭As-only KL，P43原As20则保留该KL。二者父训练历史不同。因此新模型唯一父是SAT18自己的AS_CE20 student＋ownEMA，主比较是同父 SAME_CE6；相对原P43 As30的差异不能全部归因于信道刷新。

## 固定矩阵与数据

| 逻辑条件 | 角色 | 训练来源及预算 |
|---|---|---|
| NOAS30 | P43原NoAs参照 | 已完成原E30，复用预测／评分 |
| AS30 | P43原fullKL As参照 | 已完成原E30，复用预测／评分 |
| SAME_CE6 | SAT18 AS_CE同IQ CE控制 | 已完成自身20＋10，复用预测／评分 |
| INDEP_CE6 | 本轮唯一新行B01 | 从相同SAT18 AS_CE20追加10轮 |

固定F＋CROSS_RX、SIG_TIME_X3／统一Sinc／CENTER_UNITS／joint160／默认cosine30，900722参数／206状态键。模型与原头正常联合更新，无额外部署分支或诊断分类器。完整L6300／hidden U56700／V27000，RX1/3/4/6/8、day1/2/3、六TX、split392005；无heldRX/Ldev。父epoch20／4440更新、source-only scratch／final_only，无未知上游或目标选模。沿用已验证角色句柄，不重复数据验证。

两次As来自相同raw物理L，A0保留原命名流，A1用独立传播／噪声流；硬件同RX/day/eq session，非TX定义，两个接收质量与补偿状态各自重算。全部合法有标签As均参与CE，不设困难学生置信度门。原raw／ordinary KL、dropout、SupCon ordinary128、U规则／ownEMA、loss尺度保留。As-only KL两边均关闭，CE均值和L总1.25／6L768保持。没有新Ag、额外一致性损失或AsU。

IQ2×256／25MHz，practical之前原逐包RMS／center=false，内部CENTER_UNITS保持。train fc2.462GHz、full residual／post_sync、EQ=false且G=1、quality adaptation／phase tracking／IQ compensation／AGC开启、reference_snr_db30／quality reference20、reflect。真leo_practical仍为pinned20261015_r02/experiments/adv3b02_xuc/code实现，发布后记录__file__。正式固定IQ、fc2.45GHz和六practical不变；新增噪声比不冒称总SNR。

## 两项固定源判据与E50

冻结最终学生后读取原完整V27000 clean与同一V1800×2draw×3practical。候选仅在以下两项同时满足时有总E50资格：

1. fullV clean−原NOAS30 ≥−0.1个绝对百分点；
2. pooled mid／low urban−max(原AS30，SAME_CE6) ≥＋0.5个百分点。

任一FAIL关闭固定方案，不扫窗口／阈值／新seed救跑。所有行仍完成clean＋六固定practical。源函数不读取正式成绩，主owner先保存源决定，再读取本轮正式指标。正式0.1pp阶段目标、源晋级和可重复双提升分开报告，不因正式分数改判。

## 执行、成本与停止规则

精确code commit将在实现冻结后填入experiment.json。本地工作树codex/phase1-urban-sigtime-20261014；远端普通N607现有CVS-RFFI Python3.10.19／Torch2.1.0+cu121／NumPy2.2.5，FP32／CPU2，无安装。CWD PROJECT/releases/cvs_FcrossRX_full_As_pair6_E30_20261018_r01，命令 python -u experiments/cvs_full_channel_pair68/runner.py；输出PROJECT/runs/20261018-phase1-FcrossRX-fullAsPair6-e30-s2026092701-m1-r01，唯一owner codex/root/full-As-pair68。GPU按单次preflight后队列分配，包含其他owner每GPU最多2进程／至少12000MB空余，保护健康其他实验。

首步一次真实AS_CE20 student＋ownEMA及一次合成6L反传无query smoke，预期772样本前向／0更新／0真实IQ，通过即训练。一条新tail10model-epochs／2220更新，累计30／6660；预期native3162962＋source12600＋formal1176002＋smoke772＝4352336新增远端NN，本地真实NN0。第二完整信道增加物理模拟次数，不能宣称零成本；旧三个E30、父E20和其教师成本单列，不重训／重预测／重评分。

只对错误角色／来源／runtime、非有限值、输出碰撞或错误As配对／session／CE预算语义触发所属run技术停止；保留产物，不自动重试。科学低性能不停止。一个commit、聚焦负例、唯一P0/P1、单归档比对／compile、一次启动后PID/CWD/argv/GPU/log增长核实按现有最小流程执行，不增加gate。

训练冻结后逐row自动固定七视图预测，再独立truth-last。报告全部accuracy／macroF1／最差RX／跨RX-day／错误修复新增、源state／lock／新增噪声区间、默认头间隔／固定集合保持、view0/view1/combined物理覆盖、完整可用训练曲线、实际NN／更新／墙钟和缺项。只有seed2026092701，不宣称稳定双提升或一定达到low60%。


实际代码 f520ae98791a80775b6b5923183fcffba7e91c73 已提交／推送并独立核实远端 identical。单归档SHA ca5482c667f9e39ccd9626c1b52536f8cb996184b6345295120f09ea7409e8b1、远端17个Python文件编译均VERIFIED；一次真实AS_CE20 student／ownEMA及合成6L反传smoke PASS772NN，0真实IQ／0更新。GPU0的所属PID/CWD/argv、实际计算进程和日志增长均VERIFIED。实际native args相对同父SAME只有run_id/candidate_id/output_dir差异；完整角色EXACT_MATCH，ownEMA严格恢复，原pinned leo_practical真实__file__已保存。


源规则已固定关闭（先于主owner正式指标读取）：fullVclean对原NoAs −0.029630pp PASS；pooledmidlow对同父SAME＋0.069444pp，未达＋.5pp，FAIL；low对SAME−0.027778pp。对原As的midlow＋.319444pp含前20KL历史差，不能归为独立完整信道收益。两条件任一FAIL即无E50，不扫seed/namespace/剂量救跑；正式七视图已经按状态完成，等待完整CPU分析。实际父cohort独立source诊断没有参与该决定。


最终交付：同父 INDEP−SAME 的 urban＋0.114286pp、clean＋0.019643pp；对同预算 NoAs 的 urban＋28.302778pp、clean−0.013690pp，正式单 seed 满足0.1pp阶段要求。相对原As urban−0.065675pp；正式low51.245833%，没有达到60%或可重复双提升。所有七视图已完成；详细F1／最差RX／分层／成本及缺项见 final_analysis.md、delivery.json。实际父source-only诊断另12602NN，本轮新增总4364938NN；未启动E50／扩seed。
