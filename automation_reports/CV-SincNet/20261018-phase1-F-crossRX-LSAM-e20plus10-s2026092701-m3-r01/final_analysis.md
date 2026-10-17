# L-only SAM 匹配 E30：完整单种子结果

20261018-phase1-F-crossRX-LSAM-e20plus10-s2026092701-m3-r01

状态：ANALYZED／VERIFIED。三行均完成新增10轮、累计E30、固定七视图与独立truth-last。只有一个模型seed；本项是A_s训练组织，没有新的IQ A_g。

|模式|clean %|high sub %|mid sub %|low sub %|high urban %|mid urban %|low urban %|
|---|---:|---:|---:|---:|---:|---:|---:|
|NOAS_REPLAY|85.226190|45.927381|44.232738|42.184524|44.646429|37.236310|31.663095|
|AS_REPLAY|82.702381|81.582738|79.331548|76.963690|78.214881|64.421429|53.442857|
|AS_LSAM|82.827381|81.333929|79.031548|76.679167|77.993452|64.141071|53.193452|

|比较|clean pp|mid urban pp|low urban pp|
|---|---:|---:|---:|
|AS_REPLAY__NOAS_REPLAY|-2.523810|+27.185119|+21.779762|
|AS_LSAM__AS_REPLAY|+0.125000|-0.280357|-0.249405|
|AS_LSAM__NOAS_REPLAY|-2.398810|+26.904762|+21.530357|

|模式|high suburban距clean pp|mid suburban距clean pp|low suburban距clean pp|
|---|---:|---:|---:|
|NOAS_REPLAY|39.298810|40.993452|43.041667|
|AS_REPLAY|1.119643|3.370833|5.738690|
|AS_LSAM|1.493452|3.795833|6.148214|

源判据：CLOSED_FIXED_LSAM_SOURCE_RECOGNITION_FAILED。源V1800由原V中每TX/RX/day按opaqueID固定20包构成，全部5RX/90单元，clean一次、6practical各2draw；每行23400NN/21600物理处理/0更新/0U/0target。源判定函数仅消费源产物，目标结果未用于调半径、模型、父、候选或重跑。全部三行均为预登记必测，无论source判据如何均完成固定测试。

|源AS_LSAM相对AS_REPLAY|增益 pp|要求|
|---|---:|---|
|clean|-0.111111|≥0|
|practical_mid_urban|-1.500000|≥0|
|practical_low_urban|-0.611111|≥+.5|
|practical_high|-0.500000|≥−.25|
|practical_mid|-0.611111|≥−.25|
|practical_low_suburban|-0.611111|≥−.25|

若源判据失败，关闭固定rho.05／存储坐标／L-only范围／父／seed／优化与预算配方；不以正式目标结果救回。若源判据通过，仅为单种子源资格，不从目标成绩自动扩seed。

全部五源RX[1,3,4,6,8]、day[1,2,3]、完整L6300/U56700隐藏TX/V27000、split392005和256点保持；未留出训练RX或使用内部L_dev协议。F三路SIG_TIME／统一Sinc／joint160／原cosine6×160 scale30／NoExplicitCFO及必要补偿保持，骨干与默认头联合更新，单独拟合头0。

NoAs用原REPEAT20，As两行严格共用原CONCAT20；每行恢复自身原EMA并重置AdamW/cos10追加10轮，累计E30／6660更新。两种treatment父的权重不同，源角色与初始化规则一致，均固定源final20，无目标挑选或不匹配父继承。旧ordinary保留，所以NoAs不是完全无增强。L中稳定共有ordinary.5，第二对NoAs为同一ordinary.5，As为完整practical(raw).5，SupCon.1和原U规则保持；没有在ordinary后再次施加完整星地信道。

三行均8次L前向和每步2次额外L VJP；重放原4份已生成输入及逐forward随机失活，不重抽ordinary或As。中性控制不改总梯度；SAM仅把L梯度替换为固定参数球邻域的L梯度，然后精确恢复权重、RNG和buffers，原clip5/AdamW/EMA各一次；原U及其余目标在原权重的梯度保留。rho.05在F实际存储坐标含CENTER25K q，不声称物理Hz不变，也不是IQ变换。

完整读取三行2220步、10epoch、370池、4440条L/U序列、2220条参考/units/replay记录；原始物理输入、U序列及全部生效参数按允许差异核对，实际总8LNN与原U/教师预算分别验证。标量重放记录与执行中的精确tensor恢复断言一致，但没有从标量重构所有实际梯度张量；不能将梯度范数、loss变化或健康检查当识别收益。

正式378个CM／63个分区／21配对、源39个CM与RX计数已独立复算，分析0IQ/CP/NN/新更新/原始truth。具体RX/TX/day全部保留见stratified_results.md；修复、新增错误、双方clean正确集合与clean∩scene联合正确见result.json。

新增真实训练＋源资格＋正式预测NN 14792052；真实父CP烟测4、开发合成NN18432／更新6／额外FVJP24另列；两份唯一原E20父NN 7322164。新增真实更新6660／训练extraVJP13320，唯一累计15540更新／70 model-epochs；各model累计30轮。逐行teacher、NN、实际物理As生成、source/score/墙钟保留于result.json，不隐藏额外前向或父训练成本。

正式目标判据：{"clean_net_positive": false, "lowurban_above60": false, "each_suburban_gap_at_most2pp": false, "source_qualified": false, "IQ_Ag_proved": false, "cross_seed_proved": false, "joint_goal_complete": false}。来源或单seed改善均不足以证明IQ A_g、Ag×As交互或联合目标；当前没有新增B10/B11，因此四因子交互未定义，不填0，也不叠加其他实验增益。

图中L/CE是原权重下的完整实际日志，SAM的更新梯度来自参数邻域重放，不能把原权重loss曲线称为SAM最终优化目标值。V clean为已见源健康指标，不是跨RX部署结果。
