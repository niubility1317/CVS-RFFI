# 研究纠偏：完整源训练与普通星地拼接参照

2026-10-17，按用户最新指令执行。基线仍为 F＋跨 RX 同 TX，模型仍三路 SIG_TIME／统一 Sinc／joint160／默认 cosine 头正常联合训练／NoExplicitCFO，预算每模型 scratch E20（13L＋7U、4440更新）。不改结构、头、原 U 或正式 received IQ。

## 已确认的问题

最近多轮主要推进源统计、输入可辨识性和轻度 IQ 入口反证。它们产生了可保留的负结果，但没有直接改善 practical 身份识别；以这些活动作为主研发方向，偏离了本轮星地性能目标。健康、可拟合或统计风险下降均不算识别收益。已有部分新组织也没有超过其匹配的普通星地对照，不能用更复杂来代替实际增量。

已有普通配对 practical 的五源留出结果：一个模型 seed 下 low urban 平均 +14.081481 pp，urban 均值 +20.039506 pp，clean 平均 +0.881481 pp，但 RX6/RX8 的 clean 分别 −3.648148/−2.129630 pp。它证明了该旧源折下的实际星地适应，也保留 clean 负结果；不能推导完整源训练的绝对性能或把跨实验分数直接排序。原源折全部拟合四个RX，训练 L5040/U45360；完整源是 L6300/U56700，二者不是同训练覆盖。减少覆盖的实际降幅尚无完整匹配结果，不能编造一个数值。见[原五折完整报告](E:/type10-7/automation_reports/CV-SincNet/20261016-phase1-FcrossRX-practical-source-confirm-s2026092701-m8-r01/report.md)。

## 立即生效的训练契约

后续新训练使用全部原 source RX1/3/4/6/8、day1/2/3、6TX，完整 L6300／U56700（TX隐藏）／原 V27000。训练不再留出一个源RX，也不建立L5580/Ldev720内部替代契约。原目标RX与源RX的物理隔离继续保持；不把目标放回训练。原V仅健康、诊断和合法固定规则，不能反向传播或更新EMA。新训练从零，不复用任何源留出父权重。

“所有源留出折先通过才可完整训练”不再是本研究的入口。既有关闭结果和健康旧任务保持；用户这次新指令明确要求完整源训练，允许建立普通拼接参照，不能把这解释为旧固定候选已成功或抹掉其失败。A_g 仍是最终目标的一部分，暂不以新的统计入口检验阻塞 A_s 的直接性能实验。

## 当前唯一训练假设与最小参照

当前检验重点是：在当前 F＋跨RX同TX和完整源覆盖下，普通 clean/source＋practical 配对监督能否取得实际星地净收益，并保持clean。先给它一个公平、可复用的参照位置，再研究新方案；不以旧架构、E50或减少源覆盖的结果替代。

最小配置为同一个预定seed2026092701的三个scratch E20：

|行|监督与作用|L前向／物理包|每模型L前向包量|
|---|---|---:|---:|
|NATIVE|原F＋跨RX同TX，两次同ordinary IQ的RDrop，原SupCon|2|1136640|
|REPEAT|两对使用同一ordinary IQ；两对各0.5权重，SupCon只第一对|4|2273280|
|CONCAT|0.5普通源IQ对＋0.5完整practical(raw)对；原SupCon只普通源对|4|2273280|

REPEAT用于隔离额外前向／随机子网络计算的作用，CONCAT−REPEAT衡量普通星地视图替换的实际增量，NATIVE保留原底座本身。每个RDrop对用同一个已生成IQ，不重抽增强；旧ordinary不冒称新Ag。共同普通源对在两个匹配行中权重均0.5，另一个槽显式由重复ordinary换为practical；不宣称保持了NATIVE的完整1.0 ordinary监督。没有新增raw分支或新的对齐损失。

practical直接作用原规范化L，避免ordinary的PA/AWGN/multipath前置叠加；全部合法satellite L监督，不设学生高置信门槛。训练沿用原最朴素C1的low_suburban/mid_urban/low_urban均匀先验及完整传播、噪声、质量、补偿、AGC与边界依赖，不调弱测试信道。训练和正式测试随机流分离，原U不新增增强。三模型合计60model-epoch／13320更新，每模型仍E20，不是一个模型训练E60；额外前向、信道生成、原U/教师、评价与墙钟实测分列。

所有固定行final20冻结后逐行自动完成既有正式clean＋六practical预测，全部视图预测固定后独立truth-last评分。报告准确率、F1、urban均值、最差RX、分层、成对修复/新错及各suburban绝对值和对自身clean差。2pp作为预先固定的suburban接近探索量化，low urban>60%仍是最终目标，不预告必过。目标分数不用于调参、模型选择、候选晋级或选择性重跑。

若CONCAT相对REPEAT无星地净收益，或只靠降低clean缩小gap，就否定本假设，不声称简单拼接已解决联合目标；仍完整保留参照，不按得分扫描配比或补幸运seed。后续新方案必须直接对同契约CONCAT报告增量，不能只比无星地模型。

配置见[完整源参照配置](../configs/cvs_full_source_concat_reference_e20.json)与[活动契约](../configs/cvs_joint_augmentation_F_crossrx_e20.json)。当前全源loader适配、三行目标及逐行自动评分入口已实现并登记 `20261017-phase1-F-crossRX-fullsource-concat-e20-s2026092701-m3-r01`；本地33协议反例及语法PASS，真实F合成检查和发布前最小验证待执行，尚未启动新训练，没有新训练、IQ、checkpoint、NN、更新或target访问。不是训练成功报告。
