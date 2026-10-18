# 后续候选的历史证据范围

本记录只核查既有源域证据与实际实现，不读取新正式分数，不启动候选，不改变健康E50四行。当前目标按用户最新要求执行：星地识别优先、匹配clean下降≤0.1个绝对百分点，最终仍需可重复双提升；不强制Ag+As或独立泛化收益，不留训练RX。常规E30、好配方及匹配对照可总E50，不默认增加到E70或加载E200。

## clean教师→urban U：已有当前F契约实跑

P39：20261018-phase1-FcrossRX-cleanurban-U-e30-s2026092701-m3-r01，已ANALYZED。F/SIG_TIME_X3/统一Sinc/joint160/defaultcos/CROSS_RX、原L6300/U56700隐藏/V27000、全部RX1/3/4/6/8和day1/2/3，无新留域。B0/URAW/UAS各scratch20+自身10，累计E30/6660更新。它们L为4次前向，当前P43/P50为6次；不能跨配方把前向成本差异归因U机制。

三行均有额外963900个U学生前向：B0对这条额外前向不加loss，URAW用raw，UAS用完整practical；原U流程保留。UAS额外U视图全部As，占两个U学生前向50%，其中low suburban/mid urban/low urban为321253/321561/321086。teacher复用自身EMA的原clean/raw out_w，无额外U teacher前向；T=1 softmax，q=clip((maxp−.95)/.05,0,1)，batchmean qKL，权重.16配5epoch ramp。不要求urban学生高置信，也不继承旧U多重门控。

teacher_eval NN=965700（原U963900＋首次V资格1800），teacher_train NN=0；自身EMA按原步更新。实际正q占88.6696%，旧门控接受比例1.4727%。U真值精度未知，不能用V的teacher正确率代替U精度。

合法源域：fullV clean B0/URAW/UAS为98.0556/98.0889/98.0778%，保护条件通过；UAS相对同成本URAW的四个mid/low×draw增量依次+0.1111/+1.5000/+1.0556/+1.2778个百分点，均为正，但当时预登记要求每格≥1，首格未满足，记录CLOSED_EXACT_C4_SOURCE_NEGATIVE且formal_scores_read=false。这只是该配方未满足当时的门槛，**不是clean损伤，也不证明所有SSL无效**。旧run结论保留；若重新研究须说明改变哪个关键假设和比较目的，保留同预算同NN控制，不能仅换名称或把当前较宽松目标静默写回旧协议。

[原源域判定](<E:/type10-7/automation_reports/CV-SincNet/20261018-phase1-FcrossRX-cleanurban-U-e30-s2026092701-m3-r01/source_development_decision.json>)；实现为experiments/cvs_Uurban39。更早SSL_CONF/SSL_WEAK E50属于single SIG_TIME并叠C2课程/C3，不是当前F的独立证据。

## 单侧默认头间隔：旧C3存在，当前F独立实跑未找到

精确公式位于[旧objective.py](<E:/type10-7/code/worktrees/phase1_urban_sigtime_20261014/experiments/cvs_urban_sigtime/objective.py>)：clean正确、stopgrad、rho=.5、clamp m_max1、cosine logits/30、连续q。其design明确single SIG_TIME；C3叠在C2课程上。旧registration的m_max字段继承不能证明当前F执行。当前cleanretain37/P43/P50是CE、同IQ RDrop和既有SupCon；P40是0训练的默认头梯度诊断；semantic_order50是embedding triplet/SupCon，class_W_dependency=false，均不能冒充此候选。

已核查canonical登记与两个工作树，未找到当前F＋CROSS_RX＋原全L/U/V＋all5RX下该单侧间隔的独立实跑和源判定。既没有当前契约下的有效证据，也没有精确匹配的失败证据。这个缺项不构成自动开跑许可或性能保证；本轮仍先完成固定原As预算的三seed复现，保持默认头和已有前向成本的约束。
