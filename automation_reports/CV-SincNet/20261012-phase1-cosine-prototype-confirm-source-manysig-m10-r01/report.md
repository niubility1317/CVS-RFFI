# 普通余弦原型：独立模型种子源确认预登记

上一轮源端对照出现普通分类器信号：2701五RX，NATIVE_HEAD61.7111%、COSINE_PROTO66.0593%，平均+4.3481个百分点，4/5RX非负，最差44.6296%→44.7963%。SOFT_FINITE62.7778%，对普通Mahalanobis虽有+1.1630个百分点、但3/5RX非负，未过预登记；不降低门槛。这里不把原型收益归给软度量或多网络。

新假设只来自合法source，固定两个此前本矩阵未跑的模型seed2026092702/03，各五RX，共10个从零原CVS E20。训练路径与2701CLEAN_REPLAY完全匹配：每模型20×222=4440更新、mainclean128+extraeval160纯CE，reference显式CFO槽屏蔽，无U、EMA、卫星、辅助网络或祖先权重。

每份模型在四RX L5040训练，整个heldRX在骨干、分类头、教师、统计拟合中均未使用。按最后epoch20冻结，严格重载自己同row权重。合法同一fitL实际joint160单位向量按六类取均值并归一化，所有类别同公式余弦；不学习温度、协方差或判别头。heldRX V5400先固定native/proto两预测再评分，并核对native与训练收尾原记录一致。原V只评估、不拆角色，所有15TX类对、RX/day、混淆、macroF1保留。

## 冻结确认条件

每个新seed源平均prototype-native至少+2个百分点；两个seed合计十RX比较至少8非负；每seed原型最差RX不低于原头最差超过1个百分点。全过才发布固定全源三seed E20同模型两读出最终确认，逐row自动clean+六practical全部prediction固定后独立truth-last评分。未通过就保留，不择seed或修改条件。既有target为已暴露开发基准，不称新的独立盲测。

预算仍按现行E20，没有擅自拉长。当前阶段仅source-only，未访问目标或独立target支持。新不可覆盖root、owner、Gitcommit、命令、环境/CWD/GPU和产物详experiment.json。独立一次P0/P1、真实checkpoint无querysmoke、一次archive比对/compile后继续，每GPU最多3总CUDA进程。系统技术故障才停止本run，低性能完整保留，其他健康任务保持。
