# 独立支线：固定RX1两教师源互补小试草稿

用户fab17893附件：首版只训练两位等规模教师，证明其能纠正既有错误后再考虑完整库与学生蒸馏。实现位于 experiments/cvs_crossfit_teacher50/；尚无发布器／完整比较器／登记或真实训练，本稿不声称可发布矩阵已完成。最新31cc8a22附件将语义增强作为下一核心开发，本支线保留为独立工作，不与Dropout、bridge或关系损失组合。

|要求|状态|证据／剩余项|
|---|---|---|
|固定原SIG_TIME/default head，无新增部署模块|verified|真实模型863730参数、相同seed初态相等|
|两个教师同4464 L，TX/day配额相同，排除RX1完整|verified|source subset单测：每TX/day248、每TX744；普通组全5RX、排除组4RX；物理顺序改变不改结果|
|无U iterator/forward、无EMA／辅助损失|verified|actual args关闭；installed原222步预算在ForbiddenU上从不迭代U的测试|
|原头与t2/SIG projection端到端CE|verified|真实CE梯度与RX1/非法样本负例测试|
|真实数据过滤、实际每批物理ID与全继承来源|pending|select/install/own_model guards已写；待真实源CP/训练证据|
|E50=11100真实更新与训练资源计量|pending|设计固定，未执行；不能把草稿当训练结果|
|冻结后RX1源L_dev144纠正/破坏/真类概率/day|pending|比较器待实现，不使用U真值或target|
|原native门控拒绝区域|pending|需要固定A0源replay并明确与实际U决策不同；尚未实现|
|一次必要审查、登记、发布与独立读回|pending|未启动，后续按最小流程，不额外审批|
|原教师互补条件与源推进结果|pending|source-only阈值在design草稿预定，无实际互补证据|
|五对教师库＋BASE/SAMPLE-KD/RX-KD学生|deferred|先两教师source证伪，不立即扩库；不对齐独立教师特征|

计数：verified4、pending6、deferred1、rejected0、blocked0。4项本地检查PASS（1.468s），只有源构造／CE接线的代码证据，尚无教师权重、性能、clean代价或跨seed结果。主风险是排除RX后反而增加高置信度错误；E50及内部L_dev契约不能冒充A的E20或V方案。
