# 独立支线：固定RX1两教师源互补小试草稿

用户fab17893附件：首版只训练两位等规模教师，证明其能纠正既有错误后再考虑完整库与学生蒸馏。实现位于 experiments/cvs_crossfit_teacher50/。正在补齐 evaluate.py／gate.py／dispatch.py／publish.py／登记；已发布并训练中，尚无最终教师互补结果。主线HQ E50健康运行，本支线保持独立，不与Dropout、bridge或关系损失组合。源小试模型仍各E50，首轮不扩教师库或KD学生。

|要求|状态|证据／剩余项|
|---|---|---|
|固定原SIG_TIME/default head，无新增部署模块|verified|真实模型863730参数、相同seed初态相等|
|两个教师同4464 L，TX/day配额相同，排除RX1完整|verified|source subset单测：每TX/day248、每TX744；普通组全5RX、排除组4RX；物理顺序改变不改结果|
|无U iterator/forward、无EMA／辅助损失|verified|actual args关闭；installed原222步预算在ForbiddenU上从不迭代U的测试|
|原头与t2/SIG projection端到端CE|verified|真实CE梯度与RX1/非法样本负例测试|
|真实数据过滤、实际每批物理ID与全继承来源|pending|select/install/own_model guards已写；待真实源CP/训练证据|
|E50=11100真实更新与训练资源计量|pending|设计固定，未执行；不能把草稿当训练结果|
|冻结后RX1源L_dev144纠正/破坏/真类概率/day|pending|evaluate/compare已接通；纠正/破坏、概率与空区域检查PASS，真实指标待训练冻结|
|原native门控拒绝区域|pending|原native阈值/temporal/strong重放已接通；真实模型functional强视图/状态检查PASS；明确不是历史U决定，真实分层待运行|
|一次必要审查、登记、发布与独立读回|verified|一次P0/P1 PASS；正式登记；archive/compile/真实CPsmoke PASS；PID3176235/3176260/3176337与GPU3/6/CWD/argv/训练流增长独立VERIFIED|
|原教师互补条件与源推进结果|pending|source-only阈值在design草稿预定，无实际互补证据|
|五对教师库＋BASE/SAMPLE-KD/RX-KD学生|deferred|先两教师source证伪，不立即扩库；不对齐独立教师特征|

计数：verified5、pending5、deferred1、rejected0、blocked0。4项本地检查PASS（1.468s），只有源构造／CE接线的代码证据，尚无教师权重、性能、clean代价或跨seed结果。主风险是排除RX后反而增加高置信度错误；E50及内部L_dev契约不能冒充A的E20或V方案。

新增4项评价检查PASS.700s；pipeline原3项2PASS/1fixture str.name失败，Path化后该源U守卫定向PASS.004s。compileall PASS。源真实机制守卫在CP加载前拒绝U/未完成暴露，旧门控实际native函数及train Dropout/BN临时buffer行为已用真实固定网络验证。总11唯一检查分批执行。仍无真实教师准确率或源条件通过，不扩库/学生。

实际初期读回两行4轮、各4464独立训练包；EXCLUDE实际coverage RX只3/4/6/8。真实全E50暴露/源互补仍待完成，不能据初期状态宣称正收益。
