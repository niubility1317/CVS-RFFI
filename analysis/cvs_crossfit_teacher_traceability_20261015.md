# 独立支线：固定 RX1 两教师源互补小试已完成

用户fab17893附件：首版只训练两位等规模教师，证明其能纠正既有错误后再考虑完整库与学生蒸馏。实现位于 experiments/cvs_crossfit_teacher50/。训练、评价、门控回放、发布及登记均已完成；真实源域互补结果 FAIL。主线HQ E50健康运行，本支线保持独立，不与Dropout、bridge或关系损失组合。源小试模型仍各E50，首轮不扩教师库或KD学生。

|要求|状态|证据／剩余项|
|---|---|---|
|固定原SIG_TIME/default head，无新增部署模块|verified|真实模型863730参数、相同seed初态相等|
|两个教师同4464 L，TX/day配额相同，排除RX1完整|verified|source subset单测：每TX/day248、每TX744；普通组全5RX、排除组4RX；物理顺序改变不改结果|
|无U iterator/forward、无EMA／辅助损失|verified|actual args关闭；installed原222步预算在ForbiddenU上从不迭代U的测试|
|原头与t2/SIG projection端到端CE|verified|真实CE梯度与RX1/非法样本负例测试|
|真实数据过滤、实际每批物理ID与全继承来源|verified|两行最终completion/initialization/inner contract/完整coverage实际匹配4464；EXCLUDE仅RX3/4/6/8，scratch无继承/target|
|E50=11100真实更新与训练资源计量|verified|两行各50/11100，完整22200步解析；train1420800/eval2754000每行，实际墙钟及限幅记录已取回|
|冻结后RX1源L_dev144纠正/破坏/真类概率/day|verified|SAMPLE clean144/144，EXCLUDE102/144，修复0/新错42；七视图、概率/间隔/day/混淆矩阵完整|
|原native门控拒绝区域|verified|真实source replay RX1 rejected136，净修复-41；gate不读TX，非历史U决定；强分支原状态未改变|
|一次必要审查、登记、发布与独立读回|verified|一次P0/P1 PASS；正式登记；archive/compile/真实CPsmoke PASS；PID3176235/3176260/3176337与GPU3/6/CWD/argv/训练流增长独立VERIFIED|
|原教师互补条件与源推进结果|verified|实际FAIL：clean净-42，拒绝区净-41，高置信错0→6.25%；low源净-24；不扩教师库/学生|
|五对教师库＋BASE/SAMPLE-KD/RX-KD学生|deferred|先两教师source证伪，不立即扩库；不对齐独立教师特征|

计数：verified10、pending0、deferred1、rejected0、blocked0。启动前4项本地检查PASS（1.468s），当时仅有源构造／CE接线证据。现已完成真实权重与源域评价：clean损失42/144，高置信错增加6.25个百分点；无跨seed或正式目标结果。E50及内部L_dev契约不能冒充A的E20或V方案。

启动前新增4项评价检查PASS.700s；pipeline原3项2PASS/1fixture str.name失败，Path化后该源U守卫定向PASS.004s。compileall PASS。源真实机制守卫在CP加载前拒绝U/未完成暴露，旧门控实际native函数及train Dropout/BN临时buffer行为已用真实固定网络验证。总11唯一检查分批执行。最终真实源准确率与条件FAIL已取回，不扩库/学生。

实际初期读回两行4轮、各4464独立训练包；EXCLUDE实际coverage RX只3/4/6/8。最终全50轮/11100步和完整4464源覆盖已验证，不把初期状态当作性能收益。

最终真实源证据已完成，3 图已生成并逐图检查。clean 对照144/144存在天花板，无法证明纠正既有clean错误；六practical仍全部净负。source-only、seed1、无部署学生及正式测试收益。历史“待完成”叙述为当时状态，最终以上表与本段为准。
