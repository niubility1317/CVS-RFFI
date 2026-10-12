# 跨接收机身份关系训练：固定15行

报告9.3要求将辅助目标直接对准身份判别。本批保留旧门控＋cos、L身份CE、同两固定源CE教师的U全样本软KL；随机20轮／4440更新，无显式CFO拼接、EMA、星地增强或新辅助头。三seed为2026092701/02/03。固定final20逐行自动clean168000，prediction固定后独立truth-last评分。V仅E5/10/15/20健康检查。设计在读取上一批24行成绩前固定。

|方案|关系层|原型规则|
|---|---|---|
|L30_CE|无关系；额外30L普通CE|样本预算控制|
|REL_FINAL_MERGED|最终joint160|合并其他4RX|
|REL_FINAL_SEPARATE|最终joint160|分别保留其他4RX|
|REL_MULTI_MERGED|浅time_down、中t2.pool的均值＋population std，最终joint160|合并|
|REL_MULTI_SEPARATE（主方案）|浅／中／最终等层平均|分别保留|

每4次身份更新执行一次支持读出；444步预热后从448到4440，共999次，每次30个合法L（6TX×5RX各1个，循环day1/2/3，同一步同day）。排除主L128查询中全部物理ID，支持自身也不重复。每个查询RX r只使用其他4RX的24个参考。编码器已经见过全部五源RX，此任务是已见RX间读出迁移，不能冒充未知RX验证。

支持由当前学生eval/no_grad读出，恢复模式且核验buffer不变。中心c_r是24个独立参考的均值；q=normalize(h_query-c_r)。合并规则p_t=normalize(mean_RX(h_ref[t,RX]-c_r))，a_t=cos(q,p_t)/0.1。分RX规则p_tj=normalize(h_ref[t,j]-c_r)，a_t=logsumexp_j(cos(q,p_tj)/0.1)-log4。全部6TX竞争，CE(a,y)仅经查询真实特征更新身份网络。组合层关系CE取均值，weight1×固定888步ramp。身份CE/U软KL保留。

L30_CE使用同支持选择／间隔／ramp，但支持普通train前向和反向；主RNG恢复。关系参考eval/no_grad，二者匹配额外样本与前向数，不声称完全同算力。每行额外29970L、999前向；控制有梯度参考999次，关系参考0次。原型无训练参数，无额外优化器步。完整源日志分别记录身份头准确率/margin、关系CE/准确率/margin、两读出规则的纠错与破坏、特征范数及梯度；这些统计不能冒充更新因果修复。

复用已登记无CFO源CE3教师和上一批U缓存，只读合法U56700，不读其TX。所有来源核对原始契约/commit/final20/空祖先；源教师不按目标成绩挑选。已固定CE、SSL预测只供独立scorer作配对控制。每卡最多3总CUDA进程，独占输出、共享容量锁，不干预健康任务。自动15行完整66600步源分析／2520000 clean评分／全seed、RX/TX/day、混淆与成本报告。

本地五方案CUDA实际前向／有限梯度／AdamW入口、自身checkpoint严格往返、原型手工参考、TX置换、查询RX排除、PID独立、模式/RNG/buffer恢复已通过。作用预测和U结构新证据属于报告9.4/9.5下一阶段，不由本批关系CE替代。发布仍需独立P0/P1和远端真实L/U/teacher源预检。

发布VERIFIED：实际启动9/15行，已核验9份实际源配置，其余按容量排队。8GPU总进程实测均不超过3；真实源L/U和全部固定新noCFO CE老师CPU预检PASS。执行commit 3ec1801e104aa894335c0dd537e761bc5df7a4f0；每行自动clean，当前矩阵评分未完成。
