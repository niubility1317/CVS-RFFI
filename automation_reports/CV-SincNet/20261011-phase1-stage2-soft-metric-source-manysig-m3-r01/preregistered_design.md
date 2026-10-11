# 完整维度软度量源判别：预登记

LOCAL_VERIFIED，尚未发布。沿报告3.2、3.3、9.1检验已有身份表示中的变化度量是否能用于更好的分类。原物理42维特征读出已完整完成且增量不稳定，本轮不重复它，也不新增物理身份分支。

固定原始CE E200三父，不使用S3/Z3初始化。仅复用合法L_s/V的joint160、L_s八个局部响应与原头；所有拟合只用L_s。V只评分。无身份优化器或更新，无新增骨干前向，无target。读取source准备完成记录、实际父契约、完整6300/27000物理ID、V顺序及父margin复算。

固定矩阵：isotropic、packet、packet_L、packet_T、packet_R、packet_LTR。每个seed有全L_s拟合、5个RX留出、3个day留出，共9组；对应V全部/对应RX/对应day评分。共162个解析读出。父骨干已经见过所有源条件，留出只相对本次读出拟合，不宣称未见RX骨干泛化，也不宣称未见TX实验。不存在用V拟合的标准化、中心、协方差或可靠度。

每类中心按合法TX/day/RX cell等权。C_packet为各cell包残差协方差等权均值；L/T为原有单位明确Jacobian与I8/6先验的协方差，非真实环境分布。R在每个outer拟合集内部重建跨TX pair与留一日共享RX偏移，固定shrink=.4；以inner留出完整向量MSE计算c，再累计c*p*p^T。不会复用已见outer留出条件的S3 bank，也不把全部TX×RX变异当作可信nuisance。R不使用alpha或目标可行性，因为本次只估计变化度量，不构造位移干预。R全向量c仍非判别相关可靠度，本轮明确验证其分类作用。

C = C_packet + 对应C_L/C_T/C_R + 1e-5 I160，所有系数1沿原谱分析固定，不搜索。isotropic为I160。P=C^-1/2保留全部160个正方向，评分30*cos(zP,mu_y P)。这是均衡类中心的soft metric读出，与原cosine分类头独立对比，不把它写成新增神经网络已训练。维度、表示归一化与原信号变换保持。完整SPD矩阵的逆平方根通过白化恒等式核验；不因为六类均值秩<=5而删去其余方向。全源metric_readouts.pt只供研究追溯，不自动部署。

主候选预先固定packet_LTR、匹配控制packet，其余为分量说明，不能主候选失败后改选最高分。推进条件逐seed均满足：留RX V宏平均准确率严格高于packet和原父；全L_s拟合的V准确率比原父下降不超过0.2pp、任一RX不超过0.5pp；留day V宏均值不低于原父。通过仅表示值得设计后续身份训练，不等于target达标。失败保留全部数据，不调rho/权重/挑seed。本轮source-only不自动target；84%仍由以后冻结的三seed clean结果验收。

|ID|报告章节|要求|目标文件|状态|验证|备注|
|---|---|---|---|---|---|---|
|M01|3.2/9.1|可信变化度量与身份判别联系|metric_audit.py|implemented|完整SPD及同坐标旋转检查|不新增硬投影|
|M02|3.2|RX/day条件可迁移|metric_audit.py|implemented|外部fit隔离、内部R跨TX/day|只读出留出，父已见源条件|
|M03|9.1/9.3|先验证增量再引入新特征|本预登记|implemented|固定primary与推进规则|物理分支继续deferred|
|M04|数据角色|V仅评分、无target|metric_checks.py|verified|改V特征/标签不改变任何拟合参数|L_s角色不扩展|
|M05|执行|不可覆盖run及三seed产物|metric_publish.py|pending|独立P0/P1及远端读回|CPU2，无训练|

7项检查通过：全维SPD、共同旋转不变、外部条件与V拟合隔离、内部R留TX/day、零动作恒等、逐包推理无批依赖、缺cell拒绝。发布前按既有流程独立正确性审查和Git读回，不增设审批。
