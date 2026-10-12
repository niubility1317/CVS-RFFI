# 无显式 CFO 同路径 SSL 与工程探索（后24行）

本批固定八arm×三seed，共24随机学生；与先发布无CFO CE三行组成27行匹配矩阵。全部原生旧门控20 epoch／4440更新，reference响应slot27恒零，内部CFO补偿保留，final20自动clean168000后独立truth-last评分。原始IQ不保证无残余频率。无EMA、星地增强、Fishr或额外辅助网络。V仅健康验证，不选模。

|arm|与SSL不同处|同seed比较|
|---|---|---|
|SSL|新无CFO CE另两seed固定概率均值ALL softKL；L预热444、ramp888|SSL−CE|
|L_REPLAY|每活跃步一次额外同L clean CE前向／反向，同dropout重放、weight1×ramp|L_REPLAY−SSL|
|SSL_LR3|cosine峰值6e-4|LR3−SSL|
|SSL_WARM_LR3|6e-4峰值、444步从1e-6预热再3996步cosine|WARM−LR3，WARM−SSL|
|SSL_SAM|rho.02，全部可训练身份参数，两遍完整目标、一次AdamW，参数/RNG/buffer恢复|SAM−SSL|
|SSL_SMOOTH|仅L CE标签平滑.05，固定U软目标不变|SMOOTH−SSL|
|SSL_HIDDEN_MIX|L不同TX供体，Beta(2,2)取maxλ，特征/软标签同λ；浅time_down与中t2.pool交替，每步仅一层|HIDDEN_MIX−L_REPLAY|
|SSL_WARM_LR3_SAM|预热高LR＋SAM；工程主方案|组合−WARM，组合−SAM，组合−SSL|

主方案及所有超参沿用新报告前已冻结设计，未按新CE目标成绩改变。新CE源模型来源固定run/seed/commit/数据契约/final20/无祖先/无CFO，缓存schema独立；旧权重/旧缓存拒绝。每学生排除自己seed教师。合法U56700只含IQ/RX/day/opaqueID，教师生成恰三模型×56700读出，状态/RNG不变；不接触U TX、V或target。

每行568320L及511488U累计读出，SSL额外学生weak诊断/强噪声前向成本保留；SAM额外12432前向；replay/hidden各额外3996 L128前向和反向。隐藏供体选择/Beta/hook非零成本另说明。三教师共享缓存一次生成成本与历史训练40轮/学生分别列出。推理只用学生。

每卡最多3总CUDA进程、至少12GB空闲；唯一owner，共享锁，独立输出拒绝覆盖，失败保留产物、不重复启动、不干预健康任务。自动全106560步源日志核验、完整分层评分和实测资源报告，原登记状态与events更新。关系训练/有限作用/U前导诊断按报告另阶段，不静默叠加。

本地八arm原生CUDA损失/有限梯度/一步AdamW、SAM人工参考与故障恢复、层混合、clean replay、实际训练入口和strict自身checkpoint往返通过；新教师来源负例/U标签负例拒绝。发布仍需独立P0/P1及远端真实L/U/新源teacher预检与启动证据。

发布VERIFIED：实际启动24/24行，已核验24份实际源配置，其余按容量排队。8GPU总进程实测均不超过3；真实源L/U和全部固定新noCFO CE老师CPU预检PASS。执行commit 68d9221c07cbb77543cf2a4fe9b28e604863edbc；每行自动clean，当前矩阵评分未完成。

**VERIFIED / ANALYZED：24行训练及自动clean独立评分全部完成。** 106560完整源更新／4032000固定预测；加CE3共27行。最高三种子均值SSL_WARM_LR3为79.8986%，原SSL79.7873%，同输入CE79.4891%；主方案WARM_LR3_SAM78.8252%。未达到84%。全部方案、分层、配对差与实测成本见[final_analysis.md](final_analysis.md)。后续关系15行登记先于读取这些成绩，配置不据此改变。
