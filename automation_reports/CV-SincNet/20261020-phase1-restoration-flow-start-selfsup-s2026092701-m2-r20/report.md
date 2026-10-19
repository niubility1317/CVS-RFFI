# 固定起点与随机起点残差流匹配 R20

状态：**ANALYZED / SOURCE_NOT_QUALIFIED**。本轮没有新正式目标集成绩，持续目标尚未证实。 固定 40 轮新增训练、两行全部源评分及登记的有条件工作完成，控制器和本 run worker 均自然退出。方法 `98c1822f88baa546db267dd97efe00f10870afd3`，没有按目标反馈改变结构、阈值、排序或选择性重跑。


保持完整目标：独立恢复器输出一份 IQ，原 PAMF 从它一致计算 FFT/PA/统计/时间分支，冻结原 F50 seed2701/defaulthead。正式 low urban≥70%，其余五受损场景各距原始同 F/seed clean≤5 pp；恢复 clean 损伤单列。恢复训练全合法 L/U、0 TX标签、0 F 权重/输出、0 target；源 V 标签验收和历史监督 F 的知识来源单列。

依据用户六类报告 R5 的 sigma0=0 与 sigma0>0 区别，以及 R03 随机残差流源诊断的 7.03125% 类别切换。旧 R03 使用残差空间 sigma=.02，但混合退化与新 R07 缓存、物理求解配方不同，不能作本轮严格匹配控制。此假设和规则在 R19 源决策/正式结果未知时固定，R20 不使用 R19 新结果改结构、损失、阈值或参考。

R20 两行 `zero_start`/`noise_start` 使用相同条件速度场、完整初始权重和随机流，唯一差异为 sigma0=0/.02。只继承 R07 全量源训练的冻结 coarse/作用/原生白输入方差代理，不继承任何训练后 prior。加载父后重置 model2701；条件编码输入 received2/coarse2/lognativevariance1，在每个分辨率注入；速度场为 Core2＋时间编码。新 prior scratch E20/4940，固定最后一轮。

令 e=(x−b(r))/SCALE、z0=sigma0×epsilon、z_t=(1−t)z0+t e。独立抽取 t~Uniform[0,1]、epsilon~N(0,I)，回归 v(z_t,t,r,b,variance) 到 e−z0。目标是逐包 `SCALE²×mean((v−(e−z0))²)/mean(x²)` 的均值。两行即使 sigma0=0 也都抽同样的 epsilon 和 t，保持训练随机流匹配。teacher-path 单点端点误差只作训练诊断，不能冒充完整采样性能。

推理只使用当前 received、由其估计的上下文及预先声明的独立噪声，从 z0 出发执行16次 Euler；条件编码每包缓存一次，完整采样后得到 prior=b+SCALE×z1，再执行一次6步软MAP（warmstart为coarse）。只输出该恢复IQ，不输入clean/真实信道/真实噪声/TX/场景，不硬设Ax=r或剪裁。固定起点为确定性迭代恢复，不声称完整后验；随机起点也不因此自动证明后验正确。本轮是报告R5/R6的残差与软MAP适配，不是I2SB或Schrödinger bridge逐式复现。

模型本体对 noise_start 要求显式 epsilon；评价worker用独立torch.Generator按固定seed生成，与标签、IQ内容/ID、其他样本决策均无关。primary seed=202610202001，固定batch128；生成器只消耗随机数，不拟合或更新模型/原型/归一化。给定同一epsilon时，逐包与合批必须相同；不把改变随机数实现后的不同样本宣称数值不变。

源1800V七视图的主预测全部固定后，在固定前128条lowurban上额外用202610202002/003/004三个独立seed完整采样，原F默认头读出，报告三draw类别分歧率，阈值≤5%。这些只使用源输入与冻结F预测，不读取TX标签，不挑draw；primary不按诊断更换。零起点的分歧率为0是构造属性，不是忠实恢复的证明。额外128条是1800的子集，不增加独立样本数。全部主预测及诊断验证完成后，独立CPU才连接源V真值。

数据/预算与R17保持：6300L/56700U，每行L126000/U1134000曝光，R07原20轮15%clean/85%residual缓存，data2001、物理增强1912，训练t/epsilon独立seed202610202000。batch256、AdamW2e-4→2e-5、warm247、WD1e-4、clip1、FP32。无额外FFT/解析/F损失、身份监督、教师或校准。coarse20+action20+prior20=组件E60，F50历史E50单列；16次推理不是16份物理观测。

源资格继续保护native/R15FFT/R16IQ/R17unshared四参考：mean6NMSE≤各参考1.05倍、cleanNMSE≤.02，low/midurban各≥+.5pp、clean下降≤.1pp、最差低urbanTX≥各参考−1pp、其他五场景最差变化≥各参考−.5pp，并加原生成式≤5%类别分歧条件。合格者按low/mid均值、最差TX、最差RX最高及名称排序冻结一行；无合格者0target，有合格者自动原168000×七视图、原F50/defaulthead、独立truth-last。R17既有正式对照只在新评分后复用。

R20任何GPU worker必须等R19控制器和全部所属worker真实退出后才可启动，避免跨run合计超过两条本任务GPUworker。之后池0/1/5/7，最多两条、每卡一条、全owner cap2/free≥12GB、共享capacity4锁到CUDA PID可见；普通账户、不可覆盖root/release、唯一owner。真实2Ls/noquery入口PASS立即继续；技术失败保留输出、失败行/pending不推进，健康任务继续，不自动重试。



以下均为 1800 条 source V 的百分比，不能代替正式目标集成绩。原 F50 seed2701/defaulthead；每个源视图的 TX/RX/day 分组计数、accuracy、Macro-F1、混淆矩阵及 repair/damage 完整保留于[汇总](summary.json)和[逐组数据](row_metrics.csv)。

|源场景|native_reference|fft_reference|single_shot|unrolled_reference|zero_start|noise_start|
|---|---|---|---|---|---|---|
|clean|98.4444|98.4444|98.4444|98.4444|98.4444|98.4444|
|高仰角郊区|98.0000|97.9444|97.8333|97.8333|97.7222|97.8333|
|中仰角郊区|96.6667|96.6667|96.9444|96.7778|96.1111|96.2222|
|低仰角郊区|95.9444|96.0556|95.9444|95.9444|94.3889|94.5000|
|高仰角城区|97.0000|96.8889|97.2222|97.0000|96.7222|96.7222|
|中仰角城区|89.8889|90.1667|90.7222|91.4444|87.0556|87.4444|
|低仰角城区|82.1667|82.5000|84.3889|86.8333|78.1111|78.0556|

|源 IQ NMSE|native_reference|fft_reference|single_shot|unrolled_reference|zero_start|noise_start|
|---|---|---|---|---|---|---|
|clean|0.00009617|0.00009617|0.00009617|0.00009617|0.00009617|0.00009617|
|高仰角郊区|0.01046707|0.01042492|0.01090087|0.00898618|0.01254193|0.01253031|
|中仰角郊区|0.01539733|0.01535856|0.01554777|0.01297654|0.01876491|0.01893271|
|低仰角郊区|0.01939996|0.01926939|0.01919486|0.01615114|0.02367322|0.02399207|
|高仰角城区|0.01474365|0.01460483|0.01465043|0.01238056|0.01779032|0.01779266|
|中仰角城区|0.04131214|0.04097974|0.03884036|0.03438081|0.04963016|0.05039678|
|低仰角城区|0.06428189|0.06342117|0.05999025|0.05259345|0.07680566|0.07791764|

|候选|参考组合|波形|身份|通过|low增益pp|mid增益pp|最差TX差pp|
|---|---|---|---|---|---|---|---|
|zero_start|R16/native|False|False|False|-6.2778|-3.6667|-12.2867|
|zero_start|R15/native|False|False|False|-4.3889|-3.1111|-8.3916|
|zero_start|R17/native|False|False|False|-8.7222|-4.3889|-15.0350|
|noise_start|R16/native|False|False|False|-6.3333|-3.2778|-12.2378|
|noise_start|R15/native|False|False|False|-4.4444|-2.7222|-10.4895|
|noise_start|R17/native|False|False|False|-8.7778|-4.0000|-17.1329|

固定合格排序为 `[]`，选定 `None`。四个参考同时保护；不以平均 NMSE 替代最差 TX 条件。

|源 low urban TX|native_reference|fft_reference|single_shot|unrolled_reference|zero_start|noise_start|
|---|---|---|---|---|---|---|
|14-10|74.7508|77.4086|77.0764|78.0731|74.4186|74.7508|
|14-7|70.2341|72.2408|69.2308|73.5786|65.8863|63.5452|
|20-15|83.9161|85.3147|87.0629|91.9580|76.9231|74.8252|
|20-19|78.4983|71.3311|82.9352|83.9590|70.6485|73.7201|
|6-15|89.9351|92.5325|94.4805|95.4545|86.6883|87.0130|
|8-20|94.8882|95.2077|94.8882|97.4441|92.9712|93.2907|

两候选均未通过全部四源参考条件，因此本 run 保持 **0 target**，没有启动目标预测或 truth 评分。源域结果不代替正式 70%/5 pp 目标；没有放宽条件或选择性重跑。

本轮无源合格候选，未访问目标数据，目标配对细分诊断不适用。

|行|总参数/可训练参数|原训练秒数|batch1中位数/P90 ms|
|---|---|---|---|
|unrolled_reference|4665073/2507526|1668.396|44.028/45.129|
|zero_start|3546797/1389250|982.997|127.732/132.933|
|noise_start|3546797/1389250|975.766|129.464/133.285|

两新网络各 1389250 个可训练参数，总参数 3546797。训练每包一次速度 Core，推理每包完整 16 次 Euler 速度 Core、一次条件编码和末端 6 次软 CG。训练日志中的 teacher_path_endpoint_nmse 使用配对真值路径，只作训练诊断，不是完整采样成绩。主评价使用固定独立随机流和 batch128；没有挑选 draw，零起点的随机分歧为零是构造属性。额外三 draw 使用固定 low urban 前128条 source V 子集，不增加独立样本数；实测采样身份分歧见下表，≤5% 仅为一项保护条件。

|候选|固定子集|额外完整draw数|类别分歧率（%）|≤5%|
|---|---|---|---|---|
|zero_start|128|3|0.0000|True|
|noise_start|128|3|10.9375|False|

batch1 含冻结上下文，固定源低 urban 16 样本、两次预热，所在卡与其他负载不同，不是严格同卡速度测量。新 prior 各 E20，组件累计 E60；原 F50 E50 和历史参考成本另列。

完整日志解析：新增40轮、三参考历史60轮，135行，致命错误0，warning 0。每行4940更新、L126000/U1134000曝光，全部真实记录参与，恢复训练 TX/F/target读取均为0。28项本地聚焦检查、compile/help、一次独立P0/P1发现的源摘要收尾缺陷已failed-first修复并定向复核、N607预检、归档单次SHA比对、远端编译、真实checkpoint/原F50/训练速度梯度/16次Core及6CG入口、PID/CWD/argv/GPU及日志增长均 VERIFIED。新权重已取回本机，从登记工作树严格加载并连接原 F/defaulthead，两条真实 L_s 输出有限，逐包与合批一致性检查通过；没有参数更新或标签读取。

入口速度训练梯度范数 `{'zero_start': 0.0018330930033698678, 'noise_start': 0.0018044839380308986}` 仅证明连通性。汇总 prior_parameter_changes 记录条件编码、速度 Core 和时间编码相对初始参数的变化，以及零初始化输出投影的最终范数；不将参数更新量当作身份贡献证明。固定起点为确定性迭代，不代表完整后验；随机起点也不能自动证明后验正确。当前 received 是条件，未输入真值 TX、真实信道或真实噪声。最终性能和同 TX 统计改善仍不构成逐采样点真实性保证。单seed、同信道引擎代理、已暴露基准和未完成的新oracle反事实是边界。此轮为报告 R5/R6 的残差流匹配与软MAP适配，不是 I2SB 或 Schrödinger bridge 逐式复现。

验证命令：`python -X utf8 -m pytest tests/test_pamf_flow_start_restore.py -q`、`python -X utf8 -m compileall -q experiments/pamf_flow_start_restore`、`python -X utf8 -m experiments.pamf_flow_start_restore.runner --help`。矩阵、命令、环境/CWD、路径、GPU、停止规则见[登记](experiment.json)，[追踪记录](../../../docs/PAMF_FLOW_START_RESTORE_R20_20261020.md)为7 verified、1 deferred、0 rejected/blocked。持续目标仍保持 active，权重、数据与原始日志不提交。


训练路径单点误差与完整源采样误差来自不同路径暴露和训练/验证分布，数值差异只支持“单点误差不能替代部署采样验收”，不足以确定唯一失败原因；不将其解释为全部流匹配方法无效。


补充同批源V路径诊断：固定R07原验证缓存low urban前128条、对应clean和物理ID，0 TX标签/F权重/F输出/target；只使用已冻结R20两行E20权重，本机CPU、不更新参数。两行共享一个CPU epsilon（seed202610202105），分别报告t=0/.25/.5/.75的含真值路径单点端点误差，以及实际自由运行16步的逐步先验误差和末端6CG输出误差。含真值路径不作为可部署预测；t>0会暴露clean端点信息，CPU噪声也不冒充原CUDA主draw。新增256次源IQ恢复仅为这128条源V子集的诊断，不增加独立物理样本数，不影响既定源资格与0target决定。

代码 `tools/diagnose_pamf_flow_path.py`，3项聚焦检查覆盖同批路径区分、真值观察不改变实际输出/随机状态及冻结状态和形状约束。输入仅clean/received/physical_ids的本机私有npz，权重为本机严格核实的R20目录；命令 `python -m tools.diagnose_pamf_flow_path --inputs <private_source128.npz> --weights-root <private_R20_weights> --output <new_diagnostic.json>`。分析只检验同批误差差异，不声称已确定唯一失败原因，输入、权重与原始路径日志不提交。
