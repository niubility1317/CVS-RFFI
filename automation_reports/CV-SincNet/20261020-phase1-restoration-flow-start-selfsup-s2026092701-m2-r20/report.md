# 固定起点与随机起点：残差流匹配恢复 R20

状态：WAITING_R19，未发布。方法 commit `98c1822f88baa546db267dd97efe00f10870afd3` 已推送并独立核对远端 OID。28 项聚焦检查、compile/help、一次 P0/P1 修复及定向复核通过。R20 将等 R19 原控制器和全部所属 worker 退出后，才执行一次 N607 预检、发布和真实 checkpoint 无 query 入口；入口 PASS 立即继续。


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

|ID|来源|要求|目标文件|状态|验证|说明|
|---|---|---|---|---|---|---|
|A20-01|R5sigma消融|同速度场同初值/随机流，只改0/.02|models.py/design.py|verified|路径端点、目标、独立time/noise、完整state匹配通过|不同初始完整输出是预期|
|A20-02|R5/R6条件|received上下文多分辨率注入、16步完整采样后软MAP、一份IQ|models.py|verified|条件缓存、实际调用16、恒定速度解析式通过；N607真实入口待执行|非I2SB|
|A20-03|无监督与来源|全L/U、仅R07冻结父、scratch/0TX/F/target|training.py/runtime.py|pending|来源/schema负测、实际20/4940及曝光|单点训练诊断不作性能|
|A20-04|生成身份风险|固定128源3draw≤5%，无挑draw或标签条件|evaluation.py|pending|诊断完整性先于truth、零起点属性和随机起点实测|不替代逐TX|
|A20-05|源/目标边界|四源保护/固定排序、只有合格选定行target7|evaluation.py/runner.py|verified|未选target阻断、全部预测先truth、源成功收尾回归通过|目标不得回流|
|A20-06|执行保护|先等R19真实退出，单owner/容量/不可覆盖/真实入口|controller.py|pending|负测、一次P0P1、N607真实PID/CWD/argv/GPU/loggrowth|不热改健康R19|
|A20-07|完整交付|source7/条件target7、TX/RX/day/损伤/成本/全日志|run报告|pending|独立CPU评分、完整统计与实际seed|目标保持active|
|A20-08|真实性边界|新oracle/多seed/独立物理族/逐细节证明|后续确认|deferred|如实披露，非本轮发布前门槛|不称已完成总目标|

本地完成 28 项聚焦检查、编译和命令入口检查；一次独立 P0/P1 审查发现源摘要重复创建问题，已先复现 FileExistsError，再改为保留原 summary.json、另写 flow_source_summary.json，并将控制器选择输入同步到新摘要，定向复核 PASS。训练参数 1389250，总参数 3546797；条件编码 465920、速度 Core 917058、时间编码 6272。每训练样本调用一次速度 Core，推理为完整 16 次，不把两者的成本混报。当前 3 verified、4 pending、1 deferred，0 rejected/blocked；pending 对应实际训练、采样稳定性、远端入口与交付证据。


完整矩阵、命令、环境/CWD、路径、GPU 与停止规则见[登记](experiment.json)。
