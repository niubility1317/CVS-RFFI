# 冻结粗恢复后的恢复残差修正

状态：RUNNING / VERIFIED（源信道缓存生成）。方法commit `b01b534f52daf9cae9e940ed6ec6f9243bccbdbe`；11项新增与12项R02回归、compileall、controller help通过；一次独立P0/P1无阻断，N607预检通过。尚无本轮训练/效果结论。

本轮落实用户报告R6，三行矩阵、seed、权限、命令、成本、路径和停止规则唯一维护于[experiment.json](experiment.json)。逐项设计见[追踪](../../../docs/PAMF_RESIDUAL_REFINEMENT_R03_20261019.md)。旧run、数据、权重和日志全部保留。

固定粗恢复器是R02仅源波形A选出的复时频U-Net。其source V clean为98.7778%、low urban为79.5556%，mean six-residual NMSE为0.03197056，符合原源资格。实际父launch commit、源决策、E20/4940及0TX训练元数据已读回；完整原source物理角色和原F来源核对结论复用，不重验数据。父目标评分不是初始化/选择依据，当前固定面板已有暴露背景如实保留。

只训练恢复残差e=(x-b)/s，最终输出b+s*e_hat。b始终冻结、eval/no_grad、不进入优化器；s由第一轮固定8192条源L/U观测估计一次，共享为checkpoint常量，零V/标签/目标访问。推理仅输入当前r与固定噪声，b由r计算，不读取真实残差大小。恢复残差不称为真实硬件指纹。

确定性二次修正、残差扩散、残差流匹配各自E20，全部63000源物理样本逐轮一次；15%clean+60%residual+25%full，原生两路同latent但不串接，每轮新namespace。修正器从零开始，父恢复器E20及原F历史成本分列；累计恢复器20+20，另有一次8192条共享尺度估计，没有把整套系统称为E20。训练无TX标签、F输出/特征、CE、PL或联合F。

第一步实际F源样本no-query检查PASS后直接继续。16CPU先生成20轮新信道缓存，再一个GPU估计共享尺度，随后GPU0/1/5各一条训练/预测worker，计入其他owner cap2/free12GB。每行E20后冻结并自动接续两路source七视图预测/独立评分、生成稳定性/条件编码置换、同步batch1成本；不因低分停训。

A仅按波形误差在粗恢复与三修正中选择；B单独披露有标签source V辅助：相对粗恢复low urban+0.5个百分点、clean损失<=0.2个百分点、mean NMSE不恶化、clean NMSE<=0.02，生成式三draw切换<=5%。规则在目标前冻结，最多测试两条去重的新修正；A选粗恢复时复用父固定预测。所有行训练无TX，整体的源身份诊断/资格并不冒称无标签选模。

目标始终为父R01的原1200条固定输入，clean+六practical全视图预测固定后才独立truth-last。无target训练、改阈值、候选重排、best-of-N或选择性重跑；此阶段性配对面板不替代完整正式received IQ、多seed或独立物理族证明。技术异常保留产物、阻止未开始行，健康活动行继续；不自动重试，不修改其他owner任务。

## 发布与运行证据

实际方法commit `b01b534f52daf9cae9e940ed6ec6f9243bccbdbe` 已推送并独立读回分支OID。一次代码归档传输比较一致，远端compile/import通过；现有Python3.10.19/Torch2.1.0+cu121/NumPy2.2.5保持。独立PID/CWD/完整argv/父子关系读回匹配本run，16个数据worker运行。真实F两源样本检查PASS，TX/目标读取均为0。

控制器日志从237增长至764字节，已有1个完整epoch缓存；最新进度见私有观察证据。当前模型训练尚未开始，后续由同一controller自动完成尺度估计、三行GPU训练、源冻结与预登记目标确认。没有本轮有效性结论，不以发布或smoke冒称恢复成功。

本次继续监控原live handle，不因观察超时重启。已有权重/数据/旧输出和其他owner健康任务保持；原始机器路径、完整进程快照、原始日志和新缓存不提交。
