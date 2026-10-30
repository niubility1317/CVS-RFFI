# 源域模块梯度诊断：未通过固定研究判据

VERIFIED / ANALYZED，范围SOURCE_ONLY。旧noVar E20源checkpoint、固定512个源训练池样本（128L384U）×原缓存01/20，32批B32、1024次G输入加2Ls实际checkpoint noquery。CPU2、0GPU、0训练/F/TX/V/target，不修改当前nativeIQhint健康进程。模型全state不变，p.grad没有累积。全部32批、384组梯度范数/dot/cos/ratio及汇总已独立复算一致，未挑选子批。

预先固定判据为：指定模块必须32个cos都可定义、至少24个为负。coarseclean为18/32，actionprobe为2/32，两项均未通过；不放宽规则，不据此自动启动梯度分离或额外预训练。

|模块与相对final IQ的方向|负cos批数|平均cos|梯度范数比中位数|
|---|---:|---:|---:|
|coarse/coarse_clean_IQ|18/32|-0.038594|12.068895|
|action/action_probe|2/32|0.364921|0.199550|
|coarse/prior_IQ|0/32|0.587973|0.151071|
|action/prior_IQ|0/32|0.786514|0.091549|
|prior/prior_IQ|0/32|0.626548|0.083904|

coarse直接clean监督与final IQ在该冻结点的平均cos略负（-0.038594），但不是稳定达到24/32的冲突。权重为1时其范数中位约为final IQ coarse梯度的12.069倍，说明直接加同权重可能显著改变优化压力；这不是新权重建议，也不能据此断言旧hint负结果的因果原因。action物理探针方向多数与final IQ一致，源证据不支持直接切断其全部联合梯度。

源损失复用一次真实G前向：finalIQ1、prior.1、STFT.05、actionprobe.05，以及只用于诊断的coarse对clean IQ1。coarse/action/prior按唯一参数完整不相交划分，未使用TX或另训识别头。不存在依赖路径的梯度为零，cos明确null，未以0替代null参与均值。

局限：这是旧源final20这一冻结点、两个源缓存上的batchmean参数梯度，不是完整训练轨迹、逐样本因果、真实身份准确率或目标泛化收益。该源训练池不是独立V，0F意味着没有身份保真证据。它只回答本次模块梯度分工假设，不能证明所有单阶段或多阶段方法优劣。

实际梯度循环28.349202秒。执行7f7cfd682bf732013fbc9820e9ad4d1d77c41413，checkpointcf40/source7-63-30/20轮4940更新/ancestor0严格核实。4聚焦检查与唯一P0P1通过，一次preflight/单archive比较/compile、真实2Ls smoke、实际PID1204211/CWD/argv/CUDA空/日志增长与自然结束均已核实。完整loss/geometry/成本见results.json，源positions仅留私有artifact。默认保持，目标ACTIVE。
