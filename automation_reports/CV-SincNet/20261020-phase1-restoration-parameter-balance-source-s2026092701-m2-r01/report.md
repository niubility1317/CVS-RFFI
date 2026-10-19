# 原F指导恢复的参数梯度范数上限

状态：RUNNING / VERIFIED。目标仍为完整目标low urban≥70%，其余五practical接近原F clean85.113889%（Agent暂采用≤2个百分点口径）；分别报告同物理clean的波形差距与冻结原默认头真实TX识别/净纠错。当前最好66.384127%，用户完整目标尚未达成。

源依据为既有scalar_sum训练E5/1235/630000IQ的累计identity/wave输出梯度RMS比15.426531，仅为源训练聚合统计，不是逐step/参数比或因果证明。网络Jacobian会改变任务相对梯度，因此输出空间限制不等于参数空间限制。本轮独立检验参数坐标及分组，保持两项损失、原数据/参数/预算，不因当前residual_flow新目标准确率调整方案；source_gradient_evidence.json复用已有聚合证据，0新模型前向/更新/目标读取。

|固定行|训练操作|执行|
|---|---|---|
|既有per_frame_cap|输出IQ坐标逐帧范数上限|复用既有gradient_balance/per_frame_cap完整结果，不训练/预测或初始化|
|parameter_global|全恢复器参数向量一个scale|新GPU4|
|parameter_parts|coarse/correction各一个scale|新GPU5|

两新行均从原合法R03（method b01b534f52daf9cae9e940ed6ec6f9243bccbdbe，coarse20/corr20）完整相同state独立开始，2140840参数，无新增NN，不继承失败精修、当前control或生成权重。wave仍为pairedIQ NMSE＋0.1logpower，identity为三原F各自final160cosine＋0.3×T²KL（T2）的均值。分别对两项mean loss求全模型参数梯度gw/gi；首次保留共享图，第二次完成后无Hessian/教师更新。依named_parameters固定coarse后correction顺序flatten，全局或分两组算s=min(1,sqrt(norm²(gw)/norm²(gi)))。identity norm²≤1e-20时s=1；wave0而非零identity时s=0。停止scale梯度，merged=gw+s×gi依原shape分回p.grad，无二次batch除法，再统一clip1与AdamW更新。

每个组的原始Euclidean合并方向与wave梯度内积非负可由范数上限推出；这是原始梯度方向性质，不是AdamW动量/预条件/weight decay后的参数步或最终波形/身份保证。本候选不是论文GradNorm/PCGrad，不作冲突投影或放大identity。记录实际calls/IQ、每组scaled/conflict更新步数、scale均值/范围、wave/identity/identity-after norm²及dot前后。统计单位是task mean-loss参数梯度/更新步，不能与旧逐IQ sum-loss输出梯度计数直接等同；零实际cap不作为资格或技术失败。

复用控制按operator结构固定为既有per_frame_cap（method9866f24dab7097294d585069bfcfe529d3346ade），不依据其目标分数选控制、不重新预测、不从其初始化。相同R03、数据/缓存、seed/顺序/路由、任务loss、原三F与AdamW/E10预算；新的两次全参数反传与旧输出坐标求导再一次模型反传计算工作不同，不称同walltime/FLOP。新两行仅改变参数分组范围，实际耗时/显存另报。控制如技术缺项则报告缺项，不声称完整匹配。

原63000 L/U、R02缓存epochs1..10、.15clean/.60residual/.25full、observed+clean等权、batch256/modelseed2026092701/dataseed2026101951保持。新各jointE10/2470、L63000/U567000优化曝光、1260000IQ；原coarse/corr各累计30轮、60组件epoch、串行继承50pass。AdamW3e-5/warm247/cos至3e-6、WD1e-4/clip1、FP32/noTF32。原F50 CROSS_RX三seed2026092701/02/03、method4df7b94d0fd8ce29056bd8236364e45afa92e07f和原默认头冻结参数/统计，0L/U TX真值、0F更新、0新头拟合。推断仍原网络单帧received IQ一次coarse/corrector，无运行F、梯度/cap、clean、真实信道/TX/RX/场景、其他帧/视图或更新。

source V原1800/原F1/七视图仅诊断，无早停/资格/NMSE gate/排序/状态更新。两新固定最终E10都自动完整168000物理目标clean+六practical/三原默认头，各自argmax不投票或按环境拼接。各行七视图IQ/ID/预测全部固定后独立无CUDA CPU才连接truth/pairedclean；报告逐F/TX/RX/day accuracy/MacroF1、修复/损坏/净修复、距clean差距、无oracle对齐full256/prefix232/tail24 NMSE/logFFTpower均值/中位数/P90、训练/预测成本和相对复用控制的差值。目标结果不回流参数/预算/顺序/选择性重跑，单恢复器seed与三读出不冒充三次独立恢复训练、新盲测、Phase2或在轨性能。

60项聚焦检查和唯一一次独立P0/P1审查PASS（审查者独立复跑60项）。launcher首步实际R03/三F/四Ls no-query smoke，完整初始IQ/state一致、有限非零merged coarse/correction与native IQ梯度、F无梯度、新checkpoint往返，0V/TX/target/optimizer更新，PASS立即训练。复用同数据契约与合法父来源，不重复数据验证/哈希/签名链。

一次N607普通账号预检VERIFIED，新run/release不存在、原R03合法，GPU4/5各24124MB空闲无PID；当前residual_flow在GPU6/7实际target PID/CWD/cmd LIVE并保留。每卡1本owner、allowner<2，重叠最多4本owner GPUworkers，source≥18GB、target≥12GB，CPU2线程/scorer最多1；新source评分先等residual_flow complete/failure，再与本run target共用score.lock，实际source退出才同卡target。停止规则：非法输入/权限/代码、输出冲突、非有限/OOM、预算/梯度执行或预测失败时保留并技术失败该行，无自动重试，健康行继续，不停止/修改他人任务。

一个实际代码配置commit、不可覆盖root、一次归档本地/远端SHA比较及remotecompile/import、真实smoke、PID/CWD/cmd/GPU/日志增长证据随后核实。命令、环境/CWD、矩阵、路径、资源、停止规则与产物见experiment.json。数据、权重、逐样本预测、日志及私有机器状态不进Git。另一用户对话未被发送指令或修改。


## 实际发布与首轮核验

方法commit `fd9c8fee5370f2b3ca7a2744862e99efb59968fd`已推送并独立核实远端OID，一次归档本地/远端SHA比较及远端compile/import均PASS。实际R03/三F/四Ls no-query smoke PASS：两臂完整初始IQ/state等于原R03，coarse/correction合并梯度及原头到IQ梯度有效，F无梯度，新checkpoint往返一致，0V/TX/target/optimizer更新。两臂实际2140840参数，无新增模块；PASS立即训练。

两条实际source worker的PID/CWD/完整命令/GPU4/5及持续日志增长已核实，三F参考缓存各63000条。实际参数组/更新步约束计数：

- parameter_global：1/10轮、247/2470步，参数梯度合并247次、IQ126000条。
  all：实际cap 247步、冲突28步，平均scale0.0546821691、范围0.0175209232至0.287018746，wave/identity/identityafter norm²总和14.168569/4694.36999/14.1685689。
- parameter_parts：1/10轮、247/2470步，参数梯度合并247次、IQ126000条。
  coarse：实际cap 247步、冲突22步，平均scale0.0564548573、范围0.0173563622至0.295902342，wave/identity/identityafter norm²总和14.7636229/4585.33089/14.7636229。
  correction：实际cap 247步、冲突25步，平均scale0.0371360714、范围0.0164287072至0.0674817935，wave/identity/identityafter norm²总和0.170577067/144.187353/0.170577069。

首轮核验时尚无新源V或目标准确率。实际缩放不代表性能改善或参数步波形保证；两臂固定E10后分别自动完整目标测试，source V不作资格筛选，用户目标仍未达成。最终状态与结果以完整结果节为准。前一流修正的GPU6/7目标进程保留，CPU评分等待其完成后执行；复用另一个已完成的per_frame_cap固定控制，不重复训练/预测。
