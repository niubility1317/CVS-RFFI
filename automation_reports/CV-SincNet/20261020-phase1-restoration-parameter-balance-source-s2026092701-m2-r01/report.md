# 原F指导恢复的参数梯度范数上限

状态：ANALYZED / VERIFIED。目标仍为完整目标low urban≥70%，其余五practical接近原F clean85.113889%（Agent暂采用≤2个百分点口径）；分别报告同物理clean的波形差距与冻结原默认头真实TX识别/净纠错。当前最好66.384127%，用户完整目标尚未达成。

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


## 完整固定目标比较

两个登记行均完成E10/2470、source V诊断、168000物理目标样本的七视图预测和独立CPU truth-last评分。实际source/target/controller进程均已退出。没有用源分数筛掉一组，也没有根据目标成绩更改/重排/选择性重跑。以下是三套原F各自argmax的均值±样本标准差，三套原读出不是三次独立恢复器训练。原clean参考均值85.113888889%；历史完整特征约束候选low urban为65.309126984%，仅作已暴露基准参照，不作为初始化。

|目标环境|原R03|全参数范数上限|分组件参数范数上限|
|---|---:|---:|---:|
|clean|84.743056%|84.942659±0.476678%|84.934127±0.477819%|
|high|77.575000%|77.552183±0.233296%|77.564484±0.251935%|
|mid|76.491071%|76.649802±0.186273%|76.638492±0.194466%|
|low suburban|75.284127%|75.581349±0.124183%|75.576389±0.097939%|
|high urban|76.330952%|76.555754±0.158617%|76.562500±0.168831%|
|mid urban|70.116865%|71.388492±0.194048%|71.373214±0.198906%|
|low urban|63.392460%|65.683532±0.364106%|65.657937±0.353954%|

|识别判据|全参数范数上限|分组件参数范数上限|
|---|---:|---:|
|low urban≥70%|未达到|未达到|
|其余五环境距原clean≤2个百分点|未达到|未达到|

“接近”采用本对话Agent预先记录的≤2个百分点口径，不冒充用户指定数值。未通过者保持负结果，未因实现/训练完成或波形误差而称用户目标实现。

|逐原F的low urban|全参数范数上限|分组件参数范数上限|
|---|---:|---:|
|2026092701|66.098810%|66.062500%|
|2026092702|65.419048%|65.405357%|
|2026092703|65.532738%|65.505952%|

|距原clean差距（百分点）|全参数范数上限|分组件参数范数上限|
|---|---:|---:|
|clean|0.171230|0.179762|
|high|7.561706|7.549405|
|mid|8.464087|8.475397|
|low suburban|9.532540|9.537500|
|high urban|8.558135|8.551389|
|mid urban|13.725397|13.740675|
|low urban|19.430357|19.455952|

## 同物理clean的波形差距与净纠错

配对clean仅在七视图预测全部固定后的独立CPU评分读取，不做oracle增益/相位/时延对齐。逐环境full256/prefix232/tail24 NMSE和logFFTpower均值/中位数/P90、逐F/TX/RX/day与混淆矩阵完整保存在results.json。

|目标low urban|原received|原R03|全参数范数上限|分组件参数范数上限|
|---|---:|---:|---:|---:|
|full_nmse均值|1.987855210|0.076996611|0.076072135|0.075999371|
|prefix232_nmse均值|1.998446002|0.068349912|0.067724651|0.067671114|
|tail24_nmse均值|1.928131431|0.157928481|0.154216374|0.153932314|
|log_power_mse均值|0.160772502|0.023813475|0.023465681|0.023452704|

|low urban修复/损坏/净修复|相对received|相对原R03|
|---|---:|---:|
|全参数范数上限|188015/26551/161464|33995/22448/11547|
|分组件参数范数上限|187995/26660/161335|33690/22272/11418|

计数为三原头的决策合计，504000个重复视图决策只有168000个物理样本，不称504000个独立样本。

## 源诊断、实际执行和成本

|源V（固定F1）|clean|low urban|mid urban|
|---|---:|---:|---:|
|全参数范数上限|98.500000%|85.166667%|92.555556%|
|分组件参数范数上限|98.555556%|85.222222%|92.722222%|

源V只诊断，无候选资格/排序/早停/参数或统计更新。两臂各63000训练物理L/U、E10/2470、L63000/U567000优化曝光、1260000IQ，原coarse/correction各累计30轮、60原组件epoch、串行继承50pass；梯度合并安排同10轮，无新模块。原F和默认头冻结，0新头拟合、0L/U TX真值、0目标训练更新。

- 全参数范数上限：2140840参数无新增模块，参数梯度合并2470次、IQ1260000条；训练/保存1534.726s，含准备/等待1552.226s，峰值分配5682035712bytes；七视图目标预测799.692s。
  all：cap 2470步、冲突259步，平均scale0.0611075967、范围0.0147907324至0.790292203；wave/identity/identityafter norm²总和139.93297/30777.4591/139.93297，dot前/后387.959915/25.9016437。
- 分组件参数范数上限：2140840参数无新增模块，参数梯度合并2470次、IQ1260000条；训练/保存1527.144s，含准备/等待1544.264s，峰值分配5682039808bytes；七视图目标预测791.157s。
  coarse：cap 2470步、冲突268步，平均scale0.0606182251、范围0.0118274605至0.733421206；wave/identity/identityafter norm²总和140.840363/30192.1645/140.840367，dot前/后389.880736/27.5864088。
  correction：cap 2470步、冲突155步，平均scale0.0396401065、范围0.00939090829至0.113102123；wave/identity/identityafter norm²总和1.82433576/1391.56893/1.82433576，dot前/后5.23494867/0.199896354。

上述为task mean-loss参数梯度/更新步统计，不是逐IQ sum-loss输出梯度，scale均值不与旧输出cap直接等同。两新行同样两次全参数反传，仅分组范围不同；与旧输出cap工作量不同，成本如实报告。原始Euclidean wave方向非负不保证AdamW动量/预条件/decay之后参数步或最终波形/身份效果。全部source/target日志已完整解析，无技术错误。方法commit`fd9c8fee5370f2b3ca7a2744862e99efb59968fd`；60项聚焦检查、唯一独立P0/P1审查、实际R03/三F/四Ls及新checkpoint no-query smoke、归档比较/编译、实际PID/CWD/cmd/GPU/loggrowth证据保留。未给另一用户对话发消息。

本轮仍未达到用户完整目标。参数梯度范数上限不等于论文GradNorm/PCGrad，也不证明最终波形或身份保真保证，不能把单恢复器种子或已暴露基准当作三seed恢复结论、新独立盲测、Phase2部署或在轨性能。


## 预登记复用的固定输出空间控制

控制run `20261020-phase1-restoration-gradient-balance-source-s2026092701-m2-r01`/`per_frame_cap`在参数约束试验发布前已独立固定，不依据本试验或目标分数选择；不从其初始化，不重训或重预测。比较状态：VERIFIED_REUSED。

控制与两新行的逐轮epoch/update/路由/L-U曝光/IQ/LR相同，168000/七视图/三原F及truth-last完成，原received/R03读出逐环境一致。方法与父权重/缓存相同已在预登记固定。

|环境|复用输出空间控制|全参数上限相对控制（百分点）|分组件上限相对控制（百分点）|
|---|---:|---:|---:|
|clean|85.061111%|-0.118452|-0.126984|
|high|77.926984%|-0.374802|-0.362500|
|mid|76.995238%|-0.345437|-0.356746|
|low suburban|75.948016%|-0.366667|-0.371627|
|high urban|76.960119%|-0.404365|-0.397619|
|mid urban|72.014881%|-0.626389|-0.641667|
|low urban|66.384127%|-0.700595|-0.726190|
