# DURA从零单阶段E20：恒等残差初始化

- run_id：`20261030-phase1-DURA-direct-identityinit-s2026092701-r01`
- group_id：`phase1-dura-single-stage-efficiency`；类别：`cvs`；阶段：`Phase1`
- 配置与矩阵：[experiment.json](experiment.json)；状态记录：[events.jsonl](events.jsonl)
- 当前登记状态：ANALYZED（实际状态按events.jsonl及独立证据更新）

## 目的与对照

SameE20/4940/peak1e-3aslivefastLR;only structuredidentityinitialization:randominnerfilters,zero residual/actionheads,FIRdelta0;0ancestor/teacher;automatic7formal

本候选依据代码与合法source物理约定固定：`CompactSuccessor`的checkpoint-loading skeleton沿用FIR初始脉冲24，并用随机Linear替换原先清零的全局头；其余输出残差头也随机。已有DURA权重加载会覆盖它们，从零训练则直接从这套非恒等初值开始。该发现不证明此前全部性能差距由初始化造成，但给出单阶段优化的可证伪方向。

与正在运行的fast LR控制保持相同E20/4940、峰值1e-3/末值2e-5/warm247、全source角色与顺序、model/probe seed、目标权重、batch/optimizer/clip、266951参数及两prior/四CG。唯一机制差异是结构化残差初始化：内部滤波器保持同seed随机，coarse.temporal.out、action点头/全局头、两prior.out清零，FIR初始脉冲0。初始coarse=y、A=I、prior=y，训练后不强制恒等；所有组件一起训练，0预训练权重/学习教师/TX/F/V/target训练。

两项聚焦检查通过：真实Action/双CG的toy端到端恒等及源联合目标有限非零梯度；run/worker/score与checkpoint配置正确传播新recipe。一次独立P0/P1审查PASS。launcher首步另用实际source两条Ls检查完整图初始clean恒等容差2e-6、received梯度、严格checkpoint存取和实际冻结F50无query smoke，PASS后直接训练，无额外审批artifact。

此矩阵在fast LR正式结果尚未读取时冻结，不改变当前健康训练，不用当前或将来target分数选择父/调参。每版自动source1800×7与target168000×7、独立truth-last和同卡G/G+F成本；目标仍为六星地各自距rawclean≤3pp并保持clean，不以初始恒等或更快source下降代替正式收益。

## 实际发布与训练

VERIFIED：执行commit `d62107285bb0ae3fe952a4d0954de2cc16c8bab9`，release `20261030-DURA-direct-identityinit-r01`，archive传输SHA一致/远端编译通过。实际完整图的2Ls初始恒等检查、梯度、checkpoint严格存取和F50 noquery smoke均PASS，之后按原资源规则自动启动。CPUcontroller3596624/worker3597618、GPU2、实际CWD/cmdline和日志增长已独立读回。

已核实E7/1729、Ls44100/Us396900曝光，source IQNMSE0.0410731，0TX/F/学习教师/初始化checkpoint读取。该读回仅证明训练健康；本run正式性能尚未形成，默认DURA保持不变。

## 数据、seed与模型来源

实际数据契约、权限例外、完整seed角色、checkpoint来源和选择规则见experiment.json。
逐行配置通过config_ref/resolved_config_ref定位；待补项必须在对应生命周期补齐。

## 执行与存储

命令、环境、CWD、commit、launch owner、输出和日志路径见experiment.json。
实际PID/GPU、读取时间、remote readback、失败或替代关系在此追加，并用record命令记录证据指针。

## 结果与覆盖

尚无结果。按预登记artifact逐项记录路径和缺项；保留每row与RX/day/TX/scene/K/seed的对应关系。
源域训练完成、预测完成、评分完成及协议有效性分别陈述。不得用总索引或旧状态证明当前运行。

## 交接

记录已完成、当前run/commit、证据路径、阻塞与下一步；恢复先查原run，不重复启动。

## 完整单阶段训练和七景结果

VERIFIED/ANALYZED。20条完整epoch、4940更新，累计L126000/U1134000曝光；祖先checkpoint、学习教师、TX/F/V/target训练读取均为0。训练763.331秒（12.722分钟），266951参数，训练峰值allocated 1034.42MiB。完整stdout无traceback/OOM；source1800×7和target168000×7全部完成，controller/worker自然退出。执行commit d62107285bb0ae3fe952a4d0954de2cc16c8bab9。

全七prediction固定后独立CPU truth-last，再次独立复算126个总体/TX/RX/day分组一致。原F50/defaulthead单seed，不扩展为当前Tri-SIG-TIME SSL或Phase2/3证据。

|场景|准确率%|相对普通初始化pp|相对默认DURApp|距rawclean pp|
|---|---:|---:|---:|---:|
|clean|84.620833|-0.044643|-0.057143|0.055357|
|practical_high|81.629167|+1.634524|-1.195833|3.047024|
|practical_mid|80.130357|+1.220238|-1.430357|4.545833|
|practical_low_suburban|78.541071|+1.083333|-1.517262|6.135119|
|practical_high_urban|80.323810|+1.570833|-1.422024|4.352381|
|practical_mid_urban|73.293452|+0.987500|-2.192857|11.382738|
|practical_low_urban|66.442857|+0.801190|-2.447024|18.233333|

六个星地场景均高于普通随机初始化fastLR，lowurban66.442857%，但六景均低于默认DURA，六景3pp目标=False。原R17逐景最多下降1pp容忍=True，它不替代本次优化目标。clean较raw少93条正确（−0.055357pp），较默认少96条。

最终source IQNMSE 0.02773173略高于普通初始化0.02702048，而星地准确率更高；源IQ均值不能充分代理身份保真，不能将这一单seed现象解释为因果机制。固定最后E20，不选中间epoch。匹配同卡B1/B128 G/G+F三轮交替时延/P95/吞吐见results.json；共享负载不用于跨卡历史速度比。所有权重和原始日志保留原root/local_artifacts，不提交Git。默认DURA保持，目标ACTIVE。
