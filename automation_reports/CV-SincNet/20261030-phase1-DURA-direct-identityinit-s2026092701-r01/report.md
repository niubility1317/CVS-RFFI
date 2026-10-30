# DURA从零单阶段E20：恒等残差初始化

- run_id：`20261030-phase1-DURA-direct-identityinit-s2026092701-r01`
- group_id：`phase1-dura-single-stage-efficiency`；类别：`cvs`；阶段：`Phase1`
- 配置与矩阵：[experiment.json](experiment.json)；状态记录：[events.jsonl](events.jsonl)
- 当前登记状态：PLANNED（实际状态按events.jsonl及独立证据更新）

## 目的与对照

SameE20/4940/peak1e-3aslivefastLR;only structuredidentityinitialization:randominnerfilters,zero residual/actionheads,FIRdelta0;0ancestor/teacher;automatic7formal

本候选依据代码与合法source物理约定固定：`CompactSuccessor`的checkpoint-loading skeleton沿用FIR初始脉冲24，并用随机Linear替换原先清零的全局头；其余输出残差头也随机。已有DURA权重加载会覆盖它们，从零训练则直接从这套非恒等初值开始。该发现不证明此前全部性能差距由初始化造成，但给出单阶段优化的可证伪方向。

与正在运行的fast LR控制保持相同E20/4940、峰值1e-3/末值2e-5/warm247、全source角色与顺序、model/probe seed、目标权重、batch/optimizer/clip、266951参数及两prior/四CG。唯一机制差异是结构化残差初始化：内部滤波器保持同seed随机，coarse.temporal.out、action点头/全局头、两prior.out清零，FIR初始脉冲0。初始coarse=y、A=I、prior=y，训练后不强制恒等；所有组件一起训练，0预训练权重/学习教师/TX/F/V/target训练。

两项聚焦检查通过：真实Action/双CG的toy端到端恒等及源联合目标有限非零梯度；run/worker/score与checkpoint配置正确传播新recipe。一次独立P0/P1审查PASS。launcher首步另用实际source两条Ls检查完整图初始clean恒等容差2e-6、received梯度、严格checkpoint存取和实际冻结F50无query smoke，PASS后直接训练，无额外审批artifact。

此矩阵在fast LR正式结果尚未读取时冻结，不改变当前健康训练，不用当前或将来target分数选择父/调参。每版自动source1800×7与target168000×7、独立truth-last和同卡G/G+F成本；目标仍为六星地各自距rawclean≤3pp并保持clean，不以初始恒等或更快source下降代替正式收益。

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
