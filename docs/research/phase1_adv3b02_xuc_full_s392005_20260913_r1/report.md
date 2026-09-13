# FULL9完整扩展矩阵与发布验收

2026-09-13；run：`phase1_adv3b02_xuc_full_s392005_20260913_r1`。
当前状态：本地实现及聚焦验证完成，远端CUDA验收及正式发布待执行；此状态不代表训练已启动。

## 授权与变更原因

用户明确选择“扩展分支也全部加入，重新设计矩阵”。本次启用原生A1曾关闭的DAOT tangent/nuisance/fingerprint/relation/prototype及RC4 negative/anchor/卫星Hard监督，并保留已指定七行的X/U/C区别。三个互斥teacher模式不同时叠加，采用three_view与robust_deployment；tangent采用branch_selective。全开指这些损失路径可按定义参与目标，不是把所有互斥算法或条件门槛同时强制通过。

旧DR7保留为历史探索结果。已发现其U学生forward错误使用GRL=1，本次统一修复为原生RC4的GRL=0；旧结果不能继续称为原生A1完整等价实现。旧49步预算也改为与完整U轮次一致的222步。新旧同时改变预算、GRL和扩展，不能用新旧差值单独归因任何一个修复。

## 九组矩阵

|新行|DAOT/RC4|X|Ux_normalized|C控制|课程|
|---|---|---|---|---|---|
|F-M00|无，CORE90参考|关|关|关|固定|
|F-A1|原生A1启用项，修正GRL|关|关|关|固定|
|F-M14|完整扩展|开|开|被动审计，无控制动作|曝光匹配能力课程|
|F-M11|完整扩展|关|关|原C2|原能力课程|
|F-M05|完整扩展|开|开|被动审计，无控制动作|固定|
|F-M08|完整扩展|开|开|原C2|原能力课程|
|F-M07|完整扩展|关|开|原C2|原能力课程|
|F-M12|完整扩展|开|开|可靠性C*|曝光匹配能力课程|
|F-M13|完整扩展|开|开|可靠性C*|固定|

X权重0.05，Ux_normalized权重0.01，均保留原clean四个P4Q4K2块定义。F-M08/F-M12/F-M13都是完整联合方案，分别比较原C2、可靠性C*加课程、可靠性C*固定课程。C2与C*为控制器替代，不在同一行同时发出冲突动作。F-M14/F-M05按原矩阵定义关闭控制动作。

F-M00明确为匹配222步及FastTrust学习率的CORE90参考；普通U仍从E131启用，U批量256，因此不称为旧M00的逐项复刻。F-A1是CORE90承载网络上的A1目标参考，不冒充原生M09训练器。两组参考不宣称全部扩展开启。保留历史M00/M09/M10用于描述性对照，不重复发布其相同配置。

主要对比：F-M08对F-M07识别给定U+C2上的X差异；F-M07对F-M11识别给定C2上的U差异；F-M12对F-M14识别给定曝光课程上的C*；F-M13对F-M05识别固定课程上的C*；F-M12对F-M13识别给定C*上的课程；F-M14对F-M05识别被动审计条件下课程。F-A1/F-M00比较承载网络上的原生A1目标整体，完整扩展行与F-A1的差异包含X/U/C，不能解释为单独扩展的因果效应。九组不是全因子矩阵，没有单独完整DR且无X/U/C的第十组。

## 参数、机制与继承来源

DAOT七项基础权重：orbit_z=0.5、orbit_logit=0.2、orbit_proto=0.2、orbit_relation=0.05、tangent=0.05、nuisance=0.1、fingerprint=0.1；teacher三view，robust_deployment聚合；tangent_sample_ratio=0.25。清空daot_ablation后再应用原生校验，防止A1消融配置把扩展重新置零。

原生L侧执行七项，U侧执行orbit四项；未把L-only分支虚报为U侧执行。原型来自本run已接受的L更新，E21前要求六类原型齐备，每次目标评估前快照detach。teacher为本run从零训练的EMA。anchor在完成E20后、进入E21前冻结本run EMA，保存source_anchor_E20.pth与anchor_provenance.json；无外部初始化/resume/teacher checkpoint。源V只读校准，不反传。

RC4：H/P-set/P-conditional/N/domain/self/anchor/satellite全部启用。H=0.6、P=0.4、N=0.2、domain=0.16、self=0.1、satellite=0.1、feature_anchor=0.05。H/P/N独立有效预算各0.1。原生total_identity预算模式仅允许H/P并禁止N，因此选择独立预算模式，total_identity_effective_budget=0；这不是关闭N。P-conditional尾段终值0.2，防止E200又变为零权重。其他阈值按原生源V校准，Hard精度目标0.98，P覆盖目标0.95，P精度目标0.98，N误排除目标0.01。卫星Hard样本从实际H路由选出，生成真实信道IQ后进入学生forward。

每个主更新先冻结teacher输出、RC4路由、卫星视图、原型和scale起点。EG两场均计算完整目标，scale与EMA只在接受主更新后提交一次，探测场不污染持久状态。U学生GRL=0保留原生分离语义，不能通过额外身份干路对抗梯度解释损失已启用。

## 数据、预算与调度

ManySig.pkl、equalized=1；源RX=1/3/4/6/8，源day=1/2/3；目标RX=0/2/5/7/9/10/11，目标day=0/1/2/3；六个注册TX。沿用既有物理角色契约：L=6300、U=56700、V=27000，比例0.07/0.63/0.30。U标签与TX元数据隐藏，训练不读取目标IQ/truth。复用已验证源角色和目标input包，不重新构造数据或重复校验。

全部seed=392005、scratch、FP32、E200；每轮222次主更新，L128/U256，末U批124条，完整U一轮恰好56700条；每行44400次接受更新。L使用完整128批循环或原grid采样，预算与成员曝光记录绑定ticket。DR八行从E1处理U；CORE90参考普通U从E131。AdamW lr=0.0002、weight_decay=0.0001、clip=5、EMA=0.999。FastTrust学习率按origin ticket epoch：E1为4e-5，E5达2e-4；原生余弦到E160的2e-5；E161骨干4e-6，E181骨干1e-6，其他参数尾段2e-5。

E1–20：主干、RC4 domain/self训练，身份路由及DAOT保持原生warmup。E21：同run anchor冻结，orbit/nuisance/fingerprint及RC4身份项进入日程；orbit从E21–60渐增。E61：tangent进入日程，并按原生E61–140渐增。源V校准E1/21/41/91/161。关键边界拆ticket窗口，不能把未完成E20的EMA当锚点。每GPU最多两个训练实验，计入已启动未初始化的训练/预测进程，空闲显存至少12000MiB；保护其他owner。此进程扫描不是跨owner原子锁，发布前必须核实其他owner调度状态。

全部九行E200冻结后进行clean及三LEO场景独立推理：每场景168000条，每行672000条，总6048000条预测固定后独立scorer连接truth。测试按最终E200，不依据目标指标选checkpoint、调参或决定重跑。当前目标集是既有研究benchmark，不称作新盲测确认。

## 验收证据与边界

本地23项pytest通过，覆盖旧目标保持、EG、数据隐藏、课程与预算、完整U覆盖、九行父方法配置、fingerprint条件梯度及激活状态。F-M13及F-M00各通过真实train_xuc入口的1步合成检查；这只是入口诊断，不算正式epoch。

full_fields_03通过E1/21/61/131/181完整合成目标及两场EG检查。受控H/P/N各8条时，H/P-set/P-conditional/N/satellite/anchor均有非零梯度；orbit四项、nuisance及nuisance_head、E61后的tangent亦有非零梯度。受控mask仅在check脚本，不进入正式训练。该场景fingerprint恒为0；额外原生bundle测试在固定minimum=0.5下分别证明未满足时非零梯度、满足时归零。不能因此宣称正式数据上的fingerprint已经有非零梯度。

正式七行每轮落盘activation_status.json，分别报告配置、预定阶段、实际H/P/N数量、非零损失、梯度探针和C*真实动作。梯度探针每1000步一次，损失与路由每步记录；零梯度不自动等于断线，非零总损失不能代表每个分支有效。C*无可靠证据时不发CORRECT/CATCHUP，不能放宽门槛制造激活。条件持续不触发需报告覆盖事实，不能宣称全部实际触发。

独立P0/P1审查：新目标函数与增量发布入口无未关闭阻断项；异矩阵恢复已拒绝，失败行从零新run重试，健康E200仅复用于推理，不覆盖旧产物。SSH超时先核对实际进程及产物再决定后续，不能重复发布。

发布前2026-09-13 19:36香港时间实查：旧XUC15与DR7共22行全部SCORED，两个owner均已退出；其他seedscan有6个活跃任务、pending为空、每GPU单进程准入。GPU0空闲，其余每卡一个GPU进程；GPU4的其他任务保持运行；项目盘空闲约7.7TB。详细见remote_preflight.json。

现有每小时监控xuc15为PAUSED；本次保留暂停状态，发布后更新其追踪对象。不能报告监控已经自动运行。

预计时间暂不按旧49步训练速度外推精确数值；本次更新数约4.53倍且新增多view、tangent及anchor计算，后续应使用新run阶段实测吞吐分别估计E21/E61后的余时，并包含全部预测与评分时间。
