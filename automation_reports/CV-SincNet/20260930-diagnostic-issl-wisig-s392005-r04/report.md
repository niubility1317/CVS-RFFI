# ISSL修复版新类学习固定预算验证

- run_id：`20260930-diagnostic-issl-wisig-s392005-r04`
- group_id：`newclass-registration-issl-repair`；类别：`diagnostic`；阶段：`external_execution_acceptance`
- 配置与矩阵：[experiment.json](experiment.json)；状态记录：[events.jsonl](events.jsonl)
- 当前登记状态：ANALYZED（2026-09-30；真实产物/全部日志读回VERIFIED）

## 目的与对照

修复KD学生断梯度、SSL梯度累积、queue转置/初始化归一化/容量、冻结teacher及EMA BN状态；保留作者架构/损失权重/温度/LR/阶段流程

## 数据、seed与模型来源

实际数据契约、权限例外、完整seed角色、checkpoint来源和选择规则见experiment.json。
逐行配置通过config_ref/resolved_config_ref定位；待补项必须在对应生命周期补齐。

## 执行与存储

命令、环境、CWD、commit、launch owner、输出和日志路径见experiment.json。
实际PID/GPU、读取时间、remote readback、失败或替代关系在此追加，并用record命令记录证据指针。

## 结果与覆盖

完整结果与修复机制见下方；固定预算、缺项和来源边界保持明确。
源域训练完成、预测完成、评分完成及协议有效性分别陈述。不得用总索引或旧状态证明当前运行。

## 交接

记录已完成、当前run/commit、证据路径、阻塞与下一步；恢复先查原run，不重复启动。

## 修复与预登记科学判定

修复版保留作者网络、logits对比空间、随机裁剪、旧样本重复、Adam/LR、损失权重与温度、base→SSL→transfer及独立incremental流程。KD保留原温度CE标量且恢复学生计算图，未额外乘T²。teacher冻结并eval；key在no_grad下前向，以EMA同步参数与浮点BN状态、复制整数计数。所有queue向量统一归一化、按transpose计算点积并精确保留最后K条。

预算：base=50、SSL=20、transfer=100、独立incremental=100epoch。两个run在任何新query评分前固定，从零训练，r04不加载r03状态。每GPU最多两个训练进程，本次串行；/root唯一launch owner。

执行判定：loss/梯度有限，KD学生梯度实际非零、key/teacher无反向梯度，queue规范、teacher状态不变、新head行更新；四个checkpoint恢复后预测逐条一致。学习效果同时报告训练旧/新类准确率与独立query旧/新类准确率/H/Macro-F1，不把短预算低分当系统故障或用query调参。B及其衍生指标N/A。原数据/全文缺项不阻断用户授权的WiSig修复诊断，也不据此宣称原论文完整复现。

## 修复后真实WiSig验证结果（VERIFIED）

实现`issl_fixed_v1`，代码commit `014dc059ad24f9cd2d6f280c46e821f06f377d91`；旧3类、新3类，K=128，query每类64，RX1-1/2021_03_01。实际预算base=50、SSL=20、transfer=100、独立incremental=100epoch。完全scratch；新teacher/key仅继承本run，未加载历史checkpoint。

|阶段/分支|旧类准确率|新类准确率|H|
|---|---|---|---|
|A：适应前旧类|100.0000%|N/A|N/A|
|B：仅旧类support适应|N/A|N/A|N/A|
|C：SSL后监督transfer，全注册类竞争|100.0000%|100.0000%|100.0000%|
|独立incremental对照C|100.0000%|75.0000%|85.7143%|

主分支Macro-F1=1.000000，独立增量Macro-F1=0.874510。旧类适应提升及B→C注册后下降N/A，因为作者流程没有独立B；主分支最终新旧差0.0000个百分点。A→C旧类净变化+0.0000个百分点，包含SSL与联合监督训练，不混称仅注册影响。K×新增类数仅此预登记row（K128×新增3），未扩展其他K/新增数。

|训练阶段最后一轮|训练旧类准确率|训练新类准确率|loss|
|---|---|---|---|
|base|100.0000%|N/A|0.000014|
|downstream_transfer|100.0000%|100.0000%|0.000007|
|incremental_baseline|100.0000%|77.3438%|0.207293|

完整训练机制核验：SSL的360步中360步KD学生梯度非零；新增head实际更新。独立incremental的600步中600步KD学生梯度非零。SSL每步清梯度，负队列归一化并转置，最大长度1000；teacher全程冻结且状态未改变。SSL loss范围2.903054至3.843369，梯度范围0.009704至0.069893。

实际1710训练步，所有loss/梯度有限；完整读取1980行逐步/摘要JSONL、对应CSV及1982行stdout，紧凑epoch摘要270行。384个唯一query物理ID预测完整；4个checkpoint strict重载，A/C/独立增量C的全部预测与固定artifact逐条一致。训练IQ上的真实checkpoint smoke通过且未访问query。

RTX5070Ti、torch2.10.0+cu128、ssr-gpu；方法端到端69.6554秒，Torch峰值allocated=300798976字节（286.86MiB），模型参数及可训练参数均3847424。阶段训练/训练集评估耗时见epoch摘要；独立推理耗时、主机峰值内存、完整常驻状态、新增传输字节数未单独测量，记N/A，不声明星载省算力。首次成功进程读回时训练已结束，PID/argv由启动artifact、CWD/GPU由实际resolved_config及CUDA峰值核对，不冒称取得了运行中的nvidia进程快照。

主分支训练旧/新类与query旧/新类均已有效学习，独立增量新类query为75%。该固定预算诊断消除了此前新类始终0的现象。与r02/r03相比同时包含训练预算差异，不能把提升全部归因于KD或某一项修复，也不能证明该数据切片以外的泛化。

结论边界：真实同接收机/同日6类WiSig无LEO诊断，不是正式CVS性能比较；未确认跨接收机、跨日或LEO条件。ISSL全文/原始数据仍缺，修复的是作者实现中可定位的问题，不宣称原论文所有公式或原始数据准确率复现。评分只在固定prediction后执行，没有反馈调参、选模或追加重跑。
