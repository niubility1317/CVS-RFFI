# DURA 解析 RF 保真损失单阶段恢复对照

- run_id：`20261030-phase1-DURA-source-analyticRF-s2026092701-r01`
- group_id：`phase1-dura-source-analytic-RF-auxiliary`；类别：`cvs`；阶段：`Phase1`
- 配置与矩阵：[experiment.json](experiment.json)；状态记录：[events.jsonl](events.jsonl)
- 当前登记状态：ANALYZED（实际状态按events.jsonl及独立证据更新）

## 目的与对照

同noVar scratchE20/4940，唯一source梯度权重.009527944261805803×五解析RF误差；复用一次G输出，自动全七景truth-last。

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


## 最小研究说明

固定source梯度可行性诊断得到的唯一alpha=.009527944261805803，在当前noVar四项目标上增加五支解析RF归一误差之和。保留从零单阶段E20/4940、原20缓存、15%clean、训练seed/预算/LR。复用一次G恢复输出；每批clean解析变换no_grad，restored变换可反传；没有训练好F权重/特征/分类头进入训练，不增加部署支路或另训头。此前R01用过解析RF损失且失败，本次不继承其有效性。

原128Ls/384Us双视图的输出梯度资格均已通过并独立复算，冻结权重不再做网格或目标调参。source输出梯度可行性不等于G参数方向或TX/目标有效性。实际G仍只使用合法sourceclean配对和模拟action；TX/V/target训练读取0。sourceF研究背景如实保留，不声称整个研发从未使用F信息。

GPU仅4/6/7择一新增1个本任务进程，总computePID最多3、剩余至少12000MiB；不干预其他任务。首步真实模型2Ls smoke、原默认头和解析冻结检查，通过即继续。最终E20自动profile、source1800×7及target168000×7，逐角色全部预测完成后独立truth-last评分；源域负结果不能取消新版本正式测试。逐场景3pp、clean保持、R17单景1pp容忍、净纠错和完整训练/部署成本均报告。默认DURA保持，当前目标未完成。


## 启动核实

VERIFIED / RUNNING。执行commit `0a3b0767dc5c76824a7ec94aa2cda600b5c77a74`，8项聚焦检查与本候选唯一独立P0/P1审查PASS；一次N607预检、单归档两端SHA、远端compile通过。控制器PID903741，首步真实2Ls G梯度/严格重载/原F默认头smoke及解析提取器冻结检查PASS，随后直接进入训练。

独立读取实际worker PID904106，GPU6；真实CWD、完整argv和CUDA_VISIBLE_DEVICES匹配发布路径/commit，worker日志2377字节并继续增长。容量限制由固定dispatcher在准入时核实：加本任务前可见PID<3、空闲至少12000MiB。启动后外部Phase2预测进程新增，当前同卡包括本任务共4个computePID，不能宣称持续总数<=3。独立/proc起始时刻证实其中另一个预测worker比本任务晚启动1.72秒；其他后续预测也在本任务之后启动。没有提高本任务准入上限或追加本任务worker，不热停健康任务；共享负载影响如实保留。其余owner未停止、修改或重启。本run只有一个训练row，固定E20后自动全七景，不将启动状态当成实验完成。


## 完整训练与正式七景结果

VERIFIED / ANALYZED。完整20条epoch/4940更新，L126000/U1134000曝光，从零单阶段、0恢复器祖先/训练好F指导/神经教师/直接TX/V/target训练。五解析变换保持冻结，2520000次解析输入对应每访clean与restored各一次，单次G恢复输出复用。完整20轮分支误差之和及固定alpha加权总loss逐项复算一致，方差辅助仍仅记录、不参与目标。执行commit `0a3b0767dc5c76824a7ec94aa2cda600b5c77a74`。

source1800×7和target168000×7全部固定后各自独立CPU truth-last，再次独立复算126个总体/TX/RX/day、MacroF1及净纠错计数一致。完整日志无Traceback/OOM/数值失败，控制器与worker自然退出。已暴露基准、单seed，不声称新的独立盲测或稳定复现。

|场景|准确率%|相对noVar E20 pp|相对默认DURA pp|距raw clean pp|净纠错（相对raw同景）|
|---|---:|---:|---:|---:|---:|
|clean|84.661310|-0.080952|-0.016667|0.014881|-25|
|practical_high|79.604762|-1.299405|-3.220238|5.071429|+65305|
|practical_mid|78.569643|-0.836905|-2.991071|6.106548|+64937|
|practical_low_suburban|77.379762|-0.489286|-2.678571|7.296429|+64428|
|practical_high_urban|78.520833|-1.477976|-3.225000|6.155357|+65222|
|practical_mid_urban|72.402381|-1.990476|-3.083929|12.273810|+63704|
|practical_low_urban|66.466071|-2.004762|-2.423810|18.210119|+60760|

六景3pp验收=False，clean不低于raw=False，原R17逐景最多下降1pp容忍=False。默认DURA保持，结果不回流分支选择、alpha、轮次、父权重或目标分组权重；没有选择性重跑。

G训练循环及保存1092.857秒，解析变换准备0.521秒，完整G训练流程1093.996秒/18.233分钟；控制器启动至完整分析1428.170秒/23.803分钟，目标七景预测280.480秒。源模拟缓存初次构建和源梯度校准成本另计，未把共享GPU时段耗时差归因于单一方法。保留启动后外部新增任务造成同卡4个可见PID的事实，本任务始终只有1个GPU worker。

仍266951参数/2prior/4CG/27FFT，无额外推理支路，同卡交替G/G+F测量见matched_profile。源V诊断仅1800条预定面板，不扩展为整个V或TX细节证明。解析RF可行性和源误差下降均不能替代正式性能；R01历史解析损失负结果保留，不宣称整条研发从未使用F信息。
