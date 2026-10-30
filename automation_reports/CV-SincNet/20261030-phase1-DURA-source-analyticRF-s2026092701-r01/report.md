# DURA 解析 RF 保真损失单阶段恢复对照

- run_id：`20261030-phase1-DURA-source-analyticRF-s2026092701-r01`
- group_id：`phase1-dura-source-analytic-RF-auxiliary`；类别：`cvs`；阶段：`Phase1`
- 配置与矩阵：[experiment.json](experiment.json)；状态记录：[events.jsonl](events.jsonl)
- 当前登记状态：PLANNED（实际状态按events.jsonl及独立证据更新）

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
