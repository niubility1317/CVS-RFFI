# 更新CVS网络的Phase1修复：时序门控与学习率固定对照32行

- run_id：`20261008-phase1-reference-repair-manysig-m32-r01`
- group_id：`cvs-reference-phase1-source-grounded-repair`；类别：`cvs`；阶段：`Phase1`
- 配置与矩阵：[experiment.json](experiment.json)；状态记录：[events.jsonl](events.jsonl)
- 当前登记状态：PLANNED（实际状态按events.jsonl及独立证据更新）

## 目的与对照

从代码和源日志发现的问题出发，2×2门控/学习率对照及CE、LEO、独立域对抗、MixStyle适配验证；全部固定对照从零训练，不按历史目标成绩筛选。

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

## 固定修复设计

本轮比较8组×4模型seed，所有模型使用更新后的reference_response网络。每行从零训练200轮、每轮222次更新；源域物理L/U/V分别6300/56700/27000，U隐藏TX标签，V只读。固定E200，无候选筛除。修复依据为源日志与实际代码路径，不根据已经暴露的目标分数挑方法或调参数。

|组|改变因素|
|---|---|
|ce_cosine|CE监督、无LEO、余弦LR|
|leo_cosine|CE＋原登记LEO、余弦LR|
|pseudo_batch_constant|原随机批内邻居门控、恒定LR|
|pseudo_bank_constant|仅改为物理样本跨轮稳定性门控|
|pseudo_batch_cosine|仅改为余弦LR|
|pseudo_bank_cosine|同时修复门控与LR|
|bank_cosine_domain|上组＋原权重域对抗，伪标签域门控始终关闭|
|bank_cosine_mixstyle|上组＋原参数MixStyle，保留原适配层位置|

四个伪标签核心对照均为LEO＋两阶段＋EMA＋伪标签。阈值和伪标签损失权重不放宽。跨轮门控要求同一源U物理样本在相邻两轮高置信预测一致，不依赖随机批中碰到邻居；第一轮不放行。LR从E1的0.0002连续余弦降至E200的0.000001，不启用MUSE，不改变优化器分组或原有裁剪。

MixStyle仍只作用于time_down/t1，参考响应分支绕过它；本轮将其作为明确限定的对照，不宣称已解决所有增强与物理指纹兼容问题。域对抗保持lambda_domain=1、lambda_adv=0.35，先测试去除门控耦合后的作用，不盲目调整权重。

## 验证与自动收尾

完整32行源训练固定后，全部224个模型×视图预测完成后才由独立进程打开truth。测试复用已有VALIDATED_ONCE的clean和六个完整residual场景，每视图168000相同物理query。输出Accuracy、Macro-F1、逐RX/TX混淆矩阵、四seed均值/标准差与成对差值、源门控采用率、实际损失/LR/梯度、时间/内存。历史benchmark已暴露，不宣称全新盲测。失败保留，无按低成绩停止、回退旧权重或选择性重跑。Phase2/K/新增类指标不适用。

## 当前状态

PLANNED。性能提升尚待实际完整测试证明。已有实验不停止、不热改、不重启。新run最多4个并行worker，每GPU总进程不超过2；预CUDA训练入口可被旧控制器识别。

## 资源排队修复

独立审查指出旧控制器不共享原子GPU预约，R5切R6时可能与新owner补位竞争。新owner发布后先记录WAITING_PRIOR_OWNERS，按/proc核实旧recover、sixscene_after及evaluation_recovery等真实owner退出，再开始训练；每次补位仍检查旧owner。新训练/预测入口分别命名train_repair.py、predict_repair.py，可被原有预约扫描识别。排队不等于模型训练开始，启动证据分别记录。

## 发布核验与交接

核验时间：2026-10-08T00:45:24.951643+08:00。发布状态VERIFIED；实际代码commit：`ae783c81cd4949f93a80dbd2e80c4b2568eb99ea`；控制器PID：975107。本地29项相关测试通过，本地和N607各8组×5个关键epoch合成数据真实训练循环冒烟通过，均保存并重载自身scratch checkpoint；不把冒烟当成正式性能结果。

当前QUEUED/WAITING_PRIOR_OWNERS，0/32正式训练开始，0/32正式测试完成。正在等待原recover PID4170278、sixscene_after PID4170279和evaluation_recovery PID4176081真实退出。旧12个R5训练worker保留。控制器每30秒检查旧owner，后续最多4个新worker，并持续执行每GPU总进程最多2限制；完成后自动固定32行、生成224个预测视图、独立truth-last评分及成对对照报告。

最新真实状态读取脚本：`evidence/inspect_remote.py`，当前证据：`evidence/remote_readback.json`。不得重新发布或重复启动此run。正式源worker启动后的实际参数、scratch来源、PID/cwd、完整日志由source包装器写入各row/source；当前尚无这些正式训练产物。后续应核实各阶段产物，再回收scores/summary/paired_results/resources/analysis并更新原登记、镜像Git、提交推送。性能是否改善尚未证明。
