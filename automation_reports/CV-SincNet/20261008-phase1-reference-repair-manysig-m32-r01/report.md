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

## 用户授权直接启动：控制器接管

2026-10-08用户明确要求“直接启动”。解除此前等待旧启动者全部退出的排队条件；仅替换本run从未启动worker的等待owner，既有32行配置、数据/seed/E200/44400预算与worker release ae783c81保持不变。原R5/R6健康进程不停止、不热改。新控制器最多4个worker，仅向完全空闲GPU提交，读取所有/proc显式CUDA_VISIBLE_DEVICES预约（含旧-m source），新train/predict入口仍可被旧owner识别。不占用已有1个进程的GPU最后名额，以降低同时补位的竞争风险。预测与评分继续沿用原先的完整源冻结和truth-last链路。

实际handoff前须核对旧owner的PID/cwd/argv/start_ticks、WAITING状态、无launch/row产物及全部32个配置，再向该唯一等待PID发送SIGTERM；核实退出后以独占direct_owner文件启动新控制器。2项预约/僵尸进程测试已通过。

## 直接启动结果：VERIFIED

截至2026-10-08T08:17:29.908425+08:00，新控制器1170536实际运行，原本run等待控制器975107已退出。4行训练、28行待调度；无failure产物。原3个控制器及4个R5训练进程均存活，8张GPU实际各1个CUDA计算进程。新worker与原32行科学配置一致，均scratch初始化；没有访问target。

| 行 | GPU | PID | 已完成epoch |
|---|---:|---:|---:|
| ce_cosine-s2026092701 | 0 | 1170554 | 3 / 200 |
| leo_cosine-s2026092701 | 3 | 1170563 | 2 / 200 |
| pseudo_batch_constant-s2026092701 | 4 | 1170572 | 2 / 200 |
| pseudo_bank_constant-s2026092701 | 6 | 1170581 | 2 / 200 |

控制器commit：`f519f9baaa61b18e03804b7b67eaf542481e706e`；训练worker仍为`ae783c81cd4949f93a80dbd2e80c4b2568eb99ea`。容量修复3项测试通过，独立P0/P1复核PASS。真实进程、CUDA占用、生效参数和逐epoch日志已交叉确认，证据见[evidence/direct_start_readback.json](evidence/direct_start_readback.json)，后续只读检查入口为[evidence/inspect_remote_direct.py](evidence/inspect_remote_direct.py)。旧等待状态记录保留为历史。

尚无本修复矩阵的测试成绩。队列继续完成32行固定E200训练，全部冻结后自动执行clean及6个星地场景预测，再独立truth-last评分；测试结果不回流选模或调参。

## 2026-10-08 15:13结果核查

VERIFIED：5/32行完成E200与44400次更新，0行正在训练，27行排队，0/32行完成测试。控制器存活，无failure产物；截至15:14，旧R6任务共16个CUDA进程，每GPU两个。当前控制器仅向完全空闲GPU补位，本次只读核查未改调度或停止其他任务。

下表仅为源域V结果，27000个样本；全部为model seed 2026092701，固定E200。不能解释为跨接收机目标测试或四seed结论。训练耗时为各run实测墙钟时间，受共享GPU影响，不是独占硬件速度对照。

| 方法 | 源V准确率 | 最差源RX准确率 | 训练耗时(h) |
|---|---:|---:|---:|
| ce_cosine | 98.515% | 96.148% | 1.82 |
| leo_cosine | 98.611% | 96.278% | 3.43 |
| pseudo_bank_constant | 98.696% | 96.519% | 3.83 |
| pseudo_batch_constant | 98.696% | 96.352% | 3.87 |
| pseudo_batch_cosine | 98.600% | 96.259% | 4.11 |

跨轮门控修复已实际生效：E132至E200，恒定LR下伪标签平均采用率由batch的1.695%升至bank的97.393%；U真值不可用，因此这是采用率，不是伪标签正确率。两者最终源V准确率均98.696%，最差源RX由96.352%升至96.519%（+0.167个百分点）。只能证明门控覆盖恢复，尚不能证明目标泛化改善。

LEO＋cosine相对CE＋cosine源V提高0.096个百分点，最差源RX提高0.130个百分点；实测训练墙钟时间3.43h对1.82h。pseudo_batch_cosine最终源V为98.600%，比恒定LR的98.696%低0.096个百分点；均为单seed描述，不据此选模或调参。

已解析5行各完整200条epoch记录（1000条），并扫描5行全文训练stdout异常标记；无Traceback/OOM/RuntimeError/FloatingPointError命中，所有epoch的非有限loss/grad跳步均为0。未对逐step压缩日志做全面诊断。

source_matrix_frozen、prediction/complete、scoring_complete、scores、summary均未生成。按原预登记须32行全部完成源训练与冻结，再执行7视图预测和独立truth-last评分。本次没有提前测试、修改模型/参数或重排队列。证据：[progress_20261008_1513.json](evidence/progress_20261008_1513.json)。

## 2026-10-09完成19行测试

截至2026-10-09T09:39:17.606358+08:00，训练完成19/32行、4行运行、9行等待。已完成19行均有clean与六星地测试成绩；本次独立补测14行，加上此前5行。主比较为八方法共同前两个seed。测试结果仅报告，不回流原32行训练、选模或重排；未干预健康进程。详见[19行测试报告](../20261009-evaluation-reference-repair-completed14-manysig-m14-r01/evidence/results19_zh.md)。
