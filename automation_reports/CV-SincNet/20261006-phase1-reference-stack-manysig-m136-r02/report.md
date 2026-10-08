# reference_response原Phase1剩余机制：R2至R6共136行：故障恢复r02

- run_id：`20261006-phase1-reference-stack-manysig-m136-r02`
- group_id：`cvs-reference-original-phase1-overlay`；类别：`cvs`；阶段：`Phase1`
- 配置与矩阵：[experiment.json](experiment.json)；状态记录：[events.jsonl](events.jsonl)
- 当前登记状态：PLANNED（实际状态按events.jsonl及独立证据更新）

## 目的与对照

修复进程退出竞态后新建独立运行。复用16行已完成、从零训练且未接触目标的R2产物；剩余120行按原矩阵从零训练。预算、源域选择及目标测试计划不变。

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

## 故障恢复方案及验证

旧控制器退出竞态已复现并修复：进程退出时消失的cmdline/cwd视为已退出，再核对完成产物；真实存活身份变化仍报错。新控制器拥有新worker，不再接管旧PID。24项相关测试通过，含队列完成自动补位、R2禁止重训、目标访问门控及七视图完整性。独立审查无P0/P1。

R2保持原目录、配置、权重和日志；发布前逐行核对源契约、E200/44400、scratch来源及无目标访问。R3至R6在新目录按原方案从零训练，16并发、每卡最多2行。保留此前验证的执行加速。所有136行源域冻结后执行预登记测试；clean及六residual各168000同物理query，全部预测固定后独立truth-last评分并复算。当前尚未启动，实际commit/PID/日志将发布后登记。

## 2026-10-06恢复发布核验

2026-10-06T11:23:46.162959+08:00：VERIFIED。R2的16行完成产物通过实际源角色及完整scratch来源核验，保持旧目录不变。R3已有16行训练，完成第8至11轮，所有PID存活且日志增长；每GPU两行，另外12行R3排队，R4/R5/R6共92行按源域冻结顺序继续。训练控制器PID=4170278，release commit=d2a1ef36b41827d151e9f7c6927cc3ef9b8e9cb2。执行加速实测生效，源验证缓存每epoch命中106次、0次miss；原FP32、E200×222预算不变。

发布r02曾在启动前因误用契约字段classes而失败；独立读回确认无submit及run目录，未启动进程，原release保留。r03以完整expected源契约逐项匹配修复，远端Torch2.1 CPU兼容检查VERIFIED。

随后审查发现原预测入口具有同一schema缺陷。已发布独立测试修复run `20261006-phase1-reference-stack-sixscene-manysig-m136-r03`，PID=4176081，commit=3b5a73a6930cbdfa5db0ddd11ab49326256904ec，当前WAITING_PARENT_SCHEMA_FAILURE。它等待全部源域冻结、旧预测入口自然退出且已知故障证据匹配后，在独立目录完成原混合测试，再完成clean及六完整residual视图的预测、truth-last评分和独立复算。旧训练/等待器未停止、未热改；旧失败记录继续保留。当前本批目标测试完成0/136，不能宣称实验全部完成。

## 2026-10-06下午进度与详细测试结果

2026-10-06T16:13:09.802859+08:00：新增矩阵训练完成28/136，R3另16行运行且自动补位正常；R2至R6尚无目标测试。R1全部16行完成预测、独立评分及复算，状态ANALYZED；本次已更正本地旧RUNNING登记。完整场景/seed/RX/TX指标、640条原始评分与当前队列见[详细报告](evidence/status_20261006_pm/report.md)。六个完整residual环境仍无测试结果，等待全部源训练冻结。

## 2026-10-06即时测试已完成

2026-10-06T18:15:10.394812+08:00：当前训练完成32/136（R2为16/16，R3为16/28），R3另外12行继续运行，无故障。用户明确要求立即测试已完成模型，已把快照全部32个固定E200模型以独立run `20261006-phase1-reference-completed-sixscene-manysig-m32-r01` 在clean与六完整residual环境测试完毕。37632000次预测、3136条评分、独立truth-last复算和本地混淆矩阵再核验均VERIFIED。仅使用4个空余测试名额，未停止、重启或热修改训练。原136行后续测试任务保持不变；本快照结果不反馈源域选择。详见[全部新测试结果](../20261006-phase1-reference-completed-sixscene-manysig-m32-r01/evidence/completed_test/detailed_results_zh.md)。

## 2026-10-07新增48行补测完成

2026-10-07T23:26:45.254153+08:00：训练完成80/136，当前r5共12行继续训练，无故障。R2完成16、R3完成28、R4完成32、R5 rc4完成4。本次新增48行与昨天32行互斥，共80行全部完成clean及六完整residual测试。合计94080000次预测、7840条评分，truth-last独立复算及本地混淆矩阵复核VERIFIED。20组均为4seed，R5 base尚未完成，rc4暂不作配对增益解释。训练控制器及后续136行原定测试队列保持不变。详见[80行详细结果](../20261007-phase1-reference-completed-sixscene-manysig-m48-r01/evidence/combined80/detailed_results_zh.md)。

## 2026-10-08机制收益深入分析

已完成[新骨干与原Phase1机制的详细分析](evidence/mechanism_analysis_20261008/analysis_zh.md)：完整解析96个已完成模型、19200轮、3712000步；复算7840条评分记录。确认LEO仍将本次混合卫星视图均值提高19.218个百分点，MixStyle附加收益消失，DG存在clean与残差信道取舍；定位batch内时间邻居门控约1.7%的覆盖限制、源域选择差异及骨干替换后的梯度路径变化。运行中R5与未开始R6不作完整方法判定，未改训练、选模或目标预测。证据与局限见报告。

## 用户停止原后续实验

2026-10-08T18:26:57.910565+08:00：用户明确要求停止原来的后续实验及正在进行的任务。本136行实验链已停止，独立两次进程读回均无残留：训练控制器4170278、16个R6训练进程，以及测试等待器4170279/4176081全部退出。28行待运行R6取消后续执行，自动测试等待终止；不自动恢复。92行已完成模型与七环境测试结果、16行未完成训练的已有日志/权重等产物全部保留。其他独立实验未停止。原queue_state保留为停止前历史快照，当前权威状态为本次STOPPED及停止证据。

停止脚本已发送全部TERM；随后监测遇到进程退出期间cmdline变化而报错。未重复发送停止任务，后续独立两次核验原19个PID及release匹配进程均不存在，实际停止结果VERIFIED。
