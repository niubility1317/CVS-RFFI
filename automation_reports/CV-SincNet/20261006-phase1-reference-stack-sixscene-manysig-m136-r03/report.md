# reference_response136行独立测试修复：原混合视图及clean+六完整residual

- run_id：`20261006-phase1-reference-stack-sixscene-manysig-m136-r03`
- group_id：`cvs-reference-phase1-stack-sixscene-evaluation`；类别：`cvs`；阶段：`Phase1`
- 配置与矩阵：[experiment.json](experiment.json)；状态记录：[events.jsonl](events.jsonl)
- 当前登记状态：PLANNED（实际状态按events.jsonl及独立证据更新）

## 目的与对照

保持训练进程和不可变release不变；等待全部source冻结及旧预测入口因已知契约schema缺陷自然失败退出，在新目录执行原混合测试及七完整视图测试，独立truth-last评分与复算。

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

## 独立测试接续

R3已有16行健康训练，不停止、重启或热修改。已定位两条旧预测入口读取不存在的契约字段，必在目标query读取前失败。此独立任务等待全部136行合法源域冻结、旧控制器自然退出，并严格核实该故障与无旧预测产物后，才使用修复后的入口在新目录执行原混合测试，再完成clean及六个完整residual视图。任何其他故障都停止并保留产物。训练权重、矩阵、预算、源域选择规则、数据和全部测试分层不变。

## 2026-10-06恢复发布核验

2026-10-06T11:23:46.162959+08:00：VERIFIED。R2的16行完成产物通过实际源角色及完整scratch来源核验，保持旧目录不变。R3已有16行训练，完成第8至11轮，所有PID存活且日志增长；每GPU两行，另外12行R3排队，R4/R5/R6共92行按源域冻结顺序继续。训练控制器PID=4170278，release commit=d2a1ef36b41827d151e9f7c6927cc3ef9b8e9cb2。执行加速实测生效，源验证缓存每epoch命中106次、0次miss；原FP32、E200×222预算不变。

发布r02曾在启动前因误用契约字段classes而失败；独立读回确认无submit及run目录，未启动进程，原release保留。r03以完整expected源契约逐项匹配修复，远端Torch2.1 CPU兼容检查VERIFIED。

随后审查发现原预测入口具有同一schema缺陷。已发布独立测试修复run `20261006-phase1-reference-stack-sixscene-manysig-m136-r03`，PID=4176081，commit=3b5a73a6930cbdfa5db0ddd11ab49326256904ec，当前WAITING_PARENT_SCHEMA_FAILURE。它等待全部源域冻结、旧预测入口自然退出且已知故障证据匹配后，在独立目录完成原混合测试，再完成clean及六完整residual视图的预测、truth-last评分和独立复算。旧训练/等待器未停止、未热改；旧失败记录继续保留。当前本批目标测试完成0/136，不能宣称实验全部完成。

## 2026-10-06下午进度与详细测试结果

2026-10-06T16:13:09.802859+08:00：新增矩阵训练完成28/136，R3另16行运行且自动补位正常；R2至R6尚无目标测试。R1全部16行完成预测、独立评分及复算，状态ANALYZED；本次已更正本地旧RUNNING登记。完整场景/seed/RX/TX指标、640条原始评分与当前队列见[详细报告](evidence/status_20261006_pm/report.md)。六个完整residual环境仍无测试结果，等待全部源训练冻结。
