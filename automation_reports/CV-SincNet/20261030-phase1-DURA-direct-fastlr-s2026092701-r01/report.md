# DURA从零单阶段E20：源拟合学习率匹配

- run_id：`20261030-phase1-DURA-direct-fastlr-s2026092701-r01`
- group_id：`phase1-dura-single-stage-efficiency`；类别：`cvs`；阶段：`Phase1`
- 配置与矩阵：[experiment.json](experiment.json)；状态记录：[events.jsonl](events.jsonl)
- 当前登记状态：PLANNED（实际状态按events.jsonl及独立证据更新）

## 目的与对照

MatchedscratchE20sameinit/data/loss/budget;onlypeakLR2e-4to1e-3;sourcefull20epochunderfitdiagnosis;noancestorweights/teacher;all7formal

本候选的直接依据是完整source训练日志：从零E20的最终IQNMSE0.04014097，常规续训0.02096943；峰值LR2e-4沿用的是旧权重微调设置，从零20轮末仍有源拟合缺口。它是优化器日程假设，不认定全部误差必然来自学习率。本run在Agent读取该pilot正式分数之前已登记。

唯一科学变更为峰值LR从2e-4到1e-3；固定warmup247、最终2e-5、E20/4940、batch256、AdamW/WD/clip、随机初始化/modelseed、source shuffle/probes/目标权重及所有数据/七景矩阵。仍无祖先checkpoint、学习教师或TX/F/V/target训练。三个子模式run/worker/score均进入同一fast_lr模块，customschema严格加载同一新配置；新不可覆盖root。测试已验证子进程传播与LR端点/预算，一次独立P0/P1审查PASS。此前两个恢复队列已全部结束，不会重新启动；当前任务GPUworker仍合计最多2。

每版均完整测试；不按source表现取消正式预测，不按目标结果调参或换父。成功仍要求六个星地场景逐个距同一rawclean≤3pp并保持clean性能，同时核算累计训练成本。只完成更快的source拟合不视为目标完成。

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
