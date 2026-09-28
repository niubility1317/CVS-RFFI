# 固定Phase1的D92-SCV新旧类完整配对确认

- run_id：`20260928-phase2-d92-scv-confirmation-manytx-m4-r02`
- group_id：`d92-fixed-phase1-support-cv-upgrade`；类别：`cvs`；阶段：`Phase2-joint-confirmation`
- 配置与矩阵：[experiment.json](experiment.json)；状态记录：[events.jsonl](events.jsonl)
- 当前登记状态：ANALYZED（实际状态按events.jsonl及独立证据更新）

## 目的与对照

4个固定新模型seed、3个与本轮源/目标RX不重合的接收机；全K、全部新类规模及5支持seed，对原D92与冻结SCV做一次完整评分。

## 数据、seed与模型来源

实际数据契约、权限例外、完整seed角色、checkpoint来源和选择规则见experiment.json。
逐行配置通过config_ref/resolved_config_ref定位；待补项必须在对应生命周期补齐。

## 执行与存储

命令、环境、CWD、commit、launch owner、输出和日志路径见experiment.json。
实际PID/GPU、读取时间、remote readback、失败或替代关系在此追加，并用record命令记录证据指针。

## 结果与覆盖

完整7236条结果已评分和汇总。候选未达到新旧类全面提升目标，未晋级。详见[完整汇总](results/summary/report.md)和[产物定位](results/artifacts.json)。
源域训练完成、预测完成、评分完成及协议有效性分别陈述。不得用总索引或旧状态证明当前运行。

## 交接

记录已完成、当前run/commit、证据路径、阻塞与下一步；恢复先查原run，不重复启动。

Manual technical recovery after diagnosis, not performance rerun.15 runner tests passed including absent reference.classes, actual/ground class binding and validated capsule reuse. No model, algorithm, matrix or data changes.

VERIFIED r02 supervisor2354788 and all4 matched final checkpoint exports. Four CPU paired prediction lanes running; no target truth read. Current release841f189690884f0ae51ad8778099fc576e48d849.

VERIFIED complete7236 paired results; all processes terminal. Scientific acceptance FAILED: old/new/H deltas pp at K1(+6.46,-7.00,-3.87),K5(+7.82,+0.26,+2.63),K10(+3.00,-0.85,+0.24),K20(-0.23,-0.04,-0.11). Goal active; candidate not promoted. Full metrics/plots and compact fit CSV preserved. Auxiliary source training scope awaiting user clarification.

![新旧类与H](results/summary/old_new_h.png)

![全部K和新类规模的变化](results/summary/delta_all_k_new.png)
