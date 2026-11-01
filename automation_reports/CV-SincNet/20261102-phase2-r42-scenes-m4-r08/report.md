# R42 固定单场景 K5/N20 恢复

- run_id：`20261102-phase2-r42-scenes-m4-r08`
- group_id：`phase2-frozen-R42-group1-completion`；类别：`cvs`；阶段：`Phase2-B-C`
- 配置与矩阵：[experiment.json](experiment.json)；状态记录：[events.jsonl](events.jsonl)
- 当前登记状态：PLANNED（实际状态按events.jsonl及独立证据更新）

## 目的与对照

After the user-prioritized MIX completion, resume original preregistered fixed LS/MU/LU scene-only comparison, retaining prior outputs without reuse of target scores for tuning.

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

2026-11-02 运行核实：VERIFIED RUNNING: original fixed single-scene K5/N20 scope; four actual-source no-query smokes PASS, 222 closed scene predictions indexed for exact reuse, no new source updates, single-scene scoring only after all 1260 predictions fixed. MIX outcome not used for method selection.

## 原 Group1 完整对照已汇总

VERIFIED：本run单场景1260任务/27720行与已交付r07的672 MIX任务/139776行均各自冻结预测后独立评分。三单场景每景420单元；原生10方法各1932个配对单元、46704个指标行，49728条逐单元配对与232组聚合及预算独立核对。原生DA仅B、CIL仅C；CIL旧类标签预算与R42不同，不声称标签等价。ISSL用户停止，Tweak/CSIL N/A，Hu scaled保持独立身份。完整 [Group1结果总览](results_group1/report.md)、[单场景与A/B/C/DURA因子](results_group1/scene_factor_report.md)、[外部方法](results_group1/external_report.md)。

MIX及场景各自独立truth-last；单场景在MIX结果公开后执行但方法在此前固定，不作为新盲测。冷C/B后C仅固定资源点测量且全核重拟合，无新准确率测试；MACs/FLOPs/能耗未测。
