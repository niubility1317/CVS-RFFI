# D92-BranchLocalRidge冻结方法rx3完整重复基准

- run_id：`20260929-phase2-d92-branch-local-ridge-repeat-rx3-m4-r01`
- group_id：`d92-fixed-phase1-branch-local-ridge-repeated-benchmark`；类别：`cvs`；阶段：`Phase2-repeated-benchmark`
- 配置与矩阵：[experiment.json](experiment.json)；状态记录：[events.jsonl](events.jsonl)
- 当前登记状态：ANALYZED（实际状态按events.jsonl及独立证据更新）

## 目的与对照

固定Phase1及support诊断前冻结的LocalRidge公式；复用BranchRidge原始received五块特征，在当前row support估计局部核带宽、中心化与trace尺度，一次ridge拟合，全部K同式。

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

LocalRidge frozen full-matrix repeated benchmark preregistered; not launched. Source-free cached Phase1 provenance; original D92, BranchRidge and BranchInteraction references fixed. No score-driven changes.

VERIFIED readonly preflight:8 frozen BranchRidge original-received caches bound to capsules/checkpoints; all destination/release paths absent; CPU only, no source/query score reads. Native stdlib publisher uses verified F:/App/miniconda3/python.exe; broken original Conda junction is not used. Frozen core unchanged; current-support-only fullmatrix benchmark prepared with3unchanged baselines.

VERIFIED RUNNING: supervisor326110 and4ownedCPU predictors have matched live argv/CWD; all4model rows show completed fit/prediction progress. Runtime commitde10068cd1168ab064c835bf341ddb0f63e1231c; no score files read. Continue original run; no restart, retune or partial interpretation. Both cohorts must complete before score download.


<!-- LOCAL_RIDGE_FINAL_RESULTS_20260929 -->

## 完整重复基准结果

全部7236条评分完成，进程退出；指标算术、三基线逐任务绑定及完整成本审计均为VERIFIED。四个K的新旧联合均值中，旧类、新类和H相对原D92、BranchRidge、BranchInteraction均提高；不等于每个任务或分层都改善，K1优势较小。

完整结果与成本见[产物索引](results/artifacts.json)，解释见[LocalRidge结果报告](E:/type10-7/docs/D92_BRANCH_LOCAL_RIDGE_RESULT_20260929.md)。用户2026-09-29明确要求“先不用独立验证，继续提升”；新增独立数据验证暂缓，继续仅基于合法support的query-blind研发。本轮已有结果不回流调参、拼接方法或选择性重跑。

VERIFIED full repeated benchmark complete:4800pairedcandidatefits/9648jointscores; allconfusion arithmetic and threebaseline identity, fullfit resourceaudit pass. PerK jointold/new/H positive vsall3baselines; K1small and localdeclines preserved. User explicitly deferred independentvalidation and requested continued optimization; next design remains query-blind and support-only. Goal ACTIVE.
