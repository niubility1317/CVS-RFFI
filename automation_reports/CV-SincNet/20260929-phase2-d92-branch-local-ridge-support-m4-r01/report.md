# BranchLocalRidge局部径向核与单样本代理support诊断

- run_id：`20260929-phase2-d92-branch-local-ridge-support-m4-r01`
- group_id：`d92-branch-local-ridge-support-development`；类别：`diagnostic`；阶段：`Phase2-support-only-diagnostic`
- 配置与矩阵：[experiment.json](experiment.json)；状态记录：[events.jsonl](events.jsonl)
- 当前登记状态：ANALYZED（实际状态按events.jsonl及独立证据更新）

## 目的与对照

固定单视图分支表示，对比原分支岭回归、交互岭回归及训练折内确定尺度的径向核岭回归；完整物理OOF及全部support内1-shot anchors。

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

LocalRidge已预登记，尚未发布或启动。2026-09-29只读N607核查确认8份缓存绑定一致、输出与release目录不存在；复用原始support缓存，不加载模型、不读取源样本或query。联合合成测试仅一项Windows高精度测试参照失败，其余测试通过；修复参照和独立P0/P1审查正在进行，不能据此声称性能改善。

LocalRidge本地相关验证已通过：联合四文件测试只有一项Windows高精度参照失败，改为80位Decimal独立参照后，失败项及两处同期变更的3项复测全部通过；未放宽方法阈值。独立P0/P1审查见docs/D92_BRANCH_LOCAL_RIDGE_P0_20260929.md。证据见evidence/local_validation_20260929.json。此状态仅代表代码验证，尚无性能结论。

VERIFIED: LocalRidge support-only supervisor PID294733 and four owned CPU children have matching live argv/cwd; runtime96f74ac9dcb06d72923324f3cd44a63dac65a2de. All four initial lanes produced first trueK1 numerical record; remaining four lanes pending. No query/source input or checkpoint reload; no performance conclusion.

后续独立读回readback_1790689250.json确认同一主进程及4个children仍存活且命令匹配；首批4条rx3分别完成40/41/41/41个parent，共163/4800。已完成真实K20、26注册类的诊断，日志末尾未见错误。其余4条rx1仍排队；仅记录运行进度，不据部分指标调参或停止。

VERIFIED: all8 lanes/4800 parents complete;1200 trueK1 numeric-only,3600 physicalOOF parents,42000 proxy anchors,158400 actual factorizations,0 optimizer steps. Supervisor and owned children exited. No query/source input. Full evidence analysis next: tools/analyze_d92_branch_local_ridge_probe.py --spec configs/d92_branch_local_ridge_support_20260929.json --analysis-release d92_branch_local_ridge_analysis_20260929_r01. No performance conclusion yet.

VERIFIED: complete support analysis (4800 parents) passes both frozen per-parent-K A/B screens vs both controls. Standard OOF old/new/H positive in all12 K x positive-newcount strata; proxy old means slightly negative, and K10/new20 proxy H vs BranchRidge is -0.011pp. Not strict joint improvement; trueK1 numerical-only. Runtime96f74ac9; analysis09d4a0db. Method stays frozen. Full interpretation docs/D92_BRANCH_LOCAL_RIDGE_SUPPORT_RESULT_20260929.md. Next: authorized full repeated query benchmark, no new independent-data claim.
