# Phase-equivariant DURA seven-view ground test

- run_id：`20261107-phase1-dura-phase-equivariant-ground-test-s2026092701-r01`
- group_id：`phase1-dura-phase-equivariant`；类别：`cvs`；阶段：`Phase1-restoration`
- 配置与矩阵：[experiment.json](experiment.json)；状态记录：[events.jsonl](events.jsonl)
- 当前登记状态：ANALYZED；正式七景预测和独立评分完成。

## 目的与对照

Frozen DURA with a source-only complex phase-equivariant correction conditioned on received IQ, anchor, innovation and physical variance.

## 数据、seed与模型来源

实际数据契约、权限例外、完整seed角色、checkpoint来源和选择规则见experiment.json。
逐行配置通过config_ref/resolved_config_ref定位；待补项必须在对应生命周期补齐。

## 执行与存储

命令、环境、CWD、commit、launch owner、输出和日志路径见experiment.json。
实际PID/GPU、读取时间、remote readback、失败或替代关系在此追加，并用record命令记录证据指针。

## 结果与覆盖

七景各 168000 条预测全部固定后独立 truth-last 评分。六个星地场景相对 DURA 均下降 0.0613 至 0.3077 个百分点；仅 1/6 个场景满足相对 raw clean 下降不超过 3% 的目标，无论采用百分点还是相对百分比口径。逐景、配对纠错和 TX／RX／day 见 [正式结果](analysis.md)。

## 交接

记录已完成、当前run/commit、证据路径、阻塞与下一步；恢复先查原run，不重复启动。
