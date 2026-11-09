# PAMF 时域、PA 与频谱统计完整实验

- run_id：`20261109-phase1-pamf-split-spectrum-paper-m226-r01`
- group_id：`phase1-pamf-split-spectrum-paper`；类别：`comparison`；阶段：`Phase1`
- 配置与矩阵：[experiment.json](experiment.json)；状态记录：[events.jsonl](events.jsonl)
- 当前登记状态：RUNNING（2026-11-09 启动快照；实际进度以远端运行记录为准）

## 目的与对照

226 rows: separate PA IQ and spectral statistics, eleven data settings, five training conditions, PL permutation and E200; direct and matched DURA readouts.

## 数据、seed与模型来源

实际数据契约、权限例外、完整seed角色、checkpoint来源和选择规则见experiment.json。
逐行配置通过config_ref/resolved_config_ref定位；待补项必须在对应生命周期补齐。

## 执行与存储

命令、环境、CWD、commit、launch owner、输出和日志路径见experiment.json。
实际PID/GPU、读取时间、remote readback、失败或替代关系在此追加，并用record命令记录证据指针。

## 结果与覆盖

15 个匹配已有模型及 528 条外部方法记录已在远端准备阶段核对并接入。211 个新增模型依预登记逐行训练、冻结预测及独立评分，完整矩阵结果尚未生成。按预登记artifact逐项记录路径和缺项；保留每row与RX/day/TX/scene/K/seed的对应关系。
源域训练完成、预测完成、评分完成及协议有效性分别陈述。不得用总索引或旧状态证明当前运行。

## 交接

实际执行提交为 `0a94a2e0bef5936b68b31783ccd4c8ef8b7f8240`；发布后独立核验见 [launch_verification.json](launch_verification.json)。原 190 行任务按原计划继续；本 run 不接管其他 owner。恢复前核对本 run 的进程和已有产物，不重复启动。
