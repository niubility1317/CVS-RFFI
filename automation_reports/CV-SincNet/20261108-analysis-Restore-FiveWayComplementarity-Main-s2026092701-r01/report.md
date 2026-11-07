# 原受损与四个恢复输出的五方互补性

- run_id：`20261108-analysis-Restore-FiveWayComplementarity-Main-s2026092701-r01`
- group_id：`fixed-five-way-prediction-complementarity`；类别：`diagnostic`；阶段：`Phase1`
- 配置与矩阵：[experiment.json](experiment.json)；状态记录：[events.jsonl](events.jsonl)
- 当前登记状态：ANALYZED（实际状态按events.jsonl及独立证据更新）

## 目的与对照

复用四个固定恢复输出及原始受损预测，计算五方正确性组合及不可部署的理论并集。

## 数据、seed与模型来源

实际数据契约、权限例外、完整seed角色、checkpoint来源和选择规则见experiment.json。
逐行配置通过config_ref/resolved_config_ref定位；待补项必须在对应生命周期补齐。

## 执行与存储

命令、环境、CWD、commit、launch owner、输出和日志路径见experiment.json。
实际PID/GPU、读取时间、remote readback、失败或替代关系在此追加，并用record命令记录证据指针。

## 结果与覆盖

独立 CPU truth-last 分析完成，14 组与原逐对评分一致。见 [分析报告](analysis.md) 和 [总体计数](aggregate_counts.json)。
源域训练完成、预测完成、评分完成及协议有效性分别陈述。不得用总索引或旧状态证明当前运行。

## 交接

记录已完成、当前run/commit、证据路径、阻塞与下一步；恢复先查原run，不重复启动。
