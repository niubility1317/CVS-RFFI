# 接收机均衡融合源域诊断

- run_id：`20261108-phase1-dura-direct-balanced-mix-source-probe-s2026092701-r01`
- group_id：`phase1-dura-direct-balanced-mix-source-probe`；类别：`cvs`；阶段：`Phase1-restoration`
- 配置与矩阵：[experiment.json](experiment.json)；状态记录：[events.jsonl](events.jsonl)
- 当前登记状态：PLANNED（实际状态按events.jsonl及独立证据更新）

## 目的与对照

冻结均衡门控，比较 DURA、直接恢复器及融合输出在不重叠源 pilot 上的误差与融合比例。

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

## 已完成

N607 CPU 两线程后台诊断 PID 1089611 完成，真实 8 条样本 smoke 通过，完整 672 条源 pilot 分析见 analysis.md。测试含历史探索，不能推断地面正式识别结果。
