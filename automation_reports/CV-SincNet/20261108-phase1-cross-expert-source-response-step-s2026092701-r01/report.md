# 时频网络源域波形与响应单步对照

- run_id：`20261108-phase1-cross-expert-source-response-step-s2026092701-r01`
- group_id：`phase1-cross-expert-source-response-step`；类别：`cvs`；阶段：`Phase1-restoration`
- 配置与矩阵：[experiment.json](experiment.json)；状态记录：[events.jsonl](events.jsonl)
- 当前登记状态：PLANNED（实际状态按events.jsonl及独立证据更新）

## 目的与对照

以相同新初始化与源样本比较输出头波形损失和波形加幅度三次响应损失的内存单步，不保存模型。

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

64 条源记录的两种输出头单步诊断已完成；九项针对性测试、独立 P0/P1 审查及真实源 smoke 通过。PID 1178882 事后查询仍存在，远端日志和结果确认完成。详细结果与边界见 [analysis.md](analysis.md)。
