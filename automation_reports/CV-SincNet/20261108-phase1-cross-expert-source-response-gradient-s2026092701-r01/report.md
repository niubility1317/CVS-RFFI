# 时频跨恢复网络源域响应梯度

- run_id：`20261108-phase1-cross-expert-source-response-gradient-s2026092701-r01`
- group_id：`phase1-cross-expert-source-response-gradient`；类别：`cvs`；阶段：`Phase1-restoration`
- 配置与矩阵：[experiment.json](experiment.json)；状态记录：[events.jsonl](events.jsonl)
- 当前登记状态：PLANNED（实际状态按events.jsonl及独立证据更新）

## 目的与对照

使用与在训模型相同种子的全新初始化，比较源波形误差与四类合成微扰响应的输出头梯度，不加载在训状态。

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

真实源 checkpoint 与初始化输出 smoke 通过；四组共 32 条源记录的输出头梯度诊断完成，范围和限制见 [analysis.md](analysis.md)。不加载或更改在训状态。

N607 后台启动返回 PID 1170375；任务在第一次事后查询前完成，未捕获运行中的 PID 快照。远端日志和结果确认 source smoke 与完整计算完成。九项针对性测试及独立 P0/P1 审查通过。
