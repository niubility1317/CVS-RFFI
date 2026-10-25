# 跨接收机身份参照三变体单种子源留出验证

- run_id：`20261026-phase1-PAMF-RXReference-sourceHold-m3-r01`
- group_id：`phase1-pamf-receiver-reference-20261025`；类别：`cvs`；阶段：`Phase1`
- 配置与矩阵：[experiment.json](experiment.json)；状态记录：[events.jsonl](events.jsonl)
- 当前登记状态：PLANNED（实际状态按events.jsonl及独立证据更新）

## 目的与对照

同seed01原E30+20至E50，raw/同G/双G身份参照三变体；新增分支排除RX1，完整源V八视图与既有PLControl配对，源阈值选择后另行扩种子。

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

## 发布前范围

详见 [最小预登记](preregistration.md)。三个方法变体仅有一个student seed2026092701；不是三种子测试。固定原E30+20、共同源剂量0.07145844915553692及默认头联合训练，完整源V八视图逐行自动评估；源条件通过后未来三种子矩阵另行登记。本run不访问目标数据，不替换默认模型。两项新检查与compile通过，一次独立P0/P1审查PASS；真实权重联合backward smoke为launcher第一步。
