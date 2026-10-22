# PAMF 三核心机制：源侧检查接口修复及55任务健康接管

- run_id：`20261022-phase1-PAMF-core-control-r02`
- group_id：`phase1-pamf-three-core-functional-factorial`；类别：`diagnostic`；阶段：`Phase1`
- 配置与矩阵：[experiment.json](experiment.json)；状态记录：[events.jsonl](events.jsonl)
- 当前登记状态：PLANNED（实际状态按events.jsonl及独立证据更新）

## 目的与对照

Two supervised objectives and one complete native selective-SSL group; fixed CE scaffold/backbone/head. Full-minus-group main table plus full factorial, without score-dependent regrouping.

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

## 发布前接口修复

r01 在健康任务接管之前退出，真实 checkpoint 源检查已 PASS；新增分组 smoke 直接调用内部方法，未触发 ordinary IQ 捕获。已本地复现，改成原训练外层调用；11 项聚焦测试及一次原问题定点复核 PASS。科学训练不变，r02 使用新不可覆盖目录，旧失败产物保留。
