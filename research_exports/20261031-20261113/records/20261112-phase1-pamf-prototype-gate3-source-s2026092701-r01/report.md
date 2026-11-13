# 10类 PAMF 分支筛选三票门槛源域对照

- run_id：`20261112-phase1-pamf-prototype-gate3-source-s2026092701-r01`
- group_id：`phase1-pamf-branch-prototype-tuning`；类别：`cvs`；阶段：`Phase1`
- 配置与矩阵：experiment.json；状态记录：events.jsonl
- 当前登记状态：STOPPED（保留既有产物）

## 目的与对照

10类三票分支筛选对比原四票；复用同划分同种子的本轮 E20，从第 21 轮独立续训。

## 数据、seed与模型来源

实际数据契约、权限例外、完整seed角色、checkpoint来源和选择规则见experiment.json。
逐行配置通过config_ref/resolved_config_ref定位；待补项必须在对应生命周期补齐。

## 执行与存储

命令、环境、CWD、commit、launch owner、输出和日志路径见experiment.json。
实际PID/GPU、读取时间、remote readback、失败或替代关系在此追加，并用record命令记录证据指针。

## 结果与覆盖

2026-11-12：用户要求取消源域留出并保持原始数据配置后，精确停止了本试验所属控制器及子进程；独立进程检查确认停止，其他 owner 未受影响。该未完成试验没有正式测试结果，已有产物保留。后续改用完整源数据的独立 run。

匹配的四票源域基准已完成。[计算环境] 真实 checkpoint 无查询 smoke 通过，候选已从相同源域开发划分、相同种子的 E20 在 GPU 0 续训；独立读回进程和运行配置，尚无门槛对照评分。执行代码 commit `e2ef742496865659103f05c90c98602d4479542d`。

按预登记 artifact 逐项记录路径和缺项；保留每 row 与 RX/day/TX/scene/K/seed 的对应关系。
源域训练完成、预测完成、评分完成及协议有效性分别陈述。不得用总索引或旧状态证明当前运行。

## 交接

记录已完成、当前run/commit、证据路径、阻塞与下一步；恢复先查原run，不重复启动。
