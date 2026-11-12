# 10类 PAMF 分支筛选三票门槛源域对照

- run_id：`20261112-phase1-pamf-prototype-gate3-source-s2026092701-r01`
- group_id：`phase1-pamf-branch-prototype-tuning`；类别：`cvs`；阶段：`Phase1`
- 配置与矩阵：[experiment.json](experiment.json)；状态记录：[events.jsonl](events.jsonl)
- 当前登记状态：QUEUED（实际状态按 events.jsonl 及独立证据更新）

## 目的与对照

10类三票分支筛选对比原四票；复用同划分同种子的本轮 E20，从第 21 轮独立续训。

## 数据、seed与模型来源

实际数据契约、权限例外、完整seed角色、checkpoint来源和选择规则见experiment.json。
逐行配置通过config_ref/resolved_config_ref定位；待补项必须在对应生命周期补齐。

## 执行与存储

命令、环境、CWD、commit、launch owner、输出和日志路径见experiment.json。
实际PID/GPU、读取时间、remote readback、失败或替代关系在此追加，并用record命令记录证据指针。

## 结果与覆盖

2026-11-12：N607 发布和远端编译完成，控制器已独立读回并等待匹配的四票源域基准完成。当前没有 GPU worker、训练或评分结果；基准完成后才会执行真实 checkpoint 无查询 smoke 与候选训练。执行代码 commit `e2ef742496865659103f05c90c98602d4479542d`。

按预登记 artifact 逐项记录路径和缺项；保留每 row 与 RX/day/TX/scene/K/seed 的对应关系。
源域训练完成、预测完成、评分完成及协议有效性分别陈述。不得用总索引或旧状态证明当前运行。

## 交接

记录已完成、当前run/commit、证据路径、阻塞与下一步；恢复先查原run，不重复启动。
