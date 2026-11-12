# 1% 标签 PAMF 分支原型温度源域内部开发试验

- run_id：`20261112-phase1-pamf-prototype-label01-source-s2026092701-r01`
- group_id：`phase1-pamf-branch-prototype-tuning`；类别：`cvs`；阶段：`Phase1`
- 配置与矩阵：[experiment.json](experiment.json)；状态记录：[events.jsonl](events.jsonl)
- 当前登记状态：RUNNING（实际状态按 events.jsonl 及独立证据更新）

## 目的与对照

在 1% 标签配置从零训练三组相同划分的 Hybrid＋分支原型，仅在源 L 内部开发集选温度。

## 数据、seed与模型来源

实际数据契约、权限例外、完整seed角色、checkpoint来源和选择规则见experiment.json。
逐行配置通过config_ref/resolved_config_ref定位；待补项必须在对应生命周期补齐。

## 执行与存储

命令、环境、CWD、commit、launch owner、输出和日志路径见experiment.json。
实际PID/GPU、读取时间、remote readback、失败或替代关系在此追加，并用record命令记录证据指针。

## 结果与覆盖

2026-11-12：N607 发布及远端编译完成，真实 checkpoint 无查询 smoke 通过。源域单种子三候选分别在 GPU 4、5、7 训练，已独立读回所属进程及配置，暂未记录失败。执行代码 commit `dbe204445e85fdcfed0b62f0c8cda0c56531ab56`。本试验仅进行源域开发，尚无目标预测或性能结果。

按预登记 artifact 逐项记录路径和缺项；保留每 row 与 RX/day/TX/scene/K/seed 的对应关系。
源域训练完成、预测完成、评分完成及协议有效性分别陈述。不得用总索引或旧状态证明当前运行。

## 交接

记录已完成、当前run/commit、证据路径、阻塞与下一步；恢复先查原run，不重复启动。
