# 6 个源接收机 PAMF 分支原型五温度源域调优

- run_id：`20261112-phase1-pamf-prototype-m6-source-s2026092701-r01`
- group_id：`phase1-pamf-branch-prototype-tuning`；类别：`cvs`；阶段：`Phase1`
- 配置与矩阵：experiment.json；状态记录：events.jsonl
- 当前登记状态：STOPPED（保留既有产物）

## 目的与对照

在 6 个源接收机配置下从零训练五种原型温度，只用源域开发集选参。

## 数据、seed与模型来源

实际数据契约、权限例外、完整seed角色、checkpoint来源和选择规则见experiment.json。
逐行配置通过config_ref/resolved_config_ref定位；待补项必须在对应生命周期补齐。

## 执行与存储

命令、环境、CWD、commit、launch owner、输出和日志路径见experiment.json。
实际PID/GPU、读取时间、remote readback、失败或替代关系在此追加，并用record命令记录证据指针。

## 结果与覆盖

2026-11-12：用户要求取消源域留出并保持原始数据配置后，精确停止了本试验所属控制器及子进程；独立进程检查确认停止，其他 owner 未受影响。该未完成试验没有正式测试结果，已有产物保留。后续改用完整源数据的独立 run。

2026-11-12：[计算环境] 发布及远端编译完成，真实 checkpoint 无查询 smoke 通过。五个温度候选分别在 GPU 2、6、7、1、3 训练，已独立读回进程、命令与显卡；GPU 2、6 与其他用户进程共享，未停止或修改这些进程。执行代码 commit `705740f448c42080df7996ae02fbb0cfb66040d3`。本试验仅进行源域开发，尚无目标预测或性能结果。

按预登记 artifact 逐项记录路径和缺项；保留每 row 与 RX/day/TX/scene/K/seed 的对应关系。
源域训练完成、预测完成、评分完成及协议有效性分别陈述。不得用总索引或旧状态证明当前运行。

## 交接

记录已完成、当前run/commit、证据路径、阻塞与下一步；恢复先查原run，不重复启动。
