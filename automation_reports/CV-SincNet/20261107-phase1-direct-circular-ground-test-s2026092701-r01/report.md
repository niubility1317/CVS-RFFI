# 共轭相位等变恢复七景测试

- run_id：`20261107-phase1-direct-circular-ground-test-s2026092701-r01`
- group_id：`phase1-direct-circular`；类别：`cvs`；阶段：`Phase1-restoration`
- 配置与矩阵：[experiment.json](experiment.json)；状态记录：[events.jsonl](events.jsonl)
- 当前状态：固定 20 轮源训练已完成，七景预测运行中；预测 worker PID 454794。

## 目的与对照

从接收 IQ 直接恢复，增加局部共轭特征并保持全局相位等变。

## 数据、seed与模型来源

实际数据契约、权限例外、完整seed角色、checkpoint来源和选择规则见experiment.json。
逐行配置通过config_ref/resolved_config_ref定位；待补项必须在对应生命周期补齐。

## 执行与存储

命令、环境、CWD、commit、launch owner、输出和日志路径见experiment.json。
实际PID/GPU、读取时间、remote readback、失败或替代关系在此追加，并用record命令记录证据指针。

## 结果与覆盖

源预测 smoke 通过 112 条，正式七景预测正在运行；全部视图固定后才独立 truth-last 评分并与 DURA 配对，目前尚无识别性能结果。
源域训练完成、预测完成、评分完成及协议有效性分别陈述。不得用总索引或旧状态证明当前运行。

## 交接

记录已完成、当前run/commit、证据路径、阻塞与下一步；恢复先查原run，不重复启动。
