# DURA 留一视图教师七景测试

- run_id：`20261107-phase1-dura-leaveout-multiview-ground-test-s2026092701-r01`
- group_id：`phase1-dura-leaveout-multiview`；类别：`cvs`；阶段：`Phase1-restoration`
- 配置与矩阵：[experiment.json](experiment.json)；状态记录：[events.jsonl](events.jsonl)
- 当前登记状态：PLANNED（实际状态按events.jsonl及独立证据更新）

## 目的与对照

冻结 DURA，多尺度修正训练时使用同一源记录另外两视图的 DURA 输出均值作为教师。

## 数据、seed与模型来源

实际数据契约、权限例外、完整seed角色、checkpoint来源和选择规则见experiment.json。
逐行配置通过config_ref/resolved_config_ref定位；待补项必须在对应生命周期补齐。

## 执行与存储

命令、环境、CWD、commit、launch owner、输出和日志路径见experiment.json。
实际PID/GPU、读取时间、remote readback、失败或替代关系在此追加，并用record命令记录证据指针。

## 结果与覆盖

源训练队列等待 GPU，尚无冻结候选或预测。固定七景全部完成后执行独立 truth-last 评分，与同输入 DURA 和 raw clean 对照。

## 交接

记录已完成、当前run/commit、证据路径、阻塞与下一步；恢复先查原run，不重复启动。
