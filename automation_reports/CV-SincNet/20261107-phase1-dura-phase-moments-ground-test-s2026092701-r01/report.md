# DURA 复矩条件相位七景测试

- run_id：`20261107-phase1-dura-phase-moments-ground-test-s2026092701-r01`
- group_id：`phase1-dura-phase-moments`；类别：`cvs`；阶段：`Phase1-restoration`
- 配置与矩阵：[experiment.json](experiment.json)；状态记录：[events.jsonl](events.jsonl)
- 当前状态：源训练已完成，七景预测运行中。

## 目的与对照

冻结 DURA，以逐记录二阶和四阶复数统计增强条件相位混合。

## 数据、seed与模型来源

实际数据契约、权限例外、完整seed角色、checkpoint来源和选择规则见experiment.json。
逐行配置通过config_ref/resolved_config_ref定位；待补项必须在对应生命周期补齐。

## 执行与存储

命令、环境、CWD、commit、launch owner、输出和日志路径见experiment.json。
实际PID/GPU、读取时间、remote readback、失败或替代关系在此追加，并用record命令记录证据指针。

## 结果与覆盖

固定最终第 5 轮后，使用相同 Tri-SIG-TIME SSL Main split 默认头，逐景预测各 168000 条；全部预测固定后独立评分，并逐记录比较 DURA。两个条件相位版本均完成固定预测和评分后再进行匹配比较。目前尚无识别性能结果。
源域训练完成、预测完成、评分完成及协议有效性分别陈述。不得用总索引或旧状态证明当前运行。

## 交接

记录已完成、当前run/commit、证据路径、阻塞与下一步；恢复先查原run，不重复启动。
