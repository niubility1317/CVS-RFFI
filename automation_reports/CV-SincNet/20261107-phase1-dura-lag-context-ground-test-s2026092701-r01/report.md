# DURA 长距离相关恢复七景测试

- run_id：`20261107-phase1-dura-lag-context-ground-test-s2026092701-r01`
- group_id：`phase1-dura-lag-context`；类别：`cvs`；阶段：`Phase1-restoration`
- 配置与矩阵：[experiment.json](experiment.json)；状态记录：[events.jsonl](events.jsonl)
- 当前登记状态：QUEUED；源训练固定最终轮后自动预测。

## 目的与对照

冻结 DURA，使用非循环 20/40 延迟的相位对齐差异作为源训练修正网络输入。

## 数据、seed与模型来源

实际数据契约、权限例外、完整seed角色、checkpoint来源和选择规则见experiment.json。
逐行配置通过config_ref/resolved_config_ref定位；待补项必须在对应生命周期补齐。

## 执行与存储

命令、环境、CWD、commit、launch owner、输出和日志路径见experiment.json。
实际PID/GPU、读取时间、remote readback、失败或替代关系在此追加，并用record命令记录证据指针。

## 结果与覆盖

固定 final5 后自动执行 clean 及六种 practical 场景、每景 168000 条；全部预测完成后独立 truth-last，并与固定 DURA 逐记录配对，报告逐景和 TX/RX/day。当前尚无正式预测与身份识别结果。

## 交接

记录已完成、当前run/commit、证据路径、阻塞与下一步；恢复先查原run，不重复启动。
