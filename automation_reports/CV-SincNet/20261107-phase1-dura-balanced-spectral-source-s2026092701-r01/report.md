# DURA 频段平衡 GPU 训练

- run_id：`20261107-phase1-dura-balanced-spectral-source-s2026092701-r01`
- group_id：`phase1-dura-balanced-spectral`；类别：`cvs`；阶段：`Phase1-restoration`
- 配置与矩阵：[experiment.json](experiment.json)；状态记录：[events.jsonl](events.jsonl)
- 当前登记状态：RUNNING；N607 GPU 0 上控制器 PID 3953857、训练 PID 3954047 已核实。

## 目的与对照

冻结 DURA，多尺度修正同时优化源 IQ 与四频段归一化误差。

## 数据、seed与模型来源

实际数据契约、权限例外、完整seed角色、checkpoint来源和选择规则见experiment.json。
逐行配置通过config_ref/resolved_config_ref定位；待补项必须在对应生命周期补齐。

## 执行与存储

命令、环境、CWD、commit、launch owner、输出和日志路径见experiment.json。
实际PID/GPU、读取时间、remote readback、失败或替代关系在此追加，并用record命令记录证据指针。

## 结果与覆盖

本机[真实 DURA GPU 技术检查](technical_smoke.md)通过。N607 真实 DURA 源 smoke 使用 64 个源 ID × 三视图，通过；正式训练已完成第 3/5 轮，累计训练日志计时 439.4 秒，第 3 轮源损失 0.0268752。该源损失不是识别准确率。七景预测和独立 truth-last 尚未开始，暂无识别结果。

## 交接

固定五轮训练完成后自动执行七景预测和独立评分；恢复时核实唯一 owner，不重复启动。
