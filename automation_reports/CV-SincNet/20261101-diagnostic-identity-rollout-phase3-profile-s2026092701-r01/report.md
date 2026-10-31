# 完整轨迹版与 DURA 的 Phase3 系统计时

- run_id：`20261101-diagnostic-identity-rollout-phase3-profile-s2026092701-r01`
- group_id：`phase3-identity-rollout-matched-latency`；类别：`diagnostic`；阶段：`Phase3-fixed-system-synthetic-profile`
- 配置与矩阵：[experiment.json](experiment.json)；状态记录：[events.jsonl](events.jsonl)
- 当前登记状态：PLANNED（实际状态按events.jsonl及独立证据更新）

## 目的与对照

完整轨迹 E30 全部预测与评分完成后，用相同合成输入和冻结系统在同 GPU 计时，不读真实 IQ 或真值。

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

## 排队：VERIFIED

固定代码 `aaf104120b879be72b9c290a57ae7b8e3dfa33bb` 的 CPU 控制器 PID 2148644 等待完整轨迹 E30 所有预测和评分结束；尚未启动 GPU 测试。完成真实模型与合成输入 smoke、检查独占 GPU 后才计时。

## 等待进程停止：VERIFIED

只停止本任务等待中的 CPU 控制器 PID 2148644；尚无 GPU 提交、smoke 或计时。原因是准入清单遗漏另一对话的直接 train/predict 入口；使用独立 r02 修复，原记录和日志保留。
