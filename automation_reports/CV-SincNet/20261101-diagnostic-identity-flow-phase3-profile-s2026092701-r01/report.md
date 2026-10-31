# 条件流与 DURA 的完整 Phase3 系统延迟对照

- run_id：`20261101-diagnostic-identity-flow-phase3-profile-s2026092701-r01`
- group_id：`phase3-identity-flow-matched-latency`；类别：`diagnostic`；阶段：`Phase3-fixed-system-synthetic-profile`
- 配置与矩阵：[experiment.json](experiment.json)；状态记录：[events.jsonl](events.jsonl)
- 当前登记状态：PLANNED（实际状态按events.jsonl及独立证据更新）

## 目的与对照

仅待 E30 固定预测和评分完成后，在同 GPU 用两条合成 G10 输入比较完整 G+F0+E0+PCD+密度阈值计算；不读目标 IQ 或真值。

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

## 排队状态：VERIFIED

固定代码 `ce7574630cc1ba0f41c3ead6eb9036004a4c6201` 的 CPU 控制器 PID 2131031 正等待 E30 全部固定评测完成；尚未开始 GPU 测速。实际模型、合成 G10 输入 smoke 与独占 GPU 检查通过后才计时。
