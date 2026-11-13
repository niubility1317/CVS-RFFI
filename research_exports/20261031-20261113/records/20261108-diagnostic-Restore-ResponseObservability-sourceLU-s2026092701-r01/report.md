# 源域微弱变化的可观察性与恢复响应

- run_id：`20261108-diagnostic-Restore-ResponseObservability-sourceLU-s2026092701-r01`
- group_id：`source-directional-observability-and-response`；类别：`diagnostic`；阶段：`Phase1`
- 配置与矩阵：experiment.json；状态记录：events.jsonl
- 当前登记状态：PLANNED（实际状态按events.jsonl及独立证据更新）

## 目的与对照

复用 224 条源 IQ 的已冻结响应，按同条模拟 Action 和标量噪声强度预设分箱，检查响应偏差与可观察性代理。

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

## 固定源响应与噪声代理分析完成

VERIFIED / ANALYZED。复用224条源L_s/U_s、原模拟Action/标量方差和三版冻结响应；七场景×四方向×三模型×四预定区间及all汇总共384行，以独立复数FIR公式复算一致。低档城区32条中各方向20条的作用能量/标量噪声代理小于1；不把该比例解释为真实身份可恢复上限。无新神经前向、TX标签、V/OE或目标读取。 完整分析
