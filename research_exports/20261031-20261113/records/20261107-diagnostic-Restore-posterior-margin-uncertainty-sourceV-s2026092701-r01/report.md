# 冻结识别边界方向上的源端后验不确定性

- run_id：`20261107-diagnostic-Restore-posterior-margin-uncertainty-sourceV-s2026092701-r01`
- group_id：`source-Gaussian-posterior-identity-margin-geometry`；类别：`diagnostic`；阶段：`Phase1`
- 配置与矩阵：experiment.json；状态记录：events.jsonl
- 当前登记状态：PLANNED（实际状态按events.jsonl及独立证据更新）

## 目的与对照

固定源 V120×7，复用已知模拟算子标量噪声的实际恢复 IQ 和分数。只求原冻结 F0 的 clean 输入导数与高斯协方差投影。

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

## 源端后验方向结果

低档城区在已知模拟算子和标量噪声的同一高斯模型下，剩余 IQ 方差占源先验 17.42%。投影到原 F0 clean 第一与第二名的分数差，剩余方差比例为 20.49%，汇总所有五个竞争类别为 19.90%。

完整分析
