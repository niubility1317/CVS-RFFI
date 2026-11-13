# 源端波形混合先验与原高斯配对比较

- run_id：`20261107-diagnostic-Restore-waveform-mixture-sourceV-s2026092701-r01`
- group_id：`source-moment-matched-waveform-prior`；类别：`diagnostic`；阶段：`Phase1`
- 配置与矩阵：experiment.json；状态记录：events.jsonl
- 当前登记状态：PLANNED（实际状态按events.jsonl及独立证据更新）

## 目的与对照

仅使用原 63000 条 L_s/U_s clean IQ 拟合四组。先验总体均值与协方差固定，分别在相同盲估和已知模拟 Action/噪声下对照。

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

## 源端结果已核实

固定源 V120 的低档城区，盲估信道下真实 TX 正确数由 98 升至 102，4 条修复、0 条损伤。已知模拟信道和噪声时，两种先验均为 107/120。同批 raw clean 为 118/120。

完整分析
