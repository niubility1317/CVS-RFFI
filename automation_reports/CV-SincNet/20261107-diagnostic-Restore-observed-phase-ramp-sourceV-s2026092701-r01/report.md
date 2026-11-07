# 逐包相位与漂移的源端接收观测校正

- run_id：`20261107-diagnostic-Restore-observed-phase-ramp-sourceV-s2026092701-r01`
- group_id：`frozen-received-only-phase-ramp-calibration`；类别：`diagnostic`；阶段：`Phase1`
- 配置与矩阵：[experiment.json](experiment.json)；状态记录：[events.jsonl](events.jsonl)
- 当前登记状态：PLANNED（实际状态按events.jsonl及独立证据更新）

## 目的与对照

固定源 V120 七景。冻结 BF 盲估、相位边缘化、相位加漂移、已知模拟对照。盲估三路共用一次 Cholesky。

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

## 结果已核实

低档城区的 120 条源记录中，原盲估、相位校正、相位加漂移和已知模拟路径的 F(clean) 判决一致率依次为 81.67%、86.67%、76.67% 和 90.00%。两条校正的配对修复／损伤分别为 7／1 和 4／10。

[完整分析](analysis.md)

## 源端真实 TX 复算

同一 120 条源记录的四路七景结果全部冻结并完成独立核对后，另做原源 V 真值评分。低档城区整体相位校正从 98/120 提高到 104/120，净增 6 条真实 TX 正确，修复 7 条、损伤 1 条。这七条修复与原来的 clean 判决一致性修复逐条相同。clean 仍为 118/120，整体相位版七景正确数均至少持平。

[完整分析](analysis.md)
