# 恢复域监督剂量与恢复后轻噪声的源域可行性诊断

- run_id：`20261025-diagnostic-PAMF-DURA-dose-noise-m1-r02`
- group_id：`phase1-pamf-restored-IQ-ssl-source-20261025`；类别：`cvs`；阶段：`Phase1`
- 配置与矩阵：[experiment.json](experiment.json)；状态记录：[events.jsonl](events.jsonl)
- 当前登记状态：PLANNED（实际状态按events.jsonl及独立证据更新）

## 目的与对照

一个E30原普通PL学生在12个L/U源批次求完整原L+普通raw PL和新增raw/G软监督梯度；不更新参数。随后三冻结教师、完整源V检验40dB恢复后复噪声。

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

固定矩阵见 [预登记](preregistration.md)。r01收尾首批非持久buffer恢复失败；r02仅技术修复，阈值和科学矩阵不变。
