# 原始与 DURA 恢复 IQ 的冻结源域配对诊断

- run_id：`20261025-diagnostic-PAMF-DURA-paired-source-m3-r01`
- group_id：`phase1-pamf-restored-IQ-ssl-source-20261025`；类别：`cvs`；阶段：`Phase1`
- 配置与矩阵：[experiment.json](experiment.json)；状态记录：[events.jsonl](events.jsonl)
- 当前登记状态：PLANNED（实际状态按events.jsonl及独立证据更新）

## 目的与对照

三份原普通PL固定E30及同契约G-LU；完整源V比较原始、恢复、三种受损及对应恢复视图。仅诊断，不开展新的F/G优化或目标访问。

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

固定源域配对矩阵和最小发布证据见 [预登记](preregistration.md)。本 run 不包含新的 F/G 训练。

## 完整源诊断完成

状态：ANALYZED，外部执行 VERIFIED。完整 V27000×8视图×3冻结F、108000次G前向与真实 L smoke 完成，159.60秒。raw三教师共识精度99.968297%、覆盖93.459259%，源质量/利用率可行性通过。没有新的 F/G/分类头训练，未打开 U IQ/U真值/目标 IQ；参数与 buffers 不变。

全部端点、配对计数及解释见 [源域分析](source_analysis.md)、[执行证据](source_result_evidence.json)。科学执行commit为f75c8fcd039ecce460790ea9092bf443d66e66d6；本次后续收尾代码仅修状态覆盖方式，不更改该 run 的科学代码身份。原完整产物保存后出现终态 FileExistsError，已备份旧状态、同步已有完成状态并独立读回，未重跑前向；原日志保留。

该源诊断不代表最终性能目标完成。下一步仍按既定计划先分析合法源梯度剂量和恢复后弱扰动稳定性，再冻结新的训练矩阵。
