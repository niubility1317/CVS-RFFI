# LORA全面验收两轮真实WiSig回归

- run_id：`20260930-diagnostic-lora-wisig-s392005-r02`
- group_id：`newclass-registration-source-acceptance`；类别：`diagnostic`；阶段：`external_execution_acceptance`
- 配置与矩阵：[experiment.json](experiment.json)；状态记录：[events.jsonl](events.jsonl)
- 当前登记状态：PLANNED（实际状态按events.jsonl及独立证据更新）

## 目的与对照

修复输入/评分/契约路径后完整2epoch功能回归；checkpoint重载预测一致，原论文不使用WiSig时不启动完整预算复现实验

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

范围：每阶段2epoch真实WiSig功能验收，非论文准确率或正式CVS对比。条件判断与40项测试见 paper_reproduction/newclass_registration_20260930/COMPREHENSIVE_ACCEPTANCE.md。
