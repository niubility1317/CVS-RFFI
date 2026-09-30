# LORA真实WiSig源代码完整链路验收

- run_id：`20260930-diagnostic-lora-wisig-s392005-r01`
- group_id：`newclass-registration-source-acceptance`；类别：`diagnostic`；阶段：`external_execution_acceptance`
- 配置与矩阵：[experiment.json](experiment.json)；状态记录：[events.jsonl](events.jsonl)
- 当前登记状态：PLANNED（实际状态按events.jsonl及独立证据更新）

## 目的与对照

固定作者逻辑、从零训练，各阶段1epoch，保留原始算法行为；执行验收而非论文准确率或正式CVS比较

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

本次范围：真实WiSig执行诊断，不等于原论文数值复现或正式CVS对比。使用本次scratch状态，没有外部权重继承。详细逻辑/修改见 paper_reproduction/newclass_registration_20260930/README.md。独立P0/P1审查及定点复审通过。
