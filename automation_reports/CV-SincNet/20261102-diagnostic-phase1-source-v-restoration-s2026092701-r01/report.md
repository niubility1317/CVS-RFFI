# Phase1 独立源验证集冻结恢复器诊断

- run_id：`20261102-diagnostic-phase1-source-v-restoration-s2026092701-r01`
- group_id：`phase1-source-v-frozen-restoration-diagnostic`；类别：`diagnostic`；阶段：`Phase1-source-validation`
- 配置与矩阵：[experiment.json](experiment.json)；状态记录：[events.jsonl](events.jsonl)
- 当前登记状态：PLANNED（实际状态按events.jsonl及独立证据更新）

## 目的与对照

从源 V 中不按标签抽取 4096 条，冻结 Tri-SIG-TIME SSL、DURA 和复数频谱候选，在 clean 与三个模拟星地视图上全部预测封存后独立连接源 V 真值，不拟合或更新任何模型。

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

发布前核对发现登记命令误写本机解释器路径，已改为 N607 CVS-RFFI Python；此前仅发布代码并编译，未启动源 V 诊断。
