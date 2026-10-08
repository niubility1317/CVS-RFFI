# R5新增完成12行立即补测：base、DAOT、DAOT＋RC4四seed

- run_id：`20261008-evaluation-reference-r5-completed-manysig-m12-r01`
- group_id：`cvs-reference-completed-snapshot-sixscene`；类别：`cvs`；阶段：`evaluation`
- 配置与矩阵：[experiment.json](experiment.json)；状态记录：[events.jsonl](events.jsonl)
- 当前登记状态：PLANNED（实际状态按events.jsonl及独立证据更新）

## 目的与对照

固定R5尚未测试的base/daot/both各四seed，合并既有rc4结果形成完整16行对照；无目标反馈。

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

## 固定测试范围

2026-10-08 18:08核实R5全部16行完成。其中rc4四seed已测试，本次仅补测base、daot、both四seed共12行。各模型固定E200/44400步且scratch来源匹配，全部84模型视图预测固定后独立truth-last评分。原32行修复组仍5行完成且已测试，没有新增可测试权重。

使用原fadf4be测试release的验证与推理逻辑；独立输出，不覆盖已有产物，不更改健康训练。无query真实权重smoke在每个模型推理前执行。结果合并既有rc4四seed，保留所有结果，不回流训练/选模。单个串行推理worker，不增加训练进程。
