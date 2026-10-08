# reference_response新增12行R5补测：clean及六residual

- run_id：`20261008-phase1-reference-completed-sixscene-manysig-m12-r01`
- group_id：`cvs-reference-completed-snapshot-sixscene`；类别：`cvs`；阶段：`Phase1`
- 配置与矩阵：[experiment.json](experiment.json)；状态记录：[events.jsonl](events.jsonl)
- 当前登记状态：PLANNED（实际状态按events.jsonl及独立证据更新）

## 目的与对照

固定完成时刻新增R5 base/daot/both各4seed，与已测试80行互斥。

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

## 即时补测

R5新完成base/daot/both共12行，固定各自E200/44400步，排除已评分80行。用户要求立即测试并放宽进程限制，本次仅临时增加最多4个推理worker，每卡额外最多1个、总进程最多3、启动前剩余显存至少12GB；原16个训练进程保持不变。全部12×7×168000预测固定后独立truth-last评分与复算。

## 重复发布已避免

发布器在本地未提交文件检查处退出，尚未SCP或启动。随后核实另一个聊天已发布相同12行评估 `20261008-evaluation-reference-r5-completed-manysig-m12-r01`，实际commit=cbc1fadab057be8620163d94cf2b8c7fabf289e8、PID=1466353。逐行完整配置一致；本run和release远端均不存在，状态VERIFIED_NOT_LAUNCHED。本预登记标记REPLACED，不再启动；直接复用另一launch owner的独立评估产物。原代码与预登记保留追溯。

## 共享评估完成与92行汇总

2026-10-08T18:20:02.150950+08:00：共享run新增12行评分ANALYZED，14112000次预测、1176条评分。累计92行、108192000次预测、9016条评分；本地全部混淆矩阵复算通过。训练仍按原源域选择规则运行，目标结果不回流。详见[92行完整结果](evidence/combined92/detailed_results_zh.md)。本run本身未启动。
