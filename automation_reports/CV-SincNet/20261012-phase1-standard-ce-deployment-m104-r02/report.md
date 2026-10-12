# 十三模型八种子：端到端推理时延、内存显存与FP32模型存储实测

- run_id：`20261012-phase1-standard-ce-deployment-m104-r02`
- group_id：`phase1-standard-ce-deployment-cost`；类别：`diagnostic`；阶段：`Phase1 deployment diagnostic`
- 配置与矩阵：[experiment.json](experiment.json)；状态记录：[events.jsonl](events.jsonl)
- 当前登记状态：PLANNED（实际状态按events.jsonl及独立证据更新）

## 目的与对照

Frozen E200 real checkpoints; synthetic CPU IQ to CPU labels, no dataset/truth/prediction scoring. GPU0 sequential fixed matrix.

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

## 修复与固定范围

r01 在远端编译依赖导入时失败，尚未启动模型测量，release/archive 保留。本版以标准库读取 Linux /proc/self/statm 采样 RSS，避免安装软件；矩阵、权重、种子、时延与显存口径不变。原候选唯一 P0/P1 审查及换行转义定点复核通过，修复依赖后复用该检查范围；launcher 首步仍执行真实 checkpoint no-query smoke。
