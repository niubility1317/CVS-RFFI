# 十三模型八种子：端到端推理时延、内存显存与FP32模型存储实测

- run_id：`20261012-phase1-standard-ce-deployment-m104-r01`
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

## 固定测量计划

13 种网络 × 8 个既有 E200 权重，共 104 个模型、208 个批大小测量。CPU IQ 到 CPU 标签的软件链路测量，不接触真实数据、query 或 truth，不重测准确率。固定矩阵、命令、来源、环境、GPU、输出和技术停止规则见 experiment.json。

所有模型复用已核实的 source-only scratch 训练数据契约；首步重新核对来源并执行真实权重 no-query smoke，成功后直接串行测量。已有协议负向检查 2 项通过；实际发布 Git OID 在 release_commit.txt 和 submit.json 记录。独立 P0/P1 审查发现远程脚本换行转义问题，已改为 raw string 并待针对性复核。
