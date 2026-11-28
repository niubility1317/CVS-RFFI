# LoRa 与 WiSig 共同实增益及有界 R17 末步先验筛查

- run_id：`20261128-phase1-shared-boundedprior-gain-s2026112957-r01`
- group_id：`phase1-crossdata-shared-dura-realgain`；类别：`comparison`；阶段：`Phase1`
- 配置与矩阵：[experiment.json](experiment.json)；状态记录：[events.jsonl](events.jsonl)
- 当前登记状态：PLANNED（实际状态按events.jsonl及独立证据更新）

## 目的与对照

对可训练末步先验施加原 R17 波形 1% 均方根限幅，训练与推理使用同一限制。

## 数据、seed与模型来源

实际数据契约、权限例外、完整seed角色、checkpoint来源和选择规则见experiment.json。
逐行配置通过config_ref/resolved_config_ref定位；待补项必须在对应生命周期补齐。

## 执行与存储

命令、环境、CWD、commit、launch owner、输出和日志路径见experiment.json。
实际PID/GPU、读取时间、remote readback、失败或替代关系在此追加，并用record命令记录证据指针。

远端发布 VERIFIED：提交 `eb58403887570a2550fbdf83d4f25dca1289af98`，单归档本地/远端 SHA 一致，远端编译通过。控制器 PID 4068748，LoRa 4068769/GPU 6、WiSig 4068770/GPU 4；两侧真实权重源 U smoke 均通过，PID/CWD/GPU 和日志增长已核实。独立 P0/P1 审查未发现阻断缺陷；相关本地测试 6 项通过。

## 结果与覆盖

尚无结果。按预登记artifact逐项记录路径和缺项；保留每row与RX/day/TX/scene/K/seed的对应关系。
状态 RUNNING；逐包原 R17 波形均方根 1% 限制用于训练和推理，权重由各自源 U 教师指导获得；先前 WiSig 源 V 暴露与退化促成假设。后续自动固定预测并独立评分。不得用总索引或旧状态证明当前运行。

## 交接

记录已完成、当前run/commit、证据路径、阻塞与下一步；恢复先查原run，不重复启动。
