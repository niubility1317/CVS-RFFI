# Phase2 ProtoNet-CDA→MoPC 支持集数值诊断（GPU4）

- run_id：`20261108-phase2-protonet-mopc-support-gpu4-s2026092701-r01`
- group_id：`phase2-group2-protonet-mopc-nonfinite`；类别：`diagnostic`；阶段：`Phase2-B-C`
- 配置与矩阵：[experiment.json](experiment.json)；状态记录：[events.jsonl](events.jsonl)
- 当前登记状态：PLANNED（实际状态按events.jsonl及独立证据更新）

## 目的与对照

对 B r09 任务22的原源权重、K10/N20支持集和原GPU4单次检查梯度统计与非有限边界。

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

## 诊断范围与验证

仅在原 GPU4 上使用原 ProtoNet 源 checkpoint 和任务 22 的 K10/N20 支持集，按原 B/C 配方单次检查数值边界；保留原 33 份部分预测，不读取 query 或真值，不生成预测。记录器异常后才比较 FP32 原范数与 FP64 诊断范数，数值 guard 保持失败退出。5 项定向测试及独立 P0/P1、针对性复核通过。

## 实际运行与结果

原 GPU4 单次源和支持集诊断已完成。PID 1813443、start_ticks 63887344 的进程身份与 GPU 可见性已核实；真实源 checkpoint 的无 query smoke 通过，B 支持原型建立后 MoPC C 共完成 320 次更新。所检查边界和张量未发现非有限值，数值 guard 未触发，因此未获得故障时的 FP32／FP64 范数比较。原故障未复现，根因仍未知；一次正常运行不证明故障已修复。原 33 份部分预测保留，未读取 query 或真值，也未写预测。
