# Phase2 MoPC 原 B 状态确定性双次支持诊断（GPU7）

- run_id：`20261109-phase2-protonet-mopc-savedb-deterministic-gpu7-s2026092702-r01`
- group_id：`phase2-group2-protonet-mopc-nonfinite`；类别：`diagnostic`；阶段：`Phase2-B-C`
- 配置与矩阵：experiment.json；状态记录：events.jsonl
- 当前登记状态：PLANNED（实际状态按events.jsonl及独立证据更新）

## 目的与对照

同一实际保存 B 和原支持集在确定性 CUDA 设置下执行两次诊断，比较状态和故障；仅诊断，不产生预测。

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

## 诊断结果

真实源检查通过；从同一已保存 B 状态及原支持集，在确定性 CUDA 设置下连续两次各完成 160 次更新，均未观察到非有限值，模型状态和原型逐项相同。原 r13 故障未复现，根因仍未知；确定性内核与正式运行设置不同，此结果不证明修复或与正式结果等价。启动记录指定物理 GPU7；进程过快退出，未及时独立读取其 GPU 可见性，因此该项为 UNKNOWN。原 10 份部分预测及 B 状态保留；诊断未读取 query/真值，也未生成预测。证据：local_artifacts/20261109-phase2-protonet-mopc-savedb-deterministic-gpu7-s2026092702-r01/completion_verified.json。
