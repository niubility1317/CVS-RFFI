# Phase2 B 主干：修正 K/N 格式后接续未启动 MoPC

- run_id：`20261109-phase2-bsource-savedb-mopc-m4-r15`
- group_id：`phase2-scalr-group2-da-native-cil-cross`；类别：`comparison`；阶段：`Phase2-B-C`
- 配置与矩阵：[experiment.json](experiment.json)；状态记录：[events.jsonl](events.jsonl)
- 当前登记状态：PLANNED（实际状态按events.jsonl及独立证据更新）

## 目的与对照

r14 exited before creating a run root, claim, worker or prediction because the same K/N points were compared as JSON lists vs in-memory tuples. Correct value comparison and continue 1890 unstarted MoPC tasks from r11 B; retain 14 completed and 11 quarantined B tasks, plus canceled RadioNet. Prior numerical failure remains unresolved; stop admission and drain on any failure.

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

## Retained numerical failure

After the user requested starting the remaining tasks despite the unresolved earlier issue, real source and saved-B smoke passed and 16 workers ran within two per GPU. Fifteen tasks closed (240 predictions) while protonet_cda-s2026092702-c0195 failed at K10/N20, MoPC stage 2 epoch 10 iteration 40. The first instrumented nonfinite boundary was support logits; loss, cross-entropy and 18 gradient tensors were nonfinite. Eleven partial predictions and the worker log are retained. The first observed boundary is not a proven mathematical root. The controller stopped admission and drained; 1874 jobs were unstarted. Existing full and partial results, earlier quarantines and canceled RadioNet remain. No scoring or global prediction barrier opened. Evidence: local_artifacts/phase2_group2_bnoradio_20261108/r15_failure_detail_verified.json and r15_inventory_verified.json.
