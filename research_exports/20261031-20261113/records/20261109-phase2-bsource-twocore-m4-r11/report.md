# Phase2 B 主干：保存共享 B 状态并先计算两个 C 核心

- run_id：`20261109-phase2-bsource-twocore-m4-r11`
- group_id：`phase2-scalr-group2-da-native-cil-cross`；类别：`comparison`；阶段：`Phase2-B-C`
- 配置与矩阵：experiment.json；状态记录：events.jsonl
- 当前登记状态：PLANNED（实际状态按events.jsonl及独立证据更新）

## 目的与对照

Reuse 9677 complete ancestor tasks, retain 9 quarantines with 201 partials; run Orthogonal and LoRa for 1272 ProtoNet-CDA and 634 DADDA unstarted tasks, preserving each actual B model and MoPC anchor. MoPC remains pending; cancel 1932 unfinished RadioNet.

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

## 范围、共享 B 与屏障

仅对祖先链中 1,906 个完全未开始的非 RadioNet 任务执行 Orthogonal 和 LoRa，九处隔离及 9,677 个完整任务原样保留。每任务每 K 保存该次实际 B 模型、原型与 MoPC anchor；未来 MoPC 必须使用同一保存状态，不重新拟合 B 后冒充共享。子集结果以独立状态标记，MoPC 未完成、评分与全局预测屏障均关闭。原失败停止规则和每 GPU 最多 2 个进程保持。38 项定向测试、独立 P0/P1 及针对性复核通过；真实 source-only/no-query smoke 包括 B 状态保存与读回。

## Running observation

Real source-only smoke passed for six checkpoints; saved-B round trips passed for ProtoNet-CDA and DADDA. The two-core run dispatched, and 9 subset tasks were fixed at the observed checkpoint. The live process check found 9 workers, at most 2 per GPU, and no failed tasks. Saved B states and in-flight predictions were observed. MoPC remains pending; the full prediction and scoring barriers remain closed. Previous completed, quarantined, partial, and canceled RadioNet outputs remain preserved. Evidence: local_artifacts/phase2_group2_bnoradio_20261108/r11_running_verified.json.

## Two-core subset fixed; MoPC pending

All 1,906 previously unstarted tasks (1,272 ProtoNet-CDA and 634 DADDA) have fixed Orthogonal and LoRa predictions. Independent inventory verified 23,192 prediction files and 3,844 per-K saved B states, with the original 9,677 complete tasks, nine quarantines, and 1,932 canceled RadioNet tasks retained. The controller and workers exited without recorded failures. This is a partial result: MoPC is pending, previous failures remain unresolved, and neither the full prediction barrier nor scoring was opened. Evidence: local_artifacts/phase2_group2_bnoradio_20261108/r11_completion_inventory_verified.json and r11_2111_verified.json.
