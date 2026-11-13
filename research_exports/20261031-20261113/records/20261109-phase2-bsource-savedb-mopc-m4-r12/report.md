# Phase2 B 主干：从实际保存的 B 状态接续 MoPC

- run_id：`20261109-phase2-bsource-savedb-mopc-m4-r12`
- group_id：`phase2-scalr-group2-da-native-cil-cross`；类别：`comparison`；阶段：`Phase2-B-C`
- 配置与矩阵：experiment.json；状态记录：events.jsonl
- 当前登记状态：PLANNED（实际状态按events.jsonl及独立证据更新）

## 目的与对照

Run MoPC for the 1906 previously unstarted jobs from their exact r11 B anchors; preserve 9677 complete ancestors, nine quarantines with 201 partial predictions, and 1932 canceled RadioNet tasks. Nonfinite root cause remains unknown. First-nonfinite tracing preserves original method and failure guard.

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

## Failed; outputs retained

The controller stopped admission and drained after eight workers raised the old-support identity guard before any MoPC predictions. Each failed output directory and log is preserved, with zero prediction files; r11 saved B states and prior results remain untouched. Source inspection indicates the diagnostic helper returns support bank indices while the C loader returns physical sample IDs; this mismatch requires validation and a targeted repair. This failure does not establish the cause of earlier MoPC nonfinite gradients. Full prediction and scoring barriers remain closed. Evidence: local_artifacts/phase2_group2_bnoradio_20261108/r12_running_verified.json and r12_failure_detail_verified.json.
