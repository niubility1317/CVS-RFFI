# Phase2 B 主干：修正支持身份后从已保存 B 接续 MoPC

- run_id：`20261109-phase2-bsource-savedb-mopc-m4-r13`
- group_id：`phase2-scalr-group2-da-native-cil-cross`；类别：`comparison`；阶段：`Phase2-B-C`
- 配置与矩阵：[experiment.json](experiment.json)；状态记录：[events.jsonl](events.jsonl)
- 当前登记状态：PLANNED（实际状态按events.jsonl及独立证据更新）

## 目的与对照

Correct bank-position to physical-ID comparison; preserve r12 eight pre-registration failure directories and logs with zero predictions. Resume 1906 MoPC tasks from their actual r11 B anchors; retain nine earlier science quarantines and canceled RadioNet. Nonfinite root cause remains unknown.

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

## Failed with new nonfinite observations

The support identity repair passed targeted tests and was observed to execute actual MoPC registrations from saved r11 B anchors. The controller drained after two new tasks failed; 14 MoPC-only tasks closed and the two failed tasks retain 10 partial prediction files each. For c0189 the first instrumented nonfinite boundary was support logits; for c0192 it was a batch-normalization running variance. Both subsequently had nonfinite gradient elements. These are first observed instrumented boundaries, not proven first mathematical operations or root causes. Existing r11 predictions/B states, r12 evidence, prior quarantines, and cancellation are retained. The full prediction and scoring barriers remain closed. Evidence: local_artifacts/phase2_group2_bnoradio_20261108/r13_failure_state_verified.json and r13_failure_detail_verified.json.
