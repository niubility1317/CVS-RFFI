# Native comparison paired adaptation and registration

- run_id：`20260930-phase12-native-baselines-practical-m5-r01`
- group_id：`native-residual-noeq-exact-manysig-source-manytx-target-20260930`；类别：`comparison`；阶段：`Phase1+Phase2`
- 配置与矩阵：[experiment.json](experiment.json)；状态记录：[events.jsonl](events.jsonl)
- 当前登记状态：PLANNED（实际状态按events.jsonl及独立证据更新）

## 目的与对照

Nine distinct source model families, five fixed seeds, scratch final200; matched residual_noeq capsules and truth-last scoring. Native mechanisms retained; dataset and fixed-budget extensions disclosed.

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

## Actual dispatcher state

Status: `RUNNING`; owner: `codex/root/native-comparisons-20260930`; dispatcher PID: `98355`; release commit: `577825849c5c69edfbec9e258948b5621e4c643a`.

Counts: `{"QUEUED": 45}`. Per-row PID/GPU/log/config evidence: `/home/szu2070436088/2510044040/CV-SincNet/paper_reproduction/runs/20260930-phase12-native-baselines-practical-m5-r01/dispatcher/state.json` and experiment.json row runtime fields. Scoring: `{"p1": "WAITING", "p2": "WAITING"}`.
