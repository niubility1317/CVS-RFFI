# Tri phase branch consistency receiver and label stability

## Fixed experiment

Each of receiver split 1, receiver split 2, 1% labeled and 5% labeled has three
matched seeds 2026092701–03. Continue the corresponding original paper Tri SSL
source-only scratch E20 through its own E10+E20 with the previously fixed
time/path/reference accepted-U common-phase consistency, weight 0.05. Use the
original default head, complete three-view shuffled architecture and original
SSL guard. Its choice follows exposed Main split results and is not blind.

Freeze all 12 candidate predictions over clean and six practical scenes before
independent truth-last scoring. Match each row to its own historical paper Tri
SSL and ordinary PL, including physical packet IDs, seed, data contract and
DURA-LU readout. PL used natural ordering and a different learning mechanism;
differences are not isolated phase effects. Include per-scene paired corrected
and missed packets, TX/RX/day, acceptance and measured resource cost. Do not use
results to alter this matrix. Preserve negative results and historical exposure.

N607 GPUs 0–7 are authorized, with at most two observed or pending processes per
card, free-memory admission and no interference with other owners. A systemic
technical fault stops new dispatch, without stopping healthy foreign tasks.
Command, environment, paths, exact rows and seed roles are in experiment.json;
release uses its Git HEAD, one archive transfer check and one remote compile.

- run_id：`20261111-phase1-pamf-tri-phase-complement-splits-m12-r02`
- group_id：`phase1-trisig-phase-complement-split-stability`；类别：`comparison`；阶段：`Phase1`
- 配置与矩阵：[experiment.json](experiment.json)；状态记录：[events.jsonl](events.jsonl)
- 当前登记状态：PLANNED（实际状态按events.jsonl及独立证据更新）

## 目的与对照

Fixed accepted source-U common-phase consistency on time/path/reference across two receiver partitions and two label fractions. Selected after exposed Main target results; no new setting target feedback.

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
