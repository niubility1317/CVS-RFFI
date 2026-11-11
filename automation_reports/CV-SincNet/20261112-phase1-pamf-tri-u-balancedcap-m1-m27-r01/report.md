# Tri SSL source-derived pseudoclass capacity study

## Fixed experiment

The previous source-U run showed similar total accepted pseudoclass counts but
strict per-batch balancing retained only about 81% of accepted sample exposures;
this batch investigates the source-observed cost of batch-count fluctuations.
For each batch, only pseudo classes present among originally accepted source-U
examples contribute to the minimum count. The class-wise cap is the floor of
that minimum times 1.125, 1.25 or 1.5. A single-present-class batch is unchanged;
missing classes are not fabricated and uniform true class frequency is not
assumed. Within each cap, the fixed comparison selects by seeded random,
original confidence, or reference-branch physical-view Jensen–Shannon divergence.

Nine combinations are each trained at seeds 2026092701–03, continuing the
same-setting same-seed original Tri E20 through E10+E20. All use the original
confidence gate, full multi-branch network, fixed time permutation and gradient
guard. Physical ranking uses source-observable response coherence to attenuate
only the reference projection for a detached teacher view; the student model
and deployed inference are unchanged. All arms execute the same two extra
teacher forwards; class counts are equal for different rankings only when the
incoming pseudo labels and acceptance are equal. Separate trajectories may
diverge. This is class-wise pseudo sample selection, distinct from the other
chat's same-packet and branch contrastive losses and cross-day relations.

The m1 target setting and previous results have been exposed; this batch is
development evidence and not an independent blind test. Freeze all 27 models
and all seven-scene predictions before independent scoring. Report all
registered combinations and controls, same-seed and same-packet DURA-LU
comparisons with Tri and natural-order PL, corrected/broken, TX/RX/day, U class
counts, time and failures. No target feedback is used to change the current
matrix or choose its frozen checkpoints. The goal of more than two percentage
points above PL in every scene remains unmet. No leo_weak.

N607 GPUs 0–7 allow up to four observed or pending experiment processes per card
subject to free memory; other owners' healthy tasks stay untouched. Stop new
dispatch on systemic technical failure.

- run_id：`20261112-phase1-pamf-tri-u-balancedcap-m1-m27-r01`
- group_id：`phase1-trisig-physical-balancedcap-m1`；类别：`comparison`；阶段：`Phase1`
- 配置与矩阵：[experiment.json](experiment.json)；状态记录：[events.jsonl](events.jsonl)
- 当前登记状态：PLANNED（实际状态按events.jsonl及独立证据更新）

## 目的与对照

Three source-derived relaxed pseudoclass caps, each with physical, confidence, and random ranking at three seeds. Prior target results exposed.

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
