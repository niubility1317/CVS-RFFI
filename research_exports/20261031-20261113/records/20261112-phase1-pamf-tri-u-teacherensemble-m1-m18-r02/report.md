# Tri SSL physical branch teacher ensemble

## Fixed experiment

The original Tri SSL retains its source-U pseudo-label acceptance, U views,
gradient guard, three-view time ordering and labeled objective. A separate
detached temperature-2 teacher target averages the full prediction and a
physical branch-attenuated prediction of each raw U IQ packet. The added KL
uses the same packet's first original Tri student view with ordinary additive
noise and training-mode dropout. It uses only originally accepted U packets;
neither U identity truth nor target IQ is used. The student retains the full
deployment network, and no trained teacher or mask is added at inference.

Six predeclared arms, each with seeds 2026092701–03, compare full-only teacher,
reference adaptive teacher, batch-mean uniform reference teacher, a stronger
reference weight, reference plus frequency projections, and reference plus
the PA main projection. Attenuation uses the fixed source-measured response
coherence scale `1-clamp(100*(1-rho),0,0.75)`. The PA arm scales `pa_proj`
only; `pa_stats_proj` and `pa_delta` remain unscaled, so it is not a full PA
branch ablation. The teacher mixture is 50/50 and the additional loss weight is
0.25 except for the registered weight-1 arm. Each continues its matched E20
for E10+E20. The earlier r01 registration was not launched and is superseded
because its PA-arm scope was ambiguous.

This differs from another chat's cross-day source-U relations, L-anchor
certificates and feature consistency, and from our previous student-masked KL:
only teacher logits use branch attenuation here; the student is the existing
unmasked noisy view. These are related consistency/self-distillation ideas,
not a proven novelty. Historical target results and this setting were exposed;
this batch supplies development evidence, not an independent blind test.
All 18 seven-scene predictions are fixed before independent truth-side scoring.
Compare matched Tri and ordinary PL DURA-LU by scene, seed, TX/RX/date and
same-packet corrected/broken; PL has different temporal order and recipe.
Greater than 2 percentage points over PL in each scene remains an unmet goal.
No leo_weak, target-driven tuning within the matrix, or selective test reruns.

[计算环境] GPUs 0–7 may share at most four observed or pending experiment processes
per card subject to free-memory admission. Other owner tasks remain untouched.
Systemic technical failure stops new dispatch. Track two extra teacher forwards,
accepted U exposure, branch scale, wall time and any failure.

- run_id：`20261112-phase1-pamf-tri-u-teacherensemble-m1-m18-r02`
- group_id：`phase1-trisig-physical-teacher-ensemble-m1`；类别：`comparison`；阶段：`Phase1`
- 配置与矩阵：experiment.json；状态记录：events.jsonl
- 当前登记状态：PLANNED（实际状态按events.jsonl及独立证据更新）

## 目的与对照

Six fixed source-U full and branch-view teacher ensemble rules at three seeds; prior target results exposed. Different from the other chat cross-day relations and feature consistency.

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
