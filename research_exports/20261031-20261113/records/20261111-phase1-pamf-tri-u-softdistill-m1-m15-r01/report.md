# Tri SSL branch-view soft distribution distillation

## Fixed experiment

The original Tri SSL continues its ordinary pseudo-label selection and U losses.
For selected source-U packets, a detached evaluation-mode full-network soft
distribution is compared with the prediction on the same original IQ with a
reference branch attenuation `1-clamp(100*(1-rho),0,0.75)`, derived from
the measurable repeat coherence. The temperature-2 KL loss uses only packets
accepted by the original Tri pseudo-label rule. Teacher and student see the same
IQ, and the identity control has near-zero KL. No target inputs or U labels are
used, and the extra branch masks do not exist in the deployed classifier.

Five predeclared controls each use seeds 2026092701–03: adaptive reference
weight 0.25 or 1, batch-mean uniform reference weight 1, identity weight 1,
and adaptive reference plus frequency weight 1. Each continues the matched
single-source-RX seed E20 for E10+E20, with the original E50 budget. Record
accepted packet counts, two additional evaluation forwards, scale and runtime.
This logit-distribution objective differs from the other chat's cross-day
relations, supervised-anchor certificates and fused-feature consistency; it
is a consistency/distillation candidate rather than a proven innovation.

The m1 target setting and previous results were exposed. This batch is
development evidence, not a fresh blind test. All 15 checkpoints and seven-view
predictions must finish before independent truth-side scoring. Compare matched
Tri SSL and ordinary PL on each view and seed, same-packet corrected/broken,
TX/RX/date and cost. Ordinary PL uses natural order and a different training
recipe. Greater than 2 percentage points over PL in each view remains an
unmet goal. No leo_weak or target-driven within-matrix tuning.

[计算环境] GPUs 0–7 admit at most four observed or pending experiment processes per
card and require sufficient free memory. Other owners' jobs stay untouched;
systemic technical failure stops further dispatch.

- run_id：`20261111-phase1-pamf-tri-u-softdistill-m1-m15-r01`
- group_id：`phase1-trisig-physical-softdistill-m1`；类别：`comparison`；阶段：`Phase1`
- 配置与矩阵：experiment.json；状态记录：events.jsonl
- 当前登记状态：PLANNED（实际状态按events.jsonl及独立证据更新）

## 目的与对照

Five fixed source-U same-IQ soft distribution controls at three seeds each. Historical target setting/results exposed. Different from the other chat feature consistency and cross-day branches.

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
