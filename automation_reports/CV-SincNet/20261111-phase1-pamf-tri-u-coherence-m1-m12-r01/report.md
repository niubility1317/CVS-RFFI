# Tri SSL coherence-guided branch pseudo labels single source receiver

## Fixed experimental mechanism

Use complete old Tri-SIG-TIME SSL with its original jointly trained default
head. For accepted source-U packets, obtain the existing reference response's
normalized 20-sample repeat coherence ρ. For a selected U batch, create one
extra student feature view with reference-branch scale `0.25 + 0.75ρ`, keep
the original pseudo label, and apply its CE inside the original U gradient
guard. No U labels, target IQ, new classifier, or inference-time masks are used.
The signal statistic measures repeat similarity, not a calibrated SNR or a
guarantee that a particular TX feature is nuisance. PA and time paths remain
available. The selected source-U examples, coherence and actual attenuation
counts are recorded.

Four fixed arms, each seeds 2026092701–03, use the own-setting scratch E20 and
continue E10+E20: (1) adaptive reference scale, CE weight 0.25; (2) uniform
reference scale equal to the same batch's adaptive mean, weight 0.25; (3)
adaptive reference plus frequency scale, weight 0.25; (4) adaptive reference
scale, weight 1.0. These isolate example-dependent scaling, affected branches
and dose. This is different from the other chat's cross-day pseudo-label
association and branch-gradient certificate; output and ownership are separate.

The selected single-source-RX data setting follows exposed historical test
results and earlier candidate tests have also been exposed. Consequently this
is exploratory development, not a clean independent blind comparison. No result
from this matrix can alter its hyperparameters, final checkpoints, scheduled
seeds or scenarios. Freeze all 12 clean/six practical prediction sets before
independent truth-side scoring; compare each with same-packet, same-setting,
same-seed original Tri and ordinary PL DURA-LU. Report individual scenes and
corrected/broken packets, TX/RX/date, U acceptance, costs and failures. PL has
different training mechanism and natural temporal order, so whole-model
differences cannot be attributed solely to this intervention. Target is >2
percentage points over PL in each scene plus stronger Tri results; unachieved
until independently scored, and cannot establish other settings automatically.

This approach relates to [semi-supervised RF consistency regularization](https://arxiv.org/abs/2304.14795)
and [multi-head co-training](https://arxiv.org/abs/2107.04795), but uses the
existing fused head and no independently trained heads. These precedents and
the controls preclude claiming innovation without further evidence. Authorized
N607 GPUs 0–7 may share cards, at most four observed/pending experiment
processes per card and available-memory admission. Never stop or alter other
owner processes; systemic technical failure stops new dispatch.

- run_id：`20261111-phase1-pamf-tri-u-coherence-m1-m12-r01`
- group_id：`phase1-trisig-coherence-pseudo-label-m1`；类别：`comparison`；阶段：`Phase1`
- 配置与矩阵：[experiment.json](experiment.json)；状态记录：[events.jsonl](events.jsonl)
- 当前登记状态：PLANNED（实际状态按events.jsonl及独立证据更新）

## 目的与对照

Four preregistered source-U branch attenuation conditions use physical repeat coherence and existing head pseudo-label CE; includes matched mean uniform control, frequency ablation and stronger weight. Fixed after historical target exposure, no within-matrix test feedback.

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
