# Tri SSL branch-view pseudo-label consensus

## Fixed experiment

Original Tri SSL supplies a pseudo label and its ordinary 0.95 acceptance on
source U. A second evaluation-mode prediction on the same packet attenuates
the reference branch (or reference and frequency branches) according to its
repeat-coherence-derived scale `1-clamp(100*(1-ρ),0,0.75)`. The new acceptance
is the intersection of original accepted samples with agreement between the
full and attenuated predictions, optionally requiring perturbed confidence at
least 0.95. No U label, target IQ, new classifier or deployment mask is used.
Original Tri U losses and gradient guard run on the filtered set.

Six fixed arms with seeds 2026092701–03 each: adaptive reference agreement,
batch-mean uniform reference agreement, adaptive reference agreement plus weak
0.95 confidence, mean-matched uniform reference confidence, adaptive reference
plus frequency confidence, and a random-count control. The last draws randomly
from its own raw-accepted batch with exactly the count its own adaptive
confidence gate would keep; trajectories of different arms need not have
identical counts. Each uses its same-setting same-seed original scratch E20 and
continues E10+E20. Track raw/retained U exposure, coherence, scaling and costs.

This is distinct from the other chat's supervised-anchor certificates,
cross-day pseudo-label relations, and fused-feature alignment: here only same-U
packet prediction agreement changes the original pseudo-label mask. It tests a
related consistency idea, not a proven novelty. The single-source-RX setting
and previous test results were exposed; this is development rather than an
independent blind test. All 18 models and seven-scene predictions are fixed
before independent truth-side scoring. Compare all scenes to matched Tri and
ordinary PL DURA-LU, report same-packet corrected/broken, TX/RX/date, seeds,
runtime and failures. PL uses natural order and a different training recipe.
The goal of >2 percentage points over PL in each scene is not yet established.

N607 GPUs 0–7 may share at most four observed or pending experiment processes
per card, subject to free-memory admission; other owner jobs remain untouched.
Systemic technical failures stop new dispatch. No leo_weak, target-driven
within-matrix tuning, selective runs, or additional trained classification head.

- run_id：`20261111-phase1-pamf-tri-u-consensus-m1-m18-r01`
- group_id：`phase1-trisig-physical-pseudo-consensus-m1`；类别：`comparison`；阶段：`Phase1`
- 配置与矩阵：[experiment.json](experiment.json)；状态记录：[events.jsonl](events.jsonl)
- 当前登记状态：PLANNED（实际状态按events.jsonl及独立证据更新）

## 目的与对照

Six fixed source-U pseudo-label acceptance rules compare full and branch-scaled predictions, including uniform and random-count controls, at three seeds each. Different from other chat cross-day branch association. Historical target setting/results exposed.

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
