# Tri SSL source-U channel consistency single source receiver

## Fixed experiment

The user selected the single-source-receiver setting after historical target
results showed the largest ordinary PL advantage over original Tri among the
11 matched paper settings: 0.7062 percentage points averaged across seven
scenes, 0.6941 across six practical scenes, and at most 1.2144 in low suburban.
This is explicitly target-informed setting selection and is not independent
blind evidence. New test outcomes cannot adjust this fixed matrix.

Keep the full old three-view Tri SSL, fixed permutation, supervised objectives
and jointly trained default head. Each seed uses its own m1-natural source-only
scratch E20 and continues E10+E20. Four fixed source-U interventions use all
five actual branches, weight 0.05 inside the original U gradient guard: bounded
CFO phase ramp; 0.15 complex echo delayed one sample; the same echo delayed
three samples; and CFO ramp plus three-sample echo. Each uses three seeds
2026092701–03. No supervised-only variant or trained auxiliary classifier.

The CFO ramp changes phase by at most ±π/8 across 256 samples (about 6.1 kHz
at 25 MHz); echo delays are 40/120 ns, phase uniform on ±π, zero padded and
RMS normalized. Both parameters in the combined arm share one random draw.
These are exploratory synthetic settings, not deployment channel estimates;
CFO may carry TX identity, so invariance can be harmful. Under one source RX,
the original cross-RX positive pair and extrapolation are naturally unavailable;
we retain that original behavior instead of fabricating cross-RX partners.

Freeze all 12 new clean plus six practical predictions before independent
truth-side scoring. Reuse each own-setting, same-seed paper Tri and ordinary PL
and DURA-LU physical samples, then compare scenario accuracy, same-packet
corrected/broken counts, TX/RX/date, seed variation, U acceptance and resources.
The user target is an improvement exceeding 2 percentage points over PL in
each scene and stronger performance versus Tri, not an achieved result. PL
uses natural temporal order and a different training method; comparisons cannot
attribute the full difference to one component. No leo_weak or test-driven
recipe changes. N607 GPUs 0–7, only if no external observed/pending GPU worker, at most two own processes per card;
protect other owner jobs, preserve all outputs, systemic faults stop dispatch.
Other chat works on cross-day pseudo-label association and branch-gradient
protection. This run uses different CFO/multipath view consistency and its own
code, output root, process owner and resources; it never modifies their jobs.

Physical channel augmentation and contrastive RFF learning are prior work,
including [distance-invariant augmented learning](https://ojs.aaai.org/index.php/AAAI/article/view/37005)
and [residual-channel contrastive learning](https://arxiv.org/abs/2412.08885).
No novelty or general superiority is presumed.

The earlier r01 briefly shared GPUs with the other chat; its own dispatcher and
workers were stopped and verified absent, with outputs retained and no target
predictions. r02 admits a GPU only when no external CUDA process or recognized
pending experiment worker is present. This is an admission-time check and cannot
prevent another owner subsequently starting work on the same GPU.

Two r02 workers were temporarily paused when the other chat subsequently used
their GPUs. The user then explicitly allowed sharing with up to four experiment
processes per GPU; those two same workers were resumed after checking their
identity and current process counts. The running r02 dispatcher keeps its more
restrictive original admission configuration. Paused wall time must not be
reported as pure training cost.

- run_id：`20261111-phase1-pamf-tri-u-channels-m1-m12-r02`
- group_id：`phase1-trisig-unlabeled-channel-consistency-m1`；类别：`comparison`；阶段：`Phase1`
- 配置与矩阵：[experiment.json](experiment.json)；状态记录：[events.jsonl](events.jsonl)
- 当前登记状态：PLANNED（实际状态按events.jsonl及独立证据更新）

## 目的与对照

Four pre-fixed CFO/multipath U perturbations on single source RX; three seeds each. User selected setting after historical target exposure, where PL most exceeds Tri. Distinct from other owner cross-day and branch-gradient mechanisms; no target feedback for new matrix.

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
