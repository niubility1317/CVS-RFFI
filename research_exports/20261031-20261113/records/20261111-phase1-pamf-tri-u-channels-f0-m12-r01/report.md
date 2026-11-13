# Tri SSL source-U channel consistency four arms

## Fixed methods and comparisons

Each arm uses the original full Tri-SIG-TIME SSL and the same-seed, same Main
contract scratch E20 checkpoint, followed by E10+E20. Only accepted source-U
packets enter a five-branch consistency term, weighted 0.05 within the original
U-gradient guard; all supervised losses, original U mechanism and default head
remain. Four arms, each with seeds 2026092701–03, are declared together:

1. A linearly varying phase over each 256-sample packet, bounded by ±π/8 from
   beginning to end (about ±6.1 kHz at 25 MHz).
2. A single complex echo of amplitude 0.15 and delay 1 sample (40 ns).
3. The same echo with delay 3 samples (120 ns).
4. The CFO perturbation plus the 3-sample echo.

The echo phase is uniform on ±π and output RMS is matched to the input; the
combined arm derives CFO and echo phase from the same random draw. Delays
use zero padding rather than wraparound. These are controlled synthetic stress
settings, not estimates of actual deployment channels. CFO can carry TX identity
as well as RX effects, so imposing invariance can harm identification. All arms
and seeds are retained, including negative cases; frequency/PA branches are
included rather than assumed invariant a priori. The parameters were fixed as
bounded exploratory values, not fitted to target data. Main split target results
from earlier work have been exposed; neither the resulting exploration nor this
test should be described as a new independent blind study.

Freeze all 12 × seven scene predictions before independent truth-side scoring.
Compare each with the matching historical Tri SSL and ordinary PL physical
packets, seeds and DURA-LU scenario, report scene accuracy and paired corrected
and missed packets, TX/RX/day, U acceptance, training runtime and failures.
Original Tri uses the same fixed temporal permutation; ordinary PL uses natural
order and a different learning mechanism, so this comparison is not an isolated
estimate of the perturbation. No `leo_weak` scenarios. The candidate matrix,
paths, command, environment and all GPU limits are in experiment.json. Stop new
dispatch on systemic technical failure and preserve healthy foreign tasks.

Physical channel augmentation and contrastive RFF learning are prior research,
including [distance-invariant augmented learning]([已省略链接])
and [residual-channel contrastive learning]([已省略链接]).
Frequency offset may itself be an identity feature as discussed by
[CFO-assisted RFF identification]([已省略链接]).
Our question is whether these perturbations help this full multi-branch SSL
under matched controls; no novelty or general superiority is presumed.

- run_id：`20261111-phase1-pamf-tri-u-channels-f0-m12-r01`
- group_id：`phase1-trisig-unlabeled-channel-consistency`；类别：`comparison`；阶段：`Phase1`
- 配置与矩阵：experiment.json；状态记录：events.jsonl
- 当前登记状态：PLANNED（实际状态按events.jsonl及独立证据更新）

## 目的与对照

Four predeclared channel perturbations: bounded CFO phase ramp, one-sample echo, three-sample echo, and CFO plus three-sample echo. Three matched model seeds each; consistent on all actual branches. Prior Main target outcomes exposed; no blind or test-driven selection claim.

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
