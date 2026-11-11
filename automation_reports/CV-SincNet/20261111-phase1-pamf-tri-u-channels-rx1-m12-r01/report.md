# Tri SSL source-U channel consistency receiver split 1

## Fixed experiment

User selected receiver split 1 after the historical target scores showed the
largest ordinary PL advantage over Tri among the available settings: maximum
single-scene advantage 0.5208 percentage points, and mean across seven scenes
0.3111 points. This is explicitly target-informed setting selection, not blind
evidence or permission to adjust this run using target outcomes. Use the exact
RX1 source contract and original same-seed paper Tri SSL E20, then E10+E20.

Four arms and seeds 2026092701–03 are fixed before these candidate predictions:
bounded common CFO phase ramp; 0.15 complex echo delayed one sample; the same
echo delayed three samples; and ramp plus three-sample echo. Source-U accepted
packets get consistency on all five actual branches, weight 0.05 inside the
original U guard. No target label or target IQ is used in training, and the
full old three-view shuffled network, supervised mechanism and default jointly
trained head remain. The stopped Main run never reached target prediction.
The CFO ramp is ±π/8 total phase across 256 samples (about 6.1 kHz at 25 MHz);
echo delays 40/120 ns, phase uniform on ±π, zero padded and RMS normalized.
The combined arm derives both parameters from the same random draw. These are
exploratory synthetic parameters, not measured channel estimates. CFO can also
encode TX identity and its invariance may harm performance.

All 12 new seven-scene predictions must be frozen before independent truth-side
scoring. Match each seed to its own original Tri and ordinary PL RX1 model,
physical packets and DURA-LU views. Report per-scene accuracy and corrected /
broken packets, TX/RX/day, seed variability, U acceptance and resources. The
target is more than 2 percentage points over PL in **each** scene, alongside
improvement over Tri; the average cannot conceal any scene regression. This
target is not an achieved result. Ordinary PL uses natural temporal order and a
different training mechanism; don't attribute the whole difference to a single
component. No leo_weak; no target-driven recipe adjustment or selective rerun.

Physical channel augmentation and RFF contrastive learning are established
research, including [distance-invariant augmented learning](https://ojs.aaai.org/index.php/AAAI/article/view/37005)
and [residual-channel contrastive learning](https://arxiv.org/abs/2412.08885).
No novelty or general superiority is presumed.

- run_id：`20261111-phase1-pamf-tri-u-channels-rx1-m12-r01`
- group_id：`phase1-trisig-unlabeled-channel-consistency-rx1`；类别：`comparison`；阶段：`Phase1`
- 配置与矩阵：[experiment.json](experiment.json)；状态记录：[events.jsonl](events.jsonl)
- 当前登记状态：PLANNED（实际状态按events.jsonl及独立证据更新）

## 目的与对照

Four prior fixed unlabeled CFO/multipath perturbations on receiver split 1, three seeds each. Setting selected explicitly after historical target exposure because PL beats Tri most; new scores cannot select or modify this fixed matrix.

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
