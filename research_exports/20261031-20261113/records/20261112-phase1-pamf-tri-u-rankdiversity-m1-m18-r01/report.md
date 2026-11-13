# Tri SSL physical rank diversity

## Fixed experiment

Use the original Tri source-U pseudo classes and threshold. In each batch,
retain the minimum count among accepted pseudo classes that actually occur;
missing classes are not fabricated, and a batch with only one present class is
unchanged. Within each class rank the same packet's full-view and physically
attenuated-view logits by Jensen–Shannon divergence. Fixed policies select the
highest divergence, samples closest to the class median, or a mixture of the
two tails: `ceil(k/2)` lowest plus `floor(k/2)` highest. This tests whether
discarding physically sensitive packets loses transmitter-identifying signals;
neither sensitivity nor pseudo labels are assumed to be ground truth.

Six arms combine the three policies with reference-only or reference plus
frequency main/statistical projections; each uses seeds 2026092701–03, the same
setting/seed original Tri E20, and E10+E20 continuation. Full classifier,
temporal order, original pseudo threshold and gradient guard stay in place.
All branches are evaluated in the deployed network without new inference
machinery; the two extra branch-view teacher forwards are training only.
The prior strict-quota low-sensitivity reference and reference+frequency arms,
random and confidence rankings, and original Tri/PL are fixed comparators. All
comparisons must match input caches, physical IDs, model seed and source parent;
different trajectories need not have identical per-batch selection counts.

This class-wise pseudo selection differs from the other chat's same-packet
and per-branch contrastive objectives. Target data and prior target outcomes
were exposed; the present batch is development evidence, not an independent
blind test. Freeze all 18 models and seven-scene predictions before independent
truth-side scoring. Report all arms, same-packet corrected/broken, TX/RX/day,
U counts, runtime and failures. No target-driven tuning within the batch, no
leo_weak. The goal of over two percentage points above PL in every scene remains
unmet.

[计算环境] GPUs 0–7 permit no more than four observed or pending experiment processes
per card subject to free memory; preserve other owners' healthy tasks. Systemic
technical failure stops new dispatch.

- run_id：`20261112-phase1-pamf-tri-u-rankdiversity-m1-m18-r01`
- group_id：`phase1-trisig-physical-rank-diversity-m1`；类别：`comparison`；阶段：`Phase1`
- 配置与矩阵：experiment.json；状态记录：events.jsonl
- 当前登记状态：PLANNED（实际状态按events.jsonl及独立证据更新）

## 目的与对照

Six fixed source-U physical sensitivity quantile selections at three seeds with matched prior low-sensitivity and generic controls. Historical target results exposed.

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
