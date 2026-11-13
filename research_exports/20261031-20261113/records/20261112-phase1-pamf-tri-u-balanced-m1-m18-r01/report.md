# Tri SSL balanced physical pseudo selection

## Fixed experiment

The original Tri SSL generates source-U pseudo classes and applies its existing
confidence threshold. On each U batch, count only the accepted pseudo classes
that actually occur; for each such class keep up to the smallest count across
those present classes. The relaxed arm permits twice that minimum. A batch
with a single present pseudo class is not balanced, and missing pseudo classes
are never fabricated. This is an experimental sampling assumption, not a claim
that the true source or target class distribution is uniform.

Within each class, select by seeded random rank, full-view confidence, or the
Jensen–Shannon divergence between full and physically attenuated branch-view
predictions. Physical reference attenuation uses the fixed source-measured
repeat-coherence scale `1-clamp(100*(1-rho),0,0.75)`. Registered controls
compare adaptive with batch-mean uniform scaling, add frequency projections,
and compare relaxed and strict class caps. Every arm computes the same two
extra teacher forwards for equal compute, while only the chosen ranking changes
its acceptance. Random and confidence arms have the same per-batch class counts
as the strict physical arm for the same incoming pseudo labels; separate
training trajectories need not preserve the same counts.

Six arms each use seeds 2026092701–03 and continue matched source E20 to E50.
Keep the full network, Tri temporal order and original SSL gradient guard;
no target input, U identity truth or extra inference component is used. This
selection mechanism differs from the other chat's same-packet or per-branch
contrastive losses and cross-day pseudo relations. Balanced pseudo selection
and consistency-based filtering are related existing ideas; novelty and
physical usefulness require comparison with the controls.

Historical target results and this setting were exposed, so this is development
evidence rather than an independent blind test. Freeze all 18 models and all
seven-scene predictions before independent truth-side scoring. Compare matched
Tri and natural-order ordinary PL DURA-LU, each scene and seed, same-packet
corrected/broken, TX/RX/date, source U accepted class counts, runtime and
failures. The goal of more than 2 percentage points above PL in each scene
remains unmet. No leo_weak or target-driven tuning within this batch.

[计算环境] GPUs 0–7 allow at most four observed or pending experiment processes per
card, subject to free memory. Other owners' healthy tasks remain untouched;
systemic technical failures stop new dispatch.

- run_id：`20261112-phase1-pamf-tri-u-balanced-m1-m18-r01`
- group_id：`phase1-trisig-physical-balanced-pseudo-m1`；类别：`comparison`；阶段：`Phase1`
- 配置与矩阵：experiment.json；状态记录：events.jsonl
- 当前登记状态：PLANNED（实际状态按events.jsonl及独立证据更新）

## 目的与对照

Six fixed source-U pseudoclass balance and physical sensitivity ranking rules at three seeds, with random and confidence controls. Historical target results exposed.

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
