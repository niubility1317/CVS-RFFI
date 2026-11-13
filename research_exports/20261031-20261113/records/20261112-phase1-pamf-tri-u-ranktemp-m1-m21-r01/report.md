# Tri SSL physical pseudo ranking temperature and branch

## Fixed experiment

Under the original Tri threshold and strict pseudoclass minimum quota, use
same-packet full and physically attenuated teacher logits to rank source-U
examples. The prior source run showed mean full/attenuated JS divergence around
`5e-5` to `7e-5`; temperature 2 or 4 may make differences between highly
confident logits observable. Compute JS after dividing both logits by the
registered temperature; the original pseudo class and threshold are unchanged.

Seven fixed arms: reference projection at temperatures 2 and 4; reference plus
frequency main and statistical projections at 2 and 4; reference plus PA main
projection at 1, 2 and 4. The PA statistical projection and `pa_delta` are not
attenuated. Each arm uses three seeds and the same-setting same-seed original
Tri E20, continues E10+E20, and retains the existing U gradient guard. Compare
same-seed frozen results with original Tri, natural-order PL and the preceding
strict-quota reference T1, reference+frequency T1, random and confidence
controls. The latter are from an earlier full matrix; all physical IDs, input
caches, model seeds, parent and network contracts must match before pairing.

Source-U labels and target IQ are never used for training; student and deployed
network stay intact. This ranking experiment differs from the other chat's
cross-day and contrastive objectives. The target setting and prior target
results were already exposed. These are development results, not independent
blind evidence, and the current matrix/checkpoints cannot be tuned by target
outcomes. Freeze all 21 seven-scene predictions before independent truth-side
scoring. Report every arm against Tri and PL plus same-packet corrected/broken,
TX/RX/date, source-U counts, time and failures. No leo_weak. The goal of more
than 2 percentage points above PL in each scene remains unmet.

[计算环境] GPUs 0–7 permit at most four observed or pending experiment processes per
card subject to free memory; protect other owners' healthy tasks. Systemic
technical failure stops new dispatch.

- run_id：`20261112-phase1-pamf-tri-u-ranktemp-m1-m21-r01`
- group_id：`phase1-trisig-physical-rank-temperature-m1`；类别：`comparison`；阶段：`Phase1`
- 配置与矩阵：experiment.json；状态记录：events.jsonl
- 当前登记状态：PLANNED（实际状态按events.jsonl及独立证据更新）

## 目的与对照

Seven fixed temperature and physical branch perturbation rankings at three seeds with matched prior selection controls. Historical target results exposed.

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
