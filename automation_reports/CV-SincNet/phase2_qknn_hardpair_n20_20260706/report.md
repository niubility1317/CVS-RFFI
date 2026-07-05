# phase2_qknn_hardpair_n20_20260706

## Objective

在当前`qKNN`目标模式下，生成N20目标新类的`HP08 hard-pair`特征，用于验证十个新类扩展到二十个新类时是否还能保持稳定。该实验只保留`K=5`和`K=10`两个锚点，不扩大K数量。

## Hypothesis

已有十新类最好结果来自`NORM main view + HP08 aux view`。当前失败边界是：HP08目标域特征只覆盖十个`target_unknown`，不能作为二十新类证据。因此先导出真正的N20 HP08目标域特征；如果N20仍坍塌，问题应转向特征/注册质量，而不是继续添加qKNN标量网格。

## Protocol

| item | value |
|---|---|
| ground model | `ADV3B02_CORE90_SOFT_E200` |
| Stage2-C target receiver | `rx=7-14` |
| source receivers | `rx=0,1,2,3,4,5,6` |
| old TX | `14-10,14-7,20-15,20-19,6-15,8-20` |
| target-new TX count | `20` |
| target-new TX | `10-10,11-10,18-5,19-3,2-13,2-5,3-8,4-10,8-18,8-3,1-1,1-10,1-11,1-12,1-14,1-15,1-16,1-18,1-19,1-2` |
| LEO view | `leo_clear_weak,leo_low_elev_weak,leo_rain_weak` with `simplified_leo_residual` |
| export cap | `max_export_samples_per_tx=80` |
| K anchors | `K=10(query_per_class=70)`,`K=5(query_per_class=75)` |
| support/query rule | target-old and target-new support/query all from target receiver domain; query labels audit only |

The `proxy_unknown` list excludes all 20 target-new TX labels. This avoids target-new leakage into proxy hard-pair training.

## Local Changes

| file | purpose |
|---|---|
| `code/scripts/launch_phase2_qknn_hardpair_n20_v1.sh` | N20 HP08 hard-pair feature export plus strict `K=5,K=10` standalone qKNN sanity probes |
| `automation_reports/CV-SincNet/phase2_qknn_hardpair_n20_20260706/report.md` | experiment design, launch evidence, and result handoff |

## Verification Before Sync

| command | result |
|---|---|
| `bash -n code/scripts/launch_phase2_qknn_hardpair_n20_v1.sh` | PASS |
| local split guard: target-new vs proxy/old overlap check | PASS; `new_count=20`,`proxy_count=115`,`overlap=[]`,`old_overlap=[]` |

## N607 Launch Plan

| item | value |
|---|---|
| remote root | `/home/szu2070436088/2510044040/CV-SincNet` |
| remote script | `/home/szu2070436088/2510044040/CV-SincNet/code/scripts/launch_phase2_qknn_hardpair_n20_v1.sh` |
| run root | `/home/szu2070436088/2510044040/CV-SincNet/runs/phase2_qknn_hardpair_n20_20260706` |
| log root | `/home/szu2070436088/2510044040/CV-SincNet/logs/phase2_qknn_hardpair_n20_20260706` |
| GPU | prefer `GPU=5` if still idle at launch |
| server env | `/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python` |

Planned command:

```bash
cd /home/szu2070436088/2510044040/CV-SincNet
mkdir -p logs/phase2_qknn_hardpair_n20_20260706
nohup env GPU=5 PROFILE=HP08 RUN_ROOT=/home/szu2070436088/2510044040/CV-SincNet/runs/phase2_qknn_hardpair_n20_20260706 \
  bash code/scripts/launch_phase2_qknn_hardpair_n20_v1.sh \
  > logs/phase2_qknn_hardpair_n20_20260706/launch_HP08.out 2>&1 &
```

Expected outputs:

| artifact | path |
|---|---|
| HP08 N20 feature | `runs/phase2_qknn_hardpair_n20_20260706/MANYNEW20_HARDPAIR_HP08/ADV3B02_CORE90_SOFT_E200_PHASE1_HARDPAIR_HP08_N20/features_hardpair_HP08_n20.npz` |
| HP08 N20 clean feature | `runs/phase2_qknn_hardpair_n20_20260706/MANYNEW20_HARDPAIR_HP08/ADV3B02_CORE90_SOFT_E200_PHASE1_HARDPAIR_HP08_N20/features_clean_hardpair_HP08_n20.npz` |
| standalone K10 sanity | `runs/phase2_qknn_hardpair_n20_20260706/HP08/qknn_eval/n20_k10_coreproto_hardpair_HP08.csv` |
| standalone K5 sanity | `runs/phase2_qknn_hardpair_n20_20260706/HP08/qknn_eval/n20_k5_sourceguard_hardpair_HP08.csv` |

## Success Criteria

The launch itself is not success evidence. The target remains:

| scope | K | required |
|---|---:|---|
| ten new classes | 5,10 | every new class `>=75%` |
| more new classes | 5,10 | no collapse; next N20 comparison should approach the ten-class floor and not rely on larger K |
| K relation | 5 vs 10 | K=5 mean new accuracy not more than 5pp below K=10 |

## Risks

- The HP08 representation may still not separate dense `1-*` ManyTx families.
- Standalone HP08 qKNN is only a sanity probe. The main comparison after export must rerun dual-view qKNN with the existing N20 NORM feature and this new HP08 feature.
- Training uses active N607 resources. Before launch, recheck GPU occupancy and keep short-lived SSH/SCP only.

## Startup Retry 1

Initial remote launch:

| item | value |
|---|---|
| launch PID | `3205938` |
| GPU | `5` |
| status | failed during hard-pair parsing before training |
| log | `logs/phase2_qknn_hardpair_n20_20260706/launch_HP08.out` |

Failure:

```text
ValueError: cannot resolve hard pair TX token '1-1'
```

Cause: the N20 target-new list intentionally moved `1-1` and `1-18` out of `proxy_unknown`, but the inherited N10 hard-pair list still referenced these labels as proxy classes. This was a protocol guard, not a model failure.

Repair: removed only the four hard-pair entries containing target-new proxy labels:

```text
10-7:1-1,19-19:1-1,1-18:14-11,1-18:1
```

The repaired hard-pair list keeps 32 proxy-only entries and does not restore any target-new label to proxy training.

Retry launch:

| item | value |
|---|---|
| command log | `logs/phase2_qknn_hardpair_n20_20260706/launch_HP08_retry1.out` |
| wrapper PID | `3207361` |
| train PID | `3207363` |
| GPU | `5` |
| startup health | PASS |
| latest observed progress | epoch `10.0` logged |
| GPU5 state | `utilization=27%`,`memory.used=1481 MiB` |
| local SSH cleanup | no local `ssh.exe` process or established N607 SSH connection remained after the timed-out launch command |

Startup evidence:

```text
{"epoch": 10.0, "loss": 3.4078712005615235, ... "proxy_unknown_hard_pair": 0.0002745220424840227, "proxy_unknown_hard_old": 0.0014085482470691205}
```

The retry command itself timed out locally before returning a PID, but the subsequent short-lived checks confirmed the remote process is running and healthy. Continue monitoring via short SSH checks only.

## Completed Results

Remote run completed and artifacts were copied locally:

| artifact | local path |
|---|---|
| HP08 N20 feature | `E:\type10-7\automation_reports\CV-SincNet\phase2_qknn_hardpair_n20_20260706\artifacts\features_hardpair_HP08_n20.npz` |
| HP08-only K10 | `artifacts\n20_k10_coreproto_hardpair_HP08.json` |
| HP08-only K5 | `artifacts\n20_k5_sourceguard_hardpair_HP08.json` |
| NORM+HP08 fixed K10 | `artifacts\n20_k10_norm_hp08_dualview.json` |
| NORM+HP08 fixed K5 | `artifacts\n20_k5_norm_hp08_dualview.json` |
| NORM+HP08 adaptive v8 K10 | `artifacts\n20_k10_norm_hp08_adaptive_v8.json` |
| NORM+HP08 adaptive v8 K5 | `artifacts\n20_k5_norm_hp08_adaptive_v8.json` |

All rows use the maximum-query split available from the 80-sample-per-class feature file: `K=10` uses 70 query samples per class, and `K=5` uses 75 query samples per class.

Summary:

| route | K | old_acc | min_old | new_acc | min_new | raw support stored | verdict |
|---|---:|---:|---:|---:|---:|---:|---|
| HP08 only fixed | 10 | 79.52% | 64.29% | 74.14% | 42.86% | 0 | failed |
| HP08 only fixed | 5 | 77.33% | 66.67% | 66.47% | 42.67% | 0 | failed |
| NORM+HP08 fixed | 10 | 82.38% | 67.14% | 74.93% | 54.29% | 0 | failed |
| NORM+HP08 fixed | 5 | 80.00% | 69.33% | 70.60% | 42.67% | 0 | failed |
| NORM+HP08 adaptive v8 | 10 | 95.71% | 88.57% | 74.86% | 57.14% | 0 | failed |
| NORM+HP08 adaptive v8 | 5 | 97.11% | 92.00% | 74.20% | 57.33% | 0 | failed |

The adaptive v8 result is currently the best stability row because it keeps old classes high and keeps K=5 within 0.66pp of K=10 on mean new accuracy. It still fails the active goal because the new-class floor remains far below 75%.

Adaptive v8 per-class details:

| TX | K10 acc | K5 acc |
|---|---:|---:|
| `14-10` | 97.14% | 98.67% |
| `14-7` | 88.57% | 93.33% |
| `20-15` | 100.00% | 98.67% |
| `20-19` | 88.57% | 92.00% |
| `6-15` | 100.00% | 100.00% |
| `8-20` | 100.00% | 100.00% |
| `10-10` | 90.00% | 89.33% |
| `11-10` | 58.57% | 66.67% |
| `18-5` | 64.29% | 57.33% |
| `19-3` | 71.43% | 60.00% |
| `2-13` | 57.14% | 57.33% |
| `2-5` | 82.86% | 85.33% |
| `3-8` | 84.29% | 90.67% |
| `4-10` | 88.57% | 89.33% |
| `8-18` | 75.71% | 85.33% |
| `8-3` | 78.57% | 81.33% |
| `1-1` | 71.43% | 72.00% |
| `1-10` | 85.71% | 89.33% |
| `1-11` | 91.43% | 90.67% |
| `1-12` | 70.00% | 73.33% |
| `1-14` | 70.00% | 58.67% |
| `1-15` | 85.71% | 74.67% |
| `1-16` | 72.86% | 65.33% |
| `1-18` | 61.43% | 60.00% |
| `1-19` | 74.29% | 74.67% |
| `1-2` | 62.86% | 62.67% |

Interpretation:

- N20 HP08 feature export is valid and improves the N20 fixed dual-view floor from the previous NORM/HEAD baseline, but not enough to meet `min_new>=75%`.
- Adaptive v8 solves part of the stability issue: old classes no longer collapse, and K=5 is close to K=10 without a separate hand-tuned K policy.
- The remaining failure is concentrated in a repeatable hard subset: `11-10`,`18-5`,`2-13`,`1-14`,`1-18`,`1-2`, plus K5 degradation on `19-3`. This is now a representation/enrollment-quality problem, not simply a KNN storage or scalar scoring problem.
- Storage property remains aligned with the qKNN innovation requirement: no raw support samples are stored; the rows store compressed support codes, class prototypes, transform scalars, and small residual/logistic state.

Current goal status: active, not achieved.

## Aux Alignment Guard and Credible Baseline Reset

Follow-up inspection found that the archived `NORM+HP08` dual-view N20 rows were not sample-aligned:

| check | result |
|---|---:|
| NORM sample rows | 11,760 |
| HP08 sample rows | 11,760 |
| full metadata-key intersection (`tx/rx/day/eq/sig/role/scenario`) | 1,374 |
| source intersection | 2 |
| target-old intersection | 8 |
| target-unknown intersection | 698 |

Therefore the previous `NORM+HP08` dual-view rows are now treated as diagnostic-contaminated, not valid aligned multi-view evidence. The root cause is that HP08 export used a different export seed (`421900`) from the NORM/HEAD N20 export lineage. `features_n20_head.npz` was checked and is fully aligned with `features_n20_norm.npz`.

Code repair:

- `phase2_support_metric_qknn_probe.py` now requires auxiliary feature files to match the primary file on `tx_ids`,`dataset_role`,`sat_scenarios`,`rx_ids`,`channel_views`, and optional sample keys `day_ids`,`eq_ids`,`sig_ids`.
- `launch_phase2_qknn_hardpair_n20_v1.sh` now exposes `EXPORT_SEED`, defaulting to `4070391`, so HP08 can be re-exported on the same sample selection as the NORM N20 feature file.

Verification:

| command | result |
|---|---|
| `conda run -n ssr-gpu python -m py_compile code\scripts\phase2_support_metric_qknn_probe.py` | PASS |
| `bash -n code/scripts/launch_phase2_qknn_hardpair_n20_v1.sh` | PASS |
| NORM+misaligned-HP08 guard check | expected failure: `aux_feature_npz metadata mismatch for rx_ids` |

Credible rerun baselines after the guard:

| feature route | K | old_acc | min_old | new_acc | min_new | verdict |
|---|---:|---:|---:|---:|---:|---|
| NORM only adaptive v9 | 10 | 92.62% | 78.57% | 70.71% | 51.43% | failed |
| NORM+aligned HEAD adaptive v9 | 10 | 92.14% | 78.57% | 71.07% | 51.43% | failed |
| NORM only adaptive v9 | 5 | 92.22% | 78.67% | 68.67% | 42.67% | failed |
| NORM+aligned HEAD adaptive v9 | 5 | 91.78% | 77.33% | 68.80% | 44.00% | failed |

Interpretation:

- The true aligned N20 baseline is lower than the earlier contaminated NORM+HP08 rows.
- The aligned HEAD auxiliary view does not solve the many-new floor collapse.
- The next necessary experiment is an aligned HP08 re-export using `EXPORT_SEED=4070391`, followed by the same strict `K=10`/`K=5` maximum-query qKNN evaluation. This is a representation-quality repair attempt, not a K expansion.

Current goal status: active, not achieved.

## Post-v9 Floor Diagnostics

These diagnostics keep the K anchors fixed at `K=10` and `K=5`; no larger K setting was introduced. Query remains the maximum available split: 70 per class for `K=10`, 75 per class for `K=5`.

Artifacts:

| diagnostic | artifact |
|---|---|
| v9 K10 seed scan | `artifacts\n20_k10_norm_hp08_adaptive_v9_seedscan5.csv` |
| v9 K5 seed scan | `artifacts\n20_k5_norm_hp08_adaptive_v9_seedscan5.csv` |
| K10 transductive proto single point | `artifacts\n20_k10_v9_transductive_w006.json` |
| K10 query-proto refine single point | `artifacts\n20_k10_v9_qproto_w006.json` |
| K10 strong support-LOO grid | `artifacts\n20_k10_v8_loo_strong_grid.csv` |
| K5 strong support-LOO grid | `artifacts\n20_k5_v8_loo_strong_grid.csv` |
| K10 support-bias grid | `artifacts\n20_k10_v8_support_bias_grid.csv` |
| K10 class-diag metric grid | `artifacts\n20_k10_v8_classdiag_grid.csv` |

Seed scan summary:

| K | seed | old_acc | min_old | new_acc | min_new | weakest new classes |
|---:|---:|---:|---:|---:|---:|---|
| 10 | 421031 | 96.43% | 90.00% | 77.07% | 58.57% | `2-13` 58.57%,`11-10` 62.86%,`1-14` 64.29%,`19-3` 65.71%,`1-18` 67.14%,`1-2` 70.00% |
| 10 | 421033 | 95.71% | 87.14% | 74.50% | 51.43% | `2-13` 51.43%,`1-2` 58.57%,`1-18` 62.86%,`11-10` 62.86%,`1-12` 68.57%,`18-5` 68.57% |
| 5 | 421037 | 97.11% | 92.00% | 74.80% | 60.00% | `1-14` 60.00%,`1-18` 60.00%,`18-5` 60.00%,`19-3` 60.00%,`2-13` 60.00%,`1-2` 62.67% |
| 5 | 421038 | 94.22% | 86.67% | 69.60% | 37.33% | `2-13` 37.33%,`18-5` 45.33%,`19-3` 48.00%,`11-10` 50.67%,`1-18` 53.33%,`1-12` 58.67% |

The seed scan shows the collapse is not a single unlucky support draw. Even the best scanned floor is only 58.57% for `K=10` and 60.00% for `K=5`; worse seeds collapse much lower.

Mechanism checks:

| route | K | old_acc | min_old | new_acc | min_new | conclusion |
|---|---:|---:|---:|---:|---:|---|
| v9 baseline | 10 | 95.71% | 88.57% | 75.00% | 57.14% | reference |
| transductive proto `w=0.06` | 10 | 95.48% | 87.14% | 75.00% | 57.14% | no floor gain |
| query-proto refine `w=0.06` | 10 | 95.48% | 87.14% | 75.00% | 57.14% | no floor gain |
| strong support-LOO best | 10 | 95.71% | 88.57% | 75.57% | 61.43% | modest floor gain, still failed |
| strong support-LOO best | 5 | 97.11% | 92.00% | 74.80% | 60.00% | no gain beyond v9 |
| support-bias grid best | 10 | 95.71% | 88.57% | 74.86% | 57.14% | no floor gain |
| class-diag metric grid best | 10 | 95.48% | 87.14% | 75.00% | 57.14% | no floor gain; high storage cost |

The timed-out broader transductive grid produced no JSON/CSV evidence and is excluded. Its leftover local `conda`/`python` processes were stopped before later diagnostics.

Interpretation:

- Current qKNN compression and adaptive scoring are not the primary bottleneck in the N20 setting. Multiple compressed score-level repairs preserve old-class accuracy but cannot lift the repeated weak classes to 75%.
- The hard classes are stable across seed and mechanism checks: especially `2-13`,`1-18`,`1-2`,`18-5`, with K-specific weakness on `19-3`,`1-14`,`11-10`.
- Query-graph/transductive refinement does not repair the floor under the current feature geometry, so the next credible path is representation or enrollment-quality repair for these hard new classes, not larger K or more per-K tuning.

Current goal status: active, not achieved.

## Adaptive v9 Support-LOO Rescue

Objective: improve the N20 many-new qKNN route without adding more K anchors. The only anchors remain `K=10` and `K=5`, using the maximum query budget from the 80-sample-per-class feature file: `K=10` uses 70 query samples per class, and `K=5` uses 75 query samples per class.

Implementation change:

- Added `dualview_support_v9` / `stable_dualview_v9` in `code/scripts/phase2_support_metric_qknn_probe.py`.
- The v9 policy adaptively enables support-LOO pair rescue from support geometry: `support_hardness`, `class_load`, and `k_reliability`.
- Fixed adaptive parameter plumbing so support-LOO rescue values from the policy are actually passed into `_evaluate_metric_qknn`.
- Storage remains compressed: no raw support vectors are stored. v9 stores quantized support codes, class prototypes, transform scalars, residual/logistic scalars, and 32 support-LOO pair-rescue scalars.

Verification:

| command | result |
|---|---|
| `conda run -n ssr-gpu python -m py_compile code\scripts\phase2_support_metric_qknn_probe.py` | PASS |
| `n20_k10_norm_hp08_adaptive_v9.json` | completed |
| `n20_k5_norm_hp08_adaptive_v9.json` | completed |

The earlier large floor-grid attempt timed out before producing output files and is not used as evidence. The leftover local `conda`/`python` processes from that timed-out command were identified as the same diagnostic command and stopped before the v9 reruns.

v8 to v9 comparison:

| route | K | old_acc | min_old | new_acc | min_new | support-LOO pairs | stored support-LOO scalars | raw support stored | verdict |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| adaptive v8 | 10 | 95.71% | 88.57% | 74.86% | 57.14% | 0 | 0 | 0 | failed |
| adaptive v9 | 10 | 95.71% | 88.57% | 75.00% | 57.14% | 8 | 32 | 0 | failed |
| adaptive v8 | 5 | 97.11% | 92.00% | 74.20% | 57.33% | 0 | 0 | 0 | failed |
| adaptive v9 | 5 | 97.11% | 92.00% | 74.80% | 60.00% | 8 | 32 | 0 | failed |

Adaptive v9 per-class details:

| TX | K10 acc | K5 acc |
|---|---:|---:|
| `14-10` | 97.14% | 98.67% |
| `14-7` | 88.57% | 93.33% |
| `20-15` | 100.00% | 98.67% |
| `20-19` | 88.57% | 92.00% |
| `6-15` | 100.00% | 100.00% |
| `8-20` | 100.00% | 100.00% |
| `10-10` | 90.00% | 89.33% |
| `11-10` | 60.00% | 70.67% |
| `18-5` | 65.71% | 60.00% |
| `19-3` | 71.43% | 60.00% |
| `2-13` | 57.14% | 60.00% |
| `2-5` | 82.86% | 85.33% |
| `3-8` | 84.29% | 90.67% |
| `4-10` | 88.57% | 89.33% |
| `8-18` | 75.71% | 85.33% |
| `8-3` | 78.57% | 81.33% |
| `1-1` | 67.14% | 72.00% |
| `1-10` | 85.71% | 88.00% |
| `1-11` | 91.43% | 89.33% |
| `1-12` | 68.57% | 73.33% |
| `1-14` | 72.86% | 60.00% |
| `1-15` | 87.14% | 76.00% |
| `1-16` | 72.86% | 68.00% |
| `1-18` | 61.43% | 60.00% |
| `1-19` | 75.71% | 74.67% |
| `1-2` | 62.86% | 62.67% |

Interpretation:

- v9 improves mean new accuracy modestly and raises the K5 new-class floor from 57.33% to 60.00%, while preserving high old-class performance.
- v9 still fails the active floor target because the K10 minimum remains 57.14% and the K5 minimum remains 60.00%, both far below 75%.
- The main remaining weak classes are stable across v8/v9: `2-13`,`1-18`,`1-2`,`18-5`, with additional K-specific weakness on `19-3`,`1-14`,`1-16`. This points to representation/enrollment separability rather than a pure classifier-head storage issue.
- The current adaptive direction is still useful: it adds a support-derived rescue mechanism without per-K hand tuning, and it keeps `K=5` within 0.20pp of `K=10` on mean new accuracy. The next optimization should target hard-class representation or support selection quality, not larger K.

Current goal status: active, not achieved.
