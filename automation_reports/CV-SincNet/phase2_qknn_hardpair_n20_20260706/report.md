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

## Adaptive v22 Load-Gated Query Cluster

Objective: continue qKNN optimization without adding K anchors. `K=5` and `K=10` remain the only anchors, using the maximum query split from the 80-sample-per-class feature file. A diagnostic mismatch was first corrected: the earlier current-code K5 legacy check accidentally used `pool_per_old=10,pool_per_new=10`; the valid maximum-query K5 protocol is `pool_per_old=5,pool_per_new=5`, leaving 75 query samples per class. With the corrected split, current code exactly reproduces the archived v15 K5 row.

Implementation change:

- Added `dualview_support_v22` / `stable_dualview_v22` in `code/scripts/phase2_support_metric_qknn_probe.py`.
- v22 keeps the compressed qKNN head: no raw support vectors are stored; persistent state is quantized support codes, class prototypes, transform scalars, residual/logistic scalars, support-proxy scalars, and support-LOO rescue scalars.
- v22 adds a temporary unlabeled-query cluster alignment term. Query clusters are batch-local and discarded after scoring. The persistent classifier still stores no query state.
- v22 adapts by support geometry rather than per-K hand tuning: low `k_reliability` enables class-balanced support-proxy selection; query-cluster weight is generated from `k_reliability`, `class_load`, and `stable_gate`; a high-load/low-K overload gate disables query-cluster injection for N20 K5 where it hurts floor.

Verification:

| command | result |
|---|---|
| `conda run -n ssr-gpu python -m py_compile code\scripts\phase2_support_metric_qknn_probe.py` | PASS |
| `n10_k5_v22_final_seed421037.csv` | completed |
| `n10_k10_v22_final_seed421029.csv` | completed |
| `n20_k5_v22_final_seed421037.csv` | completed |
| `n20_k10_v22_final_seed421029.csv` | completed |

Final maximum-query summary:

| scope | K | query/class | old_acc | min_old | new_acc | min_new | query-cluster rows | raw support stored | verdict |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| N10 | 5 | 75 | 91.56% | 77.33% | 86.00% | 64.00% | 750 | 0 | improves mean, floor still failed |
| N10 | 10 | 70 | 92.14% | 77.14% | 85.43% | 64.29% | 700 | 0 | improves mean, floor still failed |
| N20 | 5 | 75 | 92.22% | 78.67% | 69.33% | 48.00% | overload-gated | 0 | no regression vs v15, failed |
| N20 | 10 | 70 | 92.62% | 78.57% | 70.14% | 51.43% | gated out | 0 | no regression vs v15, failed |

Comparison to corrected v15:

| scope | K | v15 new_acc | v15 min_new | v22 new_acc | v22 min_new | delta |
|---|---:|---:|---:|---:|---:|---|
| N10 | 5 | 85.07% | 64.00% | 86.00% | 64.00% | +0.93pp mean, same floor |
| N10 | 10 | 85.29% | 64.29% | 85.43% | 64.29% | +0.14pp mean, same floor |
| N20 | 5 | 69.33% | 48.00% | 69.33% | 48.00% | same after overload fallback |
| N20 | 10 | 70.14% | 51.43% | 70.14% | 51.43% | same after gate fallback |

Per-class accuracy:

| TX | N10 K5 | N10 K10 | N20 K5 | N20 K10 |
|---|---:|---:|---:|---:|
| `10-10` | 90.67% | 91.43% | 80.00% | 84.29% |
| `11-10` | 76.00% | 78.57% | 60.00% | 55.71% |
| `18-5` | 94.67% | 95.71% | 50.67% | 62.86% |
| `19-3` | 97.33% | 95.71% | 56.00% | 62.86% |
| `2-13` | 64.00% | 64.29% | 49.33% | 51.43% |
| `2-5` | 84.00% | 81.43% | 81.33% | 78.57% |
| `3-8` | 88.00% | 91.43% | 86.67% | 80.00% |
| `4-10` | 88.00% | 87.14% | 88.00% | 88.57% |
| `8-18` | 84.00% | 77.14% | 82.67% | 70.00% |
| `8-3` | 93.33% | 91.43% | 77.33% | 72.86% |
| `1-1` | - | - | 69.33% | 57.14% |
| `1-10` | - | - | 86.67% | 84.29% |
| `1-11` | - | - | 85.33% | 85.71% |
| `1-12` | - | - | 66.67% | 62.86% |
| `1-14` | - | - | 48.00% | 68.57% |
| `1-15` | - | - | 76.00% | 80.00% |
| `1-16` | - | - | 65.33% | 70.00% |
| `1-18` | - | - | 48.00% | 57.14% |
| `1-19` | - | - | 68.00% | 72.86% |
| `1-2` | - | - | 61.33% | 57.14% |

Interpretation:

- v22 is a safe incremental qKNN improvement for N10: it raises mean new accuracy without reducing the weakest class and keeps K5 slightly above K10 on mean new accuracy.
- v22 does not solve the active target. The ten-class floor remains 64.00%/64.29%, below the 75% requirement.
- For N20, the overload gate correctly prevents query-cluster collapse, but the result remains at the v15 boundary. The many-new failure is dominated by separability of `1-14`,`1-18`,`2-13`,`11-10`,`18-5`,`19-3`,`1-2`, not by raw support storage or KNN extensibility.
- The next credible optimization is representation/enrollment repair for these hard ManyTx classes, or a support-selection protocol that improves per-class separability while preserving the same K=5/K=10 budget and no-query-label fitting rule.

Current goal status: active, not achieved.

## v21 Self-Gated Query Cluster Diagnostic

Objective: continue optimizing qKNN without adding K anchors. The only K settings remain `K=10` and `K=5`; query stays at the feature-file maximum, namely 70 query samples per class for `K=10` and 75 for `K=5`.

Implementation changes:

- Restored the legacy support-LOO pair rescue branch used by the earlier v15/v19 lineage when `support_loo_pair_rescue_proto_neighbors<=0` and `support_loo_pair_rescue_proto_min_sim>1.0`. This prevents the v20 risk-pair logic from contaminating legacy baselines.
- Added `dualview_support_v21` / `stable_dualview_v21` with self-gated query-cluster alignment. The method keeps only compressed support prototypes as persistent memory; query clusters are temporary batch-local state and are discarded after scoring.
- Added an unsupervised query-cluster gate: cluster scores are injected only when quota cluster assignment agrees sufficiently with the current qKNN score top-1 and has adequate score margin. No query labels are used for fitting or gating.
- Added a low-reliability support-LOO rescue gate for v21: when `k_reliability<0.15`, pair rescue is disabled so low-shot support mistakes are not amplified.

Verification:

| check | result |
|---|---|
| `conda run -n ssr-gpu python -m py_compile code\scripts\phase2_support_metric_qknn_probe.py` | PASS |
| N10 K10 v15 legacy check | restored historical row: old 92.14%, new 85.29%, min_new 64.29% |
| N10 K10 v21 self-gate | no query-cluster injection; same as restored v15 |
| N10 K5 v21 reliable gate | completed; still failed floor target |

Key result rows:

| route | K | query per class | old_acc | min_old | new_acc | min_new | weakest new classes | verdict |
|---|---:|---:|---:|---:|---:|---:|---|---|
| v15 legacy check | 10 | 70 | 92.14% | 77.14% | 85.29% | 64.29% | `2-13` 64.29%,`8-18` 75.71%,`11-10` 78.57% | baseline restored, target failed |
| v21 self-gate | 10 | 70 | 92.14% | 77.14% | 85.29% | 64.29% | `2-13` 64.29%,`8-18` 75.71%,`11-10` 78.57% | safe fallback, no gain |
| current v15 check | 5 | 75 | 91.78% | 77.33% | 81.33% | 53.33% | `2-13` 53.33%,`11-10` 56.00%,`2-5` 81.33% | current low-K legacy drift remains |
| v21 reliable gate | 5 | 75 | 91.78% | 77.33% | 82.67% | 56.00% | `2-13` 56.00%,`11-10` 69.33%,`2-5` 78.67% | small recovery, target failed |

Interpretation:

- The K10 baseline regression is repaired, so future K10 comparisons are again meaningful.
- The self-gated query-cluster idea is safe in this split because it refuses to inject unreliable query structure; however, it currently behaves as a fallback rather than an accuracy improver.
- The low-K failure is not solved. Current code still does not reproduce the archived `n10_k5_v15_lowshot_hybrid_seed421037.csv` row (`new_acc=85.07%`, `min_new=64.00%`). The mismatch is concentrated in automatic support-guided proxy pair selection and should be fixed before using v21 as a promotable result.
- Active goal remains unmet: ten-new-class floor is still below 75%, and N20 remains known to collapse well below the target under current compressed qKNN geometry.

Next technical route:

1. Recover the archived K5 v15 proxy-pair behavior or introduce a support-only reliability objective that chooses proxy bundles without query labels.
2. Re-run strict anchors after that repair: N10 K10, N10 K5, N20 K10, N20 K5.
3. Treat larger enrollment-pool support selection only as a separate diagnostic unless the label-budget claim is explicitly changed.

Current goal status: active, not achieved.

## v13 Adaptive Support-Proxy Direction Rescue

Objective: remove the manual `hard_focus` dependency from the useful qKNN proxy route. The new `dualview_support_v13` policy mines hard pairs from target support only: support leave-one-out errors plus high-similarity support prototype pairs. It then uses `proxy_unknown` class prototypes to generate compressed analogy directions. The deployed state stores only proxy-direction scalar metadata and class/prototype state; it does not store raw support samples.

Local implementation:

| file | change |
|---|---|
| `code/scripts/phase2_support_metric_qknn_probe.py` | added automatic support-LOO/prototype hard-pair mining and `dualview_support_v13` adaptive proxy weights |
| `code/scripts/phase2_support_guided_proxy_pair_miner.py` | generalized external hard-pair miner from old-only `hard_old` to arbitrary registered `hard_label`, including new-vs-new pairs |

Verification:

| command | result |
|---|---|
| `conda run -n ssr-gpu python -m py_compile code\scripts\phase2_support_metric_qknn_probe.py code\scripts\phase2_support_guided_proxy_pair_miner.py` | PASS |

Final v13 summary, maximum query split:

| scope | K | query per class | old_acc | min_old | new_acc | min_new | auto pairs | stored proxy scalars | verdict |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| N10 | 10 | 70 | 92.14% | 77.14% | 85.29% | 64.29% | 8 | 24 | failed target floor |
| N10 | 5 | 75 | 91.56% | 77.33% | 85.47% | 61.33% | 8 | 24 | failed target floor |
| N20 | 10 | 70 | 92.62% | 78.57% | 70.14% | 51.43% | 8 | 24 | failed target floor |
| N20 | 5 | 75 | 92.22% | 78.67% | 70.00% | 46.67% | 8 | 24 | failed target floor |

N10 v13 per-class details:

| TX | K10 correct/total | K10 acc | K5 correct/total | K5 acc |
|---|---:|---:|---:|---:|
| `14-10` | 67/70 | 95.71% | 72/75 | 96.00% |
| `14-7` | 56/70 | 80.00% | 58/75 | 77.33% |
| `20-15` | 70/70 | 100.00% | 74/75 | 98.67% |
| `20-19` | 54/70 | 77.14% | 58/75 | 77.33% |
| `6-15` | 70/70 | 100.00% | 75/75 | 100.00% |
| `8-20` | 70/70 | 100.00% | 75/75 | 100.00% |
| `10-10` | 64/70 | 91.43% | 68/75 | 90.67% |
| `11-10` | 55/70 | 78.57% | 55/75 | 73.33% |
| `18-5` | 67/70 | 95.71% | 72/75 | 96.00% |
| `19-3` | 67/70 | 95.71% | 72/75 | 96.00% |
| `2-13` | 45/70 | 64.29% | 46/75 | 61.33% |
| `2-5` | 57/70 | 81.43% | 64/75 | 85.33% |
| `3-8` | 64/70 | 91.43% | 66/75 | 88.00% |
| `4-10` | 61/70 | 87.14% | 66/75 | 88.00% |
| `8-18` | 53/70 | 75.71% | 62/75 | 82.67% |
| `8-3` | 64/70 | 91.43% | 70/75 | 93.33% |

N20 v13 per-class details:

| TX | K10 correct/total | K10 acc | K5 correct/total | K5 acc |
|---|---:|---:|---:|---:|
| `14-10` | 67/70 | 95.71% | 73/75 | 97.33% |
| `14-7` | 57/70 | 81.43% | 59/75 | 78.67% |
| `20-15` | 70/70 | 100.00% | 74/75 | 98.67% |
| `20-19` | 55/70 | 78.57% | 59/75 | 78.67% |
| `6-15` | 70/70 | 100.00% | 75/75 | 100.00% |
| `8-20` | 70/70 | 100.00% | 75/75 | 100.00% |
| `10-10` | 59/70 | 84.29% | 61/75 | 81.33% |
| `11-10` | 39/70 | 55.71% | 47/75 | 62.67% |
| `18-5` | 44/70 | 62.86% | 41/75 | 54.67% |
| `19-3` | 44/70 | 62.86% | 41/75 | 54.67% |
| `2-13` | 36/70 | 51.43% | 35/75 | 46.67% |
| `2-5` | 55/70 | 78.57% | 61/75 | 81.33% |
| `3-8` | 56/70 | 80.00% | 64/75 | 85.33% |
| `4-10` | 62/70 | 88.57% | 66/75 | 88.00% |
| `8-18` | 49/70 | 70.00% | 61/75 | 81.33% |
| `8-3` | 51/70 | 72.86% | 59/75 | 78.67% |
| `1-1` | 40/70 | 57.14% | 52/75 | 69.33% |
| `1-10` | 59/70 | 84.29% | 65/75 | 86.67% |
| `1-11` | 60/70 | 85.71% | 64/75 | 85.33% |
| `1-12` | 44/70 | 62.86% | 50/75 | 66.67% |
| `1-14` | 48/70 | 68.57% | 43/75 | 57.33% |
| `1-15` | 56/70 | 80.00% | 55/75 | 73.33% |
| `1-16` | 49/70 | 70.00% | 49/75 | 65.33% |
| `1-18` | 40/70 | 57.14% | 40/75 | 53.33% |
| `1-19` | 51/70 | 72.86% | 52/75 | 69.33% |
| `1-2` | 40/70 | 57.14% | 44/75 | 58.67% |

Interpretation:

- v13 improves deployability and removes hand-written hard-pair focus, but it does not meet the active target. The old-class target remains stable, while the new-class floor is still far below 75%.
- The automatic pair selector repeatedly locks onto one support-hard pair per split (`11-10->18-5` for N10 K10, `8-3->2-5` for N10 K5, `1-15->19-3` or `1-14->1-16` for N20), which helps some classes but leaves `2-13`, `11-10`, `18-5`, `19-3`, and several `1-*` classes under-separated.
- The best hand-guided proxy route remains a useful diagnostic signal: manually focusing `2-13` against hard competitors reached N10 K5 `min_new=72.00%`, but the current automatic selector has not yet recovered that behavior.
- Next route should make hard-pair selection class-floor aware without using query labels. A candidate is a support-only fairness objective that reserves at least one proxy-direction bundle for each low-confidence support class instead of allowing one pair to consume all `top_pairs`.

Current goal status: active, not achieved.

## v16/v17/v18 Proxy Bundle Diagnostics

Objective: test whether the remaining low-floor classes can be repaired by replacing single-pair selection with richer compressed proxy evidence. Three variants were checked:

| variant | mechanism | intent |
|---|---|---|
| v16 | per-class proxy bundles, 4 proxy directions per selected risk class | avoid v13 one-pair collapse while keeping multiple directions per class |
| v17 | support-proxy analogy scoring, no class balance | mimic the external hand-guided proxy miner's scoring rule |
| v18 | support-proxy analogy scoring plus class balance | combine hand-miner scoring with low-class coverage |

Verification:

| command | result |
|---|---|
| `conda run -n ssr-gpu python -m py_compile code\scripts\phase2_support_metric_qknn_probe.py` | PASS |

Support-prototype diagnostic:

| K | target class | nearest support competitors |
|---:|---|---|
| 10 | `2-13` | `11-10` 0.9591, `18-5` 0.9030, `3-8` 0.8884 |
| 5 | `2-13` | `11-10` 0.8874, `18-5` 0.7967, `3-8` 0.7855 |

This confirms that the manual successful route (`2-13` vs `11-10`) is visible from support prototypes; the remaining failure is candidate gating/scoring, not query-label leakage.

Maximum-query result summary:

| variant | scope | K | old_acc | new_acc | min_new | stored proxy scalars | verdict |
|---|---|---:|---:|---:|---:|---:|---|
| v16 | N10 | 10 | 92.14% | 83.57% | 57.14% | 48 | failed, worse than v15 |
| v16 | N10 | 5 | 91.56% | 85.33% | 62.67% | 48 | failed, worse than v15 K5 floor |
| v16 | N20 | 10 | 92.62% | 68.86% | 48.57% | 48 | failed, worse than v15 |
| v16 | N20 | 5 | 92.22% | 70.07% | 48.00% | 48 | failed, tied floor but larger state |
| v17 | N10 | 10 | 92.14% | 84.29% | 61.43% | 24 | failed |
| v17 | N10 | 5 | 91.56% | 85.33% | 61.33% | 24 | failed |
| v18 | N10 | 10 | 92.14% | 79.14% | 54.29% | 24 | failed, harmful |
| v18 | N10 | 5 | 91.56% | 83.47% | 56.00% | 24 | failed, harmful |
| v18 | N20 | 10 | 92.62% | 65.57% | 31.43% | 24 | failed, harmful |
| v18 | N20 | 5 | 92.22% | 70.20% | 45.33% | 24 | failed |

Interpretation:

- v16 shows that simply adding more compressed proxy directions is not enough. It increases stored state from 24 to 48 scalars and often hurts the floor.
- v17/v18 show that the hand-miner analogy score alone is not sufficient. The useful manual result came from selecting the right hard pair and weight together; automatic support-only scoring still over-selects harmful bundles.
- Current best automatic route remains v15: it gives a small K5 stability gain without K10 regression. The next meaningful change should add an explicit support-only validation gate that simulates candidate bundle application on support leave-one-out scores and rejects bundles that reduce support class floor or mean. Without that gate, proxy directions can be plausible geometrically but harmful at query time.

Current goal status: active, not achieved.

## v19 Support-LOO-Gated Proxy Check

Objective: test the support-only validation gate proposed after v16/v17/v18. The new policy `dualview_support_v19` still uses compressed support-proxy directions, but automatically generated proxy rows are accepted only if applying the row to support leave-one-out scores does not reduce support class floor or support mean. This keeps the route deployable: no raw support vectors are stored in the classifier head, and no query labels are used for gating.

Implementation change:

- Added `support_guided_proxy_gate` plus floor/mean tolerance arguments to `code/scripts/phase2_support_metric_qknn_probe.py`.
- Added `_gate_support_guided_proxy_rows(...)` to validate candidate proxy rows on support leave-one-out scores before query scoring.
- Added `dualview_support_v19` / `stable_dualview_v19` adaptive policies.
- Added CSV fields for gate enablement and support-LOO before/after floor/mean.

Verification:

| command | result |
|---|---|
| `conda run -n ssr-gpu python -m py_compile code\scripts\phase2_support_metric_qknn_probe.py` | PASS |
| `n10_k10_v19_gated_proxy_seed421029.csv` | completed |
| `n10_k5_v19_gated_proxy_seed421037.csv` | completed |
| `n20_k10_v19_gated_proxy_seed421029.csv` | completed |
| `n20_k5_v19_gated_proxy_seed421037.csv` | completed |

Maximum-query v19 summary:

| scope | K | query per class | old_acc | min_old | new_acc | min_new | gate support floor before->after | accepted proxy rows | proxy pairs | verdict |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|---|
| N10 | 10 | 70 | 92.14% | 77.14% | 84.86% | 65.71% | 89.38%->89.38% | 1 | `11-10->18-5` | failed target floor |
| N10 | 5 | 75 | 91.56% | 77.33% | 85.60% | 61.33% | 72.50%->76.25% | 8 | `8-3->2-5` repeated | failed, worse than v15 floor |
| N20 | 10 | 70 | 92.62% | 78.57% | 70.14% | 51.43% | 66.15%->66.15% | 1 | `1-15->19-3` | failed, tied v15 floor |
| N20 | 5 | 75 | 92.22% | 78.67% | 70.07% | 48.00% | 54.62%->55.38% | 1 | `1-14->1-16` | failed, tied v15 floor |

N10 v19 per-class details:

| TX | K10 correct/total | K10 acc | K5 correct/total | K5 acc |
|---|---:|---:|---:|---:|
| `14-10` | 67/70 | 95.71% | 72/75 | 96.00% |
| `14-7` | 56/70 | 80.00% | 58/75 | 77.33% |
| `20-15` | 70/70 | 100.00% | 74/75 | 98.67% |
| `20-19` | 54/70 | 77.14% | 58/75 | 77.33% |
| `6-15` | 70/70 | 100.00% | 75/75 | 100.00% |
| `8-20` | 70/70 | 100.00% | 75/75 | 100.00% |
| `10-10` | 64/70 | 91.43% | 68/75 | 90.67% |
| `11-10` | 55/70 | 78.57% | 55/75 | 73.33% |
| `18-5` | 65/70 | 92.86% | 72/75 | 96.00% |
| `19-3` | 67/70 | 95.71% | 72/75 | 96.00% |
| `2-13` | 46/70 | 65.71% | 46/75 | 61.33% |
| `2-5` | 56/70 | 80.00% | 64/75 | 85.33% |
| `3-8` | 64/70 | 91.43% | 66/75 | 88.00% |
| `4-10` | 61/70 | 87.14% | 67/75 | 89.33% |
| `8-18` | 53/70 | 75.71% | 62/75 | 82.67% |
| `8-3` | 63/70 | 90.00% | 70/75 | 93.33% |

N20 v19 per-class details:

| TX | K10 correct/total | K10 acc | K5 correct/total | K5 acc |
|---|---:|---:|---:|---:|
| `14-10` | 67/70 | 95.71% | 73/75 | 97.33% |
| `14-7` | 57/70 | 81.43% | 59/75 | 78.67% |
| `20-15` | 70/70 | 100.00% | 74/75 | 98.67% |
| `20-19` | 55/70 | 78.57% | 59/75 | 78.67% |
| `6-15` | 70/70 | 100.00% | 75/75 | 100.00% |
| `8-20` | 70/70 | 100.00% | 75/75 | 100.00% |
| `10-10` | 59/70 | 84.29% | 61/75 | 81.33% |
| `11-10` | 39/70 | 55.71% | 48/75 | 64.00% |
| `18-5` | 44/70 | 62.86% | 41/75 | 54.67% |
| `19-3` | 44/70 | 62.86% | 41/75 | 54.67% |
| `2-13` | 36/70 | 51.43% | 36/75 | 48.00% |
| `2-5` | 55/70 | 78.57% | 61/75 | 81.33% |
| `3-8` | 56/70 | 80.00% | 64/75 | 85.33% |
| `4-10` | 62/70 | 88.57% | 66/75 | 88.00% |
| `8-18` | 49/70 | 70.00% | 61/75 | 81.33% |
| `8-3` | 51/70 | 72.86% | 59/75 | 78.67% |
| `1-1` | 40/70 | 57.14% | 52/75 | 69.33% |
| `1-10` | 59/70 | 84.29% | 65/75 | 86.67% |
| `1-11` | 60/70 | 85.71% | 64/75 | 85.33% |
| `1-12` | 44/70 | 62.86% | 50/75 | 66.67% |
| `1-14` | 48/70 | 68.57% | 43/75 | 57.33% |
| `1-15` | 56/70 | 80.00% | 55/75 | 73.33% |
| `1-16` | 49/70 | 70.00% | 48/75 | 64.00% |
| `1-18` | 40/70 | 57.14% | 40/75 | 53.33% |
| `1-19` | 51/70 | 72.86% | 52/75 | 69.33% |
| `1-2` | 40/70 | 57.14% | 44/75 | 58.67% |

Interpretation:

- v19 is useful as a safety gate, not as the current best route. It prevents some support-LOO degradation and records whether accepted proxy rows are support-consistent, but the query floor remains dominated by `2-13` and the dense `1-*` groups.
- The gate is not sufficient because support-LOO improvement does not reliably transfer to query floor. N10 K5 is the clearest example: support floor rises from 72.50% to 76.25%, while query floor stays at 61.33%.
- Current best automatic route remains v15 for the active route family. v19 should be kept as a diagnostic/safety component, but the next improvement must change the candidate objective toward the repeatedly weak classes rather than validating the same candidate pool.

Current goal status: active, not achieved.

## v20 Risk-Covered Pair Rescue Diagnostics

Objective: test whether the repeatedly weak classes can be covered more explicitly without storing raw support samples and without adding new K anchors. The policy `dualview_support_v20` keeps the same `K=10` and `K=5` maximum-query protocol, disables the harmful v19 proxy rows, and expands support-LOO pair rescue with a support-only class-risk score plus prototype-neighbor candidate discovery.

Implementation change:

- Extended support-LOO pair rescue to score candidate pairs from three support-only sources: actual support-LOO mistakes, support-score runner-up labels, and nearest support-class prototypes.
- Added a class-risk term from support-LOO class accuracy, support margin, and prototype similarity, then filtered directions where a lower-risk class would subtract from a higher-risk class.
- Added `support_loo_pair_rescue_proto_neighbors`, `support_loo_pair_rescue_proto_min_sim`, and `support_loo_pair_rescue_pairs` fields so the accepted compressed pair bundle is auditable.
- Added `dualview_support_v20` / `stable_dualview_v20` adaptive policies. The policy derives pair count, prototype threshold, and rescue weight from `new_count`, support-size reliability, and class-load; no per-K parameter table is introduced.

Verification and artifacts:

| item | result |
|---|---|
| `conda run -n ssr-gpu python -m py_compile code\scripts\phase2_support_metric_qknn_probe.py` | PASS |
| `artifacts\n10_k10_v20_riskpair_seed421029.csv` | completed |
| `artifacts\n10_k5_v20_riskpair_seed421037.csv` | completed |
| `artifacts\n20_k10_v20_riskpair_seed421029.csv` | completed |
| `artifacts\n20_k5_v20_riskpair_seed421037.csv` | completed |

Maximum-query v20 summary:

| scope | K | query per class | old_acc | min_old | new_acc | min_new | pair count | pair weight | verdict |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| N10 | 10 | 70 | 92.14% | 77.14% | 84.86% | 62.86% | 10 | 0.0822 | failed,worse than v15/v19 floor |
| N10 | 5 | 75 | 91.56% | 77.33% | 85.47% | 58.67% | 10 | 0.0482 | failed,worse than v15/v19 floor |
| N20 | 10 | 70 | 92.62% | 78.57% | 70.79% | 52.86% | 13 | 0.0933 | failed,small floor gain only |
| N20 | 5 | 75 | 92.22% | 78.67% | 69.60% | 48.00% | 13 | 0.0600 | failed,tied v15 floor |

N10 v20 per-class details:

| TX | K10 correct/total | K10 acc | K5 correct/total | K5 acc |
|---|---:|---:|---:|---:|
| `14-10` | 67/70 | 95.71% | 72/75 | 96.00% |
| `14-7` | 56/70 | 80.00% | 58/75 | 77.33% |
| `20-15` | 70/70 | 100.00% | 74/75 | 98.67% |
| `20-19` | 54/70 | 77.14% | 58/75 | 77.33% |
| `6-15` | 70/70 | 100.00% | 75/75 | 100.00% |
| `8-20` | 70/70 | 100.00% | 75/75 | 100.00% |
| `10-10` | 64/70 | 91.43% | 68/75 | 90.67% |
| `11-10` | 52/70 | 74.29% | 54/75 | 72.00% |
| `18-5` | 67/70 | 95.71% | 72/75 | 96.00% |
| `19-3` | 67/70 | 95.71% | 72/75 | 96.00% |
| `2-13` | 44/70 | 62.86% | 44/75 | 58.67% |
| `2-5` | 57/70 | 81.43% | 65/75 | 86.67% |
| `3-8` | 64/70 | 91.43% | 66/75 | 88.00% |
| `4-10` | 61/70 | 87.14% | 67/75 | 89.33% |
| `8-18` | 54/70 | 77.14% | 63/75 | 84.00% |
| `8-3` | 64/70 | 91.43% | 70/75 | 93.33% |

N20 v20 per-class details:

| TX | K10 correct/total | K10 acc | K5 correct/total | K5 acc |
|---|---:|---:|---:|---:|
| `14-10` | 67/70 | 95.71% | 73/75 | 97.33% |
| `14-7` | 57/70 | 81.43% | 59/75 | 78.67% |
| `20-15` | 70/70 | 100.00% | 74/75 | 98.67% |
| `20-19` | 55/70 | 78.57% | 59/75 | 78.67% |
| `6-15` | 70/70 | 100.00% | 75/75 | 100.00% |
| `8-20` | 70/70 | 100.00% | 75/75 | 100.00% |
| `10-10` | 60/70 | 85.71% | 61/75 | 81.33% |
| `11-10` | 40/70 | 57.14% | 45/75 | 60.00% |
| `18-5` | 44/70 | 62.86% | 39/75 | 52.00% |
| `19-3` | 48/70 | 68.57% | 40/75 | 53.33% |
| `2-13` | 37/70 | 52.86% | 36/75 | 48.00% |
| `2-5` | 56/70 | 80.00% | 62/75 | 82.67% |
| `3-8` | 56/70 | 80.00% | 64/75 | 85.33% |
| `4-10` | 61/70 | 87.14% | 66/75 | 88.00% |
| `8-18` | 49/70 | 70.00% | 61/75 | 81.33% |
| `8-3` | 50/70 | 71.43% | 58/75 | 77.33% |
| `1-1` | 38/70 | 54.29% | 54/75 | 72.00% |
| `1-10` | 59/70 | 84.29% | 65/75 | 86.67% |
| `1-11` | 64/70 | 91.43% | 65/75 | 86.67% |
| `1-12` | 39/70 | 55.71% | 53/75 | 70.67% |
| `1-14` | 48/70 | 68.57% | 40/75 | 53.33% |
| `1-15` | 59/70 | 84.29% | 54/75 | 72.00% |
| `1-16` | 49/70 | 70.00% | 46/75 | 61.33% |
| `1-18` | 41/70 | 58.57% | 37/75 | 49.33% |
| `1-19` | 52/70 | 74.29% | 52/75 | 69.33% |
| `1-2` | 41/70 | 58.57% | 46/75 | 61.33% |

Interpretation:

- v20 is a useful diagnostic and instrumentation step, but it is not the current best route. It raises N20 K10 floor from the v15/v19 51.43% boundary to 52.86%, but harms N10 and does not improve the N20 K5 48.00% floor.
- The accepted pair bundles are now auditable, but support-risk coverage alone still fails to identify query-transfer-safe corrections. The repeatedly weak classes remain `2-13`,`1-18`,`1-2`,`11-10`,`18-5`, and several dense `1-*` classes.
- Current best automatic route remains v15 for this route family. v20 should be kept as a compressed, support-only diagnostic component, not promoted as the deployment classifier head.

Current goal status: active, not achieved.

## v14/v15 Class-Floor-Aware Support-Proxy Check

Objective: address the v13 failure mode where all automatic proxy directions collapse onto one support-hard pair. v14 adds class-balanced round-robin selection over support-only hard-pair candidates. v15 keeps the same compressed proxy-direction representation but makes the balance gate adaptive: low-shot support (`adaptive_k_reliability<0.25`) uses class-balanced selection; higher-reliability support keeps v13's concentrated bundle.

Verification:

| command | result |
|---|---|
| `conda run -n ssr-gpu python -m py_compile code\scripts\phase2_support_metric_qknn_probe.py` | PASS |

v15 maximum-query summary:

| scope | K | query per class | old_acc | min_old | new_acc | min_new | balance gate | stored proxy scalars | verdict |
|---|---:|---:|---:|---:|---:|---:|---|---:|---|
| N10 | 10 | 70 | 92.14% | 77.14% | 85.29% | 64.29% | off | 24 | failed target floor |
| N10 | 5 | 75 | 91.56% | 77.33% | 85.07% | 64.00% | on | 24 | failed target floor |
| N20 | 10 | 70 | 92.62% | 78.57% | 70.14% | 51.43% | off | 24 | failed target floor |
| N20 | 5 | 75 | 92.22% | 78.67% | 69.33% | 48.00% | on | 24 | failed target floor |

Comparison against v13:

| scope | K | v13 min_new | v15 min_new | delta |
|---|---:|---:|---:|---:|
| N10 | 10 | 64.29% | 64.29% | +0.00pp |
| N10 | 5 | 61.33% | 64.00% | +2.67pp |
| N20 | 10 | 51.43% | 51.43% | +0.00pp |
| N20 | 5 | 46.67% | 48.00% | +1.33pp |

v15 N10 per-class details:

| TX | K10 correct/total | K10 acc | K5 correct/total | K5 acc |
|---|---:|---:|---:|---:|
| `14-10` | 67/70 | 95.71% | 72/75 | 96.00% |
| `14-7` | 56/70 | 80.00% | 58/75 | 77.33% |
| `20-15` | 70/70 | 100.00% | 74/75 | 98.67% |
| `20-19` | 54/70 | 77.14% | 58/75 | 77.33% |
| `6-15` | 70/70 | 100.00% | 75/75 | 100.00% |
| `8-20` | 70/70 | 100.00% | 75/75 | 100.00% |
| `10-10` | 64/70 | 91.43% | 68/75 | 90.67% |
| `11-10` | 55/70 | 78.57% | 57/75 | 76.00% |
| `18-5` | 67/70 | 95.71% | 71/75 | 94.67% |
| `19-3` | 67/70 | 95.71% | 73/75 | 97.33% |
| `2-13` | 45/70 | 64.29% | 48/75 | 64.00% |
| `2-5` | 57/70 | 81.43% | 61/75 | 81.33% |
| `3-8` | 64/70 | 91.43% | 65/75 | 86.67% |
| `4-10` | 61/70 | 87.14% | 64/75 | 85.33% |
| `8-18` | 53/70 | 75.71% | 61/75 | 81.33% |
| `8-3` | 64/70 | 91.43% | 70/75 | 93.33% |

v15 N20 per-class details:

| TX | K10 correct/total | K10 acc | K5 correct/total | K5 acc |
|---|---:|---:|---:|---:|
| `14-10` | 67/70 | 95.71% | 73/75 | 97.33% |
| `14-7` | 57/70 | 81.43% | 59/75 | 78.67% |
| `20-15` | 70/70 | 100.00% | 74/75 | 98.67% |
| `20-19` | 55/70 | 78.57% | 59/75 | 78.67% |
| `6-15` | 70/70 | 100.00% | 75/75 | 100.00% |
| `8-20` | 70/70 | 100.00% | 75/75 | 100.00% |
| `10-10` | 59/70 | 84.29% | 60/75 | 80.00% |
| `11-10` | 39/70 | 55.71% | 45/75 | 60.00% |
| `18-5` | 44/70 | 62.86% | 38/75 | 50.67% |
| `19-3` | 44/70 | 62.86% | 42/75 | 56.00% |
| `2-13` | 36/70 | 51.43% | 37/75 | 49.33% |
| `2-5` | 55/70 | 78.57% | 61/75 | 81.33% |
| `3-8` | 56/70 | 80.00% | 65/75 | 86.67% |
| `4-10` | 62/70 | 88.57% | 66/75 | 88.00% |
| `8-18` | 49/70 | 70.00% | 62/75 | 82.67% |
| `8-3` | 51/70 | 72.86% | 58/75 | 77.33% |
| `1-1` | 40/70 | 57.14% | 52/75 | 69.33% |
| `1-10` | 59/70 | 84.29% | 65/75 | 86.67% |
| `1-11` | 60/70 | 85.71% | 64/75 | 85.33% |
| `1-12` | 44/70 | 62.86% | 50/75 | 66.67% |
| `1-14` | 48/70 | 68.57% | 36/75 | 48.00% |
| `1-15` | 56/70 | 80.00% | 57/75 | 76.00% |
| `1-16` | 49/70 | 70.00% | 49/75 | 65.33% |
| `1-18` | 40/70 | 57.14% | 36/75 | 48.00% |
| `1-19` | 51/70 | 72.86% | 51/75 | 68.00% |
| `1-2` | 40/70 | 57.14% | 46/75 | 61.33% |

Interpretation:

- v15 is a small but real stability improvement over v13 for K=5 without harming K=10, using a single adaptive rule derived from support size reliability.
- It still fails the active objective: N10/N20 class floors remain below 75%, and N20 mean new accuracy remains around 70%.
- Support-bias calibration was tested as a compressed per-class scalar route and did not change the N10 class floor in this split. The next useful route is not more scalar bias; it should change how low-floor classes such as `2-13`, `11-10`, `18-5`, `1-14`, and `1-18` obtain proxy evidence, likely with per-class proxy bundles plus a support-only validation gate that rejects harmful bundles.

Current goal status: active, not achieved.

## v12 Compressed Pairwise Linear Head Check

Objective: add a qKNN variant that keeps the KNN-style extensibility but avoids persisting raw support samples. The new route is `dualview_support_v12`: it inherits v11 ASLR and adds a support-LOO-selected compressed pairwise linear head for hard new-class pairs.

Mechanism:

- Hard pairs are selected only from support-LOO errors; query labels are audit-only.
- For each selected unordered hard pair, the method fits a ridge linear boundary in feature space using only that pair's K-shot support.
- The deployed state stores only the learned coefficient vector and bias per selected pair, not raw support examples.
- The initial aggressive version overfit support: N10 K10 support-LOO floor rose to 80.00%, but query floor fell to 57.14%. The committed v12 therefore uses a conservative adaptive gate: `linear_weight=clip((0.004+0.006*k_reliability)*linear_gate,0,0.008)`, `top_pairs<=3`, `alpha=10.0`, `clip=0.5`.

Verification:

| command / artifact | result |
|---|---|
| `conda run -n ssr-gpu python -m py_compile code\scripts\phase2_support_metric_qknn_probe.py` | PASS |
| `n10_k10_norm_only_adaptive_v12_safe_seed421029.json` | completed |
| `n10_k5_norm_only_adaptive_v12_safe_seed421037.json` | completed |
| `n20_k10_norm_only_adaptive_v12_safe_seed421029.json` | completed |
| `n20_k5_norm_only_adaptive_v12_safe_seed421037.json` | completed |
| `n20_k5_v11_linear_safe_grid_seed421037.json` | completed; best floor still 44.00% |

Strict NORM-only maximum-query v12 results:

| new count | K | seed | old_acc | min_old | new_acc | min_new | rescue scalars | linear scalars | verdict |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 10 | 10 | 421029 | 92.14% | 77.14% | 84.57% | 61.43% | 16 | 322 | failed |
| 10 | 5 | 421037 | 91.56% | 77.33% | 85.60% | 61.33% | 16 | 322 | failed |
| 20 | 10 | 421029 | 92.62% | 78.57% | 70.64% | 51.43% | 32 | 483 | failed |
| 20 | 5 | 421037 | 92.22% | 78.67% | 68.73% | 44.00% | 32 | 483 | failed |

Per-TX accuracy, N10:

| TX | role | K10 acc | K5 acc |
|---|---|---:|---:|
| `14-10` | old | 95.71% | 96.00% |
| `14-7` | old | 80.00% | 77.33% |
| `20-15` | old | 100.00% | 98.67% |
| `20-19` | old | 77.14% | 77.33% |
| `6-15` | old | 100.00% | 100.00% |
| `8-20` | old | 100.00% | 100.00% |
| `10-10` | new | 91.43% | 90.67% |
| `11-10` | new | 75.71% | 73.33% |
| `18-5` | new | 94.29% | 96.00% |
| `19-3` | new | 95.71% | 96.00% |
| `2-13` | new | 61.43% | 61.33% |
| `2-5` | new | 81.43% | 85.33% |
| `3-8` | new | 91.43% | 88.00% |
| `4-10` | new | 87.14% | 89.33% |
| `8-18` | new | 75.71% | 82.67% |
| `8-3` | new | 91.43% | 93.33% |

Per-TX accuracy, N20:

| TX | role | K10 acc | K5 acc |
|---|---|---:|---:|
| `14-10` | old | 95.71% | 97.33% |
| `14-7` | old | 81.43% | 78.67% |
| `20-15` | old | 100.00% | 98.67% |
| `20-19` | old | 78.57% | 78.67% |
| `6-15` | old | 100.00% | 100.00% |
| `8-20` | old | 100.00% | 100.00% |
| `10-10` | new | 84.29% | 81.33% |
| `11-10` | new | 55.71% | 62.67% |
| `18-5` | new | 62.86% | 50.67% |
| `19-3` | new | 67.14% | 54.67% |
| `2-13` | new | 51.43% | 48.00% |
| `2-5` | new | 78.57% | 81.33% |
| `3-8` | new | 80.00% | 85.33% |
| `4-10` | new | 88.57% | 88.00% |
| `8-18` | new | 70.00% | 81.33% |
| `8-3` | new | 72.86% | 77.33% |
| `1-1` | new | 57.14% | 69.33% |
| `1-10` | new | 84.29% | 86.67% |
| `1-11` | new | 85.71% | 85.33% |
| `1-12` | new | 62.86% | 65.33% |
| `1-14` | new | 68.57% | 44.00% |
| `1-15` | new | 82.86% | 73.33% |
| `1-16` | new | 70.00% | 58.67% |
| `1-18` | new | 58.57% | 53.33% |
| `1-19` | new | 74.29% | 69.33% |
| `1-2` | new | 57.14% | 58.67% |

Interpretation:

- v12 satisfies the compression requirement: raw support storage remains zero; the extra pairwise state is 322 scalars for N10 and 483 scalars for N20 in these runs.
- It does not satisfy the active performance goal. N10 minimum new-class accuracy remains about 61%, and N20 minimum new-class accuracy remains 51.43% for K10 and 44.00% for K5.
- The mean new-class drop from N10 to N20 remains too large: 13.93pp at K10 and 16.87pp at K5, so the requested no-collapse rule is still violated.
- The useful result is negative but actionable: compressed qKNN heads can be made deployment-friendly and non-destructive, but the remaining floor is dominated by hard representation/enrollment classes (`2-13`, `11-10`, `18-5`, `1-14`, `1-18`, `1-2`) rather than by raw-support storage or simple KNN score calibration.

Current goal status: active, not achieved.

## Support Quality and Scenario-Balanced Diagnostics

Objective: continue qKNN optimization without increasing K or adding per-K hand tuning. Two non-query-label routes were checked after v12:

1. `scenario_balanced_assignment`: use the known LEO scenario batch structure during assignment.
2. `support_quality_weight`: compute one support-quality scalar per stored support code from support-LOO truth margin, then use those scalars to reweight compressed prototypes and topm local KNN scores.

Implementation update:

- Added support-LOO quality weighting to `code/scripts/phase2_support_metric_qknn_probe.py`.
- The deployed state remains compressed: no raw support samples are stored; the extra state is one scalar per quantized support code.
- The quality score uses only support labels and support-LOO predictions. Query labels remain audit-only.

Verification:

| command / artifact | result |
|---|---|
| `conda run -n ssr-gpu python -m py_compile code\scripts\phase2_support_metric_qknn_probe.py` | PASS |
| `n10_k10_v11_support_quality_grid_seed421029.csv` | completed,90 rows |
| `n10_k10_v11_support_quality_strong_grid_seed421029.csv` | completed,36 rows |
| `n10_k10_v11_scenariobal_seed421029.json` | completed |
| `n10_k5_v11_scenariobal_seed421037.json` | completed |
| `n20_k10_v11_scenariobal_seed421029.json` | completed |
| `n20_k5_v11_scenariobal_seed421037.json` | completed |

Support-quality result, N10 K10:

| route | rows | best old_acc | best new_acc | best min_new | extra stored scalars | verdict |
|---|---:|---:|---:|---:|---:|---|
| conservative support-quality grid | 90 | 92.14% | 84.57% | 61.43% | 160 | no improvement |
| strong support-quality grid | 36 | 91.43% | 82.57% | 60.00% | 160 | worsened |

Scenario-balanced assignment result:

| new count | K | old_acc | min_old | new_acc | min_new | verdict |
|---:|---:|---:|---:|---:|---:|---|
| 10 | 10 | 58.33% | 20.00% | 44.00% | 34.29% | rejected |
| 10 | 5 | 54.67% | 18.67% | 43.20% | 29.33% | rejected |
| 20 | 10 | 45.71% | 20.00% | 30.29% | 15.71% | rejected |
| 20 | 5 | 40.00% | 18.67% | 28.00% | 17.33% | rejected |

Interpretation:

- Scenario-balanced assignment is not usable here. It over-constrains the batch by LEO scenario and destroys both old and new accuracy.
- Support-quality weighting is deployment-friendly and aligns with the compression requirement, but it does not raise the hard-class floor. Strong weighting actually reduces mean new accuracy.
- This further narrows the failure: the dominant hard classes are not fixed by support reliability reweighting, support-bias, pair-axis, pairwise linear heads, transductive query prototypes, class-diagonal local metrics, or scenario assignment. The remaining credible path is representation/enrollment repair, not another scalar KNN head.

Current goal status: active, not achieved.

## Adaptive v11 ASLR Check

Objective: continue optimizing qKNN without adding K values. The only anchors remain `K=10` and `K=5`. This check adds `dualview_support_v11`, an ASLR policy: Adaptive Support-LOO Rescue. ASLR keeps the v9 compressed qKNN backbone and increases the support-LOO pair-rescue strength from support geometry, K reliability, and new-class load. It stores no raw support samples.

Implementation change:

- Added `dualview_support_v11` / `stable_dualview_v11` to `code/scripts/phase2_support_metric_qknn_probe.py`.
- v11 inherits the v9 pair-logreg, old-residual, local competition, source-old guard, and support-LOO rescue path.
- v11 uses adaptive rescue weight `clip((0.10 + 0.30 * k_reliability) * rescue_gate, 0.05, 0.20)`, so K=5 and K=10 are handled by one formula rather than separate per-K parameter files.
- A first aggressive v11 draft using column-rank calibration plus transductive/dense query refinement was tested and discarded because it lowered the N20 floor; the committed v11 keeps those query-state refinements disabled.

Verification:

| command / artifact | result |
|---|---|
| `conda run -n ssr-gpu python -m py_compile code\scripts\phase2_support_metric_qknn_probe.py` | PASS |
| `n10_k10_norm_only_adaptive_v11_seed421029.json` | completed |
| `n10_k5_norm_only_adaptive_v11_seed421037.json` | completed |
| `n20_k10_norm_only_adaptive_v11_v9plus_seed421029.json` | completed |
| `n20_k5_norm_only_adaptive_v11_v9plus_seed421037.json` | completed |

Strict NORM-only maximum-query results:

| new count | K | old_acc | min_old | new_acc | min_new | stored qcodes | weakest new classes |
|---:|---:|---:|---:|---:|---:|---:|---|
| 10 | 10 | 92.14% | 77.14% | 84.57% | 61.43% | 160 | `2-13` 61.43%,`11-10` 75.71%,`8-18` 75.71% |
| 10 | 5 | 91.56% | 77.33% | 85.60% | 61.33% | 80 | `2-13` 61.33%,`11-10` 73.33%,`8-18` 82.67% |
| 20 | 10 | 92.62% | 78.57% | 70.64% | 51.43% | 260 | `2-13` 51.43%,`11-10` 55.71%,`1-1` 57.14%,`1-2` 57.14%,`1-18` 58.57% |
| 20 | 5 | 92.22% | 78.67% | 68.67% | 42.67% | 130 | `1-14` 42.67%,`2-13` 48.00%,`18-5` 50.67%,`1-18` 53.33% |

Interpretation:

- v11 does not achieve the active target. Even at ten new classes, the floor is only about 61%, driven mainly by `2-13`.
- The N10 mean new accuracy is high, 84.57% for K=10 and 85.60% for K=5, so the failure is not average recognition. It is a minimum-class stability failure.
- When moving from 10 to 20 new classes, mean new accuracy drops from 84.57% to 70.64% for K=10 and from 85.60% to 68.67% for K=5, far exceeding the requested 3pp-per-10-new-class bound.
- Current qKNN compression remains deployment-friendly: the strongest rows store quantized support codes and small scalar state, with `stored_raw_support_count=0`. However, classifier-head adaptation alone is insufficient for the hard classes.

Current goal status: active, not achieved. Next credible step is not larger K and not query transductive tuning; it should target `2-13` first via representation/enrollment repair or a support-selection protocol that remains scientifically honest about the labeled budget.

## Hard-Class Floor Diagnosis After v11

Objective: determine whether the current failure is caused by qKNN head tuning, unlucky strict K-shot support selection, or a representation-level hard class. All diagnostics keep the K anchors fixed at `K=10` and `K=5`; no larger K value is introduced.

Artifacts:

| diagnostic | artifact |
|---|---|
| N10 K10 support-bias grid | `artifacts\n10_k10_v11_support_bias_grid_seed421029.json` |
| N10 K10 prediction debug | `artifacts\n10_k10_v11_preddebug_predictions.csv` |
| N10 K10 pair-axis grid | `artifacts\n10_k10_v11_pair_axis_grid_seed421029.json` |
| N10 K10 transductive grid | `artifacts\n10_k10_v9_transductive_grid_seed421029.json` |
| N10 K10 classdiag metric grid | `artifacts\n10_k10_v11_classdiag_grid_seed421029.json` |
| N10 K10 strict seed scan | `artifacts\n10_k10_v11_seedscan120.json` |
| N10 K5 strict seed scan | `artifacts\n10_k5_v11_seedscan120.json` |

Key findings:

| check | K | rows/seeds | best new_acc | best min_new | best `2-13` | pass 75% floor | interpretation |
|---|---:|---:|---:|---:|---:|---:|---|
| v11 strict seed scan | 10 | 120 seeds | 86.71% at seed421082 | 71.43% | 71.43% | 0 | support draw helps but cannot reach floor |
| v11 strict seed scan | 5 | 120 seeds | 86.80% at seed421064 | 66.67% | 66.67% | 0 | K5 has same hard-class bottleneck |
| support-bias grid | 10 | 20 rows | 84.57% | 61.43% | 61.43% | 0 | class prior/bias is not the limiting factor |
| pair-axis grid | 10 | 24 rows | 84.57% | 61.43% | 61.43% | 0 | simple pair prototype axis does not separate `2-13` |
| classdiag metric grid | 10 | 270 rows | 85.14% | 61.43% | 61.43% | 0 | support-derived local feature weighting lifts mean but not floor |
| transductive query-proto grid | 10 | 288 rows | 84.43% | 60.00% | 60.00% | 0 | unlabeled query prototypes do not repair the hard pair |

Prediction-level confusion at N10 K10 v11 seed421029:

| truth | correct | main wrong assignments | raw top1 evidence |
|---|---:|---|---|
| `2-13` | 43/70=61.43% | `11-10`:15,`10-10`:5,`3-8`:3 | raw top1 is `11-10` for 20/70 |
| `11-10` | 53/70=75.71% | `2-13`:16 | raw top1 is `2-13` for 11/70 |

Interpretation:

- The active target remains unmet: ten new classes still fail the per-class floor, and the 10-to-20-new-class drop is far larger than 3pp.
- The bottleneck is now sharply localized: `2-13` and `11-10` form a reciprocal hard pair. Balanced assignment and support-bias cannot fix it because the failure is pair-boundary separability, not missing class quota.
- Strict K-shot seed scans show that enrollment quality matters, but even the best 120-seed strict K10 support draw reaches only 71.43% minimum class accuracy. Therefore a support-only selector can improve stability but is not enough to prove the requested 75% floor.
- The next aligned optimization should move to representation/enrollment repair for the `2-13`/`11-10` pair, or add a new compressed pairwise head that changes the local pair feature geometry rather than only reweighting existing qKNN scores.

Current goal status: active, not achieved.

## HP08REF Aligned Evaluation and v10 Aux Gate

HP08REF completion:

| item | result |
|---|---|
| remote feature copied back | PASS |
| local artifact dir | `E:\type10-7\automation_reports\CV-SincNet\phase2_qknn_hardpair_n20_20260706\artifacts\aligned_HP08REF` |
| NORM vs HP08REF row count | 11760 vs 11760 |
| aligned metadata keys | `tx_ids,dataset_role,sat_scenarios,rx_ids,channel_views,day_ids,eq_ids,sig_ids` all exact match |
| local SSH residue after copy | none observed |

Implementation update:

- Added `dualview_support_v10` / `stable_dualview_v10` in `code/scripts/phase2_support_metric_qknn_probe.py`.
- v10 keeps v9's adaptive support-LOO rescue, but adds a support-only auxiliary-view reliability gate.
- The gate computes primary-view and auxiliary-view support LOO accuracy without query labels. If the auxiliary view has weak minimum-class support LOO, the effective auxiliary score weight is reduced to zero.
- This avoids storing raw support samples. The extra stored metadata is scalar only: effective aux weight, gate factor, and support LOO diagnostics.

Verification:

| command | result |
|---|---|
| `conda run -n ssr-gpu python -m py_compile code\scripts\phase2_support_metric_qknn_probe.py` | PASS |
| K10 HP08REF v9 seed5 | completed |
| K5 HP08REF v9 seed5 | completed |
| K10 HP08REF v10-floor seed5 | completed |
| K5 HP08REF v10-floor seed5 | completed |

Result summary:

| route | K | seed scope | old_acc | min_old | new_acc | min_new | effective_aux | weakest new |
|---|---:|---|---:|---:|---:|---:|---:|---|
| NORM-only v9 | 10 | full120 | 92.62% | 78.57% | 70.71% | 51.43% | 0.00% | `2-13` 51.43%,`11-10` 55.71%,`1-2` 57.14%,`1-18` 58.57% |
| NORM-only v9 | 5 | full120 | 92.22% | 78.67% | 68.67% | 42.67% | 0.00% | `1-14` 42.67%,`2-13` 48.00%,`18-5` 50.67%,`1-18` 53.33% |
| NORM+HP08REF v9 | 10 | seed5 | 94.29% | 85.71% | 63.86% | 48.57% | 22.00% | `1-12` 48.57%,`1-2` 50.00%,`2-13` 50.00%,`1-14` 51.43% |
| NORM+HP08REF v9 | 5 | seed5 | 94.44% | 85.33% | 48.80% | 28.00% | 22.00% | `1-1` 28.00%,`1-14` 30.67%,`1-18` 37.33%,`1-11` 40.00% |
| NORM+HP08REF v10-floor | 10 | seed5 | 92.62% | 80.00% | 44.79% | 21.43% | 0.00% | `1-1` 21.43%,`1-12` 27.14%,`1-2` 30.00%,`1-11` 37.14% |
| NORM+HP08REF v10-floor | 5 | seed5 | 77.11% | 36.00% | 41.87% | 22.67% | 0.00% | `1-1` 22.67%,`1-11` 26.67%,`1-12` 26.67%,`1-14` 28.00% |

Detailed current credible best, NORM-only v9 full120:

| TX | role | K10 acc | K5 acc |
|---|---|---:|---:|
| `14-10` | old | 95.71% | 97.33% |
| `14-7` | old | 81.43% | 78.67% |
| `20-15` | old | 100.00% | 98.67% |
| `20-19` | old | 78.57% | 78.67% |
| `6-15` | old | 100.00% | 100.00% |
| `8-20` | old | 100.00% | 100.00% |
| `10-10` | new | 84.29% | 81.33% |
| `11-10` | new | 55.71% | 62.67% |
| `18-5` | new | 62.86% | 50.67% |
| `19-3` | new | 68.57% | 54.67% |
| `2-13` | new | 51.43% | 48.00% |
| `2-5` | new | 78.57% | 82.67% |
| `3-8` | new | 78.57% | 85.33% |
| `4-10` | new | 87.14% | 88.00% |
| `8-18` | new | 68.57% | 81.33% |
| `8-3` | new | 70.00% | 77.33% |
| `1-1` | new | 60.00% | 69.33% |
| `1-10` | new | 84.29% | 86.67% |
| `1-11` | new | 90.00% | 84.00% |
| `1-12` | new | 62.86% | 65.33% |
| `1-14` | new | 68.57% | 42.67% |
| `1-15` | new | 84.29% | 73.33% |
| `1-16` | new | 70.00% | 58.67% |
| `1-18` | new | 58.57% | 53.33% |
| `1-19` | new | 72.86% | 69.33% |
| `1-2` | new | 57.14% | 58.67% |

Interpretation:

- The alignment repair succeeded. HP08REF is now valid for dual-view experiments, unlike HP08/HP08A.
- The aligned HP08REF view is not a promotable improvement: v9 dual-view lowers new-class performance, especially at K=5.
- v10 demonstrates a useful safety idea: support-only minimum-class reliability can automatically reject a harmful auxiliary view. However, in the tested seed5 window, support selection itself is poor, so rejecting HP08REF does not solve the N20 floor.
- Current strongest credible qKNN evidence remains NORM-only v9 full120, but it still fails the active goal: K10 min_new 51.43% and K5 min_new 42.67%, both below the required 75%.
- Next optimization should move away from HP08 hard-pair auxiliary views and toward support selection/enrollment quality or representation repair for `2-13`,`1-14`,`18-5`,`1-18`,`1-2`,`11-10`.

Current goal status: active, not achieved.

## 05:44 Sync and SSH Cleanup Note

05:44 CST执行本地到N607同步：本报告同步到`/home/szu2070436088/2510044040/CV-SincNet/automation_reports/CV-SincNet/phase2_qknn_hardpair_n20_20260706/report.md`，并用本地`Get-FileHash`和远端`sha256sum`核对一致。同步后发现既有本地`ssh.exe`残留PID`15320`，命令为早前`phase2_qknn_hardpair_n20_aligned_ref_20260706`的HP08REF nohup launch通道；已只关闭本地SSH客户端并复查为`NO_SSH_PROCESS`、`NO_N607_OR_BRIDGE_ESTABLISHED_22`。该清理不代表停止远端训练或诊断任务。

## Aligned HP08 Relaunch Plan

Timestamp: 2026-07-06 05:35 CST

Objective: re-export HP08 hard-pair N20 features with the same sample-selection seed as the aligned NORM/HEAD feature lineage, then rerun strict `K=10` and `K=5` qKNN maximum-query evaluation. This directly addresses the aux feature misalignment found above.

Local version state:

| file | purpose |
|---|---|
| `code/scripts/phase2_support_metric_qknn_probe.py` | fail closed when `aux_feature_npz` is not sample-aligned |
| `code/scripts/launch_phase2_qknn_hardpair_n20_v1.sh` | expose `EXPORT_SEED`, default `4070391`, for aligned HP08 export |

Verification:

| location | command | result |
|---|---|---|
| local | `conda run -n ssr-gpu python -m py_compile code\scripts\phase2_support_metric_qknn_probe.py` | PASS |
| local | `bash -n code/scripts/launch_phase2_qknn_hardpair_n20_v1.sh` | PASS |
| remote N607 | `/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python -m py_compile code/scripts/phase2_support_metric_qknn_probe.py` | PASS |
| remote N607 | `bash -n code/scripts/launch_phase2_qknn_hardpair_n20_v1.sh` | PASS |

Sync destination:

| local | remote |
|---|---|
| `E:\type10-7\github_publish\CVS-RFFI-repo\code\scripts\phase2_support_metric_qknn_probe.py` | `/home/szu2070436088/2510044040/CV-SincNet/code/scripts/phase2_support_metric_qknn_probe.py` |
| `E:\type10-7\github_publish\CVS-RFFI-repo\code\scripts\launch_phase2_qknn_hardpair_n20_v1.sh` | `/home/szu2070436088/2510044040/CV-SincNet/code/scripts/launch_phase2_qknn_hardpair_n20_v1.sh` |

Remote preflight:

| item | evidence |
|---|---|
| direct SSH | PASS via `tools\n607_ssh_preflight.ps1` |
| project root | `/home/szu2070436088/2510044040/CV-SincNet` visible |
| GPU5 | idle, 10 MiB used |
| existing qKNN/HP08 process | none |
| other active processes | Python jobs on GPU2/GPU3 only |

Planned command:

```bash
cd /home/szu2070436088/2510044040/CV-SincNet
nohup env GPU=5 PROFILE=HP08A EXPORT_SEED=4070391 \
  RUN_ROOT=/home/szu2070436088/2510044040/CV-SincNet/runs/phase2_qknn_hardpair_n20_aligned_20260706 \
  bash code/scripts/launch_phase2_qknn_hardpair_n20_v1.sh \
  > logs/phase2_qknn_hardpair_n20_aligned_20260706/launch_HP08A.out 2>&1 &
```

Expected outputs:

| output | remote path |
|---|---|
| aligned HP08 feature | `runs/phase2_qknn_hardpair_n20_aligned_20260706/MANYNEW20_HARDPAIR_HP08A/ADV3B02_CORE90_SOFT_E200_PHASE1_HARDPAIR_HP08A_N20/features_hardpair_HP08A_n20.npz` |
| K10 sanity qKNN | `runs/phase2_qknn_hardpair_n20_aligned_20260706/HP08A/qknn_eval/n20_k10_coreproto_hardpair_HP08A.json` |
| K5 sanity qKNN | `runs/phase2_qknn_hardpair_n20_aligned_20260706/HP08A/qknn_eval/n20_k5_sourceguard_hardpair_HP08A.json` |

Success check after completion:

1. Copy the aligned HP08 feature locally.
2. Verify full metadata alignment against `features_n20_norm.npz`.
3. Rerun NORM+aligned-HP08 `dualview_support_v9` under `K=10` and `K=5`.
4. Compare against credible NORM-only and NORM+HEAD baselines, not against the contaminated NORM+HP08 rows.

Current goal status: active, pending aligned HP08 run.

## Aligned HP08 Startup Evidence

Launch result:

| item | value |
|---|---|
| wrapper PID | `3232538` |
| train PID | `3232542` |
| GPU | `5` |
| log | `logs/phase2_qknn_hardpair_n20_aligned_20260706/launch_HP08A.out` |
| run root | `runs/phase2_qknn_hardpair_n20_aligned_20260706` |
| export seed | `4070391` |
| local SSH cleanup | no local `ssh.exe` N607 connection remained after startup check |

Startup health:

```text
{"epoch": 5.0, "loss": 3.5946702880859376, ... "proxy_unknown_hard_pair": 0.00019177311612293124, "proxy_unknown_hard_old": 0.0009558753594756126}
GPU5: utilization=23%, memory.used=1481 MiB
```

The initial launch SSH command returned exit code 1 despite printing `launch_pid=3232538`; follow-up short SSH checks confirmed the remote wrapper and train process are running. Treat the launch as active, with health based on the follow-up process/log/GPU evidence rather than the launch command exit code.

Next check: wait for completion, copy `features_hardpair_HP08A_n20.npz`, verify full alignment against NORM, and only then rerun dual-view qKNN.

## HP08A Completion and Reference-Filter Repair

The first aligned-seed attempt completed, but still did not match the NORM reference on `rx_ids`,`day_ids`, and `sig_ids`:

| file | status |
|---|---|
| `aligned_HP08A\features_hardpair_HP08A_n20.npz` | copied locally |
| `aligned_HP08A\n20_k10_coreproto_hardpair_HP08A.json` | copied locally |
| `aligned_HP08A\n20_k5_sourceguard_hardpair_HP08A.json` | copied locally |

Alignment check:

| key | aligned with NORM |
|---|---|
| `tx_ids` | yes |
| `dataset_role` | yes |
| `sat_scenarios` | yes |
| `channel_views` | yes |
| `eq_ids` | yes |
| `rx_ids` | no |
| `day_ids` | no |
| `sig_ids` | no |

HP08A-only sanity results:

| route | K | old_acc | min_old | new_acc | min_new | verdict |
|---|---:|---:|---:|---:|---:|---|
| HP08A only | 10 | 81.19% | 71.43% | 75.71% | 61.43% | failed |
| HP08A only | 5 | 77.56% | 61.33% | 68.13% | 52.00% | failed |

Conclusion: changing only the export seed is insufficient. The NORM lineage used a particular sample-key set that cannot be recovered reliably by seed guessing.

Implemented next repair:

- `train_apply_phase1_iq_preadapter_20260703.py` now supports `--export_reference_npz`.
- When a reference NPZ is provided, each export role is filtered and ordered by the reference `tx/rx/day/eq/sig/role` keys.
- The export dataset bypasses `max_export_samples_per_tx` cap under reference-filter mode so required reference samples are not removed before filtering.
- `launch_phase2_qknn_hardpair_n20_v1.sh` now forwards optional `EXPORT_REFERENCE_NPZ`.

Verification:

| location | command | result |
|---|---|---|
| local | `conda run -n ssr-gpu python -m py_compile code\scripts\train_apply_phase1_iq_preadapter_20260703.py code\scripts\phase2_support_metric_qknn_probe.py` | PASS |
| local | `bash -n code/scripts/launch_phase2_qknn_hardpair_n20_v1.sh` | PASS |
| remote N607 | `/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python -m py_compile code/scripts/train_apply_phase1_iq_preadapter_20260703.py code/scripts/phase2_support_metric_qknn_probe.py` | PASS |
| remote N607 | `bash -n code/scripts/launch_phase2_qknn_hardpair_n20_v1.sh` | PASS |
| remote reference file | `runs/phase2_qknn_hardpair_n20_aligned_ref_20260706/reference/features_n20_norm.npz` present |

Next planned command:

```bash
cd /home/szu2070436088/2510044040/CV-SincNet
nohup env GPU=5 PROFILE=HP08REF EXPORT_SEED=4070391 \
  EXPORT_REFERENCE_NPZ=/home/szu2070436088/2510044040/CV-SincNet/runs/phase2_qknn_hardpair_n20_aligned_ref_20260706/reference/features_n20_norm.npz \
  RUN_ROOT=/home/szu2070436088/2510044040/CV-SincNet/runs/phase2_qknn_hardpair_n20_aligned_ref_20260706 \
  bash code/scripts/launch_phase2_qknn_hardpair_n20_v1.sh \
  > logs/phase2_qknn_hardpair_n20_aligned_ref_20260706/launch_HP08REF.out 2>&1 &
```

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
