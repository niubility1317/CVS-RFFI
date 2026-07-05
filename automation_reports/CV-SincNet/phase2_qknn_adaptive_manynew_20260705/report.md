# phase2_qknn_adaptive_manynew_20260705

## Objective

继续强化qKNN，使其不再依赖K=5和K=10两套手工参数；同一策略需要自适应K和新类别数量。当前硬约束：

- 只评估`K=5`和`K=10`，不扩大K集合。
- 十个新类内每个新类准确率不低于75%。
- `K=5`性能相对`K=10`下降不超过5个百分点。
- 增加新类数量时性能不坍塌；每增加十个新类，准确率下降不超过3个百分点。

## Protocol

- Ground model: `ADV3B02_CORE90_SOFT_E200`。
- Stage2-C target receiver domain: `R_t=7-14`，与source receivers不相交。
- Old TX: `14-10,14-7,20-15,20-19,6-15,8-20`。
- Ten seen-new TX: `10-10,11-10,18-5,19-3,2-13,2-5,3-8,4-10,8-18,8-3`。
- Support/query均来自`R_t=7-14`且使用LEO弱星地信道视图；query标签只用于审计，不用于拟合。
- `proxy_unknown`属于源接收机域，不作为Stage2-C目标新类证据。

## Adaptive qKNN v1

新增`adaptive_qknn_policy=dualview_support_v1`。策略只读取support特征、support标签、K和新类数量，不读取query标签：

1. 计算support类原型最大相似度、P90相似度、平均类内半径，得到`support_hardness`。
2. 根据新类数量得到`class_load`，根据每类support数量得到`k_reliability`。
3. 当`support_hardness`高或`class_load`高时，自动进入稳定路径：`diag_whiten_fisher`、较高`proto_mix`、较强pair Gaussian、old source guard、关闭ridge/core增强。
4. 当support更可靠且类别簇不拥挤时，才逐步启用ridge head和core prototype增强。

该策略不保存原始support IQ样本；部署侧保存量化support embedding/prototype、两路transform标量和少量策略标量。

## Local Verification

Changed file:

- `github_publish/CVS-RFFI-repo/code/scripts/phase2_support_metric_qknn_probe.py`

Verification:

| command | result |
|---|---|
| `conda run -n ssr-gpu python -m py_compile code\scripts\phase2_support_metric_qknn_probe.py` | PASS |

## Ten-New-Class Result

Feature views:

- Main: `MANYNEW10_CONFLICT_NORM_features_leo_repaired.npz`
- Aux: `features_hardpair_HP08.npz`

| K | policy | old_acc | min_old | new_acc | min_new | K5-vs-K10 gap |
|---:|---|---:|---:|---:|---:|---:|
| 10 | `dualview_support_v1` | 87.86% | 74.29% | 87.71% | 77.14% | reference |
| 5 | `dualview_support_v1` | 88.89% | 72.00% | 85.87% | 76.00% | -1.85 pp |

K=10 per-class:

| class | role | acc |
|---|---|---:|
| `14-10` | old | 88.57% |
| `14-7` | old | 78.57% |
| `20-15` | old | 92.86% |
| `20-19` | old | 74.29% |
| `6-15` | old | 94.29% |
| `8-20` | old | 98.57% |
| `10-10` | new | 77.14% |
| `11-10` | new | 84.29% |
| `18-5` | new | 95.71% |
| `19-3` | new | 88.57% |
| `2-13` | new | 78.57% |
| `2-5` | new | 85.71% |
| `3-8` | new | 92.86% |
| `4-10` | new | 95.71% |
| `8-18` | new | 88.57% |
| `8-3` | new | 90.00% |

K=5 per-class:

| class | role | acc |
|---|---|---:|
| `14-10` | old | 96.00% |
| `14-7` | old | 84.00% |
| `20-15` | old | 92.00% |
| `20-19` | old | 72.00% |
| `6-15` | old | 89.33% |
| `8-20` | old | 100.00% |
| `10-10` | new | 76.00% |
| `11-10` | new | 78.67% |
| `18-5` | new | 93.33% |
| `19-3` | new | 93.33% |
| `2-13` | new | 76.00% |
| `2-5` | new | 84.00% |
| `3-8` | new | 92.00% |
| `4-10` | new | 92.00% |
| `8-18` | new | 85.33% |
| `8-3` | new | 88.00% |

Interpretation: 十新类目标已由同一自适应策略达成；K=10最低新类相对原K=10最佳从75.71%提升到77.14%，K=5保持最低新类76.00%。旧类平均满足OLD80，但旧类floor仍未达到75%，不能声明旧类floor完成。

## N20 Scaling Plan

N607 read-only preflight: direct `N607` PASS；project root存在；GPU 2-7空闲。

ManyTx target receiver `7-14`可用非旧TX数量：136类，每类至少80条目标接收机样本。

N20 seen-new TX:

```text
10-10,11-10,18-5,19-3,2-13,2-5,3-8,4-10,8-18,8-3,1-1,1-10,1-11,1-12,1-14,1-15,1-16,1-18,1-19,1-2
```

新增十类来自默认proxy候选池，因此N20导出/训练时必须从`PROXY_UNKNOWN_TX_IDS`中移除这20个target seen-new TX，避免目标新类进入proxy训练。

Remote launch target:

- Run ID: `phase2_qknn_adaptive_manynew20_20260705`
- Existing launcher base: `code/scripts/launch_phase2_adv3b02_manynew10_conflict_protected_20260705.sh`
- After export: use synced adaptive `phase2_support_metric_qknn_probe.py` on NORM main view + HEAD aux view for K=5 and K=10.

## N20 Launch

Remote sync/verification:

| item | evidence |
|---|---|
| synced script | `/home/szu2070436088/2510044040/CV-SincNet/code/scripts/phase2_support_metric_qknn_probe.py` |
| remote syntax | `python -m py_compile code/scripts/phase2_support_metric_qknn_probe.py` PASS |
| remote SHA256 | `a4d5f2a134820847f6e317b6bda6ba6c1d534ba8717db62b727f8f79aa0bdc16` |

Active remote jobs at launch time:

| PID | GPU | purpose |
|---:|---:|---|
| `3025184` | 0 | existing Phase1 `EPOC_R2_OLD_FLOOR`, not touched |
| `3025608` | 1 | existing Phase1 `EPOC_R2_BALANCED_SEP`, not touched |
| `3052325` | 2 | N20 `MANYNEW10_CONFLICT_NORM` feature train/export |
| `3052326` | 3 | N20 `MANYNEW10_CONFLICT_HEAD` feature train/export |

Launch command summary:

```bash
cd /home/szu2070436088/2510044040/CV-SincNet
nohup env RUN_ID=phase2_qknn_adaptive_manynew20_20260705 \
  RUNS_ROOT=/home/szu2070436088/2510044040/CV-SincNet/runs/phase2_qknn_adaptive_manynew20_20260705 \
  LOG_ROOT=/home/szu2070436088/2510044040/CV-SincNet/logs/phase2_qknn_adaptive_manynew20_20260705 \
  GPUS_CSV=2,3 \
  TARGET_NEW_TX_IDS=<20 target-new tx labels> \
  PROXY_UNKNOWN_TX_IDS=<default proxy list minus the 20 target-new tx labels> \
  SEED=4070720 \
  bash code/scripts/launch_phase2_adv3b02_manynew10_conflict_protected_20260705.sh \
  > logs/phase2_qknn_adaptive_manynew20_20260705/driver.nohup.out 2>&1 &
```

Expected feature outputs:

| view | path |
|---|---|
| NORM main | `/home/szu2070436088/2510044040/CV-SincNet/runs/phase2_qknn_adaptive_manynew20_20260705/PHASE2_MANYNEW10_RX7_14/MANYNEW10_CONFLICT_NORM/features_leo_repaired.npz` |
| HEAD aux | `/home/szu2070436088/2510044040/CV-SincNet/runs/phase2_qknn_adaptive_manynew20_20260705/PHASE2_MANYNEW10_RX7_14/MANYNEW10_CONFLICT_HEAD/features_leo_repaired.npz` |

Note: the reused launcher still names the case directory `PHASE2_MANYNEW10_RX7_14`, but the command-line `TARGET_NEW_TX_IDS` contains 20 labels. Interpret this run as N20, not N10.

Next inspection: after both train/export jobs finish, run adaptive qKNN with `adaptive_qknn_policy_grid=dualview_support_v1` for `K=5` and `K=10`, then compare N20 new_acc against the N10 87.71%/85.87% reference and check whether the drop is within3pp.

## N20 Adaptive Result

Adaptive qKNN was run on:

- Main view: `MANYNEW10_CONFLICT_NORM/features_leo_repaired.npz`
- Aux view: `MANYNEW10_CONFLICT_HEAD/features_leo_repaired.npz`

| new count | K | old_acc | min_old | new_acc | min_new | drop vs N10 same K | verdict |
|---:|---:|---:|---:|---:|---:|---:|---|
| 20 | 10 | 75.00% | 54.29% | 64.50% | 40.00% | -23.21 pp | failed |
| 20 | 5 | 66.89% | 45.33% | 60.87% | 24.00% | -25.00 pp | failed |

K=10 N20 per-new accuracy:

| class | acc |
|---|---:|
| `10-10` | 61.43% |
| `11-10` | 55.71% |
| `18-5` | 62.86% |
| `19-3` | 64.29% |
| `2-13` | 40.00% |
| `2-5` | 74.29% |
| `3-8` | 72.86% |
| `4-10` | 85.71% |
| `8-18` | 74.29% |
| `8-3` | 72.86% |
| `1-1` | 50.00% |
| `1-10` | 81.43% |
| `1-11` | 85.71% |
| `1-12` | 58.57% |
| `1-14` | 61.43% |
| `1-15` | 70.00% |
| `1-16` | 71.43% |
| `1-18` | 42.86% |
| `1-19` | 52.86% |
| `1-2` | 51.43% |

K=5 N20 per-new accuracy:

| class | acc |
|---|---:|
| `10-10` | 49.33% |
| `11-10` | 53.33% |
| `18-5` | 40.00% |
| `19-3` | 38.67% |
| `2-13` | 48.00% |
| `2-5` | 80.00% |
| `3-8` | 81.33% |
| `4-10` | 89.33% |
| `8-18` | 76.00% |
| `8-3` | 76.00% |
| `1-1` | 65.33% |
| `1-10` | 84.00% |
| `1-11` | 85.33% |
| `1-12` | 65.33% |
| `1-14` | 24.00% |
| `1-15` | 57.33% |
| `1-16` | 53.33% |
| `1-18` | 40.00% |
| `1-19` | 60.00% |
| `1-2` | 50.67% |

NORM-only K=10消融：

| setting | old_acc | min_old | new_acc | min_new |
|---|---:|---:|---:|---:|
| NORM only, no aux | 75.48% | 55.71% | 65.00% | 42.86% |

Interpretation: N20 collapse is not caused by HEAD aux fusion; it remains when using NORM only. The failure is therefore feature-space/class-density related. The added `1-*` classes introduce a dense intra-family cluster, and several original classes (`2-13`, `10-10`, `11-10`, `18-5`, `19-3`) also drop sharply. The current adaptive v1 solves K adaptation for ten classes, but does not yet solve class-count scaling. The next algorithmic change should add multi-new-class cluster handling, not another scalar aux-weight grid.

Recommended next route:

1. Add support-only hierarchical qKNN: first assign to coarse support clusters, then classify within cluster with local balanced assignment.
2. Add class-density-aware quotas: classes in dense clusters get local competition only against nearest support-prototype neighbors, avoiding global quota pressure from unrelated easy classes.
3. Add per-class support compactness gates: when a class has high support radius or high nearest-prototype similarity, use multi-prototype/codebook storage instead of a single centroid.
4. Re-run N10 and N20 with the same `K=5,K=10` anchors and require N20 drop within3pp before attempting N30.

## Adaptive v2 Diagnostic

Code change:

- Added `role_balanced_assignment`: performs closed-set Hungarian quota assignment separately inside the known old-query and new-query partitions. This prevents global assignment from swapping old-query quota with new-query quota when the number of new classes grows.
- Added `local_competition`: a support-prototype-neighbor score adjustment. It stores no raw support sample; it only needs the compressed class prototype graph and a small neighbor list.
- Added `query_proto_refine`: optional transductive refinement from provisional pseudo labels to temporary query-batch prototypes. This also stores no raw support sample; deployment storage is unchanged, and runtime memory is one temporary prototype per class for the current batch.
- Added `adaptive_qknn_policy=dualview_support_v2`: for class-load above the ten-new-class anchor it enables role-balanced assignment and a small local-competition weight. For the ten-new-class anchor it reduces to the v1 policy, so the prior K=5/K=10 best rows are preserved.

Local verification:

```text
conda run -n ssr-gpu python -m py_compile code\scripts\phase2_support_metric_qknn_probe.py
```

Remote sync/verification:

| item | evidence |
|---|---|
| synced script | `/home/szu2070436088/2510044040/CV-SincNet/code/scripts/phase2_support_metric_qknn_probe.py` |
| remote syntax | `python3 -m py_compile code/scripts/phase2_support_metric_qknn_probe.py` PASS |
| remote SHA256 | `6b733070fb6ca846184ce6c25a306acc6b5a0532a05ae838e7af2046a3d33685` |
| N607 action boundary | synced and compiled only; no new experiment launched |

Result summary:

| scope | K | method | old_acc | min_old | new_acc | min_new | key settings |
|---|---:|---|---:|---:|---:|---:|---|
| N10 | 10 | `dualview_support_v2` | 87.86% | 74.29% | 87.71% | 77.14% | same as v1 |
| N10 | 5 | `dualview_support_v2` | 88.89% | 72.00% | 85.87% | 76.00% | same as v1 |
| N20 | 10 | `dualview_support_v1` | 75.00% | 54.29% | 64.50% | 40.00% | baseline |
| N20 | 5 | `dualview_support_v1` | 66.89% | 45.33% | 60.87% | 24.00% | baseline |
| N20 | 10 | `dualview_support_v2` | 92.14% | 78.57% | 68.29% | 44.29% | role balance + local competition |
| N20 | 5 | `dualview_support_v2` | 92.44% | 78.67% | 65.53% | 36.00% | role balance + local competition |
| N20 | 10 | `v2+labelprop` best diagnostic | 91.67% | 77.14% | 68.86% | 45.71% | `labelprop_weight=0.1` |
| N20 | 10 | `v2+query_proto_refine` best diagnostic | 92.14% | 78.57% | 67.93% | 47.14% | `query_proto_refine_weight=0.05,topm=50` |

Interpretation:

- The previous N20 collapse had a large assignment component. In the reproduced K=10 v1 row, raw top1 old-query accuracy was 88.33%, but global balanced assignment assigned 89 old-query samples to new labels and 89 new-query samples to old labels. `role_balanced_assignment` removes this cross-role quota swap and lifts N20 K=10 old_acc from 75.00% to 92.14%.
- The ten-new-class target is still satisfied for K=5 and K=10, and K=5 remains within5pp of K=10 on new_acc.
- The twenty-new-class target is not satisfied. v2 improves N20 K=10 new_acc by +3.79pp and min_new by +4.29pp, but min_new remains 44.29%. Label propagation and query-prototype refinement only raise the best observed min_new to 45.71% and 47.14%, respectively.
- Current evidence therefore says the remaining N20 bottleneck is not primarily KNN storage or assignment; it is feature separability in dense new-class families. The weakest classes still need better representation or a stronger class-family-aware adaptation stage before claiming "more new classes do not collapse".

Deployment/storage note:

- v2 still does not store raw support samples. Persistent state remains transform scalars, class prototypes, optional aux transform scalars, and small prototype-neighbor metadata.
- Runtime overhead is low: role-balanced assignment uses the same Hungarian assignment scale as the existing closed-set quota solver, local competition stores 104 prototype-neighbor edges for the N20 K=10 row, and query-prototype refinement stores at most one temporary prototype per class when enabled.

Current verdict: v2 is a useful stabilization step for old/new quota coupling, but the active goal is not complete because N20 min_new is far below75%.

## 2026-07-06 Local Follow-up: compressed diagnostic heads

Objective: continue optimizing qKNN for many-new-class stability under the same Phase2-C protocol, without increasing the reported K grid beyond `K=5,K=10` and without storing raw support samples.

Local code change:

- Added explicit `transductive_proto_*` diagnostic switches in `code/scripts/phase2_support_metric_qknn_probe.py`. This branch builds temporary support-anchored query prototypes for the current batch only. It does not add persistent raw-support storage.
- The unverified `dualview_support_v3` policy entry was removed from the adaptive policy dispatcher after local diagnostics showed no improvement. Therefore promoted adaptive policies remain `dualview_support_v1` and `dualview_support_v2`; transductive prototype refinement is diagnostic-only unless explicitly enabled by CLI grid arguments.

Local verification:

```text
conda run -n ssr-gpu python -m py_compile code\scripts\phase2_support_metric_qknn_probe.py
```

PASS.

N20 K=10 seed/policy follow-up:

| candidate | K | seed | old_acc | min_old | new_acc | min_new | persistent extra storage | verdict |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| `dualview_support_v2`, original seed | 10 | 421029 | 92.14% | 78.57% | 68.29% | 44.29% | compressed prototypes + local edges | baseline v2 |
| seed/policy search best | 10 | 421023 | 91.43% | 77.14% | 72.14% | 52.86% | compressed prototypes + local edges | improved but failed |
| ridge-head small grid best | 10 | 421023 | 91.43% | 77.14% | 72.14% | 52.86% | none beyond v2 | no gain |
| pair-logreg small grid best | 10 | 421023 | 91.43% | 77.14% | 72.36% | 52.86% | 4 scalars in best row | tiny mean gain, no floor gain |
| class-diag metric grid best | 10 | 421023 | 91.43% | 77.14% | 72.14% | 52.86% | none in best row | no gain |
| pair-axis manual-v2 grid best | 10 | 421023 | 91.43% | 77.14% | 72.29% | 52.86% | one pair axis in best row | tiny mean gain, no floor gain |
| support-bias LOO grid best | 10 | 421023 | 91.43% | 77.14% | 72.14% | 52.86% | none in best row | no gain |
| transductive-proto diagnostic best | 10 | 421023 | 91.43% | 77.14% | 72.21% | 52.86% | temporary query prototypes only | no floor gain |

Worst N20 K=10 classes for the best seed/policy row:

| truth class | acc | main wrong predictions |
|---|---:|---|
| `2-13` | 52.86% | `11-10`:11, `1-18`:8, `1-2`:5, `1-14`:4 |
| `1-18` | 52.86% | `11-10`:15, `2-13`:5, `1-14`:3, `1-2`:3 |
| `18-5` | 54.29% | `1-18`:16, `1-16`:14 |
| `1-2` | 57.14% | `1-19`:6, `3-8`:6, `4-10`:5, `2-5`:4 |
| `11-10` | 58.57% | `2-13`:16, `1-18`:8, `1-14`:4 |
| `1-14` | 61.43% | `18-5`:11, `1-10`:7, `1-16`:7 |

Interpretation:

- The best N20 row now has good old-class performance (`old_acc=91.43%`, `min_old=77.14%`) and improves new-class mean accuracy to `72.14%`, but the new-class floor remains `52.86%`. The active many-new-class stability goal is therefore not complete.
- The remaining failures are not solved by support-compressed classifier heads, per-pair scalar/axis correction, class-diagonal metric correction, support LOO biasing, or temporary transductive prototypes. The error pattern is dense new/new confusion among specific support-prototype neighborhoods, especially `2-13`/`11-10`/`1-18` and `18-5`/`1-16`.
- Current evidence says the next useful route should not be another scalar KNN head grid. It should either improve the embedding/feature extractor for dense ManyTx families, or introduce a protocol-explicit admission/defer mechanism for support classes whose compressed prototypes are inseparable. If no defer/reject is allowed, the present embedding does not support a valid claim that twenty new classes remain above the 75% per-class floor.

Current goal status: active, not achieved.

## 2026-07-06 Continuation: fast quota assignment for full max-query qKNN

Objective: make the receiver-domain qKNN route executable at full max-query size, then test one shared adaptive domain-refine rule under only `K=5,K=10`.

Code change:

- Added `--fast_role_balanced_assignment` in `code/scripts/phase2_support_metric_qknn_probe.py`.
- The previous exact role-balanced assignment materialized one slot per query sample and used Hungarian assignment inside old/new query partitions. For max-query ADV3B02 this implied about `9540x9540` and `7920x7920` assignment matrices.
- The new fast path keeps the same closed-set role quota semantics but uses a deterministic greedy transport repair: start from per-row argmax, then move the lowest-loss rows from overfull classes to underfull classes until exact class quotas are satisfied.
- This is not the exact Hungarian optimum, so all rows below are explicitly marked `fast quota`. It is a deployable approximation and removes the full-query runtime blocker.

Local verification:

```text
conda run -n ssr-gpu python -m py_compile code\scripts\phase2_support_metric_qknn_probe.py
```

PASS. Full max-query K=10 and K=5 rows completed locally in seconds to tens of seconds instead of timing out.

Fast quota max-query summary:

| feature | K | query old/class | query new/class | candidate | old_acc | min_old | seen_new_acc | min_seen_new | status |
|---|---:|---:|---:|---|---:|---:|---:|---:|---|
| ADV3B02 Stage2-C LEO | 10 | 1590 | 990 | fast quota baseline | 72.37% | 51.70% | 30.38% | 16.16% | completed |
| ADV3B02 Stage2-C LEO | 10 | 1590 | 990 | fast quota `rx_scenario/new,w=0.25` | 72.37% | 51.70% | 43.23% | 27.98% | completed |
| ADV3B02 Stage2-C LEO | 10 | 1590 | 990 | fast quota `rx_scenario/new,w=0.5` | 72.37% | 51.70% | 42.71% | 28.28% | best K10 floor |
| ADV3B02 Stage2-C LEO | 10 | 1590 | 990 | fast quota `rx_scenario/new,w=0.75` | 72.37% | 51.70% | 41.64% | 27.27% | stability candidate |
| ADV3B02 Stage2-C LEO | 5 | 1595 | 995 | fast quota baseline | 72.56% | 52.35% | 22.86% | 10.45% | completed |
| ADV3B02 Stage2-C LEO | 5 | 1595 | 995 | fast quota `rx_scenario/new,w=0.25` | 72.56% | 52.35% | 36.34% | 26.13% | completed |
| ADV3B02 Stage2-C LEO | 5 | 1595 | 995 | fast quota `rx_scenario/new,w=0.5` | 72.56% | 52.35% | 36.98% | 26.93% | completed |
| ADV3B02 Stage2-C LEO | 5 | 1595 | 995 | fast quota `rx_scenario/new,w=0.75` | 72.56% | 52.35% | 37.61% | 27.34% | best shared stability |

Shared-weight interpretation:

- `w=0.75` is the best shared setting for the active K-stability requirement: K=10 seen-new is 41.64%, K=5 seen-new is 37.61%, so K=5 is 4.03pp below K=10 and meets the "within 5pp" stability clause.
- The same setting improves K=10 over fast baseline by +11.26pp mean new and +11.11pp min-new; it improves K=5 by +14.75pp mean new and +16.89pp min-new.
- The old-class metrics are unchanged by construction because `domain_refine_scope=new` leaves old-class columns untouched under role-balanced assignment.
- The active accuracy floor is still not achieved: the best shared K=10 minimum new class is 27.27%, and the best shared K=5 minimum new class is 27.34%, far below 75%.

K=10 per-class detail for fast quota `rx_scenario/new,w=0.75`:

| role | tx | accuracy |
|---|---|---:|
| old | 14-10 | 64.91% |
| old | 14-7 | 61.76% |
| old | 20-15 | 77.30% |
| old | 20-19 | 51.70% |
| old | 6-15 | 87.48% |
| old | 8-20 | 91.07% |
| new | 1-10 | 35.45% |
| new | 1-12 | 43.84% |
| new | 1-14 | 44.95% |
| new | 1-16 | 32.83% |
| new | 1-18 | 54.55% |
| new | 1-8 | 37.88% |
| new | 10-11 | 27.27% |
| new | 10-4 | 56.36% |

K=5 per-class detail for fast quota `rx_scenario/new,w=0.75`:

| role | tx | accuracy |
|---|---|---:|
| old | 14-10 | 67.08% |
| old | 14-7 | 60.69% |
| old | 20-15 | 76.61% |
| old | 20-19 | 52.35% |
| old | 6-15 | 87.59% |
| old | 8-20 | 91.03% |
| new | 1-10 | 34.57% |
| new | 1-12 | 33.27% |
| new | 1-14 | 38.39% |
| new | 1-16 | 38.49% |
| new | 1-18 | 47.04% |
| new | 1-8 | 33.27% |
| new | 10-11 | 27.34% |
| new | 10-4 | 48.54% |

Conclusion:

- This is concrete progress for the qKNN route: max-query is now runnable, the K=5/K=10 stability condition is met under a single shared weight, and the method still stores compressed support/domain prototypes rather than raw support samples.
- It is not a successful final result. The remaining blocker is representation separability for the weakest new class, especially `10-11`, not KNN storage or assignment runtime.

Current goal status: active, not achieved.

## 2026-07-06 ADV3B02 Stage2-C follow-up: receiver-domain new-scope qKNN

Objective: continue the qKNN route on the complete ADV3B02 Stage2-C LEO feature with only `K=5,K=10`, max available query count, and no raw support storage. The tested idea is receiver-domain compressed prototypes: use observable `rx_id|sat_scenario` domains to refine only new-class score columns while preserving the old-class closed-set head.

Code change:

- Added metadata domain construction in `code/scripts/phase2_support_metric_qknn_probe.py`: `scenario`, `rx`, `channel`, `rx_scenario`, and `rx_channel`.
- Added `domain_refine_key_grid`, `domain_refine_weight_grid`, and `domain_refine_scope_grid`.
- Added `domain_refine_scope=new|old|all`; `new` stores and scores only new-class domain prototypes. With 8 new classes and 15 observed `rx_scenario` domains, persistent new-domain state is 120 compressed prototypes, not raw support samples.
- Added role-balanced row pruning for `scope=new`: when old/new query partitions are known, only new-query rows are domain-refined.

Local verification:

```text
conda run -n ssr-gpu python -m py_compile code\scripts\phase2_support_metric_qknn_probe.py
conda run -n ssr-gpu python code\scripts\phase2_support_metric_qknn_probe.py ... --query_per_old 100 --query_per_new 100 --domain_refine_key_grid rx_scenario --domain_refine_weight_grid 0.5 --domain_refine_scope_grid new
```

PASS. Smoke output: `adv3b02_full_stage2c/smoke_k10_domainrefine_newscope_q100.json`, with `domain_refine_scope=new`, `domain_refine_domain_count=15`, `stored_domain_refine_prototype_count=120`, and `stored_raw_support_count=0`.

Full max-query runtime boundary:

- Full max-query `scope=new` was attempted twice for K=10 and exceeded 180 seconds each time on the local Windows host, even after row/column pruning. The remaining bottleneck is `base._class_scores(... scenario_aware=True)` over receiver-scenario partitions.
- Therefore the max-query `scope=new` row below is reported as a strict role-balanced inference, not as a completed full-run JSON. Under role-balanced assignment, old-query rows use only old-class columns and new-query rows use only new-class columns; `scope=new` leaves old columns identical to the v3 baseline and leaves new columns identical to the completed `rx_scenario/all` domain-refine run.

Max-query summary:

| feature | K | query old/class | query new/class | candidate | old_acc | min_old | seen_new_acc | min_seen_new | status |
|---|---:|---:|---:|---|---:|---:|---:|---:|---|
| ADV3B02 Stage2-C LEO | 10 | 1590 | 990 | `dualview_support_v3` baseline | 72.57% | 52.45% | 27.80% | 17.88% | completed |
| ADV3B02 Stage2-C LEO | 10 | 1590 | 990 | `rx_scenario/all,w=0.5` | 63.69% | 51.89% | 44.72% | 33.13% | completed |
| ADV3B02 Stage2-C LEO | 10 | 1590 | 990 | `rx_scenario/new,w=0.5` | 72.57% | 52.45% | 44.72% | 33.13% | inferred from role-balanced column independence |
| ADV3B02 Stage2-C LEO | 5 | 1595 | 995 | `dualview_support_v3` baseline | 72.36% | 51.29% | 25.83% | 14.17% | completed |
| ADV3B02 Stage2-C LEO | 5 | 1595 | 995 | `rx_scenario/all,w=0.5` | 52.11% | 33.17% | 32.90% | 18.29% | completed |
| ADV3B02 Stage2-C LEO | 5 | 1595 | 995 | `rx_scenario/new,w=0.5` | 72.36% | 51.29% | 32.90% | 18.29% | inferred from role-balanced column independence |

K=10 per-class detail for inferred `rx_scenario/new,w=0.5`:

| role | tx | accuracy |
|---|---|---:|
| old | 14-10 | 67.30% |
| old | 14-7 | 60.69% |
| old | 20-15 | 76.54% |
| old | 20-19 | 52.45% |
| old | 6-15 | 87.42% |
| old | 8-20 | 91.01% |
| new | 1-10 | 39.70% |
| new | 1-12 | 46.77% |
| new | 1-14 | 48.79% |
| new | 1-16 | 33.13% |
| new | 1-18 | 57.17% |
| new | 1-8 | 38.59% |
| new | 10-11 | 46.46% |
| new | 10-4 | 47.17% |

K=5 per-class detail for inferred `rx_scenario/new,w=0.5`:

| role | tx | accuracy |
|---|---|---:|
| old | 14-10 | 67.21% |
| old | 14-7 | 60.25% |
| old | 20-15 | 76.68% |
| old | 20-19 | 51.29% |
| old | 6-15 | 87.71% |
| old | 8-20 | 91.03% |
| new | 1-10 | 28.64% |
| new | 1-12 | 44.62% |
| new | 1-14 | 29.45% |
| new | 1-16 | 21.91% |
| new | 1-18 | 46.93% |
| new | 1-8 | 32.96% |
| new | 10-11 | 18.29% |
| new | 10-4 | 40.40% |

Interpretation:

- The new-scope receiver-domain variant is a better qKNN compression direction than all-scope domain refine: it preserves old-class accuracy while lifting K=10 new mean from 27.80% to 44.72% and new floor from 17.88% to 33.13%.
- It still fails the active goal. The K=10 new floor remains 41.87pp below the 75% floor, and K=5 remains much weaker than K=10.
- The practical deployment cost is small in memory terms: one global qKNN support codebook plus 120 new-class receiver-domain prototypes for this setting. The current blocker is not storage but separability and the unoptimized Python scoring path.

Current goal status: active, not achieved.

## 2026-07-06 Continuation: ADV3B02 full Stage2-C feature qKNN max-query audit

Objective: test whether the complete `ADV3B02_CORE90_SOFT_E200` frozen Stage2-C LEO multi-receiver feature package can serve as a stronger representation-side input for the qKNN route under the active `K=5,K=10` goal. This run keeps the K axis fixed and uses the maximum available query count for each class.

Feature source:

| item | value |
|---|---|
| remote feature | `/home/szu2070436088/2510044040/CV-SincNet/runs/phase2_adv3b02_frozen_manytx_unknown_diag_20260706/ADV3B02_CORE90_FROZEN_QKNN8_M1_TO_ALL_R5/features_stage2c_leo_multirx.npz` |
| local copy | `E:\type10-7\automation_reports\CV-SincNet\phase2_adv3b02_frozen_manytx_unknown_diag_20260706\artifacts\features_stage2c_leo_multirx.npz` |
| SHA256 | `AD788749019D1EA6FBA67BE49F8A241CB3A114153ACB8CCF1F84C2DE7DA9A0A4` |
| roles | `source=13440,target_old=9600,target_new=8000,target_unknown=8000,proxy_unknown=9350` |
| old classes | `6` classes, `1600` target-old samples per class |
| seen-new classes | `8` classes, `1000` target-new samples per class |
| LEO views | `leo_clear_weak,leo_low_elev_weak,leo_rain_weak` |

Local/N607 handling:

- N607 direct preflight passed.
- The remote feature package was copied with `scp` and hash-verified locally.
- No new remote training or remote evaluation was launched.
- SSH/SCP sessions were checked after use; no persistent N607 or bridge connection remained.

Code change:

- Added `dualview_support_v3` / `stable_dualview_v3` to `code/scripts/phase2_support_metric_qknn_probe.py`.
- The new adaptive policy fixes a v2 weakness: v2 used `class_load=(new_count-10)/20`, so an 8-new-class task had `class_load=0` and did not automatically enable role-balanced assignment even when support geometry was hard.
- v3 makes class load continuous from two new classes upward: `class_load=clip((new_count-2)/18)`.
- v3 enables role-balanced assignment when `class_load>0` or support hardness is high, and applies local competition with a nonzero low-class-count floor. This is an adaptive rule over support geometry, `K`, and class count; it is not per-K custom tuning.

Local verification:

```text
conda run -n ssr-gpu python -m py_compile code\scripts\phase2_support_metric_qknn_probe.py
```

PASS.

Max-query result summary:

| feature | K | query old/class | query new/class | candidate | old_acc | min_old | seen_new_acc | min_seen_new | verdict |
|---|---:|---:|---:|---|---:|---:|---:|---:|---|
| ADV3B02 full Stage2-C LEO | 10 | 1590 | 990 | role-balanced qKNN | 72.51% | 52.26% | 27.77% | 17.78% | failed |
| ADV3B02 full Stage2-C LEO | 10 | 1590 | 990 | `dualview_support_v3` | 72.57% | 52.45% | 27.80% | 17.88% | tiny gain, failed |
| ADV3B02 full Stage2-C LEO | 5 | 1595 | 995 | role-balanced qKNN | 72.32% | 51.22% | 25.84% | 14.27% | failed |
| ADV3B02 full Stage2-C LEO | 5 | 1595 | 995 | `dualview_support_v3` | 72.36% | 51.29% | 25.83% | 14.17% | no meaningful gain, failed |

Per-class result for `dualview_support_v3`, K=10:

| role | class | accuracy |
|---|---|---:|
| old | `14-10` | 67.30% |
| old | `14-7` | 60.69% |
| old | `20-15` | 76.54% |
| old | `20-19` | 52.45% |
| old | `6-15` | 87.42% |
| old | `8-20` | 91.01% |
| seen-new | `1-10` | 32.02% |
| seen-new | `1-12` | 24.95% |
| seen-new | `1-14` | 22.02% |
| seen-new | `1-16` | 28.28% |
| seen-new | `1-18` | 24.75% |
| seen-new | `1-8` | 26.87% |
| seen-new | `10-11` | 17.88% |
| seen-new | `10-4` | 45.66% |

Per-class result for `dualview_support_v3`, K=5:

| role | class | accuracy |
|---|---|---:|
| old | `14-10` | 67.21% |
| old | `14-7` | 60.25% |
| old | `20-15` | 76.68% |
| old | `20-19` | 51.29% |
| old | `6-15` | 87.71% |
| old | `8-20` | 91.03% |
| seen-new | `1-10` | 36.58% |
| seen-new | `1-12` | 33.47% |
| seen-new | `1-14` | 18.49% |
| seen-new | `1-16` | 14.17% |
| seen-new | `1-18` | 23.92% |
| seen-new | `1-8` | 20.20% |
| seen-new | `10-11` | 16.48% |
| seen-new | `10-4` | 43.32% |

Interpretation:

- The v3 adaptive gate is a real method improvement over v2 because it removes a discontinuity at ten new classes and makes role balancing/competition respond to support hardness and class count continuously.
- It does not solve the active goal. Even with the complete ADV3B02 full Stage2-C LEO feature package and maximum query count, both old and seen-new accuracy are far below the target, and the weakest new classes remain below 20%.
- K=5 is within 5pp of K=10 in mean seen-new accuracy on this feature package, but both are far below the required accuracy and floor.
- The strongest evidence now points away from additional qKNN score-head tweaks and toward representation-side retraining/adaptation before qKNN enrollment. In particular, the frozen ADV3B02 LEO Stage2-C features do not preserve target receiver identity geometry strongly enough for the current qKNN memory head.

Current goal status: active, not achieved.

## 2026-07-06 Local Follow-up: query-structure separability audit

Objective: test whether the current N20 feature file contains a usable unlabeled query-cluster structure for an adaptive transductive qKNN variant. This is a diagnostic only; query labels are used only for the oracle cluster-naming audit.

Diagnostic setup:

- Feature file: `n20_features/features_n20_norm.npz`.
- Split: seed421023, `old_role=target_old`, `new_role=target_unknown`, `pool_per_old=10`, `pool_per_new=10`.
- New labels are read from the fingerprinted current rerun: `10-10,11-10,18-5,19-3,2-13,2-5,3-8,4-10,8-18,8-3,1-1,1-10,1-11,1-12,1-14,1-15,1-16,1-18,1-19,1-2`.
- Method: spherical K-means on new-query embeddings, initialized by K-shot support prototypes.
- Audit modes:
  - `oracle`: assign clusters to true labels with Hungarian matching. This uses query labels only to measure whether the query geometry is clusterable.
  - `support_named`: assign clusters to labels using support-prototype similarity. This is the deployable naming approximation.

Result summary:

| scope | K | query/class | mode | new_acc | min_new | weakest classes |
|---|---:|---:|---|---:|---:|---|
| N20 | 10 | 70 | oracle cluster naming | 30.57% | 0.00% | `1-10` 0.00%,`19-3` 5.71%,`1-1` 5.71% |
| N20 | 10 | 70 | support-named clusters | 24.07% | 0.00% | `1-14` 0.00%,`19-3` 2.86%,`3-8` 2.86% |
| N20 | 5 | 75 | oracle cluster naming | 31.13% | 1.33% | `1-10` 1.33%,`19-3` 6.67%,`18-5` 10.67% |
| N20 | 5 | 75 | support-named clusters | 26.93% | 2.67% | `1-10` 2.67%,`1-2` 2.67%,`1-12` 4.00% |

Interpretation:

- This diagnostic does not replace qKNN evaluation, but it explains why the recent KNN-head variants are not lifting the N20 floor: the current query embeddings do not form stable class clusters for the hardest ManyTx groups.
- The failure remains severe even when query labels are allowed only for oracle cluster naming, so a pure transductive qKNN assignment mechanism is unlikely to reach `min_new>=75%` on this feature file.
- The next credible route is to restore/reproduce the archived stronger feature state or add representation-side adaptation before the qKNN memory head. More support-compressed score surgery is not a high-probability path to the active target.

Current goal status: active, not achieved.

## 2026-07-06 Local Follow-up: support LOO pair-rescue diagnostic

Objective: test a more targeted compressed qKNN variant for dense new-class confusion. Instead of storing raw support samples or sweeping a generic classifier head, the new diagnostic uses leave-one-out support predictions to identify directed confusion pairs, then stores a small pairwise linear margin for the selected pairs.

Code change:

- Added `support_loo_pair_rescue_*` CLI grids in `code/scripts/phase2_support_metric_qknn_probe.py`.
- The mechanism uses support-only LOO errors to select dense-class confusion pairs such as `truth -> predicted competitor`.
- Persistent extra state is small: each selected pair stores 4 scalar coefficients. In the best N20 K=10 row below, 5 directed pairs were used, adding 20 scalars.
- Raw support samples are still not stored; `stored_raw_support_count=0`.

Local verification:

```text
conda run -n ssr-gpu python -m py_compile code\scripts\phase2_support_metric_qknn_probe.py
```

PASS.

Result summary:

| scope | K | seed | candidate | old_acc | min_old | new_acc | min_new | extra persistent state | verdict |
|---|---:|---:|---|---:|---:|---:|---:|---:|---|
| N20 | 10 | 421023 | v2 best before LOO pair rescue | 91.43% | 77.14% | 72.14% | 52.86% | 0 pair-rescue scalars | baseline |
| N20 | 10 | 421023 | LOO pair rescue best | 91.43% | 77.14% | 72.64% | 52.86% | 20 scalars | tiny mean gain, floor unchanged |
| N20 | 5 | 421023 | manual-v2 no LOO pair rescue | 91.67% | 77.14% | 68.07% | 45.71% | 0 pair-rescue scalars | baseline for same seed |
| N20 | 5 | 421023 | LOO pair rescue using K10-best settings | 91.67% | 77.14% | 68.29% | 47.14% | 20 scalars | small floor gain, still failed |

Interpretation:

- The LOO-pair mechanism is a cleaner qKNN compression variant than raw-support KNN: it converts support-set failure modes into a few stored pairwise scalars and adapts naturally to `K` and class count through the number of observed LOO confusion pairs.
- It does not solve the active goal. N20 K=10 still has `min_new=52.86%`, and N20 K=5 still has `min_new=47.14%`.
- The useful evidence is negative but specific: even when the pair selection is driven by the exact support-set dense-cluster errors, the deployed query floor does not move. This strengthens the conclusion that the remaining collapse is primarily representation/embedding separability, not KNN storage, quota assignment, or a missing small compressed head.

Current goal status: active, not achieved. Recommended next route is representation-side repair or a protocol-explicit enrollment quality gate; continuing to add scalar KNN heads is unlikely to reach the 75% per-class floor for twenty new classes.

## 2026-07-06 Local Follow-up: dense-cluster query prototype diagnostic

Objective: test a cluster-local qKNN compression variant for dense ManyTx new-class groups, still under the same `K=5,K=10` anchors and without storing raw support samples.

Code change:

- Added `dense_cluster_*` CLI grids in `code/scripts/phase2_support_metric_qknn_probe.py`.
- The mechanism builds a support-prototype graph inside the selected role, forms dense class clusters, then uses only the current query batch to build temporary cluster-local query prototypes before the final assignment.
- Persistent state is unchanged: class prototypes and small graph/parameter metadata only. No raw support sample is stored; the query prototypes are runtime-only.
- Added reproducibility fields to each row: `old_role`,`new_role`,`query_per_old`,`query_per_new`,`exclude_pool_from_query`.

Local verification:

```text
conda run -n ssr-gpu python -m py_compile code\scripts\phase2_support_metric_qknn_probe.py
```

PASS.

Result summary from the current HEAD rerun:

| scope | K | query/class | candidate | old_acc | min_old | new_acc | min_new | dense-cluster state | verdict |
|---|---:|---:|---|---:|---:|---:|---:|---:|---|
| N20 | 10 | 70 | current weight=0 baseline | 92.38% | 80.00% | 44.36% | 24.29% | 0 clusters | baseline for current rerun |
| N20 | 10 | 70 | best active dense-cluster | 92.38% | 80.00% | 44.36% | 24.29% | 5 clusters,20 temporary query prototypes | no gain |
| N20 | 5 | 75 | current weight=0 baseline | 92.00% | 78.67% | 38.67% | 20.00% | 0 clusters | baseline for current rerun |
| N20 | 5 | 75 | best active dense-cluster | 92.00% | 78.67% | 39.00% | 21.33% | 4 clusters,20 temporary query prototypes | tiny gain, failed |

Important reproducibility note:

- The previously archived N20 seed421023 `dualview_support_v2` row remains in this report as historical evidence: `K=10 old_acc=91.43%,min_old=77.14%,new_acc=72.14%,min_new=52.86%`.
- Re-running the current HEAD with the same visible high-level settings did not reproduce that archived row; it produced the current baselines above. The support geometry fields match, but the stored `max_offdiag_proto_sim/class_radii` evidence differs, so the historical row should be treated as an archived result rather than a current rerun.
- This is exactly why the script now records `old_role/new_role/query_per_*` and `exclude_pool_from_query` directly in each output row.

Interpretation:

- Dense-cluster query prototypes do not solve the many-new-class floor problem. K=10 has no measurable gain; K=5 gains only +0.33pp mean new accuracy and +1.33pp min-new in the current rerun.
- The result is still useful negative evidence: even a role-local, support-graph-derived, no-raw-support transductive variant cannot lift dense ManyTx families toward the 75% per-class floor.
- The active goal remains open. The next route should move away from post-hoc qKNN score surgery and toward representation-side repair or explicit enrollment-quality gating under the Stage2-C protocol.

## 2026-07-06 Local Follow-up: reproducibility hardening and old-residual new-class qKNN

Objective: harden the N20 qKNN evidence trail and test whether new-class collapse is caused by old-class receiver-domain attractors. This keeps the goal anchors at `K=5,K=10` and does not expand the K grid.

Code change:

- Added output-level SHA256 fingerprints for `feature_npz` and `aux_feature_npz`.
- Added split fingerprints to every result row: global support/query digests and per-label support/query digests.
- Added `old_residual_new_*` grids in `code/scripts/phase2_support_metric_qknn_probe.py`.
- The old-residual mechanism estimates a compressed old-class prototype subspace from support prototypes, removes that subspace from support/query embeddings, then blends residual-space scores into new-class scores only.
- Persistent extra state is compressed: old-subspace basis, old center, and new residual prototypes. Raw support samples are not stored; `stored_raw_support_count=0`.

Local verification:

```text
conda run -n ssr-gpu python -m py_compile code\scripts\phase2_support_metric_qknn_probe.py
```

PASS.

Reproducibility audit:

| artifact | value |
|---|---|
| feature_npz | `n20_features/features_n20_norm.npz` |
| feature_npz_sha256 | `aab99af6d7022f422a61eac8018884790d1497c8f3f1f17aa65a02bdcf1432e2` |
| aux_feature_npz | `n20_features/features_n20_head.npz` |
| aux_feature_npz_sha256 | `637bae47ace04b87f9e72c42d006c44218ce8ccc58b819343e38dafca85aa140` |
| K=10 support_index_sha16 | `4a07f9cba0d2708f` |
| K=10 old_query_index_sha16 | `4bc84cba8c5c2001` |
| K=10 new_query_index_sha16 | `bb7d5c3a3574c068` |
| K=10 query_index_sha16 | `26d9c84b19bc26b7` |

Important boundary: the archived N20 seed421023 `dualview_support_v2` row (`K=10 old_acc=91.43%,min_old=77.14%,new_acc=72.14%,min_new=52.86%`) is not reproduced by the current feature file and current HEAD. The current rerun with matching visible high-level settings gives `old_acc=92.38%,min_old=80.00%,new_acc=44.36%,min_new=24.29%`. Future rows now carry feature and split fingerprints so this drift cannot be hidden.

Prediction diagnosis from the current K=10 rerun:

- Raw new-query top-1 predictions are heavily pulled to old labels, especially `8-3` (198), `11-10` (152), `20-19` (141), `1-10` (108), `20-15` (89), `1-16` (87), and `6-15` (79).
- Role-balanced assignment gives each new class the expected 70 assignments, but it cannot repair the wrong new/new score geometry.
- Weak current K=10 classes include `1-1` (24.29%), `1-2` (24.29%), `1-12` (28.57%), `1-11` (37.14%), and `11-10` (37.14%).

Result summary from the current HEAD reproducible rerun:

| scope | K | query/class | candidate | old_acc | min_old | new_acc | min_new | extra persistent state | verdict |
|---|---:|---:|---|---:|---:|---:|---:|---:|---|
| N20 | 10 | 70 | current fingerprinted baseline | 92.38% | 80.00% | 44.36% | 24.29% | 0 old-residual scalars | reproducible baseline |
| N20 | 10 | 70 | old-residual best (`weight=0.2,rank=5,proto_mix=0.4`) | 92.38% | 80.00% | 46.79% | 24.29% | 4160 scalars | +2.43pp mean new, floor unchanged |
| N20 | 5 | 75 | current fingerprinted baseline | 92.00% | 78.67% | 38.67% | 20.00% | 0 old-residual scalars | reproducible baseline |
| N20 | 5 | 75 | old-residual best (`weight=0.2,rank=5,proto_mix=0.4`) | 92.00% | 78.67% | 39.60% | 25.33% | 4160 scalars | +0.93pp mean new, +5.33pp floor |

Interpretation:

- Old-residual scoring is a valid qKNN compression innovation candidate because it replaces raw-support storage with a support-derived old-subspace basis and residual prototypes.
- It does not achieve the active goal. N20 remains far below the `min_new>=75%` requirement, and K=5 remains more than 5 percentage points below K=10 in mean new accuracy on the current fingerprinted rerun.
- The evidence now separates two facts: the historical archived row was stronger but is not currently reproducible from the local feature file, while the reproducible current feature geometry has severe old-attractor and dense ManyTx separability failures.
- The next optimization should target adaptive enrollment/support-quality repair or representation-side adaptation rather than more scalar score blending.

Current goal status: active, not achieved.
