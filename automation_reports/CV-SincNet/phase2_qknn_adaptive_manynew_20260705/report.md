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
