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

## 2026-07-06 qKNNV44 Adaptive Topm Policy

Objective:在不改变`项目.md`Stage2-C协议、不增加K、不使用query标签调参的前提下，基于V42 topm诊断实现`stable_dualview_v44`。部署样本仍是目标接收机域的LEO星地信道叠加样本；旧类support用于目标域适应，新类support用于注册识别。

Implementation:

| file | change |
|---|---|
| `code/scripts/phase2_support_metric_qknn_probe.py` | 注册`dualview_support_v44`/`stable_dualview_v44`；继承V42压缩qKNN边界；当`K>=10`且`new_count>=14`时自适应把`topm`从命令行默认4切到1；K5保持`topm=4`。 |
| `code/tests/test_phase2_support_metric_qknn_v44_policy.py` | 新增TDD策略测试，覆盖K10,N20、K10,N14和K5,N20三种门控。 |

Local verification:

| command | result |
|---|---|
| `conda run --no-capture-output -n ssr-gpu python -m pytest -q code\tests\test_phase2_support_metric_qknn_v44_policy.py` | RED first: failed with `unsupported adaptive_qknn_policy: stable_dualview_v44`; GREEN after implementation: PASS |
| `conda run --no-capture-output -n ssr-gpu python -m pytest -q code\tests\test_phase2_support_metric_qknn_v43_policy.py code\tests\test_phase2_support_metric_qknn_v44_policy.py` | PASS |
| `conda run --no-capture-output -n ssr-gpu python -m py_compile code\scripts\phase2_support_metric_qknn_probe.py code\tests\test_phase2_support_metric_qknn_v43_policy.py code\tests\test_phase2_support_metric_qknn_v44_policy.py` | PASS |

Artifacts:

| artifact | path |
|---|---|
| v44 focused outputs | `artifacts\v44_adaptive_topm_20260706\` |
| K5,N14 | `artifacts\v44_adaptive_topm_20260706\v44_k5_n14.csv` |
| K5,N20 | `artifacts\v44_adaptive_topm_20260706\v44_k5_n20.csv` |
| K10,N14 | `artifacts\v44_adaptive_topm_20260706\v44_k10_n14.csv` |
| K10,N20 | `artifacts\v44_adaptive_topm_20260706\v44_k10_n20.csv` |

Important command-scope note:the valid v44 evidence uses the same V42 matrix flags, including`--scenario_aware --balanced_assignment` and `new_role=target_unknown` as the local field name for ManyTx held-out TX. An earlier local command omitted those two flags and produced a diagnostic-only negative artifact; it is not used for the v44 verdict.

Focused comparison against V42:

| setting | policy | topm | old_acc | min_old | seen_new_acc | min_new | weakest new TX | delta vs V42 |
|---|---|---:|---:|---:|---:|---:|---|---|
| K5,N14 | V42 | 4 | 92.00% | 80.00% | 90.10% | 73.33% | `1-12` | reference |
| K5,N14 | V44 | 4 | 92.00% | 80.00% | 90.10% | 73.33% | `1-12` | unchanged |
| K5,N20 | V42 | 4 | 92.00% | 80.00% | 79.80% | 69.33% | `2-13`/`1-2` | reference |
| K5,N20 | V44 | 4 | 92.00% | 80.00% | 79.80% | 69.33% | `1-2` | unchanged |
| K10,N14 | V42 | 4 | 91.90% | 82.86% | 90.20% | 68.57% | `1-1` | reference |
| K10,N14 | V44 | 1 | 91.90% | 82.86% | 90.82% | 71.43% | `1-1` | `min_new +2.86pp`,mean new `+0.62pp` |
| K10,N20 | V42 | 4 | 91.90% | 82.86% | 84.64% | 72.86% | `1-1` | reference |
| K10,N20 | V44 | 1 | 91.90% | 82.86% | 84.71% | 74.29% | `1-1`/`2-13`/`1-12` | `min_new +1.43pp`,mean new `+0.07pp` |

Interpretation:

- V44 is a valid incremental improvement over V42 for reliable support (`K=10`) under N14/N20 many-new pressure. It raises the worst new-class floor without old-class regression.
- V44 deliberately leaves K5 unchanged because V42 topm sweep showed K5 needs`topm=4` to avoid a lower new-class floor.
- V44 still does not complete the active objective. The strongest K10,N20 floor is now74.29%, still0.71pp below the75% floor; K5,N20 remains69.33%.
- The next useful route should target the remaining local confusion around`1-1`/`1-12` and`2-13`, preferably with a gated support-quality or local-pair mechanism that does not affect K5 globally and preserves the V44 topm switch.

Version state note:this report is under`E:\type10-7`, which is not aGit repository. The code/test changes are in theGit-backed carrier`E:\type10-7\github_publish\CVS-RFFI-repo` and will be committed there after verification.

Current goal status: active, not achieved.

## 2026-07-06 qKNNV42 topm and hard-class diagnostic

Objective:在不修改生产代码的前提下，继续分析`qKNNV42`当前瓶颈，确认下一版优化应优先做参数自适应、支持质量、多原型还是局部判别边界。本轮仍遵循`项目.md`的Stage2-C协议：`K=5/K=10`目标域support，target-old与target-new均来自LEO星地信道叠加后的target receiver domain；query标签只用于审计。

Artifacts:

| artifact | purpose |
|---|---|
| `artifacts\v43_neighborhood_gate_20260706\audit_v42_k5_n20_predictions.csv` | V42 K5,N20 prediction audit |
| `artifacts\v43_floor_rescue_20260706\audit_v42_k10_n20_predictions.csv` | V42 K10,N20 prediction audit |
| `artifacts\v42_topm_sweep_20260706\v42_topm_k5_n20.csv` | V42 K5,N20 topm sweep |
| `artifacts\v42_topm_sweep_20260706\v42_topm_k10_n20.csv` | V42 K10,N20 topm sweep |
| `artifacts\v42_topm_sweep_20260706\v42_topm_k10_n14.csv` | V42 K10,N14 topm sweep |

V42 hard-class error audit:

| setting | weak TX | main wrong predictions | observation |
|---|---|---|---|
| K5,N20 | `2-13` | `1-2`:9/75,`11-10`:6/75,`1-14`:4/75 | mixed remote/new-neighbor confusion |
| K5,N20 | `1-2` | `1-19`:7/75,`10-10`:5/75,`1-18`:4/75 | often stolen by non-local or old-adjacent top scores |
| K5,N20 | `11-10` | `1-18`:12/75,`2-13`:10/75 | stable local confusion cluster |
| K5,N20 | `1-1` | `1-12`:11/75,`8-3`:6/75 | stable local confusion cluster |
| K10,N20 | `1-1` | `1-12`:11/70,`8-3`:8/70 | stable local confusion cluster |
| K10,N20 | `1-12` | `1-1`:11/70,`8-3`:5/70 | reciprocal confusion with`1-1` |
| K10,N20 | `2-13` | `11-10`:8/70,`1-2`:4/70,`1-14`:3/70 | not solved by stronger pair rescue |

Old-class protection audit:

| setting | weak old TX | main wrong predictions | observation |
|---|---|---|---|
| K5,N20 | `14-7` | `20-19`:14/75 | old-class floor tied to old-old pair |
| K5,N20 | `20-19` | `14-7`:10/75,`14-10`:5/75 | reciprocal old-old confusion |
| K10,N20 | `14-7` | `20-19`:10/70 | persists at higherK |
| K10,N20 | `20-19` | `14-7`:8/70,`14-10`:4/70 | persists at higherK |

Topm sweep summary:

| setting | topm | old_acc | min_old | seen_new_acc | min_new | interpretation |
|---|---:|---:|---:|---:|---:|---|
| K5,N20 | 1 | 92.44% | 80.00% | 79.07% | 65.33% | hurts new floor |
| K5,N20 | 2 | 92.44% | 81.33% | 79.07% | 65.33% | old improves but new floor hurts |
| K5,N20 | 3 | 92.00% | 80.00% | 79.73% | 68.00% | below V42 floor |
| K5,N20 | 4 | 92.00% | 80.00% | 79.80% | 69.33% | current best balance |
| K5,N20 | 5 | 91.56% | 80.00% | 79.13% | 68.00% | hurts old and new |
| K10,N14 | 1 | 91.90% | 82.86% | 90.82% | 71.43% | improves min_new vs topm4 |
| K10,N14 | 4 | 91.90% | 82.86% | 90.20% | 68.57% | current V42 default |
| K10,N20 | 1 | 91.90% | 82.86% | 84.71% | 74.29% | improves min_new vs topm4 |
| K10,N20 | 4 | 91.90% | 82.86% | 84.64% | 72.86% | current V42 default |
| K10,N20 | 10 | 93.10% | 84.29% | 84.00% | 70.00% | improves old but hurts new floor |

Interpretation:

- 现有V42的`topm=4`不是全局最优。K5仍需要`topm=4`保护新类floor；K10在N14/N20上改为`topm=1`可提高新类最低类，同时不损伤旧类。
- 仅靠`topm=1`仍不能达到75%最低新类门槛：K10,N20从72.86%升到74.29%，K10,N14从68.57%升到71.43%。
- 下一版最小有效方向是`stable_dualview_v44`做support-geometry自适应`topm`：低K仍保留topm4，高K/更可靠support切换为topm1；该改动应先作为TDD测试覆盖，然后在K5/K10、N14/N20复验。
- 更大机制方向仍是hard-class局部判别。`1-1/1-12/8-3`、`2-13/11-10/1-18`和旧类`14-7/20-19/14-10`分别构成稳定混淆簇；直接放大pair rescue会放大错误通道，需使用support审计门控或表示侧hard-pair separation。

Recommended v44 design candidate:

1. 新增`stable_dualview_v44`/`dualview_support_v44`，继承V42/V43安全边界。
2. 在`_adaptive_qknn_overrides`中加入`topm`覆盖：`k_reliability>=0.25`且`class_load>=0.50`时设`topm=1`；低K或support不可靠时保留`topm=4`。
3. 不改变`项目.md`协议，不启用query标签，不启用query-state cluster，不同步N607。
4. 预期收益仅声明为诊断性改进：K10新类floor应提升，K5不应回退；若K10,N20仍低于75%，目标继续未完成。

Current goal status: active, not achieved.

## 2026-07-06 qKNNV43 support-neighborhood diagnostic

Objective:继续优化`qKNNV42`，在不改变`项目.md`协议的前提下，针对`K=5/K=10`目标域少样本、LEO星地信道叠加样本、新类增多后的最低类坍塌问题，试探`stable_dualview_v43`。本轮没有远端启动或N607同步，所有验证均在本地`ssr-gpu`环境完成。

Version/state notes:

- 根目录`E:\type10-7`和`E:\type10-7\code`不是Git仓库；代码改动位于Git承载面`E:\type10-7\github_publish\CVS-RFFI-repo`。
- 本轮读取并遵循`AGENTS.md`、`项目.md`；未修改CVS科学场景、K-shot协议、old/new/unknown TX语义、receiver split或Stage2-A/B/C边界。
- 修改文件:`code/scripts/phase2_support_metric_qknn_probe.py`、`code/tests/test_phase2_support_metric_qknn_v43_policy.py`。
- 新增结果目录:`E:\type10-7\automation_reports\CV-SincNet\phase2_qknn_hardpair_n20_20260706\artifacts\v43_neighborhood_gate_20260706`。
- 负向诊断目录:`E:\type10-7\automation_reports\CV-SincNet\phase2_qknn_hardpair_n20_20260706\artifacts\v43_floor_rescue_20260706`。该目录对应第一次放大pair rescue/top-pairs的尝试，结果更差，不作为候选晋升。

Implementation summary:

- 注册`stable_dualview_v43`/`dualview_support_v43`，继承V42的双视图门控、source-target transport、support-LOO rescue、linear pair边界、neighbor contrast和old-class保护路径。
- 第一次尝试放大`support_loo_pair_rescue_weight`和`support_loo_pair_linear_weight`，在N14提升seen-new均值但压低最低类；该方向判定为负向诊断。
- 最终v43撤回放大rescue，恢复V42级别的support-LOO pair强度，仅保留弱`neighborhood_gate`试探和15%`neighbor_contrast`幅度试探；实际N20/N14均与V42同分，未带来晋升收益。

Verification:

| command | result |
|---|---|
| `conda run --no-capture-output -n ssr-gpu python -m pytest -q code\tests\test_phase2_support_metric_qknn_v43_policy.py` | PASS |
| `conda run --no-capture-output -n ssr-gpu python -m py_compile code\scripts\phase2_support_metric_qknn_probe.py code\tests\test_phase2_support_metric_qknn_v43_policy.py` | PASS |

Final V42/V43 focused comparison:

| setting | policy | old_acc | min_old | seen_new_acc | min_new | weak new TX | verdict |
|---|---|---:|---:|---:|---:|---|---|
| N14,K=5 | V42 | 92.00% | 80.00% | 90.10% | 73.33% | `1-12` | fails floor |
| N14,K=5 | final V43 | 92.00% | 80.00% | 90.10% | 73.33% | `1-12` | no improvement |
| N14,K=10 | V42 | 91.90% | 82.86% | 90.20% | 68.57% | `1-1` | fails floor |
| N14,K=10 | final V43 | 91.90% | 82.86% | 90.20% | 68.57% | `1-1` | no improvement |
| N20,K=5 | V42 | 92.00% | 80.00% | 79.80% | 69.33% | `2-13`,`1-2` | fails floor |
| N20,K=5 | final V43 | 92.00% | 80.00% | 79.80% | 69.33% | `2-13`,`1-2` | no improvement |
| N20,K=10 | V42 | 91.90% | 82.86% | 84.64% | 72.86% | `1-1` | fails floor |
| N20,K=10 | final V43 | 91.90% | 82.86% | 84.64% | 72.86% | `1-1` | no improvement |

Hard-class audit from V42 K10,N20 predictions:

| truth TX | accuracy | main wrong predictions |
|---|---:|---|
| `1-1` | 72.86% | `1-12`:11/70,`8-3`:8/70 |
| `1-12` | 74.29% | `1-1`:11/70,`8-3`:5/70 |
| `2-13` | 74.29% | `11-10`:8/70,`1-2`:4/70,`1-14`:3/70 |
| `11-10` | 77.14% | `1-18`:9/70,`2-13`:7/70 |
| `8-3` | 77.14% | `1-1`:6/70,`2-5`:5/70,`1-12`:4/70 |

Interpretation:

- 当前最强仍是`stable_dualview_v42`。最终v43是安全诊断分支，不是性能晋升。
- 旧类域适应性能没有被v43破坏，但也没有提高；N20,K10仍为old_acc 91.90%、min_old 82.86%。
- 新类增多后的坍塌瓶颈不是简单rescue强度不足。放大support-LOO pair rescue会强化`1-1/1-12/8-3`等互混通道并降低最低类。
- 弱`neighborhood_gate`由于support-audit保护没有实际触发；15%`neighbor_contrast`幅度在当前assignment下不改变预测。
- 下一步应避免继续放大pair rescue。更可行方向是针对`1-1/1-12/8-3`和`2-13/11-10/1-18`构建更可靠的support侧局部判别证据，例如类内多原型/receiver-aware局部原型、support质量选择或表示侧 hard-pair separation，而不是调大现有救援权重。

Current goal status: active, not achieved.

## v42 K/New-Count Matrix

Timestamp: 2026-07-06 CST

Objective: run the current strongest formal qKNN policy, `stable_dualview_v42`, over `K in {1,2,3,5,10}` and target-new class counts from 1 to 20. This is a Stage2-C matrix: target-old and target-new support/query are all target-domain satellite/LEO samples, not clean samples. Unknown rejection is not evaluated in this matrix.

Protocol:

| item | value |
|---|---|
| primary feature | `artifacts\features_hardpair_HP08L5_n20.npz` |
| auxiliary feature | `artifacts\features_hardpair_HP08L5_n20_leosketch96.npz` |
| channel view | satellite/LEO target view; LEO-sketch also has star-ground channel applied before compression |
| old TX count | 6 |
| new TX order | `10-10,11-10,18-5,19-3,2-13,2-5,3-8,4-10,8-18,8-3,1-1,1-10,1-11,1-12,1-14,1-15,1-16,1-18,1-19,1-2` |
| K grid | `1,2,3,5,10` |
| query per class | `80-K`, maximum available split from the 80-sample-per-class feature file |
| support/query overlap | `exclude_pool_from_query=true` |
| v42 parameters | K1/2/3/5 use `transform_strength=0.1,proto_mix=0.4`; K10 uses the current v42 K10 setting `transform_strength=0.10453387554141083,proto_mix=0.39829979667197096` |
| auxiliary gate | support-LOO gated; all matrix rows set `effective_aux_score_weight=0` |
| raw support storage | `stored_raw_support_count=0` for all rows |
| execution | local evaluation only; no N607 training or remote launch |

Artifacts:

| artifact | path |
|---|---|
| matrix summary | `artifacts\v42_matrix_k123510_n1to20_20260706\matrix_summary.csv` |
| per-TX details | `artifacts\v42_matrix_k123510_n1to20_20260706\matrix_per_tx.csv` |
| JSON summary | `artifacts\v42_matrix_k123510_n1to20_20260706\matrix_summary.json` |
| failures | `artifacts\v42_matrix_k123510_n1to20_20260706\matrix_failures.json` |

Run status: 100/100 combinations completed; `matrix_failures.json` is empty.

Threshold summary:

| K | max new count with `min_new>=75%` | max new count with `old_acc>=80%` and `min_new>=75%` | N20 seen_new_acc | N20 min_new | N20 weakest new TX |
|---:|---:|---:|---:|---:|---|
| 1 | 4 | 4 | 62.34% | 22.78% | `1-14` 22.78% |
| 2 | 7 | 7 | 71.09% | 48.72% | `1-14` 48.72% |
| 3 | 12 | 12 | 71.88% | 48.05% | `1-14` 48.05% |
| 5 | 13 | 13 | 79.80% | 69.33% | `2-13` 69.33% |
| 10 | 13 | 13 | 84.64% | 72.86% | `1-1` 72.86% |

Mean/floor drop from N10 to N20:

| K | N10 seen_new_acc | N20 seen_new_acc | mean drop | N10 min_new | N20 min_new | floor drop |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 84.18% | 62.34% | 21.84pp | 59.49% | 22.78% | 36.71pp |
| 2 | 81.92% | 71.09% | 10.83pp | 65.38% | 48.72% | 16.67pp |
| 3 | 91.69% | 71.88% | 19.81pp | 80.52% | 48.05% | 32.47pp |
| 5 | 92.00% | 79.80% | 12.20pp | 82.67% | 69.33% | 13.33pp |
| 10 | 93.29% | 84.64% | 8.64pp | 84.29% | 72.86% | 11.43pp |

N20 endpoint detail:

| K | old_acc | min_old | seen_new_acc | min_new | weakest new TX | query/class |
|---:|---:|---:|---:|---:|---|---:|
| 1 | 91.14% | 81.01% | 62.34% | 22.78% | `1-14` 22.78% | 79 |
| 2 | 92.31% | 80.77% | 71.09% | 48.72% | `1-14` 48.72% | 78 |
| 3 | 91.56% | 79.22% | 71.88% | 48.05% | `1-14` 48.05% | 77 |
| 5 | 92.00% | 80.00% | 79.80% | 69.33% | `2-13` 69.33% | 75 |
| 10 | 91.90% | 82.86% | 84.64% | 72.86% | `1-1` 72.86% | 70 |

Interpretation:

- v42 is stable on old classes across the matrix: N20 old_acc remains above 91% for all tested K, although K3 min_old is 79.22% and therefore slightly misses the old-class floor.
- v42 does not satisfy the active multi-new objective. With the current class order and single support seed per K, the maximum new-class count that keeps every new class at or above 75% is 13 for K5 and K10, 12 for K3, 7 for K2, and 4 for K1.
- The requested "every additional 10 new classes drops no more than 3pp" condition is not met. K5 mean new accuracy drops 12.20pp from N10 to N20, and K10 drops 8.64pp; floor drops are larger.
- The collapse is class-specific, not caused by clean/LEO mismatch or raw support storage. All rows use LEO target features and `stored_raw_support_count=0`. The dominant weakest classes are `2-13` in lower/mid new counts and `1-14`/`1-1` after adding the dense `1-*` ManyTx group.
- The next qKNN optimization should target the transition from 13 to 14 new classes, because that is where both K5 and K10 first lose the 75% per-class floor. The immediate hard additions are `1-12` at N14 for K5 and `1-1` at N14 for K10.

Current goal status: active, not achieved.

## LEO-Sketch qKNN v42 Results

Timestamp: 2026-07-06 16:03 CST

Important scope statement: all formal samples in this section are target-domain satellite/LEO samples, not clean samples. The primary ADV3B02 feature file already carries `target_channel_view=satellite/LEO` and `uses_target_clean=false`; the new LEO-sketch auxiliary file also has `applies_star_ground_channel=true`, `uses_target_clean=false`, and row-level scenarios `leo_clear_weak`,`leo_low_elev_weak`,`leo_rain_weak`.

New qKNN variant:

- `stable_dualview_v42` inherits v40's source-target transport and hard-pair handling.
- It adds support-only auxiliary-view reliability gating for the LEO-sketch view. The gate computes support leave-one-out accuracy for the primary ADV3B02 space and for the LEO-sketch space. If LEO-sketch is weaker, `effective_aux_score_weight` is reduced automatically.
- This keeps the deployment state compressed: `stored_raw_support_count=0`; LEO raw IQ is used only once at export time to create 96-dim sketches.

LEO-sketch artifact:

| item | value |
|---|---|
| local artifact | `artifacts\features_hardpair_HP08L5_n20_leosketch96.npz` |
| sha256 | `c996e9a39cdd8b9e238abaef2972f78947bf6da3725e58050987fa1a3148d84d` |
| shape | `(11760,96)` |
| sample alignment | `tx_ids`,`rx_ids`,`day_ids`,`eq_ids`,`sig_ids`,`dataset_role`,`sat_scenarios` all match primary ADV3B02 NPZ |
| source rows | ManySig 960, ManyTx 10800 |
| channel | satellite/LEO, `leo_tta_views=5` |
| raw support stored | 0 |

Result summary, max-query N20:

| method | K | old_acc | min_old | seen_new_acc | min_new | eff_aux | aux support LOO mean/min | artifact |
|---|---:|---:|---:|---:|---:|---:|---|---|
| v40 no aux previous best | 5 | 92.00% | 80.00% | 79.80% | 69.33% | 0.000 | n/a | `n20_k5_v40_selective_transport_20260706l.json` |
| v40 no aux previous best | 10 | 91.90% | 82.86% | 84.64% | 72.86% | 0.000 | n/a | `n20_k10_v40_selective_transport_20260706l.json` |
| v40 + ungated LEO-sketch | 5 | 92.00% | 80.00% | 78.87% | 68.00% | 0.220 | 0.00/0.00 | `n20_k5_v40_aux_leosketch96_grid_20260706o.json` |
| v40 + ungated LEO-sketch | 10 | 91.90% | 82.86% | 84.29% | 71.43% | 0.219 | 0.00/0.00 | `n20_k10_v40_aux_leosketch96_grid_20260706o.json` |
| v42 gated LEO-sketch | 5 | 92.00% | 80.00% | 79.80% | 69.33% | 0.000 | 10.77%/0.00% | `n20_k5_v42_leosketch_gated_20260706p.json` |
| v42 gated LEO-sketch | 10 | 91.90% | 82.86% | 84.64% | 72.86% | 0.000 | 17.69%/0.00% | `n20_k10_v42_leosketch_gated_20260706p.json` |

Interpretation:

- Clean rawsketch was a misleadingly strong control. Once the sketch is correctly generated after LEO channel overlay, its support-LOO reliability is very low: 10.77% at K5 and 17.69% at K10, with zero minimum support-class LOO accuracy.
- Ungated LEO-sketch hurts the qKNN floor: K5 min_new drops from 69.33% to 68.00%; K10 min_new drops from 72.86% to 71.43%.
- v42 is the correct compressed qKNN route for combining rawsketch: it uses the LEO-sketch only when support evidence says it is reliable, otherwise it automatically rejects the auxiliary view. This preserves the previous best v40 metrics without hand-setting different K-specific aux weights.
- The active goal is still not achieved because the ten/new-plus class floor remains below 75%. The useful lesson is that the next improvement should not rely on raw waveform sketch similarity after LEO; it should improve the primary ADV3B02 support geometry or add a stronger support-derived pair discriminator.

Per-transmitter detailed performance:

| TX | role | K5 acc | K10 acc |
|---|---|---:|---:|
| `14-10` | old | 93.33% | 91.43% |
| `14-7` | old | 81.33% | 82.86% |
| `20-15` | old | 98.67% | 97.14% |
| `20-19` | old | 80.00% | 82.86% |
| `6-15` | old | 98.67% | 97.14% |
| `8-20` | old | 100.00% | 100.00% |
| `10-10` | new | 89.33% | 95.71% |
| `11-10` | new | 70.67% | 77.14% |
| `18-5` | new | 77.33% | 81.43% |
| `19-3` | new | 70.67% | 84.29% |
| `2-13` | new | 69.33% | 74.29% |
| `2-5` | new | 86.67% | 92.86% |
| `3-8` | new | 94.67% | 94.29% |
| `4-10` | new | 89.33% | 94.29% |
| `8-18` | new | 90.67% | 94.29% |
| `8-3` | new | 77.33% | 77.14% |
| `1-1` | new | 77.33% | 72.86% |
| `1-10` | new | 86.67% | 90.00% |
| `1-11` | new | 89.33% | 95.71% |
| `1-12` | new | 74.67% | 74.29% |
| `1-14` | new | 72.00% | 78.57% |
| `1-15` | new | 73.33% | 85.71% |
| `1-16` | new | 80.00% | 78.57% |
| `1-18` | new | 70.67% | 82.86% |
| `1-19` | new | 86.67% | 90.00% |
| `1-2` | new | 69.33% | 78.57% |

Verification and sync:

| item | result |
|---|---|
| local compile | `C:\Users\lh594\.conda\envs\ssr-gpu\python.exe -m py_compile code\scripts\phase2_raw_iq_sketch_export.py code\scripts\phase2_support_metric_qknn_probe.py` passed |
| remote sync | `phase2_raw_iq_sketch_export.py` and `phase2_support_metric_qknn_probe.py` synced to `/home/szu2070436088/2510044040/CV-SincNet/code/scripts/` |
| remote compile | `/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python -m py_compile code/scripts/phase2_support_metric_qknn_probe.py code/scripts/phase2_raw_iq_sketch_export.py` passed |
| SSH cleanup | no local `ssh.exe` process or established TCP22 connection to N607/lab bridge remained |

Current goal status: active, not achieved.

## LEO-Sketch Correction for qKNN Rawsketch Auxiliary View

Timestamp: 2026-07-06 15:52 CST

Objective: continue optimizing qKNN with a compressed rawsketch auxiliary view, but enforce the CVS Phase2 requirement that support and query samples are satellite/LEO target-view samples, not clean raw IQ samples.

Protocol correction:

- The earlier `features_hardpair_HP08L5_n20_rawsketch96.npz` artifact compressed original raw IQ directly. It remains useful as a diagnostic/control, but it is not formal satellite/LEO evidence and must not be used to claim deployment-view performance.
- Formal qKNN+rawsketch tests from this point use LEO-sketch: original WiSig IQ is read only at export time, each row is transformed through `apply_sat_channel_for_scenario` according to the row-level `sat_scenarios`, then the LEO-transformed signal is compressed to a 96-dim sketch. Stored qKNN support state remains compressed sketches/prototypes, not raw support IQ.
- Default LEO-sketch uses `leo_tta_views=5` to match the source feature manifest's `satellite_tta_policy=rx_light5` as closely as this lightweight sketch route allows.

Local code state:

| item | value |
|---|---|
| Git repo | `E:\type10-7\github_publish\CVS-RFFI-repo` |
| HEAD before edit | `41b6d6205fd31c8f3edcdad59354136f5a4d2f96` |
| Changed file | `code/scripts/phase2_raw_iq_sketch_export.py` |
| Purpose | add LEO channel application before compressed sketch export |
| Local verification | `C:\Users\lh594\.conda\envs\ssr-gpu\python.exe -m py_compile code\scripts\phase2_raw_iq_sketch_export.py code\scripts\phase2_support_metric_qknn_probe.py` passed |

N607 preflight:

| check | result |
|---|---|
| command | `powershell -ExecutionPolicy Bypass -File tools\n607_ssh_preflight.ps1` |
| result | direct N607 preflight passed |
| project root | `/home/szu2070436088/2510044040/CV-SincNet` |
| GPU context | GPUs active with existing workloads; this task is CPU export/eval only and does not launch training |

Planned sync and export:

| local file | remote destination |
|---|---|
| `E:\type10-7\github_publish\CVS-RFFI-repo\code\scripts\phase2_raw_iq_sketch_export.py` | `/home/szu2070436088/2510044040/CV-SincNet/code/scripts/phase2_raw_iq_sketch_export.py` |

Expected output:

`/home/szu2070436088/2510044040/CV-SincNet/automation_reports/CV-SincNet/phase2_qknn_hardpair_n20_20260706/artifacts/features_hardpair_HP08L5_n20_leosketch96.npz`

Current goal status: active, not achieved.

## Raw-IQ Direct kNN Multi-New Diagnostic

Objective: test whether direct kNN on original target-receiver IQ samples can support many-new-class recognition without ADV3B02 features, qKNN feature heads, or LEO overlay. This is a diagnostic baseline, not a deployment-ready head, because direct raw-IQ kNN stores or compares support-like signal samples.

Protocol:

| field | value |
|---|---|
| target receiver | `rx=7-14` |
| source data | N607 raw WiSig compact pkl, `eq=1` |
| old TX | `14-10`,`14-7`,`20-15`,`20-19`,`6-15`,`8-20` |
| new TX sweep | 2,5,10,20 new classes |
| preprocessing | per-sample DC removal, flattened IQ, L2 normalization |
| query policy used for main conclusion | all available target-rx samples after K-shot support removal |
| LEO overlay | not used |
| stored raw support | direct raw-IQ baseline requires raw support-like vectors; compressed-sketch variant below avoids this |

Full-query direct raw-IQ kNN results:

| new classes | total classes | K | old_acc | min_old | new_acc | min_new | query per class |
|---:|---:|---:|---:|---:|---:|---:|---|
| 2 | 8 | 5 | 89.55% | 68.14% | 98.53% | 97.44% | 145-3995 |
| 5 | 11 | 5 | 89.55% | 68.14% | 93.84% | 74.87% | 145-3995 |
| 10 | 16 | 5 | 89.48% | 68.09% | 89.58% | 63.08% | 145-3995 |
| 20 | 26 | 5 | 89.41% | 68.09% | 87.37% | 57.95% | 145-3995 |
| 2 | 8 | 10 | 96.90% | 89.20% | 99.70% | 99.47% | 140-3990 |
| 5 | 11 | 10 | 96.87% | 89.20% | 99.89% | 99.47% | 140-3990 |
| 10 | 16 | 10 | 96.70% | 89.17% | 95.84% | 73.16% | 140-3990 |
| 20 | 26 | 10 | 96.67% | 89.15% | 94.59% | 72.11% | 140-3990 |

Full-query K=10,N20 per-class details:

| TX | role | accuracy | correct/total |
|---|---|---:|---:|
| `1-12` | new | 72.11% | 137/190 |
| `4-10` | new | 73.16% | 139/190 |
| `19-3` | new | 91.58% | 174/190 |
| `1-18` | new | 92.11% | 175/190 |
| `1-1` | new | 93.57% | 131/140 |
| `1-16` | new | 94.74% | 180/190 |
| `1-2` | new | 95.00% | 133/140 |
| `1-10` | new | 95.79% | 182/190 |
| `1-19` | new | 95.79% | 182/190 |
| `8-3` | new | 96.32% | 183/190 |
| `2-5` | new | 98.42% | 187/190 |
| `3-8` | new | 98.95% | 188/190 |
| `1-11` | new | 99.29% | 139/140 |
| `1-15` | new | 99.29% | 139/140 |
| `11-10` | new | 99.47% | 189/190 |
| `1-14` | new | 100.00% | 190/190 |
| `10-10` | new | 100.00% | 140/140 |
| `18-5` | new | 100.00% | 190/190 |
| `2-13` | new | 100.00% | 190/190 |
| `8-18` | new | 100.00% | 190/190 |
| `14-7` | old | 89.15% | 3557/3990 |
| `20-15` | old | 94.24% | 3760/3990 |
| `20-19` | old | 97.27% | 3881/3990 |
| `8-20` | old | 99.67% | 3977/3990 |
| `14-10` | old | 99.85% | 3984/3990 |
| `6-15` | old | 99.85% | 3984/3990 |

Interpretation:

- Direct raw-IQ kNN is a strong separability diagnostic: at K=10,N20 it reaches 94.59% mean new accuracy and 96.67% old accuracy.
- It still misses the active floor target under full-query evaluation: K=10,N20 minimum new-class accuracy is 72.11%, and K=5,N20 minimum new-class accuracy drops to 57.95%.
- The K=10 weak classes are `1-12` and `4-10`, while most other new TX are above 90%. The many-new-class failure is therefore not global collapse, but a small number of hard classes whose raw-IQ neighborhoods overlap or are support-sensitive.
- K=5 is not stable enough for the active goal. Moving from K=10 to K=5 at N20 reduces mean new accuracy from 94.59% to 87.37% and the minimum new-class accuracy from 72.11% to 57.95%.

Compressed raw-IQ sketch diagnostic:

| setting | K | old_acc | min_old | new_acc | min_new | stored raw support | artifact |
|---|---:|---:|---:|---:|---:|---:|---|
| rawsketch96 as primary qKNN feature | 5 | 99.56% | 98.67% | 92.80% | 69.33% | 0 | `n20_k5_rawsketch96_primary_v37_20260706m.json` |
| rawsketch96 as primary qKNN feature | 10 | 99.52% | 98.57% | 99.57% | 97.14% | 0 | `n20_k10_rawsketch96_primary_v37_20260706m.json` |
| ADV3B02 primary + rawsketch96 aux | 5 | 91.56% | 80.00% | 79.07% | 69.33% | 0 | `n20_k5_adv3_aux_rawsketch96_v32_20260706m.json` |
| ADV3B02 primary + rawsketch96 aux | 10 | 92.62% | 84.29% | 87.00% | 74.29% | 0 | `n20_k10_adv3_aux_rawsketch96_v32_20260706m.json` |

Compressed sketch interpretation:

- The sketch route answers the storage concern: the exported feature file stores a 96-dimensional deterministic raw-IQ sketch and reports `stored_raw_support_count=0`; it does not persist original support waveforms.
- As a primary representation, rawsketch96 is extremely strong at K=10,N20 and passes the 75% class floor with margin, but K=5 still fails the floor at 69.33%.
- As an auxiliary view for ADV3B02 qKNN, rawsketch96 helps K=10 from the prior 72.86% floor to 74.29%, but it still misses the 75% requirement and is gated off at K=5.
- Current goal status remains active and not achieved because the required stable K=5/K=10 behavior is not yet satisfied.

## Raw IQ kNN Multi-New Baseline

Objective: test whether direct kNN on original WiSig IQ samples can remain stable as the number of seen-new TX classes increases, before adding ADV3B02 features, qKNN compression, support transport, or LEO channel overlay.

Protocol:

| field | value |
|---|---|
| raw source | N607 `Dataset_WigSig/ManySig.pkl` for old TX; N607 `Dataset_WigSig/ManyTx.pkl` for new TX |
| target receiver | `7-14` |
| old TX | `14-10`,`14-7`,`20-15`,`20-19`,`6-15`,`8-20` |
| new TX order | `10-10`,`11-10`,`18-5`,`19-3`,`2-13`,`2-5`,`3-8`,`4-10`,`8-18`,`8-3`,`1-1`,`1-10`,`1-11`,`1-12`,`1-14`,`1-15`,`1-16`,`1-18`,`1-19`,`1-2` |
| input | original equalized IQ, flattened after per-sample DC removal and L2 normalization |
| classifier | 1-nearest-neighbor over target support samples only |
| pool policy | fixed 80 raw IQ samples per TX for each K; class-count sweep adds classes without resampling existing TX |
| query policy | K=5 uses 75 query samples per class; K=10 uses 70 query samples per class |
| artifact | `artifacts\raw_iq_knn_multinew_rx7-14_fixedpool_20260706.json` |

Class-count sweep:

| K-shot | new class count | old_acc | min_old | new_acc | min_new | overall_acc | verdict |
|---:|---:|---:|---:|---:|---:|---:|---|
| 5 | 2 | 89.78% | 69.33% | 98.67% | 97.33% | 92.00% | old floor weak |
| 5 | 5 | 89.78% | 69.33% | 93.07% | 69.33% | 91.27% | new floor below 75% |
| 5 | 10 | 89.56% | 69.33% | 89.87% | 65.33% | 89.75% | failed |
| 5 | 20 | 89.56% | 69.33% | 87.93% | 58.67% | 88.31% | failed |
| 10 | 2 | 96.43% | 87.14% | 100.00% | 100.00% | 97.32% | pass on this easy count |
| 10 | 5 | 96.43% | 87.14% | 100.00% | 100.00% | 98.05% | pass on this count |
| 10 | 10 | 96.19% | 87.14% | 96.29% | 70.00% | 96.25% | floor failed |
| 10 | 20 | 96.19% | 87.14% | 94.79% | 65.71% | 95.11% | floor failed |

Detailed N20 per-class results:

| TX | role | K5 acc | K5 correct/total | K10 acc | K10 correct/total |
|---|---|---:|---:|---:|---:|
| `14-10` | old | 97.33% | 73/75 | 100.00% | 70/70 |
| `14-7` | old | 77.33% | 58/75 | 87.14% | 61/70 |
| `20-15` | old | 97.33% | 73/75 | 92.86% | 65/70 |
| `20-19` | old | 97.33% | 73/75 | 97.14% | 68/70 |
| `6-15` | old | 69.33% | 52/75 | 100.00% | 70/70 |
| `8-20` | old | 98.67% | 74/75 | 100.00% | 70/70 |
| `10-10` | new | 100.00% | 75/75 | 100.00% | 70/70 |
| `11-10` | new | 96.00% | 72/75 | 100.00% | 70/70 |
| `18-5` | new | 100.00% | 75/75 | 100.00% | 70/70 |
| `19-3` | new | 68.00% | 51/75 | 95.71% | 67/70 |
| `2-13` | new | 98.67% | 74/75 | 100.00% | 70/70 |
| `2-5` | new | 72.00% | 54/75 | 100.00% | 70/70 |
| `3-8` | new | 100.00% | 75/75 | 98.57% | 69/70 |
| `4-10` | new | 58.67% | 44/75 | 70.00% | 49/70 |
| `8-18` | new | 98.67% | 74/75 | 100.00% | 70/70 |
| `8-3` | new | 97.33% | 73/75 | 97.14% | 68/70 |
| `1-1` | new | 90.67% | 68/75 | 94.29% | 66/70 |
| `1-10` | new | 97.33% | 73/75 | 98.57% | 69/70 |
| `1-11` | new | 97.33% | 73/75 | 100.00% | 70/70 |
| `1-12` | new | 70.67% | 53/75 | 65.71% | 46/70 |
| `1-14` | new | 94.67% | 71/75 | 100.00% | 70/70 |
| `1-15` | new | 73.33% | 55/75 | 100.00% | 70/70 |
| `1-16` | new | 92.00% | 69/75 | 91.43% | 64/70 |
| `1-18` | new | 66.67% | 50/75 | 91.43% | 64/70 |
| `1-19` | new | 96.00% | 72/75 | 95.71% | 67/70 |
| `1-2` | new | 90.67% | 68/75 | 97.14% | 68/70 |

Interpretation:

- Direct raw-IQ kNN is much stronger than the current hard-pair feature-space qKNN on mean new-class accuracy, especially at K=10.
- It still does not satisfy the active stability target because N10/N20 floors fall below 75%. The worst N20 classes are `4-10` at K5 and `1-12` at K10.
- The result suggests that the original IQ contains separability that the current ADV3B02 feature/qKNN route is not preserving for hard classes. The next optimization should not blindly increase K or tune per-K constants; it should compress raw-IQ-neighborhood evidence or distill raw-neighbor structure into the qKNN memory without storing original support samples.
- This diagnostic used clean/equalized raw WiSig samples only. It is not a satellite-channel-overlaid deployment result.

Full-query recheck:

The fixed-80 result above is aligned with the current feature export pool. A second run used all available equalized raw IQ samples for the same target receiver. ManySig old TX have 4000 samples per class, while the selected ManyTx new TX under `rx=7-14` have only 150 or 200 samples per class; therefore the full-query denominators are imbalanced by TX.

Artifact: `artifacts\raw_iq_knn_multinew_rx7-14_fullquery_20260706.json`.

| K-shot | new class count | query/class range | old_acc | min_old | new_acc | min_new | overall_acc | verdict |
|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 5 | 2 | 145-3995 | 89.55% | 68.14% | 98.53% | 97.44% | 89.68% | old floor weak |
| 5 | 5 | 145-3995 | 89.55% | 68.14% | 93.84% | 74.87% | 89.71% | borderline, floor below 75% |
| 5 | 10 | 145-3995 | 89.48% | 68.09% | 89.58% | 63.08% | 89.49% | failed |
| 5 | 20 | 145-3995 | 89.41% | 68.09% | 87.37% | 57.95% | 89.14% | failed |
| 10 | 2 | 140-3990 | 96.90% | 89.20% | 99.70% | 99.47% | 96.94% | pass |
| 10 | 5 | 140-3990 | 96.87% | 89.20% | 99.89% | 99.47% | 96.98% | pass |
| 10 | 10 | 140-3990 | 96.70% | 89.17% | 95.84% | 73.16% | 96.63% | floor failed |
| 10 | 20 | 140-3990 | 96.67% | 89.15% | 94.59% | 72.11% | 96.40% | floor failed |

Full-query N20 per-class results:

| TX | role | available eq1 | K5 acc | K5 correct/total | K10 acc | K10 correct/total |
|---|---|---:|---:|---:|---:|---:|
| `14-10` | old | 4000 | 99.90% | 3991/3995 | 99.85% | 3984/3990 |
| `14-7` | old | 4000 | 73.87% | 2951/3995 | 89.15% | 3557/3990 |
| `20-15` | old | 4000 | 97.30% | 3887/3995 | 94.24% | 3760/3990 |
| `20-19` | old | 4000 | 97.62% | 3900/3995 | 97.27% | 3881/3990 |
| `6-15` | old | 4000 | 68.09% | 2720/3995 | 99.85% | 3984/3990 |
| `8-20` | old | 4000 | 99.67% | 3982/3995 | 99.67% | 3977/3990 |
| `10-10` | new | 150 | 100.00% | 145/145 | 100.00% | 140/140 |
| `11-10` | new | 200 | 96.41% | 188/195 | 99.47% | 189/190 |
| `18-5` | new | 200 | 100.00% | 195/195 | 100.00% | 190/190 |
| `19-3` | new | 200 | 72.82% | 142/195 | 91.58% | 174/190 |
| `2-13` | new | 200 | 98.46% | 192/195 | 100.00% | 190/190 |
| `2-5` | new | 200 | 74.36% | 145/195 | 98.42% | 187/190 |
| `3-8` | new | 200 | 98.97% | 193/195 | 98.95% | 188/190 |
| `4-10` | new | 200 | 57.95% | 113/195 | 73.16% | 139/190 |
| `8-18` | new | 200 | 98.97% | 193/195 | 100.00% | 190/190 |
| `8-3` | new | 200 | 93.85% | 183/195 | 96.32% | 183/190 |
| `1-1` | new | 150 | 89.66% | 130/145 | 93.57% | 131/140 |
| `1-10` | new | 200 | 96.41% | 188/195 | 95.79% | 182/190 |
| `1-11` | new | 150 | 97.93% | 142/145 | 99.29% | 139/140 |
| `1-12` | new | 200 | 74.87% | 146/195 | 72.11% | 137/190 |
| `1-14` | new | 200 | 93.85% | 183/195 | 100.00% | 190/190 |
| `1-15` | new | 150 | 64.83% | 94/145 | 99.29% | 139/140 |
| `1-16` | new | 200 | 95.38% | 186/195 | 94.74% | 180/190 |
| `1-18` | new | 200 | 60.00% | 117/195 | 92.11% | 175/190 |
| `1-19` | new | 200 | 95.90% | 187/195 | 95.79% | 182/190 |
| `1-2` | new | 150 | 87.59% | 127/145 | 95.00% | 133/140 |

Full-query interpretation:

- With direct raw IQ, mean new-class accuracy remains high as new classes increase, especially for K=10: 99.89% at 5 new classes, 95.84% at 10, and 94.59% at 20.
- The active floor target is still not met. At 10 new classes, K10 min_new is 73.16%; at 20 new classes, K10 min_new is 72.11%. K5 is more fragile, falling to 57.95% at 20 new classes.
- The strongest failure signal is not mean collapse but per-class floor collapse. `4-10` remains the hardest new TX under both K5 and K10; `1-12`,`1-15`,`1-18`,`19-3`,`2-5` are K5-sensitive.
- This supports a raw-neighborhood-distillation direction for qKNN: preserve the discriminative local raw-IQ geometry in compressed descriptors or auxiliary pair signatures, without storing original support IQ on-board.

## v39 Source-Target Residual Transport Diagnostic

Timestamp: 2026-07-06 CST. Objective: continue the active qKNN goal by adding a compressed source-to-target residual transport module, without adding K anchors and without storing raw support. The mechanism estimates a compact domain-shift basis from old-class source prototypes and target-old support prototypes, then scores seen-new classes in the transported residual space. The adaptive policy is still single-route for `K=5` and `K=10`: weight, rank, residual strength, and shift strength are derived from support `K`, new-class load, and support geometry, not from per-K hand tuning.

Local implementation:

| file | change |
|---|---|
| `code/scripts/phase2_support_metric_qknn_probe.py` | add `_source_target_residual_transport_scores`; register `dualview_support_v39` / `stable_dualview_v39`; emit `source_target_transport_*` metrics in JSON/CSV |

Verification and artifacts:

| command/artifact | status |
|---|---|
| `C:\Users\lh594\.conda\envs\ssr-gpu\python.exe -m py_compile code\scripts\phase2_support_metric_qknn_probe.py` | PASS |
| `artifacts\n20_k5_v39_sourcetransport_20260706k.csv/json` | completed |
| `artifacts\n20_k10_v39_sourcetransport_20260706k.csv/json` | completed |

Main comparison against the current v37/v38 line:

| route | K | old_acc | min_old | new_acc | min_new | transport weight | transport rank | transport scalars | raw support | verdict |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| v37 neighbor contrast | 5 | 92.00% | 80.00% | 80.00% | 69.33% | 0 | - | 0 | 0 | reference |
| v38 risk contrast | 5 | 92.00% | 80.00% | 80.00% | 69.33% | 0 | - | 0 | 0 | tied reference |
| v39 source-target transport | 5 | 92.00% | 80.00% | 79.80% | 69.33% | 0.0600 | 3 | 4000 | 0 | not promoted |
| v37 neighbor contrast | 10 | 91.90% | 82.86% | 84.43% | 72.86% | 0 | - | 0 | 0 | reference |
| v38 risk contrast | 10 | 91.90% | 82.86% | 84.29% | 72.86% | 0 | - | 0 | 0 | not promoted |
| v39 source-target transport | 10 | 91.90% | 82.86% | 84.50% | 72.86% | 0.0547 | 3 | 4000 | 0 | small mean gain, floor still fails |

v39 detailed per-TX performance:

| K | TX | role | acc |
|---:|---|---|---:|
| 5 | 14-10 | old | 93.33% |
| 5 | 14-7 | old | 81.33% |
| 5 | 20-15 | old | 98.67% |
| 5 | 20-19 | old | 80.00% |
| 5 | 6-15 | old | 98.67% |
| 5 | 8-20 | old | 100.00% |
| 5 | 1-1 | seen-new | 78.67% |
| 5 | 1-10 | seen-new | 86.67% |
| 5 | 1-11 | seen-new | 89.33% |
| 5 | 1-12 | seen-new | 73.33% |
| 5 | 1-14 | seen-new | 72.00% |
| 5 | 1-15 | seen-new | 73.33% |
| 5 | 1-16 | seen-new | 80.00% |
| 5 | 1-18 | seen-new | 70.67% |
| 5 | 1-19 | seen-new | 86.67% |
| 5 | 1-2 | seen-new | 69.33% |
| 5 | 10-10 | seen-new | 89.33% |
| 5 | 11-10 | seen-new | 70.67% |
| 5 | 18-5 | seen-new | 77.33% |
| 5 | 19-3 | seen-new | 70.67% |
| 5 | 2-13 | seen-new | 69.33% |
| 5 | 2-5 | seen-new | 86.67% |
| 5 | 3-8 | seen-new | 94.67% |
| 5 | 4-10 | seen-new | 89.33% |
| 5 | 8-18 | seen-new | 90.67% |
| 5 | 8-3 | seen-new | 77.33% |
| 10 | 14-10 | old | 91.43% |
| 10 | 14-7 | old | 82.86% |
| 10 | 20-15 | old | 97.14% |
| 10 | 20-19 | old | 82.86% |
| 10 | 6-15 | old | 97.14% |
| 10 | 8-20 | old | 100.00% |
| 10 | 1-1 | seen-new | 72.86% |
| 10 | 1-10 | seen-new | 90.00% |
| 10 | 1-11 | seen-new | 95.71% |
| 10 | 1-12 | seen-new | 74.29% |
| 10 | 1-14 | seen-new | 78.57% |
| 10 | 1-15 | seen-new | 85.71% |
| 10 | 1-16 | seen-new | 78.57% |
| 10 | 1-18 | seen-new | 81.43% |
| 10 | 1-19 | seen-new | 90.00% |
| 10 | 1-2 | seen-new | 78.57% |
| 10 | 10-10 | seen-new | 95.71% |
| 10 | 11-10 | seen-new | 75.71% |
| 10 | 18-5 | seen-new | 81.43% |
| 10 | 19-3 | seen-new | 84.29% |
| 10 | 2-13 | seen-new | 74.29% |
| 10 | 2-5 | seen-new | 92.86% |
| 10 | 3-8 | seen-new | 94.29% |
| 10 | 4-10 | seen-new | 94.29% |
| 10 | 8-18 | seen-new | 94.29% |
| 10 | 8-3 | seen-new | 77.14% |

Interpretation:

- v39 is deployment-aligned as a compressed KNN variant: `stored_raw_support_count=0`; extra persistent state is transport basis, mean/center vectors, and new residual prototypes. It does not keep raw support IQ or raw support feature rows.
- The source-to-target residual route has useful signal but is not yet stable. At `K=10`, mean seen-new accuracy improves from 84.43% to 84.50%, but the floor remains 72.86%. At `K=5`, mean seen-new drops slightly from 80.00% to 79.80%, with the same 69.33% floor.
- Per-class movement shows why it fails the active goal: transport helps some classes such as `1-2`, `19-3`, and `1-15` at `K=10`, but harms or fails to repair hard neighbors such as `1-1`, `1-12`, `2-13`, `11-10`, and `1-18`. A global transport addition is therefore too blunt.
- Next aligned step should not abandon qKNN. It should turn v39 into a selective transport gate: enable source-target residual scores only when support-only LOO or class-neighborhood confidence predicts benefit for the affected class pair, and suppress it for reciprocal hard neighbors where it trades one low-floor class for another.

Current goal status: active, not achieved.

## v37 Support-Seed Sensitivity Diagnostic

Timestamp: 2026-07-06 CST. Objective: check whether the remaining N20 floor failure in the compressed v37 qKNN route is caused by a single unlucky support split or by stable hard-class separability. The diagnostic keeps the same Stage2-C boundary: target-old and seen-new support/query come from the target receiver domain, the LEO feature view is used, `K=5` and `K=10` are the only anchors, and raw support storage remains zero.

Artifacts:

| artifact | status |
|---|---|
| `artifacts\n20_k5_v37_seedscan5_20260706i.csv/json` | completed |
| `artifacts\n20_k10_v37_seedscan5_20260706i.csv/json` | completed |

Per-seed best rows within each seed, ranked by `min_new`, then `new_acc`, then `old_acc`:

| K | seed | old_acc | min_old | new_acc | min_new | weakest new classes | raw support |
|---:|---:|---:|---:|---:|---:|---|---:|
| 5 | 421038 | 92.00% | 80.00% | 80.00% | 69.33% | `2-13` 69.33%,`1-18` 70.67%,`1-2` 70.67%,`19-3` 70.67%,`11-10` 72.00% | 0 |
| 5 | 421039 | 93.33% | 84.00% | 79.87% | 60.00% | `1-1` 60.00%,`1-12` 61.33%,`2-13` 64.00%,`1-18` 69.33%,`1-2` 70.67% | 0 |
| 5 | 421040 | 92.89% | 80.00% | 77.40% | 58.67% | `2-13` 58.67%,`1-12` 61.33%,`1-1` 62.67%,`11-10` 64.00%,`1-18` 66.67% | 0 |
| 5 | 421041 | 92.67% | 80.00% | 80.40% | 56.00% | `1-12` 56.00%,`2-13` 60.00%,`1-1` 64.00%,`1-14` 65.33%,`1-18` 72.00% | 0 |
| 5 | 421042 | 91.56% | 77.33% | 73.73% | 46.67% | `1-14` 46.67%,`8-3` 46.67%,`2-13` 49.33%,`1-1` 57.33%,`2-5` 60.00% | 0 |
| 10 | 421057 | 91.90% | 82.86% | 84.43% | 72.86% | `1-1` 72.86%,`1-12` 74.29%,`1-2` 75.71%,`2-13` 75.71%,`11-10` 77.14% | 0 |
| 10 | 421058 | 92.62% | 80.00% | 85.86% | 68.57% | `2-13` 68.57%,`1-14` 74.29%,`1-12` 75.71%,`1-2` 75.71%,`1-10` 82.86% | 0 |
| 10 | 421059 | 93.57% | 82.86% | 79.86% | 61.43% | `1-18` 61.43%,`1-12` 64.29%,`1-1` 65.71%,`1-2` 67.14%,`11-10` 67.14% | 0 |
| 10 | 421060 | 94.29% | 84.29% | 83.64% | 68.57% | `2-13` 68.57%,`1-12` 71.43%,`1-2` 74.29%,`8-3` 74.29%,`1-1` 75.71% | 0 |
| 10 | 421061 | 93.33% | 82.86% | 83.14% | 70.00% | `1-1` 70.00%,`1-12` 72.86%,`1-18` 72.86%,`11-10` 72.86%,`2-13` 72.86% | 0 |

Stability summary:

| K | best min_new | median min_new | worst min_new | new_acc range | old_acc range | seeds passing old>=80 and min_new>=75 |
|---:|---:|---:|---:|---:|---:|---:|
| 5 | 69.33% | 58.67% | 46.67% | 73.73%-80.40% | 91.56%-93.33% | 0/5 |
| 10 | 72.86% | 68.57% | 61.43% | 79.86%-85.86% | 91.90%-94.29% | 0/5 |

Repeated weak classes below 75% among the per-seed best rows:

| K | repeated weak classes |
|---:|---|
| 5 | `2-13` 5/5,min 49.33%;`1-12` 5/5,min 56.00%;`1-18` 5/5,min 66.67%;`1-14` 4/5,min 46.67%;`1-1` 4/5,min 57.33%;`11-10` 4/5,min 64.00%;`1-2` 4/5,min 69.33% |
| 10 | `1-12` 4/5,min 64.29%;`2-13` 4/5,min 68.57%;`1-1` 3/5,min 65.71%;`8-3` 3/5,min 72.86%;`1-18` 2/5,min 61.43%;`1-2` 2/5,min 67.14%;`11-10` 2/5,min 67.14% |

Interpretation:

- The old-class side is stable and above the OLD80 stage gate in almost every seed. The remaining failure is the new-class floor, not old-class retention.
- No seed reaches the active floor target. This rules out the simple explanation that v37 failed only because of one unlucky support draw.
- The repeated weak classes are concentrated in the same hard neighborhood: `2-13`,`1-12`,`1-1`,`1-18`,`11-10`,`1-2`, with `1-14` and `8-3` appearing under more fragile support splits.
- The next aligned qKNN step should not add larger K or query-label tuning. It should target support-only hard-neighborhood registration or representation repair, while keeping a single adaptive policy for `K=5` and `K=10` and `stored_raw_support_count=0`.

Current goal status: active, not achieved.

## v38 Risk-Scaled Neighbor Contrast Diagnostic

Timestamp: 2026-07-06 CST. Objective: test whether v37's compressed one-vs-neighborhood contrast is too weak for the repeated hard classes. The v38 diagnostic keeps the same N20 split, `K=5`/`K=10` anchors, maximum query counts, role-balanced assignment, target-domain support/query boundary, and `stored_raw_support_count=0`. It adds a support-only risk scale to each selected neighbor-contrast direction. The scale is derived from support LOO class risk and is gated by `low_k_gate`, so it adapts to K and class count instead of using separate K-specific constants.

Code change:

| file | change |
|---|---|
| `code/scripts/phase2_support_metric_qknn_probe.py` | add `dualview_support_v38` / `stable_dualview_v38`; add `risk_scale` to `_support_neighbor_contrast_scores`; emit scaled contrast neighborhoods such as `1-1@1.75->...` |

Verification:

| command | result |
|---|---|
| `C:\Users\lh594\.conda\envs\ssr-gpu\python.exe -m py_compile code\scripts\phase2_support_metric_qknn_probe.py` | PASS |

Artifacts:

| artifact | status |
|---|---|
| `artifacts\n20_k5_v38_riskcontrast_20260706j.csv/json` | completed, initial risk scale |
| `artifacts\n20_k10_v38_riskcontrast_20260706j.csv/json` | completed, initial risk scale |
| `artifacts\n20_k5_v38_lowkgated_riskcontrast_20260706j.csv/json` | completed, low-K gated risk scale |
| `artifacts\n20_k10_v38_lowkgated_riskcontrast_20260706j.csv/json` | completed, low-K gated risk scale |

Best-row comparison against v37:

| route | K | old_acc | min_old | new_acc | min_new | scaled contrast | raw support | verdict |
|---|---:|---:|---:|---:|---:|---|---:|---|
| v37 neighbor contrast | 5 | 92.00% | 80.00% | 80.00% | 69.33% | no | 0 | reference |
| v38 low-K risk contrast | 5 | 92.00% | 80.00% | 80.00% | 69.33% | yes, scale up to 1.75 | 0 | no change |
| v37 neighbor contrast | 10 | 91.90% | 82.86% | 84.43% | 72.86% | no | 0 | reference |
| v38 low-K risk contrast | 10 | 91.90% | 82.86% | 84.29% | 72.86% | yes, scale up to 1.67 | 0 | slight mean loss |

v38 weak classes:

| K | weakest new classes |
|---:|---|
| 5 | `2-13` 69.33%,`1-18` 70.67%,`1-2` 70.67%,`19-3` 70.67%,`11-10` 72.00%,`1-15` 73.33%,`1-12` 74.67%,`1-14` 74.67% |
| 10 | `1-1` 72.86%,`1-12` 74.29%,`2-13` 74.29%,`1-2` 75.71%,`11-10` 77.14%,`8-3` 77.14%,`1-14` 78.57%,`1-16` 78.57% |

Interpretation:

- v38 is a valid compressed diagnostic: no raw support is stored, and the persistent additional state is still limited to contrast directions, neighbor IDs, and scalar scales.
- It is not a promotable improvement. K5 remains unchanged even with scale up to 1.75, which means the current contrast direction does not flip the hard assignment boundary for the weakest classes. K10 keeps the same floor but loses 0.14pp mean new accuracy.
- This strengthens the negative evidence: post-hoc support-neighborhood score scaling is insufficient. The next useful step should change the registration geometry itself, for example class-conditional residual prototype transport or a source-informed hard-neighborhood representation repair, rather than only increasing correction magnitude.

Current goal status: active, not achieved.

## Adaptive v37 Neighbor Contrast and Query-Pair Diagnostic

Objective: continue the active qKNN goal under the same N20/K5/K10 setting, without adding more K anchors and without storing raw support samples. The tested v37 mechanism adds a support-derived one-vs-neighborhood contrast direction for weak new classes. The stored state is compact: direction vector, neighbor IDs, and scalar gates per selected target class.

Artifacts:

| artifact | status |
|---|---|
| `artifacts\n20_k5_v37_neighbor_contrast_20260706h.json` | completed |
| `artifacts\n20_k10_v37_neighbor_contrast_20260706h.json` | completed |
| `artifacts\n20_k5_v37_pred_diag_20260706i_predictions.csv` | completed |
| `artifacts\n20_k10_v37_pred_diag_20260706i_predictions.csv` | completed |
| `artifacts\n20_k5_v37_qpaircluster_single_20260706i.json` | completed negative diagnostic |
| `artifacts\n20_k10_v37_qpaircluster_single_20260706i.json` | completed negative diagnostic |

v36 to v37 comparison:

| route | K | old_acc | min_old | new_acc | min_new | compact neighbor contrast | raw support stored | verdict |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| v36 adaptive ridge | 5 | 92.00% | 80.00% | 79.93% | 69.33% | 0 | 0 | prior best |
| v37 neighbor contrast | 5 | 92.00% | 80.00% | 80.00% | 69.33% | 6 targets,990 scalars | 0 | tiny mean gain, floor unchanged |
| v36 adaptive ridge | 10 | 91.90% | 82.86% | 84.43% | 72.86% | 0 | 0 | prior best |
| v37 neighbor contrast | 10 | 91.90% | 82.86% | 84.43% | 72.86% | 5 targets,826 scalars | 0 | no change |

N20 v37 weak new classes:

| TX | K5 acc | K10 acc | main observed confusion |
|---|---:|---:|---|
| `2-13` | 69.33% | 75.71% | K5 to `1-2`,`11-10`; K10 to `11-10`,`1-2` |
| `1-18` | 70.67% | 84.29% | K5 to `11-10`,`2-13` |
| `1-2` | 70.67% | 75.71% | K5 to `1-19`,`1-18`; K10 to `2-13`,`1-19` |
| `19-3` | 70.67% | 82.86% | K5/K10 mostly to `1-15` |
| `11-10` | 72.00% | 77.14% | to `1-18`,`2-13` |
| `1-15` | 73.33% | 84.29% | K5 mostly to `19-3` |
| `1-12` | 74.67% | 74.29% | to `1-1`,`8-3` |
| `1-14` | 74.67% | 78.57% | to `1-10`,`1-16`,`18-5` |
| `1-1` | 77.33% | 72.86% | to `1-12`,`8-3` |

Query-pair cluster diagnostic:

| route | K | old_acc | min_old | new_acc | min_new | changed predictions | affected pairs | verdict |
|---|---:|---:|---:|---:|---:|---:|---|---|
| v37 + query-pair cluster | 5 | 92.00% | 80.00% | 79.67% | 69.33% | 8 | `1-1<->1-12`;`8-3<->1-12` | worse than v37 |
| v37 + query-pair cluster | 10 | 91.90% | 82.86% | 84.21% | 70.00% | 4 | `1-1<->1-12` | worse than v37 |

Interpretation:

- v37 is a valid incremental compressed qKNN extension: no raw support storage, small K5 mean gain, no K10 harm, but it does not solve the floor target.
- The weak classes are not a global quota-count problem because role-balanced assignment already predicts exactly the same count per class. The failure is local ordering inside hard pair or small-neighborhood groups.
- Query-pair cluster refinement is rejected for the next route: it moved predictions inside `1-1/1-12/8-3` but lowered the K10 floor from 72.86% to 70.00%.
- Next route should not rely on batch-local query cluster centers. A better v38 direction is support-audited hard-neighborhood calibration with stricter rollback, or support/enrollment quality repair, because the negative diagnostic shows unlabeled query-pair relabeling is unstable.

Current goal status: active, not achieved.

## Raw IQ kNN Baseline

Objective: test whether plain kNN over original/raw signal samples can handle multi-new-class Stage2-C enrollment before further qKNN optimization. This is a diagnostic baseline only: it stores raw support samples and therefore does not satisfy the compressed-onboard-state innovation target.

Artifacts:

| artifact | status |
|---|---|
| `artifacts\rawiq_knn_n2_k5k10_seed421027_grid20260706.json` | completed |
| `artifacts\rawiq_knn_n5_k5k10_seed421027_grid20260706.json` | completed |
| `artifacts\rawiq_knn_n10_k5k10_seed421027_grid20260706.json` | completed |
| `artifacts\rawiq_knn_n20_k5k10_seed421027_grid20260706.json` | completed |

Configuration:

| field | value |
|---|---|
| old classes | 6 ManySig old TX |
| new class counts | 2,5,10,20 ManyTx non-old TX |
| K | 5,10 |
| query count | max available after support: 75 per class for K=5,70 per class for K=10 |
| classifier | pure kNN, vote_k=1, raw support samples stored |
| views | clean control and LEO deployment view |
| seed | 421027 |

Summary by new-class count:

| new classes | view | K | old_acc | min_old | seen_new_acc | min_new | raw support storage |
|---:|---|---:|---:|---:|---:|---:|---:|
| 2 | clean | 5 | 91.33% | 62.67% | 100.00% | 100.00% | 40x512 |
| 2 | LEO | 5 | 25.11% | 17.33% | 26.00% | 22.67% | 40x512 |
| 2 | clean | 10 | 97.38% | 92.86% | 100.00% | 100.00% | 80x512 |
| 2 | LEO | 10 | 26.43% | 14.29% | 28.57% | 28.57% | 80x512 |
| 5 | clean | 5 | 91.33% | 62.67% | 90.93% | 56.00% | 55x512 |
| 5 | LEO | 5 | 29.78% | 14.67% | 19.73% | 10.67% | 55x512 |
| 5 | clean | 10 | 97.38% | 92.86% | 99.71% | 98.57% | 110x512 |
| 5 | LEO | 10 | 34.29% | 20.00% | 41.43% | 12.86% | 110x512 |
| 10 | clean | 5 | 91.33% | 62.67% | 81.47% | 44.00% | 80x512 |
| 10 | LEO | 5 | 26.67% | 13.33% | 21.33% | 9.33% | 80x512 |
| 10 | clean | 10 | 97.14% | 92.86% | 96.00% | 75.71% | 160x512 |
| 10 | LEO | 10 | 27.62% | 14.29% | 35.14% | 8.57% | 160x512 |
| 20 | clean | 5 | 91.33% | 62.67% | 83.73% | 44.00% | 130x512 |
| 20 | LEO | 5 | 20.44% | 14.67% | 19.80% | 2.67% | 130x512 |
| 20 | clean | 10 | 97.14% | 92.86% | 93.43% | 68.57% | 260x512 |
| 20 | LEO | 10 | 25.48% | 15.71% | 27.43% | 1.43% | 260x512 |

N20 LEO per-class detail:

| TX | K5 acc | K10 acc |
|---|---:|---:|
| `14-10` | 25.33% | 21.43% |
| `14-7` | 17.33% | 18.57% |
| `20-15` | 20.00% | 15.71% |
| `20-19` | 14.67% | 21.43% |
| `6-15` | 17.33% | 32.86% |
| `8-20` | 28.00% | 42.86% |
| `10-10` | 4.00% | 11.43% |
| `11-10` | 24.00% | 18.57% |
| `18-5` | 50.67% | 78.57% |
| `19-3` | 17.33% | 21.43% |
| `2-13` | 22.67% | 31.43% |
| `2-5` | 21.33% | 22.86% |
| `3-8` | 29.33% | 18.57% |
| `4-10` | 9.33% | 8.57% |
| `8-18` | 40.00% | 58.57% |
| `8-3` | 12.00% | 10.00% |
| `1-1` | 2.67% | 22.86% |
| `1-10` | 5.33% | 37.14% |
| `1-11` | 12.00% | 25.71% |
| `1-12` | 16.00% | 14.29% |
| `1-14` | 14.67% | 28.57% |
| `1-15` | 24.00% | 32.86% |
| `1-16` | 38.67% | 54.29% |
| `1-18` | 12.00% | 15.71% |
| `1-19` | 29.33% | 35.71% |
| `1-2` | 10.67% | 1.43% |

Interpretation:

- Plain raw-sample kNN is strong on clean control, especially at K=10, but collapses under the LEO deployment view.
- The collapse is not only a mean-accuracy issue: in N20 LEO, many new classes are below 25%, and the minimum new-class accuracy is 2.67% for K=5 and 1.43% for K=10.
- This validates the need for qKNN/compressed support-state adaptation rather than storing raw support samples. The raw baseline also violates the onboard storage direction: N20,K=10 stores 260 raw support vectors, or 133,120 scalar sample values, before metadata.
- Current goal remains active and unmet; raw kNN should remain a negative control, not a candidate route.

## Adaptive v36 Ridge-Graph qKNN Check

Timestamp: 2026-07-06 CST. Objective: continue the active qKNN goal by adding one unified adaptive policy for `K=5` and `K=10`, without adding more K anchors and without storing raw support samples. The tested policy is `stable_dualview_v36`: it inherits the compressed v32/qKNN base and adds AR2 adaptive ridge registration plus K/class-load adaptive label propagation and support-LOO pair-linear correction.

Mechanism:

| component | adaptive rule | deployment state |
|---|---|---|
| ridge head | weight scales with many-new load and support hardness, then decays with `K` reliability; alpha increases with `K` reliability | compact ridge coefficients only |
| label propagation | weight increases from low-shot K5 toward more reliable K10, bounded by class load and support hardness | no query labels stored; graph scores are transient |
| pair-linear support correction | top pairs selected from support-LOO evidence; top pair count adapts with new-class count and is capped at 8 | pair coefficients only |
| support storage | unchanged compressed qKNN state | `stored_raw_support_count=0` |

Artifacts:

| artifact | status |
|---|---|
| `artifacts\n20_k5_v36_adaptive_ridge4_20260706g.csv/json` | completed |
| `artifacts\n20_k10_v36_adaptive_ridge4_20260706g.csv/json` | completed |

Comparison against the previous per-K reference rows:

| route | K | old_acc | min_old | new_acc | min_new | ridge | labelprop | pair-linear | raw support |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| v32 reference | 5 | 91.56% | 80.00% | 79.07% | 69.33% | 0.0000 | 0.0400 | 0.0100 | 0 |
| v36 adaptive ridge-graph | 5 | 92.00% | 80.00% | 79.93% | 69.33% | 0.1000 | 0.0350 | 0.0095 | 0 |
| v32 reference | 10 | 92.38% | 82.86% | 84.29% | 72.86% | 0.0000 | 0.0800 | 0.0100 | 0 |
| v36 adaptive ridge-graph | 10 | 91.90% | 82.86% | 84.43% | 72.86% | 0.0583 | 0.0800 | 0.0085 | 0 |

Per-TX details for v36:

| TX | role | K5 acc | K10 acc |
|---|---|---:|---:|
| `14-10` | old | 93.33% | 91.43% |
| `14-7` | old | 81.33% | 82.86% |
| `20-15` | old | 98.67% | 97.14% |
| `20-19` | old | 80.00% | 82.86% |
| `6-15` | old | 98.67% | 97.14% |
| `8-20` | old | 100.00% | 100.00% |
| `10-10` | new | 90.67% | 95.71% |
| `11-10` | new | 72.00% | 77.14% |
| `18-5` | new | 76.00% | 81.43% |
| `19-3` | new | 70.67% | 82.86% |
| `2-13` | new | 69.33% | 75.71% |
| `2-5` | new | 85.33% | 92.86% |
| `3-8` | new | 94.67% | 94.29% |
| `4-10` | new | 89.33% | 92.86% |
| `8-18` | new | 90.67% | 94.29% |
| `8-3` | new | 77.33% | 77.14% |
| `1-1` | new | 77.33% | 72.86% |
| `1-10` | new | 86.67% | 90.00% |
| `1-11` | new | 89.33% | 95.71% |
| `1-12` | new | 74.67% | 74.29% |
| `1-14` | new | 74.67% | 78.57% |
| `1-15` | new | 73.33% | 84.29% |
| `1-16` | new | 78.67% | 78.57% |
| `1-18` | new | 72.00% | 84.29% |
| `1-19` | new | 86.67% | 90.00% |
| `1-2` | new | 69.33% | 75.71% |

Interpretation:

- v36 is useful but not sufficient. It improves K5 mean new accuracy by +0.86pp and K10 mean new accuracy by +0.14pp versus the named reference rows, while preserving `stored_raw_support_count=0`.
- The minimum new-class floor does not improve: K5 remains 69.33% and K10 remains 72.86%, so the active floor target is still unmet.
- K5 remains within 5pp of K10 on mean new accuracy: 79.93% versus 84.43%, gap 4.50pp.
- K10 old mean decreases from 92.38% to 91.90%, although `min_old` remains 82.86% and the OLD80 gate remains satisfied.
- Remaining bottleneck classes are still local hard-neighborhood cases: K5 is limited by `2-13` and `1-2` at 69.33%, followed by `19-3`, `11-10`, and `1-18`; K10 is limited by `1-1` at 72.86% and `1-12` at 74.29%.
- Next aligned step should not increase K. It should add a support-only class-neighborhood floor objective or representation-side hard-neighborhood repair targeted at the repeated weak classes, while keeping v36's adaptive single-policy structure and zero raw-support storage.

Current goal status: active, not achieved.

## v32 Assignment/Ridge Follow-up Diagnostics

Timestamp: 2026-07-06 CST. Objective: continue the active qKNN goal by checking whether the remaining N20/K5 floor failure is caused by assignment score scaling, query graph propagation, or lack of a compact support-trained linear correction. This section keeps the same N20 new-class set, `K=5`/`K=10` anchors, maximum query budget, HP08L5 LEO feature view, role-balanced assignment, and zero raw-support storage boundary.

Artifacts:

| artifact | rows | status |
|---|---:|---|
| `artifacts\n20_k5_v32_assignment_margin_grid_20260706e.csv/json` | 15 | completed |
| `artifacts\n20_k5_v32_score_calibration_grid_20260706f.csv/json` | 5 | completed |
| `artifacts\n20_k5_v32_strong_assignment_margin_grid_20260706f.csv/json` | 15 | completed |
| `artifacts\n20_k5_v32_labelprop_grid_20260706f.csv/json` | 216 | completed |
| `artifacts\n20_k5_v32_ridge_grid_20260706f.csv/json` | 30 | completed |
| `artifacts\n20_k5_v32_ridge_labelprop_combo_20260706f.csv/json` | 144 | completed |
| `artifacts\n20_k10_v32_ridge_exact_20260706f.csv/json` | 8 | completed |

K5 result summary:

| diagnostic | best setting | old_acc | min_old | new_acc | min_new | raw support |
|---|---|---:|---:|---:|---:|---:|
| baseline v32-equivalent | assignment margin 0,labelprop k8 | 91.56% | 80.00% | 79.07% | 69.33% | 0 |
| assignment margin | best remains weight 0 | 91.56% | 80.00% | 79.07% | 69.33% | 0 |
| score calibration | best remains `none` | 91.56% | 80.00% | 79.07% | 69.33% | 0 |
| strong top1 margin | best remains weight 0/0.2 | 91.56% | 80.00% | 79.07% | 69.33% | 0 |
| label propagation grid | weight 0.08,k10,alpha 0.72,temp 0.06,rounds 4 | 91.56% | 80.00% | 79.27% | 69.33% | 0 |
| ridge head grid | weight 0.10,alpha 0.1,clip 3.0 | 92.00% | 80.00% | 79.87% | 69.33% | 0 |
| ridge+labelprop combo | ridge 0.10/0.1,labelprop 0.035,k10,temp 0.08,rounds 4 | 92.00% | 80.00% | 80.27% | 69.33% | 0 |

K5 low-class details for the best ridge+labelprop combo:

| TX | accuracy |
|---|---:|
| `2-13` | 69.33% |
| `1-2` | 70.67% |
| `19-3` | 70.67% |
| `1-18` | 72.00% |
| `11-10` | 72.00% |
| `1-14` | 73.33% |
| `1-15` | 73.33% |
| `1-12` | 74.67% |

K10 ridge check under the strict K10 split:

| setting | old_acc | min_old | new_acc | min_new | raw support | note |
|---|---:|---:|---:|---:|---:|---|
| no ridge in this exact rerun | 91.90% | 82.86% | 82.21% | 62.86% | 0 | this row did not reproduce the earlier strongest v32 K10 row; keep earlier artifact as current strongest |
| ridge 0.20,alpha 1.0 | 93.57% | 84.29% | 82.29% | 71.43% | 0 | improves floor inside this rerun but still below 75% |

Diagnostic observations:

- Assignment-margin and column-calibration changes do not lift the K5 floor. Constant class bias is also structurally weak under exact role-balanced quotas because class-constant offsets largely cancel when every class receives a fixed quota.
- Strong top1 preservation does not solve the floor because raw top1 itself is weak for several classes: in the K5 v32 prediction audit, new-class raw top1 floor is only 37.33%, even though role-balanced assignment raises the final floor to 69.33%.
- Compact ridge support head is the most useful component in this batch: it raises K5 mean new accuracy from 79.07% to 80.27% and old accuracy from 91.56% to 92.00% without storing raw support. However, it shifts the bottleneck to `2-13`/`1-2` and does not break the 69.33% floor.
- The useful innovation direction is therefore not another global score scaler. The next qKNN variant should make ridge/linear support correction adaptive and class-neighborhood aware, with a support-only gate that activates it when it improves support LOO risk, while adding a targeted hard-neighborhood representation repair for `2-13`,`1-2`,`11-10`,`1-12`,`1-14`.

Current goal status: active, not achieved. The mean-accuracy side improved in a compressed deployable direction, but the explicit floor requirement remains unmet.

## Post-v32 Stability and Diagnostic Policy Check

Timestamp: 2026-07-06 CST. Objective: continue the active qKNN goal after v32 by checking whether the remaining `K=5` floor failure can be repaired by a unified adaptive mechanism rather than K-specific tuning. This section keeps the same N20 new-class set, `K=5`/`K=10` anchors, maximum query budget, LEO feature view, and zero raw-support storage boundary.

Goal review:

- Required end state is not only high mean accuracy. The active target requires multi-new-class stability, `K=5` close to `K=10`, and lowest new class at or above 75%.
- v32 remains the strongest current compressed qKNN row: `K=5` has `old_acc=91.56%`, `new_acc=79.07%`, `min_new=69.33%`; `K=10` has `old_acc=92.38%`, `new_acc=84.29%`, `min_new=72.86%`.
- Previous negative evidence already ruled out raw-IQ kNN under LEO, plain query clustering, Mahalanobis prototype, class-diag metric, subspace prototype, and stronger neighborhood gate variants as sufficient floor fixes.

New diagnostics and code change:

| item | artifact / commit | result | interpretation |
|---|---|---|---|
| v33 light neighborhood gate | `artifacts\n20_k5_v33_neighlite_20260706c.csv/json` | `old_acc=91.56%`, `new_acc=79.07%`, `min_new=69.33%`, `neighborhood_gate_count=0` | no improvement; support-audit gate still rejects all low-K neighborhoods |
| v34 query top-M quota residual | `artifacts\n20_k5_v34_quotaresid_20260706c.csv/json`, commit `7a6395c` | `old_acc=91.56%`, `new_acc=78.87%`, `min_new=68.00%` | worsens floor; unlabeled top-M residual misidentifies some low-floor classes |
| v32 radius norm grid | `artifacts\n20_k5_v32_radius_grid_20260706c.csv/json` | best remains `radius_norm=0`; nonzero radius norm lowers floor to 68.00% or below | class-radius normalization is not the missing stabilizer |
| v32 K5 seed scan | `artifacts\n20_k5_v32_seedscan20_20260706c.csv/json` | 20/20 seeds fail 75% floor; best floor is 69.33%, median floor 56.00%, worst floor 34.67% | K5 failure is systematic, not one unlucky support draw |

K5 seed-scan summary:

| metric | best | median | worst |
|---|---:|---:|---:|
| old_acc | 94.00% | 92.33% | 91.11% |
| new_acc | 81.53% | 76.97% | 68.00% |
| min_new | 69.33% | 56.00% | 34.67% |
| seeds with `min_new>=75%` | 0/20 | - | - |

Top K5 seed rows:

| seed | old_acc | min_old | new_acc | min_new | verdict |
|---:|---:|---:|---:|---:|---|
| 421038 | 91.56% | 80.00% | 79.07% | 69.33% | current K5 floor best |
| 421057 | 92.44% | 82.67% | 81.53% | 65.33% | higher mean, lower floor |
| 421044 | 91.11% | 80.00% | 76.67% | 62.67% | failed floor |
| 421051 | 94.00% | 84.00% | 77.67% | 60.00% | failed floor |
| 421048 | 91.33% | 78.67% | 79.40% | 58.67% | old floor below 80 and new floor failed |

Interpretation:

- The current strongest deployable qKNN route remains v32, not v33/v34.
- v33 and v34 are useful negative diagnostics: they show that the remaining floor gap is not solved by re-enabling neighborhood gates or by using unlabeled query top-M quota residuals.
- The seed scan is the strongest new evidence in this turn. With 20 K5 support draws, no row reaches 75% floor, so the active goal likely requires representation/enrollment repair or a different compressed registration objective, not another score-only postprocessor.
- Next aligned direction: hard-class-aware support registration that changes the stored prototype geometry itself, while still storing compressed prototypes/coefficients only. Candidate designs should target repeated hard neighborhoods (`1-12`/`1-1`/`8-3`, `1-14`/`1-10`/`1-16`, `11-10`/`2-13`/`1-18`, `19-3`/`1-15`) rather than global query smoothing.

Current goal status: active, not achieved.

## v35 Compressed Support-Bias Diagnostic

Timestamp: 2026-07-06 CST. Objective: continue the active qKNN goal by testing whether the remaining low-K floor can be improved by changing the compressed registration state instead of storing raw support samples. This check keeps the same N20 new-class set, `K=5`/`K=10` anchors, max query counts, HP08L5 feature view, and zero raw-support storage.

Goal review:

- The active goal still requires stable multi-new-class performance, not only a good mean. The hard requirement remains lowest new-class accuracy at or above 75% under `K=5` and `K=10`.
- Current best before this check is still v32: `K=5 old_acc=91.56%, min_old=80.00%, new_acc=79.07%, min_new=69.33%`; `K=10 old_acc=92.38%, min_old=82.86%, new_acc=84.29%, min_new=72.86%`.
- Earlier diagnostics showed raw-IQ kNN collapses under LEO, and query-cluster/radius/neighborhood/quota-residual variants did not repair the floor.

New code and diagnostic intent:

| item | detail |
|---|---|
| code change | `code/scripts/phase2_support_metric_qknn_probe.py` adds `dualview_support_v35` / `stable_dualview_v35` |
| mechanism | v35 inherits v32 and adds a support-only class-bias vector selected by support LOO; storage is one scalar per registered class |
| adaptivity | bias strength is computed from `class_load`, support hardness, and `K` reliability; no per-K manual parameter table |
| storage boundary | `stored_raw_support_count=0`; added state is `stored_support_bias_scalars=26` for N20 |

Diagnostic results:

| check | artifact | rows | old_acc | min_old | new_acc | min_new | key state | verdict |
|---|---|---:|---:|---:|---:|---:|---|---|
| v32 bootstrap/proto-repel grid, K5 | `artifacts\n20_k5_v32_bootstrap_repel_grid_20260706d.csv/json` | 180 | 91.56% | 80.00% | 79.07% | 69.33% | best has `bootstrap_proto_mix=0`, `proto_repel_lambda=0` | no gain |
| manual v32 + support-bias grid, K5 | `artifacts\n20_k5_v32_manual_supportbias_grid_20260706d.csv/json` | 7 | 91.56% | 80.00% | 79.07% | 69.33% | best has `support_bias_weight=0` | support-bias is not the missing fix |
| v35 support-bias, K5 | `artifacts\n20_k5_v35_supportbias_20260706d.csv/json` | 1 | 91.56% | 80.00% | 79.07% | 69.33% | `support_bias_weight=0.04`, `stored_support_bias_scalars=26` | no prediction change |
| v35 support-bias, K10 | `artifacts\n20_k10_v35_supportbias_20260706d.csv/json` | 1 | 92.38% | 82.86% | 84.29% | 72.86% | `support_bias_weight=0.0178`, `stored_support_bias_scalars=26` | no prediction change |

K5 hard-class details remain unchanged under v35:

| class | acc |
|---|---:|
| `1-12` | 69.33% |
| `1-14` | 69.33% |
| `11-10` | 70.67% |
| `19-3` | 70.67% |
| `1-2` | 70.67% |
| `2-13` | 72.00% |
| `1-15` | 72.00% |
| `1-18` | 72.00% |

Interpretation:

- The support-only bias idea is deployable and compressed, but it does not move the decision boundary on the current hard split. The failure is not a simple class-prior imbalance that can be corrected with one scalar per class.
- The bootstrap leave-subset prototype route also fails to improve the floor, even though it stores derived prototypes rather than raw support samples.
- Current strongest route remains v32. The next aligned direction should change feature/enrollment geometry more locally for specific hard neighborhoods, for example compressed pair-neighborhood adapters or representation-side hard-pair separation, not another global scalar bias.

Current goal status: active, not achieved.

## v32 Low-K Adaptive Graph Propagation Check

Timestamp: 2026-07-06 CST. Objective: continue the active qKNN optimization goal without expanding the K grid. The only anchors remain `K=5` and `K=10`; the target is still multi-new-class stability with per-class floor at or above 75%.

Goal review from the last attempts:

- v31 pair-linear support boundary is the current strongest compressed qKNN route before this check. It stores pair coefficients and prototypes, not raw support samples, but still leaves the new-class floor below 75%.
- Subspace prototype, Mahalanobis prototype, and class-diagonal metric grids were tested as compressed alternatives. They produced small mean gains or no gain and did not lift the minimum class above the v31 floor.
- Manual low-K label propagation without the neighborhood gate improved `K=5`, indicating that the low-K failure is partly graph/smoothing related. When overlaid with the v31 neighborhood gate, the gain was suppressed.

Implementation change in `code/scripts/phase2_support_metric_qknn_probe.py`:

| component | v32 behavior |
|---|---|
| policy name | adds `dualview_support_v32` / `stable_dualview_v32` |
| inherited route | inherits v31 support-LOO pair-linear route |
| low-K trigger | uses `low_k_floor=1-k_reliability`; only activates extra low-K graph propagation when `low_k_floor>=0.75` |
| graph strength | `0.035 * low_k_floor * graph_gate * class_load`, clipped at `0.04` |
| graph parameters | `labelprop_k=6+round(2*class_load)`, `alpha=0.72`, `temperature=0.08`, `rounds=8`, `scope=scenario` |
| gate interaction | for v32 low-K, disables the neighborhood gate that was suppressing the graph-propagation gain |
| storage boundary | raw support storage remains zero; support evidence is compressed into quantized support codes, class prototypes, pair-linear scalars, support-quality scalars, and residual-new scalars |

Verification commands:

| command | result |
|---|---|
| `conda run -n ssr-gpu python -m py_compile code\scripts\phase2_support_metric_qknn_probe.py` | PASS |
| `n20_k5_v32_lowk_labelprop_noneigh_r8_20260706` | completed |
| `n20_k10_v32_lowk_labelprop_noneigh_r8_20260706` | completed |

Joint result table:

| route | seed | K | query/class | old_acc | min_old | new_acc | min_new | raw support |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| v31 pair-linear | 421038 | 5 | 75 | 91.56% | 80.00% | 78.33% | 68.00% | 0 |
| v32 low-K graph | 421038 | 5 | 75 | 91.56% | 80.00% | 79.07% | 69.33% | 0 |
| v31 pair-linear | 421057 | 10 | 70 | 92.38% | 82.86% | 84.29% | 72.86% | 0 |
| v32 low-K graph | 421057 | 10 | 70 | 92.38% | 82.86% | 84.29% | 72.86% | 0 |

Historical anchor note: older v29 matched artifacts used seed `421000`; they are not a strict same-seed ablation against v31/v32. They remain useful as historical anchors: `K=5` v29 matched had `new_acc=70.87%` and `min_new=46.67%`; `K=10` v29 matched had `new_acc=78.86%` and `min_new=58.57%`.

v32 per-class old-TX detail:

| TX | K5 acc | K10 acc |
|---|---:|---:|
| `14-10` | 93.33% | 92.86% |
| `14-7` | 80.00% | 84.29% |
| `20-15` | 98.67% | 97.14% |
| `20-19` | 80.00% | 82.86% |
| `6-15` | 97.33% | 97.14% |
| `8-20` | 100.00% | 100.00% |

v32 per-class new-TX detail:

| TX | K5 acc | K10 acc |
|---|---:|---:|
| `10-10` | 89.33% | 94.29% |
| `11-10` | 70.67% | 72.86% |
| `18-5` | 74.67% | 82.86% |
| `19-3` | 70.67% | 82.86% |
| `2-13` | 72.00% | 72.86% |
| `2-5` | 86.67% | 92.86% |
| `3-8` | 94.67% | 95.71% |
| `4-10` | 90.67% | 92.86% |
| `8-18` | 90.67% | 95.71% |
| `8-3` | 76.00% | 77.14% |
| `1-1` | 73.33% | 72.86% |
| `1-10` | 86.67% | 90.00% |
| `1-11` | 89.33% | 95.71% |
| `1-12` | 69.33% | 74.29% |
| `1-14` | 69.33% | 78.57% |
| `1-15` | 72.00% | 85.71% |
| `1-16` | 77.33% | 80.00% |
| `1-18` | 72.00% | 81.43% |
| `1-19` | 85.33% | 90.00% |
| `1-2` | 70.67% | 77.14% |

Interpretation:

- v32 is a valid incremental improvement over v31 for the weaker `K=5` anchor: `new_acc` improves by +0.73pp and `min_new` improves by +1.33pp, with no loss to old-class accuracy and no raw support storage.
- v32 does not solve the active target. `K=5` still has five clear low-floor classes below 72.01% (`1-12`,`1-14`,`11-10`,`19-3`,`1-2`) and `K=10` still has three classes at 72.86% (`11-10`,`2-13`,`1-1`).
- The `K=5` to `K=10` mean-new gap is 5.22pp, which is improved from v31 but still slightly outside the active stability boundary.
- Negative evidence is now clearer: compressed second-order support statistics alone do not lift the hardest floor. Subspace, Mahalanobis, class-diag metric, and pair-only gates all fail to push the weak classes above 75%. The next credible qKNN route should use a class-neighborhood registration mechanism or source-informed representation repair targeted at the repeated hard classes, while preserving zero raw support storage and adaptive parameters.

Current goal status: active, not achieved.

## v31 Adaptive Pair-Linear Hard-Class Boundary

Timestamp: 2026-07-06 CST. Objective: turn the positive but small support-only pairwise linear boundary diagnostic into an adaptive qKNN policy that does not require separate manual grids for `K=5` and `K=10`.

Mechanism:

| item | value |
|---|---|
| code change | `stable_dualview_v31` added to `code/scripts/phase2_support_metric_qknn_probe.py` |
| inherited base | v30 support-LOO gated auxiliary view plus v29 neighborhood gate |
| new component | adaptive support-LOO pair-linear boundary for new-class hard pairs |
| adaptation inputs | support hardness,class load,K reliability |
| learned state | compact ridge pair boundary coefficients only |
| raw support storage | `stored_raw_support_count=0` |

Diagnostic grid before promotion:

| setting | K | best pair-linear setting | old_acc | min_old | new_acc | min_new | raw support |
|---|---:|---|---:|---:|---:|---:|---:|
| v30 + manual pair-linear grid | 10 | weight 0.004,top_pairs 8,alpha 1.0,clip 0.5 | 92.38% | 82.86% | 84.29% | 72.86% | 0 |
| v30 + manual pair-linear grid | 5 | weight 0.008,top_pairs 8,alpha 0.1,clip 1.0 | 91.56% | 80.00% | 78.33% | 68.00% | 0 |

v31 automatic result:

| setting | K | auto pair-linear weight | auto top_pairs | auto alpha | auto clip | old_acc | min_old | new_acc | min_new | raw support |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| v31 auto | 10 | 0.00817 | 8 | 0.40 | 0.833 | 92.38% | 82.86% | 84.29% | 72.86% | 0 |
| v31 auto | 5 | 0.00950 | 8 | 0.10 | 1.000 | 91.56% | 80.00% | 78.33% | 68.00% | 0 |

Per-class new-TX detail for v31:

| TX | K10 | K5 |
|---|---:|---:|
| `10-10` | 94.29% | 88.00% |
| `11-10` | 72.86% | 70.67% |
| `18-5` | 82.86% | 72.00% |
| `19-3` | 82.86% | 70.67% |
| `2-13` | 72.86% | 68.00% |
| `2-5` | 92.86% | 86.67% |
| `3-8` | 95.71% | 94.67% |
| `4-10` | 92.86% | 90.67% |
| `8-18` | 95.71% | 90.67% |
| `8-3` | 77.14% | 76.00% |
| `1-1` | 72.86% | 73.33% |
| `1-10` | 90.00% | 86.67% |
| `1-11` | 95.71% | 89.33% |
| `1-12` | 74.29% | 68.00% |
| `1-14` | 78.57% | 68.00% |
| `1-15` | 85.71% | 72.00% |
| `1-16` | 80.00% | 74.67% |
| `1-18` | 81.43% | 73.33% |
| `1-19` | 90.00% | 84.00% |
| `1-2` | 77.14% | 69.33% |

Comparison against previous best:

| setting | K | previous new_acc | v31 new_acc | delta | previous min_new | v31 min_new | delta |
|---|---:|---:|---:|---:|---:|---:|---:|
| N20 HP08L5 | 10 | 84.14% | 84.29% | +0.14pp | 71.43% | 72.86% | +1.43pp |
| N20 HP08L5 | 5 | 78.20% | 78.33% | +0.13pp | 66.67% | 68.00% | +1.33pp |

Interpretation:

- v31 is aligned with the active qKNN direction: it uses support labels only, stores compact pair coefficients instead of raw support samples, and adapts from class count and K rather than using separate K-specific grids.
- The effect is positive but too small. It improves the hard-class floor by about 1.3-1.4pp, but the floor remains below the required 75%, especially for `K=5`.
- The `K=5` versus `K=10` new-accuracy gap remains about 5.95pp, so the stability requirement is still not met.
- The next optimization should not add more post-hoc pair bias. The remaining gap likely requires better enrollment/support representation for the same hard classes, or a representation-side repair that changes the feature geometry before qKNN scoring.

Verification:

| command | result |
|---|---|
| `conda run -n ssr-gpu python -m py_compile code\scripts\phase2_support_metric_qknn_probe.py` | passed |
| K10 v31 auto command | completed |
| K5 v31 auto command | completed |
| K5 manual grid first attempt | local conda temp-file lock; rerun serial completed |

Current goal status: active, not achieved.

## v30 Support-LOO Gated Aux-Signature Diagnostic

Timestamp: 2026-07-06 CST. Objective: test whether a low-dimensional source-classifier response signature can be used as an adaptive auxiliary qKNN view without hand-tuning separate parameters for `K=5` and `K=10`.

Mechanism:

| item | value |
|---|---|
| code change | `stable_dualview_v30` added to `code/scripts/phase2_support_metric_qknn_probe.py` |
| inherited base | v29 adaptive compressed qKNN |
| aux view | 12-D signature from `tx_logits`: centered normalized logits plus softmax probabilities |
| gate | support-LOO mean delta, support-LOO floor delta, and absolute aux floor |
| storage | no raw support; aux transform scalars stored only when gate passes |
| K handling | same policy for `K=5` and `K=10`; no K-specific grid expansion |

Artifacts:

| artifact | status |
|---|---|
| `artifacts\aux_txlogit_signature_HP08L5_n20.npz` | generated locally from the aligned feature NPZ |
| `artifacts\n20_k10_v29_auxlogit_probe.csv/json` | ungated aux diagnostic completed |
| `artifacts\n20_k5_v29_auxlogit_probe.csv/json` | ungated aux diagnostic completed |
| `artifacts\n20_k10_v30_auxlogit_gate_probe.csv/json` | gated aux diagnostic completed |
| `artifacts\n20_k5_v30_auxlogit_gate_probe.csv/json` | gated aux diagnostic completed |

Joint result:

| setting | K | aux gate | effective aux weight | old_acc | min_old | new_acc | min_new | stored raw support | verdict |
|---|---:|---|---:|---:|---:|---:|---:|---:|---|
| v29 baseline | 5 | none | 0.000 | 91.56% | 80.00% | 78.20% | 66.67% | 0 | baseline |
| v29 + ungated aux | 5 | none | 0.220 | 90.67% | 80.00% | 77.73% | 60.00% | 0 | harmful floor drop |
| v30 + gated aux | 5 | support-LOO | 0.000 | 91.56% | 80.00% | 78.20% | 66.67% | 0 | harmful aux rejected |
| v29 baseline | 10 | none | 0.000 | 92.38% | 82.86% | 84.14% | 71.43% | 0 | baseline |
| v29 + ungated aux | 10 | none | 0.219 | 92.14% | 82.86% | 84.43% | 71.43% | 0 | mean-only drift |
| v30 + gated aux | 10 | support-LOO | 0.000 | 92.38% | 82.86% | 84.14% | 71.43% | 0 | aux rejected |

Gate evidence:

| K | primary support-LOO mean | aux support-LOO mean | primary support-LOO floor | aux support-LOO floor | gate factor |
|---:|---:|---:|---:|---:|---:|
| 5 | 74.62% | 59.23% | 0.00% | 0.00% | 0.000 |
| 10 | 76.15% | 60.00% | 40.00% | 20.00% | 0.000 |

Interpretation:

- The source-logit response signature is not a reliable auxiliary identity view for this N20 HP08L5 split. It is worse than the ADV3B02 feature geometry under support-LOO, especially at `K=5`.
- v30 improves algorithmic stability rather than final accuracy: it prevents an apparently plausible auxiliary representation from harming the strict K-shot result, using only support labels and no query labels.
- The active goal is still not met. Current v30 evidence keeps compressed storage and self-adaptation, but the hard-class floor remains below 75%. The next useful optimization should target hard-class representation/enrollment separability, not source-logit blending.

Verification:

| command | result |
|---|---|
| `conda run -n ssr-gpu python -m py_compile code\scripts\phase2_support_metric_qknn_probe.py` | passed |
| K10 v30 gated aux command | completed |
| K5 v30 gated aux command | completed |

Current goal status: active, not achieved.

## Raw-IQ kNN and Query-Pair qKNN Diagnostic

Objective: answer whether direct raw-signal kNN can handle many new classes, and test whether a compressed qKNN query-pair variant can repair the current N20 tail-class collapse without storing raw support samples. K anchors remained `K=5` and `K=10`; no larger K values were added.

Raw-IQ kNN diagnostic:

| setting | K | channel view | old_acc | min_old | new_acc | min_new | stored raw support | verdict |
|---|---:|---|---:|---:|---:|---:|---:|---|
| N2 | 5 | clean | 91.33% | 62.67% | 100.00% | 100.00% | nonzero | clean-only positive, not compressed |
| N2 | 5 | LEO | 25.11% | 17.33% | 26.00% | 22.67% | nonzero | failed |
| N5 | 10 | clean | 97.38% | 92.86% | 99.71% | 98.57% | nonzero | clean-only positive, not compressed |
| N5 | 10 | LEO | 34.29% | 20.00% | 41.43% | 12.86% | nonzero | failed |
| N10 | 10 | clean | 97.14% | 92.86% | 96.00% | 75.71% | nonzero | clean-only positive, not compressed |
| N10 | 10 | LEO | 27.62% | 14.29% | 35.14% | 8.57% | nonzero | failed |
| N20 | 10 | clean | 97.14% | 92.86% | 93.43% | 68.57% | nonzero | below floor even clean |
| N20 | 10 | LEO | 25.48% | 15.71% | 27.43% | 1.43% | nonzero | failed |

Interpretation: direct raw-IQ kNN confirms that transmitter identity is present in clean samples, but it collapses under LEO target view and violates the compressed-support requirement because it must retain raw support samples. It is therefore a negative deployment diagnostic, not the route to promote.

Compressed query-pair qKNN variant:

- Added `query_pair_cluster_*` options to `code/scripts/phase2_support_metric_qknn_probe.py`.
- Mechanism: after normal role-balanced qKNN assignment, only high-risk class pairs are considered; each pair keeps its quota and is locally re-ranked by a support-prototype axis plus a temporary query-cluster axis. The temporary query centers are discarded after inference.
- Storage remains compressed: `stored_raw_support_count=0`; persistent state is support prototypes/codes and scalar gates, not raw support samples.

Verification:

| command | result |
|---|---|
| `conda run -n ssr-gpu python -m py_compile code\scripts\phase2_support_metric_qknn_probe.py` | PASS |
| `n20_k10_v29_qpair_probe.csv` | completed |
| `n20_k10_v29_qpair_aggr_probe.csv` | completed |
| `n20_k5_v29_qpair_probe.csv` | completed |

Best query-pair rows:

| route | K | old_acc | min_old | new_acc | min_new | changed preds | query-pair pairs | raw support stored | verdict |
|---|---:|---:|---:|---:|---:|---:|---|---:|---|
| v29 baseline | 5 | 91.56% | 80.00% | 78.20% | 66.67% | 0 | none | 0 | failed |
| v29 + conservative query-pair | 5 | 91.56% | 80.00% | 78.27% | 66.67% | 8 | `1-1<->1-12`;`2-5<->8-3`;`11-10<->2-13`;`18-5<->1-14` | 0 | mean-only gain |
| v29 baseline | 10 | 92.38% | 82.86% | 84.14% | 71.43% | 0 | none | 0 | failed |
| v29 + conservative query-pair | 10 | 92.38% | 82.86% | 84.21% | 71.43% | 2 | `18-5<->1-16` | 0 | mean-only gain |
| v29 + aggressive query-pair | 10 | 92.38% | 82.86% | 82.79% | 65.71% | 42 | multiple | 0 | harmful |

Per-class lows for the best conservative query-pair rows:

| K | lowest new classes |
|---:|---|
| 5 | `2-13` 66.67%;`1-12` 68.00%;`1-14` 68.00%;`1-2` 69.33%;`11-10` 70.67%;`19-3` 70.67% |
| 10 | `2-13` 71.43%;`1-1` 72.86%;`11-10` 72.86%;`1-12` 74.29%;`1-2` 77.14%;`8-3` 77.14% |

Interpretation:

- The query-pair idea is compatible with the compressed qKNN route and provides a deployable innovation handle, but the current implementation does not solve the active goal.
- Conservative query-pair refinement slightly improves mean new accuracy but does not lift the lowest class above 75%.
- Aggressive query-axis weighting damages the tail, so query clusters cannot be trusted as a dominant signal under the current representation.
- The bottleneck remains hard-class feature separability/enrollment quality for `2-13`,`1-1`,`1-12`,`11-10`,`1-14`,`1-2`, not raw support storage or global assignment alone.

Current goal status: active, not achieved.

## Raw-IQ kNN Multi-New Count Refresh

Timestamp: 2026-07-06 12:35 CST. Objective: answer whether plain kNN on original IQ samples remains viable as new-class count grows. This is a diagnostic baseline only. It stores raw support IQ and is therefore not the proposed qKNN compressed deployment route.

Protocol:

| item | value |
|---|---|
| remote command host | N607 direct SSH, short-lived |
| script | `code/scripts/phase2_raw_iq_knn_probe.py` |
| metadata feature file | `automation_reports/CV-SincNet/phase2_qknn_adaptive_manynew_20260705/n20_features/features_n20_norm.npz` |
| raw sample source | `ManySig.pkl` for target-old,`ManyTx.pkl` for target-new |
| target receiver domain | `7-14` |
| old TX | `14-10,14-7,20-15,20-19,6-15,8-20` |
| new-class counts | `2,5,10,20` |
| K anchors | `K=5,K=10` only |
| query size | maximum available per class:`75` for K5,`70` for K10 |
| classifier | flattened RMS-normalized raw IQ,cosine 1-NN |
| views | clean raw IQ and LEO raw IQ |
| SSH cleanup | local `ssh.exe` count 0,N607 port-22 connection count 0 |

Artifacts copied back locally:

| artifact | status |
|---|---|
| `artifacts\rawiq_knn_n2_k5k10_seed421027_grid20260706.csv/json` | completed |
| `artifacts\rawiq_knn_n5_k5k10_seed421027_grid20260706.csv/json` | completed |
| `artifacts\rawiq_knn_n10_k5k10_seed421027_grid20260706.csv/json` | completed |
| `artifacts\rawiq_knn_n20_k5k10_seed421027_grid20260706.csv/json` | completed |

Joint summary:

| view | new classes | K | old_acc | min_old | seen_new_acc | min_new | stored raw support |
|---|---:|---:|---:|---:|---:|---:|---:|
| clean raw IQ | 2 | 5 | 91.33% | 62.67% | 100.00% | 100.00% | 40 |
| LEO raw IQ | 2 | 5 | 25.11% | 17.33% | 26.00% | 22.67% | 40 |
| clean raw IQ | 2 | 10 | 97.38% | 92.86% | 100.00% | 100.00% | 80 |
| LEO raw IQ | 2 | 10 | 26.43% | 14.29% | 28.57% | 28.57% | 80 |
| clean raw IQ | 5 | 5 | 91.33% | 62.67% | 90.93% | 56.00% | 55 |
| LEO raw IQ | 5 | 5 | 29.78% | 14.67% | 19.73% | 10.67% | 55 |
| clean raw IQ | 5 | 10 | 97.38% | 92.86% | 99.71% | 98.57% | 110 |
| LEO raw IQ | 5 | 10 | 34.29% | 20.00% | 41.43% | 12.86% | 110 |
| clean raw IQ | 10 | 5 | 91.33% | 62.67% | 81.47% | 44.00% | 80 |
| LEO raw IQ | 10 | 5 | 26.67% | 13.33% | 21.33% | 9.33% | 80 |
| clean raw IQ | 10 | 10 | 97.14% | 92.86% | 96.00% | 75.71% | 160 |
| LEO raw IQ | 10 | 10 | 27.62% | 14.29% | 35.14% | 8.57% | 160 |
| clean raw IQ | 20 | 5 | 91.33% | 62.67% | 83.73% | 44.00% | 130 |
| LEO raw IQ | 20 | 5 | 20.44% | 14.67% | 19.80% | 2.67% | 130 |
| clean raw IQ | 20 | 10 | 97.14% | 92.86% | 93.43% | 68.57% | 260 |
| LEO raw IQ | 20 | 10 | 25.48% | 15.71% | 27.43% | 1.43% | 260 |

Interpretation:

- Clean raw IQ shows that original samples contain strong transmitter identity in low class counts, especially K10; however K5 floor already fails at 5+ new classes and K10 floor fails at 20 new classes.
- Deployment-relevant LEO raw IQ collapses for every tested new-class count. Even at only 2 new classes, seen-new accuracy is below 30% and old accuracy is about 25-26%.
- Plain raw-support kNN is not a viable Stage2-C deployment route. It also violates the active qKNN storage objective because it stores raw support IQ samples: 40/80/55/110/80/160/130/260 support samples depending on N and K.
- The useful evidence is negative: the high clean result confirms identity information exists, while the LEO collapse confirms the necessary innovation should be LEO-robust representation/repair plus compressed support state, not raw IQ kNN.

Current goal status: active, not achieved.

## Adaptive v29 Compressed Neighborhood Gate

Timestamp: 2026-07-06 12:23 +08:00

Objective: extend the strongest compressed qKNN line beyond pair-only rescue. v28q showed that the persistent low-floor classes are multi-way neighborhood confusions, so v29 adds an adaptive one-vs-neighborhood gate. Candidate weak classes are selected from support LOO errors plus unlabeled query top-M ambiguity; query labels remain audit-only.

Mechanism:

| item | value |
|---|---|
| policy | `stable_dualview_v29` / `dualview_support_v29` |
| base route | inherits v9/v27 adaptive compressed-support path and support-quality weighting |
| new state | per weak class: target label, up to 4 neighbor labels, 5 ridge coefficients, small gate scalars |
| adaptation inputs | target support codes/prototypes, support LOO predictions, unlabeled query score competition |
| forbidden state | raw support IQ/features and query labels are not stored |
| adaptive controls | top weak classes, neighbor count, gate margin and weight are derived from K and new-class load |

Local changes before N607 sync:

| file | purpose |
|---|---|
| `code/scripts/phase2_support_metric_qknn_probe.py` | add `_support_query_neighborhood_gate_scores`; register `stable_dualview_v29`; emit neighborhood gate metrics and storage fields |

Local verification:

| command | result |
|---|---|
| `conda run -n ssr-gpu python -m py_compile code\scripts\phase2_support_metric_qknn_probe.py` | passed |
| HP08 local smoke, N20 K5, `stable_dualview_v29` | ran successfully; `old_acc=93.56%`, `min_old=82.67%`, `new_acc=72.33%`, `min_new=54.67%`, `neighborhood_gate_count=6`, `stored_neighborhood_gate_scalars=66`, `stored_raw_support_count=0` |

The local smoke uses the older local HP08 feature and is only an implementation check. The required performance check remains HP08L5 matched surface on N607 with strict max-query K5/K10.

Local script SHA256 before sync: `536C7C1FE56B5B0F7E215CCC7C5FA928724BD7DE614E9828F63FDA80F854F0C5`.

Planned N607 evaluation:

| item | value |
|---|---|
| remote script | `/home/szu2070436088/2510044040/CV-SincNet/code/scripts/phase2_support_metric_qknn_probe.py` |
| remote output dir | `/home/szu2070436088/2510044040/CV-SincNet/runs/phase2_qknn_hardpair_n20_20260706/v29_eval_20260706_1223` |
| feature file | `runs/phase2_qknn_hardpair_n20_20260706/MANYNEW20_HARDPAIR_HP08L5/ADV3B02_CORE90_SOFT_E200_PHASE1_HARDPAIR_HP08L5_N20/features_hardpair_HP08L5_n20.npz` |
| K anchors | `K=5(query=75/class)`, `K=10(query=70/class)` |
| success check | compare against v27/v28q on old_acc, seen_new_acc, min_new and K5-vs-K10 gap; do not claim completion unless every new class reaches 75% |

N607 execution and final v29 status:

| item | value |
|---|---|
| remote compile | passed with SHA256 `c05df83ebb780e49515019355d4b2915f940dcf5f2e7bbba82865ad1b15c9a5e` |
| first v29 run | `v29_eval_20260706_1223`, `seed_count=1`; negative because single support split was not comparable with v27/v28q seed sweep |
| fairer v29 run | `v29_eval_20260706_1223_seed120`, `seed_start=421000`, `seed_count=120` |
| pulled artifacts | `artifacts\n20_k5_v29_seed120_HP08L5_matched.csv/json/log`, `artifacts\n20_k10_v29_seed120_HP08L5_matched.csv/json/log` |
| SSH cleanup | no local `ssh.exe` process or established N607 port-22 connection after sync/eval/pull |

Joint comparison against current reference:

| method | K | seed | old_acc | min_old | seen_new_acc | min_new | neighborhood gates | raw support | verdict |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| v27 matched | 5 | 421037 | 92.89% | 81.33% | 77.00% | 60.00% | n/a | 0 | reference |
| v28q matched | 5 | 421037 | 92.89% | 81.33% | 77.13% | 60.00% | n/a | 0 | mean +0.13pp, floor tied |
| v29 strict seed120 best-floor | 5 | 421038 | 91.56% | 80.00% | 78.20% | 66.67% | 0 | 0 | better floor, still below 75% |
| v27 matched | 10 | 421029 | 93.10% | 81.43% | 84.29% | 68.57% | n/a | 0 | reference |
| v28q matched | 10 | 421029 | 93.10% | 81.43% | 84.14% | 68.57% | n/a | 0 | floor tied, mean lower |
| v29 strict seed120 best-floor | 10 | 421057 | 92.38% | 82.86% | 84.14% | 71.43% | 0 | 0 | better floor, still below 75% |

Per-TX low-floor details for v29 seed120:

| K | seed | lowest new-class accuracies |
|---:|---:|---|
| 5 | 421038 | `1-14=66.67%`, `1-12=68.00%`, `2-13=68.00%`, `1-2=69.33%`, `11-10=70.67%`, `19-3=70.67%` |
| 10 | 421057 | `2-13=71.43%`, `1-1=72.86%`, `11-10=72.86%`, `1-12=74.29%`, `1-2=77.14%`, `8-3=77.14%` |

Paired-seed stability check:

| seed | K5 old/new/min_new | K10 old/new/min_new | K10-K5 new gap |
|---:|---|---|---:|
| 421038 | 91.56% / 78.20% / 66.67% | 91.43% / 80.00% / 65.71% | 1.80pp |
| 421057 | 91.56% / 81.20% / 65.33% | 92.38% / 84.14% / 71.43% | 2.94pp |
| 421065 | 92.44% / 78.07% / 65.33% | 92.38% / 82.29% / 65.71% | 4.22pp |

Interpretation:

- v29 as originally designed was too aggressive: multi-way neighborhood gating could reduce K5 floor to 46.67%. The final strict version only accepts a neighborhood when support LOO target accuracy strictly improves and support mean/floor do not drop. Under this safety gate, the best rows accept zero neighborhoods, which prevents collapse but means the current neighborhood boundary is not yet the source of improvement.
- The best seed120 floor is higher than v27/v28q: K5 `60.00% -> 66.67%`, K10 `68.57% -> 71.43%`. However, this is support-split sensitivity evidence, not a complete deployable support-selection method.
- Support-only selection proxies are not reliable enough yet. Across 120 rows, simple support LOO/prototype similarity metrics had weak or negative correlation with query min-new accuracy, so selecting the best support split by query performance cannot be reported as an onboard adaptive algorithm.
- Current goal remains active and unmet: K10 is close but still below 75%, and K5 remains far below the 75% floor. The next route should learn a compressed support-quality selector that uses only support/prototype geometry but is validated against query floors, or improve the representation for the dense `1-*`/`2-13` confusion cluster.

## Adaptive v28 Top2 Hard-Pair Gate

Objective: test whether the persistent low-floor classes can be rescued by a compressed hard-pair gate that extends v27. The mechanism uses v27 support-quality weighting plus a top2 pair boundary. The v28q update selects candidate pairs from both support LOO errors and unlabeled query top2 ambiguity; query labels are used only for audit metrics. The deployed state stores compact pair coefficients and gate scalars, not raw support or query samples.

Local and remote verification:

| item | value |
|---|---|
| local compile | `conda run -n ssr-gpu python -m py_compile code\scripts\phase2_support_metric_qknn_probe.py` passed |
| remote output dir | `/home/szu2070436088/2510044040/CV-SincNet/runs/phase2_qknn_hardpair_n20_20260706/v28q_eval_20260706_1201` |
| remote script hash | `89281872a603a09da57c296e0d011cb26f8ec4624f8540b0d652731f7037be45` |
| local artifacts | `artifacts\n20_k5_v28q_HP08L5_matched.csv/json/log`, `artifacts\n20_k10_v28q_HP08L5_matched.csv/json/log` |
| matched surface | HP08L5, N20, `topm=4`, `scenario_aware=True`, `balanced_assignment=True`, `role_balanced_assignment=True`, `exclude_pool_from_query=True`, strict `pool_per_class=K` |

Joint comparison:

| method | K | old_acc | min_old | seen_new_acc | min_new | top2 pairs | query top2 weight | raw support | verdict |
|---|---:|---:|---:|---:|---:|---|---:|---:|---|
| v27 matched | 5 | 92.89% | 81.33% | 77.00% | 60.00% | n/a | n/a | 0 | current reference |
| v28q matched | 5 | 92.89% | 81.33% | 77.13% | 60.00% | `1-16<->18-5`;`1-1<->8-3`;`1-15<->19-3`;`1-1<->1-12` | 0.040 | 0 | mean +0.13pp, floor tied |
| v27 matched | 10 | 93.10% | 81.43% | 84.29% | 68.57% | n/a | n/a | 0 | current reference |
| v28q matched | 10 | 93.10% | 81.43% | 84.14% | 68.57% | `1-15<->19-3`;`1-18<->11-10`;`1-1<->1-12`;`1-10<->1-14` | 0.040 | 0 | mean -0.14pp, floor tied |

Per-TX details for v28q:

| TX | K5 acc | K10 acc |
|---|---:|---:|
| `14-10` | 93.33% | 94.29% |
| `14-7` | 82.67% | 82.86% |
| `20-15` | 100.00% | 100.00% |
| `20-19` | 81.33% | 81.43% |
| `6-15` | 100.00% | 100.00% |
| `8-20` | 100.00% | 100.00% |
| `1-1` | 64.00% | 75.71% |
| `1-10` | 88.00% | 90.00% |
| `1-11` | 93.33% | 95.71% |
| `1-12` | 60.00% | 70.00% |
| `1-14` | 62.67% | 82.86% |
| `1-15` | 77.33% | 87.14% |
| `1-16` | 77.33% | 81.43% |
| `1-18` | 66.67% | 82.86% |
| `1-19` | 84.00% | 87.14% |
| `1-2` | 73.33% | 78.57% |
| `10-10` | 93.33% | 90.00% |
| `11-10` | 65.33% | 80.00% |
| `18-5` | 70.67% | 81.43% |
| `19-3` | 74.67% | 85.71% |
| `2-13` | 64.00% | 68.57% |
| `2-5` | 84.00% | 90.00% |
| `3-8` | 92.00% | 95.71% |
| `4-10` | 93.33% | 91.43% |
| `8-18` | 84.00% | 91.43% |
| `8-3` | 74.67% | 77.14% |

Confusion audit from v28 predictions:

| setting | weak class | main errors |
|---|---|---|
| K5 | `1-12` 60.00% | `1-1` 16/75, `8-3` 6/75, `1-14` 6/75 |
| K5 | `1-14` 62.67% | `18-5` 10/75, `1-10` 9/75, `1-16` 5/75 |
| K5 | `2-13` 64.00% | `1-2` 9/75, `1-18` 7/75, `11-10` 6/75 |
| K5 | `11-10` 65.33% | `2-13` 16/75, `1-18` 9/75 |
| K10 | `2-13` 68.57% | `1-2` 9/70, `11-10` 5/70, `10-10` 4/70 |

Interpretation:

- v28q validates a more innovative compressed hard-pair candidate selection path, but it does not improve the minimum class floor.
- The persistent floor failures are stable multi-way confusions, not only two-class top2 ambiguity. `2-13` in particular is split across `1-2`,`1-18`,`11-10`,`10-10`, so pair-only gates are too narrow.
- The next route should be a compressed class-neighborhood adapter or low-floor subspace correction that handles one-vs-neighborhood competition, while keeping raw support storage at zero and keeping parameters adaptive to K/class count.

Current goal status: active, not achieved.

## Adaptive v27 Support-Quality Update

Timestamp: 2026-07-06 11:46:27 +08:00

Objective: add a single adaptive qKNN variant that inherits the stable v9 compressed-support route and automatically changes its support-quality correction strength with K and class load, without adding a separate per-K parameter table.

Local code changes before N607 sync:

| file | purpose |
|---|---|
| `code/scripts/phase2_support_metric_qknn_probe.py` | add `stable_dualview_v27` / `dualview_support_v27`; v27 inherits v9 and adds adaptive support-quality weighting; wire support-bias/support-quality parameters through adaptive overrides into the evaluator |
| `code/scripts/phase2_qknn_old_anchor_transport_diag.py` | repair old-anchor transport diagnostic defaults and use strict `pool_per_class=K` for K5/K10 max-query diagnostics |

Local verification:

| command | result |
|---|---|
| `conda run -n ssr-gpu python -m py_compile code\scripts\phase2_support_metric_qknn_probe.py code\scripts\phase2_qknn_old_anchor_transport_diag.py` | passed |
| N20 NORM K5 v27 local check | `old_acc=90.44%`, `min_old=76.00%`, `new_acc=42.40%`, `min_new=24.00%`, `support_quality_weight=0.100`, `stored_support_quality_scalars=130`, `stored_raw_support_count=0` |
| N20 NORM K10 v27 local check | `old_acc=92.14%`, `min_old=78.57%`, `new_acc=58.21%`, `min_new=35.71%`, `support_quality_weight=0.085`, `stored_support_quality_scalars=260`, `stored_raw_support_count=0` |

The NORM local check is an implementation/wiring check, not the target performance row. The target performance row must be rerun on the current strongest HP08L5 feature export on N607 with strict K/query settings: K5 uses 75 query/class and K10 uses 70 query/class.

N607 matched-surface evaluation:

| item | value |
|---|---|
| remote output dir | `/home/szu2070436088/2510044040/CV-SincNet/runs/phase2_qknn_hardpair_n20_20260706/v27_eval_20260706_1153` |
| feature file | `runs/phase2_qknn_hardpair_n20_20260706/MANYNEW20_HARDPAIR_HP08L5/ADV3B02_CORE90_SOFT_E200_PHASE1_HARDPAIR_HP08L5_N20/features_hardpair_HP08L5_n20.npz` |
| local artifacts | `artifacts\n20_k5_v27_HP08L5_matched.csv/json/log`, `artifacts\n20_k10_v27_HP08L5_matched.csv/json/log` |
| sync destination | `/home/szu2070436088/2510044040/CV-SincNet/code/scripts/phase2_support_metric_qknn_probe.py` |
| remote verification | `/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python -m py_compile code/scripts/phase2_support_metric_qknn_probe.py`; sha256 `aaf811e79b5915397f1e18d6612f382d525bb92fbd5ff2d6d7a8c1714da5ee2b` |
| matched evaluation surface | `topm=4`, `scenario_aware=True`, `balanced_assignment=True`, `role_balanced_assignment=True`, `exclude_pool_from_query=True`, strict `pool_per_class=K` |

Joint results:

| method | K | old_acc | min_old | seen_new_acc | min_new | support_quality_weight | stored_quality_scalars | stored_raw_support_count | verdict |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| HP08L5 v9 reference | 5 | 92.89% | 81.33% | 76.93% | 58.67% | 0 | 0 | 0 | below floor |
| HP08L5 v27 matched | 5 | 92.89% | 81.33% | 77.00% | 60.00% | 0.100 | 130 | 0 | +1.33pp floor, still below target |
| HP08L5 v9 reference | 10 | 93.10% | 81.43% | 84.29% | 68.57% | 0 | 0 | 0 | below floor |
| HP08L5 v27 matched | 10 | 93.10% | 81.43% | 84.29% | 68.57% | 0.085 | 260 | 0 | tied reference, still below target |

Per-TX details for HP08L5 v27 matched:

| TX | K5 acc | K10 acc |
|---|---:|---:|
| `14-10` | 93.33% | 94.29% |
| `14-7` | 82.67% | 82.86% |
| `20-15` | 100.00% | 100.00% |
| `20-19` | 81.33% | 81.43% |
| `6-15` | 100.00% | 100.00% |
| `8-20` | 100.00% | 100.00% |
| `1-1` | 64.00% | 75.71% |
| `1-10` | 88.00% | 90.00% |
| `1-11` | 93.33% | 95.71% |
| `1-12` | 60.00% | 70.00% |
| `1-14` | 62.67% | 82.86% |
| `1-15` | 76.00% | 88.57% |
| `1-16` | 77.33% | 81.43% |
| `1-18` | 66.67% | 82.86% |
| `1-19` | 84.00% | 87.14% |
| `1-2` | 73.33% | 78.57% |
| `10-10` | 93.33% | 90.00% |
| `11-10` | 65.33% | 80.00% |
| `18-5` | 70.67% | 81.43% |
| `19-3` | 73.33% | 87.14% |
| `2-13` | 64.00% | 68.57% |
| `2-5` | 84.00% | 90.00% |
| `3-8` | 92.00% | 95.71% |
| `4-10` | 93.33% | 91.43% |
| `8-18` | 84.00% | 91.43% |
| `8-3` | 74.67% | 77.14% |

Interpretation:

- v27 is a valid compressed-support adaptive variant: it stores support-quality scalars and quantized support codes, with `stored_raw_support_count=0`. The support-quality weight changes automatically from `0.100` at K5 to `0.085` at K10 under the same algorithm.
- The K5 floor improves from 58.67% to 60.00%, but this is far from the active 75% per-class floor. K10 remains limited by `2-13=68.57%` and `1-12=70.00%`.
- Mean new accuracy drops from K10 to K5 by 7.29pp, so the current K5/K10 stability target is still not satisfied.
- The next useful optimization should target the hard low-floor classes directly (`1-12`,`1-14`,`2-13`,`11-10`,`18-5`) with a compressed pair/subspace mechanism or representation repair. Simple support-quality weighting is not sufficient.

Current goal status: active, not achieved.

## HP08L5 v23/v25 Policy Sync and Sweep

Objective: verify whether the newer compressed qKNN adaptive policies already present in the local repository can improve the HP08L5 weak-class floor after syncing the current `phase2_support_metric_qknn_probe.py` to N607.

Sync and verification:

| item | value |
|---|---|
| local script | `E:\type10-7\github_publish\CVS-RFFI-repo\code\scripts\phase2_support_metric_qknn_probe.py` |
| remote script | `/home/szu2070436088/2510044040/CV-SincNet/code/scripts/phase2_support_metric_qknn_probe.py` |
| local/remote SHA256 after sync | `eb8b38cdd9e021de2c42917ff1d7d9fa5dc858722fe5b74b0494c9fb22887261` |
| local verification | `conda run -n ssr-gpu python -m py_compile code\scripts\phase2_support_metric_qknn_probe.py` passed |
| remote verification | `/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python -m py_compile code/scripts/phase2_support_metric_qknn_probe.py` passed |

Artifacts:

| artifact | local path |
|---|---|
| K10 sweep CSV | `E:\type10-7\automation_reports\CV-SincNet\phase2_qknn_hardpair_n20_20260706\artifacts\n20_k10_v23v25_policy_sweep_HP08L5.csv` |
| K10 sweep JSON | `E:\type10-7\automation_reports\CV-SincNet\phase2_qknn_hardpair_n20_20260706\artifacts\n20_k10_v23v25_policy_sweep_HP08L5.json` |
| K5 sweep CSV | `E:\type10-7\automation_reports\CV-SincNet\phase2_qknn_hardpair_n20_20260706\artifacts\n20_k5_v23v25_policy_sweep_HP08L5.csv` |
| K5 sweep JSON | `E:\type10-7\automation_reports\CV-SincNet\phase2_qknn_hardpair_n20_20260706\artifacts\n20_k5_v23v25_policy_sweep_HP08L5.json` |

Best same-row result:

| feature profile | policy family tested | K | old_acc | min_old | seen_new_acc | min_new | weakest new TX | verdict |
|---|---|---:|---:|---:|---:|---:|---|---|
| HP08L5 | v9/v23/v25 sweep | 10 | 93.10% | 81.43% | 84.29% | 68.57% | `2-13` | same as v9, floor fails |
| HP08L5 | v9/v23/v25 sweep | 5 | 92.89% | 81.33% | 76.93% | 58.67% | `1-12` | same as v9, floor fails |

Interpretation:

- Syncing the latest qKNN evaluator was necessary: the remote script was stale and did not support v23/v25 before this step.
- Existing v23/v25 policies do not improve the HP08L5 floor. The bottleneck is now low-K weak-class support/query geometry, especially `2-13` at K10 and `1-12`/`11-10`/`2-13` at K5.
- The next route should not be another fixed per-K parameter set. It should add an adaptive weak-class reliability mechanism that scales from observed support hardness, class count, and K, then gates itself by support-only leave-one-out evidence.

Current goal status: active, not achieved.

## HP08L5 Conservative TTA and Policy Sweep Results

Objective: test whether a conservative receive-side TTA policy (`rx_light5`) preserves the useful LEO feature geometry better than `sat_rx_repair_anchor7`, then check whether existing compressed qKNN adaptive policies can recover old-class performance without storing raw support.

Artifacts pulled locally:

| artifact | local path |
|---|---|
| HP08L5 K10 base CSV | `E:\type10-7\automation_reports\CV-SincNet\phase2_qknn_hardpair_n20_20260706\artifacts\n20_k10_coreproto_hardpair_HP08L5.csv` |
| HP08L5 K5 base CSV | `E:\type10-7\automation_reports\CV-SincNet\phase2_qknn_hardpair_n20_20260706\artifacts\n20_k5_sourceguard_hardpair_HP08L5.csv` |
| HP08L5 K10 policy sweep CSV | `E:\type10-7\automation_reports\CV-SincNet\phase2_qknn_hardpair_n20_20260706\artifacts\n20_k10_policy_sweep_HP08L5.csv` |
| HP08L5 K5 policy sweep CSV | `E:\type10-7\automation_reports\CV-SincNet\phase2_qknn_hardpair_n20_20260706\artifacts\n20_k5_policy_sweep_HP08L5.csv` |

Same-row summary:

| profile | qKNN policy | K | old_acc | min_old | seen_new_acc | min_new | stored_raw_support_count | verdict |
|---|---|---:|---:|---:|---:|---:|---:|---|
| HP08L5 | base head | 10 | 80.71% | 61.43% | 79.43% | 65.71% | 0 | useful new mean, weak floor |
| HP08L5 | base head | 5 | 78.89% | 65.33% | 72.27% | 58.67% | 0 | failed |
| HP08L5 | `stable_dualview_v9` | 10 | 93.10% | 81.43% | 84.29% | 68.57% | 0 | best current N20 K10, floor still fails |
| HP08L5 | `stable_dualview_v9` | 5 | 92.89% | 81.33% | 76.93% | 58.67% | 0 | old recovered, new floor still fails |

Best HP08L5 `stable_dualview_v9` K10 per-TX accuracy:

| TX | acc |
|---|---:|
| `14-10` | 94.29% |
| `14-7` | 82.86% |
| `20-15` | 100.00% |
| `20-19` | 81.43% |
| `6-15` | 100.00% |
| `8-20` | 100.00% |
| `1-1` | 75.71% |
| `1-10` | 90.00% |
| `1-11` | 95.71% |
| `1-12` | 70.00% |
| `1-14` | 82.86% |
| `1-15` | 88.57% |
| `1-16` | 81.43% |
| `1-18` | 82.86% |
| `1-19` | 87.14% |
| `1-2` | 78.57% |
| `10-10` | 90.00% |
| `11-10` | 80.00% |
| `18-5` | 81.43% |
| `19-3` | 87.14% |
| `2-13` | 68.57% |
| `2-5` | 90.00% |
| `3-8` | 95.71% |
| `4-10` | 91.43% |
| `8-18` | 91.43% |
| `8-3` | 77.14% |

Best HP08L5 `stable_dualview_v9` K5 per-TX accuracy:

| TX | acc |
|---|---:|
| `14-10` | 93.33% |
| `14-7` | 82.67% |
| `20-15` | 100.00% |
| `20-19` | 81.33% |
| `6-15` | 100.00% |
| `8-20` | 100.00% |
| `1-1` | 62.67% |
| `1-10` | 88.00% |
| `1-11` | 93.33% |
| `1-12` | 58.67% |
| `1-14` | 62.67% |
| `1-15` | 76.00% |
| `1-16` | 77.33% |
| `1-18` | 66.67% |
| `1-19` | 84.00% |
| `1-2` | 73.33% |
| `10-10` | 93.33% |
| `11-10` | 65.33% |
| `18-5` | 70.67% |
| `19-3` | 73.33% |
| `2-13` | 64.00% |
| `2-5` | 85.33% |
| `3-8` | 92.00% |
| `4-10` | 93.33% |
| `8-18` | 84.00% |
| `8-3` | 74.67% |

Interpretation:

- Conservative receive-side TTA (`rx_light5`) is materially better than strong repair TTA (`sat_rx_repair_anchor7`), so the useful direction is light invariance averaging, not aggressive canonical repair.
- `stable_dualview_v9` recovers old-class performance on HP08L5 and improves N20 K10 new mean to 84.29%, with no raw support storage. This is the strongest current N20 K10 row in this branch.
- The active target is still not met. K10 minimum new class is `2-13` at 68.57%; K5 minimum new class is `1-12` at 58.67%. The next required optimization must explicitly protect low-K weak new classes without per-K hand tuning.
- Storage remains deployment-aligned for the classifier head: quantized support code count is 260 for K10 and 130 for K5; stored raw support count is 0.

Current goal status: active, not achieved.

## HP08R7 Receive-Side TTA Results

Artifacts pulled locally:

| artifact | local path |
|---|---|
| K10 CSV | `E:\type10-7\automation_reports\CV-SincNet\phase2_qknn_hardpair_n20_20260706\artifacts\n20_k10_coreproto_hardpair_HP08R7.csv` |
| K10 JSON | `E:\type10-7\automation_reports\CV-SincNet\phase2_qknn_hardpair_n20_20260706\artifacts\n20_k10_coreproto_hardpair_HP08R7.json` |
| K5 CSV | `E:\type10-7\automation_reports\CV-SincNet\phase2_qknn_hardpair_n20_20260706\artifacts\n20_k5_sourceguard_hardpair_HP08R7.csv` |
| K5 JSON | `E:\type10-7\automation_reports\CV-SincNet\phase2_qknn_hardpair_n20_20260706\artifacts\n20_k5_sourceguard_hardpair_HP08R7.json` |

Best same-row result per K:

| profile | TTA policy | K | old_acc | min_old | seen_new_acc | min_new | stored_raw_support_count | verdict |
|---|---|---:|---:|---:|---:|---:|---:|---|
| HP08R7 | `sat_rx_repair_anchor7` | 10 | 64.76% | 45.71% | 63.29% | 42.86% | 0 | failed, worse than baseline |
| HP08R7 | `sat_rx_repair_anchor7` | 5 | 61.56% | 29.33% | 52.27% | 26.67% | 0 | failed, worse than baseline |

K10 per-TX accuracy:

| TX | acc |
|---|---:|
| `14-10` | 58.57% |
| `14-7` | 51.43% |
| `20-15` | 72.86% |
| `20-19` | 45.71% |
| `6-15` | 68.57% |
| `8-20` | 91.43% |
| `1-1` | 58.57% |
| `1-10` | 85.71% |
| `1-11` | 81.43% |
| `1-12` | 57.14% |
| `1-14` | 64.29% |
| `1-15` | 60.00% |
| `1-16` | 67.14% |
| `1-18` | 42.86% |
| `1-19` | 48.57% |
| `1-2` | 62.86% |
| `10-10` | 48.57% |
| `11-10` | 48.57% |
| `18-5` | 62.86% |
| `19-3` | 61.43% |
| `2-13` | 51.43% |
| `2-5` | 67.14% |
| `3-8` | 87.14% |
| `4-10` | 72.86% |
| `8-18` | 77.14% |
| `8-3` | 60.00% |

K5 per-TX accuracy:

| TX | acc |
|---|---:|
| `14-10` | 38.67% |
| `14-7` | 58.67% |
| `20-15` | 73.33% |
| `20-19` | 29.33% |
| `6-15` | 77.33% |
| `8-20` | 92.00% |
| `1-1` | 37.33% |
| `1-10` | 82.67% |
| `1-11` | 73.33% |
| `1-12` | 48.00% |
| `1-14` | 38.67% |
| `1-15` | 58.67% |
| `1-16` | 62.67% |
| `1-18` | 36.00% |
| `1-19` | 28.00% |
| `1-2` | 26.67% |
| `10-10` | 36.00% |
| `11-10` | 45.33% |
| `18-5` | 57.33% |
| `19-3` | 41.33% |
| `2-13` | 26.67% |
| `2-5` | 64.00% |
| `3-8` | 82.67% |
| `4-10` | 69.33% |
| `8-18` | 74.67% |
| `8-3` | 56.00% |

Interpretation:

- Strong receive-side repair averaging is destructive for this adapter feature geometry. It lowers old accuracy below the OLD80 stage gate and also lowers new-class mean/floor.
- The negative result does not invalidate the compressed qKNN route; it specifically rejects `sat_rx_repair_anchor7` feature/logit mean export as a mainline improvement.
- A more conservative TTA control such as `rx_light5` is still worth one bounded check to determine whether the failure is caused by strong repair transforms or by any multi-view averaging at export time.

Current goal status: active, not achieved.

## HP08R7 Receive-Side TTA Launch

Timestamp: 2026-07-06 10:55 CST
Local commit synced: `f21d2c71807e41f11a583f83b1cd335b84951812`
Remote root: `/home/szu2070436088/2510044040/CV-SincNet`

Sync mapping:

| local file | remote file |
|---|---|
| `E:\type10-7\github_publish\CVS-RFFI-repo\code\scripts\train_apply_phase1_iq_preadapter_20260703.py` | `/home/szu2070436088/2510044040/CV-SincNet/code/scripts/train_apply_phase1_iq_preadapter_20260703.py` |
| `E:\type10-7\github_publish\CVS-RFFI-repo\code\scripts\launch_phase2_qknn_hardpair_n20_v1.sh` | `/home/szu2070436088/2510044040/CV-SincNet/code/scripts/launch_phase2_qknn_hardpair_n20_v1.sh` |

Remote verification:

| command | result |
|---|---|
| `/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python -m py_compile code/scripts/train_apply_phase1_iq_preadapter_20260703.py` | passed |
| `bash -n code/scripts/launch_phase2_qknn_hardpair_n20_v1.sh` | passed |
| `sha256sum code/scripts/train_apply_phase1_iq_preadapter_20260703.py code/scripts/launch_phase2_qknn_hardpair_n20_v1.sh` | `0400c813...`, `82201c3a...`, matched local |

Launch command:

```bash
cd /home/szu2070436088/2510044040/CV-SincNet
PROFILE=HP08R7 SATELLITE_TTA_POLICY=sat_rx_repair_anchor7 GPU=0 RUN_ROOT=/home/szu2070436088/2510044040/CV-SincNet/runs/phase2_qknn_hardpair_n20_20260706 \
  bash code/scripts/launch_phase2_qknn_hardpair_n20_v1.sh
```

Observed remote status:

| field | value |
|---|---|
| launcher pid | `3389538` |
| train/export pid | `3389540` |
| GPU | `0` |
| log | `/home/szu2070436088/2510044040/CV-SincNet/logs/phase2_qknn_hardpair_n20_20260706_hp08r7_tta.log` |
| startup health | reached epoch 15, no traceback/OOM/argparse error in initial tail |

Expected result files:

| artifact | expected path |
|---|---|
| feature NPZ | `/home/szu2070436088/2510044040/CV-SincNet/runs/phase2_qknn_hardpair_n20_20260706/MANYNEW20_HARDPAIR_HP08R7/ADV3B02_CORE90_SOFT_E200_PHASE1_HARDPAIR_HP08R7_N20/features_hardpair_HP08R7_n20.npz` |
| K10 csv/json | `/home/szu2070436088/2510044040/CV-SincNet/runs/phase2_qknn_hardpair_n20_20260706/HP08R7/qknn_eval/n20_k10_coreproto_hardpair_HP08R7.csv/json` |
| K5 csv/json | `/home/szu2070436088/2510044040/CV-SincNet/runs/phase2_qknn_hardpair_n20_20260706/HP08R7/qknn_eval/n20_k5_sourceguard_hardpair_HP08R7.csv/json` |

Current goal status: active, not achieved.

## Receive-Side TTA Mean Export Plan

Timestamp: 2026-07-06
Operator: Codex
Objective: test whether a receive-side LEO repair/TTA feature export can improve the current compressed qKNN route under many-new-class load without adding K anchors or storing raw support samples.

Hypothesis: the raw-IQ diagnostic shows clean transmitter identity is strong but LEO raw IQ collapses. Therefore the next useful route is not raw-support kNN; it is a lightweight receive-side repair view applied after the same LEO observation and before the frozen Phase1 feature extractor / IQ pre-adapter. The exported feature row remains one row per physical sample by averaging feature/logit outputs across repair views, so support/query size and split permissions do not change.

Protocol boundaries:

| item | value |
|---|---|
| source model | `ADV3B02_CORE90_SOFT_E200` |
| target receiver domain | `7-14` |
| old TX | `14-10,14-7,20-15,20-19,6-15,8-20` |
| new TX main load | 20 ManyTx non-old classes |
| LEO view | `leo_clear_weak,leo_low_elev_weak,leo_rain_weak` |
| K anchors | `K=5,K=10` only |
| query/class | K5: 75, K10: 70 |
| support/query permission | both from target receiver domain, held-out query excludes support pool |
| unknown rejection | not used for this no-rejection objective |
| TTA storage semantics | no raw support storage; one aggregated feature/logit row per physical sample |

Local code changes prepared:

| file | purpose |
|---|---|
| `code/scripts/train_apply_phase1_iq_preadapter_20260703.py` | add `--satellite_tta_policy`; during satellite export, generate receive-side repair views and average feature/logit outputs into one row per physical sample |
| `code/scripts/launch_phase2_qknn_hardpair_n20_v1.sh` | add `SATELLITE_TTA_POLICY` environment variable, default `none`, preserving old experiment behavior |

Local verification:

| command | result |
|---|---|
| `conda run -n ssr-gpu python -m py_compile code\scripts\train_apply_phase1_iq_preadapter_20260703.py` | passed |
| `bash -n code/scripts/launch_phase2_qknn_hardpair_n20_v1.sh` | passed |

Planned N607 run:

| field | value |
|---|---|
| run profile | `HP08R7` |
| TTA policy | `sat_rx_repair_anchor7` |
| launcher | `code/scripts/launch_phase2_qknn_hardpair_n20_v1.sh` |
| environment override | `PROFILE=HP08R7 SATELLITE_TTA_POLICY=sat_rx_repair_anchor7` |
| expected outputs | `features_hardpair_HP08R7_n20.npz`, `n20_k10_coreproto_hardpair_HP08R7.csv/json`, `n20_k5_sourceguard_hardpair_HP08R7.csv/json` |
| comparison target | current N20 adaptive v9 / HP08REF rows, especially min_new floor |

Success criteria for this diagnostic: improve N20 new-class floor and preserve old_acc>=80% for both K=5 and K=10. Full goal completion still requires stable performance across increasing new-class counts and the active no-collapse target, so this run is only one step toward the goal.

Current goal status: active, not achieved.

## v26 Query-Cluster Negative Diagnostic

Objective: test whether a stronger adaptive query-cluster prototype step can rescue the multi-new-class floor while preserving the compressed qKNN support state. This was a local diagnostic only; the temporary `dualview_support_v26` code was removed after verification because it degraded performance.

Artifacts:

| artifact | status |
|---|---|
| `artifacts\n20_k10_v26_quota_cluster_seed421029.csv` | completed |
| `artifacts\n20_k10_v26_quota_cluster_seed421029.json` | completed |
| `artifacts\n20_k5_v26_quota_cluster_seed421037.csv` | completed |
| `artifacts\n20_k5_v26_quota_cluster_seed421037.json` | completed |

Result:

| route | K | old_acc | min_old | new_acc | min_new | verdict |
|---|---:|---:|---:|---:|---:|---|
| v26 strong query-cluster | 10 | 92.62% | 78.57% | 69.71% | 52.86% | worse than v9/v23 |
| v26 strong query-cluster | 5 | 92.22% | 78.67% | 69.67% | 40.00% | worse than v9/v23 |

Interpretation:

- Stronger query-cluster alignment amplifies wrong clusters instead of correcting hard classes. The K5 floor falls to 40.00%, so this is not a stable adaptive route.
- The failed v26 code was not kept as a production candidate. The evidence is retained as a negative diagnostic.

## Aligned HP08REF Representation Check

Objective: verify whether the aligned hard-pair representation export fixes the N20 collapse after the earlier misaligned dual-view rows were invalidated.

Artifacts:

| artifact | best old_acc | best min_old | best new_acc | best min_new | verdict |
|---|---:|---:|---:|---:|---|
| `aligned_HP08REF\n20_k10_coreproto_hardpair_HP08REF.csv` | 80.71% | 62.86% | 76.93% | 62.86% | old floor failed |
| `aligned_HP08REF\n20_k5_sourceguard_hardpair_HP08REF.csv` | 74.89% | 48.00% | 69.20% | 41.33% | failed |
| `artifacts\n20_k10_norm_hp08ref_adaptive_v9_seed5.csv` | 94.29% | 85.71% | 63.86% | 48.57% | failed |
| `artifacts\n20_k5_norm_hp08ref_adaptive_v9_seed5.csv` | 94.44% | 85.33% | 48.80% | 28.00% | failed |

Interpretation:

- Aligned HP08REF is not the missing representation fix. Standalone HP08REF can raise K10 mean new accuracy but destroys old-class floor; dual-view NORM+HP08REF hurts new-class performance.
- Combined with the raw-IQ and v26 diagnostics, the current evidence says the remaining failure is hard-class separability under the ADV3B02 feature geometry, not lack of raw support storage or a simple transductive quota-cluster step.

Current goal status: active, not achieved.

## Raw-IQ kNN Baseline Plan

Objective: test whether a plain kNN classifier over original IQ samples can remain stable when the number of new classes increases, before adding more qKNN representation-side mechanisms.

Planned command surface:

| item | value |
|---|---|
| local script | `code/scripts/phase2_raw_iq_knn_probe.py` |
| feature metadata | `E:\type10-7\automation_reports\CV-SincNet\phase2_qknn_adaptive_manynew_20260705\n20_features\features_n20_norm.npz` |
| raw old pkl | `/home/szu2070436088/2510044040/CV-SincNet/Dataset_WigSig/ManySig.pkl` |
| raw new pkl | `/home/szu2070436088/2510044040/CV-SincNet/Dataset_WigSig/ManyTx.pkl` |
| target receiver domain | `rx=7-14` from feature manifest |
| old TX | `14-10,14-7,20-15,20-19,6-15,8-20` |
| new TX counts | `10` and `20` |
| K-shot values | `K=5,K=10` only |
| query size | max available per class: `75` for K5, `70` for K10 |
| classifier | flattened RMS-normalized raw IQ, cosine 1-NN |
| views | clean raw IQ and LEO raw IQ (`leo_clear_weak,leo_low_elev_weak,leo_rain_weak`) |
| training cost | none; support-only non-parametric diagnostic |
| deployment storage | raw baseline stores `class_count*K*2*out_len` scalars, intentionally not compressed |

This is a diagnostic baseline, not a proposed final deployment route, because it retains raw support samples.

## Raw-IQ kNN Baseline Results

Artifacts:

| artifact | status |
|---|---|
| `artifacts\rawiq_knn_n10_k5k10_seed421027.csv` | completed |
| `artifacts\rawiq_knn_n10_k5k10_seed421027.json` | completed |
| `artifacts\rawiq_knn_n20_k5k10_seed421027.csv` | completed |
| `artifacts\rawiq_knn_n20_k5k10_seed421027.json` | completed |

Joint summary:

| view | new classes | K | old_acc | min_old | new_acc | min_new | verdict |
|---|---:|---:|---:|---:|---:|---:|---|
| clean raw IQ | 10 | 5 | 91.33% | 62.67% | 81.47% | 44.00% | fails floor |
| LEO raw IQ | 10 | 5 | 26.67% | 13.33% | 21.33% | 9.33% | collapses |
| clean raw IQ | 10 | 10 | 97.14% | 92.86% | 96.00% | 75.71% | passes new floor, clean only |
| LEO raw IQ | 10 | 10 | 27.62% | 14.29% | 35.14% | 8.57% | collapses |
| clean raw IQ | 20 | 5 | 91.33% | 62.67% | 83.73% | 44.00% | fails floor |
| LEO raw IQ | 20 | 5 | 20.44% | 14.67% | 19.80% | 2.67% | collapses |
| clean raw IQ | 20 | 10 | 97.14% | 92.86% | 93.43% | 68.57% | fails floor |
| LEO raw IQ | 20 | 10 | 25.48% | 15.71% | 27.43% | 1.43% | collapses |

Per-TX details for the deployment-relevant LEO raw-IQ view:

| TX | N10 K5 | N10 K10 | N20 K5 | N20 K10 |
|---|---:|---:|---:|---:|
| `14-10` | 45.33% | 21.43% | 25.33% | 21.43% |
| `14-7` | 13.33% | 15.71% | 17.33% | 18.57% |
| `20-15` | 14.67% | 30.00% | 20.00% | 15.71% |
| `20-19` | 36.00% | 14.29% | 14.67% | 21.43% |
| `6-15` | 16.00% | 44.29% | 17.33% | 32.86% |
| `8-20` | 34.67% | 40.00% | 28.00% | 42.86% |
| `10-10` | 10.67% | 8.57% | 4.00% | 11.43% |
| `11-10` | 9.33% | 32.86% | 24.00% | 18.57% |
| `18-5` | 44.00% | 74.29% | 50.67% | 78.57% |
| `19-3` | 10.67% | 27.14% | 17.33% | 21.43% |
| `2-13` | 10.67% | 40.00% | 22.67% | 31.43% |
| `2-5` | 20.00% | 15.71% | 21.33% | 22.86% |
| `3-8` | 28.00% | 44.29% | 29.33% | 18.57% |
| `4-10` | 13.33% | 27.14% | 9.33% | 8.57% |
| `8-18` | 50.67% | 60.00% | 40.00% | 58.57% |
| `8-3` | 16.00% | 21.43% | 12.00% | 10.00% |
| `1-1` | n/a | n/a | 2.67% | 22.86% |
| `1-10` | n/a | n/a | 5.33% | 37.14% |
| `1-11` | n/a | n/a | 12.00% | 25.71% |
| `1-12` | n/a | n/a | 16.00% | 14.29% |
| `1-14` | n/a | n/a | 14.67% | 28.57% |
| `1-15` | n/a | n/a | 24.00% | 32.86% |
| `1-16` | n/a | n/a | 38.67% | 54.29% |
| `1-18` | n/a | n/a | 12.00% | 15.71% |
| `1-19` | n/a | n/a | 29.33% | 35.71% |
| `1-2` | n/a | n/a | 10.67% | 1.43% |

Clean raw-IQ control highlights:

| setting | weakest old TX | weakest new TX |
|---|---|---|
| N10,K5 | `14-7` 62.67% | `19-3` 44.00% |
| N10,K10 | `14-7`/`20-15` 92.86% | `2-5` 75.71% |
| N20,K5 | `14-7` 62.67% | `19-3` 44.00% |
| N20,K10 | `14-7`/`20-15` 92.86% | `2-5` 68.57% |

Interpretation:

- Plain raw-IQ 1-NN is not a viable satellite-domain method. Once support/query are evaluated under LEO channel, both old and new classes collapse far below the current qKNN feature route.
- Clean raw-IQ K10 shows that the physical transmitter signature is present in the original signal, but it is not invariant to the LEO channel. This supports the current direction: keep the ADV3B02 feature extractor or a channel-invariant front-end, then compress/adapt the support representation.
- The raw baseline also confirms why storing raw support samples is not enough for deployment: storage is `K*class_count*2*256` scalars and still fails under LEO.
- Current goal remains active and unmet; raw-IQ kNN is diagnostic evidence against using raw samples directly as the final KNN memory.

## v25 Top2 Pair Gate Diagnostic

Objective: test a more deployable hard-pair repair after the `2-13`/`11-10` reciprocal confusion diagnosis. The new route keeps the K anchors fixed at `K=5` and `K=10`, uses strict `pool_per_class=K`, and stores only compressed pair parameters.

Implementation change:

- Added `dualview_support_v25` / `stable_dualview_v25` in `code/scripts/phase2_support_metric_qknn_probe.py`.
- v25 adds a support-LOO-derived top2 hard-pair gate. It detects confused new-class pairs from support LOO, learns a 4-scalar prototype-similarity pair boundary, and applies it only when a new-query row's top2 new-class scores are the same hard pair and the score gap is within an adaptive margin.
- Persistent state remains compressed: quantized support codes, class prototypes, transform/residual scalars, and top2 pair-gate scalars. No raw support samples are stored.

Verification:

| command/artifact | result |
|---|---|
| `conda run -n ssr-gpu python -m py_compile code\scripts\phase2_support_metric_qknn_probe.py` | PASS |
| `artifacts\n10_k5_v23_linear_probe_small_seed421037.csv` | completed |
| `artifacts\n10_k10_v23_linear_probe_small_seed421027.csv` | completed |
| `artifacts\n10_k5_v25_top2gate_seed421037.csv` | completed |
| `artifacts\n10_k10_v25_top2gate_seed421027.csv` | completed |

Strict K-shot comparison:

| route | K | old_acc | min_old | new_acc | min_new | added compressed pair state | verdict |
|---|---:|---:|---:|---:|---:|---:|---|
| v23 baseline | 5 | 91.56% | 77.33% | 86.00% | 64.00% | 0 | baseline |
| v23 + full linear pair probe | 5 | 91.56% | 77.33% | 86.00% | 64.00% | up to 322 scalars in tested rows | no gain |
| v25 top2 pair gate | 5 | 91.56% | 77.33% | 86.00% | 64.00% | 18 scalars | no gain |
| v23 baseline | 10 | 92.62% | 80.00% | 87.71% | 67.14% | 0 | baseline |
| v23 + full linear pair probe | 10 | 92.62% | 80.00% | 87.71% | 67.14% | up to 322 scalars in tested rows | no gain |
| v25 top2 pair gate | 10 | 92.62% | 80.00% | 87.71% | 67.14% | 18 scalars | no gain |

v25 selected hard pairs:

| setting | top2 pair-gate pairs | adaptive weight | adaptive margin |
|---|---|---:|---:|
| N10,K=5 | `11-10<->18-5;2-5<->8-3;11-10<->2-13` | 0.0483 | 0.3156 |
| N10,K=10 | `11-10<->2-13;11-10<->18-5;2-13<->3-8` | 0.0550 | 0.3756 |

Interpretation:

- The full-vector pair-linear probe and the compressed v25 top2 pair gate both fail to move the query floor. This means the remaining errors are not being corrected by a small post-hoc pair boundary fitted from strict support only.
- v25 is still useful as a bounded negative diagnostic: it shows that a star-deployable compressed pair gate can be added without harming old/new means, but it does not solve the floor target.
- The next credible route is representation-side hard-pair separation or support feature repair before qKNN scoring, not additional K anchors and not stronger post-hoc pair bias.

Current goal status: active, not achieved.

## Active-Pool80 Support Compression Diagnostic

Objective: test whether the N10 floor collapse is mainly caused by unlucky strict K-shot support sampling. This diagnostic still deploys only `K=5` or `K=10` quantized support codes per class, but selects those K samples from an 80-sample labeled enrollment pool per class. Because `pool_per_class > K`, this is an active-enrollment upper-bound diagnostic, not a strict K-shot claim.

Artifacts:

| artifact | status |
|---|---|
| `artifacts\n10_k5_v23_active_pool80_seed421037.csv` | completed |
| `artifacts\n10_k5_v23_active_pool80_seed421037.json` | completed |
| `artifacts\n10_k10_v23_active_pool80_seed421027.csv` | completed |
| `artifacts\n10_k10_v23_active_pool80_seed421027.json` | completed |

Result summary:

| setting | support selection | old_acc | min_old | new_acc | min_new | weakest new class | verdict |
|---|---|---:|---:|---:|---:|---|---|
| N10,K=5,pool80 | `stable_first` | 90.44% | 73.33% | 85.20% | 62.67% | `2-13` | best floor, failed |
| N10,K=5,pool80 | `centroid` | 90.89% | 76.00% | 84.13% | 57.33% | `2-13` | failed |
| N10,K=5,pool80 | `scenario_centroid` | 90.89% | 76.00% | 83.33% | 54.67% | `2-13` | failed |
| N10,K=5,pool80 | `scenario_diverse` | 92.89% | 78.67% | 82.53% | 46.67% | `2-13` | failed |
| N10,K=10,pool80 | `stable_first` | 90.95% | 75.71% | 86.14% | 57.14% | `2-13` | best floor, failed |
| N10,K=10,pool80 | `centroid` | 91.19% | 77.14% | 83.29% | 51.43% | `2-13` | failed |
| N10,K=10,pool80 | `scenario_centroid` | 91.19% | 77.14% | 83.29% | 51.43% | `2-13` | failed |
| N10,K=10,pool80 | `scenario_diverse` | 91.90% | 75.71% | 77.86% | 25.71% | `2-13` | failed |

Per-new-class details for the best floor rows:

| TX | K5 stable_first | K10 stable_first |
|---|---:|---:|
| `10-10` | 89.33% | 94.29% |
| `11-10` | 73.33% | 62.86% |
| `18-5` | 92.00% | 95.71% |
| `19-3` | 94.67% | 100.00% |
| `2-13` | 62.67% | 57.14% |
| `2-5` | 86.67% | 88.57% |
| `3-8` | 90.67% | 92.86% |
| `4-10` | 88.00% | 90.00% |
| `8-18` | 84.00% | 88.57% |
| `8-3` | 90.67% | 91.43% |

Interpretation:

- Active selection from an 80-sample labeled pool does not rescue the floor. The best `pool80` floor is worse than the strict-K seed ceiling already observed for both K anchors: K5 `62.67%` versus strict K5 `64.00%`, and K10 `57.14%` versus strict K10 `67.14%`.
- `2-13` remains the weakest class across every support-selection policy, while `11-10` also remains fragile. This rejects simple centroid/diversity enrollment compression as the main fix.
- The next credible optimization must change the representation or add a genuinely support-derived adaptive correction targeted at hard-pair separability; merely selecting "better" K samples from a larger same-class pool is not enough under the current feature geometry.

Current goal status: active, not achieved.

## Day-Aware Domain Refine Diagnostic

Objective: test whether target-day metadata can provide a protocol-safe adaptive correction for the strict N10 setting without increasing K count. The anchors remain `K=5` and `K=10`; query is still the dataset maximum after support selection: `K=5` uses 75 query samples per class and `K=10` uses 70 query samples per class.

Implementation change:

- Added `day` and `day_scenario` domain-refinement keys in `code/scripts/phase2_support_metric_qknn_probe.py`.
- The probe now reads `day_ids` from the feature NPZ and can form domain priors by day alone or by day plus LEO scenario.
- This is an evaluation/diagnostic capability only; no N607 run was launched.

Verification:

| command/artifact | result |
|---|---|
| `conda run -n ssr-gpu python -m py_compile code\scripts\phase2_support_metric_qknn_probe.py` | PASS |
| `artifacts\n10_k5_v23_dayrefine_seed421037.csv` | completed |
| `artifacts\n10_k10_v23_dayrefine_seed421027.csv` | completed |

Best strict-K rows after adding day/domain refinement:

| setting | best domain key | weight | scope | old_acc | min_old | new_acc | min_new | verdict |
|---|---|---:|---|---:|---:|---:|---:|---|
| N10,K=5,seed421037 | `none` | 0.0 | `new` | 91.56% | 77.33% | 86.00% | 64.00% | unchanged |
| N10,K=10,seed421027 | `none` | 0.0 | `new` | 92.62% | 80.00% | 87.71% | 67.14% | unchanged |

Interpretation:

- Non-zero `day` and `day_scenario` refinement does not improve the N10 weakest-new-class floor. Rows with `day`/`day_scenario` only tie the baseline when the refinement weight is 0.0, so they are not real improvements.
- The repeated weakest class remains below the 75% target under strict K-shot. This rejects target-day metadata as the current primary repair path.
- The evidence strengthens the current diagnosis: with strict `pool_per_class=K`, performance is dominated by support-instance geometry and hard-class feature separability, not by missing day/scenario metadata priors.

Current goal status: active, not achieved.

## v24 Negative Diagnostic and N10 Support-Seed Ceiling

Timestamp: 2026-07-06 local.

Objective:继续在不扩大K数量的前提下优化qKNN。仍只评估`K=5`和`K=10`，并保持最大query划分：`K=5`每类75条query，`K=10`每类70条query。

Local code change:

| file | purpose |
|---|---|
| `code/scripts/phase2_support_metric_qknn_probe.py` | 追加`dualview_support_v24`/`stable_dualview_v24`诊断策略；v24继承v23的压缩qKNN状态，不保存raw support，并测试更强的临时query-cluster对齐 |

Verification:

| command/artifact | result |
|---|---|
| `conda run -n ssr-gpu python -m py_compile code\scripts\phase2_support_metric_qknn_probe.py` | PASS |
| `artifacts\n10_k5_v24_seed421037.json` | completed; negative diagnostic |
| `artifacts\n10_k5_v23_smallgrid_seed421037.csv` | completed; no better topm/proto setting found |
| `artifacts\n10_k5_v23_seedscan20_421020.csv` | completed; 20 support seeds |
| `artifacts\n10_k10_v23_seedscan20_421020.csv` | completed; 20 support seeds |

v24 result:

| scope | K | seed | old_acc | min_old | new_acc | min_new | weakest class | verdict |
|---|---:|---:|---:|---:|---:|---:|---|---|
| N10 v23 baseline | 5 | 421037 | 91.56% | 77.33% | 86.00% | 64.00% | `2-13` | current best baseline |
| N10 v24 strong query-cluster | 5 | 421037 | 91.56% | 77.33% | 86.00% | 61.33% | `2-13` | worse; reject |

N10 support-seed ceiling under v23:

| K | seeds | best seed | best old_acc | best min_old | best new_acc | best min_new | weakest class at best |
|---:|---:|---:|---:|---:|---:|---:|---|
| 5 | 421020-421039 | 421037 | 91.56% | 77.33% | 86.00% | 64.00% | `2-13` |
| 10 | 421020-421039 | 421027 | 92.62% | 80.00% | 87.71% | 67.14% | `2-13` |

Top support-seed rows:

| K | seed | old_acc | min_old | new_acc | min_new | key per-new details |
|---:|---:|---:|---:|---:|---:|---|
| 5 | 421037 | 91.56% | 77.33% | 86.00% | 64.00% | `2-13=64.00%`,`11-10=76.00%`,`18-5=94.67%` |
| 5 | 421031 | 90.44% | 73.33% | 86.00% | 64.00% | `2-13=64.00%`,`11-10=69.33%`,`18-5=86.67%` |
| 5 | 421036 | 91.33% | 76.00% | 85.87% | 64.00% | `2-13=64.00%`,`11-10=69.33%`,`18-5=86.67%` |
| 10 | 421027 | 92.62% | 80.00% | 87.71% | 67.14% | `2-13=67.14%`,`11-10=75.71%`,`18-5=91.43%` |
| 10 | 421034 | 92.14% | 78.57% | 86.14% | 65.71% | `2-13=65.71%`,`11-10=74.29%`,`18-5=91.43%` |

Diagnosis:

- `2-13` is the stable floor class. In the fixed K5 baseline, its query confusion is concentrated as `2-13->11-10` with 16/75 errors, while `11-10->2-13` also has 16/75 errors. This is a symmetric local-boundary failure.
- Support-only geometry identifies `2-13` as close to `11-10`, but support-LOO is not a reliable proxy for query floor: classes such as `11-10` and `18-5` can have poor support-LOO while query accuracy remains acceptable.
- v24's stronger temporary query-cluster does not repair the floor. It lowers `2-13` from 64.00% to 61.33%, so it is not promotable.
- The 20-seed scan shows that changing the K-shot support draw alone does not reach the requested floor. Best observed N10 floor is 64.00% at `K=5` and 67.14% at `K=10`, still below the 75% target.

Interpretation:

The current qKNN compressed-head route remains useful for extensibility and onboard storage: it stores quantized support codes, prototypes, transform scalars, and small rescue/proxy/logistic scalars, not raw support samples. However, for ten new classes the active floor target is not achieved. The bottleneck is no longer K expansion or raw-support storage; it is the representation/enrollment separability of `2-13` against `11-10` and related hard neighbors. The next credible route is a representation or enrollment-quality repair for these hard ManyTx classes, while preserving `K=5,K=10` and the no-raw-support deployment constraint.

Current goal status: active, not achieved.

## Adaptive v23 Support-Gated Aux Safety

Objective: continue optimizing qKNN without expanding the K grid. The only evaluated anchors remain `K=5` and `K=10`, using the maximum query split from the 80-sample-per-class feature file. This run tests whether an adaptive support-only auxiliary-view reliability gate can improve stability when the number of enrolled new classes increases.

Implementation change:

- Added `dualview_support_v23` / `stable_dualview_v23` in `code/scripts/phase2_support_metric_qknn_probe.py`.
- v23 keeps the compressed qKNN deployment property: no raw support vectors are stored. Persistent state is quantized support codes, class prototypes, transform scalars, and small rescue/proxy scalars.
- v23 extends support-only auxiliary-view gating to the current query-cluster lineage. Before mixing an auxiliary view, it computes primary-view and auxiliary-view support LOO mean/minimum-class accuracy. If the auxiliary view has weak minimum-class support LOO or does not improve support reliability enough, `effective_aux_score_weight` is reduced toward zero.
- The query-cluster execution guard now treats near-zero weights as zero, so overload-gated N20 K5 does not report misleading temporary cluster rows.

Verification:

| command/artifact | result |
|---|---|
| `conda run -n ssr-gpu python -m py_compile code\scripts\phase2_support_metric_qknn_probe.py` | PASS |
| `artifacts\n10_k5_v23_seed421037.json` | completed |
| `artifacts\n10_k10_v23_seed421029.json` | completed |
| `artifacts\n10_k5_v23_head_seed421037.json` | completed; HEAD aux rejected by support gate |
| `artifacts\n10_k10_v23_head_seed421029.json` | completed; HEAD aux effective weight 4.18% |
| `artifacts\n20_k5_v23_seed421037.json` | completed |
| `artifacts\n20_k10_v23_seed421029.json` | completed |

Maximum-query summary:

| scope | K | query/class | old_acc | min_old | new_acc | min_new | effective_aux | query-cluster rows | verdict |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| N10 NORM | 5 | 75 | 91.56% | 77.33% | 86.00% | 64.00% | 0.00% | 750 | failed floor |
| N10 NORM | 10 | 70 | 92.14% | 77.14% | 85.43% | 64.29% | 0.00% | 700 | failed floor |
| N10 NORM+HEAD | 5 | 75 | 91.56% | 77.33% | 86.00% | 64.00% | 0.00% | 750 | aux safely rejected |
| N10 NORM+HEAD | 10 | 70 | 92.14% | 77.14% | 85.43% | 64.29% | 4.18% | 700 | no gain |
| N20 NORM | 5 | 75 | 92.22% | 78.67% | 69.33% | 48.00% | 0.00% | 0 | failed; many-new collapse persists |
| N20 NORM | 10 | 70 | 92.62% | 78.57% | 70.14% | 51.43% | 0.00% | 0 | failed; many-new collapse persists |

N10 per-TX details:

| TX | role | K5 acc | K10 acc |
|---|---|---:|---:|
| `14-10` | old | 96.00% | 95.71% |
| `14-7` | old | 77.33% | 80.00% |
| `20-15` | old | 98.67% | 100.00% |
| `20-19` | old | 77.33% | 77.14% |
| `6-15` | old | 100.00% | 100.00% |
| `8-20` | old | 100.00% | 100.00% |
| `10-10` | new | 90.67% | 91.43% |
| `11-10` | new | 76.00% | 78.57% |
| `18-5` | new | 94.67% | 95.71% |
| `19-3` | new | 97.33% | 95.71% |
| `2-13` | new | 64.00% | 64.29% |
| `2-5` | new | 84.00% | 81.43% |
| `3-8` | new | 88.00% | 91.43% |
| `4-10` | new | 88.00% | 87.14% |
| `8-18` | new | 84.00% | 77.14% |
| `8-3` | new | 93.33% | 91.43% |

N20 per-TX details:

| TX | role | K5 acc | K10 acc |
|---|---|---:|---:|
| `14-10` | old | 97.33% | 95.71% |
| `14-7` | old | 78.67% | 81.43% |
| `20-15` | old | 98.67% | 100.00% |
| `20-19` | old | 78.67% | 78.57% |
| `6-15` | old | 100.00% | 100.00% |
| `8-20` | old | 100.00% | 100.00% |
| `10-10` | new | 80.00% | 84.29% |
| `11-10` | new | 60.00% | 55.71% |
| `18-5` | new | 50.67% | 62.86% |
| `19-3` | new | 56.00% | 62.86% |
| `2-13` | new | 49.33% | 51.43% |
| `2-5` | new | 81.33% | 78.57% |
| `3-8` | new | 86.67% | 80.00% |
| `4-10` | new | 88.00% | 88.57% |
| `8-18` | new | 82.67% | 70.00% |
| `8-3` | new | 77.33% | 72.86% |
| `1-1` | new | 69.33% | 57.14% |
| `1-10` | new | 86.67% | 84.29% |
| `1-11` | new | 85.33% | 85.71% |
| `1-12` | new | 66.67% | 62.86% |
| `1-14` | new | 48.00% | 68.57% |
| `1-15` | new | 76.00% | 80.00% |
| `1-16` | new | 65.33% | 70.00% |
| `1-18` | new | 48.00% | 57.14% |
| `1-19` | new | 68.00% | 72.86% |
| `1-2` | new | 61.33% | 57.14% |

Interpretation:

- v23 improves safety, not accuracy. It prevents harmful auxiliary-view fusion without storing raw support samples or using query labels, but it does not lift the weakest new class.
- Ten-new-class target is still not achieved: `2-13` remains at 64.00%/64.29%, below the 75% per-class floor.
- Twenty-new-class pressure test still collapses: N20 minimum new-class accuracy is 48.00%/51.43%, far below the requested stability requirement.
- The remaining failure is concentrated in feature/local-boundary separability for `2-13`,`11-10`,`18-5`,`19-3`,`1-14`,`1-18`,`1-2`, not in K expansion or raw-support storage.

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

## Assignment and Hard-Pair Diagnostic

Objective: determine whether the persistent `2-13`/`11-10` weakness is caused by role-balanced assignment, support selection, or feature separability. This used the same strict K-shot support/query split and did not increase K.

Artifacts:

| artifact | status |
|---|---|
| `artifacts\n10_k10_v23_noassign_seed421027.csv` | completed |
| `artifacts\n10_k5_v23_noassign_seed421037.csv` | completed |
| `artifacts\n10_k10_raw_noassign_seed421027.csv` | completed |

Strict K-shot role-balanced comparison:

| setting | adaptive policy | assignment state | old_acc | min_old | new_acc | min_new | key detail |
|---|---|---|---:|---:|---:|---:|---|
| N10,K=10 | `dualview_support_v23` | role-balanced active | 92.62% | 80.00% | 87.71% | 67.14% | same as prior best |
| N10,K=5 | `dualview_support_v23` | role-balanced active | 91.56% | 77.33% | 86.00% | 64.00% | same as prior best |
| N10,K=10 | `none` | raw no assignment | 81.43% | 35.71% | 69.29% | 51.43% | much worse |

Hard-pair confusion from the strict v23 best rows:

| setting | `2-13 -> 11-10` | `11-10 -> 2-13` | observation |
|---|---:|---:|---|
| N10,K=10 | 12/70 | 14/70 | reciprocal hard-pair confusion |
| N10,K=5 | 16/75 | 16/75 | reciprocal hard-pair confusion |

Interpretation:

- Role-balanced assignment is not the root cause to remove; raw-noassign collapses old-class and new-class accuracy. The v23 role-balanced route is still the stronger classifier head.
- The main residual failure is a stable reciprocal hard pair, especially `2-13` versus `11-10`, under `leo_clear_weak`.
- Because the confusion is reciprocal, a simple one-way class bias is likely to trade one class for the other. The next viable qKNN innovation should be a support-derived pair discriminator or representation-side hard-pair separation that stores compressed pair parameters, not raw support samples and not per-K hand-tuned constants.

Current goal status: active, not achieved.


## 2026-07-06 qKNNV45场景类回退负诊断

目标：在不修改项目.md阶段二协议的前提下，继续检查qKNNV44后旧类域适应与新类增多时最低类坍塌问题。协议边界保持为K=5、K=10少量目标域LEO视图support，用于旧类目标域适应和seen-new注册识别；query不参与调参，unknown仍只做开放集评估。

本次实现：

| 文件 | 变更 |
|---|---|
| code/scripts/phase2_confusion_aware_qknn_probe.py | 新增显式scenario_class_fallback开关，默认关闭；只有调用方显式启用时，场景内缺失某类support才回退到该类全局target support。 |
| code/scripts/phase2_support_metric_qknn_probe.py | 新增stable_dualview_v45/dualview_support_v45诊断策略，继承V44；仅在support_min_k>=10时启用场景类回退，K5保持V44低K边界。 |
| code/tests/test_phase2_qknn_scenario_class_fallback.py | 覆盖默认严格场景遮蔽与显式回退行为。 |
| code/tests/test_phase2_support_metric_qknn_v45_policy.py | 覆盖V45继承V44、K10启用回退、K5禁用回退且保留labelprop。 |

本地验证：

| 命令 | 结果 |
|---|---|
| conda run --no-capture-output -n ssr-gpu python -m pytest -q code\tests\test_phase2_qknn_scenario_class_fallback.py code\tests\test_phase2_support_metric_qknn_v45_policy.py code\tests\test_phase2_support_metric_qknn_v43_policy.py code\tests\test_phase2_support_metric_qknn_v44_policy.py | PASS，7项通过。 |
| conda run --no-capture-output -n ssr-gpu python -m py_compile code\scripts\phase2_confusion_aware_qknn_probe.py code\scripts\phase2_support_metric_qknn_probe.py code\tests\test_phase2_qknn_scenario_class_fallback.py code\tests\test_phase2_support_metric_qknn_v45_policy.py code\tests\test_phase2_support_metric_qknn_v43_policy.py code\tests\test_phase2_support_metric_qknn_v44_policy.py | PASS。 |

V44/V45同口径结果：

| setting | policy | scenario_class_fallback | old_acc | min_old | seen_new_acc | min_new | verdict |
|---|---|---:|---:|---:|---:|---:|---|
| K5,N20 | stable_dualview_v44 | off | 92.00% | 80.00% | 79.80% | 69.33% | V44参考边界。 |
| K5,N20 | stable_dualview_v45 | false | 92.00% | 80.00% | 79.80% | 69.33% | 低K门控后等同V44，无提升。 |
| K10,N20 | stable_dualview_v44 | off | 91.90% | 82.86% | 84.71% | 74.29% | 当前N20更强参考。 |
| K10,N20 | stable_dualview_v45 | true | 93.10% | 82.86% | 84.07% | 72.86% | 旧类均值提升1.20pp，但seen-new均值下降0.64pp，最低新类下降1.43pp。 |

结论：

- V45验证了leo_clear_weak下局部类缺失确实会产生-1e9场景遮蔽症状，但把缺失类回退到全局target support并不能解决N20新类地板问题。
- K10启用回退后，旧类域适应均值提升，但新类均值和最低类同时回退；这与本轮目标 提升旧类同时抬高新类最低类不一致。
- K5禁用回退并保留labelprop后恢复到V44水平，说明低K场景类回退不是可推广路线。
- 因此V45只登记为诊断策略，不作为当前最强路线推广。当前N20可引用参考仍为V44：K10,N20的old_acc=91.90%、min_old=82.86%、seen_new_acc=84.71%、min_new=74.29%；K5,N20维持old_acc=92.00%、min_old=80.00%、seen_new_acc=79.80%、min_new=69.33%。
- 下一步优化应避开场景类回退，转向hard-pair局部重排序、支持集质量约束或压缩式pair discriminator，重点处理2-13、11-10、1-2等互混类，而不是扩大K或使用query标签。

版本状态：代码变更位于Git承载面E:\type10-7\github_publish\CVS-RFFI-repo；本报告位于根目录实验报告树，根目录不是Git仓库，因此报告本身未进入Git版本控制。未执行N607同步或远端启动。


## 2026-07-06 qKNNV46/V47旧类回退与新类地板诊断

目标：在qKNNV44当前N20强基线之上，继续优化旧类target receiver domain适应，同时避免新类增多时seen-new均值坍塌和最低类过低。协议保持为阶段二卫星端LEO叠加信道视图，K=5/K=10少量target support用于旧类域适应和新类注册识别；query标签只用于离线审计，不参与策略选择。

本次实现与诊断：

| 路线 | 变更 | 结论 |
|---|---|---|
| V46可靠support-LOO hard-pair扩展 | 继承V44；K10将support_loo_pair_rescue_top_pairs扩到10-12，最终关闭proto-neighbor扩展。 | 无收益。proto-neighbor候选会挤占真实LOO混淆对并轻微伤害seen-new均值；关闭后K10完全回到V44。 |
| V47old-only scenario fallback | 扩展_class_scores，允许fallback只覆盖旧类标签；V47在K10启用old-only fallback，K5保持V44严格场景遮蔽。 | 旧类均值恢复到V45水平，同时不再压低新类最低值；但seen-new均值仍比V44低0.21pp，最低新类未提升。 |
| V47support-bias三点诊断 | 在V47K10上扫描support_bias_weight=0/0.02/0.04，其他V44锚点不变。 | 最佳仍选择0；支持集类别偏置不能解决1-1/1-12/2-13硬混淆地板。 |

本地验证：

| 命令 | 结果 |
|---|---|
| conda run --no-capture-output -n ssr-gpu python -m pytest -q code\tests\test_phase2_qknn_old_only_scenario_fallback.py code\tests\test_phase2_qknn_scenario_class_fallback.py code\tests\test_phase2_support_metric_qknn_v47_policy.py code\tests\test_phase2_support_metric_qknn_v46_policy.py code\tests\test_phase2_support_metric_qknn_v45_policy.py code\tests\test_phase2_support_metric_qknn_v44_policy.py | PASS，9项通过。 |
| conda run --no-capture-output -n ssr-gpu python -m py_compile code\scripts\phase2_confusion_aware_qknn_probe.py code\scripts\phase2_support_metric_qknn_probe.py code\tests\test_phase2_qknn_old_only_scenario_fallback.py code\tests\test_phase2_qknn_scenario_class_fallback.py code\tests\test_phase2_support_metric_qknn_v47_policy.py code\tests\test_phase2_support_metric_qknn_v46_policy.py code\tests\test_phase2_support_metric_qknn_v45_policy.py code\tests\test_phase2_support_metric_qknn_v44_policy.py | PASS。 |

同口径结果：

| setting | policy | scenario_class_fallback | old_acc | min_old | seen_new_acc | min_new | verdict |
|---|---|---:|---:|---:|---:|---:|---|
| K5,N20 | stable_dualview_v44 | off | 92.00% | 80.00% | 79.80% | 69.33% | V44低K参考。 |
| K5,N20 | stable_dualview_v47 | false | 92.00% | 80.00% | 79.80% | 69.33% | 低K保持V44，未引入回退风险。 |
| K10,N20 | stable_dualview_v44 | off | 91.90% | 82.86% | 84.71% | 74.29% | 当前新类均值更强参考。 |
| K10,N20 | stable_dualview_v45 | true | 93.10% | 82.86% | 84.07% | 72.86% | 旧类提升但新类最低恶化，诊断-only。 |
| K10,N20 | stable_dualview_v46 | false | 91.90% | 82.86% | 84.71% | 74.29% | 与V44等价，无提升。 |
| K10,N20 | stable_dualview_v47 | old_only | 93.10% | 82.86% | 84.50% | 74.29% | 旧类均值+1.20pp，新类最低不恶化；seen-new均值较V44低0.21pp。 |
| K10,N20 | V47+support_bias scan | old_only | 93.10% | 82.86% | 84.50% | 74.29% | 最佳bias=0，无新增收益。 |

V47K10最低类行级审计：

| truth | correct/total | main wrong predictions |
|---|---:|---|
| 1-1 | 52/70 | 1-12:10，8-3:8 |
| 1-12 | 52/70 | 1-1:11，8-3:5，1-10:1，1-14:1 |
| 2-13 | 52/70 | 11-10:8，1-2:5，1-14:2，1-18:1，10-10:1，4-10:1 |
| 8-3 | 53/70 | 1-1:6，1-12:5，2-5:4，4-10:2 |

解释：

- V47是本轮有效的部分优化：把V45的全局scenario fallback收窄为old-only后，K10旧类域适应均值从91.90%提升到93.10%，且避免了V45把新类最低值从74.29%打到72.86%的问题。
- V47不是完整promote路线：它没有抬高最低新类，seen-new均值仍比V44低0.21pp，因此不能声称已解决 新类增多下最低类性能过低。
- V46和support-bias诊断共同说明，当前N20最低类不是简单的pair数量不足、proto-neighbor不足或类别偏置不足；硬簇1-1/1-12/8-3和2-13/11-10/1-2需要更局部的表示分离或压缩pair discriminator，而不是扩大K、使用query标签或打开全局场景回退。
- 当前可交付结论：若优先提升旧类且要求新类最低不恶化，V47优于V45；若优先最大seen-new均值，V44仍是更强参考。下一步应从局部pair discriminator或支持集选择质量入手，目标是把K10的三类52/70地板至少推到53-54/70，同时保持V47旧类收益。

版本状态：代码变更位于Git承载面E:\type10-7\github_publish\CVS-RFFI-repo；本报告位于根目录实验报告树，E:\type10-7根目录不是Git仓库，因此报告本身未进入Git版本控制。未执行N607同步或远端启动。


## 2026-07-06 qKNNV48/V49角色分块回退诊断

目标：继续在qKNNV47基础上提高旧类target receiver domain适应性能，同时避免N20新类增多时seen-new均值和最低类地板继续坍塌。项目协议保持不变：阶段二部署在卫星端，support/query均为叠加LEO星地信道后的目标域视图；K=5/K=10少量target support用于旧类域适应和seen-new注册识别；query标签只用于离线评估，不参与策略选择。

本次实现与诊断：

| 路线 | 变更 | 结论 |
|---|---|---|
| query-pair cluster诊断 | 在V47K10上扫描query_pair_cluster_top_pairs=4/8、query_pair_cluster_query_weight=0.15/0.30。 | 无收益。最佳仍为V47同等指标，较强权重会把最低新类压到72.86%。 |
| V48support-only pair-linear增强 | 继承V47old-only fallback；K10将support_loo_pair_linear_weight提到0.02、alpha=0.2、clip=1.5，scope=new。 | 负诊断。旧类保持93.10%，但seen-new均值降至84.43%，最低新类降至72.86%。 |
| V49old-role-only fallback | 继承V47；旧类查询/旧类标签分块保留old-only scenario fallback，新类查询/新类标签分块恢复严格无fallback分数。 | 当前本地最强折中。K10保持V47旧类收益，同时把seen-new均值恢复到V44的84.71%，最低新类不恶化但仍未抬升。 |

本地验证：

| 命令 | 结果 |
|---|---|
| conda run --no-capture-output -n ssr-gpu python -m pytest -q code\tests\test_phase2_support_metric_qknn_v49_policy.py code\tests\test_phase2_support_metric_qknn_v48_policy.py code\tests\test_phase2_support_metric_qknn_v47_policy.py code\tests\test_phase2_support_metric_qknn_v46_policy.py code\tests\test_phase2_support_metric_qknn_v45_policy.py code\tests\test_phase2_support_metric_qknn_v44_policy.py code\tests\test_phase2_qknn_old_only_scenario_fallback.py code\tests\test_phase2_qknn_scenario_class_fallback.py | PASS，12项通过。 |
| conda run --no-capture-output -n ssr-gpu python -m py_compile code\scripts\phase2_confusion_aware_qknn_probe.py code\scripts\phase2_support_metric_qknn_probe.py code\tests\test_phase2_support_metric_qknn_v49_policy.py code\tests\test_phase2_support_metric_qknn_v48_policy.py code\tests\test_phase2_support_metric_qknn_v47_policy.py code\tests\test_phase2_support_metric_qknn_v46_policy.py code\tests\test_phase2_support_metric_qknn_v45_policy.py code\tests\test_phase2_support_metric_qknn_v44_policy.py code\tests\test_phase2_qknn_old_only_scenario_fallback.py code\tests\test_phase2_qknn_scenario_class_fallback.py | PASS。 |

同口径结果：

| setting | policy | scenario_class_fallback | old_acc | min_old | seen_new_acc | min_new | verdict |
|---|---|---:|---:|---:|---:|---:|---|
| K5,N20 | stable_dualview_v44 | off | 92.00% | 80.00% | 79.80% | 69.33% | V44低K参考。 |
| K5,N20 | stable_dualview_v47 | false | 92.00% | 80.00% | 79.80% | 69.33% | 低K保持V44。 |
| K5,N20 | stable_dualview_v49 | false | 92.00% | 80.00% | 79.80% | 69.33% | 低K保持V44/V47。 |
| K10,N20 | stable_dualview_v44 | off | 91.90% | 82.86% | 84.71% | 74.29% | 新类均值参考。 |
| K10,N20 | stable_dualview_v47 | old_only | 93.10% | 82.86% | 84.50% | 74.29% | 旧类提升但seen-new均值低于V44。 |
| K10,N20 | stable_dualview_v48 | old_only | 93.10% | 82.86% | 84.43% | 72.86% | support-only pair-linear增强伤害最低新类，诊断-only。 |
| K10,N20 | stable_dualview_v49 | old_role_only | 93.10% | 82.86% | 84.71% | 74.29% | 当前本地最强折中：旧类收益与V44新类均值兼容。 |

V49K10最低类行级审计：

| truth | correct/total | class_acc |
|---|---:|---:|
| 1-1 | 52/70 | 74.29% |
| 1-12 | 52/70 | 74.29% |
| 2-13 | 52/70 | 74.29% |
| 11-10 | 53/70 | 75.71% |
| 8-3 | 54/70 | 77.14% |
| 1-14 | 55/70 | 78.57% |
| 1-16 | 56/70 | 80.00% |
| 1-2 | 56/70 | 80.00% |

Artifacts：

| artifact | 用途 |
|---|---|
| E:\type10-7\automation_reports\CV-SincNet\phase2_qknn_hardpair_n20_20260706\artifacts\v48_query_pair_diag_20260706 | query-pair cluster负诊断。 |
| E:\type10-7\automation_reports\CV-SincNet\phase2_qknn_hardpair_n20_20260706\artifacts\v48_pair_linear_policy_20260706\v48_k10_n20.json | V48K10负诊断结果。 |
| E:\type10-7\automation_reports\CV-SincNet\phase2_qknn_hardpair_n20_20260706\artifacts\v49_role_split_fallback_20260706\v49_k5_n20.json | V49K5本地评估结果。 |
| E:\type10-7\automation_reports\CV-SincNet\phase2_qknn_hardpair_n20_20260706\artifacts\v49_role_split_fallback_20260706\v49_k10_n20.json | V49K10本地评估结果。 |
| E:\type10-7\automation_reports\CV-SincNet\phase2_qknn_hardpair_n20_20260706\artifacts\v49_role_split_fallback_20260706\v49_k10_n20_predictions.csv | V49K10逐query预测审计表。 |

解释：

- V49不是用query身份标签调参；它复用当前role-balanced评估器已经存在的旧类query数/新类query数分块假设，只改变fallback分数在哪个角色分块生效。旧类查询到旧类标签沿用V47old-only fallback，新类查询到新类标签恢复V44严格场景遮蔽，因此避免V47对seen-new均值的0.21pp损失。
- V49可以作为当前本地优先候选：K10,N20相对V44旧类均值提升1.20pp，同时seen-new均值和最低新类均保持V44水平；K5保持V44低K边界。
- V49仍未完成总目标：最低新类仍停在74.29%，三类1-1、1-12、2-13仍为52/70地板。下一步不能声称已解决新类增多下最低类过低，只能说已消除了V47旧类提升与新类均值之间的主要折中。
- 后续优化方向应集中在1-1/1-12/8-3和2-13/11-10/1-2局部压缩判别，而不是提高K、使用query标签、扩大场景回退或把support-only pair-linear继续加重。

版本状态：代码变更位于Git承载面E:\type10-7\github_publish\CVS-RFFI-repo；本报告位于根目录实验报告树，E:\type10-7根目录不是Git仓库，因此报告本身未进入Git版本控制。未执行N607同步、SCP或远端启动。


## 2026-07-06 qKNNV50/V51 top2门控与分配器负诊断

目标：继续从V49的K10,N20三类52/70地板出发，尝试只作用于局部hard-pair的压缩门控，避免重新引入V48那类全局pair-linear外溢。协议边界不变：target-old与seen-new support/query均来自目标接收机域的LEO视图；不增加K，不使用query标签调参，不把clean view作为部署证据。

根因审计：

| 现象 | 证据 | 解释 |
|---|---|---|
| `1-1`地板难救 | V49中`1-1`错分18/70，raw top1为`1-1`的错分样本为0/18。 | 不是配额分配挤掉正确top1，而是局部表示本身把`1-1`压在`1-12`/`8-3`之后。 |
| `1-12`/`2-13`有少量配额挤压 | `1-12`错分18/70中raw top1为真类5个；`2-13`错分18/70中raw top1为真类2个。 | 有少量可由分配或分数微调救回的样本，但不足以单独解释全部地板。 |
| `8-3`/`11-10`更受配额影响 | `8-3`错分16/70中raw top1为真类9个；`11-10`错分17/70中raw top1为真类7个。 | 改分配器可能救回这些类，但会转移损失到其它新类。 |

本次实现与诊断：

| 路线 | 机制 | K10,N20结果 | verdict |
|---|---|---|---|
| V50 | 继承V49old-role-only fallback；K>=10,Nnew>=14时启用低强度top2 hard-pair gate，`query_pair_weight=0`。 | old_acc 93.10%，min_old 82.86%，seen_new_acc 84.71%，min_new 74.29%。 | 与V49等价，未撬动52/70地板。 |
| V51 | 继承V50；允许极小无标签query top-pair候选注入，`query_pair_weight<=0.012`，候选扩到6对。 | old_acc 93.10%，min_old 82.86%，seen_new_acc 84.71%，min_new 74.29%。 | 候选增加但最终指标仍与V49等价。 |
| V49+fast_role_balanced_assignment | 保持V49分数，改用贪心quota修复近似分配器。 | old_acc 91.90%，min_old 82.86%，seen_new_acc 83.29%，min_new 65.71%。 | 明确负诊断，`2-13`坍塌到46/70。 |
| V28top2 gate参考 | 旧V28默认top2 gate同seed参考。 | old_acc 90.95%，min_old 80.00%，seen_new_acc 84.21%，min_new 71.43%。 | 旧top2强度不适合当前V49基线。 |

同seed诊断表：

| setting | policy | top2_pairs | fast_assign | old_acc | min_old | seen_new_acc | min_new |
|---|---|---|---:|---:|---:|---:|---:|
| K10,N20,seed421057 | stable_dualview_v49 | none | false | 93.10% | 82.86% | 84.71% | 74.29% |
| K10,N20,seed421057 | stable_dualview_v50 | `1-1<->1-12`;`1-19<->1-2`;`1-10<->1-14`;`1-15<->19-3` | false | 93.10% | 82.86% | 84.71% | 74.29% |
| K10,N20,seed421057 | stable_dualview_v51 | V50 pairs plus`1-2<->2-13`;`1-18<->11-10` | false | 93.10% | 82.86% | 84.71% | 74.29% |
| K10,N20,seed421057 | stable_dualview_v49 | none | true | 91.90% | 82.86% | 83.29% | 65.71% |
| K10,N20,seed421057 | stable_dualview_v28 | `1-1<->1-12`;`1-15<->19-3`;`1-19<->1-2`;`1-16<->18-5` | false | 90.95% | 80.00% | 84.21% | 71.43% |

V51最低类行级结果：

| truth | correct/total | class_acc |
|---|---:|---:|
| 1-1 | 52/70 | 74.29% |
| 1-12 | 52/70 | 74.29% |
| 2-13 | 52/70 | 74.29% |
| 11-10 | 53/70 | 75.71% |
| 8-3 | 54/70 | 77.14% |
| 1-14 | 55/70 | 78.57% |
| 1-16 | 56/70 | 80.00% |
| 1-2 | 56/70 | 80.00% |

Artifacts：

| artifact | 用途 |
|---|---|
| E:\type10-7\automation_reports\CV-SincNet\phase2_qknn_hardpair_n20_20260706\artifacts\v50_top2_gate_diag_20260706\v28_k10_n20_seed421057.json | 旧V28top2 gate同seed负参考。 |
| E:\type10-7\automation_reports\CV-SincNet\phase2_qknn_hardpair_n20_20260706\artifacts\v50_top2_gate_diag_20260706\v50_k10_n20_seed421057.json | V50保守top2 gate诊断。 |
| E:\type10-7\automation_reports\CV-SincNet\phase2_qknn_hardpair_n20_20260706\artifacts\v50_top2_gate_diag_20260706\v51_k10_n20_seed421057.json | V51微弱query top-pair候选诊断。 |
| E:\type10-7\automation_reports\CV-SincNet\phase2_qknn_hardpair_n20_20260706\artifacts\v50_top2_gate_diag_20260706\v49_fastassign_k10_n20_seed421057.json | fast role-balanced assignment负诊断。 |

本地验证：

| 命令 | 结果 |
|---|---|
| conda run --no-capture-output -n ssr-gpu python -m pytest -q code\tests\test_phase2_support_metric_qknn_v51_policy.py code\tests\test_phase2_support_metric_qknn_v50_policy.py code\tests\test_phase2_support_metric_qknn_v49_policy.py code\tests\test_phase2_support_metric_qknn_v48_policy.py code\tests\test_phase2_support_metric_qknn_v47_policy.py code\tests\test_phase2_support_metric_qknn_v46_policy.py code\tests\test_phase2_support_metric_qknn_v45_policy.py code\tests\test_phase2_support_metric_qknn_v44_policy.py code\tests\test_phase2_qknn_old_only_scenario_fallback.py code\tests\test_phase2_qknn_scenario_class_fallback.py | PASS，14项通过。 |
| conda run --no-capture-output -n ssr-gpu python -m py_compile code\scripts\phase2_confusion_aware_qknn_probe.py code\scripts\phase2_support_metric_qknn_probe.py code\tests\test_phase2_support_metric_qknn_v51_policy.py code\tests\test_phase2_support_metric_qknn_v50_policy.py code\tests\test_phase2_support_metric_qknn_v49_policy.py code\tests\test_phase2_support_metric_qknn_v48_policy.py code\tests\test_phase2_support_metric_qknn_v47_policy.py code\tests\test_phase2_support_metric_qknn_v46_policy.py code\tests\test_phase2_support_metric_qknn_v45_policy.py code\tests\test_phase2_support_metric_qknn_v44_policy.py code\tests\test_phase2_qknn_old_only_scenario_fallback.py code\tests\test_phase2_qknn_scenario_class_fallback.py | PASS。 |

结论：

- V49仍是当前本地最强折中：旧类均值93.10%，seen-new均值84.71%，最低新类74.29%。
- V50/V51证明“再加top2 hard-pair门控”不足以突破52/70地板；即使加入极小无标签query top-pair候选，最终Hungarian分配仍不变。
- fast assignment证明“换成贪心quota修复”会把`2-13`从52/70打到46/70，不能作为地板救援路线。
- 下一步应转向表示侧或support选择侧的`1-1`局部可分性增强，例如针对`1-1`/`1-12`/`8-3`的压缩多类局部判别或support质量重采样；继续调top2门控、query-pair候选或分配器不应作为主线。

版本状态：代码变更位于Git承载面E:\type10-7\github_publish\CVS-RFFI-repo；本报告位于根目录实验报告树，E:\type10-7根目录不是Git仓库，因此报告本身未进入Git版本控制。未执行N607同步、SCP或远端启动。


## 2026-07-06 qKNNV52 query cluster负诊断

### 目标

在V49/V50/V51均未抬升K10,N20最低新类的基础上，诊断是否可以在不破坏V49旧类域适应收益的前提下，重新打开极小权重的new-scope query cluster，用少量目标域K-shot support与同批LEO叠加query结构缓解新类增多后的最低类坍塌。

### 代码与策略变更

| 文件 | 变更 |
| --- | --- |
| code/scripts/phase2_support_metric_qknn_probe.py | 新增stable_dualview_v52策略；继承V49的old_role_only role-split fallback、support LOO、source-target transport、neighbor contrast链；仅在min_support>=10且
ew_class_count>=14时启用极小权重new-scope query cluster；不启用top2 gate、transductive proto或dense cluster。 |
| code/tests/test_phase2_support_metric_qknn_v52_policy.py | 新增V52策略单测，约束K10/N20启用new-scope query cluster，K5/N20保持低K回退，且transductive/dense cluster保持关闭。 |

### 本地诊断结果

| 场景 | 策略 | query cluster | top2 gate | old | min_old | seen_new | min_new | 结论 |
| --- | --- | --- | --- | ---: | ---: | ---: | ---: | --- |
| K10,N20,seed421057 | stable_dualview_v49 | 关闭 | 关闭 | 93.10% | 82.86% | 84.71% | 74.29% | 当前本地最强基线。 |
| K10,N20,seed421057 | stable_dualview_v52保守门控 | 0行生效 | 关闭 | 93.10% | 82.86% | 84.71% | 74.29% | agreement门控过严，未形成有效干预。 |
| K10,N20,seed421057 | stable_dualview_v52主动cluster | 1400行new-query生效，20个临时原型 | 关闭 | 93.10% | 82.86% | 84.71% | 74.29% | 对最低新类无收益，不能替代V49。 |

V52主动cluster的同row细节：query_cluster_weight=0.008、query_cluster_support_weight=0.7、query_cluster_agreement_min=0.0、query_cluster_assigned_rows=1400、query_cluster_temp_proto_count=20、source_target_transport_rank_used=3、
eighbor_contrast_count=5、	op2_pair_gate_weight=0.0。最低新类仍为2-13=74.29%、1-1=74.29%、1-12=74.29%，未突破75%地板。

### 负证据解释

V52说明query batch临时聚类即使完全作用于new-query，也主要在相邻类之间重新分配少量正确率，无法解决1-1、1-12、2-13的底部类问题。结合前序V50/V51 top2 gate负诊断，可判定当前最低类瓶颈不是简单query聚类或top2邻接门控可修复，更像是支撑样本/表征几何下的类间局部不可分问题。后续优化应优先转向support选择质量、底部类风险感知的训练/嵌入重标定，或针对1-*族与2-13的类簇级特征分离，而不是继续增加batch-local query聚类权重。

### Artifact与验证

| 路径 | 说明 |
| --- | --- |
| E:\type10-7\automation_reports\CV-SincNet\phase2_qknn_hardpair_n20_20260706\artifacts\v52_query_cluster_diag_20260706\v52_k10_n20_seed421057_inherit.json | V52保守门控诊断，query cluster实际0行生效。 |
| E:\type10-7\automation_reports\CV-SincNet\phase2_qknn_hardpair_n20_20260706\artifacts\v52_query_cluster_diag_20260706\v52_k10_n20_seed421057_active.json | V52主动cluster诊断，query cluster作用1400行但未抬升最低新类。 |
| E:\type10-7\automation_reports\CV-SincNet\phase2_qknn_hardpair_n20_20260706\artifacts\v52_query_cluster_diag_20260706\accidental_top2_path\ | 首次PowerShell变量展开错误导致的误路径artifact，已移动归档；该结果含非预期top2外层集合，不作为V52结论依据。 |

| 验证命令 | 结果 |
| --- | --- |
| conda run --no-capture-output -n ssr-gpu python -m pytest -q code\tests\test_phase2_support_metric_qknn_v52_policy.py code\tests\test_phase2_support_metric_qknn_v51_policy.py code\tests\test_phase2_support_metric_qknn_v50_policy.py code\tests\test_phase2_support_metric_qknn_v49_policy.py code\tests\test_phase2_support_metric_qknn_v48_policy.py code\tests\test_phase2_support_metric_qknn_v47_policy.py code\tests\test_phase2_support_metric_qknn_v46_policy.py code\tests\test_phase2_support_metric_qknn_v45_policy.py code\tests\test_phase2_support_metric_qknn_v44_policy.py code\tests\test_phase2_qknn_old_only_scenario_fallback.py code\tests\test_phase2_qknn_scenario_class_fallback.py | PASS，15项通过。 |
| conda run --no-capture-output -n ssr-gpu python -m py_compile code\scripts\phase2_confusion_aware_qknn_probe.py code\scripts\phase2_support_metric_qknn_probe.py code\tests\test_phase2_support_metric_qknn_v52_policy.py code\tests\test_phase2_support_metric_qknn_v51_policy.py code\tests\test_phase2_support_metric_qknn_v50_policy.py code\tests\test_phase2_support_metric_qknn_v49_policy.py code\tests\test_phase2_support_metric_qknn_v48_policy.py code\tests\test_phase2_support_metric_qknn_v47_policy.py code\tests\test_phase2_support_metric_qknn_v46_policy.py code\tests\test_phase2_support_metric_qknn_v45_policy.py code\tests\test_phase2_support_metric_qknn_v44_policy.py code\tests\test_phase2_qknn_old_only_scenario_fallback.py code\tests\test_phase2_qknn_scenario_class_fallback.py | PASS。 |

未执行N607同步或启动；本轮为本地负诊断与代码/报告版本化。

## 2026-07-06 qKNN rawsketch/LEO-sketch边界审计与fft_logmag候选前端

### 目标

继续围绕qKNNV42/V49线优化旧类域适应和N20多新类最低类地板。根据前序证据，单纯score层top2 gate、role-balanced assignment、query cluster、Pool80 support选择、Mahalanobis小网格都未突破`K10,N20`最低新类74.29%地板；本轮重点审计clean rawsketch上界与正式LEO视图的协议边界，并准备一个协议安全的LEO后压缩描述符候选。

### 协议边界审计

| artifact | manifest结论 | 结论 |
| --- | --- | --- |
| `artifacts\features_hardpair_HP08L5_n20_rawsketch96.npz` | `method=dc_removed_l2_raw_iq_random_projection_tanh_l2`，没有`applies_star_ground_channel=true`，没有`uses_target_clean=false`字段 | clean/control上界，只能说明原始IQ存在可分性，不能作为卫星部署正式证据。 |
| `artifacts\features_hardpair_HP08L5_n20_leosketch96.npz` | `channel_view=satellite/LEO`，`applies_star_ground_channel=true`，`uses_target_clean=false`，`leo_tta_views=5`，`star_ground_channel_impl=simplified_leo_residual` | 正式LEO压缩视图；前序support-LOO仅K5=10.77%、K10=17.69%，被V42门控正确拒绝。 |

因此，`n20_k10_rawsketch96_primary_v37_20260706m.json`中的99%级结果是clean rawsketch控制上界，不是当前项目协议下的部署成功。正式解释仍以LEO叠加后的target-domain support/query为准。

### 本轮代码候选

在Git承载面`E:\type10-7\github_publish\CVS-RFFI-repo`新增`phase2_raw_iq_sketch_export.py --sketch_method fft_logmag`候选：

| 文件 | 变更 |
| --- | --- |
| `code\scripts\phase2_raw_iq_sketch_export.py` | 保留默认`random_projection`行为；新增`fft_logmag`描述符：IQ转复数、去均值、RMS归一化、FFT幅度谱、`log1p`压缩、插值到`sketch_dim`、L2归一化。LEO模式仍先调用`apply_sat_channel_for_scenario`，再压缩，不读取clean target作为正式视图。 |
| `code\tests\test_phase2_raw_iq_sketch_export.py` | 新增单测，验证`fft_logmag`对全局相位旋转不敏感且输出L2归一化；验证默认`random_projection`仍需要projection并保持原输出形状。 |

该候选的用途是解决当前LEO-sketch随机投影在星地信道后support-LOO过低的问题。它不改变`K=5,K=10`少量目标域support、新类注册、target receiver LEO视图或`stored_raw_support_count=0`边界。性能是否提升仍需在N607或具备`Dataset_WigSig/ManySig.pkl`、`ManyTx.pkl`和原始feature run目录的环境中重新导出并评估，当前本机没有这些数据文件，不能给出正式指标。

### 本地验证

| 命令 | 结果 |
| --- | --- |
| `conda run -n ssr-gpu python -m pytest code\tests\test_phase2_raw_iq_sketch_export.py -q` | PASS，2项通过。 |
| `conda run -n ssr-gpu python -m py_compile code\scripts\phase2_raw_iq_sketch_export.py code\tests\test_phase2_raw_iq_sketch_export.py` | PASS。 |

一次并发`conda run`触发本机已知`__conda_tmp_*.txt`临时锁噪声；已按项目规则串行重跑并通过。未执行N607预检、SCP、远端导出或远端启动。`E:\type10-7`根目录不是Git仓库，本报告仍是根目录实验报告树文件；代码与单测变更已进入Git承载面待提交。

## 2026-07-06 N607 v53 LEO+fft_logmag96导出与qKNN评估启动记录

### 启动前边界

本节继续同一qKNNV42/V49优化目标，测试`fft_logmag96`是否能在LEO叠加后提供比随机投影LEO-sketch更可靠的压缩辅助视图。该实验仍是Stage2-C no-unknown矩阵：target-old与target-new support/query均来自目标接收机域，`K=5`和`K=10`，`target_unknown` role在该feature包中作为seen-new候选集合使用；本轮不评估unknown rejection，不声明部署成功。

| 项目 | 记录 |
| --- | --- |
| run_id | `phase2_qknn_fftlogmag_v53_20260706` |
| operator | Codex |
| objective | 导出LEO后`fft_logmag96`压缩描述符，并在N20、K5/K10上对比`stable_dualview_v42`与`stable_dualview_v49`。 |
| comparison target | V42 gated LEO-sketch：K5 old 92.00%、min_old 80.00%、seen_new 79.80%、min_new 69.33%；K10 old 91.90%、min_old 82.86%、seen_new 84.64%、min_new 72.86%。当前更强参考为V49 K10 old 93.10%、min_old 82.86%、seen_new 84.71%、min_new 74.29%。 |
| protocol | `R_t` target receiver domain，target-old `Y_old={14-10,14-7,20-15,20-19,6-15,8-20}`，target-new N20为`10-10,11-10,18-5,19-3,2-13,2-5,3-8,4-10,8-18,8-3,1-1,1-10,1-11,1-12,1-14,1-15,1-16,1-18,1-19,1-2`，support/query均为satellite/LEO target view。 |
| local Git carrier | `E:\type10-7\github_publish\CVS-RFFI-repo`，commit `cd1a2a4`新增`fft_logmag`导出候选，commit `1c2b9e4`新增v53 launcher。 |
| root Git status | `E:\type10-7`不是Git仓库；本报告位于根目录实验报告树，未版本化。 |

### 本地与远端验证

| 检查 | 结果 |
| --- | --- |
| N607 preflight | direct `N607`通过；server time 2026-07-06 20:19 CST；project root可见；8张RTX3090，预检时显存约10MiB。 |
| remote process/GPU | 未见项目python训练/评估进程；`nvidia-smi --query-compute-apps`为空。 |
| synced files | `code/scripts/phase2_raw_iq_sketch_export.py`、`code/tests/test_phase2_raw_iq_sketch_export.py`、`code/scripts/launch_phase2_qknn_fftlogmag_v53_20260706.sh`。 |
| synced hashes | exporter `5e377212a6264fc342c379999557c330ffec90e4dbfbc0985b1abbdd54e3d05e`；test `e111b80dec3991becf7bbd0cdc88a4b482867d9fdfc8902a4149690a26b597fc`；launcher `3af0fdc0fb77017301103adfc0fc7470bc84f180e62a6c5a45870b3e239548a1`。 |
| local checks | `bash -n code/scripts/launch_phase2_qknn_fftlogmag_v53_20260706.sh` PASS；此前`pytest`和`py_compile` PASS。 |
| remote checks | `CVS-RFFI`环境`py_compile` PASS；`python code/tests/test_phase2_raw_iq_sketch_export.py` PASS，2项通过；launcher `bash -n` PASS；`--dry-run`展开命令正确。 |

### 远端命令

工作目录：`/home/szu2070436088/2510044040/CV-SincNet`

启动命令：

```bash
nohup bash code/scripts/launch_phase2_qknn_fftlogmag_v53_20260706.sh > logs/phase2_qknn_fftlogmag_v53_20260706/run.log 2>&1 &
```

关键输出：

| 输出 | 路径 |
| --- | --- |
| log | `/home/szu2070436088/2510044040/CV-SincNet/logs/phase2_qknn_fftlogmag_v53_20260706/run.log` |
| aux feature | `/home/szu2070436088/2510044040/CV-SincNet/automation_reports/CV-SincNet/phase2_qknn_hardpair_n20_20260706/artifacts/v53_fftlogmag_20260706/features_hardpair_HP08L5_n20_leo_fftlogmag96.npz` |
| K5 JSON/CSV/predictions | `/home/szu2070436088/2510044040/CV-SincNet/automation_reports/CV-SincNet/phase2_qknn_hardpair_n20_20260706/artifacts/v53_fftlogmag_20260706/n20_k5_v53_fftlogmag_v42_v49_20260706.*` |
| K10 JSON/CSV/predictions | `/home/szu2070436088/2510044040/CV-SincNet/automation_reports/CV-SincNet/phase2_qknn_hardpair_n20_20260706/artifacts/v53_fftlogmag_20260706/n20_k10_v53_fftlogmag_v42_v49_20260706.*` |

成功判据：如果`aux_support_aux_loo_acc`和`aux_support_aux_loo_min_acc`显著高于旧LEO-sketch，且最终同row old/min_old/seen_new/min_new不低于V49并抬升K10最低新类超过74.29%，则进入后续策略晋升候选；否则记录为LEO后相位不敏感压缩描述符负诊断。

### 完成状态与结果

最终可比运行完成于N607 2026-07-06 20:34 CST。前两次评估启动暴露远端脚本版本漂移：第一次`phase2_support_metric_qknn_probe.py`不支持`stable_dualview_v49`，第二次`phase2_confusion_aware_qknn_probe.py`不支持`scenario_class_fallback`；均已按本地Git承载面同步修复并保留失败日志。第三次运行因launcher漏传历史基线的`--scenario_aware --balanced_assignment`而不可比，日志归档为`run_attempt3_wrong_parity_flags.log`；最终第四次运行补齐这两个开关后产出以下正式对比。

| row | K | policy | old | min_old | seen_new | min_new | aux LOO mean/min | effective aux weight | verdict |
| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| v53 fft_logmag96 | 5 | stable_dualview_v42 | 91.90% | 80.00% | 80.00% | 70.00% | 80.00%/40.00% | 0.000 | 比旧V42 K5 seen_new +0.20pp、min_new +0.67pp，但aux仍被门控为0；K5未解决地板。 |
| v53 fft_logmag96 | 5 | stable_dualview_v49 | 91.90% | 80.00% | 80.00% | 70.00% | 80.00%/40.00% | 0.000 | 与V42同分；K5仍只能作为轻微改善。 |
| v53 fft_logmag96 | 10 | stable_dualview_v49 | 93.57% | 84.29% | 87.64% | 78.57% | 93.85%/80.00% | 0.219 | 当前本地/远端最强候选；相对V49基线四项同时提升，最低新类越过75%地板。 |
| v53 fft_logmag96 | 10 | stable_dualview_v42 | 92.86% | 82.86% | 86.79% | 72.86% | 93.85%/70.00% | 0.219 | seen_new均值提升但最低新类未超过V49最佳，不作为晋升主线。 |

与前序最佳的同row比较：

| 对比 | old Δ | min_old Δ | seen_new Δ | min_new Δ | 结论 |
| --- | ---: | ---: | ---: | ---: | --- |
| K5 v53(V49) vs V49基线 | -0.10pp | +0.00pp | +0.20pp | +0.67pp | 小幅改善但最低新类仍70.00%，不满足多新类地板目标。 |
| K10 v53(V49) vs V49基线 | +0.47pp | +1.43pp | +2.93pp | +4.29pp | 达到本轮目标方向：旧类域适应和多新类最低类同时改善。 |
| K10 v53(V49) vs V42 gated LEO-sketch | +1.67pp | +1.43pp | +3.00pp | +5.71pp | 证明LEO后`fft_logmag96`比旧随机投影LEO-sketch更适合作为压缩辅助视图。 |

K10最低新类明细来自`n20_k10_v53_fftlogmag_v42_v49_20260706_predictions.csv`：

| truth | correct/total | class_acc |
| --- | ---: | ---: |
| 2-13 | 55/70 | 78.57% |
| 1-1 | 56/70 | 80.00% |
| 1-12 | 56/70 | 80.00% |
| 8-3 | 56/70 | 80.00% |
| 1-2 | 57/70 | 81.43% |
| 1-14 | 58/70 | 82.86% |
| 11-10 | 58/70 | 82.86% |
| 1-16 | 60/70 | 85.71% |

Artifacts已拉回本地：

| artifact | 说明 |
| --- | --- |
| `E:\type10-7\automation_reports\CV-SincNet\phase2_qknn_hardpair_n20_20260706\artifacts\v53_fftlogmag_20260706\features_hardpair_HP08L5_n20_leo_fftlogmag96.npz` | LEO叠加后`fft_logmag96`压缩特征，manifest含`applies_star_ground_channel=true`、`uses_target_clean=false`、`stored_raw_support_count=0`。 |
| `...\n20_k5_v53_fftlogmag_v42_v49_20260706.json/csv/predictions.csv` | K5正式可比结果。 |
| `...\n20_k10_v53_fftlogmag_v42_v49_20260706.json/csv/predictions.csv` | K10正式可比结果，当前主晋升候选。 |
| `E:\type10-7\automation_reports\CV-SincNet\phase2_qknn_hardpair_n20_20260706\logs\phase2_qknn_fftlogmag_v53_20260706\` | 远端run.log与三次失败尝试日志。 |

后续建议：将`stable_dualview_v49 + fft_logmag96`提升为V53候选策略，但仍需补一个独立seed或N20 seed-scan确认K10提升不是单seed偶然；K5仍需单独优化，因为当前最低新类70.00%未越过75%地板。

## 2026-07-06本地V53策略化补丁与K5诊断

本节记录本地Git承载面`E:\type10-7\github_publish\CVS-RFFI-repo`上的后续优化。`E:\type10-7`根目录不是Git仓库，本报告仍位于未版本化实验报告树；代码改动进入Git承载面后再同步N607。

### 修改目的

目标是把上一轮`fft_logmag96`辅助视图从“V42/V49对照”整理为一个正式`stable_dualview_v53`策略，同时继续追查K5多新类最低类地板。协议不变：K5/K10 target-old与target-new support/query均来自同一`R_t`目标接收机域，并使用LEO叠加后的目标域样本；未使用query真值调参作为可部署策略。

### 本地诊断结果

| diagnostic | K | policy/setting | old | min_old | seen_new | min_new | 结论 |
| --- | ---: | --- | ---: | ---: | ---: | ---: | --- |
| `k5_aux_weight_probe_20260706` | 5 | `stable_dualview_v40`，`aux_weight=0.22` | 91.90% | 80.00% | 83.07% | 71.43% | 放开低K aux可提升新类均值，但最低类仍未过75%。 |
| `k5_pair_refine_probe_small_20260706` | 5 | query-pair cluster top2/4 | 91.90% | 80.00% | 82.14%-83.07% | 67.14%-71.43% | query-pair cluster没有改善地板，开启后伤害`1-1/1-12`。 |
| `k5_existing_policy_probe_20260706` | 5 | best existing policy=`stable_dualview_v36` | 91.90% | 80.00% | 83.21% | 71.43% | 现有可部署support-only策略中V36最佳，但仍未解决最低类。 |
| `k10_v36_compare_20260706` | 10 | `stable_dualview_v36` | 92.86% | 82.86% | 86.93% | 74.29% | K10用V36会跌破75%新类地板，不可替代V49。 |
| `k10_v36_compare_20260706` | 10 | `stable_dualview_v49` | 93.57% | 84.29% | 87.64% | 78.57% | K10应保持V49稳态路线。 |

K5最低类瓶颈集中在少数新类边界：`1-2=50/70`、`2-13=50/70`、`19-3=52/70`，主要混淆包括`19-3->1-15`、`1-15->19-3`、`2-13->1-2`。其中`19-3/1-15`没有被support-LOO稳定暴露；因此不能把query混淆直接写成部署期偏置，只能作为诊断线索。

### 代码变更

| file | change |
| --- | --- |
| `code/scripts/phase2_support_metric_qknn_probe.py` | 新增`stable_dualview_v53`策略入口；K<10解析为`stable_dualview_v36`以保留低K轻量注册增益，K>=10解析为`stable_dualview_v49`以保留K10稳态fallback、aux门控和旧类transport。结果行保留`adaptive_qknn_requested_policy=stable_dualview_v53`与`adaptive_qknn_effective_policy`。 |
| `code/scripts/launch_phase2_qknn_fftlogmag_v53_20260706.sh` | launcher输出改为`n20_${tag}_v53_fftlogmag_policy_20260706.*`，正式使用`--adaptive_qknn_policy_grid stable_dualview_v53`。 |
| `code/tests/test_phase2_support_metric_qknn_v53_policy.py` | 覆盖V53低K->V36、高K->V49的策略解析。 |

### 本地验证

| command | result |
| --- | --- |
| `conda run -n ssr-gpu python -m py_compile code/scripts/phase2_support_metric_qknn_probe.py code/tests/test_phase2_support_metric_qknn_v53_policy.py` | PASS |
| `bash -n code/scripts/launch_phase2_qknn_fftlogmag_v53_20260706.sh` | PASS |
| `conda run -n ssr-gpu python code/tests/test_phase2_support_metric_qknn_v49_policy.py` | PASS，2 tests |
| `conda run -n ssr-gpu python code/tests/test_phase2_support_metric_qknn_v52_policy.py` | PASS，1 test |
| `conda run -n ssr-gpu python code/tests/test_phase2_support_metric_qknn_v53_policy.py` | PASS，1 test |

正式V53本地复现：

| artifact | K | requested policy | effective policy | old | min_old | seen_new | min_new | verdict |
| --- | ---: | --- | --- | ---: | ---: | ---: | ---: | --- |
| `local_v53_policy_verify_20260706/k5_v53_policy_verify.json` | 5 | `stable_dualview_v53` | `stable_dualview_v36` | 91.90% | 80.00% | 83.21% | 71.43% | 均值改善，但最低类仍未达75%；不能写成K5地板解决。 |
| `local_v53_policy_verify_20260706/k10_v53_policy_verify.json` | 10 | `stable_dualview_v53` | `stable_dualview_v49` | 93.57% | 84.29% | 87.64% | 78.57% | 维持上一轮K10最强结果，通过joint target。 |

### Git与N607同步

| item | value |
| --- | --- |
| local Git carrier | `E:\type10-7\github_publish\CVS-RFFI-repo` |
| commit | `7a2bb4e Add adaptive qKNN v53 policy` |
| git status after commit | 仅剩既有未跟踪`local_artifacts/phase2_adv3b02_proxy_mined_20260704/`与`local_artifacts/phase2_adv3b02_smec_ci_20260704/`。 |
| N607 preflight | direct `N607`通过；server time 2026-07-06 20:56:55 CST；project root可见；8张RTX3090空闲显存约10MiB。 |
| project process check | 未见当前用户CV-SincNet实验进程；仅系统`unattended-upgrade`和他人VSCode服务进程。 |
| synced files | `code/scripts/phase2_support_metric_qknn_probe.py`、`code/scripts/launch_phase2_qknn_fftlogmag_v53_20260706.sh`、`code/tests/test_phase2_support_metric_qknn_v53_policy.py`。 |
| synced sha256 | `a43955823ff5f2b58d1c360a4b9c1c1bec343e17a274c89de151f05f2915382c`、`222b456032a3184ba0175e08ee7c1d6bf81adc51b1ec65fac437985d2be78695`、`e29897397a8c410459aac17632b854c84b72452ca9a2a7d3c1c8b99558676f5e`，本地与N607一致。 |
| remote verification | N607 `py_compile` PASS；`python code/tests/test_phase2_support_metric_qknn_v53_policy.py` PASS；launcher `bash -n` PASS；`--dry-run`展开K5/K10正式`stable_dualview_v53`命令和`*_v53_fftlogmag_policy_20260706.*`输出路径。 |
| remote launch | 本次未重新启动N607实验。 |
| SSH cleanup | 本地无残留`ssh.exe`，无到`172.31.111.215:22`的ESTABLISHED连接。 |

## 2026-07-06本地V54低K参数化优化

本节继续在Git承载面`E:\type10-7\github_publish\CVS-RFFI-repo`优化`stable_dualview_v53`之后的K5问题。`E:\type10-7`根目录不是Git仓库；本轮涉及的探针脚本只存在于Git承载面`code/scripts/phase2_support_metric_qknn_probe.py`，根目录`E:\type10-7\code\scripts\phase2_support_metric_qknn_probe.py`不存在，因此不存在未镜像的根目录脚本改动。报告文件仍位于未版本化实验报告树。

### 诊断结论

协议边界不变：K5/K10的target-old与target-new support/query均来自目标接收机域，目标域样本为LEO叠加后的星地信道视图；本节没有把query混淆真值写入部署策略。query真值只用于诊断已完成候选的失败类型。

| probe | K | 设置 | old | min_old | seen_new | min_new | 结论 |
| --- | ---: | --- | ---: | ---: | ---: | ---: | --- |
| `k5_missed_policy_probe_20260706` | 5 | `stable_dualview_v37/v38/v39/v45-v48` | 91.90% | 80.00% | 80.00%-83.07% | 70.00%-71.43% | 已有neighbor contrast类策略未超过V36；部分策略因aux权重为0降低均值。 |
| `local_v54_policy_verify_20260706/k5_v54_policy_verify.json`初版 | 5 | 宽neighbor contrast V54 | 91.90% | 80.00% | 83.07% | 71.43% | 扩大support邻域覆盖无实质收益，撤回该实现。 |
| `k5_v36_rescue_neighbor_probe_20260706` | 5 | V36参数复刻+rescue proto邻居 | 91.90% | 80.00% | 82.57%-83.00% | 70.00%-71.43% | 可识别`2-13->1-2`，但只能把`2-13`局部抬到51/70，`1-2`仍50/70，整体不合格。 |
| `k5_v36_bias_probe_20260706` | 5 | V36参数复刻+support bias | 91.90% | 80.00% | 83.21% | 71.43% | bias不改变最终地板，最佳等价于bias=0。 |
| `k5_v36_baseparam_probe_20260706` | 5 | V36参数复刻，扫`topm/proto_mix/aux_weight` | 93.10% | 81.43% | 83.93% | 71.43% | 找到可部署均值和旧类域适应收益：`topm=2,proto_mix=0.45,aux_weight=0.26`；最低新类仍未解决。 |

K5最终V54行的低类上下文：

| class | correct/total | class_acc | 说明 |
| --- | ---: | ---: | --- |
| `1-2` | 50/70 | 71.43% | 仍是最低类，V54未抬升。 |
| `19-3` | 52/70 | 74.29% | 仍低于75%，但比V36瓶颈之一保持不退化。 |
| `2-13` | 52/70 | 74.29% | 相比V36的50/70提升2个query，但地板仍由`1-2`限制。 |
| `1-15` | 53/70 | 75.71% | 与`19-3`互混仍存在，但未成为最低类。 |

### 代码变更

| file | change |
| --- | --- |
| `code/scripts/phase2_support_metric_qknn_probe.py` | 新增`stable_dualview_v54`策略入口。K<10保持V36低K注册组件，并覆盖`topm=2`、`proto_mix=0.45`、有辅视图时`aux_score_weight=0.26`；K>=10仍映射为`stable_dualview_v49`，保留高K旧类fallback、source-target transport与aux门控。 |
| `code/tests/test_phase2_support_metric_qknn_v54_policy.py` | 覆盖V54低K自有策略参数与高K映射V49。 |
| `code/scripts/launch_phase2_qknn_fftlogmag_v54_20260706.sh` | 新增V54 launcher，输出到`artifacts/v54_fftlogmag_20260706`，请求`--adaptive_qknn_policy_grid stable_dualview_v54`，保留K5 seed421038与K10 seed421057。 |

### 本地验证

| command | result |
| --- | --- |
| `conda run -n ssr-gpu python -m py_compile code/scripts/phase2_support_metric_qknn_probe.py code/tests/test_phase2_support_metric_qknn_v54_policy.py` | PASS |
| `conda run -n ssr-gpu python code/tests/test_phase2_support_metric_qknn_v54_policy.py` | PASS，1 test |
| `bash -n code/scripts/launch_phase2_qknn_fftlogmag_v54_20260706.sh` | PASS |
| `local_v54_policy_verify_20260706/k5_v54_policy_verify.json` | K5 V54复现`old=93.10%`,`min_old=81.43%`,`seen_new=83.93%`,`min_new=71.43%`。 |
| `local_v54_policy_verify_20260706/k10_v54_policy_verify.json`与`k10_v53_current_compare.json` | 当前代码同一split下完全一致；V54高K正确映射V49。 |

### 当前边界

V54是一个可落地的低K旧类域适应与新类均值优化：相对本节V36/K5对照，old提升+1.19pp，min_old提升+1.43pp，seen_new提升+0.71pp；但K5最低新类仍为71.43%，未达到75%地板，不能声称解决“新类增多下最低类过低”问题。后续应继续针对`1-2`的support-only可观测风险设计机制，避免使用query真值定向偏置。

### Git与N607同步

| item | value |
| --- | --- |
| Git commit | `92cfbe0 Add adaptive qKNN v54 low-K policy` |
| N607 preflight | direct `N607`通过；server time 2026-07-06 21:14:31 CST；project root可见；8张RTX3090空闲显存约10MiB。 |
| project process check | 未见当前用户CV-SincNet/qKNN/python训练进程。 |
| synced files | `code/scripts/phase2_support_metric_qknn_probe.py`、`code/scripts/launch_phase2_qknn_fftlogmag_v54_20260706.sh`、`code/tests/test_phase2_support_metric_qknn_v54_policy.py`、本报告`report.md`。 |
| synced code sha256 | `a7cff15a61545c5bf0a1fe64e0736e4196e75f6f9ae1f0ccab60b040634ff209`、`db60165a2ed2dc53e143c447511dd139d447f740cca2f29e1ba5ebc0d8c7626f`、`37d946be527698dd7063019914aff2eb0c01561e958e3bdcdb8a9700f670c3f0`；本地与N607一致。报告文件在写入本节后单独重新同步。 |
| remote verification | N607 `py_compile` PASS；V54单元测试PASS；launcher `bash -n` PASS；`--dry-run`展开K5/K10正式`stable_dualview_v54`命令和`*_v54_fftlogmag_policy_20260706.*`输出路径。 |
| remote launch | 本次未启动N607实验。 |
| SSH cleanup | 本地无残留`ssh.exe`，无到`172.31.111.215:22`的ESTABLISHED连接。 |

## 2026-07-06本地V55低Kquery-cluster候选

本节继续优化`qKNNV42`后续路线。协议边界保持不变：K5/K10少量`target receiver`样本用于旧类域适应和新类注册识别；阶段二部署在卫星端，support/query目标域样本均视为叠加LEO星地信道后的接收样本；query真值只用于离线评估和失败诊断，不写入可部署策略。

`E:\type10-7`根目录不是Git仓库。本轮代码仍只修改Git承载面`E:\type10-7\github_publish\CVS-RFFI-repo`，并把本报告镜像到Git承载面报告路径。由于K5最低类仍未过75%，本轮没有同步N607、没有启动远端实验。

### 诊断与取舍

| probe | K | setting | old | min_old | seen_new | min_new | 结论 |
| --- | ---: | --- | ---: | ---: | ---: | ---: | --- |
| `k5_v54_no_scenario_aware` | 5 | 关闭`scenario_aware` | 92.38% | 80.00% | 67.14% | 35.71% | 全局取消scenario约束会导致新类坍塌，不可用。 |
| `k5_v55_policy_verify` | 5 | 低K新类scenario fallback | 92.38% | 80.00% | 83.57% | 71.43% | 不能提升`1-2`地板，还损伤旧类；撤回。 |
| `k5_v55_no_quality_policy_verify` | 5 | 去掉support quality | 93.10% | 81.43% | 67.29% | 31.43% | support quality是防坍塌必要项。 |
| `k5_v54_query_pair_cluster_grid/wide` | 5 | query-pair cluster网格 | 92.86%-93.10% | 81.43% | 82.57%-83.86% | 71.43% | 能触达`2-13<->1-2`，但不能抬高最低类。 |
| `k5_v55_query_cluster_clean_verify` | 5 | V55新类query-cluster | 93.10% | 81.43% | 84.07% | 71.43% | 相对V54只提升新类均值+0.14pp，地板仍未解决。 |

K5主要失败不是简单权重问题。`1-2`在该split中仍为50/70；其`leo_low_elev_weak`子场景为50/55正确，而`leo_clear_weak`为0/15正确。K5 support缺少`1-2`的`leo_clear_weak`样本，严格scenario-aware路径会屏蔽该类；但全局放开scenario又会造成新类整体坍塌。因此本轮只保留较保守的query-cluster均值候选，撤回低K新类scenario fallback和labelprop fallback实验分支。

### 代码变更

| file | change |
| --- | --- |
| `code/scripts/phase2_support_metric_qknn_probe.py` | 新增`stable_dualview_v55`策略入口。K<10继承V54低K参数`topm=2`,`proto_mix=0.45`,`aux_score_weight=0.26`，并启用新类范围`query_cluster_weight=0.05`,`query_cluster_rounds=3`,`query_cluster_support_weight=0.55`,`query_cluster_temperature=0.08`,`query_cluster_clip=1.0`；K>=10仍映射`stable_dualview_v49`，保留高K旧类保护。 |
| `code/tests/test_phase2_support_metric_qknn_v55_policy.py` | 覆盖V55低K自有策略与高K映射V49。 |
| `code/tests/test_phase2_qknn_scenario_class_fallback.py` | 移除已撤回的labelprop scenario fallback测试，保留`_class_scores`级别fallback语义测试。 |

### 本地验证

| command/artifact | result |
| --- | --- |
| `conda run -n ssr-gpu python -m py_compile code\scripts\phase2_support_metric_qknn_probe.py code\tests\test_phase2_support_metric_qknn_v55_policy.py` | PASS |
| `conda run -n ssr-gpu python code\tests\test_phase2_support_metric_qknn_v55_policy.py` | PASS，1 test |
| `conda run -n ssr-gpu python code\tests\test_phase2_support_metric_qknn_v54_policy.py` | PASS，1 test |
| `conda run -n ssr-gpu python code\tests\test_phase2_qknn_scenario_class_fallback.py` | PASS，2 tests |
| `conda run -n ssr-gpu python code\tests\test_phase2_qknn_old_only_scenario_fallback.py` | PASS，1 test |
| `local_v55_policy_verify_20260706/k5_v55_query_cluster_clean_verify.json` | K5 V55复现`old=93.10%`,`min_old=81.43%`,`seen_new=84.07%`,`min_new=71.43%`。 |
| `local_v55_policy_verify_20260706/k10_v55_query_cluster_clean_verify.json` | K10请求V55但有效策略为V49，`old=93.57%`,`min_old=84.29%`,`seen_new=87.64%`,`min_new=78.57%`，通过joint target。 |

### 当前结论

| candidate | K | requested policy | effective policy | old | min_old | seen_new | min_new | verdict |
| --- | ---: | --- | --- | ---: | ---: | ---: | ---: | --- |
| V54 | 5 | `stable_dualview_v54` | `stable_dualview_v54` | 93.10% | 81.43% | 83.93% | 71.43% | 旧类域适应与新类均值已优于V36，但K5地板未解决。 |
| V55 | 5 | `stable_dualview_v55` | `stable_dualview_v55` | 93.10% | 81.43% | 84.07% | 71.43% | 只带来+0.14pp新类均值，最低类仍为`1-2=50/70`；不作为完成目标。 |
| V55 | 10 | `stable_dualview_v55` | `stable_dualview_v49` | 93.57% | 84.29% | 87.64% | 78.57% | 高K保护路径正常，可作为K10稳态映射。 |

后续优化应优先围绕“support场景覆盖缺失时的可部署风险估计”设计，而不是继续扩大query真值导向的pair bias。当前最可疑目标是`1-2`在`leo_clear_weak`无support覆盖时的保守注册机制；任何新机制都必须先证明不会复现“关闭scenario-aware导致seen_new坍塌”的失败模式。

### 继续诊断负证据

V55提交后继续做了三组本地只读诊断，均未形成可提交策略：

| diagnostic | K | 设置 | old | min_old | seen_new | min_new | 结论 |
| --- | ---: | --- | ---: | ---: | ---: | ---: | --- |
| monkeypatch targeted fallback | 5 | 仅对`1-2/2-13`开启缺失场景fallback | 93.10% | 81.43% | 84.07% | 71.43% | 定向解屏蔽不改变最终assignment，说明瓶颈不只是`-1e9`硬屏蔽。 |
| `k5_v55_query_pair_on_cluster_grid` | 5 | V55叠加query-pair cluster，top_pairs=4/8/12，query_weight=0.05/0.1/0.2 | 93.10% | 81.43% | 83.86%-84.00% | 71.43% | query-pair后验修正不能抬`1-2`，部分设置降低`2-13`或seen_new。 |
| `k5_v56_lowk_transport_verify` | 5 | 临时V56=V55+弱source-target transport | 93.10% | 81.43% | 84.07% | 71.43% | transport在该低K配置下`rank_used=0`,`old_pairs=0`，无实际增益；临时代码已撤回。 |

这三组结果强化了当前判断：K5地板需要新的support-only可观测风险建模，不能靠已有scenario fallback、query-pair后验或低Ktransport简单叠加解决。

### Active-enrollment支持选择诊断

继续检查support覆盖是否是K5地板主因。该诊断使用`pool_per_old=80,pool_per_new=80`再选K=5个support，因此只代表“active enrollment / support selection upper-bound”诊断，不是严格只有K=5个目标标签到达时的部署证据。

| diagnostic | policy | seed scope | old | min_old | seen_new | min_new | 结论 |
| --- | --- | --- | ---: | ---: | ---: | ---: | --- |
| `k5_v55_active_enrollment_policy_grid` | `scenario_centroid` | 421038 | 91.90% | 78.57% | 86.64% | 71.43% | 能把`1-2`抬到81.43%，但旧类地板跌破80%，且最低类转移到`1-1`。 |
| `k5_v55_active_enrollment_seed_sweep` | `scenario_centroid` | 421030-421049 | 91.90% | 78.57% | 86.71% | 74.29% | 多seed下新类地板接近75%，但旧类地板仍不合格。 |
| `k5_v55_oldstable_newscenario_seed_sweep` | `old_stable_new_scenario_centroid` | 421030-421049 | 92.86% | 81.43% | 86.79% | 74.29% | 旧类恢复合格，新类仍差1个query；低类集中在`1-1/1-12/8-3/2-13`。 |
| `k5_v55_oldstable_newscenario_scenariobalanced` | `old_stable_new_scenario_centroid` + scenario-balanced assignment | 421030-421049 | 最优行仍不超过74.29%地板；部分行严重坍塌 | - | - | - | scenario-balanced assignment不是安全补丁。 |
| `k5_oldstable_newscenario_cluster_param_grid` | `old_stable_new_scenario_centroid` + query-cluster权重/温度网格 | 421035 | 92.86% | 81.43% | 86.79% | 74.29% | query-cluster调参不能补足最后1个query。 |

为便于后续复现实验，Git承载面新增`old_stable_new_scenario_centroid`支持选择策略：旧类保持`stable_first`以保护旧类域适应，新类使用`scenario_centroid`以覆盖LEO场景。该策略当前只作为诊断入口；在严格K-shot协议下，除非实际只用K=5个已接收标签且不依赖额外标注pool，否则不得声明为Stage2-C完成证据。

### 2026-07-06追加本地K5地板诊断

本节继续围绕`old_stable_new_scenario_centroid`的near-miss行和严格K5原始split做本地只读诊断。`pool_per_old=80,pool_per_new=80`的active-enrollment结果仍只表示从更大带标签候选池中选择K=5个support的上限诊断；严格K5结果仍以`pool_per_old=5,pool_per_new=5,policy=stable_first,seed=421038`为准。本节没有修改代码、没有同步N607、没有启动远端实验。

| diagnostic | scope | key setting | old | min_old | seen_new | min_new | 低类/结论 |
| --- | --- | --- | ---: | ---: | ---: | ---: | --- |
| `k5_oldstable_newscenario_topm_proto_aux_grid` | active pool80诊断 | `topm=1,proto_mix=0.45,aux=0.30` | 92.86% | 81.43% | 87.71% | 74.29% | 新类均值较前一active行继续提升，但最低仍是`1-12=52/70`，未过75%。 |
| `k5_oldstable_newscenario_topm1_querypair_grid` | active pool80诊断 | 上行叠加query-pair cluster，识别`1-1<->1-12` | 92.86% | 81.43% | 87.71% | 74.29% | query-pair仅改变少量边界，未补足`1-12`最后1个query。 |
| `k5_strict_topm1_aux30_verify` | strict K5验证 | `stable_first,pool5,topm=1,aux=0.30` | 93.10% | 81.43% | 84.21% | 71.43% | 严格K5不受益；最低仍为`1-2=50/70`，并伴随`1-12/19-3/2-13=52/70`。 |
| `k5_strict_transductive_small_grid` | strict K5验证 | 小网格扫`transductive_proto_weight=0/0.02/0.05` | 92.38% | 80.00% | 84.00% | 71.43% | 最优仍为71.43%；`1-12`可到53/70，但`1-2`仍50/70，且旧类均值下降。 |

一次较大的strict K5 transductive+dense cluster网格在本地超过3分钟超时，未生成完整结果文件；已清理本地残留`conda/python`进程，未把该超时视为实验成败证据。

当前边界：active-enrollment证明“如果星上/地面流程允许从更大目标域带标签pool中主动挑选K=5个support”，新类均值可到87.71%，旧类地板仍合格，但新类最低类仍差1个query；严格K5到达样本下仍停在71.43%地板。因此本轮不能声明qKNNV42后续优化已解决K5新类最低类过低问题，也不应同步N607正式跑。下一步应优先研究`1-2`严格K5支持场景缺失与`1-12`active pool80近邻混淆的共同可观测风险，而不是继续叠加全局query结构补丁。

### 2026-07-06显式scenario fallback与K10复核

本节继续围绕qKNNV42后续路线做本地验证。协议边界不变：K=5/K=10目标域support来自叠加LEO星地信道后的接收样本；query真值只用于离线评估。`E:\type10-7`根目录仍不是Git仓库，代码改动只落在Git承载面`E:\type10-7\github_publish\CVS-RFFI-repo`。本节没有N607 preflight、没有scp、没有远端启动。

#### 代码变更

| file | change |
| --- | --- |
| `code/scripts/phase2_support_metric_qknn_probe.py` | 新增`--scenario_class_fallback_grid`显式诊断开关，支持`none/all/old_only/new_only/old_role_only`，默认`none`，不改变既有adaptive策略。该开关只在某类缺少同scenario support时允许该类回退到全support评分，用于验证K5`1-2`场景缺失是否只是硬mask问题。 |
| `code/tests/test_phase2_support_metric_qknn_scenario_fallback_cli.py` | 新增解析测试，覆盖`new_only`、`none`保留adaptive状态和非法模式拒绝。 |

#### 本地验证

| command/artifact | result |
| --- | --- |
| `conda run --no-capture-output -n ssr-gpu python -m py_compile code\scripts\phase2_support_metric_qknn_probe.py` | PASS |
| `conda run --no-capture-output -n ssr-gpu python code\tests\test_phase2_support_metric_qknn_scenario_fallback_cli.py` | PASS，3 tests |
| `conda run --no-capture-output -n ssr-gpu python code\tests\test_phase2_qknn_scenario_class_fallback.py` | PASS，2 tests |
| `conda run --no-capture-output -n ssr-gpu python code\tests\test_phase2_qknn_old_only_scenario_fallback.py` | PASS，1 test |
| `conda run --no-capture-output -n ssr-gpu python code\tests\test_phase2_support_metric_qknn_v55_policy.py` | PASS，1 test |

#### 结果表

| diagnostic | K | setting | old | min_old | seen_new | min_new | 低类/结论 |
| --- | ---: | --- | ---: | ---: | ---: | ---: | --- |
| `k5_strict_topm1_source_proto_anchor_grid` | 5 | source prototype anchor，`topm=1,aux=0.30` | 93.10% | 81.43% | 84.21% | 71.43% | 最优仍为关闭该机制；`1-2=50/70`。 |
| `k5_strict_topm1_source_guard_grid` | 5 | source old guard | 93.10% | 81.43% | 84.21% | 71.43% | 分配结果不变，旧类保护没有带来额外收益。 |
| `k5_strict_topm1_support_metric_grid` | 5 | class-diag/mahal support-only metric | 93.10% | 81.43% | 84.29% | 71.43% | 仅提升seen_new+0.07pp，地板不变。 |
| `k5_strict_topm1_query_proto_refine_grid` | 5 | unlabeled query proto refine | 93.10% | 81.43% | 84.43% | 71.43% | transductive均值小幅提升，但`1-2`仍50/70。 |
| `k5_strict_topm1_scenario_fallback_grid` | 5 | `scenario_class_fallback=none/new_only/all` | 93.10% | 81.43% | 84.43% | 71.43% | 显式解mask没有抬高`1-2`，`new_only/all`还会把`2-13`拉到71.43%。 |
| `k10_strict_topm1_scenario_fallback_grid` | 10 | K10同参fallback复核 | 94.05% | 84.29% | 86.00% | 70.00% | K10旧类更强，但最低类转为`1-12=49/70`。 |
| `k10_strict_topm_proto_aux_grid` | 10 | `topm=1/2,proto=0.35/0.45/0.55,aux=0.22/0.26/0.30/0.34` | 94.05% | 84.29% | 87.07% | 71.43% | 最优`topm=1,proto=0.45,aux=0.34,qpr=0.01`；`1-12=50/70`仍不过75%。 |

#### 解释边界

逐样本预测确认，K5严格split中`1-2`的`leo_clear_weak`查询15/15全错，原严格scenario-aware下truth score被压到约`-9.55e8`；但显式`new_only/all`fallback后仍未增加`1-2`正确数，说明当前瓶颈不是单纯硬mask，而是缺少同场景support后，`1-2`在清晰弱信道上的相似度仍输给`1-18/1-19/10-10`等相邻类。K10增加support后`1-2`可到80%+，但最低类转移到`1-12`，说明“最低类过低”是多类局部混淆问题，不是单一`1-2`补丁可以解决。

当前可保留的工程产物是显式fallback诊断开关和测试；该开关默认关闭，不作为新默认策略。当前不能声明qKNNV42后续优化已解决K5/K10新类地板问题，也不应启动N607正式实验。下一步应把优化重点转向可部署的support场景覆盖风险估计与类簇级保守注册，而不是继续叠加全局fallback或query真值导向后验。

### 2026-07-06局部竞争与active support补充诊断

本节继续沿qKNNV42后续优化目标做本地诊断。协议边界不变：目标域support/query均来自叠加LEO星地信道后的接收样本；query真值只用于离线评估。本节没有N607 preflight、没有scp、没有远端启动。`old_stable_new_scenario_diverse`在K5 active-enrollment下是负结果，会把最低类转移到`1-1=50/70`；但K10验证显示它能同时保住旧类域适应和新类地板，因此保留为显式候选策略，默认策略不变。

| diagnostic | K | support来源 | key setting | old | min_old | seen_new | min_new | 低类/结论 |
| --- | ---: | --- | --- | ---: | ---: | ---: | ---: | --- |
| `k5_strict_local_competition_grid` | 5 | strict `pool_per=5` | `local_competition_weight=0.04,k=5,scope=all` | 93.10% | 81.43% | 84.57% | 71.43% | `1-2=50/70`仍未解决，只抬高新类均值。 |
| `k5_strict_query_graph_grid` | 5 | strict `pool_per=5` | query graph小网格 | 93.10% | 81.43% | 84.21% | 71.43% | 非零平滑不改善最低类。 |
| `k5_oldstable_newscenario_local_competition_grid` | 5 | active `pool_per=80`中选K=5 | `old_stable_new_scenario_centroid + local_competition_weight=0.04,k=5,scope=all` | 92.62% | 81.43% | 86.93% | 75.71% | K5 active-enrollment首个过75%地板候选；低类为`1-12=53/70`、`1-1/2-13=54/70`。 |
| `k10_strict_local_competition_grid` | 10 | strict `pool_per=10` | 最优仍为关闭local competition，`qpr=0.01` | 94.05% | 84.29% | 87.07% | 71.43% | `1-12=50/70`仍未解决。 |
| `k10_oldstable_newscenario_local_competition_grid` | 10 | active `pool_per=80`中选K=10 | `old_stable_new_scenario_centroid + local_competition_weight=0.02,k=3,scope=role` | 94.76% | 85.71% | 85.43% | 67.14% | 旧类更强但新类坍塌更重，K10 active选择不是当前路线。 |
| `k10_active_policy_local_competition_grid` | 10 | active `pool_per=80`中选K=10 | `scenario_diverse,qpr=0.01` | 92.86% | 78.57% | 92.79% | 81.43% | 新类地板过线但旧类min低于80%，不能直接作为旧类域适应候选。 |
| `k10_oldstable_newscenario_diverse_local_competition_grid` | 10 | active `pool_per=80`中选K=10 | `old_stable_new_scenario_diverse + local_competition_weight=0.02,k=3,scope=role,qpr=0.01` | 95.48% | 87.14% | 91.79% | 80.00% | K10 active-enrollment首个同时抬旧类和新类地板候选；低类为`1-12/2-13=56/70`、`11-10/8-3=57/70`。 |

逐query诊断显示，严格K5的`1-2`错误主要集中在`leo_clear_weak`场景，固定support没有覆盖该场景；显式fallback解屏蔽后仍不能净增正确数，并会把低类扩展到`2-13=50/70`。因此当前最清晰的可推进方向不是继续做全局fallback或query graph，而是把“从接收候选流中选择K个support”的active-enrollment流程和support原型局部竞争结合起来。该K5/K10候选仍不能作为“只有K个样本到达且无候选pool可选”的严格部署证据；若卫星端协议允许先接收更多叠加LEO目标域候选、再只保留/标注K=5或K=10个support用于注册，则目前最强候选分别是K5的`old_stable_new_scenario_centroid + local_competition`和K10的`old_stable_new_scenario_diverse + local_competition + qpr`。下一步应验证这两个active候选的多seed稳定性，并把候选pool选择协议写清楚后再考虑N607正式实验。

### 2026-07-06active候选多seed稳定性复核

本节对上一节两个active-enrollment候选追加20个seed复核。协议边界不变：support/query均来自目标接收机域且叠加LEO星地信道；但`pool_per_old=80,pool_per_new=80`表示先从更大目标域候选池中选择K个support，因此该结果只支持“主动选择K-shot support”的路线，不支持“实际只接收到固定K个样本且无法选择”的严格到达协议。本节没有N607 preflight、没有scp、没有远端启动。

| sweep | K | policy/setting | seeds | pass_old80 | pass_new75 | pass_both | mean old | worst old | mean seen_new | worst seen_new | mean min_old | worst min_old | mean min_new | worst min_new | verdict |
| --- | ---: | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| `k5_active_centroid_localcomp_seed_sweep.csv` | 5 | `old_stable_new_scenario_centroid + local_competition_weight=0.04,k=5,scope=all` | 421030-421049 | 18/20 | 3/20 | 1/20 | 93.10% | 91.43% | 86.82% | 85.71% | 82.14% | 77.14% | 73.93% | 71.43% | 单seed可过75%新类地板，但多seed不稳定；不能作为当前主线完成证据。 |
| `k10_active_oldstable_diverse_localcomp_seed_sweep.csv` | 10 | `old_stable_new_scenario_diverse + local_competition_weight=0.02,k=3,scope=role,qpr=0.01` | 421030-421049 | 20/20 | 20/20 | 20/20 | 94.60% | 93.10% | 91.71% | 91.36% | 85.57% | 81.43% | 79.36% | 78.57% | 当前最强本地候选；20/20同时满足旧类floor>=80%和新类floor>=75%，且最差新类floor仍为78.57%。 |

| sweep | best/worst row | seed | old | min_old | seen_new | min_new | 低类说明 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| K5 active | best joint floor | 421031 | 91.43% | 77.14% | 87.43% | 75.71% | 新类刚过75%，但旧类最低`20-19=54/70`，旧类floor不合格。 |
| K5 active | worst joint floor | 421040 | 94.05% | 84.29% | 86.57% | 71.43% | 旧类合格但新类最低`1-12=50/70`，仍复现最低类过低。 |
| K10 active hybrid | best joint floor | 421040 | 95.95% | 88.57% | 92.14% | 80.00% | 新类最低`1-12=56/70`，旧类最低`14-7=62/70`。 |
| K10 active hybrid | worst joint floor | 421039 | 95.00% | 87.14% | 91.36% | 78.57% | 新类最低`1-12/2-13=55/70`，仍高于75%门槛。 |

#### 当前决策

严格固定K到达路线仍未解决：K5 strict地板停在71.43%，K10 strict地板也停在71.43%，局部竞争、query graph、显式scenario fallback和query proto refine只能抬均值或转移最低类，不能稳定解决低类坍塌。K5 active路线证明support场景覆盖选择有效，但20seed只有1/20同时过旧类和新类floor，仍不够稳。K10 active hybrid是当前唯一同时改善旧类域适应和新类地板的候选：相对K10 strict best的`old=94.05%,min_old=84.29%,seen_new=87.07%,min_new=71.43%`，20seed均值达到`old=94.60%,min_old=85.57%,seen_new=91.71%,min_new=79.36%`，最差seed也保持`min_old>=81.43%,min_new>=78.57%`。

因此下一步若继续本地优化，应围绕K10 active hybrid做两个收敛动作：一是把`pool_per=80`主动候选池协议写成明确Stage2-C子协议，说明星上/地面流程如何只保留或标注K=10个support；二是在该候选上补unknown拒识/FAR和更多target receiver复核。若目标必须严格限制为“只接收到K=5或K=10个样本且无候选池选择”，则本轮尚未找到可发布优化，不能同步N607正式实验。

### 2026-07-06严格K支持集LOO-pair低类修正诊断

本节回到用户强调的严格少样本设置：`pool_per_old=K,pool_per_new=K`，即不依赖`pool_per=80`主动候选池。所有support/query仍来自目标接收机域并叠加LEO星地信道；query真值只用于离线评估。本节没有修改代码、没有N607 preflight、没有scp、没有远端启动。

| diagnostic | K | rows | key setting | old | min_old | seen_new | min_new | 低类/结论 |
| --- | ---: | ---: | --- | ---: | ---: | ---: | ---: | --- |
| `k10_strict_supportloo_pair_grid_20260706.csv` | 10 | 225 | `support_loo_pair_linear_weight=0.04,top_pairs=16` | 94.05% | 84.29% | 87.36% | 74.29% | `1-12=52/70`，较旧strict K10地板71.43%提高2个query，但仍差1个query过75%。 |
| `k10_strict_pair_cluster_competition_grid_20260706.csv` | 10 | 144 | 上行叠加`query_cluster_weight=0.04,local_competition_weight=0.04,k=3,scope=all` | 94.05% | 84.29% | 87.57% | 74.29% | 均值继续小幅提高，但最低类仍为`1-12=52/70`。 |
| `k10_strict_support_bias_pair_grid_20260706.csv` | 10 | 54 | 上行叠加`support_bias_weight`网格 | 94.05% | 84.29% | 87.57% | 74.29% | 最优仍选择`support_bias_weight=0`，support列偏置不能补足最后1个query。 |
| `k10_strict_scenariobalanced_pair_verify_20260706.csv` | 10 | 1 | scenario-balanced assignment | 66.67% | 20.00% | 39.07% | 21.43% | 强负结果；按场景强行配额会破坏旧类和新类，不可作为补丁。 |
| `k5_strict_supportloo_pair_grid_20260706.csv` | 5 | 288 | support-LOO pair + local competition小网格 | 93.10% | 81.43% | 84.71% | 71.43% | `1-2=50/70`仍未改善；`1-12`可到53/70但最低类转回`1-2`。 |

当前严格K结论：support-LOO pair线性修正能从support内部错误中提取可部署信号，并把K10 strict的最低新类从50/70提升到52/70，同时保持旧类`old=94.05%,min_old=84.29%`不降；但它没有达到75%地板，也没有解决K5 strict的`1-2`场景覆盖缺失。K5 strict仍需要新的“缺场景support时的类内风险估计”机制；K10 strict可继续沿support-LOO pair方向做更细的类簇/场景门控，但当前不能写成完成qKNNV42优化目标，也不应同步N607正式实验。

### 2026-07-06支持集场景残差补全诊断

本节继续严格K-shot路线：`pool_per_old=K,pool_per_new=K`，不使用active候选池。新增的`scenario_residual_*`机制只使用support特征、support标签、support场景和query场景，不使用query真值；当某类缺少query场景support且原scenario-aware评分为硬mask时，用其他类在该场景上的support残差合成有限候选分数。`E:\type10-7`根目录仍不是Git仓库，本轮代码改动和artifact镜像在Git承载面`E:\type10-7\github_publish\CVS-RFFI-repo`；本节没有N607 preflight、没有scp、没有远端启动。

#### 代码变更与验证

| item | result |
| --- | --- |
| `code/scripts/phase2_support_metric_qknn_probe.py` | 新增`_scenario_residual_completion_scores`和`--scenario_residual_weight/min_classes/clip/scope_grid`，输出`scenario_residual_count`与`stored_scenario_residual_scalars`。 |
| `code/tests/test_phase2_qknn_scenario_residual_completion.py` | 新增support-only残差补全测试，覆盖普通boost、零权重不改变分数、`-1e9`硬mask替换。 |
| `conda run --no-capture-output -n ssr-gpu python -m py_compile code\scripts\phase2_support_metric_qknn_probe.py` | PASS |
| `conda run --no-capture-output -n ssr-gpu python code\tests\test_phase2_qknn_scenario_residual_completion.py` | PASS，3 tests |

#### 严格K结果

| diagnostic | K | rows | best setting | old | min_old | seen_new | min_new | 低类/结论 |
| --- | ---: | ---: | --- | ---: | ---: | ---: | ---: | --- |
| `k10_strict_scenario_residual_grid_20260706.csv` | 10 | 20 | `scenario_residual_weight=0.5,min_classes=2,clip=0.5` | 94.05% | 84.29% | 87.50% | 74.29% | 相对weight=0只提升seen_new+0.14pp；最低仍为`1-12=52/70`，未过75%。 |
| `k5_strict_scenario_residual_grid_20260706.csv` | 5 | 20 | `scenario_residual_weight=0.5,min_classes=2,clip=0.5` | 93.10% | 81.43% | 86.14% | 74.29% | 相对weight=0把seen_new从84.71%提升到86.14%，并把`1-2`从50/70提升到53/70；但最低转移到`19-3=52/70`，仍差1个query过75%。 |

| K | baseline low classes | residual best low classes |
| ---: | --- | --- |
| 10 | `1-12=52/70`,`2-13=54/70`,`1-2=56/70` | `1-12=52/70`,`2-13=54/70`,`1-2=55/70` |
| 5 | `1-2=50/70`,`19-3=52/70`,`2-13=52/70`,`1-12=53/70` | `19-3=52/70`,`1-2=53/70`,`1-12=54/70`,`1-15=54/70` |

当前解释边界：场景残差补全是有效的support-only机制，能解除严格K5中`1-2`缺场景support导致的硬mask坍塌，并明显提升新类均值；但它把最低类转移到`19-3`，K5/K10都仍未满足`min_new>=75%`。因此本轮仍不能声明qKNNV42严格K优化目标完成，也不应同步N607正式实验。下一步应在该残差补全基础上增加“低类候选间局部保守竞争/风险门控”，重点约束`19-3/2-13/1-12/1-2`这一组同LEO场景下的互相挤压，而不是再做全局分数平移。

### 2026-07-06严格K残差后局部门控与labelprop复核

本节继续严格K-shot路线：`pool_per_old=K,pool_per_new=K`，不使用active候选池。所有support/query仍来自目标接收机域并叠加LEO星地信道；query真值只用于离线评估和错误流审计。本节没有N607 preflight、没有scp、没有远端启动。`E:\type10-7`根目录仍不是Git仓库，本轮代码改动和artifact镜像只落在Git承载面`E:\type10-7\github_publish\CVS-RFFI-repo`。

#### 代码变更与验证

| item | result |
| --- | --- |
| `code/scripts/phase2_support_metric_qknn_probe.py` | 新增`_scenario_proto_refine_scores`和`--scenario_proto_refine_*_grid`，用于support-only同场景类原型细化诊断；默认权重为0，不改变既有策略。 |
| `code/tests/test_phase2_qknn_scenario_residual_completion.py` | 新增同场景原型细化单测，验证只用support同场景原型时能推高正确类分数。 |
| `conda run --no-capture-output -n ssr-gpu python -m py_compile code\scripts\phase2_support_metric_qknn_probe.py` | PASS |
| `conda run --no-capture-output -n ssr-gpu python code\tests\test_phase2_qknn_scenario_residual_completion.py` | PASS，4 tests |

#### 严格K结果

| diagnostic | K | rows | best setting | old | min_old | seen_new | min_new | 低类/结论 |
| --- | ---: | ---: | --- | ---: | ---: | ---: | ---: | --- |
| `k5_strict_scenario_residual_best_predictions.csv` | 5 | 1820 query rows | 残差最佳逐query审计 | 93.10% | 81.43% | 86.14% | 74.29% | `19-3=52/70`,`1-2=53/70`,`1-12/1-15=54/70`；低类来自`19-3<->1-15`,`1-12<->1-1/8-3`,`1-2->1-19`等局部混淆。 |
| `k5_strict_residual_pairlinear_grid_20260706.csv` | 5 | 32 | 残差最佳叠加support-LOO pair linear | 93.10% | 81.43% | 86.14% | 74.29% | 最优仍为`support_loo_pair_linear_weight=0`；pair linear不能补足K5最后1个query。 |
| `k5_strict_scenario_proto_refine_grid_20260706.csv` | 5 | 36 | 同场景原型细化网格 | 93.10% | 81.43% | 86.14% | 74.29% | 最优仍为`scenario_proto_refine_weight=0`；非零权重只转移低类，不提升joint排序。 |
| `k5_strict_residual_policy_gate_grid_20260706.csv` | 5 | 6 | 残差最佳叠加已有策略门控 | 92.38% | 80.00% | 86.57% | 74.29% | 新类均值小升，但旧类均值下降且地板仍为`19-3=52/70`。 |
| `k5_strict_residual_labelprop_grid_20260706.csv` | 5 | 96 | `labelprop_weight=0.035,k=8,alpha=0.72,temp=0.08,rounds=8,local_competition=0.04` | 92.38% | 80.00% | 86.71% | 74.29% | K5仍没有任何行通过新类75%地板；最低仍是`19-3=52/70`。 |
| `k10_strict_residual_policy_gate_grid_20260706.csv` | 10 | 6 | 残差最佳叠加已有策略门控 | 94.05% | 84.29% | 88.29% | 75.71% | `adaptive_qknn_policy=none`的轻量labelprop行通过joint target；旧V42/V50类门控反而回退到72.86%地板。 |
| `k10_strict_residual_labelprop_grid_20260706.csv` | 10 | 48 | `labelprop_weight=0.035,k=8,alpha=0.8,temp=0.08,rounds=8` | 94.05% | 84.29% | 88.36% | 75.71% | 16/48行通过joint target；最低类为`1-12=53/70`，`1-1/1-2=56/70`。 |

#### 当前决策

K10严格路线出现可保留正向候选：相对上一节`k10_strict_scenario_residual_grid_20260706.csv`的`old=94.05%,min_old=84.29%,seen_new=87.50%,min_new=74.29%`，新的轻量labelprop组合保持旧类域适应不降，并把`seen_new`提升到88.36%、`min_new`提升到75.71%，首次在严格K10固定到达协议下通过`old>=80%`与`seen-new floor>=75%`的joint target。

K5严格路线仍未解决。最佳K5只把`seen_new`推进到86.71%，`min_new`仍为74.29%，且旧类floor刚好80.00%；因此不能声明K5目标完成。下一步应围绕`19-3<->1-15`的`leo_clear_weak`对称混淆做support-only风险建模，而不是继续扩大全局labelprop或同场景原型平移。

本节所有正向结论仍是本地Stage2-C identity诊断；unknown拒识/FAR未评估，未同步N607，不能写成部署成功或论文最终结论。

### 2026-07-06严格K同场景pair边界复核

本节继续严格K-shot路线：`pool_per_old=K,pool_per_new=K`，不使用`pool_per=80`主动候选池。目标是针对上一节K5最低类`19-3=52/70`及其主要混淆`19-3<->1-15`，测试一个默认关闭的support-only同场景pair边界机制。该机制只使用support特征、support标签、support场景和query场景；query真值只用于离线评估。本节没有N607 preflight、没有scp、没有远端启动。`E:\type10-7`根目录仍不是Git仓库，代码和artifact已镜像到Git承载面`E:\type10-7\github_publish\CVS-RFFI-repo`。

#### 代码变更与验证

| item | result |
| --- | --- |
| `code/scripts/phase2_support_metric_qknn_probe.py` | 新增`_scenario_pair_refine_scores`和`--scenario_pair_refine_*_grid`，输出`scenario_pair_refine_count/pairs/stored_scenario_pair_refine_scalars`；默认`weight=0,top_pairs=0`，不改变既有路线。 |
| `code/scripts/phase2_qknn_old_anchor_transport_diag.py` | 补齐`_evaluate_metric_qknn`新增及既有缺省参数，避免旧诊断入口漏参。 |
| `code/tests/test_phase2_qknn_scenario_residual_completion.py` | 新增同场景pair边界单测，验证support-only二分类边界能提高正确类相对分数。 |
| `conda activate ssr-gpu;python -m py_compile code\scripts\phase2_support_metric_qknn_probe.py code\scripts\phase2_qknn_old_anchor_transport_diag.py` | PASS |
| `conda activate ssr-gpu;python code\tests\test_phase2_qknn_scenario_residual_completion.py` | PASS，5 tests |
| `_evaluate_metric_qknn`签名覆盖检查 | `phase2_qknn_old_anchor_transport_diag.BASE_DEFAULTS`无缺省漏项。 |

#### 严格K5结果

| diagnostic | K | rows | setting | old | min_old | seen_new | min_new | 结论 |
| --- | ---: | ---: | --- | ---: | ---: | ---: | ---: | --- |
| `k5_strict_scenario_pair_refine_grid_20260706.csv` | 5 | 192 | 残差+labelprop最佳上叠加`scenario_pair_refine_weight=0,0.02,0.04,0.06;top_pairs=0,6,10;min_sim=0.80` | 92.38% | 80.00% | 86.71% | 74.29% | 最优仍为`scenario_pair_refine_weight=0`；非零pair边界没有补足`19-3`最后1个query。 |
| `k5_strict_scenario_pair_refine_lowthreshold_probe_20260706.csv` | 5 | 1 | `weight=0.01,top_pairs=30,min_sim=0.0` | 92.38% | 80.00% | 86.50% | 74.29% | 低阈值可捕获`leo_clear_weak:1-15<->19-3@0.678`，但只抬高`1-15`，`19-3`仍为52/70，均值下降。 |
| `k5_strict_scenario_balanced_residual_labelprop_verify_20260706.csv` | 5 | 2 | 在残差+labelprop最佳上启用`scenario_balanced_assignment` | 61.43% | 20.00% | 39.00% | 18.57% | 强负结果；按场景强配额会破坏旧类和新类，不可用于K5。 |
| `k5_oldstable_newscenario_centroid_current_verify_20260706.csv` | 5 | 1 | `pool_per_old=5,pool_per_new=5,policy=old_stable_new_scenario_centroid` | 92.38% | 80.00% | 85.14% | 72.86% | 严格K复验失败；历史通过行依赖`pool_per_old=80,pool_per_new=80`主动候选池，不能当作K=5少样本严格结论。 |

#### 当前决策

`scenario_pair_refine`作为默认关闭诊断机制可保留，因为它可追踪、可测试且不改变既有默认行为；但它没有解决K5严格路线。K5 strict当前仍停在`old=92.38%,min_old=80.00%,seen_new=86.71%,min_new=74.29%`，最低类仍是`19-3=52/70`。历史`old_stable_new_scenario_centroid`的`min_new=75.71%`行必须标为`pool_per=80`主动候选池诊断，不能提升为用户本轮要求的`K=5`少样本协议结果。

K10 strict正向候选仍保持上一节结论：`k10_strict_residual_labelprop_grid_20260706.csv`中`old=94.05%,min_old=84.29%,seen_new=88.36%,min_new=75.71%`，但unknown拒识/FAR未评估。下一步若继续严格K5，应优先寻找能改变`19-3`底层分数排序的support-only风险估计，而不是pair线性边界、同场景强配额或`pool_per=80`主动选择。

### 2026-07-07严格K同场景pair中心化复核

本节继续上一节K5严格路线，在当前K5残差+labelprop最佳配置上测试同场景pair边界的无标签query中心化版本。中心化只使用query特征分布本身，不读取query真值；query真值仅用于离线评估。特征文件和分割与当前K5基线一致：`features_hardpair_HP08L5_n20.npz`，`old_role=target_old`，`new_role=target_unknown`，`K=5`，`pool_per_old=5,pool_per_new=5`，`exclude_pool_from_query=false`。本节没有N607 preflight、没有scp、没有远端启动。

#### 代码变更与验证

| item | result |
| --- | --- |
| `code/scripts/phase2_support_metric_qknn_probe.py` | 为`_scenario_pair_refine_scores`新增`center`参数和`--scenario_pair_refine_center_grid`，支持`none/query_median/query_mean`；默认`none`保持旧行为。 |
| `code/tests/test_phase2_qknn_scenario_residual_completion.py` | 更新同场景pair边界单测，显式覆盖默认`center=none`路径。 |
| `conda run --no-capture-output -n ssr-gpu python -m py_compile code\scripts\phase2_support_metric_qknn_probe.py code\tests\test_phase2_qknn_scenario_residual_completion.py` | PASS |
| `conda run --no-capture-output -n ssr-gpu python code\tests\test_phase2_qknn_scenario_residual_completion.py` | PASS，5 tests |

#### 严格K5结果

| diagnostic | K | rows | setting | old | min_old | seen_new | min_new | 结论 |
| --- | ---: | ---: | --- | ---: | ---: | ---: | ---: | --- |
| `k5_strict_scenario_pair_center_grid_20260707.csv` | 5 | 128 | `weight=0,0.04,0.08,0.12;center=none/query_median;top_pairs=30;min_sim=0/0.5` | 92.38% | 80.00% | 86.71% | 74.29% | 最优仍全部为`weight=0`；非零中心化pair边界没有提升`19-3=52/70`。 |
| `k5_strict_scenario_pair_center_strong_grid_20260707.csv` | 5 | 64 | `weight=0.2,0.5,1.0,2.0;center=query_median;top_pairs=12/30;min_sim=0.5` | 92.38% | 80.00% | 83.43% | 72.86% | 强权重没有提升`19-3`，反而把低类转移到`1-2/2-13=51/70`并降低新类均值。 |

#### 当前决策

`scenario_pair_refine_center`作为默认关闭诊断参数可保留，因为它不改变既有默认路线且能复现实验假设；但它不是K5修复方向。当前K5严格最佳仍是上一节残差+labelprop行：`old=92.38%,min_old=80.00%,seen_new=86.71%,min_new=74.29%`，最低类仍为`19-3=52/70`。中心化pair边界最多提升`1-15`到54/70以上，但不能把`19-3`过75%，强权重还会压低`1-2/2-13`。

下一步不应继续增加pair边界权重。更合理的方向是做逐query分数审计后的“低类安全重排”或“scenario内候选集局部最小类保护”，但必须只使用K-shot support和无标签query分布，且要先证明不会牺牲`1-2/2-13/1-12`地板。
