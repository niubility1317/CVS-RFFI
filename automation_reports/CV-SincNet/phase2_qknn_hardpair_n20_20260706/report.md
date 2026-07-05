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
