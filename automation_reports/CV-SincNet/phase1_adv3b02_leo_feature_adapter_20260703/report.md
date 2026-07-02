# Phase1 ADV3B02 Source-Only LEO Feature Adapter

## Objective

根据更新后的目标，先实现目标1：只使用地面/source训练数据构造`clean -> LEO`成对监督，训练一个后置feature-space适配器，使叠加星地信道后的样本特征尽量回到clean特征空间。随后在适配后特征上继续评估目标2：基于`ADV3B02_CORE90_SOFT_E200phase1`做未知类拒识，要求`unknown_FAR<=0.05`且旧类性能下降`<=2pp`。

## Protocol Boundary

| Item | Setting |
|---|---|
| Base model | Frozen`ADV3B02_CORE90_SOFT_E200phase1` |
| Adapter training data | Source role only: source clean features paired with source LEO single-observation features |
| Target query data | Satellite/LEO only; no target clean feature is used |
| Target labels/support | Not used for adapter training or threshold fitting |
| Unknown query labels | Evaluation-only; not used for adapter training or threshold fitting |
| Deployment interpretation | Phase1-only Stage2-A style rejection diagnostic, not Phase2-B/C few-shot adaptation |

## Method

The adapter is a lightweight post-feature repair layer applied to frozen Phase1`z_id`features:

```text
z_sat -> adapter(z_sat) ~= z_clean
```

Training loss combines source-only feature alignment and old-class identity preservation:

```text
L = pair_weight * smooth_l1(adapter(z_sat), z_clean)
  + cos_weight * (1 - cosine(adapter(z_sat), z_clean))
  + proto_ce_weight * CE(cosine(adapter(z_sat), clean_source_prototypes), tx)
  + residual_weight * ||adapter(z_sat)-z_sat||^2
```

After fitting on source pairs, the same adapter is applied to the complete satellite-only single-observation NPZ for each cell. The output NPZ keeps target query rows satellite-only and records in`manifest_json`that no target clean, target labels, or unknown query labels were used.

## Local Changes

| File | Purpose |
|---|---|
| `E:\type10-7\code\scripts\fit_apply_phase1_leo_feature_adapter.py` | Fits a source-only LEO feature repair adapter and writes an adapted satellite-only feature NPZ. |
| `E:\type10-7\code\scripts\sweep_phase1_adv3b02_leo_feature_adapter_20260703.sh` | Runs 10 strict satellite single-observation cells with 5 adapter variants and 7 rejection policies. |

## Local Verification

| Command | Result |
|---|---|
| `C:\Users\lh594\.conda\envs\ssr-gpu\python.exe -m py_compile E:\type10-7\code\scripts\fit_apply_phase1_leo_feature_adapter.py` | PASS |
| `bash -n code/scripts/sweep_phase1_adv3b02_leo_feature_adapter_20260703.sh` | PASS |
| `bash -n code/scripts/sweep_phase1_adv3b02_leo_feature_adapter_20260703.sh` after adding GPU/cell sharding | PASS |
| local synthetic NPZ smoke for`fit_apply_phase1_leo_feature_adapter.py` | PASS: output rows`48`, source pairs`16`, `uses_target_clean=false`, val pair MSE after`0.007531` |

## Local Version State

`E:\type10-7\code` is not a Git repository, so changed scripts are mirrored into the Git-backed release workspace and snapshotted before N607 sync.

| File | SHA256 |
|---|---|
| `E:\type10-7\code\scripts\fit_apply_phase1_leo_feature_adapter.py` | `A048D11006FA5E13619C8AABEE54982CB8555B4BCDEFB9EC73DC4FA38D274F8A` |
| `E:\type10-7\code\scripts\sweep_phase1_adv3b02_leo_feature_adapter_20260703.sh` | `79279146CAE5BF6276A81F806C02779AFCF40285FCB7A536146AD0A6D1079D31` |

## Planned N607 Matrix

| Field | Value |
|---|---|
| Remote root | `/home/szu2070436088/2510044040/CV-SincNet` |
| Remote script | `/home/szu2070436088/2510044040/CV-SincNet/code/scripts/sweep_phase1_adv3b02_leo_feature_adapter_20260703.sh` |
| Matrix log root | `/home/szu2070436088/2510044040/CV-SincNet/logs/phase1_adv3b02_leo_feature_adapter_matrix_20260703` |
| Expected summary | `/home/szu2070436088/2510044040/CV-SincNet/logs/phase1_adv3b02_leo_feature_adapter_matrix_20260703/leo_feature_adapter_summary.csv` |
| Expected rows | 350 rows: 10 cells x 5 adapter variants x 7 rejection policies |
| GPU allocation | Up to 8 bounded shard processes: `CELL_SHARD_INDEX=0..7`, `CELL_SHARD_COUNT=8`, `GPU=0..7`; adapter fitting uses each shard's `CUDA_VISIBLE_DEVICES` |

## Adapter Variants

| Adapter | Mechanism | Purpose |
|---|---|---|
| `LEOADAPT_IDENTITY` | no repair | Baseline control on the same adapted-eval path |
| `LEOADAPT_LINR_COS` | linear residual | Low-capacity source LEO correction |
| `LEOADAPT_MLP64_BAL` | MLP residual | Balanced feature/cosine/identity repair |
| `LEOADAPT_MLP128_CE` | MLP residual with stronger prototype CE | Bias toward old-class identity preservation |
| `LEOADAPT_AFFINE` | global affine map | Test whether a global source satellite-to-clean map is enough |

## Metrics to Inspect

| Metric group | Fields |
|---|---|
| Target1 alignment | `val_pair_mse_before/after`, `val_pair_cos_before/after`, `val_proto_acc_before/after` |
| Target2 rejection | `unknown_FAR`, `known_closed_accuracy_no_reject`, `known_full_accuracy_after_reject`, `old_drop_pp_vs_closed`, `passes_dual_target` |
| Safety checks | `uses_target_clean=false`, `uses_target_labels=false`, `uses_unknown_query_for_training=false` |

## Risks

| Risk | Mitigation |
|---|---|
| Feature repair may overfit source LEO pairs and distort target receiver old-class geometry. | Include held-out source-pair validation metrics and identity/prototype CE; compare closed old accuracy by adapter. |
| Better clean alignment may also pull unknowns toward old prototypes, worsening FAR. | Keep rejection metrics in the same matrix; do not claim target2 from alignment alone. |
| Existing single-observation NPZs may be missing. | Matrix skips missing cells with explicit log lines rather than fabricating inputs. |
