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
| `E:\type10-7\code\scripts\launch_phase1_adv3b02_leo_feature_adapter_shards_20260703.sh` | Launches the matrix as 8 bounded GPU/cell shards without relying on interactive shell quoting. |
| `E:\type10-7\code\scripts\sweep_phase1_adv3b02_leo_feature_adapter_v2_20260703.sh` | V2 matrix: trains adapters on all three source LEO scenario feature pairs while applying only to strict satellite single-observation test NPZs. |

## Local Verification

| Command | Result |
|---|---|
| `C:\Users\lh594\.conda\envs\ssr-gpu\python.exe -m py_compile E:\type10-7\code\scripts\fit_apply_phase1_leo_feature_adapter.py` | PASS |
| `C:\Users\lh594\.conda\envs\ssr-gpu\python.exe -m py_compile E:\type10-7\code\scripts\fit_apply_phase1_leo_feature_adapter.py` after N607 NumPy compatibility patch | PASS |
| `bash -n code/scripts/sweep_phase1_adv3b02_leo_feature_adapter_20260703.sh` | PASS |
| `bash -n code/scripts/sweep_phase1_adv3b02_leo_feature_adapter_20260703.sh` after adding GPU/cell sharding | PASS |
| `bash -n code/scripts/launch_phase1_adv3b02_leo_feature_adapter_shards_20260703.sh` | PASS |
| `bash -n code/scripts/sweep_phase1_adv3b02_leo_feature_adapter_v2_20260703.sh` | PASS |
| local synthetic NPZ smoke for`fit_apply_phase1_leo_feature_adapter.py` | PASS: output rows`48`, source pairs`16`, `uses_target_clean=false`, val pair MSE after`0.007531` |
| local synthetic NPZ smoke after N607 NumPy compatibility patch | PASS: output rows`48`, features dtype`float32`, logits dtype`float32` |
| local synthetic multi-train NPZ smoke after V2 patch | PASS: source pairs`32`, output dtype`float32`, val pair MSE before`0.013350`, after`0.000000` |

## Local Version State

`E:\type10-7\code` is not a Git repository, so changed scripts are mirrored into the Git-backed release workspace and snapshotted before N607 sync.

| File | SHA256 |
|---|---|
| `E:\type10-7\code\scripts\fit_apply_phase1_leo_feature_adapter.py` | `94C12E73C0D15E11FB2AEC344C985B4B0ADDA8EF1343B9D1719152CCDA5A10A5` |
| `E:\type10-7\code\scripts\sweep_phase1_adv3b02_leo_feature_adapter_20260703.sh` | `79279146CAE5BF6276A81F806C02779AFCF40285FCB7A536146AD0A6D1079D31` |
| `E:\type10-7\code\scripts\launch_phase1_adv3b02_leo_feature_adapter_shards_20260703.sh` | `BD08226DCE960D04B74339B72496701C4EF0C64C9F385FDE0EA0DDD4167A66A3` |
| `E:\type10-7\code\scripts\sweep_phase1_adv3b02_leo_feature_adapter_v2_20260703.sh` | `6AF54B5C1D35119CEF840DB4DF50707FEA6FB096DF86D20F2425E2EA4A5459AD` |

## Planned N607 Matrix

| Field | Value |
|---|---|
| Remote root | `/home/szu2070436088/2510044040/CV-SincNet` |
| Remote script | `/home/szu2070436088/2510044040/CV-SincNet/code/scripts/sweep_phase1_adv3b02_leo_feature_adapter_20260703.sh` |
| Remote shard launcher | `/home/szu2070436088/2510044040/CV-SincNet/code/scripts/launch_phase1_adv3b02_leo_feature_adapter_shards_20260703.sh` |
| Matrix log root | `/home/szu2070436088/2510044040/CV-SincNet/logs/phase1_adv3b02_leo_feature_adapter_matrix_20260703` |
| Expected summary | `/home/szu2070436088/2510044040/CV-SincNet/logs/phase1_adv3b02_leo_feature_adapter_matrix_20260703/leo_feature_adapter_summary.csv` |
| Expected rows | 350 rows: 10 cells x 5 adapter variants x 7 rejection policies |
| GPU allocation | Up to 8 bounded shard processes: `CELL_SHARD_INDEX=0..7`, `CELL_SHARD_COUNT=8`, `GPU=0..7`; adapter fitting uses each shard's `CUDA_VISIBLE_DEVICES` |
| Launch command | `cd /home/szu2070436088/2510044040/CV-SincNet && bash code/scripts/launch_phase1_adv3b02_leo_feature_adapter_shards_20260703.sh` |

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

## Launch Attempts

| Time | Status | Detail |
|---|---|---|
| `2026-07-03T00:30+08:00` | FAILED_STARTUP | Manual inline launch command had shell escaping errors; it wrote `driver_shard.out` with empty shard/GPU variables and did not create a valid matrix run. |
| `2026-07-03T00:34+08:00` | FAILED_STARTUP | Dedicated shard launcher created PIDs`864506-864513`, but the first adapter fit failed on N607 because tensor-to-NumPy conversion produced an object array under the remote PyTorch/NumPy combination. |
| `2026-07-03T00:40+08:00` | PATCHED_LOCAL | `fit_apply_phase1_leo_feature_adapter.py` now computes logits directly from the adapted tensor and converts saved arrays through `.tolist()` to guarantee float32 NPZ output. |
| `2026-07-03T00:38-00:50+08:00` | COMPLETED_V1 | 8 shards completed, 50 adapter metrics and 350 rejection metrics produced. |
| `2026-07-03T00:58-01:08+08:00` | COMPLETED_V2 | 8 shards completed, 40 adapter metrics and 280 rejection metrics produced. |

## V1 Completion Result

Artifacts:

| Artifact | Local path | SHA256 |
|---|---|---|
| V1 summary CSV | `E:\type10-7\automation_reports\CV-SincNet\phase1_adv3b02_leo_feature_adapter_20260703\artifacts\leo_feature_adapter_summary.csv` | `76684A510D0C5EA5CCD667DD68BF7AE46D421C093D0D6E172E064DF290C9567B` |
| V1 shard drivers | `E:\type10-7\automation_reports\CV-SincNet\phase1_adv3b02_leo_feature_adapter_20260703\artifacts\driver_shard0.out` ... `driver_shard7.out` | pulled locally |

Overall:

| Metric | Value |
|---|---:|
| Rows | 350 |
| Adapter runs | 50 |
| Cells | 10 |
| Dual pass (`unknown_FAR<=0.05` and old drop`<=2pp`) | 0 |
| FAR-only pass | 66 |
| Old-drop-only pass | 134 |

Target1 alignment by adapter:

| Adapter | Source pairs/cell | Val MSE before | Val MSE after | Val cosine before | Val cosine after | Val proto acc before | Val proto acc after |
|---|---:|---:|---:|---:|---:|---:|---:|
| `LEOADAPT_IDENTITY` | 27 | 0.1544 | 0.1544 | 0.9064 | 0.9064 | 0.8333 | 0.8333 |
| `LEOADAPT_AFFINE` | 27 | 0.1544 | 0.1172 | 0.9064 | 0.7239 | 0.8333 | 0.5000 |
| `LEOADAPT_MLP128_CE` | 27 | 0.1544 | 0.1215 | 0.9064 | 0.6869 | 0.8333 | 0.5000 |
| `LEOADAPT_LINR_COS` | 27 | 0.1544 | 0.1225 | 0.9064 | 0.7386 | 0.8333 | 0.5000 |
| `LEOADAPT_MLP64_BAL` | 27 | 0.1544 | 0.1236 | 0.9064 | 0.6453 | 0.8333 | 0.5000 |

Interpretation: V1 improves Euclidean/MSE alignment but damages the identity direction. It makes the adapted feature numerically closer to clean while making cosine alignment and source prototype identity worse. Therefore V1 does not satisfy the real target1 intent of “LEO样本特征和clean样本特征尽量一致” in an identity-preserving sense.

Target2 summary:

| Family | Mean unknown_FAR | Mean old drop pp | Failure mode |
|---|---:|---:|---|
| Low-FAR prototype/min-threshold rows | 0.038-0.065 | 53-60pp | FAR can be reduced only by rejecting most old-class queries. |
| Old-retention rows | 0.88-0.99 | 0-2pp | Old performance is retained but unknown samples are mostly accepted. |

Conclusion: V1 is a useful negative result. It confirms that a naive feature repair adapter can make the feature MSE look better while destroying the identity geometry needed by open-set rejection.

## V2 Follow-up Design

V2 keeps the same no-target-clean/no-target-label boundary but changes target1 training:

| Change | Reason |
|---|---|
| Train on source`clean.npz` paired with source`sat_clear.npz`, `sat_low.npz`, and`sat_rain.npz` | V1 trained on only the hidden single-observation source subset, giving only 27 source pairs/cell. |
| Add `mean_shift` and `norm_mean_shift` adapters | Test conservative global LEO residual repair without high-capacity distortion of identity direction. |
| Retain a lower-alpha identity-weighted linear residual | Keep a trainable option but reduce drift from source identity prototypes. |
| Apply only to strict satellite single-observation test NPZ | No target clean view enters evaluation. |

Planned V2 matrix:

| Field | Value |
|---|---|
| Remote script | `/home/szu2070436088/2510044040/CV-SincNet/code/scripts/sweep_phase1_adv3b02_leo_feature_adapter_v2_20260703.sh` |
| Matrix log root | `/home/szu2070436088/2510044040/CV-SincNet/logs/phase1_adv3b02_leo_feature_adapter_v2_matrix_20260703` |
| Expected summary | `/home/szu2070436088/2510044040/CV-SincNet/logs/phase1_adv3b02_leo_feature_adapter_v2_matrix_20260703/leo_feature_adapter_v2_summary.csv` |
| Expected rows | 280 rows: 10 cells x 4 adapters x 7 rejection policies |

## V2 Completion Result

Artifacts:

| Artifact | Local path | SHA256 |
|---|---|---|
| V2 summary CSV | `E:\type10-7\automation_reports\CV-SincNet\phase1_adv3b02_leo_feature_adapter_20260703\artifacts\leo_feature_adapter_v2_summary.csv` | `E5B470605D9539A9E02567CB42C3CDE32711D624F77F524B6083F29C3CA37470` |
| V2 shard drivers | `E:\type10-7\automation_reports\CV-SincNet\phase1_adv3b02_leo_feature_adapter_20260703\artifacts\driver_v2_shard0.out` ... `driver_v2_shard7.out` | pulled locally |

Overall:

| Metric | Value |
|---|---:|
| Rows | 280 |
| Adapter runs | 40 |
| Cells | 10 |
| Dual pass (`unknown_FAR<=0.05` and old drop`<=2pp`) | 0 |
| FAR-only pass | 62 |
| Old-drop-only pass | 104 |

Target1 alignment by adapter:

| Adapter | Source pairs/cell | Val MSE before | Val MSE after | Val cosine before | Val cosine after | Val proto acc before | Val proto acc after |
|---|---:|---:|---:|---:|---:|---:|---:|
| `LEOADAPT2_IDENTITY` | 27 | 0.1399 | 0.1399 | 0.8229 | 0.8229 | 0.8000 | 0.8000 |
| `LEOADAPT2_MEANSHIFT` | 27 | 0.1399 | 0.1245 | 0.8229 | 0.7573 | 0.8000 | 0.8000 |
| `LEOADAPT2_LINR_ID` | 27 | 0.1399 | 0.0992 | 0.8229 | 0.6909 | 0.8000 | 0.8000 |
| `LEOADAPT2_NORMSHIFT` | 27 | 0.1399 | 0.2522 | 0.8229 | 0.7889 | 0.8000 | 0.8000 |

V2 fixes the V1 identity-prototype collapse: prototype accuracy stays at`0.8000` for all V2 adapters. However, cosine still drops for all non-identity adapters. `mean_shift` is the best conservative repair: it improves MSE from`0.1399` to`0.1245` while preserving prototype accuracy, but it still weakens cosine from`0.8229` to`0.7573`.

Target2 result:

| Best family | Mean unknown_FAR | Max unknown_FAR | Mean old drop pp | Max old drop pp | Verdict |
|---|---:|---:|---:|---:|---|
| `LEOADAPT2_MEANSHIFT` + `ADAPT2_PROTO_MAH_MIN05` | 0.0281 | 0.0425 | 56.07 | 66.24 | FAR passes, old-class retention fails badly |
| `LEOADAPT2_MEANSHIFT` + `ADAPT2_PROTO_COS_MIN05` | 0.0308 | 0.0500 | 53.43 | 63.12 | FAR nearly/passes, old-class retention fails badly |
| `LEOADAPT2_MEANSHIFT` + `ADAPT2_MLP64_SRC9999` | 0.8888 | 0.9611 | 1.77 | 4.38 | Old retention near target, FAR fails badly |

Nearest same-row candidates remain far from the dual target:

| Run | Adapter | Reject policy | unknown_FAR | Old drop pp | Closed old acc | Full old acc after reject | Verdict |
|---|---|---|---:|---:|---:|---:|---|
| `rx20_1_u1` | `LEOADAPT2_MEANSHIFT` | `ADAPT2_MLP64_SRC9999` | 0.7759 | 1.76 | 0.5591 | 0.5415 | Old retention passes but FAR fails |
| `rx20_1_u1` | `LEOADAPT2_MEANSHIFT` | `ADAPT2_MLP64_MIN05` | 0.0364 | 35.68 | 0.5591 | 0.2024 | FAR passes but old retention fails |
| `rx3_19_u10` | `LEOADAPT2_MEANSHIFT` | `ADAPT2_PROTO_COS_MIN05` | 0.0000 | 48.06 | 0.4882 | 0.0076 | FAR passes but almost all old-correct samples are rejected |

## Final Interpretation

目标1已经实现为可复现实验工具链，但当前source-only后置feature repair没有达到可用质量。V2比V1更合理：它避免了prototype身份崩塌，并证明保守全局LEO残差修补能小幅降低MSE；但它仍然降低余弦方向一致性，且不能改善目标2的old/unknown可分性。

目标2仍未达成。两个矩阵都显示同一个机制冲突：

| If threshold is strict enough for `unknown_FAR<=5%` | Old-class accuracy drops by roughly 35-66pp |
|---|---|
| If old drop is kept near`<=2pp` | `unknown_FAR` remains around 0.78-0.99 |

Conclusion: under the current available source-pair feature exports, a post-feature adapter trained only on sparse source clean/LEO pairs is insufficient. The next route should not continue small feature-space adapter sweeps. It should regenerate a real source paired dataset with many clean/LEO pairs per source TX/RX/day, then train either a raw-IQ前置去信道模块 or an identity-constrained feature adapter with explicit cosine/prototype-preservation validation before any rejection threshold is tuned.

## V3 Global Source-Pair Route

V3 responds to the V1/V2 root cause: both earlier matrices used only`27`paired source clean/LEO samples per cell, so the adapter training set was too sparse to represent the source-domain LEO residual. V3 first exports a real source-only clean/LEO feature library from`ManySig.pkl`, then trains one global adapter per satellite scenario pool and applies it to strict satellite single-observation target tests.

Protocol boundary:

| Boundary | V3 setting |
|---|---|
| Base model | `ADV3B02_CORE90_SOFT_E200phase1` frozen checkpoint |
| Adapter training data | source training TX/RX only, clean paired with generated source LEO views |
| Target evaluation data | existing strict satellite single-observation NPZ; no target clean |
| Unknown threshold selection | no target unknown query tuning |
| Phase2 few-shot | not used |

New/changed local files:

| File | Purpose | SHA256 |
|---|---|---|
| `E:\type10-7\code\scripts\export_phase1_source_leo_pair_features.py` | Export source-only clean and LEO paired Phase1 features from`ManySig.pkl` | `AA22974E400A606431AD97C4AA50D35FA8A5DF6D71F35C51D8D76311EC8C7BD7` |
| `E:\type10-7\code\scripts\sweep_phase1_adv3b02_global_source_leo_adapter_20260703.sh` | Run V3 source-pair export and global source-trained adapter matrix | `CA937A87E90774D838B9BAA069317FA0EAD7E06A33763D9DE1F9C48BFD1CA44C` |
| `E:\type10-7\code\scripts\fit_apply_phase1_leo_feature_adapter.py` | Existing adapter/evaluator reused by V3 | `94C12E73C0D15E11FB2AEC344C985B4B0ADDA8EF1343B9D1719152CCDA5A10A5` |

Local verification:

| Command | Result |
|---|---|
| `C:\Users\lh594\.conda\envs\ssr-gpu\python.exe -m py_compile E:\type10-7\code\scripts\export_phase1_source_leo_pair_features.py E:\type10-7\code\scripts\fit_apply_phase1_leo_feature_adapter.py` | PASS |
| `bash -n code/scripts/sweep_phase1_adv3b02_global_source_leo_adapter_20260703.sh` | PASS |
| `C:\Users\lh594\.conda\envs\ssr-gpu\python.exe E:\type10-7\code\scripts\export_phase1_source_leo_pair_features.py --help` | PASS |

Remote pre-launch verification/fix:

| Event | Result |
|---|---|
| N607 direct preflight | PASS; project root visible, 8 GPUs visible and idle |
| Initial remote export attempt | Failed before feature export: exporter lacked standard`build_model_from_ckpt`CLI fields such as`dataset` |
| Local fix | Added optional`dataset`, `num_classes`, `model_size`, `model_variant`, `branch_ablation`, and`sample_rate_hz`fields; checkpoint remains the default source of those values |
| Re-verification | `py_compile`PASS; `--help`shows compatibility fields |

Planned N607 execution:

| Stage | Command/log |
|---|---|
| Export source clean/LEO pairs | `DO_EXPORT=1 RUN_CELLS=0 GPU=0 bash code/scripts/sweep_phase1_adv3b02_global_source_leo_adapter_20260703.sh`; log root `/home/szu2070436088/2510044040/CV-SincNet/logs/phase1_adv3b02_global_source_leo_adapter_matrix_20260703` |
| Run target satellite cells | `SCRIPT=/home/szu2070436088/2510044040/CV-SincNet/code/scripts/sweep_phase1_adv3b02_global_source_leo_adapter_20260703.sh bash code/scripts/launch_phase1_adv3b02_leo_feature_adapter_shards_20260703.sh` |
| Expected summary | `/home/szu2070436088/2510044040/CV-SincNet/logs/phase1_adv3b02_global_source_leo_adapter_matrix_20260703/global_source_leo_adapter_summary.csv` |

Expected matrix:

| Dimension | Values |
|---|---|
| Cells | 10 strict satellite single-observation cells |
| Adapters | `LEOADAPT3_IDENTITY`, `LEOADAPT3_MEANSHIFT`, `LEOADAPT3_NORMSHIFT`, `LEOADAPT3_LINR_COS`, `LEOADAPT3_MLP_ID` |
| Reject policies | 4 head policies plus 3 prototype policies |
| Expected rows | 350 |

Success criteria stay unchanged: target1 requires improved LEO-to-clean alignment without identity/prototype collapse; target2 requires same-row`unknown_FAR<=0.05`and old-class performance drop`<=2pp`.

## V3 Completion Result

Artifacts:

| Artifact | Local path | SHA256 |
|---|---|---|
| V3 summary CSV | `E:\type10-7\automation_reports\CV-SincNet\phase1_adv3b02_leo_feature_adapter_20260703\artifacts\global_source_leo_adapter_summary.csv` | `BF84EA4D0F71E83CCD44A6484DCB7A18832E8B9AA0A7B0178492F1B96D3CBA5D` |
| Source pair manifest | `E:\type10-7\automation_reports\CV-SincNet\phase1_adv3b02_leo_feature_adapter_20260703\artifacts\source_leo_pair_manifest.json` | `9D582E4178B8A26036EE5FA38FD8DAC4C37E33055F8CBA37220050AACE6810DE` |
| V3 driver logs | `E:\type10-7\automation_reports\CV-SincNet\phase1_adv3b02_leo_feature_adapter_20260703\artifacts\driver_shard0.out` ... `driver_shard7.out` | pulled locally |

Source pair export:

| Field | Value |
|---|---:|
| Source clean samples | 9600 |
| Source LEO scenario files | 3 |
| Effective adapter source pairs | 28800 |
| Target clean used | 0 |
| Target label/support used | 0 |
| Unknown query used for training | 0 |

Overall V3 target2 result:

| Metric | Value |
|---|---:|
| Rows | 350 |
| Adapter runs | 50 |
| Cells | 10 |
| Dual pass (`unknown_FAR<=0.05` and old drop`<=2pp`) | 0 |
| FAR-only pass | 85 |
| Old-drop-only pass | 132 |

Target1 alignment by adapter:

| Adapter | Source pairs/cell | Val MSE before | Val MSE after | Val cosine before | Val cosine after | Val proto acc before | Val proto acc after | Verdict |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| `LEOADAPT3_IDENTITY` | 28800 | 0.1566 | 0.1566 | 0.8573 | 0.8573 | 0.8690 | 0.8690 | Baseline |
| `LEOADAPT3_LINR_COS` | 28800 | 0.1566 | 0.1127 | 0.8573 | 0.8558 | 0.8690 | 0.8795 | Target1 pass: MSE improves, cosine preserved, proto acc improves |
| `LEOADAPT3_MLP_ID` | 28800 | 0.1566 | 0.1044 | 0.8573 | 0.8508 | 0.8690 | 0.8807 | Target1 pass: strongest MSE, small cosine drop, proto acc improves |
| `LEOADAPT3_MEANSHIFT` | 28800 | 0.1566 | 0.1360 | 0.8573 | 0.7906 | 0.8690 | 0.7921 | Not acceptable: identity geometry drops |
| `LEOADAPT3_NORMSHIFT` | 28800 | 0.1566 | 0.1853 | 0.8573 | 0.8208 | 0.8690 | 0.8178 | Worse than baseline |

V3 finally satisfies目标1 for the two identity-constrained trainable adapters. The key change was not the adapter form alone; it was replacing the sparse`27`pair source view with a real`28800`pair source clean/LEO library. `LEOADAPT3_LINR_COS` and`LEOADAPT3_MLP_ID` both repair feature MSE without collapsing prototype identity.

Target2 still fails. The best low-FAR rows still reject too many old-class samples:

| Run | Adapter | Reject policy | unknown_FAR | Old drop pp | Closed old acc | Full old acc after reject | Verdict |
|---|---|---|---:|---:|---:|---:|---|
| `rx20_1_u1` | `LEOADAPT3_MEANSHIFT` | `ADAPT3_MLP64_MIN05` | 0.0420 | 29.68 | 0.4809 | 0.1841 | FAR passes, old retention fails |
| `rx8_8_u10` | `LEOADAPT3_MEANSHIFT` | `ADAPT3_LIN_MIN05` | 0.0486 | 31.12 | 0.5968 | 0.2856 | FAR passes, old retention fails |
| `rx20_1_u1` | `LEOADAPT3_MLP_ID` | `ADAPT3_MLP64_MIN05` | 0.0364 | 35.35 | 0.5412 | 0.1876 | FAR passes, old retention fails |

The best old-retention rows still accept most unknown samples:

| Run | Adapter | Reject policy | unknown_FAR | Old drop pp | Closed old acc | Full old acc after reject | Verdict |
|---|---|---|---:|---:|---:|---:|---|
| `rx20_1_u1` | `LEOADAPT3_MEANSHIFT` | `ADAPT3_MLP64_SRC9999` | 0.7395 | 1.88 | 0.4809 | 0.4621 | Old retention passes, FAR fails |
| `rx8_8_u1` | `LEOADAPT3_MLP_ID` | `ADAPT3_MLP64_SRC9999` | 0.8275 | 1.00 | 0.7268 | 0.7168 | Old retention passes, FAR fails |
| `rx8_8_u1` | `LEOADAPT3_IDENTITY` | `ADAPT3_MLP64_SRC9999` | 0.8325 | 0.47 | 0.7188 | 0.7141 | Old retention passes, FAR fails |

V3 interpretation:

| Claim | Status |
|---|---|
| Target1 source-only LEO feature repair | Achieved by`LEOADAPT3_LINR_COS` and`LEOADAPT3_MLP_ID` |
| Target2 unknown rejection with old drop`<=2pp` | Not achieved |
| Main remaining failure | Rejection score overlap, not feature repair alone |

Next experiment should keep the V3 repaired features and change only the rejection decision. The current threshold families are one-dimensional and still show the same tradeoff: strict thresholds reach`unknown_FAR<=5%`only by rejecting 29-60pp old accuracy, while old-preserving thresholds keep FAR around 0.74-0.99.

## V4 Score-Table Fusion Design

V4 keeps all V3 repaired feature files and trained rejection score tables fixed. It does not retrain the base model, adapter, or rejection heads. It only tests whether source/proxy-calibrated fusion of multiple rejection scores can preserve old-class accuracy while reducing unknown acceptance.

New local file:

| File | Purpose | SHA256 |
|---|---|---|
| `E:\type10-7\code\scripts\sweep_phase1_score_table_fusion_20260703.py` | Fuse V3 score tables from linear, MLP, prototype-cosine, and prototype-Mahalanobis scores using source/proxy calibration | `1A9B56F1E095F37E9F06806CFB014C55A78F61DFFDEA93F06075C4F956F457E6` |

Local verification:

| Command | Result |
|---|---|
| `C:\Users\lh594\.conda\envs\ssr-gpu\python.exe -m py_compile E:\type10-7\code\scripts\sweep_phase1_score_table_fusion_20260703.py` | PASS |

Remote compatibility note: the first V4 attempt failed because the remote`CVS-RFFI`environment has no`pandas`. The script was revised to use only standard`csv`plus`numpy`, then locally recompiled.

Planned V4 matrix:

| Dimension | Values |
|---|---|
| Input | Existing V3 score tables only |
| Adapters | `LEOADAPT3_IDENTITY`, `LEOADAPT3_LINR_COS`, `LEOADAPT3_MLP_ID` |
| Component sets | `lin_mlp`, `lin_pcos`, `mlp_pcos`, `mlp_pmah`, `lin_mlp_pcos`, `lin_mlp_pmah`, `all4` |
| Fusion methods | `max`, `mean`, `min`, `top2mean` |
| Threshold policies | `source_accept`, `min_source_proxy`, `mean_source_proxy` |
| Calibration | source/proxy rows only; target query labels not used for threshold calculation |

Expected output: `/home/szu2070436088/2510044040/CV-SincNet/logs/phase1_adv3b02_score_fusion_20260703/score_fusion_summary.csv`.

## V4 Execution Status

V4 did not produce a valid result table. Two remote attempts were launched:

| Attempt | Output | Status |
|---|---|---|
| Full fusion grid | `logs/phase1_adv3b02_score_fusion_20260703/score_fusion_summary.csv` | No CSV produced; process was CPU-bound with empty log |
| Narrow fusion grid | `logs/phase1_adv3b02_score_fusion_20260703/score_fusion_fast_summary.csv` | No CSV produced; process was CPU-bound with empty log |

Both processes were confirmed to be the agent-launched V4 diagnostic commands and were stopped by exact PID only:

| Remote PID | Command role |
|---:|---|
| 1039360 | full-grid Python process |
| 1039357 | full-grid wrapper shell |
| 1043733 | narrow-grid Python process |
| 1043730 | narrow-grid wrapper shell |

No other N607 training or evaluation process was killed. V4 is therefore an implementation-performance failure, not a scientific result. The current valid evidence for this task remains V3.

## V5 Scalar-Score Oracle Diagnostic Design

V5 is a `NON_DEPLOYMENT_DIAGNOSTIC`. It uses target query labels only to test whether any scalar score table already produced by V3 contains a threshold that could satisfy the dual target. It does not define a deployable threshold and must not be reported as source-only rejection success.

New local file:

| File | Purpose | SHA256 |
|---|---|---|
| `E:\type10-7\code\scripts\diagnose_phase1_score_oracle_20260703.py` | Scan V3 score tables and compute target-label oracle thresholds for each scalar rejection score | `F424D3AC3FFE59365D791A28B39E180732CF725FA4D3570D3905D29242FA5494` |

Local verification:

| Command | Result |
|---|---|
| `C:\Users\lh594\.conda\envs\ssr-gpu\python.exe -m py_compile E:\type10-7\code\scripts\diagnose_phase1_score_oracle_20260703.py` | PASS |

Planned remote output: `/home/szu2070436088/2510044040/CV-SincNet/logs/phase1_adv3b02_score_oracle_20260703/score_oracle_summary.csv`.

## V5 Completion Result

Artifacts:

| Artifact | Local path | SHA256 |
|---|---|---|
| V5 oracle summary | `E:\type10-7\automation_reports\CV-SincNet\phase1_adv3b02_leo_feature_adapter_20260703\artifacts\score_oracle_summary.csv` | `4351E865AF2CE1345BD43C0707668AF1467938D061736B13D7A9409C2331D497` |
| V5 oracle log | `E:\type10-7\automation_reports\CV-SincNet\phase1_adv3b02_leo_feature_adapter_20260703\artifacts\score_oracle.out` | pulled locally |

V5 scanned 210 scalar score tables from the existing V3 outputs:

| Metric | Value |
|---|---:|
| Score tables scanned | 210 |
| Target-label oracle dual pass | 0 |
| Best FAR under old drop`<=2pp` | 0.6531 |
| Best old drop under`unknown_FAR<=0.05` | 26.24pp |

Best oracle-nearest rows:

| Run | Adapter | Score | Oracle nearest unknown_FAR | Oracle nearest old drop pp | Closed old acc | Full old acc | Verdict |
|---|---|---|---:|---:|---:|---:|---|
| `rx20_1_u1` | `LEOADAPT3_MLP_ID` | `ADAPT3_LIN_MIN05`/`SRC9999` | 0.0924 | 21.74 | 0.5412 | 0.3238 | Still far from dual target |
| `rx20_1_u1` | `LEOADAPT3_LINR_COS` | `ADAPT3_LIN_MIN05`/`SRC9999` | 0.1092 | 20.41 | 0.5765 | 0.3724 | Still far from dual target |
| `rx20_1_u1` | `LEOADAPT3_LINR_COS` | `ADAPT3_MLP64_MIN05`/`SRC9999` | 0.1373 | 20.21 | 0.5765 | 0.3744 | Still far from dual target |

Diagnostic interpretation: V5 proves that the failure is not merely threshold calibration. For every existing scalar score in V3, even an invalid target-label oracle threshold cannot satisfy`unknown_FAR<=0.05`and old drop`<=2pp`. The next valid route must create a new rejection signal, not keep sweeping scalar thresholds from the same score tables.

## V6 Multi-Score Rejector Design

V6 tests whether multiple existing V3 scores can jointly create a better rejection signal. It has two modes:

| Mode | Training signal | Deployable? | Purpose |
|---|---|---|---|
| `source_proxy_train` | source old vs source proxy unknown only | Yes | Real candidate for target2 |
| `target_label_oracle_multiscore` | target closed-correct old vs target unknown | No | Diagnostic upper bound for multi-score separability |

New local file:

| File | Purpose | SHA256 |
|---|---|---|
| `E:\type10-7\code\scripts\eval_phase1_multiscore_reject_20260703.py` | Train/evaluate small linear/MLP rejectors over multiple V3 score tables with source/proxy calibration plus target-label oracle upper bound | `5D4FAB2FD571D16E0BB974C3DD4E5B4A0C5BA4446164C829DB5EE45D2B982CC1` |

Local verification:

| Command | Result |
|---|---|
| `C:\Users\lh594\.conda\envs\ssr-gpu\python.exe -m py_compile E:\type10-7\code\scripts\eval_phase1_multiscore_reject_20260703.py` | PASS |

Planned remote output: `/home/szu2070436088/2510044040/CV-SincNet/logs/phase1_adv3b02_multiscore_reject_20260703/multiscore_reject_summary.csv`.

Remote startup fix: the first V6 run failed before producing a result because the remote PyTorch/NumPy combination returned an incompatible array from`tensor.numpy()`. The script was patched to convert scores through`tolist()` before building a NumPy array.

## V6 Completion Result

Artifacts:

| Artifact | Local path | SHA256 |
|---|---|---|
| V6 multi-score summary | `E:\type10-7\automation_reports\CV-SincNet\phase1_adv3b02_leo_feature_adapter_20260703\artifacts\multiscore_reject_summary.csv` | `11D4105C327E368595C1659790B68194912247F694A10E31F18B3153584D64CA` |
| V6 log | `E:\type10-7\automation_reports\CV-SincNet\phase1_adv3b02_leo_feature_adapter_20260703\artifacts\multiscore_reject.out` | pulled locally |

V6 result:

| Mode | Rows | Dual pass | Best nearest row |
|---|---:|---:|---|
| `source_proxy_train` | 2880 | 0 | `rx20_1_u1` + `LEOADAPT3_MLP_ID` + `all4` + linear: FAR 0.0924, old drop 22.76pp |
| `target_label_oracle_multiscore` | 80 | 0 | `rx20_1_u1` + `LEOADAPT3_MLP_ID` + `lin_mlp_pmah` + linear: FAR 0.0784, old drop 34.00pp |

Best source/proxy-trained multi-score rows:

| Run | Adapter | Components | Model | unknown_FAR | Old drop pp | Verdict |
|---|---|---|---|---:|---:|---|
| `rx20_1_u1` | `LEOADAPT3_MLP_ID` | `all4` | linear | 0.0924 | 22.76 | Still far from dual target |
| `rx20_1_u1` | `LEOADAPT3_LINR_COS` | `all4` | linear | 0.1261 | 19.88 | Still far from dual target |
| `rx20_1_u1` | `LEOADAPT3_MLP_ID` | `lin_mlp_pmah` | linear | 0.0952 | 22.97 | Still far from dual target |

V6 interpretation: combining existing scalar score tables is still insufficient. Even the invalid target-label multi-score oracle cannot satisfy the dual target. The next route must introduce a new rejection signal not present in the current score tables, such as repair residual, before/after feature movement, prototype-rank stability, or class-change consistency between raw satellite features and repaired features.

## V7 Repair-Delta Rejector Design

V7 introduces a new rejection signal not present in V3-V6 score tables. It reads the original strict satellite feature NPZ and the repaired feature NPZ, then builds before/after repair features:

| Signal family | Examples |
|---|---|
| Repair magnitude | residual norm, residual/feature norm ratio, before-after cosine |
| Prototype movement | distance/cosine to predicted source prototype before and after repair |
| Logit/probability movement | confidence, entropy, margin before/after repair, probability-change norm |
| Prediction stability | whether closed-set predicted TX changes after repair |

New local file:

| File | Purpose | SHA256 |
|---|---|---|
| `E:\type10-7\code\scripts\eval_phase1_repair_delta_reject_20260703.py` | Train/evaluate source/proxy and target-label oracle rejectors over repair-delta features from raw satellite and repaired V3 NPZ files | `DA6762371AEDD9CA9C8A3EB0CACC4C148089C64AD0013B6B8EE2F1FE056B3DBF` |

Local verification:

| Command | Result |
|---|---|
| `C:\Users\lh594\.conda\envs\ssr-gpu\python.exe -m py_compile E:\type10-7\code\scripts\eval_phase1_repair_delta_reject_20260703.py` | PASS |

Planned remote output: `/home/szu2070436088/2510044040/CV-SincNet/logs/phase1_adv3b02_repair_delta_reject_20260703/repair_delta_reject_summary.csv`.

## V7 Completion Result

Artifacts:

| Artifact | Local path | SHA256 |
|---|---|---|
| V7 repair-delta summary | `E:\type10-7\automation_reports\CV-SincNet\phase1_adv3b02_leo_feature_adapter_20260703\artifacts\repair_delta_reject_summary.csv` | `2143452DE510B041BEF7A68C90823741871D0E2DCF4B858F30A559D9562AD338` |
| V7 log | `E:\type10-7\automation_reports\CV-SincNet\phase1_adv3b02_leo_feature_adapter_20260703\artifacts\repair_delta_reject.out` | pulled locally |

V7 result:

| Mode | Rows | Dual pass | Best nearest row |
|---|---:|---:|---|
| `source_proxy_train` | 1440 | 0 | `rx20_1_u1` + `LEOADAPT3_MLP_ID` + linear: FAR 0.1148, old drop 19.50pp |
| `target_label_oracle_repair_delta` | 40 | 0 | `rx20_1_u1` + `LEOADAPT3_MLP_ID` + linear: FAR 0.1877, old drop 11.59pp |

Best source/proxy-trained repair-delta rows:

| Run | Adapter | Model | unknown_FAR | Old drop pp | Verdict |
|---|---|---|---:|---:|---|
| `rx20_1_u1` | `LEOADAPT3_MLP_ID` | linear | 0.1148 | 19.50 | Still far from dual target |
| `rx20_1_u1` | `LEOADAPT3_MLP_ID` | linear | 0.0924 | 21.76 | Still far from dual target |
| `rx20_1_u1` | `LEOADAPT3_MLP_ID` | linear | 0.0896 | 22.21 | Still far from dual target |

V7 interpretation: repair-delta signals improve neither deployable rejection nor the target-label oracle enough to satisfy目标2. Combined with V5 and V6, this shows that the current Phase1 base plus post-hoc rejection surfaces do not contain a sufficient separation signal for strict sat-only unknown rejection under`old drop<=2pp`.

Current route conclusion:

| Target | Status | Evidence |
|---|---|---|
| 目标1: source-only LEO feature repair | Achieved | V3 `LEOADAPT3_LINR_COS` and `LEOADAPT3_MLP_ID` improve MSE while preserving cosine/prototype identity |
| 目标2: `unknown_FAR<=0.05` and old drop`<=2pp` | Not achieved | V3 full matrix, V5 scalar oracle, V6 multi-score oracle, and V7 repair-delta oracle all have dual pass 0 |

Next technically aligned route: stop adding post-hoc thresholds on this frozen score surface. The next experiment should modify training/evaluation features themselves, for example by adding source-side open-set negatives during Phase1-compatible adapter training, preserving pre/post residual channels in the feature NPZ, or training a source-only open-set head directly on repaired features with explicit class-conditional old retention loss and proxy unknown separation. This remains within the phase1-only boundary if it uses source old/source proxy unknown and no target labels.

## V8 K+1 Open-Set Head Design

V8 trains a source-only open-set classifier directly on V3 repaired features. Instead of binary reject/accept, it learns`K+1`classes: six source old TX classes plus one source proxy unknown class. This changes the rejection signal itself while keeping the Phase1 backbone frozen and using no target labels for training.

New local file:

| File | Purpose | SHA256 |
|---|---|---|
| `E:\type10-7\code\scripts\eval_phase1_kplus1_openset_reject_20260703.py` | Train/evaluate source-only K+1 open-set heads on repaired features with source/proxy threshold calibration | `8F19197472B9F506F98729BA04CF3151B400536CBBE47E83D1A07B568C02A711` |

Local verification:

| Command | Result |
|---|---|
| `C:\Users\lh594\.conda\envs\ssr-gpu\python.exe -m py_compile E:\type10-7\code\scripts\eval_phase1_kplus1_openset_reject_20260703.py` | PASS |

Planned remote output: `/home/szu2070436088/2510044040/CV-SincNet/logs/phase1_adv3b02_kplus1_reject_20260703/kplus1_reject_summary.csv`.

## V8 Completion Result

Remote verification and run command:

| Item | Value |
|---|---|
| N607 preflight | PASS, direct`N607`, project root visible, 8xRTX3090 idle at preflight |
| Remote cwd | `/home/szu2070436088/2510044040/CV-SincNet` |
| Remote Python | `/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python` |
| Remote command | `python -m py_compile code/scripts/eval_phase1_kplus1_openset_reject_20260703.py && python -u code/scripts/eval_phase1_kplus1_openset_reject_20260703.py --runs_root runs --out_csv logs/phase1_adv3b02_kplus1_reject_20260703/kplus1_reject_summary.csv --epochs 220` |
| Remote output | `{'rows': 1440, 'dual_pass': 0, 'out_csv': 'logs/phase1_adv3b02_kplus1_reject_20260703/kplus1_reject_summary.csv'}` |
| Local SSH cleanup | No local`ssh.exe`process and no ESTABLISHED connection to`172.31.111.215:22`after sync/run/pull |

Artifacts:

| Artifact | Local path | SHA256 |
|---|---|---|
| V8 K+1 summary | `E:\type10-7\automation_reports\CV-SincNet\phase1_adv3b02_leo_feature_adapter_20260703\artifacts\kplus1_reject_summary.csv` | `9EE018602209BC83E2BF46FA41FC4DDBBE0F8973CE0AADE963BD7DC75029B479` |
| V8 log | `E:\type10-7\automation_reports\CV-SincNet\phase1_adv3b02_leo_feature_adapter_20260703\artifacts\kplus1_reject.out` | `4A38E4A5353D0AEFCD0E2CDF4B2047761905E46644F428D292AA5EBFA72A3B11` |

V8 result summary:

| Scope | Rows | Dual pass | Interpretation |
|---|---:|---:|---|
| Source-only K+1 open-set head over V3 repaired features | 1440 | 0 | No candidate satisfies`unknown_FAR<=0.05`and reject-induced old drop`<=2pp` |

Best rows under the two constraints:

| Selection rule | Adapter | Head | Threshold policy | unknown_FAR | Old drop pp vs K+1 closed | K+1 closed old acc | Final old acc after reject | Verdict |
|---|---|---|---|---:|---:|---:|---:|---|
| Best FAR with old drop`<=2pp` | `LEOADAPT3_LINR_COS` | linear | `mean_source_proxy` | 0.7500 | 1.94 | 0.7271 | 0.7076 | FAR far above 5% |
| Best old retention with FAR`<=5%` | `LEOADAPT3_LINR_COS` | mlp | `min_source_proxy` | 0.0448 | 34.32 | 0.5765 | 0.2332 | Old-class performance collapses |
| Nearest joint row | `LEOADAPT3_LINR_COS` | mlp | `mean_source_proxy` | 0.1961 | 15.26 | 0.5765 | 0.4238 | Still misses both constraints |

Policy-level diagnostics:

| Condition | Rows | Best available behavior |
|---|---:|---|
| `unknown_FAR<=0.05` | 128 | Minimum old drop remains 34.32pp; final old acc no higher than 0.2929 in these rows |
| Reject-induced old drop`<=2pp` | 24 | Minimum unknown_FAR remains 0.7500 |

Important metric boundary: V8's`old_drop_pp_vs_closed`measures rejection damage relative to the K+1 closed head, not relative to the original Phase1/V3 old-class baseline. Therefore the stricter user requirement of keeping original old-class performance is not met either; the low-FAR rows reduce final old acc to roughly 0.23-0.29, and the best old-retention rows still keep FAR around 0.75-0.90.

Final route conclusion after V8:

| Target | Status | Evidence |
|---|---|---|
| 目标1: source-only LEO feature repair | Achieved | V3 repaired features improve MSE and preserve prototype identity without target labels |
| 目标2: `unknown_FAR<=0.05` while keeping old-class performance | Not achieved | V3, V5, V6, V7 and V8 all have dual pass 0; V8 confirms that simply adding a source-only K+1 unknown class on repaired features does not resolve the old/unknown overlap |

Next route should no longer be another threshold-only or shallow head sweep on the same exported feature table. The remaining protocol-aligned options are to retrain the Phase1-compatible repair/open-set module with an explicit class-conditional retention objective, or to regenerate source-side LEO repair views with stronger identity-preserving constraints before exporting features. Any target-label oracle or target-statistics calibration can only be marked diagnostic, not deployable phase1-only evidence.

## V9 Joint Adapter + Oldness Gate Design

V9 moves beyond post-hoc threshold/head sweeps by jointly training a Phase1-compatible feature adapter and an oldness gate. It still freezes the Phase1 backbone and obeys the phase1-only boundary.

Training data:

| Source | Role | Use |
|---|---|---|
| `source_clean.npz` | source old clean | clean target for repair and clean prototype bank |
| `source_leo_clear_weak.npz`, `source_leo_low_elev_weak.npz`, `source_leo_rain_weak.npz` | source old LEO | adapter input for clean/LEO feature repair |
| `proxy_unknown` rows from sat-only feature NPZ | source receiver non-old LEO | open-set oldness negative class |

Losses:

| Loss | Purpose |
|---|---|
| SmoothL1 + cosine pair repair | make source LEO features approach clean features |
| prototype CE to clean old prototypes | preserve old TX identity after repair |
| small residual penalty | avoid unconstrained feature drift |
| oldness BCE | separate source old from source proxy unknown |
| proxy prototype cap | push proxy unknown away from old clean prototypes |
| old confidence floor | keep old-class evidence high |

New local file:

| File | Purpose | SHA256 |
|---|---|---|
| `E:\type10-7\code\scripts\eval_phase1_joint_adapter_energy_reject_20260703.py` | Train/evaluate source-only joint adapter + oldness gate over sat-only target query NPZs, including global and class-conditional threshold policies | `7605211132074A61D8A7C319AE853DC0A119D8A7A8D7547945996FCCF891A839` |

Local verification:

| Command | Result |
|---|---|
| `C:\Users\lh594\.conda\envs\ssr-gpu\python.exe -m py_compile E:\type10-7\code\scripts\eval_phase1_joint_adapter_energy_reject_20260703.py` | PASS |

Planned remote output:

| Artifact | Remote path |
|---|---|
| V9 summary | `/home/szu2070436088/2510044040/CV-SincNet/logs/phase1_adv3b02_joint_adapter_reject_20260703/joint_adapter_reject_summary.csv` |
| V9 metrics JSON | `/home/szu2070436088/2510044040/CV-SincNet/logs/phase1_adv3b02_joint_adapter_reject_20260703/joint_adapter_reject_metrics.json` |
| V9 log | `/home/szu2070436088/2510044040/CV-SincNet/logs/phase1_adv3b02_joint_adapter_reject_20260703/joint_adapter_reject.out` |

Success criteria remain unchanged: target sat-only unknown_FAR`<=0.05`and old-class performance drop`<=2pp`, with no target clean and no target labels in training or threshold calibration.

## V9 Completion Results

Three V9 variants were executed on N607:

| Variant | Intent | Rows | Dual pass | Target1 repair status | Target2 status |
|---|---|---:|---:|---|---|
| `V9_joint_open` | stronger oldness/proxy separation | 2080 | 0 | Failed: val MSE 0.1571 -> 0.3235, cos 0.8623 -> 0.7086 | Failed |
| `V9b_retention_balanced` | reduce open-set loss and preserve LEO repair | 2080 | 0 | Pass: val MSE 0.1571 -> 0.1223, cos 0.8623 -> 0.8528, proto acc 0.8741 -> 0.8858 | Failed |
| `V9b_class_threshold` | same V9b model with class-conditional source/proxy thresholds | 4640 | 0 | Same as V9b | Failed |

Artifacts:

| Artifact | Local path | SHA256 |
|---|---|---|
| V9 summary | `E:\type10-7\automation_reports\CV-SincNet\phase1_adv3b02_leo_feature_adapter_20260703\artifacts\joint_adapter_reject_summary.csv` | `59BE74BFB42AEC04940B27124468369E785A240A56542473092F24CED996E2ED` |
| V9 metrics | `E:\type10-7\automation_reports\CV-SincNet\phase1_adv3b02_leo_feature_adapter_20260703\artifacts\joint_adapter_reject_metrics.json` | `B4EFCEF45B4E726785DF0C7BB7A7AD849E85B70E972AFF956E2AB594FF1E6E57` |
| V9b summary | `E:\type10-7\automation_reports\CV-SincNet\phase1_adv3b02_leo_feature_adapter_20260703\artifacts\joint_adapter_reject_v9b_summary.csv` | `4B0C49837F17E159C24CC84F3B43C82C946E0C4D573096B8C05F79DCCE105FCA` |
| V9b metrics | `E:\type10-7\automation_reports\CV-SincNet\phase1_adv3b02_leo_feature_adapter_20260703\artifacts\joint_adapter_reject_v9b_metrics.json` | `EC59DCDEF662532B0C0EE4C28239751222060D73E76BCA66E1E29ED899E8F43E` |
| V9b class-threshold summary | `E:\type10-7\automation_reports\CV-SincNet\phase1_adv3b02_leo_feature_adapter_20260703\artifacts\joint_adapter_reject_v9b_class_summary.csv` | `9BB005315C231C59078B68899ACC73FFBC98F86CD973A4703F8838266811FD7D` |
| V9b class-threshold metrics | `E:\type10-7\automation_reports\CV-SincNet\phase1_adv3b02_leo_feature_adapter_20260703\artifacts\joint_adapter_reject_v9b_class_metrics.json` | `61C6234640519E2B56CC0CF9579E9FAB0646CB8E2425BD3E013702692DB43406` |

Best same-row outcomes:

| Variant | Selection rule | Score/policy | unknown_FAR | Old drop pp | Final old acc | Verdict |
|---|---|---|---:|---:|---:|---|
| V9 | Best FAR with old drop`<=2pp` | `old_prob/source_accept` | 0.8207 | 1.74 | 0.5182 | FAR fails badly |
| V9 | Best old retention with FAR`<=5%` | `proto_max/proxy_far` | 0.0476 | 29.53 | 0.2403 | Old-class performance fails |
| V9b | Best FAR with old drop`<=2pp` | `old_prob/source_accept` | 0.6275 | 1.38 | 0.5279 | FAR fails badly |
| V9b | Best old retention with FAR`<=5%` | `old_prob/proxy_far` | 0.0336 | 33.32 | 0.2085 | Old-class performance fails |
| V9b class | Best class-policy low FAR | `old_prob/class_proxy_far` | 0.0112 | 33.65 | 0.4415 | Old-class performance fails |
| V9b class | Best class-policy old retention | `fused_rank/class_mean_source_proxy` | 0.1951 | 5.09 | 0.7650 | Both targets still fail |

Interpretation:

V9 confirms the suspected tradeoff. If the oldness/proxy loss is strong enough to move unknown_FAR toward the target, it destroys the repair objective and old-class retention. If the weights are adjusted so the adapter remains a valid LEO repair module, the old/proxy/target-unknown score overlap remains too large: source-calibrated thresholds either keep old classes and accept most unknowns, or reject unknowns and remove roughly one third of old-class performance.

Current conclusion after V9/V9b/V9b-class:

| Target | Status | Evidence |
|---|---|---|
| 目标1: source-only LEO repair | Achieved by V3 and preserved by V9b | V9b improves val MSE 0.1571 -> 0.1223 and proto acc 0.8741 -> 0.8858 without target clean/labels |
| 目标2: `unknown_FAR<=0.05` with old performance drop`<=2pp` | Not achieved | V3-V9 all have dual pass 0; V9b-class still has either FAR 0.6275 under old retention or old drop 33.65pp under low FAR |

Next route should change the evidence source rather than the threshold shape: the Phase1 feature space from `ADV3B02_CORE90_SOFT_E200phase1` does not currently expose a deployable source-only separation between target old and target unknown under LEO. The next aligned experiment is a true Phase1-side representation repair/retraining route, e.g. adding source-side non-old/open-set negatives and old-retention constraints during Phase1 training or during feature export, not another post-export gate.

## V10 Phase2-B Old-Class K-Shot Calibration Design

V10 changes the evidence boundary deliberately: it is no longer a zero-label/source-only rejection test. It is a Phase2-B-style diagnostic using target receiver old-class K-shot support, while still freezing the Phase1 base and using sat-only target features. It is included because V3-V9 showed that source-only post-export gates cannot separate target old from target unknown under LEO without losing old performance.

Protocol boundary:

| Item | V10 setting |
|---|---|
| Phase1 backbone | Frozen`ADV3B02_CORE90_SOFT_E200phase1`features |
| Target clean | Not used |
| Target unknown query for threshold | Not used |
| Target labels used | Only`target_old`K-shot support per old TX |
| Query | Held-out`target_old`query plus`target_unknown`query from same target receiver domain |
| Interpretation | Phase2-B old-class calibration diagnostic, not source-only Stage2-A evidence |

Mechanism:

1. Group sat-only feature rows by role/TX/RX/day/signal.
2. For each target receiver run and each old TX, choose deterministic K-shot target-old support.
3. Build target-old prototypes from support, optionally shrink toward source old prototypes.
4. Score held-out target-old query and target_unknown query by target-prototype cosine or negative L2.
5. Choose thresholds from support scores and source proxy_unknown scores only.
6. Report`unknown_FAR`, closed old accuracy, post-reject old accuracy, old drop, and dual-pass status.

New local file:

| File | Purpose | SHA256 |
|---|---|---|
| `E:\type10-7\code\scripts\eval_phase2b_kshot_oldcalib_reject_20260703.py` | Evaluate sat-only Phase2-B K-shot old-class calibration and unknown rejection over raw/V3 repaired Phase1 features, including target-label oracle diagnostic rows | `378C0431CEB3A7321CE012502ACC5EE4B8C3F94D38B5A1CC7F684D71B279133D` |

Local verification:

| Command | Result |
|---|---|
| `C:\Users\lh594\.conda\envs\ssr-gpu\python.exe -m py_compile E:\type10-7\code\scripts\eval_phase2b_kshot_oldcalib_reject_20260703.py` | PASS |

Planned remote output: `/home/szu2070436088/2510044040/CV-SincNet/logs/phase2b_kshot_oldcalib_reject_20260703/kshot_oldcalib_reject_summary.csv`.

## V10 Completion Results

V10 was run in two passes:

| Pass | Rows | Dual pass | Boundary |
|---|---:|---:|---|
| Deployable support/proxy thresholds | 187200 | 0 | Uses target-old support and source proxy_unknown only for calibration; no unknown query threshold |
| Target-label oracle diagnostic | 190080 | 0 | Adds oracle rows that use target labels/unknown query only to test whether a score-space threshold exists |

Artifacts:

| Artifact | Local path | SHA256 |
|---|---|---|
| V10 deployable summary | `E:\type10-7\automation_reports\CV-SincNet\phase1_adv3b02_leo_feature_adapter_20260703\artifacts\kshot_oldcalib_reject_summary.csv` | `817EC03344124423CE422DB1A38B4C8AB576FBB4A2986329C2592F93E14EDA40` |
| V10 deployable log | `E:\type10-7\automation_reports\CV-SincNet\phase1_adv3b02_leo_feature_adapter_20260703\artifacts\kshot_oldcalib_reject.out` | `59BA9C89AC1D71669F8E9679C22A58730C96376FCDB38EFFE92F2ACE2BDDF22F` |
| V10 oracle summary | `E:\type10-7\automation_reports\CV-SincNet\phase1_adv3b02_leo_feature_adapter_20260703\artifacts\kshot_oldcalib_reject_oracle_summary.csv` | `451371CC91A1AB01ECE508C2C50B8588BFCF02FAE924C8273F8084C94B3F7C69` |
| V10 oracle log | `E:\type10-7\automation_reports\CV-SincNet\phase1_adv3b02_leo_feature_adapter_20260703\artifacts\kshot_oldcalib_reject_oracle.out` | `961DA6BBB0AC2C95B93115A400EBB49B265D6AED260B50F6BFC5AC4418FD44D1` |

Best deployable same-row outcomes:

| Selection rule | Run | Feature | K | Metric/score/policy | unknown_FAR | Old drop pp | Closed old acc | Final old acc | Verdict |
|---|---|---|---:|---|---:|---:|---:|---:|---|
| Best FAR with old drop`<=2pp` | `rx20_1_u1` | `LEOADAPT3_LINR_COS` | 20 | neg_l2/max_score/mean_support_proxy | 0.4174 | 1.86 | 0.6140 | 0.5954 | FAR fails |
| Best old retention with FAR`<=5%` | `rx20_1_u1` | `LEOADAPT3_LINR_COS` | 1 | neg_l2/max_score/mean_support_proxy | 0.0476 | 23.01 | 0.4912 | 0.2610 | Old performance fails |
| Nearest deployable joint row | `rx20_1_u1` | `LEOADAPT3_LINR_COS` | 1 | neg_l2/max_score/mean_support_proxy | 0.0840 | 17.83 | 0.4912 | 0.3129 | Still far from dual target |

Best target-label oracle same-row outcomes:

| Selection rule | Run | Feature | K | Metric/score | unknown_FAR | Old drop pp | Closed old acc | Final old acc | Verdict |
|---|---|---|---:|---|---:|---:|---:|---:|---|
| Best FAR with old drop`<=2pp` | `rx7_7_u1` | `LEOADAPT3_MLP_ID` | 10 | neg_l2/max_score | 0.5700 | 1.53 | 0.7521 | 0.7368 | Even oracle FAR fails |
| Best old retention with FAR`<=5%` | `rx20_1_u10` | raw Phase1 sat-only | 2 | cosine/margin | 0.0498 | 23.32 | 0.4690 | 0.2358 | Old performance fails |
| Nearest oracle row | `rx20_1_u1` | `LEOADAPT3_LINR_COS` | 1 | neg_l2/max_score | 0.0952 | 16.32 | 0.4912 | 0.3279 | Even oracle misses both |

V10 interpretation:

Target-old K-shot support improves closed old-class accuracy in some receiver cells, especially with raw Phase1 or `LEOADAPT3_LINR_COS` features at higher K. However, old/unknown score distributions still overlap heavily under sat-only LEO query. The target-label oracle result is decisive for this prototype-score route: even with an invalid oracle threshold chosen using target unknown labels, there is no row satisfying both`unknown_FAR<=0.05`and old drop`<=2pp`. Therefore the failure is not only support/proxy threshold calibration; the tested Phase1 feature/prototype score itself does not expose a sufficient separation surface.

Current conclusion after V10:

| Target | Status | Evidence |
|---|---|---|
| 目标1: source-only LEO repair | Achieved | V3/V9b retain valid source-only LEO repair evidence |
| 目标2: `unknown_FAR<=0.05` with old drop`<=2pp` | Not achieved | V3-V10 all have dual pass 0; V10 target-label oracle also has dual pass 0 |

Next technically aligned action must move below post-export scoring. Either retrain a Phase1-compatible representation with explicit source non-old/open-set negatives and old retention, or add a raw-IQ level denoising/equalization front-end before feature extraction. Continuing to sweep thresholds, shallow heads, or K-shot prototype gates on the current exported feature space is unlikely to satisfy the objective.

## V11 IQ Pre-Adapter Design

V11 moves below post-export feature scoring. It trains a lightweight residual IQ pre-adapter before the frozen`ADV3B02_CORE90_SOFT_E200phase1`backbone, using only source old clean/LEO paired supervision. Target receiver old/unknown samples are exported only after training for sat-only evaluation.

Protocol boundary:

| Item | V11 setting |
|---|---|
| Phase1 backbone | Frozen`ADV3B02_CORE90_SOFT_E200phase1` |
| Train data | Source old ManySig rows from source receivers only |
| Train channel | `leo_clear_weak`,`leo_low_elev_weak`,`leo_rain_weak`source-derived LEO views |
| Target clean | Not used |
| Target labels in training | Not used |
| Unknown query threshold fitting | Not used |
| Target query view | sat-only LEO after IQ pre-adapter |
| Rejection calibration | Existing source old + source proxy_unknown roles only |

Mechanism:

1. Generate one LEO observation from each source clean IQ batch.
2. Apply a small residual Conv1D IQ adapter.
3. Pass clean and repaired IQ through the frozen Phase1 model.
4. Optimize SmoothL1/cosine feature repair plus old-class prototype/logit CE, with a residual penalty to avoid over-editing IQ.
5. Export target-old and target-unknown sat-only features for the same 10 receiver/unknown cells.
6. Evaluate existing source/proxy rejection policies, including`IQPRE_LIN_SRC1000`as the direct analogue of the earlier multi-view`MV_LIN_SRC1000`high-retention threshold.

New local files:

| File | Purpose | SHA256 |
|---|---|---|
| `E:\type10-7\code\scripts\train_apply_phase1_iq_preadapter_20260703.py` | Train source-only IQ pre-adapter and export sat-only feature NPZs | `98459279CBE95266D97AD5A0001242A1544F655EBE5D859B3F2B3CB00596CBD1` |
| `E:\type10-7\code\scripts\sweep_phase1_adv3b02_iqpre_v11_20260703.sh` | N607 launcher for V11 train/export/eval summary | `8FCC39B828AC7F27AA12901D845EFCBA51AFEFA80DE1745076C4CF32D65A03E6` |

Local verification:

| Command | Result |
|---|---|
| `C:\Users\lh594\.conda\envs\ssr-gpu\python.exe -m py_compile E:\type10-7\code\scripts\train_apply_phase1_iq_preadapter_20260703.py` | PASS |
| `bash -lc "bash -n /mnt/e/type10-7/code/scripts/sweep_phase1_adv3b02_iqpre_v11_20260703.sh"` | PASS |
| LF line-ending audit | PASS |

Startup repair note: initial N607 launch attempt failed before training because the new script did not expose the `dataset/model_size/model_variant/branch_ablation/sample_rate_hz` compatibility fields expected by`build_model_from_ckpt`. The local script was patched, recompiled, and resynced before relaunch. No V11 training process remained active after the failed attempt.

Version/snapshot state:

| Item | Path |
|---|---|
| Non-Git code snapshot | `E:\type10-7\code\snapshots\phase1_adv3b02_iqpre_v11_20260703\` |
| Git mirror script path | `E:\type10-7\github_publish\CVS-RFFI-repo\code\scripts\train_apply_phase1_iq_preadapter_20260703.py` |
| Git mirror launcher path | `E:\type10-7\github_publish\CVS-RFFI-repo\code\scripts\sweep_phase1_adv3b02_iqpre_v11_20260703.sh` |

Planned N607 variants:

| Variant | GPU | Key config | Remote summary |
|---|---:|---|---|
| `v11a` | 0 | `alpha=0.25`,`hidden_dim=32`,`epochs=45` | `/home/szu2070436088/2510044040/CV-SincNet/logs/phase1_adv3b02_iqpre_v11a_matrix_20260703/iqpre_v11a_sweep_summary.csv` |
| `v11b` | 1 | `alpha=0.40`,`hidden_dim=48`,`epochs=55` | `/home/szu2070436088/2510044040/CV-SincNet/logs/phase1_adv3b02_iqpre_v11b_matrix_20260703/iqpre_v11b_sweep_summary.csv` |

Success criteria remain unchanged:`unknown_FAR<=0.05`and old-class performance drop`<=2pp`on sat-only target query, without target clean or target unknown threshold tuning.

## V14 Completion Results

V14 completed two selective-correctness variants on N607 as bounded foreground commands. Both variants train only on source repaired features and source proxy_unknown rows; target rows are used only for final sat-only metrics.

Overall result:

| Variant | Rows | Dual pass | FAR-only pass | Old-drop-only pass | Status |
|---|---:|---:|---:|---:|---|
| `v14a` | 3200 | 0 | 970 | 104 | Completed negative |
| `v14b` | 3200 | 0 | 840 | 114 | Completed negative |
| Combined | 6400 | 0 | 1810 | 218 | Completed negative |

Artifacts:

| Artifact | Local path | SHA256 |
|---|---|---|
| V14a summary | `E:\type10-7\automation_reports\CV-SincNet\phase1_adv3b02_leo_feature_adapter_20260703\artifacts\selective_correctness_v14a_summary.csv` | `14A64804B0A66F90CDE069431A834386AD07FBCEB9E027DBBC9BBA08A08BF39D` |
| V14a best JSON | `E:\type10-7\automation_reports\CV-SincNet\phase1_adv3b02_leo_feature_adapter_20260703\artifacts\selective_correctness_v14a_best.json` | `FBFF4AF7CED4106BD6C1231E746B191DDB0F49AB68566FB9BFB2706D67EDBEE8` |
| V14a metrics JSON | `E:\type10-7\automation_reports\CV-SincNet\phase1_adv3b02_leo_feature_adapter_20260703\artifacts\selective_correctness_v14a_metrics.json` | `F604262551CC15B73863BCF5FD9B1FFF4C14CFA3E59708F1063D28D88219DC0B` |
| V14a run log | `E:\type10-7\automation_reports\CV-SincNet\phase1_adv3b02_leo_feature_adapter_20260703\artifacts\selective_correctness_v14a.out` | `186191F48FAA8832166B44302B53891A21D3406D0262A0A56E88A30CCA8D05C3` |
| V14b summary | `E:\type10-7\automation_reports\CV-SincNet\phase1_adv3b02_leo_feature_adapter_20260703\artifacts\selective_correctness_v14b_summary.csv` | `D245F2C9F82C3C9BE0225932CE525F958CCC3134BB3CE5118EEC542E38D531C0` |
| V14b best JSON | `E:\type10-7\automation_reports\CV-SincNet\phase1_adv3b02_leo_feature_adapter_20260703\artifacts\selective_correctness_v14b_best.json` | `C026A0709916AE723646660EC6771A3BFAE4F510732DD1A86E056B713D25AA5F` |
| V14b metrics JSON | `E:\type10-7\automation_reports\CV-SincNet\phase1_adv3b02_leo_feature_adapter_20260703\artifacts\selective_correctness_v14b_metrics.json` | `3E5FDE9FC123C7BFFA3D8003CD1DCB390E526D768DF21766B8041AA4FF014853` |
| V14b run log | `E:\type10-7\automation_reports\CV-SincNet\phase1_adv3b02_leo_feature_adapter_20260703\artifacts\selective_correctness_v14b.out` | `81B201EA2EDE2C7C38F96CABC623E7F755823F4A514273DBC055C94D5D3E6DE4` |

Best same-row outcomes:

| Variant | Selection rule | Run | Model/policy | unknown_FAR | Old drop pp | Closed old acc | Final old acc | Coverage | Verdict |
|---|---|---|---|---:|---:|---:|---:|---:|---|
| `v14a` | Best FAR with old drop`<=2pp` | `rx3_19_u10` | `linear/source_correct_accept q=0.005` | 0.7209 | 1.91 | 0.5076 | 0.4885 | 0.9079 | FAR fails badly |
| `v14a` | Best old retention with FAR`<=5%` | `rx20_1_u1` | `mlp/proxy_far p=0.85` | 0.0364 | 35.88 | 0.5556 | 0.1968 | 0.3118 | Old-class performance fails |
| `v14a` | Nearest joint row | `rx3_19_u10` | `mlp/source_correct_accept q=0.02` | 0.4363 | 7.56 | 0.5076 | 0.4321 | 0.7376 | Misses both |
| `v14b` | Best FAR with old drop`<=2pp` | `rx3_19_u10` | `linear/source_correct_accept q=0.005` | 0.7182 | 1.94 | 0.5321 | 0.5126 | 0.9497 | FAR fails badly |
| `v14b` | Best old retention with FAR`<=5%` | `rx20_1_u1` | `mlp/proxy_far p=0.85` | 0.0448 | 38.65 | 0.5976 | 0.2112 | 0.3212 | Old-class performance fails |
| `v14b` | Nearest joint row | `rx3_19_u10` | `mlp/source_correct_accept q=0.01` | 0.5691 | 4.18 | 0.5321 | 0.4903 | 0.8641 | Misses both |

Interpretation:

Selective-correctness training reduced the objective mismatch relative to generic oldness, but did not create a usable deployable gate. The best old-preserving rows still accept roughly72% of unknowns. The low-FAR rows again reject most correct old target samples. This adds a direct selective-classification negative result to the frozen-base evidence: even when the gate is trained to preserve source closed-set correctness rather than class membership, target old and target unknown remain overlapping under LEO.

Current status after V14:

| Target | Status | Evidence |
|---|---|---|
| 目标1: source-only LEO repair | Achieved | V3/V9b source-only adapters repair source LEO features without target clean/labels |
| 目标2: `unknown_FAR<=0.05` with old drop`<=2pp` | Not achieved | V3-V14 all have dual pass 0; V14 adds 6400 selective-correctness rows with dual pass 0 |

The remaining aligned route is no longer a different frozen-feature gate. The separation needed for`unknown_FAR<5%`with old-class drop`<2pp`must be learned before or inside the Phase1 representation, using source-side non-old/open-set negatives and LEO identity-retention, then audited again with the same sat-only target protocol.

## V11 Completion Results

V11 completed two IQ pre-adapter variants on N607:

| Variant | Rows | Dual pass | FAR-only pass | Old-drop-only pass | Status |
|---|---:|---:|---:|---:|---|
| `v11a` | 60 | 0 | 2 | 22 | Completed negative |
| `v11b` | 60 | 0 | 2 | 18 | Completed negative |
| Combined | 120 | 0 | 4 | 40 | Completed negative |

Remote execution:

| Variant | PID | GPU | Completion | Remote summary |
|---|---:|---:|---|---|
| `v11a` | `1090324` | 0 | `2026-07-03T03:30:49+08:00` | `/home/szu2070436088/2510044040/CV-SincNet/logs/phase1_adv3b02_iqpre_v11a_matrix_20260703/iqpre_v11a_sweep_summary.csv` |
| `v11b` | `1090325` | 1 | `2026-07-03T03:30:56+08:00` | `/home/szu2070436088/2510044040/CV-SincNet/logs/phase1_adv3b02_iqpre_v11b_matrix_20260703/iqpre_v11b_sweep_summary.csv` |

Local artifacts:

| Artifact | Local path | SHA256 |
|---|---|---|
| V11a summary | `E:\type10-7\automation_reports\CV-SincNet\phase1_adv3b02_leo_feature_adapter_20260703\artifacts\iqpre_v11a_sweep_summary.csv` | `BDAFC9ACEE7F6645E812CBC39A6EF8E9634780D837F0BD012F80FAFE33DE0329` |
| V11a driver | `E:\type10-7\automation_reports\CV-SincNet\phase1_adv3b02_leo_feature_adapter_20260703\artifacts\iqpre_v11a_driver.out` | `06585FDC16B81F1F630C67D60DE52226B6A37ED89D8CEB321C3CD5B73A6056D4` |
| V11a train/export log | `E:\type10-7\automation_reports\CV-SincNet\phase1_adv3b02_leo_feature_adapter_20260703\artifacts\iqpre_v11a_train_export.out` | `5834D96B52D19FCF8C584253AA2B6A313048FC7A893F2EC7D705D2D39A94625F` |
| V11b summary | `E:\type10-7\automation_reports\CV-SincNet\phase1_adv3b02_leo_feature_adapter_20260703\artifacts\iqpre_v11b_sweep_summary.csv` | `30F226B1D9DEEFA882E6A30D4A9085140B7470A41CC74DD083A13CFCEA6BCBAD` |
| V11b driver | `E:\type10-7\automation_reports\CV-SincNet\phase1_adv3b02_leo_feature_adapter_20260703\artifacts\iqpre_v11b_driver.out` | `50E569A862B852BCC4A41C8CF1FF50DCAEAA4F11FB86FCEB27E396CBA98BE96B` |
| V11b train/export log | `E:\type10-7\automation_reports\CV-SincNet\phase1_adv3b02_leo_feature_adapter_20260703\artifacts\iqpre_v11b_train_export.out` | `23035A3CB03D51538C017857E0EB3A0403589CFC8562337512B43F835247A36D` |

Best same-row outcomes:

| Selection rule | Variant | Run | Policy | unknown_FAR | Old drop pp | Closed old acc | Final old acc | Coverage | Verdict |
|---|---|---|---|---:|---:|---:|---:|---:|---|
| Best FAR with old drop`<=2pp` | `v11a` | `rx3_19_u10` | `IQPRE_LIN_SRC1000` | 0.9268 | 0.25 | 0.5442 | 0.5417 | 0.9958 | FAR fails badly |
| Best old retention with FAR`<=5%` | `v11a` | `rx20_1_u1` | `IQPRE_LIN_MIN05`/`PROXY05` | 0.0252 | 39.42 | 0.6033 | 0.2092 | 0.3025 | Old-class performance fails |
| Lowest FAR overall | `v11b` | `rx20_1_u1` | `IQPRE_LIN_MIN05`/`PROXY05` | 0.0196 | 41.58 | 0.6075 | 0.1917 | 0.2892 | Old-class performance fails |
| Nearest joint row | `v11a` | `rx20_1_u1` | `IQPRE_MLP64_MIN05`/`PROXY05` | 0.0644 | 32.17 | 0.6033 | 0.2817 | 0.4308 | Misses both, especially old retention |
| Best old retention overall | `v11a` | `rx20_1_u1` | `IQPRE_LIN_SRC9999` | 0.9328 | 0.00 | 0.6033 | 0.6033 | 0.9975 | Old retention passes, FAR fails badly |

Training behavior:

| Variant | First reported epoch | Final epoch | Final loss | Final MSE | Final cosine loss | Final CE | Final residual |
|---|---:|---:|---:|---:|---:|---:|---:|
| `v11a` | 5 | 45 | 1.3370 | 0.0712 | 0.2059 | 1.4190 | 0.0872 |
| `v11b` | 5 | 55 | 1.3351 | 0.0707 | 0.2148 | 1.3869 | 0.0890 |

Interpretation:

V11 did not solve the joint objective. The IQ pre-adapter can preserve or slightly improve old closed-set accuracy in some cells, but the source/proxy rejection scores still show the same tradeoff: high-retention thresholds accept nearly all unknowns, while low-FAR thresholds reject most old target samples. This extends the V3-V10 conclusion from feature-space/post-hoc adapters to a source-only IQ pre-adapter in front of the frozen Phase1 model.

Current status after V11:

| Target | Status | Evidence |
|---|---|---|
| 目标1: source-only LEO repair | Still achieved by V3/V9b; V11 is an additional IQ-front-end diagnostic | V11 trains and exports sat-only IQ-pre-adapted features without target clean/labels |
| 目标2: `unknown_FAR<=0.05` with old drop`<=2pp` | Not achieved | V3-V11 all have dual pass 0; V11 combined 120 rows also dual pass 0 |

Next route should stop treating`ADV3B02_CORE90_SOFT_E200phase1`as a sufficient frozen feature basis for this sat-only unknown rejection target. The remaining aligned route is to train a new Phase1 base with open-set/source non-old negatives and LEO identity-retention built into the representation objective, then repeat the same sat-only rejection audit. Under the current frozen Phase1 base, the tested post-adapter, IQ-pre-adapter, K+1, multi-score, class-conditional, K-shot and oracle threshold routes have all failed the joint target.

## V12 Heterogeneous Decision-Fusion Design

V12 tests whether the failure is caused by using a single repair branch. It does not retrain features and does not retune thresholds on target labels. Each branch keeps its own existing source/proxy calibrated accept/reject threshold; V12 only fuses branch decisions on target query samples by vote/quorum.

Protocol boundary:

| Item | V12 setting |
|---|---|
| Phase1 backbone | Frozen`ADV3B02_CORE90_SOFT_E200phase1` |
| Branches | V3 feature adapters plus V11 IQ pre-adapters |
| Threshold source | Existing per-branch source old + source proxy_unknown calibration |
| Target clean | Not used |
| Target labels in threshold | Not used |
| Unknown query threshold fitting | Not used |
| Fusion | Accept if at least K of N branch decisions accept |
| Classification source | Explicit primary branch per row; old drop compares that branch before/after reject |

New files:

| File | Purpose | SHA256 |
|---|---|---|
| `E:\type10-7\code\scripts\eval_phase1_decision_fusion_reject_20260703.py` | Fuse existing branch decisions and compute same-row old/FAR metrics | `5FA678232515C17B4FDB6819CB998426C8EEEC6793C7F58D06A104E53B0185F2` |
| `E:\type10-7\code\scripts\sweep_phase1_adv3b02_decision_fusion_v12_20260703.sh` | N607 launcher and best-row summarizer for V12 | `7178091B98D1D62A3135E8A3B9C0B7870152DDC833ABA58E5BD28381E1A24F4D` |

Local verification:

| Command | Result |
|---|---|
| `C:\Users\lh594\.conda\envs\ssr-gpu\python.exe -m py_compile E:\type10-7\code\scripts\eval_phase1_decision_fusion_reject_20260703.py` | PASS |
| `bash -lc "bash -n /mnt/e/type10-7/code/scripts/sweep_phase1_adv3b02_decision_fusion_v12_20260703.sh"` | PASS |
| LF line-ending audit | PASS |

Planned remote output:

| Artifact | Remote path |
|---|---|
| V12 summary CSV | `/home/szu2070436088/2510044040/CV-SincNet/logs/phase1_adv3b02_decision_fusion_v12_20260703/decision_fusion_v12_summary.csv` |
| V12 best JSON | `/home/szu2070436088/2510044040/CV-SincNet/logs/phase1_adv3b02_decision_fusion_v12_20260703/decision_fusion_v12_best.json` |
| V12 driver | `/home/szu2070436088/2510044040/CV-SincNet/logs/phase1_adv3b02_decision_fusion_v12_20260703/driver.out` |

## V12 Completion Results

V12 completed on N607 as a CPU-only evaluation. The first run produced the main CSV but the launcher-side best-row postprocessor failed because the remote environment lacked`pandas`; the postprocessor was patched to standard-library CSV/JSON and rerun against the completed CSV. The main V12 evaluation itself was not rerun.

Artifacts:

| Artifact | Local path | SHA256 |
|---|---|---|
| V12 summary CSV | `E:\type10-7\automation_reports\CV-SincNet\phase1_adv3b02_leo_feature_adapter_20260703\artifacts\decision_fusion_v12_summary.csv` | `5FD72BBDB463BA4102E74862F03EB47BF6178F63C9114BD14E4BB9D0CA45834A` |
| V12 best JSON | `E:\type10-7\automation_reports\CV-SincNet\phase1_adv3b02_leo_feature_adapter_20260703\artifacts\decision_fusion_v12_best.json` | `AF519F69C820A61EC680D96538B458E54229BB5B80FF465A2D2B03CF8321E95E` |
| V12 driver | `E:\type10-7\automation_reports\CV-SincNet\phase1_adv3b02_leo_feature_adapter_20260703\artifacts\decision_fusion_v12_driver.out` | `00FFFC446B59D681F075C4919793203041EA124E4FE29B5EBEC07B48273B86CF` |

Overall result:

| Rows | Dual pass | FAR-only pass | Old-drop-only pass |
|---:|---:|---:|---:|
| 1600 | 0 | 224 | 800 |

Best same-row outcomes:

| Selection rule | Cell | Branch set | Primary branch | Fusion rule | unknown_FAR | Old drop pp | Closed old acc | Final old acc | Coverage | Verdict |
|---|---|---|---|---|---:|---:|---:|---:|---:|---|
| Best FAR with old drop`<=2pp` | `rx3_19_u10` | `retention4` | `v11b_ret` | `min_accepts_4_of_4` | 0.8157 | 0.00 | 0.6023 | 0.6023 | 0.9942 | FAR fails badly |
| Best old retention with FAR`<=5%` | `rx20_1_u1` | `strict4` | `v3_lcos_strict` | `min_accepts_3_of_4` | 0.0168 | 36.84 | 0.5497 | 0.1813 | 0.2573 | Old-class performance fails |
| Lowest FAR overall | `rx20_1_u1` | `strict4`/`all8` | `v3_lcos_strict` | `min_accepts_4_of_4`/`8_of_8` | 0.0056 | 43.27 | 0.5497 | 0.1170 | 0.1579 | Old-class performance fails |
| Nearest joint row | `rx20_1_u1` | `all8` | `v3_lcos_ret` | `min_accepts_5_of_8` | 0.1232 | 16.37 | 0.5497 | 0.3860 | 0.6608 | Misses both |

Branch-set summary:

| Branch set | Rows | Min FAR | Min old drop pp | Dual pass |
|---|---:|---:|---:|---:|
| `all8` | 640 | 0.0056 | 0.00 | 0 |
| `hetero_mixed4a` | 160 | 0.0112 | 0.00 | 0 |
| `hetero_mixed4b` | 160 | 0.0112 | 0.00 | 0 |
| `iqpre_mixed4` | 160 | 0.0448 | 0.00 | 0 |
| `retention4` | 160 | 0.8157 | 0.00 | 0 |
| `strict4` | 160 | 0.0056 | 16.37 | 0 |
| `v3_mixed4` | 160 | 0.0168 | 0.00 | 0 |

Interpretation:

Decision fusion confirms that the tradeoff is not a single-branch threshold artifact. Retention-oriented branches preserve old-class accuracy but accept most unknowns. Strict branches and all-branch quorum can force`unknown_FAR`well below5%, but only by rejecting a large fraction of correctly classified old target samples. The best joint row remains at`unknown_FAR=12.32%`and old drop`16.37pp`, far from the requested`unknown_FAR<5%`and old drop`<2pp`.

Current status after V12:

| Target | Status | Evidence |
|---|---|---|
| 目标1: source-only LEO repair | Achieved by V3/V9b; V11/V12 add diagnostic coverage | V3 source-pair adapters repair source LEO features; V11 IQ pre-adapter and V12 fusion do not use target clean/labels |
| 目标2: `unknown_FAR<=0.05` with old drop`<=2pp` | Not achieved | V3-V12 all have dual pass 0; V12 1600-row heterogeneous decision fusion also dual pass 0 |

The remaining route that still moves toward the original target is no longer another frozen-Phase1 threshold or adapter combination. The evidence now points to representation-level retraining under the Phase1 protocol: source-side non-old/open-set negatives and LEO identity-retention must be learned inside the base representation, after which the same sat-only audit should be repeated.

## V13 Repair-Ensemble Manifold Design

V13 keeps the frozen`ADV3B02_CORE90_SOFT_E200phase1`backbone and does not use target clean features. It tests a narrower repair-based hypothesis: old target LEO samples should land consistently on the source old manifold after source-only LEO repair, while unknown samples should show larger repaired-manifold distance or weaker repairer agreement. This is a reverse repair view rather than a stronger augmentation view.

Protocol boundary:

| Item | V13 setting |
|---|---|
| Phase1 backbone | Frozen`ADV3B02_CORE90_SOFT_E200phase1` |
| Repair modules | Existing V3 source-only LEO adapters |
| Training/fitting data | Source rows inside repaired feature NPZs only |
| Negative calibration | `proxy_unknown` rows from source receiver/non-old LEO payloads |
| Target clean | Not used |
| Target labels in threshold | Not used |
| Unknown query threshold fitting | Not used |
| Accept rule | Source/proxy calibrated threshold on repaired-manifold oldness score |

New files:

| File | Purpose | SHA256 |
|---|---|---|
| `E:\type10-7\code\scripts\eval_phase1_repair_ensemble_manifold_reject_20260703.py` | Fit source old manifolds over repaired features and evaluate repair agreement/distance rejection | `E127200A0533367F20EAC29DF618D8AA4131C748D49C79F823B1CDA090C77C65` |
| `E:\type10-7\code\scripts\sweep_phase1_adv3b02_repair_ensemble_v13_20260703.sh` | N607 launcher and best-row summarizer for V13a/V13b | `7A5D7D87A39E754CE3104F978A3F5CF9978FAEC3C8C306BAC5473DA71C62F0C9` |

Local verification:

| Command | Result |
|---|---|
| `C:\Users\lh594\.conda\envs\ssr-gpu\python.exe -m py_compile E:\type10-7\code\scripts\eval_phase1_repair_ensemble_manifold_reject_20260703.py` | PASS |
| `bash -lc "bash -n /mnt/e/type10-7/code/scripts/sweep_phase1_adv3b02_repair_ensemble_v13_20260703.sh"` | PASS |

Version/snapshot state:

| Item | Path |
|---|---|
| Non-Git code snapshot | `E:\type10-7\code\snapshots\phase1_adv3b02_repair_ensemble_v13_20260703\` |
| Git mirror eval script path | `E:\type10-7\github_publish\CVS-RFFI-repo\code\scripts\eval_phase1_repair_ensemble_manifold_reject_20260703.py` |
| Git mirror launcher path | `E:\type10-7\github_publish\CVS-RFFI-repo\code\scripts\sweep_phase1_adv3b02_repair_ensemble_v13_20260703.sh` |

Planned N607 variants:

| Variant | Adapter set | Remote summary |
|---|---|---|
| `v13a` | identity, mean-shift, norm-shift, linear residual, MLP residual | `/home/szu2070436088/2510044040/CV-SincNet/logs/phase1_adv3b02_repair_ensemble_v13a_20260703/repair_ensemble_v13a_summary.csv` |
| `v13b` | linear residual, MLP residual | `/home/szu2070436088/2510044040/CV-SincNet/logs/phase1_adv3b02_repair_ensemble_v13b_20260703/repair_ensemble_v13b_summary.csv` |

Success criteria remain unchanged:`unknown_FAR<=0.05`and old-class performance drop`<=2pp`on sat-only target query, without target clean or target unknown threshold tuning.

## V13 Completion Results

V13 completed two repair-ensemble manifold variants on N607. V13a was launched by the first background command; the local SSH command timed out and left a stale local`ssh.exe`, which was identified as PID`38736`, stopped locally, and verified clean before further SSH work. V13b was then run as a bounded foreground command. No target clean, target threshold labels, or unknown query threshold fitting were used.

Overall result:

| Variant | Rows | Dual pass | FAR-only pass | Old-drop-only pass | Status |
|---|---:|---:|---:|---:|---|
| `v13a` | 5600 | 0 | 1925 | 280 | Completed negative |
| `v13b` | 5600 | 0 | 1948 | 298 | Completed negative |
| Combined | 11200 | 0 | 3873 | 578 | Completed negative |

Artifacts:

| Artifact | Local path | SHA256 |
|---|---|---|
| V13a summary | `E:\type10-7\automation_reports\CV-SincNet\phase1_adv3b02_leo_feature_adapter_20260703\artifacts\repair_ensemble_v13a_summary.csv` | `D31BA606A66B6C55C9839BF4185B0D4269A6DD446B53BA0FD6374932D285E61E` |
| V13a best JSON | `E:\type10-7\automation_reports\CV-SincNet\phase1_adv3b02_leo_feature_adapter_20260703\artifacts\repair_ensemble_v13a_best.json` | `BC35833B69DB2544D7153A72DB96C76E84D6ADF7ABB7EE1D90CE7C98E0B48DA5` |
| V13a metrics JSON | `E:\type10-7\automation_reports\CV-SincNet\phase1_adv3b02_leo_feature_adapter_20260703\artifacts\repair_ensemble_v13a_metrics.json` | `DA46E0C7C38D65E71DF3FD90DACBE0422C80C27108FB35DF52530DDAD0364CCC` |
| V13a driver | `E:\type10-7\automation_reports\CV-SincNet\phase1_adv3b02_leo_feature_adapter_20260703\artifacts\repair_ensemble_v13a_driver.out` | `522360E5C83DCE7A5AAA0B82F287FE9918C35DDD78B045E4C664A67E2BD81CF3` |
| V13b summary | `E:\type10-7\automation_reports\CV-SincNet\phase1_adv3b02_leo_feature_adapter_20260703\artifacts\repair_ensemble_v13b_summary.csv` | `2CD382367E7A5701795915463A12FB584D3C0564429B366169DDFC96EB7DE85F` |
| V13b best JSON | `E:\type10-7\automation_reports\CV-SincNet\phase1_adv3b02_leo_feature_adapter_20260703\artifacts\repair_ensemble_v13b_best.json` | `E6F148C4655AA178C34B79060FCAC2DC1EE7DA3F6A8A01D3131147BD73CAA8BF` |
| V13b metrics JSON | `E:\type10-7\automation_reports\CV-SincNet\phase1_adv3b02_leo_feature_adapter_20260703\artifacts\repair_ensemble_v13b_metrics.json` | `01EEFC1058C530CA189955FCEFFE7A8178E46984E877FA5AB11AAE62A8B14489` |

Best same-row outcomes:

| Variant | Selection rule | Run | Score/policy | unknown_FAR | Old drop pp | Closed old acc | Final old acc | Coverage | Verdict |
|---|---|---|---|---:|---:|---:|---:|---:|---|
| `v13a` | Best FAR with old drop`<=2pp` | `rx3_19_u10` | `neg_mean_min_mah/source_accept q=0.005` | 0.6856 | 1.06 | 0.5321 | 0.5215 | 0.9732 | FAR fails badly |
| `v13a` | Best old retention with FAR`<=5%` | `rx3_19_u10` | `mean_max_sim/mean_source_proxy q=0.05,p=0.95` | 0.0488 | 41.56 | 0.5321 | 0.1165 | 0.1356 | Old-class performance fails |
| `v13a` | Nearest joint row | `rx3_19_u10` | `mean_max_sim/source_accept q=0.05` | 0.4092 | 9.35 | 0.5321 | 0.4385 | 0.6726 | Misses both |
| `v13b` | Best FAR with old drop`<=2pp` | `rx3_19_u10` | `neg_mean_min_mah/source_accept q=0.005` | 0.6911 | 1.24 | 0.5076 | 0.4953 | 0.9603 | FAR fails badly |
| `v13b` | Best old retention with FAR`<=5%` | `rx20_1_u10` | `mean_sim_margin/mean_source_proxy q=0.05,p=0.99` | 0.0498 | 34.35 | 0.5556 | 0.2121 | 0.2176 | Old-class performance fails |
| `v13b` | Nearest joint row | `rx3_19_u10` | `mean_max_sim/source_accept q=0.05` | 0.4119 | 9.47 | 0.5076 | 0.4129 | 0.7012 | Misses both |

Interpretation:

Repair-ensemble manifold scoring did not solve the core overlap. Scores that preserve old-class performance still accept most unknowns. Scores and thresholds that push`unknown_FAR`below5% do so only by rejecting most correctly classified old target samples. This closes the main remaining frozen-base repair path: single repair adapter, IQ pre-adapter, oldness gate, heterogeneous decision fusion and repair-ensemble manifold gates all reproduce the same tradeoff under sat-only target query.

Current status after V13:

| Target | Status | Evidence |
|---|---|---|
| 目标1: source-only LEO repair | Achieved | V3/V9b source-only adapters repair source LEO features without target clean/labels |
| 目标2: `unknown_FAR<=0.05` with old drop`<=2pp` | Not achieved | V3-V13 all have dual pass 0; V13 adds 11200 repair-ensemble manifold rows with dual pass 0 |

The next meaningful route requires changing where separation is learned. Within the frozen Phase1 feature basis, every deployable source-only gate found so far exposes the same old/unknown overlap. The aligned next step is Phase1-level representation repair/retraining with source-side non-old/open-set negatives and LEO identity-retention inside the representation objective, then rerun the same sat-only audit.

## V14 Selective-Correctness Gate Design

V14 keeps the frozen`ADV3B02_CORE90_SOFT_E200phase1`backbone and reuses source-only repaired features. It changes the gate objective from generic oldness to selective correctness: accept a sample only when the repaired ensemble predicts it as an old class with source-trained evidence resembling source LEO samples that are actually classified correctly. The aim is to protect closed-set-correct old target samples while rejecting unknown and unreliable samples.

Training labels are source-only:

| Row type | V14 training label |
|---|---|
| Source old LEO, repaired-ensemble prediction equals source TX | positive / correct-old |
| Source old LEO, repaired-ensemble prediction differs from source TX | negative / unreliable-old |
| Source proxy_unknown LEO | negative / non-old |
| Target old/unknown | never used for training or threshold fitting |

New files:

| File | Purpose | SHA256 |
|---|---|---|
| `E:\type10-7\code\scripts\eval_phase1_selective_correctness_reject_20260703.py` | Train source-only selective-correctness gates and evaluate sat-only target old/unknown rows | `4A73374F0DFFFA3F1C478E9B72B8938611001C5C0674479F7E93E8D8EE00C066` |
| `E:\type10-7\code\scripts\sweep_phase1_adv3b02_selective_correctness_v14_20260703.sh` | N607 launcher and best-row summarizer for V14a/V14b | `620740DE098C7138EBC703B3C21B62759639EB7C70C21051BD94E92C484C1AAB` |

Local verification:

| Command | Result |
|---|---|
| `C:\Users\lh594\.conda\envs\ssr-gpu\python.exe -m py_compile E:\type10-7\code\scripts\eval_phase1_selective_correctness_reject_20260703.py` | PASS |
| `bash -lc "bash -n /mnt/e/type10-7/code/scripts/sweep_phase1_adv3b02_selective_correctness_v14_20260703.sh"` | PASS |

Version/snapshot state:

| Item | Path |
|---|---|
| Non-Git code snapshot | `E:\type10-7\code\snapshots\phase1_adv3b02_selective_correctness_v14_20260703\` |
| Git mirror eval script path | `E:\type10-7\github_publish\CVS-RFFI-repo\code\scripts\eval_phase1_selective_correctness_reject_20260703.py` |
| Git mirror launcher path | `E:\type10-7\github_publish\CVS-RFFI-repo\code\scripts\sweep_phase1_adv3b02_selective_correctness_v14_20260703.sh` |

Planned N607 variants:

| Variant | Adapter set | Remote summary |
|---|---|---|
| `v14a` | linear residual, MLP residual | `/home/szu2070436088/2510044040/CV-SincNet/logs/phase1_adv3b02_selective_correctness_v14a_20260703/selective_correctness_v14a_summary.csv` |
| `v14b` | identity, mean-shift, norm-shift, linear residual, MLP residual | `/home/szu2070436088/2510044040/CV-SincNet/logs/phase1_adv3b02_selective_correctness_v14b_20260703/selective_correctness_v14b_summary.csv` |

Success criteria remain unchanged:`unknown_FAR<=0.05`and old-class performance drop`<=2pp`on sat-only target query, without target clean or target unknown threshold tuning.
