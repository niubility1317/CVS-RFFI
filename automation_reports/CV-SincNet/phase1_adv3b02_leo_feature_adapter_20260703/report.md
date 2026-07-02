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
