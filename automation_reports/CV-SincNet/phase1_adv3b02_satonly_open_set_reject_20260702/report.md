# Phase1 ADV3B02 Satellite-Only Open-Set Rejection

## Objective

根据`ADV3B02_CORE90_SOFT_E200phase1`阶段基座模型重新评估未知类拒识，进一步贴近CVS Phase2部署视图：测试样本只叠加星地信道，不包含`clean`样本；不使用Phase2少样本学习、K-shot support、target label阈值拟合或unknown query调阈值。

Success gate:

| Metric | Target |
|---|---:|
| `unknown_FAR` | `<=0.05` |
| `old_drop_pp_vs_closed` | `<=2.0pp` |

## Design

This run reuses the frozen Phase1 base model and the satellite feature exports produced by the previous Phase1 multi-view run. It deliberately excludes `clean.npz` and rebuilds a satellite-only multi-view feature file from:

| View | Scenario |
|---|---|
| `sat_clear.npz` | `leo_clear_weak` |
| `sat_low.npz` | `leo_low_elev_weak` |
| `sat_rain.npz` | `leo_rain_weak` |

The rejection method remains Phase1-only:

| Component | Value |
|---|---|
| Base model | Frozen `ADV3B02_CORE90_SOFT_E200phase1` |
| Query view | satellite-only multi-view, no clean sample |
| Risk head | Linear binary risk head trained on source old vs source-side proxy unknown |
| Threshold policy | source-accept quantile sweep |
| Policies | `SATMV_LIN_SRC9999`, `SATMV_LIN_SRC99995`, `SATMV_LIN_SRC1000` |
| Old-class preservation gate | `old_drop_pp_vs_closed<=2.0` |

Validation cells:

| Target receiver | Unknown TX set |
|---|---|
| `20-1` | `10-1,10-10` |
| `20-1` | `1-16,4-10` |
| `3-19` | `10-1,10-10` |
| `3-19` | `1-16,4-10` |
| `7-14` | `10-1,10-10` |
| `7-14` | `1-16,4-10` |
| `7-7` | `10-1,10-10` |
| `7-7` | `1-16,4-10` |
| `8-8` | `10-1,10-10` |
| `8-8` | `1-16,4-10` |

## Local Changes

| File | Purpose |
|---|---|
| `E:\type10-7\code\scripts\sweep_phase1_adv3b02_satonly_source_quantile_20260702.sh` | Merge only satellite views and run the stricter Phase1-only source-quantile rejection sweep. |
| `E:\type10-7\code\scripts\sweep_phase1_adv3b02_satonly_mlp_source_quantile_20260702.sh` | Reuse the satellite-only feature files and test a stronger MLP64 risk head under the same source-quantile policies. |
| `E:\type10-7\code\scripts\eval_phase1_multiview_reject.py` | Add optional correct-known-only risk-head training while keeping default behavior unchanged. |
| `E:\type10-7\code\scripts\sweep_phase1_adv3b02_satonly_correct_known_20260702.sh` | Train risk heads with only source closed-correct old samples as positives and source closed-incorrect samples as auxiliary negatives. |
| `E:\type10-7\code\scripts\sweep_phase1_adv3b02_satonly_margin_20260702.sh` | Train source-only margin-loss risk heads that directly push source old below risk margin and proxy unknown above risk margin. |
| `E:\type10-7\code\tests\test_phase1_multiview_reject_eval.py` | Keep evaluator regression test compatible with the new explicit training-label switches. |

## Local Verification

| Command | Result |
|---|---|
| `bash -n code/scripts/sweep_phase1_adv3b02_satonly_source_quantile_20260702.sh` | PASS |
| `bash -n code/scripts/sweep_phase1_adv3b02_satonly_mlp_source_quantile_20260702.sh` | PASS |
| `conda run -n ssr-gpu python -m py_compile code\scripts\eval_phase1_multiview_reject.py code\tests\test_phase1_multiview_reject_eval.py` | PASS |
| `conda run -n ssr-gpu pytest code\tests\test_phase1_multiview_reject_eval.py -q` | PASS, 1 test; `.pytest_cache` permission warning only |
| `bash -n code/scripts/sweep_phase1_adv3b02_satonly_correct_known_20260702.sh` | PASS |
| `conda run -n ssr-gpu python -m py_compile code\scripts\eval_phase1_multiview_reject.py code\tests\test_phase1_multiview_reject_eval.py` | PASS after margin-loss addition |
| `conda run -n ssr-gpu pytest code\tests\test_phase1_multiview_reject_eval.py -q` | PASS after margin-loss addition, 1 test; `.pytest_cache` permission warning only |
| `bash -n code/scripts/sweep_phase1_adv3b02_satonly_margin_20260702.sh` | PASS |

## Planned N607 Execution

| Field | Value |
|---|---|
| Remote root | `/home/szu2070436088/2510044040/CV-SincNet` |
| Remote script | `/home/szu2070436088/2510044040/CV-SincNet/code/scripts/sweep_phase1_adv3b02_satonly_source_quantile_20260702.sh` |
| Remote matrix log root | `/home/szu2070436088/2510044040/CV-SincNet/logs/phase1_adv3b02_satonly_matrix_20260702` |
| Expected summary | `/home/szu2070436088/2510044040/CV-SincNet/logs/phase1_adv3b02_satonly_matrix_20260702/satonly_sourceq_sweep_summary.csv` |
| Expected rows | 30 rows: 10 cells x 3 policies |
| Resource profile | Eval-only CPU/Python; no Phase1 retraining and no feature re-export unless existing satellite NPZ files are missing. |

## Risks

| Risk | Mitigation |
|---|---|
| Removing clean view may increase target-old rejection and violate the 2pp old-drop gate. | Sweep only high source-accept quantiles first; if it fails, inspect same-row FAR/drop tradeoff before designing a new route. |
| Existing satellite NPZ files may be missing for a cell. | Script fails closed before evaluation for that cell and reports the missing paths. |
| N607 NumPy environment has a known `np.mean` compatibility issue. | Current evaluator already avoids boolean `np.mean` in threshold-rate metrics. |

## Linear Satellite-Only Result

The first satellite-only sweep completed 30 rows with no traceback. Each cell merged exactly three satellite views and excluded `clean.npz`.

Artifacts:

| Artifact | Local path |
|---|---|
| Linear satellite-only summary | `E:\type10-7\automation_reports\CV-SincNet\phase1_adv3b02_satonly_open_set_reject_20260702\artifacts\satonly_sourceq_sweep_summary.csv` |
| Linear satellite-only driver log | `E:\type10-7\automation_reports\CV-SincNet\phase1_adv3b02_satonly_open_set_reject_20260702\artifacts\satonly_sweep1_driver.out` |

Policy means:

| Policy | Cells | Mean unknown_FAR | Max unknown_FAR | Mean old drop pp | Max old drop pp | Dual pass |
|---|---:|---:|---:|---:|---:|---:|
| `SATMV_LIN_SRC9999` | 10 | 0.0922 | 0.1375 | 1.96 | 2.44 | 2/10 |
| `SATMV_LIN_SRC99995` | 10 | 0.0971 | 0.1475 | 1.88 | 2.38 | 2/10 |
| `SATMV_LIN_SRC1000` | 10 | 0.1063 | 0.1600 | 1.83 | 2.35 | 1/10 |

Linear same-score oracle diagnosis:

| Diagnostic | Finding |
|---|---|
| Global threshold oracle | 8/10 cells have no threshold that simultaneously satisfies `unknown_FAR<=0.05` and `old_drop_pp_vs_closed<=2.0`. |
| Class-conditional source threshold | Reduces mean FAR to 0.0239 at `q=1.0`, but raises old drop to 2.62-3.24pp across cells; 0/10 dual pass. |

Interpretation: removing the clean view makes the linear satellite-only risk score too weak. Simple source-quantile or class-conditional thresholds cannot meet both gates. The next step is a stronger source/proxy-only MLP risk head on the same satellite-only feature representation.

## MLP Satellite-Only Result

The MLP64 source-quantile sweep completed 30 rows with no traceback.

Artifacts:

| Artifact | Local path |
|---|---|
| MLP satellite-only summary | `E:\type10-7\automation_reports\CV-SincNet\phase1_adv3b02_satonly_open_set_reject_20260702\artifacts\satonly_mlp_sourceq_sweep_summary.csv` |
| MLP satellite-only driver log | `E:\type10-7\automation_reports\CV-SincNet\phase1_adv3b02_satonly_open_set_reject_20260702\artifacts\satonly_mlp_sweep1_driver.out` |

Policy means:

| Policy | Cells | Mean unknown_FAR | Max unknown_FAR | Mean old drop pp | Max old drop pp | Dual pass |
|---|---:|---:|---:|---:|---:|---:|
| `SATMV_MLP64_SRC9999` | 10 | 0.0193 | 0.0350 | 4.76 | 7.00 | 0/10 |
| `SATMV_MLP64_SRC99995` | 10 | 0.0198 | 0.0375 | 4.71 | 6.88 | 0/10 |
| `SATMV_MLP64_SRC1000` | 10 | 0.0210 | 0.0425 | 4.64 | 6.65 | 0/10 |

Interpretation: MLP64 solves the FAR side but rejects too many closed-correct target-old samples. A simple OR-combination of MLP reject with linear rescue was checked as a diagnostic; it did not produce a 10/10 dual-pass rule. The next route changes the training labels: source samples already misclassified by Phase1 are not treated as old positives for the rejection head.

## Correct-Known-Only Route

Hypothesis: because the old-class metric is `closed_correct AND accepted`, the rejection head should learn to preserve source samples that the frozen Phase1 classifier already gets right. Source old samples that are already closed-set errors do not help old full accuracy and can be used as auxiliary negative pressure together with source-side proxy unknown.

Configuration:

| Field | Value |
|---|---|
| Positive training samples | source old groups where Phase1 predicted TX equals true source TX |
| Negative training samples | proxy unknown groups plus source old groups where Phase1 predicted TX is wrong |
| Query view | satellite-only `leo_clear_weak`, `leo_low_elev_weak`, `leo_rain_weak`; no clean |
| Heads | `SATCORR_LIN`, `SATCORR_MLP64` |
| Policies | `SRC9999`, `SRC99995`, `SRC1000` |
| Success gate | `unknown_FAR<=0.05` and `old_drop_pp_vs_closed<=2.0` |

Correct-known-only result:

| Policy | Cells | Mean unknown_FAR | Max unknown_FAR | Mean old drop pp | Max old drop pp | Dual pass |
|---|---:|---:|---:|---:|---:|---:|
| `SATCORR_LIN_SRC9999` | 10 | 0.0835 | 0.1416 | 2.16 | 2.91 | 0/10 |
| `SATCORR_LIN_SRC99995` | 10 | 0.0865 | 0.1459 | 2.11 | 2.82 | 0/10 |
| `SATCORR_LIN_SRC1000` | 10 | 0.0923 | 0.1502 | 2.05 | 2.76 | 0/10 |
| `SATCORR_MLP64_SRC9999` | 10 | 0.0498 | 0.0810 | 7.86 | 14.38 | 0/10 |
| `SATCORR_MLP64_SRC99995` | 10 | 0.0498 | 0.0810 | 7.79 | 14.21 | 0/10 |
| `SATCORR_MLP64_SRC1000` | 10 | 0.0498 | 0.0810 | 7.72 | 14.15 | 0/10 |

Artifacts:

| Artifact | Local path |
|---|---|
| Correct-known summary | `E:\type10-7\automation_reports\CV-SincNet\phase1_adv3b02_satonly_open_set_reject_20260702\artifacts\satonly_correct_known_sweep_summary.csv` |
| Correct-known driver log | `E:\type10-7\automation_reports\CV-SincNet\phase1_adv3b02_satonly_open_set_reject_20260702\artifacts\satonly_correct_sweep1_driver.out` |

Additional read-only diagnostics:

| Diagnostic | Result |
|---|---|
| MLP accept OR linear rescue | Best checked deployable combinations did not exceed 1/10 dual pass. Reducing FAR raised old drop above 2pp; rescuing old correctness raised FAR. |
| Source-thresholded vote consistency | FAR remained too high; best low-drop rules had mean FAR above 0.53. |
| Source-thresholded confidence | FAR remained near 1.0 when old drop stayed below 2pp. |
| Source-thresholded cosine consistency | Low-FAR settings required old drop around 3pp or higher. |

Current conclusion: under the stricter satellite-only condition with no clean query view and `old_drop_pp_vs_closed<=2.0`, the tested Phase1-only rejection routes have not achieved the target. The previous clean+satellite `MV_LIN_SRC1000` success does not transfer to satellite-only. The nearest observed tradeoffs are:

| Route | Strength | Failure mode |
|---|---|---|
| `SATMV_LIN_*` | Keeps old drop near 2pp | FAR remains around 9-16% on many cells. |
| `SATMV_MLP64_*` | Keeps FAR below 5% | Old drop rises to 3-7pp. |
| Class-conditional/source-combination diagnostics | Can lower FAR without target labels | Old drop remains above 2pp for most cells. |

Recommended next experiment: introduce a source-only calibration objective that explicitly optimizes the two target metrics at training time, e.g. a constrained proxy-loss or Neyman-Pearson style head trained to maximize source closed-correct retention under proxy-FAR pressure, rather than post-hoc thresholding of the current BCE heads.

## Margin-Loss Route

Hypothesis: BCE heads create a poor target tradeoff because the final source-quantile threshold is post-hoc. The margin-loss route trains risk logits directly:

```text
source old loss = softplus(logit + margin)
proxy unknown loss = softplus(margin - logit)
```

This keeps training source/proxy-only and does not use target support, target labels, or unknown query tuning. The sweep tests linear heads with stronger proxy pressure and MLP heads with stronger source-retention pressure:

| Candidate family | Purpose |
|---|---|
| `SATMARG_LIN_P2/P5` | Push linear scores to reject more proxy unknown while checking whether old drop stays near 2pp. |
| `SATMARG_MLP_S2/S5/S10/S20` | Increase source retention pressure on the MLP family that previously had low FAR but high old drop. |

Git-backed mirror commit:

| Commit | Purpose |
|---|---|
| `2d601dd` | Add margin-loss training mode and satellite-only margin sweep. |

Margin-loss result:

| Policy | Cells | Mean unknown_FAR | Max unknown_FAR | Mean old drop pp | Max old drop pp | Dual pass |
|---|---:|---:|---:|---:|---:|---:|
| `SATMARG_LIN_P2_SRC9999` | 10 | 0.1712 | 0.2778 | 1.15 | 1.74 | 1/10 |
| `SATMARG_LIN_P2_SRC1000` | 10 | 0.2165 | 0.3500 | 0.93 | 1.50 | 1/10 |
| `SATMARG_LIN_P5_SRC9999` | 10 | 0.2386 | 0.3833 | 0.71 | 1.09 | 1/10 |
| `SATMARG_LIN_P5_SRC1000` | 10 | 0.3598 | 0.5611 | 0.43 | 0.65 | 0/10 |
| `SATMARG_MLP_S2_SRC9999` | 10 | 0.0346 | 0.0561 | 5.55 | 7.82 | 0/10 |
| `SATMARG_MLP_S2_SRC1000` | 10 | 0.0374 | 0.0623 | 5.22 | 7.15 | 0/10 |
| `SATMARG_MLP_S5_SRC9999` | 10 | 0.0309 | 0.0550 | 6.01 | 8.47 | 0/10 |
| `SATMARG_MLP_S5_SRC1000` | 10 | 0.0322 | 0.0575 | 5.76 | 8.00 | 0/10 |
| `SATMARG_MLP_S10_SRC9999` | 10 | 0.0147 | 0.0280 | 6.61 | 9.68 | 0/10 |
| `SATMARG_MLP_S10_SRC1000` | 10 | 0.0151 | 0.0280 | 6.36 | 9.09 | 0/10 |
| `SATMARG_MLP_S20_SRC9999` | 10 | 0.0121 | 0.0218 | 6.02 | 8.09 | 0/10 |
| `SATMARG_MLP_S20_SRC1000` | 10 | 0.0124 | 0.0249 | 5.87 | 7.71 | 0/10 |

Artifacts:

| Artifact | Local path |
|---|---|
| Margin-loss summary | `E:\type10-7\automation_reports\CV-SincNet\phase1_adv3b02_satonly_open_set_reject_20260702\artifacts\satonly_margin_sweep_summary.csv` |
| Margin-loss driver log | `E:\type10-7\automation_reports\CV-SincNet\phase1_adv3b02_satonly_open_set_reject_20260702\artifacts\satonly_margin_sweep1_driver.out` |

Updated conclusion: margin loss sharpens the same failure split instead of resolving it. Linear margin heads preserve old-class performance but leave FAR far above 5%; MLP margin heads reduce FAR but reject too many closed-correct old-class target samples. The objective remains unachieved under current Phase1-only satellite-only evidence.

Next route should change the representation or training data available to the rejector rather than only the head objective, for example source-only satellite-augmented feature adaptation or a compact calibration layer trained during Phase1 to improve satellite-only old/unknown separability before thresholding.

## Protocol Correction: Unknown LEO Single Observation

User review identified an important protocol issue: the earlier `satonly` sweeps excluded `clean.npz`, but still merged three known scenario exports (`leo_clear_weak`, `leo_low_elev_weak`, `leo_rain_weak`) as repeated views for the same query sample. That is useful as a robustness diagnostic, but it is too strong for the stricter deployment view where the receiver only observes one signal that has already passed through an unknown LEO channel. In that setting the evaluator must not receive three counterfactual LEO scenario views for the same clean source sample.

Corrected evaluation rule:

| Field | Corrected value |
|---|---|
| Query observation | One satellite-observed feature row per sample metadata key |
| Clean view | Excluded |
| Scenario knowledge at inference | Not used by the evaluator |
| Scenario source | Stable hidden assignment from the existing `leo_clear_weak`, `leo_low_elev_weak`, `leo_rain_weak` exports |
| Multi-view status | Disabled for target query; later multi-view work must use actual multiple received packets/windows or TTA views derived from the one received satellite signal |
| Threshold scope | Source old and source-side proxy unknown only; no target support, no target labels, no unknown query tuning |

New local files:

| File | Purpose |
|---|---|
| `E:\type10-7\code\scripts\eval_phase1_prototype_reject.py` | Source-only class-prototype rejection evaluator for single satellite observations. |
| `E:\type10-7\code\scripts\sweep_phase1_adv3b02_satunknown_singleview_20260702.sh` | Builds `features_satunknown_singleview.npz` by selecting one hidden LEO scenario observation per sample metadata key, then evaluates linear/MLP risk heads and prototype scores. |
| `E:\type10-7\code\tests\test_phase1_multiview_reject_eval.py` | Adds a regression test for one-observation satellite prototype rejection. |

Local verification:

| Command | Result |
|---|---|
| `C:\Users\lh594\.conda\envs\ssr-gpu\python.exe -m py_compile code\scripts\eval_phase1_multiview_reject.py code\scripts\eval_phase1_prototype_reject.py code\tests\test_phase1_multiview_reject_eval.py` | PASS |
| `bash -n code/scripts/sweep_phase1_adv3b02_satunknown_singleview_20260702.sh` | PASS |
| `C:\Users\lh594\.conda\envs\ssr-gpu\python.exe -m pytest code\tests\test_phase1_multiview_reject_eval.py -q` | PASS, 2 tests; `.pytest_cache` permission warning only |

Planned corrected N607 command:

```bash
cd /home/szu2070436088/2510044040/CV-SincNet && bash code/scripts/sweep_phase1_adv3b02_satunknown_singleview_20260702.sh
```

Expected corrected summary:

```text
/home/szu2070436088/2510044040/CV-SincNet/logs/phase1_adv3b02_satunknown_singleview_matrix_20260702/satunknown_singleview_sweep_summary.csv
```

Until this corrected single-observation matrix completes, no prior `SATMV_*` or `SATMARG_*` row should be described as satisfying the stricter “received signal has unknown LEO channel and no clean/counterfactual scenario views” condition.

Launch status:

| Field | Value |
|---|---|
| Launch time | `2026-07-02T16:32:46+08:00` on N607 |
| Remote driver PID | `4090699` |
| Remote command | `cd /home/szu2070436088/2510044040/CV-SincNet && nohup bash code/scripts/sweep_phase1_adv3b02_satunknown_singleview_20260702.sh > logs/phase1_adv3b02_satunknown_singleview_matrix_20260702/driver.out 2>&1 &` |
| Driver log | `/home/szu2070436088/2510044040/CV-SincNet/logs/phase1_adv3b02_satunknown_singleview_matrix_20260702/driver.out` |
| Summary CSV | `/home/szu2070436088/2510044040/CV-SincNet/logs/phase1_adv3b02_satunknown_singleview_matrix_20260702/satunknown_singleview_sweep_summary.csv` |
| First-cell single-observation NPZ | `rows=15362`, `contains_clean_view=false`, scenario counts `leo_clear_weak=5141`, `leo_low_elev_weak=5141`, `leo_rain_weak=5080` |

Remote verification before launch:

| Command | Result |
|---|---|
| `/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python -m py_compile code/scripts/eval_phase1_prototype_reject.py code/scripts/eval_phase1_multiview_reject.py code/tests/test_phase1_multiview_reject_eval.py` | PASS |
| `bash -n code/scripts/sweep_phase1_adv3b02_satunknown_singleview_20260702.sh` | PASS |
| `sha256sum` for synced code/report | PASS; report hash `df32374483df768c58308066fba49d768a2a991c2c8ea61e7a010018df2870f5` |

Corrected single-observation result:

| Policy | Cells | Mean unknown_FAR | Max unknown_FAR | Mean old drop pp | Max old drop pp | Dual pass |
|---|---:|---:|---:|---:|---:|---:|
| `SATUNK_LIN_SRC9999` | 10 | 0.9804 | 1.0000 | 0.06 | 0.15 | 0/10 |
| `SATUNK_LIN_SRC1000` | 10 | 0.9827 | 1.0000 | 0.05 | 0.15 | 0/10 |
| `SATUNK_MLP64_SRC9999` | 10 | 0.8800 | 0.9614 | 1.93 | 4.53 | 0/10 |
| `SATUNK_MLP64_SRC1000` | 10 | 0.9063 | 0.9657 | 1.31 | 3.24 | 0/10 |
| `SATUNK_PROTO_COS_SRC9999` | 10 | 0.9915 | 1.0000 | 0.07 | 0.29 | 0/10 |
| `SATUNK_PROTO_COS_SRC1000` | 10 | 0.9917 | 1.0000 | 0.06 | 0.24 | 0/10 |
| `SATUNK_PROTO_MAH_SRC9999` | 10 | 0.9899 | 1.0000 | 0.22 | 0.47 | 0/10 |
| `SATUNK_PROTO_MAH_SRC1000` | 10 | 0.9899 | 1.0000 | 0.22 | 0.47 | 0/10 |

Artifacts:

| Artifact | Local path |
|---|---|
| Single-observation summary | `E:\type10-7\automation_reports\CV-SincNet\phase1_adv3b02_satonly_open_set_reject_20260702\artifacts\satunknown_singleview_sweep_summary.csv` |
| Single-observation driver log | `E:\type10-7\automation_reports\CV-SincNet\phase1_adv3b02_satonly_open_set_reject_20260702\artifacts\satunknown_singleview_driver.out` |

Interpretation: after correcting the protocol to one unknown LEO observation per sample, the previous head/threshold/prototype families collapse on FAR. The old-class drop stays small for linear/prototype routes only because the threshold accepts almost all samples, including unknowns. Therefore this baseline does not satisfy the objective and should be treated as the corrected negative control. The next aligned route is receive-side TTA multi-view: generate one `x_sat`, then derive identity-preserving views from that received signal using small time shifts, residual CFO/phase hypotheses, and normalization variants.

## Receive-Side TTA Multi-View Route

Protocol: generate exactly one satellite observation `x_sat` for each raw WiSig sample, then derive multiple test-time views from that received signal. This avoids the invalid counterfactual pattern of feeding separate `leo_clear_weak`, `leo_low_elev_weak`, and `leo_rain_weak` versions of the same clean sample to the evaluator.

Implemented TTA policy:

| Policy | Views | Description |
|---|---:|---|
| `rx_light5` | 5 | `rx_base`, `rx_shift_m2`, `rx_shift_p2`, `rx_cfo_m1e4`, `rx_cfo_p1e4` |

Rationale: small time shifts and residual CFO correction hypotheses are receiver-side synchronization/preprocessing alternatives that can be derived from one received satellite signal. They do not reapply a new LEO channel and do not require knowing whether the original channel was clear, low-elevation, or rain.

New local files/changes:

| File | Purpose |
|---|---|
| `E:\type10-7\code\export_spaceborne_features.py` | Adds `--satellite_tta_policy rx_light5`, applied after `apply_sat_channel_for_scenario` and before model feature extraction. Default remains `none`. |
| `E:\type10-7\code\scripts\sweep_phase1_adv3b02_sattta_rxlight_20260702.sh` | Re-exports Phase1 frozen features with receive-side TTA and evaluates `SATTA_LIN_*` and `SATTA_MLP64_*`. |

Local verification:

| Command | Result |
|---|---|
| `C:\Users\lh594\.conda\envs\ssr-gpu\python.exe -m py_compile code\export_spaceborne_features.py code\scripts\eval_phase1_multiview_reject.py code\scripts\eval_phase1_prototype_reject.py code\tests\test_phase1_multiview_reject_eval.py` | PASS |
| `bash -n code/scripts/sweep_phase1_adv3b02_sattta_rxlight_20260702.sh` | PASS |
| Minimal `_satellite_tta_views(torch.randn(3,2,16),'rx_light5')` shape check | PASS, 5 views, all shape `(3,2,16)` |
| `C:\Users\lh594\.conda\envs\ssr-gpu\python.exe -m pytest code\tests\test_phase1_multiview_reject_eval.py -q` | PASS, 2 tests; `.pytest_cache` permission warning only |

Planned N607 command:

```bash
cd /home/szu2070436088/2510044040/CV-SincNet && bash code/scripts/sweep_phase1_adv3b02_sattta_rxlight_20260702.sh
```

Expected TTA summary:

```text
/home/szu2070436088/2510044040/CV-SincNet/logs/phase1_adv3b02_sattta_rxlight_matrix_20260702/sattta_rxlight_sweep_summary.csv
```

Launch status:

| Field | Value |
|---|---|
| Launch time | `2026-07-02T16:50:06+08:00` on N607 |
| Remote driver PID | `4107901` |
| GPU allocation | `GPU=7`, script uses `CUDA_VISIBLE_DEVICES=7` and `--device cuda:0` |
| Remote command | `cd /home/szu2070436088/2510044040/CV-SincNet && nohup env GPU=7 bash code/scripts/sweep_phase1_adv3b02_sattta_rxlight_20260702.sh > logs/phase1_adv3b02_sattta_rxlight_matrix_20260702/driver.out 2>&1 &` |
| Driver log | `/home/szu2070436088/2510044040/CV-SincNet/logs/phase1_adv3b02_sattta_rxlight_matrix_20260702/driver.out` |
| Summary CSV | `/home/szu2070436088/2510044040/CV-SincNet/logs/phase1_adv3b02_sattta_rxlight_matrix_20260702/sattta_rxlight_sweep_summary.csv` |
| Startup health | PASS; entered second cell, GPU7 memory about `2421/24576 MiB`, no traceback in driver tail |

Remote verification before launch:

| Command | Result |
|---|---|
| `/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python -m py_compile code/export_spaceborne_features.py code/scripts/eval_phase1_multiview_reject.py` | PASS |
| `bash -n code/scripts/sweep_phase1_adv3b02_sattta_rxlight_20260702.sh` | PASS |
| `sha256sum` for synced code/report | PASS; report hash `582bd9a0b781b404f0a6bffb5cb34f899456e6991c8b3ebaa1efa560eb1f01e9` before launch-status update |

Receive-side TTA result:

| Policy | Cells | Mean unknown_FAR | Max unknown_FAR | Mean old drop pp | Max old drop pp | Dual pass |
|---|---:|---:|---:|---:|---:|---:|
| `SATTA_LIN_SRC9999` | 10 | 0.9462 | 0.9889 | 0.40 | 0.75 | 0/10 |
| `SATTA_LIN_SRC1000` | 10 | 0.9533 | 0.9925 | 0.28 | 0.67 | 0/10 |
| `SATTA_MLP64_SRC9999` | 10 | 0.4712 | 0.5875 | 16.03 | 26.25 | 0/10 |
| `SATTA_MLP64_SRC1000` | 10 | 0.4747 | 0.5925 | 15.95 | 26.25 | 0/10 |

Artifacts:

| Artifact | Local path |
|---|---|
| Receive-side TTA summary | `E:\type10-7\automation_reports\CV-SincNet\phase1_adv3b02_satonly_open_set_reject_20260702\artifacts\sattta_rxlight_sweep_summary.csv` |
| Receive-side TTA driver log | `E:\type10-7\automation_reports\CV-SincNet\phase1_adv3b02_satonly_open_set_reject_20260702\artifacts\sattta_rxlight_driver.out` |

Interpretation: receive-side TTA is protocol-correct and improves the MLP FAR relative to the single-observation baseline, but it still misses the 5% FAR target by a wide margin and the MLP family damages old-class retention far beyond 2pp. Linear heads retain old classes but accept almost all unknowns. The objective remains unachieved.

Receive-side TTA low-quantile diagnostic:

| Policy | Cells | Mean unknown_FAR | Max unknown_FAR | Mean old drop pp | Max old drop pp | Dual pass |
|---|---:|---:|---:|---:|---:|---:|
| `SATTA_LIN_SRC950` | 10 | 0.5124 | 0.7253 | 11.33 | 18.75 | 0/10 |
| `SATTA_LIN_SRC980` | 10 | 0.6022 | 0.7940 | 6.93 | 12.83 | 0/10 |
| `SATTA_LIN_SRC990` | 10 | 0.6728 | 0.8541 | 4.45 | 8.67 | 0/10 |
| `SATTA_LIN_SRC995` | 10 | 0.7158 | 0.8884 | 3.70 | 7.00 | 0/10 |
| `SATTA_LIN_SRC999` | 10 | 0.8773 | 0.9528 | 1.32 | 2.42 | 0/10 |
| `SATTA_LIN_SRC9999` | 10 | 0.9462 | 0.9889 | 0.40 | 0.75 | 0/10 |
| `SATTA_LIN_SRC1000` | 10 | 0.9533 | 0.9925 | 0.28 | 0.67 | 0/10 |

Artifact:

| Artifact | Local path |
|---|---|
| Receive-side TTA low-quantile summary | `E:\type10-7\automation_reports\CV-SincNet\phase1_adv3b02_satonly_open_set_reject_20260702\artifacts\sattta_rxlight_lowq_summary.csv` |

Interpretation: source-threshold tightening exposes the same tradeoff. Lower thresholds reduce FAR only to about 51% at best and already damage old classes by more than 11pp. Near the 2pp old-drop gate, FAR remains above 87%. Under the current frozen `ADV3B02_CORE90_SOFT_E200phase1` feature space, receive-side TTA plus post-hoc source thresholds is insufficient.

## Receive-Side TTA Proxy/Prototype Diagnostic

Planned diagnostic: reuse the completed `rx_light5` feature exports and evaluate source/proxy-only thresholds that were not covered by the first TTA run:

| Family | Purpose |
|---|---|
| `SATTAD_MLP64_PROXY*` | Test whether proxy-unknown quantile thresholds can lower FAR without using target/unknown query tuning. |
| `SATTAD_MLP64_MIN*` | Use the stricter of source-accept and proxy-FAR thresholds. |
| `SATTAD_PROTO_*` | Test class-prototype TTA scores with source/proxy-only calibration. |

Boundary: this diagnostic does not regenerate target query from clean, does not use target support, and does not use target-old or unknown query metrics to choose thresholds. It only checks whether the completed `rx_light5` feature space contains an exploitable source/proxy open-set boundary.

Status update: the ad-hoc remote proxy/prototype diagnostic command disconnected before producing `sattta_rxlight_proxy_proto_summary.csv`; follow-up process checks found no remaining diagnostic process. This path is superseded by the stricter non-TTA KNN density sweep below because the current objective should not be limited to receive-side TTA.

## Single-Observation KNN Density Diagnostic

Objective: test a non-TTA rejection mechanism on the strict single-observation LEO feature files. The method freezes `ADV3B02_CORE90_SOFT_E200phase1`, keeps each target query as one hidden LEO observation, and rejects samples by kNN distance to source-old features of the predicted old TX class.

Status update: this diagnostic was implemented and synced but not launched after the operator clarified that the preferred route remains multi-view rejection. It is retained as diagnostic code only and is not the active experiment path for the current objective.

Boundary:

| Item | Setting |
|---|---|
| Feature input | `ADV3B02_CORE90_SOFT_E200_PHASE1_SATUNKNOWN_SINGLEVIEW/features_satunknown_singleview.npz` |
| Clean view | excluded |
| Per-sample observation count | 1 |
| Target support | none |
| Threshold data | source old plus source-side proxy unknown only |
| Unknown query usage | evaluation only, never threshold tuning |
| Success gate | `unknown_FAR<=0.05` and `old_drop_pp_vs_closed<=2.0` |

Local files changed:

| File | Purpose |
|---|---|
| `code/scripts/eval_phase1_knn_reject.py` | Frozen-feature kNN density evaluator with source/proxy-only thresholds. |
| `code/scripts/sweep_phase1_adv3b02_satunknown_knn_density_20260702.sh` | 10-cell sweep over cosine/euclidean kNN, source/proxy thresholds, class-conditional thresholds, and source-incorrect proxy variants. |

Local verification:

| Command | Result |
|---|---|
| `C:\Users\lh594\.conda\envs\ssr-gpu\python.exe -m py_compile code\scripts\eval_phase1_knn_reject.py` | PASS |
| `bash -n code/scripts/sweep_phase1_adv3b02_satunknown_knn_density_20260702.sh` | PASS |

Planned remote command:

```bash
cd /home/szu2070436088/2510044040/CV-SincNet && bash code/scripts/sweep_phase1_adv3b02_satunknown_knn_density_20260702.sh
```

Expected KNN summary:

```text
/home/szu2070436088/2510044040/CV-SincNet/logs/phase1_adv3b02_satunknown_knn_density_matrix_20260702/satunknown_knn_density_sweep_summary.csv
```

## Satellite Physical Multi-View Diagnostic

Objective: keep the original multi-view rejection framework, but replace invalid counterfactual clean/LEO scenario views with receive-side views generated from a single already-LEO-observed `x_sat`. This is closer to Phase2 deployment: the test signal is known to include LEO star-ground impairment, but the receiver does not know whether it is clear, low-elevation, or rain.

New view policy:

| Policy | Views | Rationale |
|---|---:|---|
| `sat_rx_phys11` | 11 | Single-observation receive-side physical views: base, residual CFO grid, one-sample timing offsets, small constant phase offsets, AGC clipping normalization, and short-FIR smoothing. |

View list:

```text
rx_base
rx_cfo_m2e4
rx_cfo_m1e4
rx_cfo_p1e4
rx_cfo_p2e4
rx_shift_m1
rx_shift_p1
rx_phase_m15deg
rx_phase_p15deg
rx_agc_clip2p5
rx_short_fir
```

Boundary:

| Item | Setting |
|---|---|
| Feature input | one LEO observation per raw sample before receive-side view generation |
| Clean view | excluded |
| Target support | none |
| Threshold data | source old plus source-side proxy unknown only |
| Unknown query usage | evaluation only, never threshold tuning |
| Evaluator | existing `eval_phase1_multiview_reject.py` consistency/rejection head |
| Success gate | `unknown_FAR<=0.05` and `old_drop_pp_vs_closed<=2.0` |

Local files changed:

| File | Purpose |
|---|---|
| `code/export_spaceborne_features.py` | Add `sat_rx_phys11` receive-side physical multi-view policy and dynamic manifest view count. |
| `code/scripts/sweep_phase1_adv3b02_satphysmv11_20260702.sh` | 10-cell sweep using the existing multi-view rejection evaluator with source/proxy threshold variants. |

Local verification:

| Command | Result |
|---|---|
| `C:\Users\lh594\.conda\envs\ssr-gpu\python.exe -m py_compile code\export_spaceborne_features.py` | PASS |
| `bash -n code/scripts/sweep_phase1_adv3b02_satphysmv11_20260702.sh` | PASS |
| importlib smoke test for `_satellite_tta_views(..., "sat_rx_phys11")` | PASS; 11 finite views, all shaped like input IQ |

Planned remote command:

```bash
cd /home/szu2070436088/2510044040/CV-SincNet && bash code/scripts/sweep_phase1_adv3b02_satphysmv11_20260702.sh
```

Expected physical multi-view summary:

```text
/home/szu2070436088/2510044040/CV-SincNet/logs/phase1_adv3b02_satphysmv11_matrix_20260702/satphysmv11_sweep_summary.csv
```

Launch and completion status:

| Field | Value |
|---|---|
| Launch time | `2026-07-02T17:34:58+08:00` on N607 |
| Remote driver PID | `4164199` |
| GPU allocation | `GPU=7`, script uses `CUDA_VISIBLE_DEVICES=7` and `--device cuda:0` |
| Remote command | `cd /home/szu2070436088/2510044040/CV-SincNet && nohup env GPU=7 bash code/scripts/sweep_phase1_adv3b02_satphysmv11_20260702.sh > logs/phase1_adv3b02_satphysmv11_matrix_20260702/driver.out 2>&1 &` |
| Driver log | `/home/szu2070436088/2510044040/CV-SincNet/logs/phase1_adv3b02_satphysmv11_matrix_20260702/driver.out` |
| Summary CSV | `/home/szu2070436088/2510044040/CV-SincNet/logs/phase1_adv3b02_satphysmv11_matrix_20260702/satphysmv11_sweep_summary.csv` |
| Completion time | `2026-07-02T17:51:55+08:00` |
| Summary rows | 100 rows, 10 cells x 10 policies |
| Startup health | PASS; first cell exported and entered evaluator, no traceback/OOM/argparse error |
| Final process cleanup | Remote experiment process ended; local SSH client cleaned up |

Remote verification before launch:

| Command | Result |
|---|---|
| `/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python -m py_compile code/export_spaceborne_features.py` | PASS |
| `bash -n code/scripts/sweep_phase1_adv3b02_satphysmv11_20260702.sh` | PASS |
| `sha256sum` for synced code/report | PASS; `export_spaceborne_features.py` hash `5894b761eedae3a58c8b0b8c3f01991b74256eb3d7edc3c1c7b587a73b5e7e48`, sweep script hash `83ffd1227d965251ee63f72b7c5bdd4fd57f5a1c29cfe59a7db0893b22d65aa3`, report hash `360f7b68cc585c9eeadc28c3146528ab8fbe1205811dcb7f6d48291570c58bd1` |

Artifacts:

| Artifact | Local path |
|---|---|
| Physical multi-view summary | `E:\type10-7\automation_reports\CV-SincNet\phase1_adv3b02_satonly_open_set_reject_20260702\artifacts\satphysmv11_sweep_summary.csv` |
| Physical multi-view driver log | `E:\type10-7\automation_reports\CV-SincNet\phase1_adv3b02_satonly_open_set_reject_20260702\artifacts\satphysmv11_driver.out` |

Policy aggregate result:

| Policy | Cells | Mean unknown_FAR | Max unknown_FAR | Mean old drop pp | Max old drop pp | Dual pass | Mean known coverage |
|---|---:|---:|---:|---:|---:|---:|---:|
| `SATPHY11_LIN_MIN05` | 10 | 0.1578 | 0.2333 | 44.30 | 51.33 | 0/10 | 0.3490 |
| `SATPHY11_LIN_PROXY05` | 10 | 0.1578 | 0.2333 | 44.30 | 51.33 | 0/10 | 0.3490 |
| `SATPHY11_MLP64_MARGIN_MIN05` | 10 | 0.1745 | 0.2675 | 40.43 | 44.75 | 0/10 | 0.3975 |
| `SATPHY11_MLP64_MARGIN_PROXY05` | 10 | 0.1745 | 0.2675 | 40.43 | 44.75 | 0/10 | 0.3975 |
| `SATPHY11_MLP64_MIN05` | 10 | 0.2510 | 0.3550 | 34.75 | 40.42 | 0/10 | 0.4743 |
| `SATPHY11_MLP64_PROXY05` | 10 | 0.2510 | 0.3550 | 34.75 | 40.42 | 0/10 | 0.4743 |
| `SATPHY11_MLP64_SRC9999` | 10 | 0.4745 | 0.6137 | 19.10 | 28.58 | 0/10 | 0.6962 |
| `SATPHY11_MLP64_MARGIN_SRC5` | 10 | 0.5150 | 0.6953 | 17.05 | 28.25 | 0/10 | 0.7282 |
| `SATPHY11_LIN_SRC999` | 10 | 0.8246 | 0.9333 | 3.65 | 8.00 | 0/10 | 0.9380 |
| `SATPHY11_LIN_SRC9999` | 10 | 0.8336 | 0.9450 | 3.18 | 6.75 | 0/10 | 0.9452 |

Best low-FAR rows:

| Run | Policy | unknown_FAR | Old drop pp | Closed old acc | Full old acc after reject | Known coverage | Dual pass |
|---|---|---:|---:|---:|---:|---:|---|
| `phase1_adv3b02_satphysmv11_rx20_1_u1_20260702` | `SATPHY11_LIN_MIN05` | 0.0224 | 43.50 | 0.6408 | 0.2058 | 0.3150 | false |
| `phase1_adv3b02_satphysmv11_rx20_1_u1_20260702` | `SATPHY11_LIN_PROXY05` | 0.0224 | 43.50 | 0.6408 | 0.2058 | 0.3150 | false |
| `phase1_adv3b02_satphysmv11_rx20_1_u1_20260702` | `SATPHY11_MLP64_MARGIN_MIN05` | 0.0280 | 40.00 | 0.6408 | 0.2408 | 0.3517 | false |
| `phase1_adv3b02_satphysmv11_rx20_1_u1_20260702` | `SATPHY11_MLP64_MARGIN_PROXY05` | 0.0280 | 40.00 | 0.6408 | 0.2408 | 0.3517 | false |

Best old-retention rows:

| Run | Policy | unknown_FAR | Old drop pp | Closed old acc | Full old acc after reject | Known coverage | Dual pass |
|---|---|---:|---:|---:|---:|---:|---|
| `phase1_adv3b02_satphysmv11_rx7_7_u10_20260702` | `SATPHY11_LIN_SRC9999` | 0.9142 | 1.17 | 0.8167 | 0.8050 | 0.9858 | false |
| `phase1_adv3b02_satphysmv11_rx7_7_u1_20260702` | `SATPHY11_LIN_SRC9999` | 0.9450 | 1.17 | 0.8167 | 0.8050 | 0.9858 | false |
| `phase1_adv3b02_satphysmv11_rx7_7_u10_20260702` | `SATPHY11_LIN_SRC999` | 0.9099 | 1.33 | 0.8167 | 0.8033 | 0.9808 | false |
| `phase1_adv3b02_satphysmv11_rx7_7_u1_20260702` | `SATPHY11_LIN_SRC999` | 0.9325 | 1.33 | 0.8167 | 0.8033 | 0.9808 | false |

Interpretation: the physical receive-side multi-view route is protocol-correct and confirms that multi-view features contain some unknown signal: 4/100 rows pass the 5% FAR gate. However, every low-FAR row rejects too many old-class samples and damages old-class full accuracy by about 40pp. Conversely, rows that keep old drop under 2pp leave FAR between about 0.80 and 0.95. The current `ADV3B02_CORE90_SOFT_E200phase1` frozen feature space still does not satisfy the joint gate `unknown_FAR<=0.05` and old drop `<=2pp` under strict single-observation LEO deployment.

## Physical Multi-View Guarded Diagnostics

Two lightweight source-calibrated rescue diagnostics were run on the completed `sat_rx_phys11` score tables/features. They do not use target labels to define thresholds; they only test whether the already-computed low-FAR rows can recover old-class retention using source-old criteria.

| Diagnostic | Input | Rule | Result |
|---|---|---|---|
| `analyze_satphysmv11_guarded_rescue_20260702.py` | `SATPHY11_*` score tables | Accept if original reject head accepts, or source-calibrated high confidence/vote/cosine multi-view stability rescues. | 0 dual-pass rows; best low-FAR rows regain old coverage only by raising FAR to 0.33-0.85. |
| `analyze_satphysmv11_hybrid_proto_rescue_20260702.py` | `SATPHY11_*` score tables plus `features_satphysmv11.npz` | Accept if original reject head accepts, or source class-prototype distance says the sample is old-like. | 0 dual-pass rows; prototype rescue reduces old drop but raises FAR to 0.36-0.91. |

Interpretation: post-hoc rescue is not enough. The low-FAR reject heads are not merely over-thresholded; the target-old samples they reject overlap strongly with unknown under both multi-view stability and source-prototype closeness. The next test should change the rejection-head training objective itself to penalize source-old rejection more strongly.

## Physical Multi-View Constrained Head Sweep

Objective: reuse existing `sat_rx_phys11` feature exports and train multi-view rejection heads with explicit source-old protection. This keeps the user-preferred multi-view rejection framework but changes the optimization objective from a balanced source/proxy separator to a constrained separator where source-old false rejection is expensive.

Boundary:

| Item | Setting |
|---|---|
| Feature input | existing `ADV3B02_CORE90_SOFT_E200_PHASE1_SATPHYSMV11/features_satphysmv11.npz` |
| Re-export features | no |
| Clean view | excluded |
| Target support | none |
| Threshold data | source old plus source-side proxy unknown only |
| Unknown query usage | evaluation only, never threshold tuning |
| Success gate | `unknown_FAR<=0.05` and `old_drop_pp_vs_closed<=2.0` |

Planned policies:

| Family | Purpose |
|---|---|
| `SATPHY11C_MLP_M{10,20,50,100}_PROXY05` | Increase source-old penalty under margin loss while preserving proxy-FAR threshold. |
| `SATPHY11C_MLP_M{20,50}_PROXY02/MIN02` | Tighten proxy quantile to test lower FAR with constrained source loss. |
| `SATPHY11C_MLP_M{20,50,100}_SRC9999` | Measure the old-retention floor when using source-accept thresholds. |
| `SATPHY11C_LIN_M{20,50}_PROXY05` | Linear constrained control. |
| `SATPHY11C_MLP_M{20,50}_COR_*` | Train only on source samples that the Phase1 closed classifier already predicts correctly. |

Local files changed:

| File | Purpose |
|---|---|
| `code/scripts/analyze_satphysmv11_guarded_rescue_20260702.py` | Source-calibrated confidence/vote/cosine rescue diagnostic. |
| `code/scripts/analyze_satphysmv11_hybrid_proto_rescue_20260702.py` | Source-prototype rescue diagnostic. |
| `code/scripts/sweep_phase1_adv3b02_satphysmv11_constrained_20260702.sh` | Reuse `sat_rx_phys11` features for constrained source-old-protected rejection heads. |

Local verification:

| Command | Result |
|---|---|
| `C:\Users\lh594\.conda\envs\ssr-gpu\python.exe -m py_compile code\scripts\analyze_satphysmv11_guarded_rescue_20260702.py` | PASS |
| `C:\Users\lh594\.conda\envs\ssr-gpu\python.exe -m py_compile code\scripts\analyze_satphysmv11_hybrid_proto_rescue_20260702.py` | PASS |
| `bash -n code/scripts/sweep_phase1_adv3b02_satphysmv11_constrained_20260702.sh` | PASS |

Planned remote command:

```bash
cd /home/szu2070436088/2510044040/CV-SincNet && bash code/scripts/sweep_phase1_adv3b02_satphysmv11_constrained_20260702.sh
```

Expected constrained summary:

```text
/home/szu2070436088/2510044040/CV-SincNet/logs/phase1_adv3b02_satphysmv11_constrained_matrix_20260702/satphysmv11_constrained_sweep_summary.csv
```

Launch and completion status:

| Field | Value |
|---|---|
| Launch time | `2026-07-02T18:08:01+08:00` on N607 |
| Remote driver PID | `4194270` |
| Feature export | reused existing `sat_rx_phys11` feature files; no re-export |
| Remote command | `cd /home/szu2070436088/2510044040/CV-SincNet && nohup bash code/scripts/sweep_phase1_adv3b02_satphysmv11_constrained_20260702.sh > logs/phase1_adv3b02_satphysmv11_constrained_matrix_20260702/driver.out 2>&1 &` |
| Driver log | `/home/szu2070436088/2510044040/CV-SincNet/logs/phase1_adv3b02_satphysmv11_constrained_matrix_20260702/driver.out` |
| Summary CSV | `/home/szu2070436088/2510044040/CV-SincNet/logs/phase1_adv3b02_satphysmv11_constrained_matrix_20260702/satphysmv11_constrained_sweep_summary.csv` |
| Completion time | `2026-07-02T18:31:15+08:00` |
| Summary rows | 160 rows, 10 cells x 16 policies |
| Final process cleanup | Remote experiment process ended; local SSH client cleaned up |

Artifacts:

| Artifact | Local path |
|---|---|
| Constrained head summary | `E:\type10-7\automation_reports\CV-SincNet\phase1_adv3b02_satonly_open_set_reject_20260702\artifacts\satphysmv11_constrained_sweep_summary.csv` |
| Constrained head driver log | `E:\type10-7\automation_reports\CV-SincNet\phase1_adv3b02_satonly_open_set_reject_20260702\artifacts\satphysmv11_constrained_driver.out` |

Policy aggregate result:

| Policy | Cells | Mean unknown_FAR | Max unknown_FAR | Mean old drop pp | Max old drop pp | Dual pass | Mean known coverage |
|---|---:|---:|---:|---:|---:|---:|---:|
| `SATPHY11C_MLP_M50_MIN02` | 10 | 0.0591 | 0.1000 | 53.10 | 61.08 | 0/10 | 0.2218 |
| `SATPHY11C_MLP_M50_PROXY02` | 10 | 0.0591 | 0.1000 | 53.10 | 61.08 | 0/10 | 0.2218 |
| `SATPHY11C_MLP_M20_MIN02` | 10 | 0.0737 | 0.1425 | 52.60 | 60.33 | 0/10 | 0.2328 |
| `SATPHY11C_MLP_M20_PROXY02` | 10 | 0.0737 | 0.1425 | 52.60 | 60.33 | 0/10 | 0.2328 |
| `SATPHY11C_MLP_M100_PROXY05` | 10 | 0.0954 | 0.1525 | 48.68 | 56.25 | 0/10 | 0.2865 |
| `SATPHY11C_MLP_M50_PROXY05` | 10 | 0.1062 | 0.1675 | 47.12 | 53.00 | 0/10 | 0.3058 |
| `SATPHY11C_MLP_M20_PROXY05` | 10 | 0.1410 | 0.2425 | 44.58 | 49.33 | 0/10 | 0.3435 |
| `SATPHY11C_MLP_M20_SRC9999` | 10 | 0.5768 | 0.7768 | 12.57 | 23.25 | 0/10 | 0.7908 |
| `SATPHY11C_MLP_M50_SRC9999` | 10 | 0.6977 | 0.8584 | 6.20 | 10.83 | 0/10 | 0.8890 |
| `SATPHY11C_MLP_M100_SRC9999` | 10 | 0.8127 | 0.9270 | 2.58 | 5.33 | 0/10 | 0.9427 |

Best dual-objective rows:

| Run | Policy | unknown_FAR | Old drop pp | Closed old acc | Full old acc after reject | Known coverage | Dual pass |
|---|---|---:|---:|---:|---:|---:|---|
| `phase1_adv3b02_satphysmv11_rx20_1_u1_20260702` | `SATPHY11C_MLP_M20_SRC9999` | 0.2605 | 14.00 | 0.6408 | 0.5008 | 0.7517 | false |
| `phase1_adv3b02_satphysmv11_rx20_1_u1_20260702` | `SATPHY11C_MLP_M50_COR_SRC9999` | 0.3081 | 13.92 | 0.6408 | 0.5017 | 0.7333 | false |
| `phase1_adv3b02_satphysmv11_rx8_8_u10_20260702` | `SATPHY11C_MLP_M20_COR_PROXY05` | 0.0865 | 36.67 | 0.7275 | 0.3608 | 0.3933 | false |
| `phase1_adv3b02_satphysmv11_rx8_8_u10_20260702` | `SATPHY11C_MLP_M50_COR_PROXY05` | 0.0703 | 38.58 | 0.7275 | 0.3417 | 0.3675 | false |

Best old-retention rows:

| Run | Policy | unknown_FAR | Old drop pp | Closed old acc | Full old acc after reject | Known coverage | Dual pass |
|---|---|---:|---:|---:|---:|---:|---|
| `phase1_adv3b02_satphysmv11_rx8_8_u10_20260702` | `SATPHY11C_MLP_M100_SRC9999` | 0.8216 | 1.00 | 0.7275 | 0.7175 | 0.9433 | false |
| `phase1_adv3b02_satphysmv11_rx8_8_u1_20260702` | `SATPHY11C_MLP_M100_SRC9999` | 0.8675 | 1.00 | 0.7275 | 0.7175 | 0.9433 | false |
| `phase1_adv3b02_satphysmv11_rx7_14_u10_20260702` | `SATPHY11C_MLP_M100_SRC9999` | 0.8944 | 1.33 | 0.8558 | 0.8425 | 0.9800 | false |
| `phase1_adv3b02_satphysmv11_rx7_14_u1_20260702` | `SATPHY11C_MLP_M100_SRC9999` | 0.9150 | 1.33 | 0.8558 | 0.8425 | 0.9800 | false |

Best low-FAR rows:

| Run | Policy | unknown_FAR | Old drop pp | Closed old acc | Full old acc after reject | Known coverage | Dual pass |
|---|---|---:|---:|---:|---:|---:|---|
| `phase1_adv3b02_satphysmv11_rx20_1_u1_20260702` | `SATPHY11C_MLP_M100_PROXY05` | 0.0140 | 47.08 | 0.6408 | 0.1700 | 0.2317 | false |
| `phase1_adv3b02_satphysmv11_rx20_1_u1_20260702` | `SATPHY11C_MLP_M20_PROXY02` | 0.0140 | 50.42 | 0.6408 | 0.1367 | 0.1858 | false |
| `phase1_adv3b02_satphysmv11_rx20_1_u1_20260702` | `SATPHY11C_MLP_M50_PROXY02` | 0.0140 | 51.25 | 0.6408 | 0.1283 | 0.1667 | false |
| `phase1_adv3b02_satphysmv11_rx8_8_u10_20260702` | `SATPHY11C_MLP_M50_PROXY02` | 0.0216 | 46.58 | 0.7275 | 0.2617 | 0.2775 | false |

Interpretation: increasing source-old penalty makes the tradeoff sharper but does not solve it. Proxy-threshold constrained heads can reduce mean FAR to about 5.9% and individual rows to about 1.4%-2.2%, but only by rejecting most old-class target samples. Source-accept constrained heads protect old classes better, including rows with old drop near 1.0pp, but FAR remains about 82%-92%. The frozen Phase1 feature space still lacks a source-only multi-view boundary that separates target unknown from target old under LEO impairment.
