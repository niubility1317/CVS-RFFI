# Mitigating Receiver Impact DA root-cause repair and validation

## Experiment identity

| Field | Value |
|---|---|
| Experiment ID | `mitigating_da_rootcause_20260710_104628` |
| Date | 2026-07-10 Asia/Hong_Kong |
| Operator | Codex with four independent/supervisory subagents |
| Objective | Explain the large Table II gap, repair confirmed paper/public-trainer mismatches, and validate the repaired Proposed method on N607 |
| Claim boundary | Closed-set WiSig UDA paper reproduction only; not CVS Stage2, new-class, open-set, satellite, or deployment evidence |

## Root-cause assessment before repair

The 48 locally available result JSON files contain 89 run rows, 840 epoch records, 330 target-evaluation records, and 200 source-pretraining records. The bad receiver pairs diverge in the first adaptation epoch: they select 93%-97% of target samples while pseudo-label precision is only 33%-49%. Later epochs enter different high-confidence class-permutation attractors; training longer does not repair the initial contamination.

| Root cause | Evidence | Repair/status |
|---|---|---|
| Random or inconsistent source model `h0` | Prior best diagnostic launchers explicitly set source pretraining to zero; paper starts from an initial model such as one trained on Rx-1 | Validation matrix includes explicit source pretraining; duration remains a reproduction setting because the paper omits it |
| Weighted CE reduction mismatch | Old code used `mean(weight*CE)`; public trainer uses PyTorch weighted mean. Old class weights reached 533-1067, changing the effective CE/KL scale | Added separate `paper_sample_mean` and `pytorch_weighted_mean` modes; public-trainer compatibility selects the latter |
| Extra E/C feature forwards | Old path updated ResNet BatchNorm statistics multiple times per domain/batch | E/C now reuses one source and one target forward, matching the public trainer |
| MINE moving-average mismatch | Old path carried `ma_et` through all seven T updates; public trainer resets each substep and passes only the last result to E/C | Public MINE path now matches the exposed trainer |
| Pseudo-state and batch-pairing mismatch | Paper Algorithm 1 resets counters per outer loop and uses `min(floor(Ns/b),floor(Nt/b))`; old strict defaults used global state/cycled target and kept partial batches | Strict defaults changed to epoch state, `zip_min`, and train `drop_last=True` |
| Wrong cross-day task | Old `d01->d23` meant days `[0,1] -> [2,3]` | Corrected to 2021-03-01 -> 2021-03-23; prior cross-day result is invalid for Table II |
| Incomplete normalization | Paper subtracts mean before power normalization; old loader only divided by RMS | Paper dataset path now centers then normalizes; N607 audit shows this changes only about 0.05% signal power, so it is not the dominant root cause |
| RNG order coupling | Full-matrix rows inherited RNG state from earlier tasks/methods | Seed is reset before each task/method model initialization |

## Paper/public-code uncertainty

The paper does not report batch size, epoch count, optimizer type, scheduler, seeds/repeat count, source-pretraining duration, target train/test split, stopping/model-selection rule, ResNet1D stem, feature dimension, FC widths, activations, or initialization. The public repository contains only a trainer and omits its experiment TOML, WiSig wrapper/split, model wrapper, MINE class, and exact ResNet constructor. Results after repair can therefore establish a stronger bounded reproduction, but not an exact author-environment reproduction.

## Local changes and verification

| Local file | Purpose |
|---|---|
| `code/dataset_wisig.py` | Optional mean centering before RMS normalization |
| `paper_reproduction/common/wisig_runtime.py` | Optional train `drop_last` |
| `paper_reproduction/mitigating_receiver_impact_da/data.py` | Correct date mapping, paper centering, floor-batch loaders |
| `paper_reproduction/mitigating_receiver_impact_da/losses.py` | Separate paper and public-trainer weighted CE reductions |
| `paper_reproduction/mitigating_receiver_impact_da/algorithm.py` | Single-forward E/C path and corrected public MINE MA handling |
| `paper_reproduction/mitigating_receiver_impact_da/train.py` | Paper defaults, per-row reseeding, CE-mode plumbing, and fail-closed claim profiles |
| `paper_reproduction/mitigating_receiver_impact_da/protocol.py` and config | Persist explicit reported/unspecified boundaries |
| `tests/test_mitigating_receiver_impact_da.py` | Regression coverage for every confirmed mismatch |

Verification completed in activated `ssr-gpu`:

| Check | Result |
|---|---|
| Python compile for all changed Python modules | PASS |
| `pytest tests/test_mitigating_receiver_impact_da.py -q` | 36 passed |
| `pytest tests/test_wisig_random_split.py tests/test_wisig_fewshot_payload.py tests/test_mitigating_receiver_impact_da.py -q` | 52 passed |
| Paper config dry-run | PASS; exposes paper constants and unspecified fields |

## N607 validation matrix

All rows run only the Proposed method. Common fixed paper values are `lr=0.0006`, `tau=0.7`, `m=7`, `lambda=0.005`, and `mu=0.5`. Formal result selection is `final`; target-label `target_loss_best` is not used.

| Candidate | Tasks | Reproduction interpretation | Key settings | Success criterion |
|---|---|---|---|---|
| `strict_paper_h0` | All five Table II tasks | Literal paper equations plus explicit source model | 20 source-pretrain epochs, 20 adaptation epochs, batch128, uniform prior, DV, paper CPL, epoch/zip/drop-last, paper CE | Improve hard-pair final mean and avoid epoch-1 pseudo-label collapse without diagnostic floor/quota |
| `released_trainer_h0` | `14-7->3-19`, `1-1->1-19` | Exposed public trainer semantics | 10 same-optimizer source epochs + 20 adaptation epochs, batch128, MINE MA, official threshold path, current class weights, PyTorch weighted CE | Determine whether the previously missing public-code details explain the hard-pair gap |
| `strict_paper_no_h0` | `14-7->3-19` | Root-cause control | Strict paper path but no source pretraining | Quantify the contribution of `h0` under the repaired loss/BN/state path |

Remote root: `/home/szu2070436088/2510044040/CV-SincNet`. Python: `/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python`. Planned logs: `paper_reproduction/logs/mitigating_da_rootcause_20260710_104628/*.out`. Planned results/checkpoints: `paper_reproduction/runs/mitigating_da_rootcause_20260710_104628/<run-id>/`.

## Preflight and risks

Direct read-only N607 preflight passed at 10:53 CST. All eight RTX 3090 GPUs were idle and no mitigating/DADDA process was active. Dataset audit confirmed six classes, 12 receivers, four dates, 4000 samples per TX/receiver across dates, and fixed `(256,2)` equalized IQ. Main remaining risks are underdetermined architecture/config, seed sensitivity, the missing author target split, and the absence of a paper-defined label-free stopping rule.

## Sync manifest

All destinations are under `/home/szu2070436088/2510044040/CV-SincNet/`.

| Local/remote relative path | SHA256 |
|---|---|
| `code/dataset_wisig.py` | `8085807eb3a4c682fba1f66447121af6c89dd8a4e5081fa47a0d25456c22f5ba` |
| `paper_reproduction/common/wisig_runtime.py` | `dbbad9c4f8b39750a7f33b1b10dcb144814a8f711cd369ac35aa8c0ddbf02061` |
| `paper_reproduction/configs/mitigating_receiver_impact_da_manysig_paper_faithful.json` | `b17b46cc0db705dba4fbd5d253ee0c3367531ddca3e93e99e6a42298dd40aef2` |
| `paper_reproduction/mitigating_receiver_impact_da/algorithm.py` | `35194b167d86d51a25876813a273ab5efe38851e118861d4f348ddb0a687fd13` |
| `paper_reproduction/mitigating_receiver_impact_da/data.py` | `4de38056d3ab58ad754e74ef1ce10c912994cacc5e03044bd4e29cbc47109306` |
| `paper_reproduction/mitigating_receiver_impact_da/losses.py` | `69c27b5020ce93c7d1288369400e167b44fe1e49a0b89f0adc3fde9d0de30547` |
| `paper_reproduction/mitigating_receiver_impact_da/protocol.py` | `69b5c41fb3a2473725e16ace4c29564aaf4bead2c85a00e68e60b284f7bbf117` |
| `paper_reproduction/mitigating_receiver_impact_da/train.py` | `85d5bfa82a80980e289e5c4f0f62bca572cc7007d34234338a11c42784ad8364` |

## Supervisory gate

The first supervisory review blocked launch because diagnostic quota/floor and target-label checkpoint runs were still emitted with ordinary `completed` status. This is fixed before sync: every result now carries `execution_status`, `reproduction_profile`, `claim_status`, and `claim_reasons`. Strict equations and exposed public-trainer semantics are separate bounded profiles. Truncated smoke runs, mixed settings, paper-external controls, and target-label model selection become `completed_diagnostic_only` and cannot be presented as formal paper reproduction.

## Launch record

Remote SHA256 values matched the sync manifest. Remote `py_compile`, launcher `bash -n`, and paper-config dry-run passed. At 2026-07-10 11:22:34 CST the following Proposed-only runs were launched; every command uses `target_model_selection=final` and none uses quota, threshold floor, class-weight clipping/smoothing, or target-label checkpoint selection.

| Run/candidate | Task | GPU | PID | Epoch plan | Output |
|---|---|---:|---:|---|---|
| `strict_paper_h0_d01_to_d23` | `d01->d23` | 0 | `1845007` | 20 source + 20 adaptation | `paper_reproduction/runs/mitigating_da_rootcause_20260710_104628/...strict_paper_h0_d01_to_d23.../results.json` |
| `strict_paper_h0_14-7_to_3-19` | `14-7->3-19` | 1 | `1845009` | 20 source + 20 adaptation | `paper_reproduction/runs/mitigating_da_rootcause_20260710_104628/...strict_paper_h0_14-7_to_3-19.../results.json` |
| `strict_paper_h0_1-1_to_1-19` | `1-1->1-19` | 2 | `1845011` | 20 source + 20 adaptation | `paper_reproduction/runs/mitigating_da_rootcause_20260710_104628/...strict_paper_h0_1-1_to_1-19.../results.json` |
| `strict_paper_h0_1-1_to_8-8` | `1-1->8-8` | 3 | `1845013` | 20 source + 20 adaptation | `paper_reproduction/runs/mitigating_da_rootcause_20260710_104628/...strict_paper_h0_1-1_to_8-8.../results.json` |
| `strict_paper_h0_7-7_to_8-8` | `7-7->8-8` | 4 | `1845015` | 20 source + 20 adaptation | `paper_reproduction/runs/mitigating_da_rootcause_20260710_104628/...strict_paper_h0_7-7_to_8-8.../results.json` |
| `released_trainer_h0_14-7_to_3-19` | `14-7->3-19` | 5 | `1845017` | 10 same-optimizer source + 20 adaptation | `paper_reproduction/runs/mitigating_da_rootcause_20260710_104628/...released_trainer_h0_14-7_to_3-19.../results.json` |
| `released_trainer_h0_1-1_to_1-19` | `1-1->1-19` | 6 | `1845019` | 10 same-optimizer source + 20 adaptation | `paper_reproduction/runs/mitigating_da_rootcause_20260710_104628/...released_trainer_h0_1-1_to_1-19.../results.json` |
| `strict_paper_no_h0_14-7_to_3-19` | `14-7->3-19` | 7 | `1845021` | 0 source + 20 adaptation | `paper_reproduction/runs/mitigating_da_rootcause_20260710_104628/...strict_paper_no_h0_14-7_to_3-19.../results.json` |

Startup health at elapsed 118 seconds: all eight PIDs were alive; GPU utilization was 3%-23% with 547-687 MiB allocated; no traceback, runtime error, OOM, killed, argument, or name-error marker was present. The low early utilization is consistent with dataset/index preparation and source-pretraining startup and will be rechecked after the required 4-5 minute window.
| `paper_reproduction/mitigating_receiver_impact_da/launch_rootcause_validation_20260710.sh` | `6cab62a62f248f3bc36c934e0a3ff708ca7c6325a5a754e4b23148a4c1cb23ef` |

## First repaired matrix results

All eight runs completed by 11:46 CST. Formal numbers below are full target-evaluation rows from the final checkpoint; history maxima use target labels only as post-hoc diagnostics and were not selected or saved as formal results.

| Profile | Task | Paper | Reproduction | Gap | Initial target `h0` | Epoch-1 pseudo precision | Epoch-1 coverage | Post-hoc history max |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| Strict equations + `h0` | `d01->d23` | 93.34% | 85.73% | -7.61pp | 84.11% | 88.28% | 99.07% | 90.00% @ epoch13 |
| Strict equations + `h0` | `14-7->3-19` | 92.42% | 30.20% | -62.22pp | 22.19% | 48.32% | 97.78% | 51.12% @ epoch11 |
| Strict equations + `h0` | `1-1->1-19` | 95.44% | 75.40% | -20.04pp | 63.41% | 46.16% | 99.58% | 76.17% @ epoch20 |
| Strict equations + `h0` | `1-1->8-8` | 99.78% | 67.70% | -32.08pp | 68.62% | 76.94% | 99.60% | 76.78% @ epoch1 |
| Strict equations + `h0` | `7-7->8-8` | 99.74% | 52.85% | -46.89pp | 38.75% | 59.14% | 99.52% | 68.20% @ epoch3 |
| Released-trainer semantics | `14-7->3-19` | 92.42% | 28.29% | -64.13pp | n/a | n/a during source-only phase | n/a | 34.80% @ epoch23 |
| Released-trainer semantics | `1-1->1-19` | 95.44% | 46.07% | -49.37pp | n/a | n/a during source-only phase | n/a | 44.23% @ epoch11 |
| Strict equations, no `h0` | `14-7->3-19` | 92.42% | 33.25% | -59.17pp | random | 27.62% | 92.85% | 42.47% @ epoch6 |

Strict five-task mean: paper 96.14%, reproduction 62.37%, gap -33.77pp.

### Per-class final accuracy for strict Proposed

| Task | Class 0 | Class 1 | Class 2 | Class 3 | Class 4 | Class 5 |
|---|---:|---:|---:|---:|---:|---:|
| `d01->d23` | 80.88% | 74.78% | 91.12% | 69.32% | 98.73% | 99.55% |
| `14-7->3-19` | 0.28% | 12.38% | 39.38% | 62.88% | 60.73% | 5.55% |
| `1-1->1-19` | 99.55% | 0.13% | 91.83% | 61.10% | 99.98% | 99.83% |
| `1-1->8-8` | 99.93% | 6.38% | 99.78% | 0.33% | 99.95% | 99.85% |
| `7-7->8-8` | 0.05% | 8.18% | 97.30% | 11.63% | 99.95% | 99.98% |

### Root-cause verdict after matrix 1

1. Confirmed semantic bugs were real, but they were not the dominant residual cause. The public-trainer path is worse on both hard receiver pairs.
2. Source training converges; target transfer does not. The hard-pair `h0` target accuracy is only 22.19%-68.62%, despite 98.54%-99.92% final source-batch accuracy.
3. CPL admits almost the whole target set in epoch 1. Because its threshold is class-curriculum scaled, `tau=0.7` does not imply 70% confidence for underrepresented predicted classes. Incorrect labels are therefore reinforced immediately.
4. The final failures are class permutations, not a uniform loss of signal information. This points to an incompatible receiver-dependent feature geometry and/or a different author target split/model wrapper, not simply insufficient epochs.
5. Exact author parity remains blocked by missing model/config/data-split artifacts. Any architecture inference or target-label-selected checkpoint must remain diagnostic-only.

## Architecture and Table III localization matrix

The next matrix keeps every published scalar explicitly fixed (`lr=0.0006`, `tau=0.7`, `m=7`, `lambda=0.005`, `mu=0.5`), writes all five values into the result JSON, uses final-checkpoint evaluation, and runs only the paper Proposed method or its paper Table III component ablations.

| Candidate | Tasks | GPUs | Purpose | Claim status |
|---|---|---|---|---|
| `template_hypothesis_v1` | all five Table II tasks | 0-4 | Test the author-linked SAME-padding/preactivation ResNet1D + 3-layer C/T hypothesis | `diagnostic_only` because exact parameters are inferred |
| `standard_da_only` | `14-7->3-19` | 5 | Compare domain alignment alone with paper Table III 76.36% | `diagnostic_only` ablation |
| `standard_da_cw` | `14-7->3-19` | 6 | Test whether pseudo-labeling causes the collapse; paper Table III 77.02% | `diagnostic_only` ablation |
| `standard_cpl_cw` | `14-7->3-19` | 7 | Test the pseudo/class-weight path without KL; paper Table III 77.11% | `diagnostic_only` ablation |

Planned launcher: `paper_reproduction/mitigating_receiver_impact_da/launch_architecture_ablation_validation_20260710.sh`. Planned remote group: `mitigating_da_arch_ablation_20260710_115000`.

### Round-2 local verification and sync manifest

Git commit: `63af9ab` (`Add receiver DA architecture and ablation validation`). Local verification in `ssr-gpu`: 62 focused/adjacent tests passed; Python compile, `git diff --check`, launcher `bash -n`, CLI option inspection, model shape smoke, and parameter-count smoke passed. Independent algorithm review first issued NO-GO for four ablation loss scalings and target-BN leakage in the all-disabled control. Both defects were repaired, seven combinations and BN-state isolation were added to the tests, and the same reviewer then issued GO with no Critical/Important finding. Files below are the only runtime files planned for round-2 sync.

| Local/remote relative path | SHA256 |
|---|---|
| `paper_reproduction/mitigating_receiver_impact_da/model.py` | `47014c73f6b0385ba46296a1a8affd70aa5564d0f4970d2ffbb8bdf1467af3d5` |
| `paper_reproduction/mitigating_receiver_impact_da/losses.py` | `6fe3db3fa111631f585bbab60621a341151969779bbe1a7f3bb01db1eed4f395` |
| `paper_reproduction/mitigating_receiver_impact_da/algorithm.py` | `010871faf555c4ba2ec0f298440312e928947bfb958328220aad64220d235393` |
| `paper_reproduction/mitigating_receiver_impact_da/train.py` | `a843da29b07e0152209ee27585044f2d0a69504e4ab5450f44ff216b48b9a390` |
| `paper_reproduction/mitigating_receiver_impact_da/launch_architecture_ablation_validation_20260710.sh` | `d1f7c0ca7cb701cab3450576b5d695cb380f4bbe9906072f7a12a2d999d70573` |
