# Mitigating Receiver Impact DA root-cause repair and validation

## Experiment identity

| Field | Value |
|---|---|
| Experiment ID | `mitigating_da_rootcause_20260710_104628` |
| Date | 2026-07-10 Asia/Hong_Kong |
| Operator | Codex with four independent/supervisory subagents |
| Objective | Explain the large Table II gap, repair confirmed paper/public-trainer mismatches, and validate the repaired Proposed method on N607 |
| Claim boundary | Closed-set WiSig UDA paper reproduction only; not CVS Stage2, new-class, open-set, satellite, or deployment evidence |

## 2026-07-13 paper-first re-audit

This report must be read as a bounded paper-equation reproduction, not an exact or strict reproduction. A fresh four-role audit separated PDF facts from implementation assumptions and the released trainer. The default full-component path matches the paper's DV direction, T-ascent/E-C-descent order, CPL equation, previous-batch class weighting, and mu/lambda objective. Exact parity remains blocked by the unpublished target train/test split, model layer details, optimizer, batch size, epochs, stopping rule, seeds/repeat count, and preprocessing provenance.

A confirmed Table III bug was found and repaired: disabling class weighting also changed the source CE scale from mu to 1.0. That changes more than omega_l(k), contrary to Eq. (10). Therefore every result generated before this repair with class weighting disabled is invalid as a paper Table III ablation and requires rerun. Full-component Proposed rows and domain-alignment-only rows are not numerically changed by this fix.

The current loader uses the same target dataset object for adaptation and final evaluation, and all runs produced before the 2026-07-13 audit logged target labels during training without using them in loss, gradients, or formal final checkpoint selection. Those existing rows are retrospectively `target-exposed diagnostic`, not independent bounded reproduction evidence. Future runs default to no training-time target-label audit. This is still a transductive implementation assumption, not a paper-confirmed split. The detailed mapping is in `analysis/mitigating_receiver_impact_da_core_mechanism_reaudit_20260713.md`.

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

All eight runs completed by 11:46 CST. These are target-exposed development diagnostics from the final checkpoint: target labels did not select the checkpoint, but they were logged during training and later used for cross-run diagnosis, so the rows are not independent label-blind reproduction evidence.

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

The dominant permutation patterns are `1-1->8-8`: class 1->3 (93.55%) and class 3->1 (99.68%); `7-7->8-8`: class 0->3 (98.08%), class 1->0 (91.75%), and class 3->1 (88.03%); `14-7->3-19`: classes 0/1/5 collapse mainly into class 3. All target classes contain 4000 samples, so class weighting observes a plausible marginal distribution even when class semantics are wrong. This explains why the `CPL+CW` mechanism can reinforce a high-confidence but label-permuted solution.

## Architecture and Table III localization matrix

The next matrix keeps every published scalar explicitly fixed (`lr=0.0006`, `tau=0.7`, `m=7`, `lambda=0.005`, `mu=0.5`), writes all five values into the result JSON, uses final-checkpoint evaluation, and runs only the paper Proposed method or its paper Table III component ablations.

| Candidate | Tasks | GPUs | Purpose | Claim status |
|---|---|---|---|---|
| `template_hypothesis_v1` | all five Table II tasks | 0-4 | Test the author-linked SAME-padding/preactivation ResNet1D + 3-layer C/T hypothesis | `diagnostic_only` because exact parameters are inferred |
| `standard_da_only` | `14-7->3-19` | 5 | Compare domain alignment alone with paper Table III 76.36% | `diagnostic_only` ablation |
| `standard_da_cw` | `14-7->3-19` | 6 | Test whether pseudo-labeling causes the collapse; paper Table III 77.02% | `diagnostic_only` ablation |
| `standard_cpl_cw` | `14-7->3-19` | 7 | Test the pseudo/class-weight path without KL; paper Table III 77.11% | `diagnostic_only` ablation |

Planned launcher: `paper_reproduction/mitigating_receiver_impact_da/launch_architecture_ablation_validation_20260710.sh`. Planned remote group: `mitigating_da_arch_ablation_20260710_115000`.

Exact server command: `cd /home/szu2070436088/2510044040/CV-SincNet && bash paper_reproduction/mitigating_receiver_impact_da/launch_architecture_ablation_validation_20260710.sh`. Python environment: `/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python`. Expected logs: `paper_reproduction/logs/mitigating_da_arch_ablation_20260710_115000/*.out`; expected results: `paper_reproduction/runs/mitigating_da_arch_ablation_20260710_115000/<run-id>/results.json`.

### Round-2 local verification and sync manifest

Git commit: `63af9ab` (`Add receiver DA architecture and ablation validation`). Local verification in `ssr-gpu`: 62 focused/adjacent tests passed; Python compile, `git diff --check`, launcher `bash -n`, CLI option inspection, model shape smoke, and parameter-count smoke passed. Independent algorithm review first issued NO-GO for four ablation loss scalings and target-BN leakage in the all-disabled control. Both defects were repaired, seven combinations and BN-state isolation were added to the tests, and the same reviewer then issued GO with no Critical/Important finding. Files below are the only runtime files planned for round-2 sync.

| Local/remote relative path | SHA256 |
|---|---|
| `paper_reproduction/mitigating_receiver_impact_da/model.py` | `47014c73f6b0385ba46296a1a8affd70aa5564d0f4970d2ffbb8bdf1467af3d5` |
| `paper_reproduction/mitigating_receiver_impact_da/losses.py` | `6fe3db3fa111631f585bbab60621a341151969779bbe1a7f3bb01db1eed4f395` |
| `paper_reproduction/mitigating_receiver_impact_da/algorithm.py` | `010871faf555c4ba2ec0f298440312e928947bfb958328220aad64220d235393` |
| `paper_reproduction/mitigating_receiver_impact_da/train.py` | `a843da29b07e0152209ee27585044f2d0a69504e4ab5450f44ff216b48b9a390` |
| `paper_reproduction/mitigating_receiver_impact_da/launch_architecture_ablation_validation_20260710.sh` | `d1f7c0ca7cb701cab3450576b5d695cb380f4bbe9906072f7a12a2d999d70573` |

### Round-2 launch record

Remote verification passed and the manifest was created at 2026-07-10 12:05:58 CST.

| Candidate/task | GPU | PID | Expected result |
|---|---:|---:|---|
| `template_hypothesis_v1` / `d01->d23` | 0 | `1866485` | `paper_reproduction/runs/mitigating_da_arch_ablation_20260710_115000/...template_hypothesis_v1_d01_to_d23.../results.json` |
| `template_hypothesis_v1` / `14-7->3-19` | 1 | `1866487` | `paper_reproduction/runs/mitigating_da_arch_ablation_20260710_115000/...template_hypothesis_v1_14-7_to_3-19.../results.json` |
| `template_hypothesis_v1` / `1-1->1-19` | 2 | `1866489` | `paper_reproduction/runs/mitigating_da_arch_ablation_20260710_115000/...template_hypothesis_v1_1-1_to_1-19.../results.json` |
| `template_hypothesis_v1` / `1-1->8-8` | 3 | `1866494` | `paper_reproduction/runs/mitigating_da_arch_ablation_20260710_115000/...template_hypothesis_v1_1-1_to_8-8.../results.json` |
| `template_hypothesis_v1` / `7-7->8-8` | 4 | `1866496` | `paper_reproduction/runs/mitigating_da_arch_ablation_20260710_115000/...template_hypothesis_v1_7-7_to_8-8.../results.json` |
| `standard_da_only` / `14-7->3-19` | 5 | `1866498` | `paper_reproduction/runs/mitigating_da_arch_ablation_20260710_115000/...standard_da_only_14-7_to_3-19.../results.json` |
| `standard_da_cw` / `14-7->3-19` | 6 | `1866500` | `paper_reproduction/runs/mitigating_da_arch_ablation_20260710_115000/...standard_da_cw_14-7_to_3-19.../results.json` |
| `standard_cpl_cw` / `14-7->3-19` | 7 | `1866502` | `paper_reproduction/runs/mitigating_da_arch_ablation_20260710_115000/...standard_cpl_cw_14-7_to_3-19.../results.json` |

Startup check at 12:06:31 CST: all eight PIDs alive, GPU utilization 21%-37%, memory 547-885 MiB, and no traceback/runtime/OOM/killed/argument/name-error marker.

Five-minute health check at 12:11:32 CST: all eight PIDs remained alive, GPU utilization 18%-35%, memory 683-1345 MiB, zero result files yet, and no error marker.

## Round-2 completed architecture and Table III results

`remote_artifacts_round2`包含8组结果JSON与完整`.out`。逐组反序列化核对后，JSON与对应`.out`语义完全一致；每个run均完成20个source-pretrain epoch和20个adaptation epoch，未发现Traceback、RuntimeError、CUDA OOM、NaN、Killed或参数错误marker。全部8个run均为target-exposed development diagnostics，且部分消融受本轮确认的旧缩放错误影响，不得作为正式论文复现成功、Table III机制证据、CVS Stage2或部署证据。

| Candidate | Task | Components/profile | Result | Paper comparator | Gap | Final claim |
|---|---|---|---:|---:|---:|---|
| `template_hypothesis_v1` | `d01->d23` | inferred template Proposed | 89.9611% | 93.34% | -3.3789pp | `diagnostic_only` |
| `template_hypothesis_v1` | `14-7->3-19` | inferred template Proposed | 52.1833% | 92.42% | -40.2367pp | `diagnostic_only` |
| `template_hypothesis_v1` | `1-1->1-19` | inferred template Proposed | 37.2250% | 95.44% | -58.2150pp | `diagnostic_only` |
| `template_hypothesis_v1` | `1-1->8-8` | inferred template Proposed | 89.6583% | 99.78% | -10.1217pp | `diagnostic_only` |
| `template_hypothesis_v1` | `7-7->8-8` | inferred template Proposed | 80.6917% | 99.74% | -19.0483pp | `diagnostic_only` |
| `standard_da_only` | `14-7->3-19` | custom toggle: DA on, pseudo/CW off | 68.5667% | Table III DA-only 76.36% | -7.7933pp | `target-exposed diagnostic` |
| `standard_da_cw` | `14-7->3-19` | custom toggle: DA/CW on, target pseudo CE off | 40.9875% | Table III DA+CW 77.02% | -36.0325pp | `pre-fix invalid/custom diagnostic` |
| `standard_cpl_cw` | `14-7->3-19` | custom toggle: target pseudo CE/CPL/CW on, DA off | 22.2750% | Table III CPL+CW 77.11% | -54.8350pp | `pre-fix invalid/custom diagnostic` |

| Five-task profile | Mean | Paper mean | Gap to paper | Interpretation |
|---|---:|---:|---:|---|
| `pytorch_template_resnet18_hypothesis_v1` | 69.9439% | 96.1440% | -26.2001pp | 比standard高7.5697pp，但架构仍为推断且差距显著 |
| `standard_resnet18` | 62.3742% | 96.1440% | -33.7698pp | 第一轮五个同run Proposed结果 |

这些旧组件结果只能描述修复前自定义开关的行为。论文没有公开“关闭CPL”时是否保留固定阈值伪标签CE；同时修复前no-class-weight路径改变了source CE缩放。因此它们不能映射为有效Table III机制结论，也不能用与论文数值的接近程度证明实现正确。

## Full multiseed Proposed-only validation plan

新launcher：`paper_reproduction/mitigating_receiver_impact_da/launch_full_multiseed_validation_20260713.sh`。矩阵为5个Table II任务×2个seed（20260711、20260712）×2个profile（`standard_resnet18`、`pytorch_template_resnet18_hypothesis_v1`）=20个Proposed-only运行。

| Wave | Runs | Seed/profile coverage | GPU allocation | Launch boundary |
|---|---:|---|---|---|
| 1 | 8 | seed20260711的4个receiver任务×2profile | GPU0-7各1个 | 必须显式`--wave 1`；本任务不启动 |
| 2 | 8 | seed20260711的cross-day×2profile；seed20260712的4个receiver任务，其中standard4项、template2项 | GPU0-7各1个 | 其他wave不得活跃；本任务不启动 |
| 3 | 4 | seed20260712剩余cross-day×2profile及template剩余2项 | GPU0-3各1个 | 其他wave不得活跃；本任务不启动 |

launcher生成并验证固定20行`expected_matrix.tsv`，要求组合和run ID唯一、wave计数严格为8/8/4；对既有run/log/result/manifest全部fail closed；使用`flock`防止并发启动器；检查其他wave manifest PID和相关训练命令；对本波涉及GPU要求现有compute process数小于2。因此在每GPU已有1个无关`phase1_dgleo`训练时，每卡再增加1个本实验进程后达到允许上限2，不会主动干预无关进程。

GPU共享会增加训练时延和运行时方差。后续结果必须记录wave、GPU、并发占用、起止时间和同run指标；不能把共享GPU导致的耗时变化解释为模型质量变化。若任一GPU已有2个compute process、相关wave仍活跃或输出路径已存在，launcher必须NO-GO。

### Supervisor review history

初版launcher因缺少run/manifest防覆盖、跨wave并发/GPU上限守卫及20项完整性manifest被监督审查判定NO-GO，未提交、未同步、未启动。修正版补齐上述硬门控后通过独立复审GO。GO只代表launcher可进入本地验证、Git提交与远端文件同步，不代表授权启动任何wave。

### 2026-07-13 local verification and sync map

| Check | Command | Result |
|---|---|---|
| 62 related tests | `python -m pytest -q -p no:cacheprovider --basetemp <workspace-temp> tests/test_wisig_random_split.py tests/test_wisig_fewshot_payload.py tests/test_mitigating_receiver_impact_da.py` in `ssr-gpu` | PASS，62/62 |
| Python syntax | `python -m py_compile paper_reproduction/mitigating_receiver_impact_da/*.py` in `ssr-gpu` | PASS |
| Launcher syntax | `bash -n paper_reproduction/mitigating_receiver_impact_da/launch_full_multiseed_validation_20260713.sh` | PASS |
| Diff whitespace | `git diff --check` | PASS；仅LF/CRLF转换提示 |

新launcher本地SHA256：`df9de171051e97d23787aaf91bbc62a11193cdb9eb9922d37ea9085c990bc1de`。

唯一远端同步映射：

| Local | Remote |
|---|---|
| `E:/type10-7/github_publish/CVS-RFFI-repo/paper_reproduction/mitigating_receiver_impact_da/launch_full_multiseed_validation_20260713.sh` | `/home/szu2070436088/2510044040/CV-SincNet/paper_reproduction/mitigating_receiver_impact_da/launch_full_multiseed_validation_20260713.sh` |

本次禁止同步报告、代码、配置或其他launcher，禁止启动wave1/2/3。

### N607 sync verification

- Pre-sync Git commit：`a3b7a331b60d6da08667aa8840c771cafb9532b4`；同步后仅amend本任务报告证据，不改变launcher内容。
- 2026-07-13 11:20 CST直接N607 preflight通过：远端用户`szu2070436088`、主机`dell-DSS8440`、项目根可见、8张RTX3090可见。
- `tools/n607_training_inventory.py --direct-only --pretty`确认唯一活跃训练族为无关`phase1_dgleo_jointp0_leoweak8r2_20260713`，其8个GPU compute子PID分别占用GPU0-7，每GPU恰好1个。
- 精确`pgrep`匹配`paper_reproduction.mitigating_receiver_impact_da.train`返回空；本实验训练进程为0。
- 仅同步`launch_full_multiseed_validation_20260713.sh`到约定远端同路径。远端SHA256为`df9de171051e97d23787aaf91bbc62a11193cdb9eb9922d37ea9085c990bc1de`，与本地一致；远端`bash -n`为PASS。
- `paper_reproduction/logs`和`paper_reproduction/runs`下不存在`mitigating_da_full_multiseed_validation_20260713*`wave输出。本任务未启动wave1/2/3。
- 每次SSH/SCP后均确认本地无`ssh.exe`且无N607或bridge TCP22 `ESTABLISHED`连接。

## Full multiseed execution record

### Wave1 pre-launch gate

- Gate time：2026-07-13 11:25:58-11:26:35 CST。
- Direct preflight：PASS；远端用户`szu2070436088`、主机`dell-DSS8440`、项目根与8张RTX3090均可见。
- GPU capacity：GPU0-7各恰有1个无关`phase1_dgleo` compute进程；本实验相关训练进程为0。wave1启动后每卡至多2个compute进程。
- Disk：`/home`剩余7.6TB。
- Prior-wave/output gate：wave1/2/3的目标run/log目录均不存在，满足首次启动条件。
- Exact launch command：`cd /home/szu2070436088/2510044040/CV-SincNet && bash paper_reproduction/mitigating_receiver_impact_da/launch_full_multiseed_validation_20260713.sh --wave 1`。
- Shared-GPU risk：本矩阵与无关phase1训练共享GPU，可能增加时延和运行时方差；不得把耗时差异解释为算法收益。
- SSH cleanup after gate：本地`ssh.exe=0`；N607/bridge TCP22 `ESTABLISHED=0`。

### Wave1 launch, health, and completion

- Launch window：约2026-07-13 11:27 CST；launcher内置3秒检查记录8项`startup_alive`，仅作为启动健康证据。
- Five-minute gate：11:31:35 CST，8个PID全部存活，每张GPU恰有2个compute进程，结果文件0，错误marker 0。日志尚未写出，但进程均为运行态并持续占用CPU/GPU，因此继续离散监控。
- Completion window：约11:36-11:38 CST。8个PID全部退出，8个结果文件全部落盘。
- Artifact destination：`E:/type10-7/automation_reports/CV-SincNet/mitigating_da_rootcause_20260710_104628/remote_artifacts_full_multiseed/wave1`。
- Artifact verification：manifest 8行、8个JSON、8个`.out`；每个run均包含20个source-pretrain epoch和20个adaptation epoch；JSON与`.out`语义一致；全部数值有限；未发现Traceback、RuntimeError、OOM、Killed、NaN、NameError或参数错误marker。所有17个文件均已计算SHA256。

| Profile | Task | Seed | GPU | PID | Final target accuracy | History max | Epoch1 pseudo acc / coverage | Final per-class accuracy | Verdict |
|---|---|---:|---:|---:|---:|---:|---:|---|---|
| standard | `14-7->3-19` | 20260711 | 0 | 3723051 | 42.2167% | 50.2214%@e1 | 50.5093% / 98.4375% | 57.65/22.53/1.23/82.15/89.10/0.65% | target-exposed diagnostic |
| standard | `1-1->1-19` | 20260711 | 1 | 3723121 | 40.3125% | 80.3894%@e7 | 68.5281% / 99.2020% | 90.48/8.80/0/42.62/99.98/0% | target-exposed diagnostic |
| standard | `1-1->8-8` | 20260711 | 2 | 3723603 | 96.7083% | 97.1800%@e16 | 75.2938% / 99.5488% | 99.58/99.20/99.80/81.85/99.98/99.85% | target-exposed diagnostic |
| standard | `7-7->8-8` | 20260711 | 3 | 3723673 | 63.4625% | 63.8745%@e20 | 58.1906% / 99.5154% | 94.17/99.88/0.03/86.72/99.98/0% | target-exposed diagnostic |
| template | `14-7->3-19` | 20260711 | 4 | 3723741 | 76.9708% | 79.3324%@e20 | 25.6996% / 96.4489% | 63.28/81.62/64.23/55.23/98.17/99.30% | target-exposed diagnostic |
| template | `1-1->1-19` | 20260711 | 5 | 3723811 | 51.6708% | 57.7916%@e7 | 54.0456% / 99.3942% | 97.38/68.30/0.53/43.70/99.95/0.18% | target-exposed diagnostic |
| template | `1-1->8-8` | 20260711 | 6 | 3723882 | 99.2833% | 99.5864%@e10 | 86.3749% / 99.5613% | 99.55/99.15/99.83/97.35/99.98/99.85% | target-exposed diagnostic |
| template | `7-7->8-8` | 20260711 | 7 | 3723954 | 79.0958% | 99.8705%@e14 | 68.3861% / 98.6506% | 87.72/12.20/74.85/99.98/99.93/99.90% | target-exposed diagnostic |

Shared-GPU observation：wave1运行期间每卡同时存在1个无关phase1进程和1个本矩阵进程；本表只解释模型指标，不以耗时作算法比较。SSH/SCP清理状态：每个远端命令和文件传输后本地`ssh.exe=0`且N607/bridge TCP22 `ESTABLISHED=0`。

### Wave2 pre-launch gate

- Gate time：2026-07-13 11:42:46 CST。
- Direct preflight：PASS。
- Previous-wave completeness：wave1 manifest 8行、results 8个、logs 8个，8个manifest PID均已退出。
- GPU capacity：GPU0-7各恰有1个无关compute进程；本实验相关训练进程为0。
- Disk：`/home`可用约7.54TiB。
- Output gate：wave2 run/log目录不存在。
- Exact launch command：`cd /home/szu2070436088/2510044040/CV-SincNet && bash paper_reproduction/mitigating_receiver_impact_da/launch_full_multiseed_validation_20260713.sh --wave 2`。
- SSH cleanup after gate：本地`ssh.exe=0`；N607/bridge TCP22 `ESTABLISHED=0`。
