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
| `paper_reproduction/mitigating_receiver_impact_da/launch_rootcause_validation_20260710.sh` | `6cab62a62f248f3bc36c934e0a3ff708ca7c6325a5a754e4b23148a4c1cb23ef` |
