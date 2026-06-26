# Progress Log

## 2026-05-19 Baseline paper/code audit
- Started current task: audit all baseline papers and corresponding generated code under `E:/type10-7/baselines`.
- Read applicable skills and project instructions.
- Confirmed `AGENTS.md` requires `conda activate ssr-gpu` before project-related tests.
- Confirmed workspace is not a Git repository.
- Replaced active `task_plan.md` with a baseline-audit-specific plan.
- Appended initial request and inventory notes to `findings.md`.
- Read `baselines/README.md`, `baselines/common/cvs_trainer.py`, `baselines/common/cvs_data.py`, `baselines/common/cvs_sat_eval.py`, `baselines/tifs2025_channel_receiver_rffi/train_cvs.py`, and `baselines/tifs2025_channel_receiver_rffi/data.py`.
- Recorded initial TIFS2025 slow-path evidence and shared trainer final-test gap in `findings.md`.
- Error: attempted a Bash heredoc under PowerShell while checking PDF extraction packages; will rerun with a PowerShell here-string.
- Extracted four baseline PDFs to `tmp/pdfs/*.txt` using `pypdf`.
- Read the main CVS training entry points for RIEI, DRIFT, receiver-agnostic RFFI, CVCNN, and receiver-agnostic fine-tuning.
- Confirmed all main baseline trainers share the same validation-gated test path.
- Parsed historical baseline `metrics.json` files and log tails to confirm final epochs were not tested after training completion.
- Superseded on 2026-06-26: root-level `run_cvs_baseline_queue.sh` is now only a compatibility wrapper; the canonical implementation is `scripts/launchers/run_cvs_baseline_queue.sh`.
- Proposed focused fix design to user: central final test/satellite evaluation in `cvs_trainer.py`, batched TIFS2025 augmentation/spectrogram path, and fine-tune satellite consistency.
- User approved the design.
- Added RED regression tests in `tests/test_baseline_training_behaviors.py`.
- RED verification:
  - `conda activate ssr-gpu; python -m unittest tests.test_baseline_training_behaviors -q` failed on missing `metrics.json["final"]` and missing TIFS `return_raw`.
  - `conda activate ssr-gpu; python -m unittest tests.test_baseline_training_behaviors.Tifs2025RawDatasetTest.test_tifs_batch_transform_applies_augmentation_once_to_whole_batch -q` failed on missing `prepare_tifs2025_batch`.
- Implemented fixes:
  - `baselines/common/cvs_trainer.py`: added final post-training test/satellite evaluation while preserving best-val checkpoint semantics.
  - `baselines/tifs2025_channel_receiver_rffi/data.py`: added batched spectrogram support and raw-return dataset mode.
  - `baselines/tifs2025_channel_receiver_rffi/train_cvs.py`: moved TIFS2025 augmentation/spectrogram work into batched train/eval code.
  - `baselines/receiver_agnostic_rffi/finetune_cvs.py`: added satellite evaluation support.
  - `baselines/common/__init__.py`: made README-style baseline module entrypoints find workspace `code/` helpers.
- Verification:
  - `conda activate ssr-gpu; python -m unittest tests.test_baseline_training_behaviors -q` passed 3 tests.
  - `conda activate ssr-gpu; python -m py_compile ...` passed for all touched Python files.
  - Baseline `--help` commands for TIFS2025, CVCNN, RIEI, DRIFT, receiver-agnostic train, and receiver-agnostic fine-tune all exited 0.
- Unrelated/pre-existing issue:
  - `tests.test_cvs_rffi_launcher` still fails with empty stdout for a staged launcher check unrelated to the baseline trainer changes.

## Session: 2026-05-15

### Phase 1: Requirements & Discovery
- **Status:** complete
- Actions taken:
  - Read the SGV-BP-FJMP design document with explicit UTF-8 encoding.
  - Confirmed `E:/type10-7` is not a Git repository.
  - Replaced stale planning files from a previous FJMP v2 task with this SGV-BP-FJMP-specific plan.
  - Compared the document's required file list and checks against existing FJMP/trainer/manifest code.
- Files created/modified:
  - `task_plan.md`
  - `findings.md`
  - `progress.md`

### Phase 2: Test-First Coverage
- **Status:** complete
- Actions taken:
  - Added `tests/test_sgv_bp_fjmp_design.py` for the documented module list, trainer SGV-BP args, SGV-BP experiment matrix, and launcher existence.
  - Ran the test before implementation and observed failures for missing modules, parser args, manifest IDs, and launcher.
- Files created/modified:
  - `tests/test_sgv_bp_fjmp_design.py`

### Phase 3: Core Implementation
- **Status:** complete
- Actions taken:
  - Added SGV generation, centered temperature calibration, base-protected fusion, SGV-BP losses, SGV-BP metrics, and paired sampler modules.
  - Added prototype aggregation modes and documented output aliases to the FJMP head.
  - Added SGV-BP parser args, base-protected fusion construction, optimizer parameter groups, prototype LR decay, and paired clean/sat SGV training loss wiring to `train_fjmp.py`.
  - Added SGV-BP EXP-00 through EXP-16 manifest entries.
  - Added SGV-BP 8-GPU launcher and summary proxy-safe score reporting.
- Files created/modified:
  - `FJMP/star_ground_view.py`
  - `FJMP/logit_calibration.py`
  - `FJMP/base_protected_fusion.py`
  - `FJMP/sgv_bp_losses.py`
  - `FJMP/sgv_bp_metrics.py`
  - `FJMP/sgv_sampler.py`
  - `FJMP/frozen_joint_prototype_head.py`
  - `FJMP/experiment_manifest.py`
  - `FJMP/summarize_experiments.py`
  - `train_fjmp.py`
  - `scripts/run_fjmp_sgv_bp_8gpu.sh`
  - `tests/test_sgv_bp_fjmp_design.py`

### Phase 4: Verification
- **Status:** complete
- Actions taken:
  - Ran targeted no-torch unit tests.
  - Verified all 263 manifest entries parse against `train_fjmp.py`.
  - Ran syntax compilation with a temporary pycache prefix.
  - Ran SGV-BP launcher bash syntax check and `EXP-04` dry-run.

### SGV-BP Experiment Expansion
- **Status:** complete
- Actions taken:
  - Expanded SGV-BP manifest from 17 to 49 experiments.
  - Marked `EXP-00` through `EXP-10` as the `CORE` batch for first-wave execution.
  - Added follow-up SGV-BP batches: `RHO`, `SAFETY`, `LOSS`, `SGV`, `PROTO`, `ZDOM`, `FUSION`, `OPT`, `SCHED`, and `SELECT`.
  - Verified `CORE` dry-run emits 11 commands.
  - Updated SGV-BP launcher default to `FULL`; full launch now queues `CORE` experiments first, then every remaining SGV-BP experiment as GPUs 0-7 free up.
- Files created/modified:
  - `FJMP/experiment_manifest.py`

## Test Results
| Test | Input | Expected | Actual | Status |
|------|-------|----------|--------|--------|
| SGV-BP red test | `python -m unittest tests.test_sgv_bp_fjmp_design` before implementation | Missing interfaces fail | 3 failures, 1 parser error | expected fail |
| Targeted no-torch unit tests | `python -m unittest tests.test_post_stage_trainers tests.test_fjmp_package_layout tests.test_sgv_bp_fjmp_design` | Pass | Ran 12 tests, OK | pass |
| Manifest/parser compatibility | parser loop over `build_experiment_manifest()` | 0 parser failures | `263 0` | pass |
| Syntax check | `python -m py_compile ...` with `PYTHONPYCACHEPREFIX` | All touched Python files compile | Passed | pass |
| Launcher syntax | `bash -n scripts/run_fjmp_sgv_bp_8gpu.sh` | valid bash syntax | Passed | pass |
| Launcher dry-run | `bash scripts/run_fjmp_sgv_bp_8gpu.sh --base-ckpt tmp_launcher_test/dummy.pth --plan EXP-04 --gpu-ids 0,1 --dry-run` | emits main EXP-04 command | Generated SGV-BP-FJMP EXP-04 command | pass |
| SGV-BP expanded manifest count | manifest summary script | 49 SGV-BP experiments and 11 CORE experiments | `sgv 49`, `core 11` | pass |
| Expanded manifest/parser compatibility | parser loop over `build_experiment_manifest()` | 0 parser failures | `295 0` | pass |
| CORE launcher dry-run | `bash scripts/run_fjmp_sgv_bp_8gpu.sh --base-ckpt tmp_launcher_test/dummy.pth --plan CORE --gpu-ids 0,1 --dry-run` | emits 11 core commands | `TOTAL_JOBS=11` | pass |
| FULL launcher dry-run | `bash scripts/run_fjmp_sgv_bp_8gpu.sh --base-ckpt tmp_launcher_test/dummy.pth --gpu-ids 0,1,2,3,4,5,6,7 --dry-run` | CORE first, then rest | `TOTAL_JOBS=49`, `EXP-00` first, `EXP-10` before `EXP-11`, status 0 | pass |

## Error Log
| Timestamp | Error | Attempt | Resolution |
|-----------|-------|---------|------------|
| 2026-05-15 | `git status --short` failed: not a Git repository. | 1 | Proceed with direct file tracking. |
| 2026-05-15 | PowerShell rejected a Bash heredoc in a parser smoke command. | 1 | Re-ran using a PowerShell here-string piped to Python. |
| 2026-05-15 | SGV-BP launcher dry-run initially failed because bash could not find `python`. | 1 | Added automatic Python executable discovery and re-ran successfully. |
| 2026-05-15 | `py_compile` hit a Windows pycache write permission/lock. | 1 | Re-ran with temporary `PYTHONPYCACHEPREFIX`; removed the temporary cache. |
| 2026-05-15 | Default Python lacks `torch` and `pytest`. | 1 | Full tensor tests are environment-blocked; no-torch tests and syntax checks pass. |
| 2026-05-15 | Recommended checkpoint path is not present in the local workspace, though user notes it exists on the server. | 1 | Did not start local training; verified the `CORE` queue by dry-run and provided the exact server command. |

### 2026-05-16 FJMP loss logging and cross-domain diagnosis
- Analyzed `5.16logs/fjmp_sgv_bp`: 48 training logs plus `EXP-00` baseline eval.
- Found SGV-BP total loss is large because the SGV/safety branch dominates total loss; old logs did not print per-loss raw values or weighted contributions.
- Found `lambda_worst_domain_view` was effectively inactive because `train_fjmp.py` did not pass domain groups into `compute_sgv_bp_losses`.
- Updated `train_fjmp.py` to print `[LOSS-FJMP-RAW]`, `[LOSS-FJMP-W]`, `[LOSS-SGV-RAW]`, `[LOSS-SGV-W]`, and `[LOSS-TOP]` per epoch, and to save raw/weighted loss columns in `metrics_epoch.csv`.
- Updated `train_fjmp.py` to pass training domain groups into SGV-BP losses so worst-domain-view loss can optimize cross-domain groups.
- Added tests covering loss breakdown formatting and domain-group wiring.
- Verification: `python -m unittest tests.test_post_stage_trainers tests.test_fjmp_package_layout tests.test_sgv_bp_fjmp_design` -> 15 tests OK; `py_compile train_fjmp.py` with temporary pycache -> OK; SGV-BP manifest parser smoke -> `sgv_rows=49 parser_failures=0`.

### 2026-05-16 Loss-design experiment batch and 8-GPU launcher
- Added a focused `LOSS-DESIGN` batch with 16 experiments (`LD-00` through `LD-15`) in `FJMP/experiment_manifest.py`.
- The batch targets post-5.16 findings: fixed worst-domain-view rerun, conservative recommended loss, SGV safe/head balance, worst-domain-view pressure, preservation/harm strength, KD lock-in, rho cap, optimizer LR, SGV strength, and long refinement.
- Added `scripts/run_fjmp_loss_design_8gpu.sh`, a wrapper around the existing dynamic queue launcher using GPUs `0,1,2,3,4,5,6,7`, `runs/fjmp_loss_design`, and `logs/fjmp_loss_design`.
- Expanded `train_fjmp.py` startup logging with `[CONFIG-BEGIN]`, `[CONFIG-RUN]`, `[CONFIG-DATA]`, `[CONFIG-MODEL]`, `[CONFIG-OPT]`, `[CONFIG-LOSS]`, `[CONFIG-SGV-LOSS]`, `[CONFIG-EVAL]`, and `[CONFIG-END]`.
- Existing epoch logs now include `[LOSS-FJMP-RAW]`, `[LOSS-FJMP-W]`, `[LOSS-SGV-RAW]`, `[LOSS-SGV-W]`, and `[LOSS-TOP]` from the prior loss logging change.
- Verification:
  - `python -m unittest tests.test_post_stage_trainers tests.test_fjmp_package_layout tests.test_sgv_bp_fjmp_design` -> 18 tests OK.
  - `python -m py_compile train_fjmp.py FJMP/experiment_manifest.py` with temporary pycache -> OK.
  - `bash -n scripts/run_fjmp_loss_design_8gpu.sh` -> OK.
  - `bash scripts/run_fjmp_loss_design_8gpu.sh --base-ckpt tmp_launcher_test/dummy.pth --gpu-ids 0,1 --dry-run` -> `TOTAL_JOBS=16`, emitted `LD-00` through `LD-15`, status 0.

### 2026-05-16 A03/A06 anchored loss attribution batch
- Parsed `5.15logs/fjmp_v2/A03_20260515_093220.log` and `5.15logs/fjmp_v2/A06_20260515_093558.log`.
- Found both A03 and A06 reached the reported `unseen_day_unseen_rx=86.83%` at epoch 7, then drifted lower by epoch 30.
- Added `A03-A06-REPRO` batch with 8 experiments (`R83-00` through `R83-07`) in `FJMP/experiment_manifest.py`.
- The batch includes A03/A06 exact controls, epoch-7 high-point repro runs, margin-preserve variants, no-KD ablation, and proto-CE ablation to identify which loss terms improve cross-domain UDU.
- Added `scripts/run_fjmp_a03_a06_repro_8gpu.sh` using the existing dynamic GPU queue with default roots `runs/fjmp_a03_a06_repro` and `logs/fjmp_a03_a06_repro`.
- Updated `train_fjmp.py` to save and print a diagnostic `best_udu_fjmp.pth` checkpoint with `[BEST-UDU] diagnostic_test_selection`.
- Updated epoch logging to include active loss-weight snapshots: `[LOSS-FJMP-WEIGHT]` and `[LOSS-SGV-WEIGHT]`, in addition to raw/weighted losses and `[LOSS-TOP]`.
- Added `--save_checkpoints`; A03/A06 attribution experiments set it to `false` so loss analysis keeps logs/CSV only and does not write model weight files.
- Added `--metrics_csv` and updated the dynamic launcher so each experiment writes `R83-xx_TIMESTAMP_metrics_epoch.csv` next to `R83-xx_TIMESTAMP.log` in the same log folder.
- Verification:
  - `python -m unittest tests.test_post_stage_trainers tests.test_fjmp_package_layout tests.test_sgv_bp_fjmp_design` -> 21 tests OK.
  - `python -m py_compile train_fjmp.py FJMP/experiment_manifest.py` with temporary pycache -> OK.
  - `bash -n scripts/run_fjmp_a03_a06_repro_8gpu.sh; bash -n scripts/run_fjmp_sgv_bp_8gpu.sh` -> OK.
  - `bash scripts/run_fjmp_a03_a06_repro_8gpu.sh --base-ckpt tmp_launcher_test/dummy.pth --gpu-ids 0,1 --dry-run` -> `TOTAL_JOBS=8`, emitted `R83-00` through `R83-07`, status 0.

## 5-Question Reboot Check
| Question | Answer |
|----------|--------|
| Where am I? | Phase 5, final handoff. |
| Where am I going? | Report changed files, usage, verification, and environment limits. |
| What's the goal? | Implement the uploaded SGV-BP-FJMP ultimate design item by item. |
| What have I learned? | The existing FJMP v2 infrastructure was useful, but SGV-BP needed new modules and true paired sat-view training wiring. |
| What have I done? | Implemented SGV-BP modules, trainer args/loss wiring, manifest entries, launcher, summary scoring, tests, and verification. |
## 2026-05-18 SSDG Review
- Started comprehensive review of SSDG-related code under `E:/type10-7/code`.
- Read project instruction requiring `conda activate ssr-gpu` before project tests.
- Confirmed workspace is not a git repository.
- Located initial SSDG files: `code/train_ssdg.py`, `code/scripts/run_sgc_ssdg_6gpu.sh`, and shared helpers in `code/train.py` / `code/dataset_wisig.py`.
- Reviewed SSDG parser, source split, temporal gate, pseudo-label stage, satellite branch, checkpoint selection, launcher layout, and reused model/loss helpers.
- Attempted `pytest` in `ssr-gpu`; blocked by missing pytest module.
- Ran `python -m unittest E:/type10-7/code/tests/test_post_stage_trainers.py` inside `ssr-gpu`; 16 tests passed.
- Added failing regression tests for SSDG risk fixes; initial RED run failed as expected for missing `_resolve_epoch_schedule`, missing `_best_score`, dry-run data construction, and absent `loss_sat_cons_l`.
- Implemented SSDG fixes in `code/train_ssdg.py`: total-epoch schedule helper, early dry-run, satellite consistency KL, `--best_metric`, and metric-based best checkpoint selection.
- Re-ran `python -m unittest E:/type10-7/code/tests/test_post_stage_trainers.py` inside `ssr-gpu`; 23 tests passed.
- Ran `python -m py_compile E:/type10-7/code/train_ssdg.py E:/type10-7/code/tests/test_post_stage_trainers.py`; passed.
- Ran direct SSDG dry-run with `--epochs 180 --lambda_sat_cons 0.04 --best_metric sat_worst_tx`; exited 0 and reported `label_epochs=150 pseudo_epochs=30 total_epochs=180`.
## 2026-05-19 CVS-RFFI baseline-matched comparison run

- Started a new active addendum to run CVS-RFFI under the same comparison settings as the baseline receiver-curriculum experiments.
- Read `C:/Users/lh594/Desktop/实验组说明.md`; confirmed the relevant design uses a selected stable baseline as the dependency for post-core SGC/prototype/SSDG experiments.
- Initial search found `code/scripts/run_baseline_pseudo_rx_curriculum_6gpu.sh` and docs/logs for the baseline curriculum setup; next step is to map this to the CVS-RFFI launcher/checkpoint `BEX02_fishr002_mixed_e170`.
- Confirmed the baseline comparison matrix is receiver curriculum rather than post-stage S/P/U: `T1,T14 x P01-P06 x K2-K7`, fixed train/test days, and all non-train receivers as test receivers.
- Checked `runs/best_base_explore/BEX02_fishr002_mixed_e170`; local directory is empty, so no checkpoint can be loaded from this workspace. The runnable path is to train CVS-RFFI from scratch with BEX02 config for each receiver-curriculum row.
- Searched for `ManySig.pkl` under the workspace, `E:/`, and `C:/Users/lh594`; none found locally, so a non-dry launch will require `WISIG_PKL` to point to the dataset.
- Added `code/scripts/run_cvs_rffi_rx_curriculum_bex02_6gpu.sh`, a dynamic 6-GPU launcher that mirrors the baseline receiver-curriculum matrix and applies BEX02 config (`epochs=170`, `mixed_orbit`, `lambda_fishr=0.02`, `fishr_min_domains=4`).
- Added a targeted dry-run test in `tests/test_cvs_rffi_launcher.py`.
- Verified with `conda activate ssr-gpu; python -m unittest tests.test_cvs_rffi_launcher.CvsRffiLauncherTest.test_bex02_receiver_curriculum_dry_run_matches_baseline_smoke -v`: PASS.
- Verified `FULL` dry-run: `PLAN=FULL TOTAL_JOBS=72`, queue at `logs/cvs_rffi_bex02_rx_curriculum/queue_FULL_20260519_233249.tsv`, scheduler dry-run log at `logs/cvs_rffi_bex02_rx_curriculum/scheduler_FULL_20260519_233249.log`.
- Attempted non-dry `FULL` launch through `ssr-gpu`; launcher stopped before starting GPU jobs because `Dataset_WigSig/ManySig.pkl` is not present locally.
- Final fresh verification: targeted dry-run unit test still passes; latest `FULL` dry-run produced queue `logs/cvs_rffi_bex02_rx_curriculum/queue_FULL_20260519_233451.tsv` with 72 lines; latest non-dry launch check again stopped on missing `WISIG_PKL` before any GPU job started.

## 2026-05-25 Star-ground comparison design
- Started a new addendum for baseline-vs-CVS-RFFI satellite-ground comparison experiments.
- Connected to remote host `N607`; remote hostname is `dell-DSS8440`.
- Confirmed `/home/szu2070436088/2510044040/CV-SincNet` exists on the remote host.
- Remote script inventory includes existing B3b/CVS-RFFI satellite and baseline comparison launchers; next step is inspecting remote env/data/log state before proposing the final experiment matrix.
- User clarified the comparison must use CVS-RFFI and compare two satellite-ground augmentation methods, rather than using a baseline trainer as the comparison method.
- Added local launcher `E:/type10-7/code/scripts/run_cvs_rffi_sat_aug_compare_5gpu.sh` first, then synced it to N607 at `/home/szu2070436088/2510044040/CV-SincNet/scripts/run_cvs_rffi_sat_aug_compare_5gpu.sh`.
- Remote syntax check passed with `bash -n`.
- Remote dry-run passed with `PLAN=CORE TOTAL_JOBS=4`, using `/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python`.
- First launch failed because remote `train.py` rejects unknown `--exp_group sat_aug_method_compare`; fixed locally to use existing `--exp_group s3_rxrobust_no_dac`, then synced again.
- Relaunched CORE queue at timestamp `20260525_182115`; scheduler log is `/home/szu2070436088/2510044040/CV-SincNet/logs/cvs_rffi_sat_aug_compare/scheduler_CORE_20260525_182115.log`.
- Running jobs: `SA01_cvs_loss_mixed` on GPU 3, `SA02_strong_view_mixed` on GPU 4, `SA03_cvs_loss_all5` on GPU 5, and `SA04_strong_view_all5` on GPU 6.
- Added the local-first-then-scp rule to `E:/type10-7/AGENTS.md` and updated the `cv-sincnet` automation prompt with the same rule.
- Early status check after launch: all four jobs reached at least epoch 3/170 and are writing satellite strict UDU metrics. At E003, strong supervised variants show higher satellite strict UDU than current weak CVS-RFFI variants, while current weak variants still have higher clean/test strict UDU; wait for full convergence before drawing conclusions.
- User requested strict replication of the baseline satellite-view method and named it `拼接星地信道增强`.
- The earlier approximate queue was briefly terminated by mistake, then restarted without overwriting partial results under remote `runs/cvs_rffi_sat_aug_compare_keep` and `logs/cvs_rffi_sat_aug_compare_keep`.
- Added local module `E:/type10-7/code/concat_sat_channel_aug.py` implementing clean+sat batch concatenation with duplicated labels/domain tensors.
- Updated local `E:/type10-7/code/train.py` with `--use_concat_sat_channel_aug` and `--concat_sat_start_epoch`; when concat mode is enabled, the old auxiliary sat consistency branch is not applied.
- Updated local `E:/type10-7/code/scripts/run_cvs_rffi_sat_aug_compare_5gpu.sh` so `SA02`/`SA04` are now strict `concat_sat` experiments rather than strong auxiliary sat-CE approximations.
- Added local test `E:/type10-7/code/tests/test_concat_sat_channel_aug.py`; local py_compile passed, while local unittest is blocked because the local `ssr-gpu` env lacks `torch`.
- Backed up remote `train.py` and launcher to `E:/type10-7/code/snapshots/concat_sat_channel_aug_20260525_183248/` before syncing.
- Synced `train.py`, `concat_sat_channel_aug.py`, the new test, and the launcher to N607.
- Remote verification passed: `python -m py_compile train.py concat_sat_channel_aug.py tests/test_concat_sat_channel_aug.py`; `python -m unittest tests.test_concat_sat_channel_aug -v` ran 2 tests OK.
- Remote strict concat comparison queue launched on GPU 7 under `runs/cvs_rffi_concat_sat_compare` and `logs/cvs_rffi_concat_sat_compare`; scheduler log `scheduler_CORE_20260525_183355.log`, launcher PID `222146`.

## 2026-05-25 Spaceborne FL-DG-FSL Prototype Synthesis

- Started a broad research synthesis requested by the user for spaceborne RFFI with FL, DG, few-shot learning, and multi-prototype heads.
- Read the local Fed-PVS-RFFI plan and collaborative prototype fusion report as reference materials.
- Updated `task_plan.md` with a new active addendum for this synthesis.
- User clarified that earlier materials are references and that related experiment results are available on N607.
- Queried N607 inventory under `/home/szu2070436088/2510044040/CV-SincNet`; found relevant run/log families for federated few-shot DG, prototype smoke, satellite augmentation comparison, SGC/SSDG, target adaptation, and baseline satellite evaluation.
- Read `C:/Users/lh594/Downloads/fed_pvs_rffi_research_plan.md`, `E:/type10-7/docs/federated_collaborative_prototype_fusion_report.md`, and `E:/type10-7/code/docs/fed_fewshot_dg_experiments.md`.
- Queried N607 final metrics for `runs/fed_fewshot_dg/*/metrics.csv`; extracted the main CE-only FL, FedProx, BEX02 local-DG, receiver-agnostic FL, and satellite-augmentation comparisons.
- Queried N607 satellite augmentation logs under `logs/cvs_rffi_sat_aug_compare_keep`, `logs/cvs_rffi_concat_sat_compare`, and `logs/b3b_asym_sat_baseline`; recorded the clean strict UDU vs satellite strict UDU tradeoff and the failed concat-sat launcher conflict.
- Queried `runs/fed_proto_smoke`; confirmed only smoke-level prototype-bank plumbing evidence is available.
- Browsed primary or near-primary sources for FedAvg, FedProx, Fishr, MixStyle, receiver-agnostic/cross-receiver RFFI, FedDG style-transfer/augmentation, FedProto, MAML/prototypical networks, and cross-domain few-shot learning.
- Wrote the integrated synthesis and roadmap to `E:/type10-7/docs/spaceborne_fed_dg_fsl_prototype_synthesis.md`.

## 2026-05-25 Fed-PVS-CPRFFI final design integration

- Began analysis of `C:/Users/lh594/Downloads/fed_pvs_cprffi_final_design.md` against the existing CVS-RFFI/CV-SincNet codebase.
- Re-read project instructions and existing planning state.
- Confirmed the design report exists and listed its headings, including StyleBank, ProtoBank, multi-prototype head, base-anchor fusion, few-shot adaptation, stage scheduling, engineering modules, experiments, and CVS-RFFI appendices.
- Confirmed the local code surface has federated training, FedProx/FedProto stats, satellite augmentation, SGC/SSDG/FJMP post-stage modules, tests, and sync manifest infrastructure.
- Read the report's core sections on StyleBank, style-anchored virtual domains, DG activation conditions, ProtoBank, conservative fusion, few-shot adaptation, stage scheduling, engineering modules, ablations, diagnostics, satellite extension, risks, and appendices D/E.
- Inspected `code/train.py`, `code/federated/fed_trainer.py`, `fed_aggregate.py`, `client_split.py`, `model_dual_cvsincnet.py`, `DataAugmentation.py`, `sat_channel.py`, `concat_sat_channel_aug.py`, FJMP/SGC prototype surfaces, federated docs, and tests.
- Main code gap identified: the current FedProto path is single-prototype and the federated local objective is not yet multi-style-view with a separate constructed `d_style`.
- Wrote the detailed integration analysis to `E:/type10-7/docs/fed_pvs_cprffi_cvs_integration_analysis.md`.
- Verified the document exists, has the expected section headings, and begins with the correct source/report metadata.

## 2026-05-25 Fed-PVS-CPRFFI strategy loophole audit

- Treated the user's confidence challenge as a technical review of the integration strategy.
- Re-read the current integration document, planning state, and the relevant local code paths.
- Verified local gaps in `code/federated/fed_trainer.py`, `code/train.py`, `code/federated/fed_aggregate.py`, `model_dual_cvsincnet.py`, `DataAugmentation.py`, `sat_channel.py`, and prototype-related modules.
- Queried N607 under `/home/szu2070436088/2510044040/CV-SincNet` for `fed_fewshot_dg`, `fed_proto_smoke`, satellite augmentation, concat-sat, and B3b asym satellite logs/metrics.
- Confirmed the important evidence anchors: CE-only receiver-day FL around 71 strict UDU, direct receiver-day BEX02 FL around 69.6-69.9 strict UDU, `FSDG49` around 75.92 strict UDU, `FSDG50` around 70.52 strict UDU, and invalid strict concat-sat SA02/SA04 due to an argument conflict.
- Enumerated the main loopholes: missing `d_style`, single-prototype FedProto, invalid concat-sat evidence, possible clean-vs-sat regression, weak FedProx contribution evidence, privacy/fingerprint leakage risk, random-vs-conditioned ambiguity, client granularity ambiguity, local-parameter aggregation risk, and premature few-shot masking.
- Wrote the revised strategy and loophole ledger to `E:/type10-7/docs/fed_pvs_cprffi_strategy_loophole_audit.md`.
- Updated `task_plan.md` and `findings.md` with the new confidence boundary and staged gates.
- Re-audited the new document and added missing confidence controls for multi-seed statistics, split/eval leakage, and local-to-N607 version drift.

## 2026-05-26 Fed-PVS-CPRFFI Phase -1/Phase 1 code implementation

- Started implementation from the approved V2 loophole-audit plan.
- Added RED tests for StyleBank/StylePacket, ProtoEvidenceBank/reliability fusion, federated `d_style` plumbing, aggregation exclusion, CLI exposure, and concat-sat launcher hygiene.
- Initial local unittest invocation using `conda activate ssr-gpu` was unreliable in PowerShell and initially showed missing torch; switching to `conda run -n ssr-gpu` provided the correct local test execution path.
- Implemented new federated modules: `style_packet.py`, `rf_style_extractor.py`, `style_bank.py`, `virtual_domain_sampler.py`, `conditioned_receiver_dg.py`, `proto_evidence_bank.py`, and `reliability_fusion.py`.
- Updated `fed_trainer.py` with optional StyleBank diagnostics, optional `style_batch_fn` expansion, constructed `d_style` routing for model/domain losses, style metrics, StyleBank summaries, and aggregation exclude-key support.
- Updated `fed_aggregate.py` with `resolve_exclude_keys` for exact-key and prefix local-only state exclusion.
- Updated `train.py` with StyleBank diagnostic and aggregation-exclusion CLI flags.
- Updated `scripts/run_cvs_rffi_sat_aug_compare_5gpu.sh` so concat-sat rows do not execute with both `--use_sat_consistency` and `--no_use_sat_consistency`.
- Created local snapshot `E:/type10-7/code/snapshots/20260526_000700_fed_pvs_phase1_impl/` before syncing to N607.
- Updated `E:/type10-7/code/SYNC_MANIFEST.txt` with the local-to-remote mapping for this sync.
- Synced changed files to `/home/szu2070436088/2510044040/CV-SincNet` on N607 with `scp`.
- Remote test pass after one fix loop: replaced complex `rfft` with complex `fft` in `RFStyleExtractor`, and corrected the conservative-fusion test sample to match `rho<=0.05` behavior.
- Final local verification: `conda run -n ssr-gpu python -m py_compile ...` passed; `conda run -n ssr-gpu python -m unittest tests.test_fed_pvs_style_bank tests.test_fed_pvs_proto_fusion tests.test_federated_d_style_plumbing tests.test_federated_aggregation tests.test_federated_train_integration tests.test_cvs_rffi_sat_aug_launcher -v` passed 12 tests, skipped 2 local bash/WSL tests.
- Final N607 verification: `/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python -m unittest tests.test_fed_pvs_style_bank tests.test_fed_pvs_proto_fusion tests.test_federated_d_style_plumbing tests.test_federated_aggregation tests.test_federated_train_integration tests.test_cvs_rffi_sat_aug_launcher -v` passed all 14 tests.

## 2026-05-26 Post-implementation confidence loop

- Re-ran the confidence challenge as a post-implementation audit.
- Found and fixed compactness issue in `d_style`: virtual batches now use compact constructed labels and preserve original style ids in metadata.
- Found and fixed domain-head label-range issue: `FederatedTrainer` skips CE-style domain losses if the constructed domain targets cannot be represented by the available head.
- Found and fixed StyleBank schema/trim issues: bank vectors use a stable numeric stat schema and trimming prioritizes count/newness.
- Found and fixed local launcher-test hang risk with subprocess timeouts.
- Local targeted RED/GREEN checks confirmed the new tests initially caught the `d_style` compactness problem, then passed after the fix.
- Local final verification: `conda run -n ssr-gpu python -m py_compile ...` passed; `conda run -n ssr-gpu python -m unittest tests.test_fed_pvs_style_bank tests.test_fed_pvs_proto_fusion tests.test_federated_d_style_plumbing tests.test_federated_aggregation tests.test_federated_train_integration tests.test_cvs_rffi_sat_aug_launcher -v` ran 16 tests: 15 passed, 1 skipped because the local bash dry-run timed out under Windows/WSL behavior.
- Synced changed files to N607 and verified remotely: `/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python -m py_compile ...` plus the same unittest suite passed 16/16.

## 2026-05-26 FL82 federated validation launch

- Created `E:/type10-7/code/scripts/run_fed_fl82_validation_4gpu.sh` for the FL82 CORE matrix.
- Created persistent report `E:/type10-7/automation_reports/CV-SincNet/20260526_004220_fl82_fed_validation/report.md`.
- Local verification passed: `conda run -n ssr-gpu python -m py_compile train.py federated/fed_trainer.py`; later dependency checks also passed for `baseline_origin_sat_view.py`, `concat_sat_channel_aug.py`, `train.py`, and `cvsrffi/*.py`.
- Created snapshot `E:/type10-7/code/snapshots/20260526_004220_fl82_fed_validation` and updated `E:/type10-7/code/SYNC_MANIFEST.txt`.
- Synced launcher, `baseline_origin_sat_view.py`, and `cvsrffi/*.py` to N607 after local checks exposed missing remote dependencies.
- Remote checks passed: `bash -n scripts/run_fed_fl82_validation_4gpu.sh`, dry-run with CORE/GPU 3,4,5,7, and `/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python -m py_compile train.py baseline_origin_sat_view.py concat_sat_channel_aug.py cvsrffi/*.py`.
- Initial launches failed and were recorded: missing `baseline_origin_sat_view`, missing `cvsrffi`, and unsupported `--exp_desc`.
- Final launch is running despite local SSH wrapper timeout. Active scheduler: `/home/szu2070436088/2510044040/CV-SincNet/logs/fl82_fed_validation/scheduler_CORE_20260526_005707.log`.
- Startup logs confirm all four FL82 jobs entered training and print clean strict UDU plus five satellite scenarios every round.
- Updated `cv-sincnet` automation with the explicit FL82 target: validate FL, pursue clean strict UDU >=82%, and monitor per-round clean/satellite testing.
- Added and verified `[SAT-TEST-SPLIT]` logging plus the `SAT_BASELINE` launcher plan for baseline-style clean+sat supervised view expansion.
- Synced the updated launcher, `federated/fed_trainer.py`, and integration test to N607 after local compile/unittest and snapshot/hash recording; remote compile, targeted unittest, and `PLAN=SAT_BASELINE --dry-run` passed.
- Launched `SAT_BASELINE` on GPUs 0,1,2. Active scheduler: `/home/szu2070436088/2510044040/CV-SincNet/logs/fl82_fed_validation/scheduler_SAT_BASELINE_20260526_014110.log`; active logs are `FL82_07_fedprox_rx_ra_bex02_baselineview_all5_r020.log`, `FL82_08_fedprox_rx_ra_bex02_baselineview_all5_stylebank_l3_r020.log`, and `FL82_09_fedprox_rx_ra_bex02_baselineview_clearleo_l3_r020.log`.
- Verified round-1 output includes requested clear_leo split metrics for `test_unseen_day_seen_rx`, `test_seen_day_unseen_rx`, and `test_unseen_day_unseen_rx`; current numbers are warmup only and below target.
- Follow-up log check reached round 2 for all three SAT_BASELINE runs with clean strict UDU around 31-33% and clear_leo splits around 30-33%; still early warmup and below target.
- Latest CORE read: `FL82_01` R035 strict UDU 67.31, `FL82_02` R033 strict UDU 67.17, `FL82_03` R033 strict UDU 73.75, and `FL82_04` R030 strict UDU 74.06; all are still running and below the 82% success threshold.
- Updated `cv-sincnet` automation with durable targets: clean strict UDU >=82 and clear_leo split floors 84.30/60.10/53.78, with active-job monitor-only behavior and local-first sync requirements.
- Corrected the formal FL constraint after user clarification: train ratio must be `0.1`, and default epoch/round count is now `200`.
- Updated local `run_fed_fl82_validation_4gpu.sh`, `test_federated_train_integration.py`, `SYNC_MANIFEST.txt`, and `AGENTS.md`; snapshotted the changed code/test/manifest files.
- Synced the updated launcher, test, and manifest to N607; remote `bash -n`, targeted unittest, hash check, and `PLAN=SAT_BASELINE --dry-run` confirmed future commands use `--wisig_train_ratio 0.1 --epochs 200 --fl_rounds 200` with `r010` run names.
- Updated the `cv-sincnet` automation so `0.2/220` active runs are historical/debug-only and cannot satisfy the current target.

## 2026-05-26 Federated log forensics and FL82 diagnosis

- Started a new evidence-first analysis of the N607 exhaustive log backup at `E:/type10-7/server_log_backups/N607/20260526_101853/exhaustive_log_files`.
- Goal is to reconstruct every federated experiment configuration, then diagnose why `fl82_fed_validation` underperforms the centralized `BEX02_fishr002_mixed_e170` CVS-RFFI baseline.
- Will add logging code only after confirming the required diagnostic data is absent from existing logs.
- First inventory found seven `fl82_fed_validation` training logs plus scheduler/queue files, and older `fed_fewshot_dg`/FedProto artifacts in both logs and runs trees.
- Existing FL82 logs contain per-round dispatch, strict clean split, satellite split, and prediction-histogram lines, so the next step is to parse them into structured CSV/Markdown before drawing conclusions.
- User added focus areas: explain domain-generalization test performance, determine whether satellite-ground channel augmentation is effective, and propose effective high-ROI optimization directions.
- Parsed 28 federated summaries, 4392 federated round records, and 170 centralized BEX02 epochs into `E:/type10-7/analysis/federated_log_forensics/`.
- Wrote detailed diagnosis report `E:/type10-7/analysis/federated_log_forensics/fl82_federated_dg_diagnosis.md`.
- Key conclusion: FL82 best clean strict UDU is `79.04%`, far below centralized `BEX02_fishr002_mixed_e170` at `85.97%`; the available logs show FL82's Fishr/domain/consistency terms are effectively `0.0`, so centralized DG was not truly replicated inside local FL clients.
- Satellite augmentation conclusion: all-scenario `baseline_view` improves satellite strict UDU by roughly 2-3 points over CVS consistency, but clean strict UDU drops; clear-leo-only overfits one scenario and hurts clean DG badly.
- Added DG activation diagnostics to `code/federated/fed_trainer.py`, with JSONL/CSV/stdout fields for domain count, Fishr eligibility, DG branch activation, satellite branch activation, and style-batch activity.
- Added regression coverage in `code/tests/test_federated_trainer_smoke.py`; local `conda run -n ssr-gpu` smoke/py_compile passed.
- Created local snapshot `E:/type10-7/code/snapshots/20260526_104621_fed_dg_diag_logging`, updated `code/SYNC_MANIFEST.txt`, synced the trainer/test/manifest to N607, and verified remote hashes plus remote unittest pass.

## 2026-05-26 StyleBank/ProtoBank design-vs-implementation audit

- Started audit against `C:/Users/lh594/Downloads/fed_pvs_cprffi_final_design.md`.
- Skill catchup found the immediately preceding FL82 diagnosis context; no unsynced code action is required for this audit.
- `git -C E:/type10-7/code diff --stat` failed because the code tree is not a git repository; use explicit file inspection and existing snapshots/manifests for version evidence.
## 2026-05-26 StyleBank/ProtoBank audit

- Compared the design report against current federated modules and FL82/fed_proto logs.
- Wrote the detailed audit to `E:\type10-7\analysis\federated_log_forensics\stylebank_protobank_design_gap_audit.md`.
- Main conclusion: StyleBank currently works as diagnostics/stat collection only; ProtoBank is not active in federated training. Current code is a partial scaffold rather than the full design-report algorithm.

## 2026-05-26 StyleBank/ProtoBank design-parity implementation

- Implemented default-on federated StyleBank training views in `code/federated/fed_trainer.py`.
- Added staged StyleBank controls and default-on CLI flags in `code/train.py`; formal defaults are now `wisig_train_ratio=0.1`, `epochs=200`, and `fl_rounds=200`.
- Added multi-remote style sampling in `code/federated/style_bank.py`.
- Wired ProtoEvidenceBank collection and conservative harm/rescue fusion diagnostics into federated training/evaluation.
- Emphasized GRL/receiver-adversarial activation on constructed `d_style` domains; console diagnostics now include `grl_rx_adv_active`.
- Added/updated tests for default StyleBank batch construction, ProtoBank evidence/fusion reporting, CLI exposure, and GRL diagnostics.
- Verification under `ssr-gpu`: focused pytest suite passed `22 passed, 1 skipped`; py_compile passed; `train.py --help` shows new flags.
- Snapshot saved under `E:\type10-7\code\snapshots\20260526_stylebank_protobank_design_parity\`.
- Synced the verified files to N607 after creating remote backup `snapshots/20260526_stylebank_protobank_design_parity`.
- Remote verification used `/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python` because `ssr-gpu` was not present on N607; remote py_compile and unittest passed `Ran 23 tests ... OK (skipped=1)`.
- Added a hard guard so non-centralized WiSig federated training refuses any `--wisig_train_ratio` other than `0.1`; re-verified locally and remotely after syncing `train.py` and `test_federated_train_integration.py`.
- Corrected the default StyleBank batch semantics for the dual-backbone architecture: generated remote-style views now use their target receiver/domain labels for `dom_head(z_dom)` and GRL/`adv_head(z_id)`, rather than compact view-index labels. Local pytest passed `22 passed, 1 skipped`; N607 targeted unittest passed `15 tests OK`.
- Fixed additional StyleBank target-domain risks: centroids now preserve target-domain metadata, merge only across compatible target-domain metadata, and default remote style replay skips styles whose target domain equals the local clean domain. Also changed immature style batches to avoid remapping synthetic mapped labels before DG gates activate.
- Added logging for `zdom_target_acc` and `grl_target_acc` so experiments can directly inspect whether `z_dom` classifies target domains and whether GRL is de-domainizing `z_id`.
- Re-verified locally under `ssr-gpu`: `24 passed, 1 skipped`; synced to N607 with backups `20260526_stylebank_domain_metadata_fix` and `20260526_stylebank_zdom_grl_metrics`; remote targeted unittest passed `17 tests OK`.
- Added startup federated configuration snapshots for future analysis: `federated_config.json`, a `logs.jsonl` `fed_config` event, and stdout `[FED-CONFIG-*]` blocks covering data split, FL, StyleBank, ProtoBank, GRL, losses, satellite augmentation, and evaluation.
- Added regression coverage in `code/tests/test_federated_trainer_smoke.py`.
- Local verification under `ssr-gpu`: focused federated pytest suite passed `24 passed, 1 skipped`; `py_compile` passed for `federated/fed_trainer.py` and `train.py`.
- Snapshot saved under `E:\type10-7\code\snapshots\20260526_federated_config_snapshot\`; synced `fed_trainer.py` and the smoke test to N607 after remote backups.
- Remote verification used `/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python`; `py_compile` passed and targeted unittest passed `Ran 12 tests ... OK (skipped=1)`.
- Enforced per-round satellite-channel testing for federated training: `train.py` now rejects `train_mode=fedavg/fedprox` with `--no_eval_sat_channel`, and default FL satellite eval expands from strict UDU only to `test_unseen_day_seen_rx,test_seen_day_unseen_rx,test_unseen_day_unseen_rx`.
- `federated_config.json` / `[FED-CONFIG-SAT]` now records `per_round_satellite_eval` / `per_round_eval` and the effective `eval_sat_on` split list.
- Added tests proving the argument guard and proving `FederatedTrainer` calls/prints extra satellite eval on every round.
- Local verification under `ssr-gpu`: `30 passed, 1 skipped`; `py_compile` passed for `train.py`, `fed_trainer.py`, `rf_style_extractor.py`, and `conditioned_receiver_dg.py`.
- Synced to N607 with backup `snapshots/20260526_federated_sat_eval_every_round`; remote verification passed `Ran 14 tests ... OK`. A stale remote StyleBank dependency was found and corrected by syncing `rf_style_extractor.py` and `conditioned_receiver_dg.py`.

## 2026-05-26 Reusable workflow packaging audit

- Extracted the image command text manually from the supplied screenshot for final response.
- Refreshed local conversation index: `conda activate ssr-gpu; python tools/conversation_index.py build` -> indexed 99 E:\type10-7 entries.
- Searched conversation index for N607 experiment workflows, FL82/StyleBank/ProtoBank diagnosis, baseline audit/augmentation, and target-domain adaptation launchers.
- Searched Codex memory and read the active `cv-sincnet` automation prompt and memory.
- Inventoried installed local skills and found no existing `C:\Users\lh594\.codex\skills\cv-sincnet-n607-automation` skill, despite memory referencing it as a related skill.
- Created `C:\Users\lh594\.codex\skills\cv-sincnet-n607-automation\SKILL.md` and fixed `agents/openai.yaml`.
- Updated the `cv-sincnet` automation with conversation-index lookup and default `--fl_client_key receiver` for formal federated launches.
- Verification: default Python failed `quick_validate.py` due missing `yaml`; `conda activate ssr-gpu; python ...quick_validate.py ...cv-sincnet-n607-automation` passed with "Skill is valid!".
- Re-ran the audit after user feedback about design-report implementation omissions.
- Searched conversation index and Codex memory for design-report implementation / gap-audit patterns, including FJMP, SGC, SGV-BP, and Fed-PVS/StyleBank/ProtoBank work.
- Inventoried existing skills and found no design-report traceability skill.
- Created `C:\Users\lh594\.codex\skills\design-report-traceability\SKILL.md` and `agents/openai.yaml`.
- First validation failed because Chinese trigger phrases in the skill file hit Windows GBK decoding; rewrote the skill as ASCII while preserving the trigger intent.
- Verification: `conda activate ssr-gpu; python ...quick_validate.py ...design-report-traceability` passed with "Skill is valid!".

## 2026-06-01 Paper-exact CVS-RFFI reproduction setup

- User requested reproduction experiments that match each comparison paper's original settings under the CVS-RFFI data configuration, with multi-subagent supervision/review.
- Read current `AGENTS.md` and confirmed hard constraints: local `ssr-gpu` verification, local-first edits, direct `N607` preflight before SSH/SCP, federated `wisig_train_ratio=0.1`, default `epochs=200`, `fl_rounds=200`, `--fl_client_key receiver`, and experiment reports before N607 launch.
- Refreshed conversation index with `conda activate ssr-gpu; python tools/conversation_index.py build`; result: indexed 177 entries.
- Searched conversation index for CVS-RFFI paper/baseline reproduction history and found prior baseline audit, baseline-vs-CVS satellite-view analysis, and earlier baseline-matched CVS-RFFI experiment requests.
- Added a new active addendum to `task_plan.md` and created `analysis/cvsrffi_paper_reproduction_traceability_20260601.md` for paper-setting parity tracking.
- Inventory found the current active baseline methods under `baselines/`: `cvcnn_ce`, `riei_fd`, `drift`, and `ra_collab`; old `tifs2025` artifacts remain under `baselines/baseline_runs` but no current trainer directory is present in the active baseline package.
- Superseded on 2026-06-26: `scripts/launchers/run_cvs_baseline_queue.sh` now defaults to `SAT_VIEW_AUG=0`; training-time satellite view augmentation is enabled only by `--sat-view-aug`. Satellite evaluation remains an additional project metric, not original paper-style training evidence.
- Another trap: individual trainers default to `wisig_train_ratio=0.2`, while the queue defaults to ratio `0.1` and val `0.9`; the reproduction commands must set the CVS-RFFI data split explicitly.
- Dispatched three read-only review subagents: paper-setting parity, code/launcher parity, and prior-run exclusion. They converged on the current runnable baseline set `cvcnn_ce/riei_fd/drift/ra_collab`, with FedRIEI blocked by no runnable entrypoint and old TIFS2025 only present as artifacts.
- Historical verification under `ssr-gpu`: all four trainer `--help` commands exited 0; Python baseline modules passed `py_compile`; launcher syntax and exact queue dry-run passed with `sat_view_aug=0`.
- Created experiment report `automation_reports/CV-SincNet/20260601_152529_paper_exact_baselines_r010/report.md` before any N607 launch.
- N607 preflight passed; initial remote inventory showed all 8 GPUs idle and target exact run/log roots absent, but remote baseline package was stale/missing current `cvcnn_ce`, `riei_fd`, and `ra_collab` directories.
- Created local snapshot under `code/snapshots/20260601_152529_paper_exact_baselines_r010/`, created remote pre-sync backup, synced the current verified baseline directories and launcher with direct `scp`, then verified remote hashes, `bash -n`, `py_compile`, and remote dry-run. Post-SSH/SCP cleanup checks showed no lingering local SSH connection.
- Launch command timed out locally before returning `LAUNCH_PID`; identified stale local `ssh.exe` PID 36052 and stopped it, verifying no N607 port-22 connection remained. Remote inspection confirmed the queue launched successfully.
- Active N607 run: four paper-original baseline/control jobs under `logs/wisig_baselines_paper_exact_seed1337_ratio010` and `runs/wisig_baselines_paper_exact_seed1337_ratio010`, PIDs 664267/664285/664293/664300 on GPUs 0/1/2/3. Startup logs reached epoch 1 for all methods; `ra_collab` entered epoch 2. No startup Traceback/OOM/argparse/import errors observed.

## 2026-06-01 Original-paper protocol correction

- User clarified the required target is complete reproduction under each original paper's original dataset protocol.
- Marked the launched CVS-RFFI/WiSig r010 queue as non-target auxiliary evidence, not original-paper protocol reproduction.
- Created `analysis/cvsrffi_original_paper_protocol_traceability_20260601.md` to track exact original dataset/protocol requirements and dataset availability.
- Per AGENTS.md safety rules, did not stop active N607 jobs because the user has not explicitly requested killing/stopping them.
- User further clarified that papers without WiSig data should align to the original settings of papers that do use WiSig.
- Implemented the first executable WiSig original-paper protocol: `--wisig_protocol drift_day1`, matching DRIFT's WiSig/ManySig Day1 receiver-disjoint setup with train receivers `1-1,14-7,7-7`, test receivers `1-19,19-2,2-1,2-19,20-1,7-14,8-8`, and 800/200 train/test sample counts.
- Added `paper_eval_window` last-N reporting so DRIFT can report `last5`, RIEI can report `last10`, and non-WiSig methods aligned to the WiSig protocol can report `aligned_wisig_last5`.
- Updated the baseline queue launcher to pass the explicit protocol, receiver groups, sample counts, disabled satellite train augmentation, and method-specific last-N windows.
- Local verification under `ssr-gpu` passed: touched Python `py_compile`, `BaselineWiSigPaperProtocolTest`, full `tests.test_baseline_training_behaviors`, launcher syntax, and `drift_day1` dry-run.
- Created N607 gate report `E:\type10-7\automation_reports\CV-SincNet\20260601_160423_wisig_drift_day1_original_protocol\report.md`.
- N607 preflight passed; GPU 4-7 were free while the earlier auxiliary CVS-RFFI-config jobs remained on GPU 0-3.
- Synced verified files to N607 after remote backup `snapshots/20260601_160423_wisig_drift_day1_original_protocol_pre_sync`.
- Remote verification passed: hashes matched, `bash -n` passed, `py_compile` passed, `BaselineWiSigPaperProtocolTest` passed, and dry-run showed the exact `drift_day1` protocol.
- Launched `wisig_original_protocol_drift_day1_seed1337` on GPU 4-7. New PIDs: `685180` cvcnn_ce, `685194` riei_fd, `685207` drift, `685214` ra_collab.
- Startup health check at about 5 minutes showed all four jobs still running and advancing, with no error keywords in target logs.
- Completion analysis on 2026-06-01 18:18 CST found the target PIDs gone, scheduler `status=0`, all four `metrics.json` files present, and no error keywords in the copied full logs/metrics.
- Remote loader split reconstruction confirmed the DRIFT Table I first-group split: Day1 `2021_03_01`, train receivers `1-1,14-7,7-7`, test receivers `1-19,19-2,2-1,2-19,20-1,7-14,8-8`, train size `14400`, test size `8400`, per test receiver `1200`.
- Final performance did not match the paper: DRIFT last5 was `52.09 +/- 1.93` vs paper Table I first-group DRIFT average `75.62`; RIEI last10 was `53.24 +/- 10.27` vs paper RIEI `62.36`; CVCNN aligned last5 was `61.80 +/- 0.71`.
## 2026-06-13 CVS-RFFI Audit Progress

- Started evidence-driven audit for ground-training to spaceborne few-shot deployment.
- Read required skills: training-log-analysis, cv-sincnet-n607-automation, dispatching-parallel-agents, planning-with-files, academic-writing.
- Appended audit plan to `task_plan.md`.
- Spawned five read-only explorer subagents:
  - Agent A: scenario and code implementation audit.
  - Agent B: data/protocol audit.
  - Agent C: results/metrics audit.
  - Agent D: target/performance/root-cause audit.
  - Agent E: reproducibility/improvement audit.
- Initial broad structure/keyword scan produced excessive output and hit access-denied on `.pytest_cache`, `code/.pytest_cache`, and `tmpwhf_cnj1`; switching to targeted evidence extraction.

## 2026-06-13 CVS-RFFI audit artifacts
- Generated AUDIT_CVS_RFFI_GROUND_TO_SPACE_FSL.md, metrics_summary.csv, evidence_map.csv, missing_experiments.md, improvement_plan.md.
- Audit conclusion: partial achievement; SFE/FTRC implemented but open-set and deployable old-class adaptation remain unproven.

## 2026-06-21 CVS automation trim and multi-agent review

- Read `AGENTS.md`, `项目.md`, current control manifest, workflow contract, state, active stage2 prompt, validator, runner template, and local conversation/memory context.
- Spawned three read-only subagents: historical/control-plane defects, code-surface changes, and project-protocol semantics. Their reviews converged on `validator PASS != launch permission`, route duplication repair, state/prompt bloat, and single-`r_sat` protocol drift.
- Added `launchability_summary` to `tools/optimizer_validate_matrix.py`, including lane-level `runner_readiness`, launchable/deferred/non-launchable counts, local-patch counts, and route-duplication repair counts.
- Added validator hard gates for launchable rows: route-duplication repair cannot be launchable; 64-row matrices require unique `registry_key` and `command_hash`; launchable Phase2 rows require exactly one target receiver and valid Stage2-B/C K-shot grid `{1,2,5,10,15,20,50}`.
- Added `tools/optimizer_state_current_view.py` to generate a compact current-decision view and keep `objective_changelog`, `target_changelog`, and lane subtrees audit-only.
- Updated `tools/optimizer_control_manifest.md`, `tools/optimizer_workflow_contract.md`, and the active `stage2_prompt.md` so `项目.md` is loaded after `AGENTS.md` and before control files.
- Corrected top-level `stage2_optimizer_state.json` Phase2 protocol default from mixed `rx7-rx11` to single default `rx7`, with `rx8-rx11` as sensitivity receivers only.
- Corrected current local next64cq launcher default `TARGET_RXS` from `7,8,9,10,11` to `7`; Phase2 remains blocked by local patch/route duplication until a repaired matrix is generated.
- Verification passed:
  - `bash -n code/scripts/launch_stage2_optimizer_20260621_211348_next64cq.sh`
  - `conda run -n ssr-gpu python -m py_compile tools/optimizer_validate_matrix.py tools/optimizer_state_current_view.py tools/optimizer_workflow_lib.py`
  - `conda run -n ssr-gpu python -m pytest -p no:cacheprovider -q code/tests/test_optimizer_workflow_tools.py` -> `25 passed`
  - `conda run -n ssr-gpu python tools/optimizer_validate_matrix.py automation_reports/CV-SincNet/stage2_optimizer_20260621_211348/artifacts/stage2_candidate_matrix.json --expected-count 64 --output automation_reports/CV-SincNet/stage2_optimizer_20260621_211348/artifacts/matrix_validation_next64cq_with_launchability.json` -> PASS with 8 launchable / 56 non-launchable rows
  - `conda run -n ssr-gpu python tools/optimizer_state_current_view.py automation_reports/CV-SincNet/stage2_optimizer_state.json --output automation_reports/CV-SincNet/stage2_optimizer_state.current_view.json`
- Verification caveat: parallel `conda run` on Windows repeatedly conflicted on the same temp file. Use sequential `conda run` for this workflow.

## 2026-06-22 K-shot policy relaxation

- User requested that the project K grid should not be rigidly fixed.
- Updated `项目.md` first, per `AGENTS.md`: Stage2-B/C K is now any positive integer, with `{1,2,5,10,15,20,50}` retained only as recommended anchors for comparable curves.
- Updated `tools/optimizer_workflow_contract.md` and `tools/optimizer_validate_matrix.py`: validator no longer rejects intermediate K values such as 3/4/8/12/16. It still requires explicit positive K for Stage2-B/C, and requires `K>20` to be labeled higher-shot, medium-shot, or saturation.
- Added `stage2_sample_protocol.k_shot_policy` to `automation_reports/CV-SincNet/stage2_optimizer_state.json`.
- Updated `code/tests/test_optimizer_workflow_tools.py`: focused tests now cover intermediate positive K acceptance, non-positive K rejection, and high-K interpretation requirement.
- Verification passed:
  - `conda run -n ssr-gpu python -m py_compile tools/optimizer_validate_matrix.py tools/optimizer_state_current_view.py tools/optimizer_workflow_lib.py`
  - `conda run -n ssr-gpu python -m pytest -p no:cacheprovider -q code/tests/test_optimizer_workflow_tools.py` -> `27 passed`
  - `conda run -n ssr-gpu python tools/optimizer_state_current_view.py automation_reports/CV-SincNet/stage2_optimizer_state.json --output automation_reports/CV-SincNet/stage2_optimizer_state.current_view.json`
  - `conda run -n ssr-gpu python tools/optimizer_validate_matrix.py automation_reports/CV-SincNet/stage2_optimizer_20260622_000742/artifacts/stage2_candidate_matrix.json --expected-count 64 --output automation_reports/CV-SincNet/stage2_optimizer_20260622_000742/artifacts/matrix_validation_next64cr_flexible_k.json`
