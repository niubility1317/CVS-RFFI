# Findings & Decisions

## Baseline Audit Request - 2026-05-19
- Current request: comprehensively review papers and corresponding code in `E:/type10-7/baselines`, check whether code matches paper descriptions, optimize very slow `tifs2025` training, and ensure every comparison method tests on the test set after training, including satellite-ground channel evaluation.
- Project test instruction: activate Conda environment with `conda activate ssr-gpu` before project-related code tests.
- Workspace is not a Git repository; use direct file inspection and planning logs for change tracking.
- Initial baseline inventory:
  - PDFs: `Toward_Channel-Robust_and_Receiver-Independent_Radio_Frequency_Fingerprint_Identification.pdf`, `Receiver-Agnostic_Radio_Frequency_Fingerprint_Identification_via_Feature_Disentanglement.pdf`, `2510.09405v1 (1).pdf`, `2207.02999v1.pdf`.
  - Method folders: `tifs2025_channel_receiver_rffi`, `receiver_agnostic_rffi`, `riei`, `drift`, `cvcnn`, and shared `common`.
  - Shared evaluation utilities include `common/cvs_trainer.py`, `common/cvs_sat_eval.py`, `common/cvs_data.py`, and baseline run/log folders.
- Initial code evidence:
  - `baselines/common/cvs_trainer.py` runs named test and satellite extra tests only inside `if gate.should_test(...)`, so the final trained model is not guaranteed to be tested after training ends.
  - `baselines/tifs2025_channel_receiver_rffi/train_cvs.py` performs a 100-epoch NT-Xent pretrain plus 100-epoch Siamese fine-tune by default, which is inherently heavier than single-stage baselines.
  - TIFS2025 pretraining uses `PretrainDataset`, whose `__getitem__` applies online RF channel augmentation and `SpectrogramTransform` twice per sample.
  - TIFS2025 Siamese training uses `SiamesePairDataset`, whose `__getitem__` applies online RF channel augmentation and `SpectrogramTransform` once for each side of the pair.
  - `build_cvs_loaders(..., transform_train=spec, transform_eval=spec)` creates spectrogram-transformed validation/test loaders, and `raw_eval_loaders = build_cvs_loaders(args, device)` creates an additional raw evaluation loader set when satellite evaluation is enabled.
  - Default `num_workers` is `0`, so the expensive online augmentation/STFT work runs synchronously in the training process unless the launcher overrides it.
  - `riei`, `drift`, `receiver_agnostic_rffi`, `cvcnn`, and TIFS2025 all call the same `run_validation_gated_training`, so final post-training test/satellite coverage can be fixed centrally.
  - `riei`, `drift`, and `cvcnn` pass satellite evaluation through `extra_test`; `receiver_agnostic_rffi` and TIFS2025 use raw loaders plus `input_transform=spec` for satellite-to-spectrogram evaluation. All are currently gated by validation improvement.
  - Historical run metrics confirm the final epoch was not tested:
    - `cvcnn_seed1337`: 200 epochs, 25 tested epochs, last epoch 200 `tested=False`, best epoch 187.
    - `drift_seed1337`: 200 epochs, 14 tested epochs, last epoch 200 `tested=False`, best epoch 121.
    - `riei_seed1337`: 200 epochs, 14 tested epochs, last epoch 200 `tested=False`, best epoch 36.
    - `tifs2025_seed1337`: 60 epochs due to early stop, 6 tested epochs, last epoch 60 `tested=False`, best epoch 30.
  - Historical `receiver_agnostic_seed1337` run directory is empty and the manifest points to a log file that is absent, so that comparison result was not produced in the inspected run.
  - No root-level `run_cvs_baseline_queue.sh` is present in the current workspace although `baselines/README.md` documents it.

## Star-ground baseline vs CVS-RFFI comparison - 2026-05-25
- User asked to design related comparison experiments and connect by SSH to implement.
- Prior local evidence showed baseline and CVS-RFFI use the same `sat_channel.py` simulator; likely causes are training objective strength, auxiliary-loss interference, MixStyle interaction, scenario coverage, and evaluation split aggregation.
- Remote host `N607` is reachable.
- Remote CV-SincNet workspace exists at `/home/szu2070436088/2510044040/CV-SincNet`.
- Remote script listing includes `run_b3b_asym_sat_baseline_8gpu.sh`, `run_cvs_rffi_rx_curriculum_bex02_6gpu.sh`, `run_sgc_ssdg_6gpu.sh`, and other relevant experiment launchers.

## Baseline Paper/Code Mapping - 2026-05-19
| Paper | Implementation | Paper expectation | Code review result |
|-------|----------------|-------------------|--------------------|
| `Toward_Channel-Robust_and_Receiver-Independent_Radio_Frequency_Fingerprint_Identification.pdf` | `baselines/tifs2025_channel_receiver_rffi` | Spectrogram representation, unsupervised NT-Xent pretraining, Siamese fine-tuning with contrastive + CE losses, online channel augmentation, single-branch inference. | Architecture/loss/stages match at a reproduction level. Main issue was efficiency: per-sample online augmentation and STFT in dataset `__getitem__`; fixed by raw dataset mode plus batched train/eval transform. |
| `Receiver-Agnostic_Radio_Frequency_Fingerprint_Identification_via_Feature_Disentanglement.pdf` | `baselines/riei` | FED split into emitter/receiver features, EC/RC classifiers, CE + MI + IE losses, alternating/intermediate updates, inference uses emitter classifier. | Code implements feature split, EC/RC/cross classifiers, MI/IE terms, and alternating update. It is an approximate reproduction using local ResNet1D and CVS split rather than the exact paper dataset protocol. |
| `2510.09405v1 (1).pdf` | `baselines/drift` | Cross-receiver DG with 1D ResNet-18 style encoder, transmitter/receiver feature split, GRL receiver discriminator, receiver center regularizer, negative MSE feature separation. | Code implements the stated model components and loss terms. Hyperparameters align with paper summary in README (`lr=1e-4`, batch 64, GRL/center/MSE weights). |
| `2207.02999v1.pdf` | `baselines/receiver_agnostic_rffi` | Spectrogram/CIS-style receiver-agnostic adversarial training with GRL, SGD momentum, validation-loss LR decay, optional fine-tuning and collaborative soft/adaptive fusion. | Code implements spectrogram training, GRL receiver classifier, validation-loss scheduling, collaborative evaluation for named tests, and now satellite evaluation in fine-tune too. CIS is represented by the local spectrogram transform rather than an exact CIS implementation. |
| N/A baseline control | `baselines/cvcnn` | Plain complex-valued CNN with CE only. | Code is a fair control path and shares the corrected final test/satellite behavior. |

## Implemented Fixes - 2026-05-19
- `baselines/common/cvs_trainer.py`
  - Added `evaluate_named_tests`.
  - Added final post-training test evaluation stored in `metrics.json["final"]`.
  - If the last epoch was not tested by the best-val gate, final evaluation now runs named test loaders plus `extra_test_fn`, which includes satellite-ground channel evaluation for all baseline trainers that pass it.
  - Preserves `metrics.json["best"]` and `best_by_val.pt` semantics; final test does not choose checkpoints.
- `baselines/tifs2025_channel_receiver_rffi/data.py`
  - `SpectrogramTransform` now preserves batch inputs as `[B,1,F,T]` while keeping single-sample output as `[1,F,T]`.
  - `PretrainDataset` and `SiamesePairDataset` support `return_raw=True` so expensive STFT can be performed in batched training/eval code instead of per sample in `__getitem__`.
- `baselines/tifs2025_channel_receiver_rffi/train_cvs.py`
  - Added `prepare_tifs2025_batch`.
  - Pretrain and Siamese training now request raw IQ samples and apply online channel augmentation + spectrogram conversion per batch.
  - Validation/test/satellite evaluation now uses raw loaders and converts to spectrogram inside `forward_eval`, avoiding duplicate transformed loader construction.
- `baselines/receiver_agnostic_rffi/finetune_cvs.py`
  - Added shared satellite evaluation args and final `extra_test_fn` support for the fine-tuning entry point.
- `baselines/common/__init__.py`
  - Adds project `code/` to `sys.path` when available so README-style `python -m baselines...` commands can import `dataset_wisig`, `training_controls`, and `sat_channel` from this workspace.
- `tests/test_baseline_training_behaviors.py`
  - Added regression tests for final post-training test/satellite behavior and TIFS2025 raw/batched transform behavior.

## Verification - 2026-05-19
- RED tests before implementation:
  - `conda activate ssr-gpu; python -m unittest tests.test_baseline_training_behaviors -q` failed because `metrics.json` lacked `final` and TIFS datasets lacked `return_raw`.
  - TIFS batch helper test then failed because `prepare_tifs2025_batch` was absent.
  - `python -m baselines.tifs2025_channel_receiver_rffi.train_cvs --help` and `python -m baselines.cvcnn.train --help` initially failed on `ModuleNotFoundError: dataset_wisig`.
- Passing verification after fixes:
  - `conda activate ssr-gpu; python -m unittest tests.test_baseline_training_behaviors -q` passed 3 tests.
  - `conda activate ssr-gpu; python -m py_compile baselines/common/__init__.py baselines/common/cvs_trainer.py baselines/tifs2025_channel_receiver_rffi/data.py baselines/tifs2025_channel_receiver_rffi/train_cvs.py baselines/receiver_agnostic_rffi/finetune_cvs.py tests/test_baseline_training_behaviors.py` passed.
  - `conda activate ssr-gpu; python -m baselines.tifs2025_channel_receiver_rffi.train_cvs --help`, `baselines.cvcnn.train --help`, `baselines.riei.train --help`, `baselines.drift.train --help`, `baselines.receiver_agnostic_rffi.train --help`, and `baselines.receiver_agnostic_rffi.finetune_cvs --help` all exited 0.
- Known unrelated/pre-existing verification issue:
  - `conda activate ssr-gpu; python -m unittest tests.test_baseline_training_behaviors tests.test_cvs_rffi_launcher -q` failed only in `tests.test_cvs_rffi_launcher`: expected launcher-missing message was absent from stdout. This test targets `scripts/run_cvs_rffi_staged_8gpu.sh`, not the baseline trainer changes above.

## Requirements
- Implement `C:/Users/lh594/Downloads/SGV_BP_FJMP_Ultimate_Design.md` without omitting items.
- Core output path must be `safe_logits = base_logits + rho(x) * [Calibrate(head_logits) - stopgrad(Calibrate(base_logits))]`.
- Final eval/checkpoint/deployment path must default to `safe_logits`, while still logging `base_logits` and `head_logits`.
- Add documented files:
  - `FJMP/star_ground_view.py`
  - `FJMP/sgv_bp_losses.py`
  - `FJMP/sgv_bp_metrics.py`
  - `FJMP/sgv_sampler.py`
  - `FJMP/logit_calibration.py`
  - `FJMP/base_protected_fusion.py`
- Update documented files:
  - `train_fjmp.py`
  - `FJMP/frozen_joint_prototype_head.py`
  - `FJMP/experiment_manifest.py`
  - `FJMP/summarize_experiments.py`
  - `scripts/run_fjmp_sgv_bp_8gpu.sh`
- Correctness checklist must cover frozen baseline, detached base logits, paired clean/sat batches, safe default eval, CE below safety/SGV consistency, no KD on hard wrong samples, reliability masking, stage-specific rho caps, calibrated delta, gate regularizers, stage-3 prototype LR decay, and checkpoint selection without real UDU.
- Metrics checklist must cover base/head/safe acc and margin, clean/sat separation, harm/rescue/net gain, rho/gate/delta strata, prototype entropy, clean-sat assignment KL, proxy score, and per-rx/per-day/per-view worst metrics.

## Research Findings
- Uploaded SGV-BP-FJMP document is UTF-8 Markdown at `C:/Users/lh594/Downloads/SGV_BP_FJMP_Ultimate_Design.md`.
- Workspace root is `E:/type10-7`.
- Workspace is not a Git repository.
- Previous `task_plan.md/findings.md/progress.md` described a different FJMP v2 harm exploration document and has been superseded for this request.
- Existing FJMP code already had frozen backbone feature extraction, prototype head, calibrated/residual fusion, harm/rescue diagnostics, and a large older FJMP v2 experiment manifest.
- Missing SGV-BP files from document section 31.1 were added.
- `train_fjmp.py` now accepts the document's recommended SGV-BP command arguments, including `--model_name SGV-BP-FJMP`, `--aggregation top2_mean`, `--fusion_mode base_protected_residual`, rho caps, SGV flags, safety/consistency loss weights, and proxy-safe selection.
- `train_fjmp.py` now wires `--use_sgv` into the training loop by generating paired sat views, forwarding the frozen baseline and FJMP head on clean/sat views, applying reliability masking, and adding SGV-BP scheduled losses.
- `FJMP/frozen_joint_prototype_head.py` now supports `top2_mean`, `trimmed_lse`, `mean`, and `max` aggregation and exposes documented aliases such as `head_logits` and `proto_assign`.
- `FJMP/experiment_manifest.py` now includes SGV-BP `EXP-00` through `EXP-16` while preserving the previous L0-L6 and priority batches.
- `scripts/run_fjmp_sgv_bp_8gpu.sh` can dry-run `EXP-04` and emits the documented main SGV-BP command.
- `FJMP/summarize_experiments.py` now reports `proxy_safe_score` and warning flags from metrics rows when present.

## Technical Decisions
| Decision | Rationale |
|----------|-----------|
| Treat document as approved design rather than asking for another spec approval. | The request explicitly says to implement modifications and optimizations from the document one by one. |
| Preserve local package structure under `FJMP/`. | The repository already moved FJMP implementation under a package with root-level shims. |
| Implement SGV-BP as additive functionality. | Existing FJMP v2 experiments and scripts should keep working. |

## Issues Encountered
| Issue | Resolution |
|-------|------------|
| Initial broad `git status` failed because no `.git` exists. | Continue with filesystem-level tracking and verification. |
| Current default Python has no PyTorch. | Could not run tensor-level unit tests here; used parser/unit tests that do not import torch plus syntax and launcher checks. |

## Resources
- `C:/Users/lh594/Downloads/SGV_BP_FJMP_Ultimate_Design.md`
- `E:/type10-7/FJMP/`
- `E:/type10-7/train_fjmp.py`
- `E:/type10-7/tests/`
- `E:/type10-7/scripts/run_fjmp_sgv_bp_8gpu.sh`

## Visual/Browser Findings
- None.

## SSDG Review Findings - 2026-05-18
- Current user request is a comprehensive review and explanation of SSDG-related code under `E:/type10-7/code`.
- Initial SSDG surface is `code/train_ssdg.py`, `code/scripts/run_sgc_ssdg_6gpu.sh`, WiSig dataset helpers, and reused functions from `code/train.py`.
- `E:/type10-7` is not a git repository, so the review uses filesystem inspection rather than git diff/PR context.
- `train_ssdg.py` implements a two-stage WiSig-only trainer: labeled supervised DG stage, then optional pseudo-label stage over a source-domain unlabeled split.
- SSDG reuses the dual CV-SincNet model and core DG losses from `train.py` / `post_stage_common.py` rather than defining a separate SSDG model package.
- Verification: `conda activate ssr-gpu; python -m pytest code/tests/test_post_stage_trainers.py -q` failed because `pytest` is not installed in `ssr-gpu`; `conda activate ssr-gpu; python -m unittest E:/type10-7/code/tests/test_post_stage_trainers.py` passed 16 tests.
- Review risks found so far: `--lambda_sat_cons` is parsed/logged but not used in the loss; best checkpoint selection uses source-val `tx_acc` only; `--dry_run` still builds data/model context before returning.

## SSDG Risk Fixes - 2026-05-18
- Added regression coverage in `code/tests/test_post_stage_trainers.py` for SSDG epoch scheduling, best metric scoring, dry-run behavior, and satellite consistency loss wiring.
- `train_ssdg.py` now treats `--epochs` as total epochs, preserving label epochs first and assigning the remainder to pseudo epochs.
- `train_ssdg.py` now returns from `--dry_run` before data/model construction.
- `train_ssdg.py` now implements `--lambda_sat_cons` as a KL consistency loss from clean logits to satellite logits and logs it separately.
- `train_ssdg.py` now supports `--best_metric` for `clean_val_tx`, `test_overall_tx`, `sat_mean_tx`, and `sat_worst_tx`, and uses it for best checkpoint updates.
- Verification after fixes: `conda activate ssr-gpu; python -m unittest E:/type10-7/code/tests/test_post_stage_trainers.py` passed 23 tests; `python -m py_compile` passed for touched files; direct `train_ssdg.py --dry_run --epochs 180 --lambda_sat_cons 0.04 --best_metric sat_worst_tx` exited 0 and skipped data/model construction.
## 2026-05-19 CVS-RFFI baseline-matched run

- User requested running CVS-RFFI with the same baseline comparison experiment settings, using strongest config/checkpoint `BEX02_fishr002_mixed_e170`.
- Desktop experiment notes define the post-core baseline matrix as SGC S1-S4, prototype-head P1-P3, and SSDG U0-U3, all dependent only on the selected stable baseline and runnable in a shared GPU queue.
- The receiver-curriculum baseline comparison launcher appears to be `code/scripts/run_baseline_pseudo_rx_curriculum_6gpu.sh`; docs describe SMOKE/CORE/FULL plans and existing logs are under `logs/baseline_supervised_rx_curriculum` and `logs/baseline_pseudo_rx_curriculum`.
- The requested comparison should mirror the receiver-curriculum matrix: target blocks `T1` and `T14`, pair IDs `P01`-`P06`, levels `K=2..7`, fixed train days `0,1`, fixed test days `2,3`, train ratio `0.2`, and all remaining receivers as test receivers.
- Local `runs/best_base_explore/BEX02_fishr002_mixed_e170` exists but is empty; the request is therefore interpreted as using the BEX02 training configuration, not loading a local checkpoint.
- No local `Dataset_WigSig/ManySig.pkl` was found under `E:/type10-7`, `E:/`, or `C:/Users/lh594`; a real launch will need `WISIG_PKL` pointed at the dataset location.
- New launcher `code/scripts/run_cvs_rffi_rx_curriculum_bex02_6gpu.sh` generated the full 72-job matrix successfully in dry-run mode. It writes to `runs/cvs_rffi_bex02_rx_curriculum` and `logs/cvs_rffi_bex02_rx_curriculum` by default.
- The actual launch is blocked before job start in this local workspace by missing `WISIG_PKL`; no GPU jobs or process IDs were created.

## 2026-05-25 CVS-RFFI star-ground augmentation comparison

## 2026-05-25 Fed-PVS-CPRFFI Final Design Integration

- User provided `C:/Users/lh594/Downloads/fed_pvs_cprffi_final_design.md` as the latest final design report for federated learning, domain generalization, few-shot adaptation, and collaborative multi-prototype heads.
- Project instructions still require local-first edits and verification before any N607 sync, plus a local experiment report for every N607 experiment launch.
- The current code surface includes `code/train.py`, `code/federated/fed_trainer.py`, `code/federated/fed_aggregate.py`, `code/federated/fedprox.py`, `code/sat_channel.py`, `code/concat_sat_channel_aug.py`, `code/FJMP`, `code/SGC`, `code/SSDG`, and `code/training_test_eval.py`.
- The new report's table of contents confirms the central architecture: RF StyleBank, style-anchored physical virtual domains, DG losses only on multi-style batches, ProtoBank, collaborative multi-prototype classifier evidence, base-anchor conservative fusion, few-shot new-domain/new-class adaptation, and staged federated training.
- `code/train.py` already exposes `--train_mode fedavg/fedprox`, `--fl_client_key receiver/receiver_day/receiver_channel/receiver_day_channel`, `--fl_local_objective ce/bex02_dg/receiver_agnostic_bex02`, FedProto stats, MixStyle, Fishr, same-TX/domain losses, satellite consistency, strict concat-sat augmentation, and satellite OOD evaluation.
- `code/federated/fed_trainer.py` currently keeps `global_proto_stats` as class sums/counts finalized into one class prototype per class, plus optional domain prototype summaries. This is a useful FedProto baseline but not the report's reliability-aware multi-prototype ProtoBank.
- `FederatedTrainer._compute_local_objective` is still single-main-view first, with optional baseline satellite concatenation or sat consistency. It does not yet construct report-style `x_local + x_style + x_phys + x_sat` multi-style batches with separate `d_style` labels.
- `DataAugmentation.py` and `sat_channel.py` already provide most of the physical transform substrate. The missing layer is conditioned sampling from remote style packets rather than random env IDs or fixed scenario configs.
- `model_dual_cvsincnet.py` already exposes `z_id`, `z_dom`, RCN stat enhancement, GRL heads, and MixStyle hooks on `time_down/t1`, matching the report's desired branch semantics and feature-style insertion points.
- `FJMP` and `SGC/v3` already contain multi-prototype/safe-fusion/harm-rescue diagnostics, but they are post-stage or adapter-style components. For Fed-PVS-CPRFFI they should be reused conceptually as inference-only evidence fusion around the federated base, not transplanted as a strong train-time replacement classifier.

- The correct experimental question is within CVS-RFFI: compare two satellite-ground augmentation methods, not CVS-RFFI versus a baseline trainer.
- Method A is current CVS-RFFI satellite auxiliary training: `--sat_cons_start_epoch 20 --lambda_sat_cls 0.10 --lambda_sat_cons 0.00`.
- Method B is the strong supervised satellite-view variant inside the same CVS-RFFI entry point: `--sat_cons_start_epoch 1 --lambda_sat_cls 1.00 --lambda_sat_cons 0.00`.
- The CORE matrix runs A/B on `mixed_orbit` and A/B on all five scenarios `clear_leo,low_elev_leo,rain_leo,storm_mp,mixed_orbit`.
- Strict evaluation is fixed to `--eval_sat_on test_unseen_day_unseen_rx` with all five satellite scenarios and `--sat_eval_max_batches -1`.
- Implementation followed the local-first rule: edited `E:/type10-7/code/scripts/run_cvs_rffi_sat_aug_compare_5gpu.sh`, synced it to `/home/szu2070436088/2510044040/CV-SincNet/scripts/run_cvs_rffi_sat_aug_compare_5gpu.sh`, then dry-ran and launched remotely.
- Remote `train.py` requires `--exp_group` to be one of its existing choices; the launcher now uses `s3_rxrobust_no_dac` and relies on `run_name=SAxx...` to identify this comparison.
- CORE queue is running from scheduler log `/home/szu2070436088/2510044040/CV-SincNet/logs/cvs_rffi_sat_aug_compare/scheduler_CORE_20260525_182115.log`.
- The `cv-sincnet` automation and `E:/type10-7/AGENTS.md` now explicitly require local edits first, then `scp` sync to N607, and prohibit remote-only edits.
- Strict baseline replication is implemented as `拼接星地信道增强` in `E:/type10-7/code/concat_sat_channel_aug.py`.
- `拼接星地信道增强` reproduces the baseline training semantics: it creates `x_sat` from the same satellite-channel simulator, returns `torch.cat([x_clean, x_sat], dim=0)`, and duplicates `y` plus raw domain labels to match the 2B batch.
- `train.py` exposes `--use_concat_sat_channel_aug`; this mode runs inside CVS-RFFI before the normal main/aux/DG losses, so all downstream supervised/domain/generalization losses see the concatenated 2B batch.
- When `--use_concat_sat_channel_aug` is enabled, the old auxiliary `use_sat_consistency` sat-CE/sat-cons branch is skipped to avoid double-counting satellite training.
- Remote strict queue launched under `/home/szu2070436088/2510044040/CV-SincNet/runs/cvs_rffi_concat_sat_compare` with logs in `/home/szu2070436088/2510044040/CV-SincNet/logs/cvs_rffi_concat_sat_compare`.

## 2026-05-25 Spaceborne FL-DG-FSL Prototype Synthesis

- User requested a broad, literature-grounded research synthesis for spaceborne RFFI combining federated learning, domain generalization, few-shot learning, and multi-prototype classification.
- Local plan `C:/Users/lh594/Downloads/fed_pvs_rffi_research_plan.md` frames the core issue as follows: in receiver-as-client federated RFFI, each client usually sees one real receiver domain, so centralized DG losses such as GRL, Fishr, same-TX consistency, MixStyle cross-domain pairing, and GroupDRO lose their local multi-domain assumptions.
- The same local plan proposes Fed-PVS-RFFI: RF StyleBank, style-anchored physical virtual domains, Identity ProtoBank, and fingerprint semantic preservation. This suggests DG should be activated only after local batches are rebuilt into multi-style batches.
- Local report `docs/federated_collaborative_prototype_fusion_report.md` argues that multi-prototype heads should be treated as conservative collaborative evidence, not as a strong standalone classifier. Client-wise prototypes should be reliability-weighted, mixture-preserved, and used for rescue/hard-sample calibration with harm monitoring.
- User clarified that previous materials are references and that related experiment results exist on the SSH server.
- Remote N607 inventory shows relevant result families under `/home/szu2070436088/2510044040/CV-SincNet`: `runs/fed_fewshot_dg`, `runs/fed_proto_smoke`, `runs/cvs_rffi_concat_sat_compare`, `runs/cvs_rffi_sat_aug_compare`, `runs/b3b_asym_sat_baseline`, `runs/sgc_ssdg_n04`, `runs/ssdg_gpu45_bex02`, `runs/sgc_v3_n04`, and target-adaptation logs.
- Preliminary integration thesis: FL is the system frame; StyleBank/physical virtual styles make DG valid under FL; few-shot adaptation is a later deployment-stage mechanism for new ground stations/satellite passes; multi-prototype evidence should stabilize identity semantics and support cautious inference-time fusion.
- N607 `runs/fed_fewshot_dg` final metrics show CE-only few-shot FL at 10% source labels reaches about 71% strict UDU for FedAvg/FedProx receiver-day clients; FedProx penalties are numerically tiny for local-epoch-2 runs and do not materially change accuracy.
- Directly moving centralized BEX02 DG into receiver-day FL (`FSDG18/FSDG19/FSDG1A/FSDG1B`) underperforms the CE-only FL controls, ending around 69.6-69.9% strict UDU. This supports the hypothesis that centralized DG losses should not be blindly placed inside single-domain clients.
- The strongest inspected FL result is `FSDG49_fedprox_receiver_ra_bex02_cvs_sat`: receiver-client FedProx + receiver-adversarial BEX02 + CVS satellite consistency, ending at 80.30% overall and 75.92% strict UDU. Its baseline-view satellite counterpart `FSDG50` drops to 70.52% strict UDU, suggesting satellite supervised expansion can hurt when it overwhelms identity learning in FL.
- N607 satellite augmentation logs show centralized CVS-RFFI/B3b strong DG reaches about 86-87% clean strict UDU, but satellite strict UDU remains around 40-46% for clear/low/rain/storm/mixed scenarios. Approximate strong sat-view supervision improves satellite strict UDU to about 48-50% on mixed training but reduces clean strict UDU to about 83-84%, revealing a clean-vs-satellite tradeoff.
- Strict concat-sat runs `SA02/SA04` failed because the launcher emitted mutually exclusive `--use_sat_consistency` and `--no_use_sat_consistency`; this is a pending experiment hygiene issue, not evidence against concat augmentation.
- FedProto smoke on N607 only validates plumbing: global prototype summary is enabled with nonzero class/domain prototype counts, but the run is too tiny to draw performance conclusions.
- External literature supports the synthesis direction: FedAvg/FedProx are the baseline FL optimizers; FedDG literature highlights privacy and lack of cross-domain interaction; CCST/FedGCA specifically address single-source clients through cross-client style/statistic augmentation; FedProto supports prototype communication/regularization under heterogeneity; RFFI receiver-agnostic papers support adversarial/disentangled receiver suppression and collaborative inference; few-shot literature supports metric/prototype heads but warns about domain shift.

## 2026-05-25 Fed-PVS-CPRFFI Strategy Loophole Audit

- The original integration strategy is structurally compatible with CVS-RFFI, but not safe to implement directly as one combined feature.
- Factually 100% confidence in final performance is impossible without controlled experiments; the only defensible 100% confidence boundary is that the revised protocol avoids known evidence/code overclaims.
- Local code confirms the main gap: `FederatedTrainer._compute_local_objective` is still single-main-view first and has no constructed `d_style` multi-style batch.
- Local code confirms current FedProto is a single class-mean mechanism (`class_sum/class_count -> class_proto`), not a reliability-aware multi-prototype ProtoBank.
- Local aggregation behavior must be revisited before adding local adapters/domain-private heads; otherwise new local-only parameters may be averaged by default.
- N607 metrics support the warning that direct BEX02-style DG inside receiver-day FL underperforms CE-only FL controls: direct DG runs end around 69.6-69.9 strict UDU versus CE/FedAvg/FedProx around 70.7-71.7 strict UDU.
- N607 metrics identify `FSDG49_fedprox_receiver_ra_bex02_cvs_sat` as the strongest inspected FL anchor: about 80.30 overall and 75.92 strict UDU.
- N607 evidence also shows satellite augmentation has a clean-vs-satellite tradeoff: stronger sat-view supervision improves satellite strict UDU but tends to reduce clean strict UDU.
- The strict concat-sat SA02/SA04 results remain invalid as method evidence because the launcher passed mutually exclusive arguments; this must be fixed before any concat-sat conclusion.
- Revised strategy V2 requires Phase -1 evidence hygiene, Phase 1 StyleBank diagnostics/no-op, Phase 2 style-conditioned augmentation without DG, Phase 3 `d_style`-only DG, Phase 4 eval-only conservative ProtoEvidenceBank fusion, and Phase 5 few-shot adaptation.
- Hard gates added: style-conditioned must beat random physical augmentation; clean strict drop must stay within a small bound; DG losses must see enough constructed style domains; prototype fusion must report rescue greater than harm; privacy/style leakage probes must pass.
- Additional confidence loopholes added after self-audit: key conclusions need multi-seed/statistical reporting, StyleBank/ProtoBank must not update from test-unseen data, and every N607 experiment must preserve local-to-remote version evidence through report/hash/sync mapping.

## 2026-05-26 Fed-PVS-CPRFFI Phase -1/Phase 1 Implementation

- Added `federated/style_packet.py` with `StylePacket` and `StyleDomainBatch`; this makes constructed `d_style` explicit instead of overloading raw receiver/day domains.
- Added `federated/rf_style_extractor.py`; it extracts class-marginalized IQ, amplitude, phase-difference, spectrum, and optional shallow-feature statistics without storing labels.
- Added `federated/style_bank.py`; it stores small server-side EMA/centroid style summaries, supports remote-client sampling, diagnostics, and byte-size accounting.
- Added `federated/virtual_domain_sampler.py` and `federated/conditioned_receiver_dg.py`; these provide the no-DG/Phase 2 substrate for explicit style-domain batch assembly and conservative style-conditioned IQ perturbation.
- Updated `FederatedTrainer` to collect optional no-op StyleBank diagnostics with `--use_fl_style_bank_stats`, log global style summaries, and accept an optional `style_batch_fn` whose `d_style` is passed to the model and DG losses while `d_raw` remains available for logging/eval.
- Updated `fed_aggregate.py` with `resolve_exclude_keys`, allowing exact-key and prefix-based local-only parameter exclusion before aggregation.
- Added `federated/proto_evidence_bank.py` and `federated/reliability_fusion.py`; these implement multi-prototype evidence retention and conservative probability-level fusion with harm/rescue reporting.
- Fixed the satellite comparison launcher hygiene issue by removing base `--use_sat_consistency` from concat-sat commands and stripping row-level `--no_use_sat_consistency` before execution.
- Added regression tests for StyleBank extraction/banking, virtual `d_style` batches, style-conditioned perturbation, ProtoEvidenceBank, conservative fusion, federated `d_style` plumbing, aggregation exclusion, CLI exposure, and concat-sat launcher dry-runs.
- Verification: local `conda run -n ssr-gpu` py_compile passed; local unittest passed 12 tests with 2 bash/WSL launcher checks skipped; N607 `/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python -m unittest ... -v` passed all 14 tests.

## 2026-05-26 Post-Implementation Loophole Loop

- Fresh audit found `d_style` labels should not use raw StyleBank `style_id` directly, because a remote style id can exceed the model domain-head dimension. `VirtualDomainSampler` now emits compact constructed labels (`clean=0`, virtual views `1..K`) and stores original style ids in metadata.
- Fresh audit found domain CE losses need a guard when constructed domains exceed the available domain-head output dimension. `FederatedTrainer` now skips CE-style domain losses in that case instead of crashing or training on invalid labels.
- Fresh audit found StyleBank vectors need a stable numeric schema across packets with different optional stats. `FederatedStyleBank` now maintains a stat-key schema and re-encodes centroids when new numeric keys appear.
- Fresh audit found StyleBank trimming should prefer higher-count/newer centroids rather than accidentally preserving old ones. Trim ordering was changed and tested.
- Fresh audit found a local launcher test could hang under Windows bash/WSL behavior. Subprocess timeouts now turn that into a local skip; N607 still runs the bash assertions normally.
- Verification after fixes: local `conda run -n ssr-gpu python -m py_compile ...` passed; local unittest for the six relevant modules ran 16 tests with 15 passed and 1 local bash/WSL skip; N607 py_compile plus the same six-module unittest suite passed 16/16.

## 2026-05-26 FL82 federated validation experiment

- User requested an SSH-run experiment to validate FL effectiveness and pursue clean strict `test_unseen_day_unseen_rx` accuracy >=82%, with clean tests every federated round and satellite-channel testing included.
- Added local launcher `E:/type10-7/code/scripts/run_fed_fl82_validation_4gpu.sh`; it defines CORE runs `FL82_01` FedAvg CE, `FL82_02` FedProx CE, `FL82_03` FedProx receiver-agnostic BEX02 CVS, and `FL82_04` local3 StyleBank diagnostics.
- Initial FL82 launcher defaults were train ratio `0.2` and rounds `220`; after user correction, formal FL82 launches now default to train ratio `0.1` and epochs/rounds `200`.
- Local report created at `E:/type10-7/automation_reports/CV-SincNet/20260526_004220_fl82_fed_validation/report.md` with objective, hypothesis, config, sync mapping, hashes, command, logs, PIDs, risks, and startup metrics.
- N607 first launch exposed missing `baseline_origin_sat_view.py`; second launch exposed missing `cvsrffi`; third launch exposed unsupported `--exp_desc`. All were fixed locally first, snapshotted/synced, and recorded in the report.
- Active N607 queue is running from scheduler `/home/szu2070436088/2510044040/CV-SincNet/logs/fl82_fed_validation/scheduler_CORE_20260526_005707.log` with logs under `/home/szu2070436088/2510044040/CV-SincNet/logs/fl82_fed_validation`.
- Startup verification confirmed all four runs reached training and printed per-round clean `[FED-TEST-SPLIT]` plus five-scenario `[FED-SAT-TEST]` metrics. Early round-1 strict UDU values are warmup-only: about 20.29%, 20.36%, 19.41%, and 27.92% for FL82_01..04 respectively.
- Updated the `cv-sincnet` hourly automation with the explicit FL82 goal, success threshold, monitoring instructions, report path, and local-first sync rules.
- Added `[SAT-TEST-SPLIT]` logging for named satellite eval splits and changed new satellite evals to `test_unseen_day_seen_rx,test_seen_day_unseen_rx,test_unseen_day_unseen_rx`.
- Added `SAT_BASELINE` runs `FL82_07`..`FL82_09` using baseline-style clean+sat supervised view expansion (`--fl_sat_aug_mode baseline_view --sat_view_prob 1.0 --sat_cons_start_epoch 1`) to target the clear_leo floors 84.30/60.10/53.78.
- `SAT_BASELINE` launched on N607 GPUs 0,1,2 from scheduler `/home/szu2070436088/2510044040/CV-SincNet/logs/fl82_fed_validation/scheduler_SAT_BASELINE_20260526_014110.log`; all three jobs reached round 1 and printed the requested clear_leo split metrics.
- Round-1 SAT_BASELINE metrics are warmup-only: `FL82_07` clean strict UDU 16.71 and clear_leo 17.24/17.10/17.15; `FL82_08` clean strict UDU 23.43 and clear_leo 27.77/28.75/27.43; `FL82_09` clean strict UDU 21.86 and clear_leo 27.27/28.07/26.73.
- Latest quick check reached round 2 without filtered Traceback/OOM/NaN: `FL82_07` clean strict UDU 31.16 and clear_leo 30.70/31.11/30.06; `FL82_08` clean strict UDU 32.16 and clear_leo 32.66/31.94/31.63; `FL82_09` clean strict UDU 32.87 and clear_leo 33.09/31.97/31.77. Still early warmup, below target.
- Latest CORE status remains below target but improving: `FL82_01` R035 strict UDU 67.31, `FL82_02` R033 strict UDU 67.17, `FL82_03` R033 strict UDU 73.75, and `FL82_04` R030 strict UDU 74.06.
- Updated the `cv-sincnet` automation again so the durable targets are clean strict UDU >=82 plus clear_leo UDSR>=84.30, SDUR>=60.10, and UDUR>=53.78, with active-job monitor-only behavior.
- User corrected a hard constraint: formal federated training must use train ratio `0.1`, not `0.2`, and default epoch/round count should be `200`.
- Updated `run_fed_fl82_validation_4gpu.sh` so future launches default to `FEWSHOT_RATIO=0.1`, `EPOCHS=200`, and `FL_ROUNDS=200`; future FL82 run names now use `r010`.
- Active `0.2/220` FL82 runs are now historical/debug-only and must not be counted as formal evidence for the current target, though they remain monitor-only unless the user explicitly asks to stop/restart them.
- Local and remote checks passed for the correction: local py_compile and targeted unittest passed; remote SHA matched, `bash -n` passed, targeted unittest passed, and remote SAT_BASELINE dry-run confirmed `--wisig_train_ratio 0.1 --epochs 200 --fl_rounds 200`.

## 2026-05-26 Federated log forensics: initial inventory

- Source backup for this analysis is `E:/type10-7/server_log_backups/N607/20260526_101853/exhaustive_log_files`; it contains both `CV-SincNet/logs/...` text logs and `CV-SincNet/runs/...` metric artifacts.
- `CV-SincNet/logs/fl82_fed_validation` contains seven main training logs: `FL82_01`..`FL82_04` CORE and `FL82_07`..`FL82_09` SAT_BASELINE, plus scheduler/queue/launcher files from failed and successful attempts.
- `FL82_02` example confirms current logs include `[SAT-EVAL]`, `[FED] dispatch`, per-round `[FED]`, `[FED-TEST]`, `[FED-TEST-SPLIT]`, `[FED-SAT-TEST]`, and `[FED-DIAG]` entries, enough to recover train mode, client key, local objective, rounds/local epochs, FedProx mu, strict UDU curves, and class prediction histograms.
- `fl82_fed_validation` logs in the backup had progressed far beyond the earlier startup status: SAT_BASELINE `FL82_09` had reached at least round 117 in the backed-up log, so this analysis should parse full available curves rather than relying on earlier progress notes.
- The backup also contains older `CV-SincNet/logs/fed_fewshot_dg` and `CV-SincNet/runs/fed_fewshot_dg` artifacts, including FSDG12-19, FSDG40-43, FSDG49-50, and FedProto smoke results, useful as FL historical anchors.

## 2026-05-26 FL82 DG and satellite findings

- `FL82_01/02` receiver-day CE baselines plateau near `73%` strict UDU; FedProx adds almost nothing over FedAvg.
- `FL82_03/04` receiver-client receiver-agnostic BEX02 improves clean strict UDU to `78.58/79.04%`, but both degrade late, with `FL82_04` collapsing to `16.67%` final strict UDU.
- The centralized anchor `BEX02_fishr002_mixed_e170` reaches `85.97%` strict UDU with real centralized `rx_day` mixed-domain batches, so FL82 is missing about `6.9` clean strict points at the same logged `0.2` ratio.
- Parsed FL82 client losses show `loss_fishr`, `loss_dom`, and `loss_cons` as `0.0`; the likely cause is that local FL clients do not provide enough simultaneous domains for these centralized DG losses to activate. Future logs now include explicit activation diagnostics to confirm this per round.
- `baseline_view all5` satellite augmentation improves satellite strict UDU versus CVS consistency, especially in `FL82_08`, but it trades off clean strict UDU. `baseline_view clearleo` maximizes clear-leo but hurts clean DG and does not generalize well to storm/mixed.
- Highest-ROI next direction is not more FedProx tuning; it is constructing verified multi-style/multi-domain local batches, then enabling Fishr/domain/consistency only when the new diagnostics prove they are active.
## 2026-05-26 StyleBank/ProtoBank effectiveness

- StyleBank has evidence of diagnostic activity in FL82 logs: style packets and 64 centroids are recorded, but `client_style_num_domains_avg` and `client_style_batch_views_avg` are missing/NaN in inspected runs. This means no normal training-time multi-style batch reconstruction occurred.
- ProtoBank design modules exist (`proto_evidence_bank.py`, `reliability_fusion.py`) but are not wired into real federated training/evaluation. Current active prototype training path is FedProto-style class-mean pull loss behind `--use_fed_proto_stats`.
- FL82 stylebank performance should not be interpreted as StyleBank benefit because the decisive StyleBank mechanism is inactive and differences are confounded by local epochs/baseline-view/satellite settings.

## 2026-05-26 Design-parity implementation findings

- The implementation now treats star-ground augmentation, StyleBank, and GRL as complementary: satellite views inject target channel distribution; StyleBank injects cross-client nuisance diversity; GRL/receiver-adversarial loss removes receiver/style shortcuts from identity features.
- Federated training now defaults to StyleBank being enabled. Remote centroids are sampled after the bank has matured, transformed through `StyleConditionedReceiverDG`, and appended as explicit `d_style` views whose labels are the target receiver/domain labels used by `dom_head(z_dom)` and GRL/`adv_head(z_id)`.
- DG/GRL losses are gated by StyleBank maturity and `fl_style_dg_min_domains`, so direct single-domain-client DG no longer pretends to be cross-domain evidence.
- ProtoEvidenceBank now collects real class evidence from client `z_id`/logits and reports conservative fusion harm/rescue diagnostics, while FedProto remains separately named as the single-class-mean baseline.
- Important implementation correction: StyleBank centroids must retain target-domain metadata and must not merge across different target domains. Without this, generated remote views could carry ambiguous or wrong `d_style`, breaking the intended `z_dom` classification and GRL signal.
- Important experiment caveat: style-conditioned IQ transforms are still pseudo-domain generation. They should be validated by `zdom_target_acc`, `style_domain_entropy`, `style_dg_ready`, `grl_target_acc`, clean strict UDU, and satellite split metrics before claiming the generated styles are physically faithful target domains.
- Future federated runs now have a machine-readable config anchor at `federated_config.json` plus a `logs.jsonl` `fed_config` event. This should eliminate ambiguity about train ratio, client split key, StyleBank gate timing, target-domain label semantics, ProtoBank fusion settings, GRL/adversarial weights, and satellite augmentation/evaluation scenarios.
- The startup stdout now includes `[FED-CONFIG-GRL]` with the explicit role of GRL: preventing receiver/channel shortcut learning so `zid` keeps transmitter-stable fingerprints while `zdom` learns the current target domain/style.
- Federated training now has a hard satellite-eval invariant: every round must include satellite-channel testing. Disabling `eval_sat_channel` for `fedavg`/`fedprox` raises immediately, and the default FL satellite eval target is the three main DG splits rather than only strict UDU.
- For future log forensics, use `[FED-SAT-TEST][Rxxx]` and `[SAT-TEST-SPLIT]` together with `federated_config.json:satellite.per_round_satellite_eval=true` as the proof that per-round star-ground robustness was measured.

## 2026-05-26 Reusable workflow packaging audit

- User requested a 30-day workflow packaging audit from image instructions, prioritizing recent Codex sessions, Codex memories/rollout summaries, Chronicle if enabled, and existing skills/agents/automations.
- Local conversation index was refreshed with `conda activate ssr-gpu; python tools/conversation_index.py build`; it indexed 99 E:\type10-7 conversation entries.
- Existing durable automation found: `C:\Users\lh594\.codex\automations\cv-sincnet\automation.toml`, active hourly local cron for N607 CV-SincNet/CVS-RFFI monitoring.
- Existing custom project skill referenced by memory, `skills/cv-sincnet-n607-automation/SKILL.md`, was not present under `C:\Users\lh594\.codex\skills`; this made the N607 workflow a high-confidence missing skill rather than a duplicate.
- Candidate: N607/CV-SincNet experiment design-sync-launch-monitor-report workflow.
  - Evidence dates: 2026-05-22 through 2026-05-26 in memory, automation memory, and conversation index; repeated SSH gating, monitor-only decisions, local-first edits, scp sync, snapshots/manifests, startup health checks, and reports.
  - Frequency/confidence: high/high.
  - Recommended form: create missing skill and extend existing automation.
  - Decision: created `C:\Users\lh594\.codex\skills\cv-sincnet-n607-automation\SKILL.md`; updated `cv-sincnet` automation prompt with conversation-index lookup and formal `--fl_client_key receiver` default.
- Candidate: federated log forensics and FL82/StyleBank/ProtoBank diagnosis.
  - Evidence dates: 2026-05-25 through 2026-05-26; repeated parsing of N607 logs, metric CSVs, satellite split metrics, and design-vs-implementation audits.
  - Frequency/confidence: medium/high.
  - Recommended form: skip separate asset for now; keep inside the N607 skill/report workflow because it depends on the same experiment records and parser outputs.
- Candidate: BEX02 target-domain adaptation launcher suite.
  - Evidence dates: 2026-05-20 through 2026-05-21; repeated clean/provided-satellite/RXxTX-balanced launcher creation and loader safeguards.
  - Frequency/confidence: medium/medium.
  - Recommended form: extend existing scripts/docs only when the next adaptation request arrives; not enough current need for a new skill.
- Candidate: baseline server-safe packaging and satellite-view augmentation.
  - Evidence dates: 2026-05-19 through 2026-05-21; repeated server startup/import fixes, baseline queue packaging, and LEO augmentation integration.
  - Frequency/confidence: medium/medium.
  - Recommended form: skip new asset; existing baseline README/tests and code fixes already cover the core workflow.
- Candidate: CVS-RFFI behavior-preserving refactor/equivalence checks.
  - Evidence dates: 2026-05-25 plus related method-semantics threads.
  - Frequency/confidence: low-medium/medium.
  - Recommended form: skip; current general code-review/refactor skills plus project memory are sufficient unless another large refactor appears.
- Candidate: Codex skill installation/name lookup.
  - Evidence dates: 2026-05-20 and 2026-05-25.
  - Frequency/confidence: medium/high.
  - Recommended form: skip; existing `skill-installer` and installed skill metadata already cover it.

## 2026-05-26 Reusable workflow packaging audit - omission reflection update

- User added a correction: when asked to act according to design reports, I often omit important parts. Treat this as actionable workflow feedback, not as a conversational aside.
- Evidence supports the pattern:
  - 2026-05-13/14: multiple requests to implement design documents item-by-item, including `frozen_joint_multi_prototype_head_design_formula_fixed.md`, `SGC_Standalone_TargetAdapt_Design_v2.md`, and `CVS_RFFI_experiment_validation_design.md`.
  - 2026-05-15/16: `SGV_BP_FJMP_Ultimate_Design.md` and FJMP v2/v3 work required exact design translation plus follow-up redesign when safe residual behavior was too conservative.
  - 2026-05-18: `PhyCon-CxRCM-SGC_设计落地报告.md` explicitly requested "逐项实现...不遗漏".
  - 2026-05-25/26: `fed_pvs_cprffi_final_design.md` first produced integration analysis, then a gap audit found StyleBank was diagnostic-only and ProtoBank inactive; a later design-parity implementation was needed.
- Candidate: design-report implementation traceability / omission guard.
  - Frequency/confidence: high/high.
  - Recommended form: new skill.
  - Why worth creating: this failure mode is costly, repeated, and has a stable workflow: extract requirements, map to files/tests, implement by ID, reverse-audit against the source report, and state strict parity vs approximation.
  - Decision: created `C:\Users\lh594\.codex\skills\design-report-traceability\SKILL.md`.
- Root-cause reflection:
  - I tended to summarize design reports into an implementation plan too early, which compresses away edge requirements.
  - I sometimes treated module existence or CLI exposure as implementation, even when the path was not reachable in real train/eval flow.
  - I sometimes verified syntax/tests without proving each design item had runtime evidence.
  - I sometimes gave final summaries at the feature level rather than a source-section traceability table, making omissions harder to see.
- New operating rule: for future "按文档/设计报告/逐项落实/不遗漏" tasks, use `design-report-traceability` first and maintain a traceability table before claiming completion.

## 2026-06-21 CVS automation trim findings

- Main deficiency: the automation had too many places where a broad PASS-like word could be over-read. The most important correction is that matrix `verdict=PASS` now remains only schema/protocol validation; lane runner decisions must inspect `launchability_summary.by_lane`.
- Route duplication is a real launch blocker, not a reviewer preference. Launchable rows carrying route-duplication/local-hook repair markers now fail validation.
- Operational state was too heavy for decision making. `stage2_optimizer_state.json` remains the evidence store, but `tools/optimizer_state_current_view.py` now provides the thin current-decision surface and explicitly marks changelogs/lane subtrees audit-only.
- Protocol drift was present: control files and state allowed multiple target receivers in one Stage2 run, while `项目.md` requires a single deployment receiver `r_sat`. The default is now single `rx7`; `rx8-rx11` are sensitivity receivers only.
- Deleted/degraded from decision surface: historical lane mirrors, old changelog text, launcher-parent context counts, clean-view success, Phase1 protocol-only PASS, and retired/duplicate route signatures. These remain evidence or diagnostics, not launch gates or success claims.
- Remaining useful next work: centralize `PHASE2_LOCAL_PATCH_REQUIRED` defer behavior in `stage2_queue_runner_template.sh`, add golden tests for `update_monitor_optimizer_closed_loop_prompt.py`, and regenerate a non-duplicate single-`r_sat` Phase2 matrix before clearing the Phase2 local-patch blocker.
