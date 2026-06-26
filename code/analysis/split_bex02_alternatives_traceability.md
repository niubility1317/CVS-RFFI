# Split-BEX02 Alternatives Traceability

Status values: `pending`, `implemented`, `local_verified`, `synced`, `n607_started`, `n607_evidenced`, `complete`, `blocked`.

| ID | Scheme | Status | Local files / knobs | Required evidence | Notes |
|---|---|---:|---|---|---|
| SBX-01 | Local Virtual-BEX02 | local_verified | `train.py`, `federated/fed_trainer.py`, `federated/virtual_domain_sampler.py`; `--fl_local_objective local_virtual_bex02`, StyleBank d_style knobs | Logs: `diag_domain_count`, `diag_fishr_domain_count`, `diag_rx_adv_active`, `diag_baseline_sat_view_active`, `global_style_summary` | Approximation only, not centralized equivalence. VMB/Split client paths now upload StylePackets, verified by smoke tests and launcher dry-run. |
| SBX-02 | Prototype-BEX02 | local_verified | `federated/fedcvs_vmb.py`, `federated/fed_trainer.py`; VMB TX/RX proto and FedProto knobs | Logs: `global_vmb_proto_summary`, proto losses, payload bytes | Must show prototypes are used in loss, not only collected. Existing VMB/proto tests plus launcher row verified locally. |
| SBX-03 | Decomposed-Gradient / Gradient-Stat | local_verified | `federated/gradient_stats.py`, `federated/fed_trainer.py`; `--fl_conflict_agg` | Logs: conflict counts, gradient cosine before/after, payload | No raw IQ; full gradients are still gradients, not private statistics. Unit test covers conflict detection and cosine-clip resolution. |
| SBX-04 | Style-Code Bank | local_verified | `federated/style_packet.py`, StyleBank modules; `--fl_style_code_dim` | Logs: style centroids, style bytes, style code dim, d_style domains | Style codes are compact metadata, not learned client-private embeddings. Unit and VMB/Split smoke tests cover fixed-dim style code serialization plus packet upload. |
| SBX-05 | Logit / Distillation Anchors | local_verified | `federated/distill_anchors.py`, `federated/fed_trainer.py`; KD knobs | Logs: `kd_loss`, `kd_active`, `anchor_count`, `teacher_correct_rate`, payload | KD must be confidence/margin gated. Unit and smoke tests cover gated anchors and trainer logging. |
| SBX-06 | Sketch / Low-Rank / Quantized Activation Tokens | local_verified | `federated/activation_tokens.py`, trainer; `--train_mode split_bex02`, token knobs | Logs: token route, shape, bits/rank, bytes, compression ratio, quant error | Compressed Split-BEX02 approximation and online feature-probe export; strict split learning still requires feature-gradient return proof. Unit and smoke tests cover quantized route, accounting, and export. |
| SBX-07 | Satellite CE-Only Integration | local_verified | `train.py`, `federated/fed_trainer.py`; `--fl_baseline_view_ce_only` | Logs: baseline CE-only active, `diag_sat_cons_active=0` when expected | Existing path stays isolated from DG losses when CE-only; launcher row and existing satellite tests remain green. |
| SBX-08 | Stage1 Strengthening | local_verified | `train.py`, `federated/fed_trainer.py`; Stage1 steps, objective, and LR multiplier | Logs: stage1 steps, stage1 objective, state averaging, transition to stage2 | Strengthens pretrain through `--fl_vmb_stage1_local_steps`, `--fl_vmb_stage1_objective ce`, and `--fl_vmb_stage1_lr_mult`; Stage1 stays CE-only for loss while still uploading StylePackets for later VMB. |
| SBX-09 | Stage2 Conflict-Aware Aggregation | local_verified | `federated/gradient_stats.py`, `federated/fed_trainer.py`; `--fl_conflict_agg` | Logs: conflicts detected/resolved, cosine before/after | `none` preserves old behavior; `cosine_clip` covered by utility and split-route smoke tests. |
| SBX-10 | Feature / Probe Diagnostics | local_verified | `evaluation/fedcvs_vmb_probe.py`, trainer config; probe knobs | Logs/artifacts: `global_feature_probe_summary`, `feature_probe_samples`, exported `.pt` features, offline four-probe script | Existing offline probes remain; trainer now exports `z_t/z_r/tx/rx` feature payloads when `--fl_probe_every` and `--feature_probe_export` are set. |
| SBX-11 | Report / Update Workflow | local_verified | `scripts/launch_split_bex02_alternatives_8gpu.sh`, `SYNC_MANIFEST.txt`, automation report | Local verification, snapshot, sync manifest, N607 proof | Local report and dry-run are updated. N607 sync/launch is deliberately pending. |

## Experiment Matrix

| GPU | Run | Main mechanism |
|---:|---|---|
| 0 | `SBX02_LVMB_r010` | Local Virtual-BEX02 / VMB approximation |
| 1 | `SBX02_PROTO_r010` | Proto-BEX02 / VMB prototypes + FedProto |
| 2 | `SBX02_FISHR_r010` | Decomposed-gradient/Fishr and conflict stats |
| 3 | `SBX02_STYLE_r010` | Style-code Bank |
| 4 | `SBX02_KDLOGIT_r010` | Logit/distillation anchors |
| 5 | `SBX02_QTOKEN_r010` | Quantized/sketched activation tokens |
| 6 | `SBX02_SATCE_r010` | Satellite CE-only branch |
| 7 | `SBX02_COMBO_r010` | Gated combination of individually passing mechanisms |

## Local Verification Evidence

- `conda activate ssr-gpu; python -m py_compile train.py federated/fed_trainer.py federated/fedcvs_vmb.py federated/gradient_stats.py federated/distill_anchors.py federated/activation_tokens.py model_dual_cvsincnet.py` -> exit 0.
- `bash -n scripts/launch_split_bex02_alternatives_8gpu.sh` -> exit 0.
- `conda activate ssr-gpu; python -m pytest tests/test_split_bex02_alternatives.py tests/test_fedcvs_vmb.py tests/test_federated_trainer_smoke.py tests/test_federated_train_integration.py tests/test_federated_aggregation.py tests/test_fed_pvs_style_bank.py tests/test_federated_d_style_plumbing.py -q` -> `61 passed, 1 skipped in 36.00s`.
- `bash -lc 'cd /mnt/e/type10-7/code && DRY_RUN=1 ROOT=/tmp/cv_sincnet RUN_ROOT=/tmp/split_bex02_launcher_dryrun PYTHON=python WISIG_PKL=/tmp/cv_sincnet/Dataset_WigSig/ManySig.pkl scripts/launch_split_bex02_alternatives_8gpu.sh'` -> exit 0, 8 run commands, one `--fl_local_objective` per command.

## Review Fixes Landed

- VMB Stage1 and Split Stage2 client paths now upload StylePackets, so `SBX02_LVMB`, `SBX02_STYLE`, and `SBX02_COMBO` no longer rely on non-VMB-only StyleBank upload behavior.
- `local_virtual_bex02` is included in the federated augmentation/SAT transform objective set, so COMBO can activate baseline-view CE-only satellite supervision.
- `--fl_vmb_stage1_objective ce` makes Stage1 loss behavior match the CE-only warmup design.
- `--no_use_logit_anchors` now disables logit-anchor upload/KD even if `--lambda_logit_kd` is positive; KD rows must opt in with `--use_logit_anchors`.
- Conflict-aware aggregation now preserves non-common gradient keys by zero-filling missing client gradients and reports `missing_gradient_entries`.
- Metrics CSV now includes `vmb_missing_gradient_entries` to make zero-filled gradient keys auditable.
- `apply_fedcvs_vmb_defaults()` now respects both `--flag value` and `--flag=value` explicit overrides.
- Launcher `QTOKEN` and `COMBO` rows now explicitly align the same VMB base hyperparameters as the other rows, and PID writing records the nohup training command PID from inside the target working directory subshell.
- Launcher non-dry-run mode validates `${ROOT}/train.py` before starting jobs.
- Launcher now defaults to `ENFORCE_ONE_RUN_PER_GPU=1` and refuses to start a row when the target GPU already has a compute process, preserving the strict one-experiment-per-card requirement.

## Remaining Gate

- No N607 sync or full server launch has been started yet.
- N607 check on 2026-05-28 12:15 CST found all eight GPUs already occupied by SA43-SA50, so SBX02 launch remains pending to avoid stacking experiments.
- Before sync, create a snapshot under `E:\type10-7\code\snapshots\<timestamp-or-run-id>\` because the workspace is not a git repository.
- After sync, update `E:\type10-7\automation_reports\CV-SincNet\20260528_split_bex02_alternatives\report.md` with remote file mappings, command lines, PIDs, log paths, and first health check evidence.
