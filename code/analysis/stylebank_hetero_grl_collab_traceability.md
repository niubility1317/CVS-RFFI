# StyleBank Heterogeneous GRL Collaborative Traceability

| ID | Source section | Requirement | Target files | Status | Verification | Notes |
|----|----------------|-------------|--------------|--------|--------------|-------|
| R1 | Design req 1 | StyleBank remains opt-in and non-StyleBank FL behavior is unchanged. | `train.py`, `federated/fed_trainer.py`, tests | verified | `python -m pytest tests -q`; `train.py --help` | Existing opt-in defaults preserved; collaborative eval is also opt-in. |
| R2 | Design req 2 | Local StyleBank batches represent clean plus remote receiver-style views. | `federated/fed_trainer.py`, `tests/test_federated_d_style_plumbing.py` | verified | `python -m pytest tests/test_federated_d_style_plumbing.py -q` | Default StyleBank batch builds clean + remote views; satellite remains separate by default. |
| R3 | Design req 3 | Constructed style labels are explicit and usable by GRL/Fishr as `d_style`. | `federated/fed_trainer.py`, `tests/test_federated_d_style_plumbing.py` | verified | `python -m pytest tests/test_federated_d_style_plumbing.py -q` | `d_style` is now contiguous virtual labels `0..K`; raw target labels stay in `d_raw` and metadata. |
| R4 | Design req 4 | Receiver-agnostic GRL avoids double-counting one adversarial head. | `federated/fed_trainer.py`, existing tests | verified | `python -m pytest tests/test_federated_trainer_smoke.py tests/test_federated_d_style_plumbing.py -q` | Previous `rx_uses_adv_head` guard remains intact. |
| R5 | Design req 5 | Fishr activates only with enough constructed style domains. | `federated/fed_trainer.py`, tests | verified | `python -m pytest tests/test_federated_d_style_plumbing.py tests/test_federated_trainer_smoke.py -q` | `fishr_min_domains` still gates `_fishr_logit_gradient_variance_loss`. |
| R6 | Design req 6 | Virtual collaborative inference reports base/fused accuracy, rescue, harm, and net gain. | `federated/reliability_fusion.py`, `federated/fed_trainer.py`, tests | verified | `python -m pytest tests/test_fed_pvs_proto_fusion.py -q` | `global_style_collab_fusion` is produced from per-round eval. |
| R7 | Design req 7 | Collaborative fusion supports soft mean and adaptive weighting. | `federated/reliability_fusion.py`, tests | verified | `python -m pytest tests/test_fed_pvs_proto_fusion.py -q` | `soft`, `adaptive`, and guarded `conservative` modes are shape-checked. |
| R8 | Design req 8 | CLI, config snapshot, logs, and metrics expose collaborative eval switches and outputs. | `train.py`, `federated/fed_trainer.py`, tests | verified | `python train.py --help`; `python -m pytest tests/test_federated_train_integration.py -q` | Config prints `[FED-CONFIG-STYLE-COLLAB]`; metrics include rescue/harm/net/base/fused. |
| R9 | Design req 9 | FL82 launcher/docs include paper-inspired StyleBank heterogeneous collaborative variant and preserve formal defaults. | `scripts/run_fed_fl82_validation_4gpu.sh`, docs, tests | verified | `bash -n scripts/run_fed_fl82_validation_4gpu.sh`; `bash scripts/run_fed_fl82_validation_4gpu.sh --plan SAT_BASELINE --dry-run` | Added `FL82_10...stylebank_collab...`; dry-run shows ratio 0.1, epochs 200, FL rounds 200, receiver clients. |
| R10 | Design req 10 | Tests prove fusion math, parser/config reachability, trainer eval reachability, and training semantics. | `tests/*` | verified | `conda activate ssr-gpu; python -m pytest tests -q` | Full local suite passed: 96 passed, 1 skipped, 1 warning, 11 subtests passed. |

## Omission Traps

- Report-only implementation without calling collaborative eval from `_evaluate`.
- CLI flag exposed but not included in config snapshot or logs.
- StyleBank batch labels still tied to raw receiver IDs, leaving virtual domains sparse and unstable.
- Fusion utility implemented but not shape-safe.
- Launcher variant added without formal FL82 defaults.
- Claiming target accuracy before an N607 formal run is completed and fully parsed.

## Verification Summary

- `conda activate ssr-gpu; python -m pytest tests/test_fed_pvs_proto_fusion.py tests/test_federated_d_style_plumbing.py tests/test_federated_train_integration.py -q` -> `19 passed, 1 skipped`
- `conda activate ssr-gpu; python -m pytest tests/test_federated_trainer_smoke.py tests/test_fed_pvs_style_bank.py -q` -> `18 passed`
- `conda activate ssr-gpu; python -m py_compile train.py federated\fed_trainer.py federated\reliability_fusion.py federated\__init__.py tests\test_fed_pvs_proto_fusion.py tests\test_federated_d_style_plumbing.py tests\test_federated_train_integration.py` -> passed
- `conda activate ssr-gpu; python train.py --help` -> passed and exposes `--use_style_collab_eval`
- `bash -n scripts/run_fed_fl82_validation_4gpu.sh` -> passed
- `bash scripts/run_fed_fl82_validation_4gpu.sh --plan SAT_BASELINE --dry-run` -> passed and includes `FL82_10_fedprox_rx_ra_bex02_stylebank_collab_all5_r010`
- `conda activate ssr-gpu; python -m pytest tests -q` -> `96 passed, 1 skipped, 1 warning, 11 subtests passed`

## Snapshot

Local snapshot: `E:\type10-7\code\snapshots\20260526_162400_stylebank_hetero_grl_collab`

Remote sync and N607 launch were not performed in this implementation step.
