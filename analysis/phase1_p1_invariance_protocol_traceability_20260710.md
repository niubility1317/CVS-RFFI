# Phase1 P1 Invariance Protocol Traceability (2026-07-10)

Scope: Phase1 source-only weak-label DG. Open-set quantities remain source/proxy diagnostics and do not establish true unknown FAR, FPR95, Stage2 success, or deployment success.

| ID | Requirement | Target files | Status | Verification | Notes |
|---|---|---|---|---|---|
| P1-SAT-01 | Make satellite training scenarios and source-val/heldout satellite evaluation scenarios disjoint, fail closed on overlap, and persist family/provenance fields. | `code/training_controls.py`; `code/SSDG/train_ssdg.py`; Phase1 launcher; tests | verified | `test_phase1_p1_protocol.py`; launcher dry-run test | Train uses `simplified_leo_residual_weak_v1`; frozen evaluation uses `legacy_satellite_physics_holdout_v1`. Scenario, family, config hash, and channel implementation overlap all fail closed. |
| P1-LEAK-02 | Add TX-conditional direct receiver/day/channel leakage losses on `z_id`, including concat clean/satellite channel labels. | `code/cvsrffi/losses.py`; `code/SSDG/train_ssdg.py`; tests | verified | direct-loss unit tests; trainer static contract tests | Group-center alignment is conditioned on TX; clean/satellite exact pairs have a separate cosine term. Skipped satellite transforms retain clean channel label `0`. |
| P1-PROBE-03 | Run deterministic frozen linear probes from source train to source val for receiver/day/channel and write non-empty accuracy/chance/excess fields. | `code/cvsrffi/leakage_probe.py`; `code/SSDG/train_ssdg.py`; tests | verified | frozen ridge probe tests; terminal fail-closed tests | Reports raw/balanced accuracy, chance, excess, counts, and status. Channel probe fits on clean versus train-residual views and evaluates on clean versus independent full-physics views. |
| P1-COMP-04 | Replace the direct-metric global class acceptance gate with receiver/day/channel-view local components while retaining global dispersion as an invariance diagnostic. | `code/cvsrffi/losses.py`; `code/SSDG/train_ssdg.py`; tests | verified | direct-metric and U_s quarantine tests | Direct loss, source-val tail safety, and U_s routing use class x receiver-day local components with separate clean/satellite views. Global-ball acceptance is disabled and global quantiles remain diagnostic only. |
| P1-CKPT-05 | Make final training weights the only selectable/exportable Phase1 checkpoint. Best/source-val metrics remain telemetry only and cannot select weights. | `code/SSDG/train_ssdg.py`; Phase1 launcher; tests | verified | final-only static contract and launcher tests | Only `final_ssdg.pth` is written. No latest, best, safe, tail-reference, or rejected model weight is saved; tail reference is metric-only and rollback is forbidden. |
| P1-GATE-06 | Bind P1 protocol/probe/component/final-only evidence into terminal fail-closed promotion readiness. | `code/SSDG/train_ssdg.py`; tests | verified | P1 terminal status and pre-export contract tests | Missing probes, implementation overlap, inactive local components, global fallback, inactive invariance, or non-final identity blocks promotion/export. |
| P1-DOC-07 | Update the experiment launcher/reporting contract and reverse-audit all P1 requirements. | launcher; this file; root planning files | verified | reverse audit plus `git diff --check` | No N607 sync or launch is part of this local implementation turn. |

## Acceptance boundary

- `sat_train_scenario_list` and `eval_sat_scenario_list` have zero scenario/family/config/channel-implementation overlap and the manifest records all provenance fields.
- Receiver/day/channel leakage probe fields are finite and include train/eval counts, class count, chance, accuracy, and excess.
- Direct geometry reports `local_component_gate=1`, component count, view-specific metrics, and `global_ball_accept=0`.
- `selected_checkpoint` is `final_ssdg.pth`; no non-final model-weight checkpoint is produced or used for heldout evaluation/prototype export.
- P1 evidence is bound to the same final checkpoint SHA256 used by heldout evaluation and endpoint export.

## Verification summary

- Syntax compilation passed for the trainer, training controls, leakage probe, losses, and focused tests.
- P0/P1 regression suite: 107 passed. Two existing AMP deprecation warnings remain non-blocking.
- Broader SSDG/baseline/concat-satellite/teacher-distillation suite: 22 passed.
- `git diff --check` passed; only the repository's existing Windows line-ending notices were emitted.

## Reverse audit result

- Verified requirements: 7.
- Deferred requirements: 0 for the requested P1 scope.
- Rejected approximations: using a renamed `leo_residual` holdout family was rejected because it retained the same channel implementation; evaluation now uses an independent full-physics implementation.
- Runtime evidence boundary: no new N607 training run was requested or launched, so actual receiver/day/channel leakage excess, strict UDU, satellite floor, and open-set proxy improvement remain unmeasured until the next experiment.
- Highest residual risk: the inherited teacher checkpoint lineage is unchanged. A future launch must record and audit teacher provenance so historical broad-domain behavior is not silently reintroduced through distillation.
