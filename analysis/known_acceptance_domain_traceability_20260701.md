# Known Acceptance Domain Optimization Traceability

Source inputs:
- `C:/Users/lh594/Downloads/CVS_RFFI_known_acceptance_domain_optimization_plan_20260701 (1).md`
- `E:/codex/home/attachments/356e8eaa-1610-4fa4-a728-acda667f39ee/pasted-text.txt`
- `AGENTS.md`
- `项目.md`

Scope boundary:
- Local implementation and verification only.
- No N607 launch or remote sync in this pass.
- Source-only proxy/synthetic reject metrics remain separate from real Stage2 unknown metrics.
- New training behavior must remain default-off unless explicitly enabled.

| ID | Source section | Requirement | Target files | Status | Verification | Notes |
|----|----------------|-------------|--------------|--------|--------------|-------|
| KAD-01 | Prompt 1 / report P0 | Make `--phase2_fuse_prototypes true` execute fusion on the normal Phase2 export path and write PT/JSON fusion fields. | `code/SSDG/train_ssdg.py`, `code/cvsrffi/phase2_prototypes.py`, `code/tests/test_phase2_prototype_fusion_export.py` | verified | `pytest ... test_phase2_prototype_fusion_export.py` PASS; `py_compile` PASS | Fixed unreachable call after `raise ImportError`; JSON/PT package now carries `fusion_config`, `fusion_components`, `fused_tx_prototypes`. |
| KAD-02 | Report P0/P2 | Add a re-export script for local component Phase2 packages without retraining. | `code/scripts/export_phase2_local_components.py` | verified | `py_compile` PASS | Script expects an existing Phase2 prototype package, not a raw training checkpoint. |
| KAD-03 | Prompt 2 | Provide a local component prototype bank with angular geometry, radius, density/NLL fallback, and old JSON compatibility. | `code/cvsrffi/component_geometry.py`, `code/cvsrffi/prototype_bank.py`, tests | verified | `pytest ... test_vacuum_gaussian_prototype_bank.py` PASS | Density/NLL unavailable fields are skipped, not faked. |
| KAD-04 | Prompt 3 | Implement hard gate where only local component core auto-accepts; tail defaults to review; outside/low-density/inter-class rejects. | `code/cvsrffi/hard_gate.py`, `code/cvsrffi/gate_metrics.py`, tests | verified | `pytest ... test_local_component_hard_gate.py test_stage2_gate_metrics.py` PASS | No global class ball fallback accept path. |
| KAD-05 | Prompt 4 | Implement shell, tail-outward, inter-class slerp, and same-class low-density bridge negative samplers. | `code/cvsrffi/negative_sampling.py`, tests | verified | `pytest ... test_negative_sampling.py` PASS | Synthetic negatives are reject benchmarks/training inputs, not real unknowns. |
| KAD-06 | Prompt 5 | Add default-off energy-in/out and reject-negative loss helpers that tolerate empty negatives. | `code/cvsrffi/losses.py`, tests | deferred | `pytest ... test_reject_energy_losses.py` PASS | Helper APIs and CLI flags are verified; full train-loop negative-logit/reject-head coupling is intentionally deferred. |
| KAD-07 | Prompt 6 | Add core/tail/outside quarantine partition and tail CVaR/overflow cap helpers. | `code/cvsrffi/tail_quarantine.py`, `code/SSDG/train_ssdg.py`, tests | deferred | `pytest ... test_tail_quarantine.py` PASS | Helper APIs and CLI flags are verified; old CE weighting is not changed by default. |
| KAD-08 | Prompt 7 | Add unlabeled pseudo-known core vs unknown-risk vs ignore mining helper. | `code/cvsrffi/unlabeled_risk_mining.py`, `code/SSDG/train_ssdg.py`, tests | deferred | `pytest ... test_unlabeled_risk_mining.py` PASS | Helper APIs and CLI flags are verified; pseudo-label training path is not changed by default. |
| KAD-09 | Prompt 8 | Add source episode safe partition helper so low-density query is not forcibly pulled into known. | `code/cvsrffi/source_episode_safe_gate.py`, tests | deferred | `pytest ... test_source_episode_safe_gate.py` PASS | Existing `source_episode_three_sigma_loss` remains unchanged by default. |
| KAD-10 | Prompt 9 | Add synthetic reject and Stage2 local-gate metric calculation scripts/helpers with real unknown metrics nullable when labels are absent. | `code/scripts/eval_synthetic_reject_benchmark.py`, `code/scripts/eval_stage2_local_gate.py`, tests | verified | `py_compile` PASS; `pytest ... test_stage2_gate_metrics.py` PASS | Does not fabricate unknown FAR when unknown labels are absent. |
| KAD-11 | Prompt 10 | Add dry-run launchers and collector for A/B/C/D mechanism matrix. | `code/scripts/launch_phase1_accept_domain_v2.sh`, `code/scripts/launch_stage2_gate_eval_v2.sh`, `code/scripts/collect_accept_domain_v2.py`, tests | verified | `bash -n` PASS; both launchers `--dry-run` PASS | Dry-run only locally; no N607 launch. |
| KAD-12 | Prompt 11 | Classify skipped-test NaN, aux grad NaN, real loss/metric NaN, and fatal NaN separately. | `code/scripts/collect_accept_domain_v2.py`, tests | verified | `pytest ... test_log_nan_parser.py` PASS | T16-T31 aborted artifacts excluded. |
| KAD-13 | Prompt 12/13 | Maintain QA checklist and report boundary: proxy/synthetic improvements are not Stage2 success. | this traceability record, final report | verified | Traceability and final handoff boundary | Final answer must state strict design parity vs approximation. |
