# Multi-disentanglement report: source-action validation

Source: user attachment `已粘贴的文本.txt`, received 2026-10-09. Implement report step 1 and bounded source-only comparisons of step 2 before any fresh E200 matrix. Existing target scores are not consumed by this experiment. The frozen identity model is not optimized in this diagnostic; auxiliary fitting and audit use disjoint physical packets from source L. No claim of new target performance.

| ID | Source | Requirement | Target files | Status | Verification | Notes |
|---|---|---|---|---|---|---|
| A01 | 10 / protocol | Fixed physical source-L fit/audit split; all views inherit packet role; U labels and V untouched | runner.py, design.py | verified | role negative checks | Audit unseen by auxiliary fitting, not unseen by original identity training |
| A02 | 3 | Three-block normalized fit plus differentiable frozen G error | actions.py | verified | gradient and scale checks | Separate original-fit control |
| A03 | 4 | Heldout zero/mean skill, energy, G error, actual/predicted margin and honest reliability | actions.py | verified | zero-energy and zero-skill tests | No same-batch training reliability |
| A04 | 8 | Self/cross-TX/shuffled/oracle diagnostics using identical interventions across TX | actions.py | verified | matching and semantic-use tests | Oracle remains diagnostic only |
| A05 | 5.4 | Contribution age and reference version bounded, no stale refresh | receiver.py | verified | sparse-refresh expiry test | Frozen reference in current source diagnostic |
| A06 | 5.4 | No sorted-class truncation; balanced unique relationship coverage | receiver.py | verified | permutation/coverage checks | Preserve strata and source RX roles |
| A07 | 5.1 | Aggregate per-packet time statistics; phase cancellation counterexample and repeatability | receiver.py | verified | x/-x/CFO test | No raw-IQ group mean |
| A08 | 5.2 | Overall matched source-RX action without unvalidated L/T subtraction | receiver.py | verified | target semantics tests | No pure receiver-hardware claim |
| A09 | 5.3 | Compare original relation loss and stop-gradient destination-source-RX class anchors | receiver.py | verified | rotation/gradient test | Group distribution, not paired transmissions |
| A10 | 6/9 | Window conditioning, delay routing, boundaries and h/block/z spectra | actions.py | verified | numeric checks and real-source outputs | Architecture/rank not changed based on target |
| A11 | 7 | Pure LT, additive/sequential/actual composition and total-action diagnostics | runner.py | verified | factorial comparisons | Learn LT only after source evidence |
| A12 | 9 | Identity component gradient norm ratio and cosine for actual/virtual losses | runner.py | verified | no parameter mutation | Native loss vs clean CE reference must be distinguished |
| A13 | 10 | Matched source clean/LEO exposure across diagnostic comparisons | runner.py | verified | deterministic exposure IDs | Existing native augmentation only |
| A14 | workflow | Provenance, registration, local tests, one P0/P1 review, immutable N607 release and readback | design.py, publish.py, dispatch.py | implemented | focused tests and launch evidence | Four seeds, up to four processes/GPU per user |
| A15 | 10.2/3 | Full identity E200 ablation and fresh test predictions | subsequent run | deferred | source audit first | Report explicitly prioritizes diagnostic before larger matrix; cannot test an unchanged identity as an optimization result |
| A16 | 7/9 | New LT training, larger rank, front-end routing changes and stronger weights | subsequent run | deferred | require source evidence | Preserve network/rank/weight in first diagnostic |

The next full training experiment must split real-IQ CE from learned-action reliability and keep main-effect weights and x11 exposure matched across no/zero/learned-LT controls. This diagnostic measures the required evidence and does not silently claim those future training changes are deployed.

Local evidence: actual CVS CPU/CUDA fits, frozen-G differentiability, physical-role negative checks, composition, deterministic source LEO, Jensen and balanced gradient sampling verified. P0/P1 review and one focused recheck PASS. A12 uses a clean-CE reference, not the full native pseudo/LEO loss; that full training diagnostic remains deferred with A15. Counts before publication: verified=13, implemented=1, deferred=2, rejected=0, blocked=0.

Local evidence: actual CVS CPU/CUDA fits, frozen-G differentiability, physical-role negative checks, composition, deterministic source LEO, Jensen and balanced gradient sampling verified. P0/P1 review and one focused recheck PASS. A12 uses a clean-CE reference, not the full native pseudo/LEO loss; that full training diagnostic remains deferred with A15. Counts before publication: verified=13, implemented=1, deferred=2, rejected=0, blocked=0.
