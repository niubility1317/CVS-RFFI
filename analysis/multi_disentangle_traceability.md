# Multi-disentanglement design implementation

Authority: user-approved design in this chat, 2026-10-09. The user explicitly requested parallel implementation without deviation. Root is the only launch owner. Existing healthy experiments and unrelated edits must remain untouched.

## Accepted design

One existing reference-response identity backbone F=G(E(x)); independent linear-response D_L, temporal D_T and cross-receiver statistical residual D_R networks; optional independently supervised factorial interaction D_LT. Each network uses its own nuisance view encoder and bounded low-rank conditional action. Auxiliary fitting uses detached current-run EMA intermediate anchors and its own optimizer. Identity training consumes frozen predicted changes and retains native supervised/semi-supervised training. Receiver observations are not exact same-packet counterfactual pairs. Inference uses only the identity model. No GRL, orthogonality, MI, auxiliary TX classifier or identity-feature concatenation.

All rows start from scratch with the same legal source physical-role contract. Legacy batch-neighbor pseudo gate, own EMA, cosine LR, full-U entropy, base clean+satellite concat and 200x222 identity updates stay fixed. Added perturbation exposures/compute are disclosed and controlled. Source V never updates state. All fixed comparison rows freeze their own E200 checkpoints before any query, and all predictions precede independent truth-last scoring.

## Implementation plan and ownership

1. Model worker: model.py / physics.py; actual E/G split, independent learned codes/operators and order-correct factorial interventions.
2. Runtime worker: runtime.py / source.py / train_worker.py; alternating training, source-only grouped RX statistics, production telemetry and auxiliary checkpoint.
3. Verification worker: checks.py / smoke.py; meaningful gradient/protocol tests and independent P0/P1 review.
4. Root: design/configuration, registration, evaluation/publication/controller, report, integration, commit/push and remote delivery.

## Traceability

| ID | Source section | Requirement | Target files | Status | Verification | Notes |
|---|---|---|---|---|---|---|
| M01 | 1,9 | Independent D_L/D_T/D_R; optional separate D_LT | model.py | verified | structural/gradient checks | No multi-head substitution |
| M02 | 2 | Actual identity intermediate E/G with exact native inference | model.py | verified | baseline/logit and checkpoint equivalence | Preserve reference identity path |
| M03 | 3 | Learned factor-specific encoders and bounded low-rank actions; zero action identity | model.py,physics.py | verified | code ablation/finite-gradient/zero-action tests | Oracle parameters supervise, never replace learned codes |
| M04 | 4 | x00,x10,x01,x11=A_T(A_L x) and independently supervised interaction | physics.py,runtime.py | verified | factorial/order/target checks | No commutativity assumption |
| M05 | 5 | D_R legal L-only quality-matched grouped statistics and weak relational identity constraint | runtime.py | verified | metadata/truth isolation and nonzero receiver update | Not exact packet pairing; no physical-purity claim |
| M06 | 6 | Auxiliary fit detached own EMA, separate optimizer; identity constraint freezes auxiliary parameters | runtime.py | verified | both-direction gradient isolation | No historical weight inheritance |
| M07 | 6 | Native old gate/cosine/EMA/U-entropy/basic concat unchanged | design.py,runtime.py,source.py | verified | production epoch smoke/config equality | 44400 identity updates |
| M08 | 7 | Independent per-factor targets, bounded total auxiliary weighting, fit reliability | runtime.py | verified | per-arm loss/weight telemetry | No all-zero free learned gate |
| M09 | 8 | Same-factor physical+residual branch described as optional extension | design.md | deferred | explicit design scope | Last design chose 1+3 main and optional interaction; no speculative extra branch in first matrix |
| M10 | 9 | Baseline/fixed intervention/unified/three single/multi/interaction controls | design.py,configs | verified | immutable 8-arm matrix checks | Four inherited model seeds; 32 fixed controls |
| M11 | 9 | Training auxiliaries removable; per-packet identity-only inference | model.py,evaluate.py | verified | stateless prediction/strict checkpoint tests | Report separate training and deployment costs |
| M12 | protocol | Scratch provenance, source-only fitting, own E200 freeze, complete truth-last test | design.py,evaluate.py,dispatch.py | verified | protocol negative tests/full synthetic scoring | Clean and six existing full proxy views; not new blind benchmark |
| M13 | logs | Detailed losses/weights/grads/mechanism execution/validation/resources | runtime.py,source.py | verified | actual incremental writer chain smoke | Missing values null with reason |
| M14 | delivery | One owner, <=4 total processes/GPU, immutable paths, preserve existing runs | publish.py,dispatch.py | verified | independent runtime readback | User's four-process override persists |
| M15 | delivery | Local tests, one P0/P1 review, commit/push/OID, report and registration | checks.py,smoke.py,report.md | verified | evidence paths recorded below | No success claim before evidence |

Evidence: [local verification](../automation_reports/CV-SincNet/20261009-phase1-multi-disentangle-manysig-m32-r01/evidence/local_verified.json). CPU/CUDA structure, gradients, zero/bounds and actual native training paths were checked; synthetic scoring exercises the complete fixed matrix without reading target data. GPU reproducibility diagnostics and limitations are recorded in the report. Mathematical factor recovery and positive test gains are not assumed. M12 is implemented and synthetically verified; formal E200 training and complete test results remain outstanding.

Launch: [independent runtime evidence](../automation_reports/CV-SincNet/20261009-phase1-multi-disentangle-manysig-m32-r01/evidence/launch_verified.json). M15 local code commit/push and remote OID were verified before publication. Formal performance remains unverified. Counts: verified=13, implemented=1, deferred=1, rejected=0, blocked=0.

Final validation: M12 verified by evidence/final_verified.json and all224 predictions plus independent truth-last recount. Final counts: verified=14, deferred=1 (optional extension), rejected=0, blocked=0. No remaining required execution; positive performance gain was not established. Full results: ../automation_reports/CV-SincNet/20261009-phase1-multi-disentangle-manysig-m32-r01/results/complete_summary.md. Earlier pending statements describe launch-time state.
