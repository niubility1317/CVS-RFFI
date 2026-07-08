# Paper Original Matrix

This matrix records paper-faithful reproduction scope for methods in this release. It is an evidence ledger, not a claim that every paper table and figure has been reproduced.

## DADDA Cross-Receiver RFFI

Paper: Feng, Fang, and Fan, "Cross-Receiver Radio Frequency Fingerprint Identification Based on Domain Adaptation With Dynamic Distribution Alignment", IEEE Internet of Things Journal, 2025.

Scope boundary: DADDA is closed-set single-source UDA. It is not CVS Stage2-A/B/C, target-new enrollment, unknown/open-set rejection, or satellite/LEO deployment evidence.

| Paper item | Release status | Code or artifact | Missing for full reproduction |
|---|---|---|---|
| Eq. (1) receiver-signal model | Documentation/protocol narrative only | `paper_reproduction/dadda_cross_receiver/paper_checklist.md` | No numerical channel-model implementation is claimed |
| Table I related-work comparison | Documentation-only paper context | This matrix | No code artifact required |
| Section III closed-set source-to-target receiver UDA | Implemented protocol scaffold | `paper_reproduction/dadda_cross_receiver/data.py` | Real WiSig formal runs |
| Fig. 3 DADDA `G_f/G_m/G_l` pipeline | Implemented as a 2-D paper-shaped variant with legacy 1-D ablation retained | `paper_reproduction/dadda_cross_receiver/model.py`; `paper_reproduction/configs/dadda_cross_receiver_manysig_paper_faithful.json` | Author-code-level layer details remain unavailable |
| Fig. 4 multiscale `G_m` | Implemented with 2-D `2x1/1x3/1x5` branch kernels | `paper_reproduction/dadda_cross_receiver/model.py` | Formal real-data validation after the correction |
| Eq. (2) MMD | Implemented and unit-tested with one shared batch kernel bandwidth | `paper_reproduction/dadda_cross_receiver/losses.py` | Formal real-data result |
| Eq. (3)-(4) LMMD | Implemented and unit-tested with one shared batch kernel bandwidth | `paper_reproduction/dadda_cross_receiver/losses.py` | Formal real-data result |
| Eq. (5) dynamic adaptive factor | Implemented with class-wise LMMD sum | `paper_reproduction/dadda_cross_receiver/losses.py` | Ablation against fixed weights |
| Eq. (6)-(10) total objective and training loop | Implemented smoke path | `paper_reproduction/dadda_cross_receiver/train.py` | Full 100-epoch real-data Table II run |
| Section V-B optimizer, schedules, and domain sample count | Implemented and config-backed, including 4000 samples per receiver-domain control | `paper_reproduction/configs/dadda_cross_receiver_manysig_paper_faithful.json` | Hardware-matched formal run evidence |
| Table II 12 receiver-transfer tasks | Task matrix implemented | `PAPER_TABLE2_TASKS` and config file | Full numerical run plus all paper baselines |
| Table II DANN/DAN/DSAN/WD/DCORAL/CDAN baselines | Not implemented in this module | Runner emits structured missing-baseline rows | Baseline runners and result table |
| Fig. 5 SNR robustness | Plan scaffold only | `paper_reproduction/dadda_cross_receiver/experiment_plans.py` | AWGN/SNR runner and plots |
| Table III module ablation | Plan scaffold only | `paper_reproduction/dadda_cross_receiver/experiment_plans.py` | Ablation runner and result table |
| Table IV dynamic alpha ablation | Fixed/dynamic alpha loss and plan scaffold | `paper_reproduction/dadda_cross_receiver/losses.py`; `experiment_plans.py` | Fixed-weight comparison runner and result table |
| Table V kernel sensitivity/params/FLOPs | Artifact schema scaffold only | `paper_reproduction/dadda_cross_receiver/experiment_plans.py` | Kernel sweep and complexity script |
| Fig. 6 A-distance | Artifact schema scaffold only | `paper_reproduction/dadda_cross_receiver/experiment_plans.py` | Feature-distance analysis script |
| Fig. 7 t-SNE | Artifact schema scaffold only | `paper_reproduction/dadda_cross_receiver/experiment_plans.py` | Visualization script and figure export |
| Fig. 8 confusion matrix | Artifact schema scaffold only | `paper_reproduction/dadda_cross_receiver/experiment_plans.py` | Confusion-matrix script and figure export |
| Table VI train/test time | Artifact schema scaffold only | `paper_reproduction/dadda_cross_receiver/experiment_plans.py` | Timing hook on declared hardware |
| Multi-source/multi-target extension | Out of current paper-faithful release scope | This matrix | Future-work extension design |

Current claim: paper-faithful DADDA method scaffold with formula-level and smoke verification. Full paper reproduction remains pending until the missing baselines, real-data metrics, ablations, robustness, visualization, complexity, and timing artifacts are produced.
