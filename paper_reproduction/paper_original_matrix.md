# Paper Original Matrix

This matrix records paper-faithful reproduction scope for methods in this release. It is an evidence ledger, not a claim that every paper table and figure has been reproduced.

## DADDA Cross-Receiver RFFI

Paper: Feng, Fang, and Fan, "Cross-Receiver Radio Frequency Fingerprint Identification Based on Domain Adaptation With Dynamic Distribution Alignment", IEEE Internet of Things Journal, 2025.

Scope boundary: DADDA is closed-set single-source UDA. It is not CVS Stage2-A/B/C, target-new enrollment, unknown/open-set rejection, or satellite/LEO deployment evidence.

| Paper item | Release status | Code or artifact | Missing for full reproduction |
|---|---|---|---|
| Section III closed-set source-to-target receiver UDA | Implemented protocol scaffold | `paper_reproduction/dadda_cross_receiver/data.py` | Real WiSig formal runs |
| Fig. 3 DADDA `G_f/G_m/G_l` pipeline | Implemented as runnable 1-D approximation | `paper_reproduction/dadda_cross_receiver/model.py` | Layer-for-layer 2-D Fig. 3 replica if required |
| Fig. 4 multiscale `G_m` | Implemented as 1-D four-branch approximation | `paper_reproduction/dadda_cross_receiver/model.py` | Exact 2-D `2x1/1x3/1x5` module |
| Eq. (2) MMD | Implemented and unit-tested | `paper_reproduction/dadda_cross_receiver/losses.py` | Formal real-data result |
| Eq. (3)-(4) LMMD | Implemented and unit-tested | `paper_reproduction/dadda_cross_receiver/losses.py` | Formal real-data result |
| Eq. (5) dynamic adaptive factor | Implemented with class-wise LMMD sum | `paper_reproduction/dadda_cross_receiver/losses.py` | Ablation against fixed weights |
| Eq. (6)-(10) total objective and training loop | Implemented smoke path | `paper_reproduction/dadda_cross_receiver/train.py` | Full 100-epoch real-data Table II run |
| Section V-B optimizer and schedules | Implemented and config-backed | `paper_reproduction/configs/dadda_cross_receiver_manysig_paper_faithful.json` | Hardware-matched formal run evidence |
| Table II 12 receiver-transfer tasks | Task matrix implemented | `PAPER_TABLE2_TASKS` and config file | Full numerical run plus all paper baselines |
| Table II DANN/DAN/DSAN/WD/DCORAL/CDAN baselines | Not implemented in this module | Runner marks unsupported methods `not_implemented` | Baseline runners and result table |
| Fig. 5 SNR robustness | Pending | Checklist only | AWGN/SNR runner and plots |
| Table III module ablation | Pending | Checklist only | Ablation runner and result table |
| Table IV dynamic alpha ablation | Pending | Checklist only | Fixed-weight comparison runner |
| Table V kernel sensitivity/params/FLOPs | Pending | Checklist only | Kernel sweep and complexity script |
| Fig. 6 A-distance | Pending | Checklist only | Feature-distance analysis script |
| Fig. 7 t-SNE | Pending | Checklist only | Visualization script and figure export |
| Fig. 8 confusion matrix | Pending | Checklist only | Confusion-matrix script and figure export |
| Table VI train/test time | Pending | Checklist only | Timing hook on declared hardware |

Current claim: paper-faithful DADDA method scaffold with formula-level and smoke verification. Full paper reproduction remains pending until the missing baselines, real-data metrics, ablations, robustness, visualization, complexity, and timing artifacts are produced.
