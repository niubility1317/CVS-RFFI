# DADDA Paper-to-Code Audit Checklist

Scope: paper-faithful closed-set UDA reproduction for Feng et al. (IEEE IoT Journal, 2025). This checklist does not certify CVS Stage2-C, target-new enrollment, unknown rejection, or satellite/LEO deployment.

| Paper item | Paper evidence | Code evidence | Test/result evidence | Current status |
|---|---|---|---|---|
| Task definition | Section III: labeled source receiver plus unlabeled target receiver; `Y_s=Y_t` | `paper_reproduction/dadda_cross_receiver/data.py`; `train.py` dry-run payload | `tests/test_dadda_cross_receiver.py::test_dadda_dry_run_declares_closed_set_uda_not_cvs` | Implemented as closed-set UDA |
| CVS boundary | `项目.md` requires CVS old/new/unknown split for Stage2-C; DADDA does not | `configs/dadda_cross_receiver_manysig_paper_faithful.json` has `cvs_extension=false` | Dry-run claim blocks | Implemented boundary guard |
| Input shape | Section V-A: `2x256` IQ samples | `DADDAFeatureExtractor`; `build_manysig_task_datasets` | Model and data task tests | Implemented |
| WiSig subset | Section V-A: six transmitters, twelve receivers, four days | `build_manysig_task_datasets` validates counts | synthetic ManySig fixture test | Implemented validation |
| Backbone `G_f` | Section IV-B and Fig. 3: modified ResNet18 | `DADDAFeatureExtractor` | model shape test | Implemented runnable approximation |
| Multiscale `G_m` | Fig. 4: four branches `2x1`, `2x1+1x3`, `2x1+1x5`, `AvgPool+2x1` | `DADDAMultiscaleExtractor` | model shape test | Implemented |
| Classifier `G_l` | Section V-B: two FC layers with 512 and 128 neurons | `DADDAClassifier`; config hidden widths | model shape test and dry-run hyperparameters | Implemented |
| MMD Eq. (2) | global alignment on `G_f(X_s),G_f(X_t)` | `mmd_loss` | objective test | Implemented |
| LMMD Eq. (3)-(4) | source one-hot labels; target predicted probabilities | `lmmd_loss` | objective test | Implemented |
| Dynamic factor Eq. (5) | `alpha=d_MMD/(d_MMD+sum d_LMMD)` and `alpha in [0,1]` | `dynamic_adaptive_factor` | objective test | Implemented |
| Total objective Eq. (6)-(9) | `CE + lambda*((1-alpha)MMD + alpha*LMMD)` | `dadda_objective`; `run_dadda_training_loop` | smoke runner history logs `alpha_mean`, `mmd_mean`, `lmmd_mean` | Implemented |
| Algorithm 1 | each batch uses `x_s,y_s,x_t`, target labels hidden for training | `run_dadda_training_loop` | smoke runner test | Implemented smoke path |
| Optimizer and schedules | Section V-B: SGD, momentum 0.9, L2 decay 0.0005, lr/lambda schedules | `train.py`; config file | dry-run hyperparameters; smoke history | Implemented |
| Table II tasks | twelve receiver-transfer tasks | `PAPER_TABLE2_TASKS`; config file | dry-run task plan | Implemented task matrix |
| Table II metrics | source-only and DADDA target-domain accuracy; baselines DANN/DAN/DSAN/WD/DCORAL/CDAN | `run_table2_reproduction` supports `source_only` and `dadda`; other baselines explicitly not implemented here | formal WiSig run required | Partially implemented; full table pending real data/GPU runs |
| SNR Fig. 5 | AWGN SNR 0-20 dB on two tasks | no SNR runner yet | pending | Pending |
| Ablation Tables III-IV | module ablations and dynamic-alpha comparison | no ablation matrix yet | pending | Pending |
| Kernel Table V | kernel sensitivity, params, FLOPs | no kernel sweep yet | pending | Pending |
| Fig. 6-8 analyses | A-distance, t-SNE, confusion matrix | no visualization scripts yet | pending | Pending |
| Time Table VI | per-epoch train/test time on the paper hardware | no timing hook yet | pending | Pending |

Known paper ambiguity: the prose names a tradeoff parameter `gamma`, while Eq. (9) and Algorithm 1 use `lambda`. This implementation follows Eq. (9) and Algorithm 1 and records the lambda schedule in dry-run output.

