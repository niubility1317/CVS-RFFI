# Orthogonal Incremental SEI Reproduction Findings

Source paper: `正交空间约束的特定辐射源小样本类增量识别方法`.

## Scope

This implementation is a paper-faithful closed-set FSCIL-SEI scaffold. It covers the method mechanics needed to reproduce the paper's base-stage orthogonal-space learning and incremental classifier-weight calibration. It does not claim formal ADS-B or WiFi table reproduction yet.

The paper setting is not CVS Stage2-B/C evidence. The paper does not define disjoint source/target receivers, target-old support, target-new support under a LEO view, or unknown-query rejection. Any CVS use must be implemented under `paper_reproduction/cvs_aligned/` with `cvs_extension=true`.

## Implemented

| Paper requirement | Implementation |
|---|---|
| Pseudo-target bound `|C| <= N <= d + 1` and simplex geometry | `paper_reproduction/orthogonal_incremental_sei/pseudo_targets.py` |
| Formula (4) orthogonal pseudo-target loss and iterative optimization | `pseudo_target_orthogonal_loss`, `optimize_pseudo_targets`, and the dry-run entrypoint; paper-protocol configs enable iterative optimization through `pseudo_target_steps`. |
| Formula (6) perturbed pseudo targets | `perturb_pseudo_targets`; default behavior is the paper's additive `target + epsilon` form, with renormalization only as an explicit implementation option. |
| Six Conv1D-BN-MaxPool encoder and cosine classifier | `model.py` |
| Base losses `Lce`, `Ls`, `Lc`, and `Linit` | `losses.py`; `Ls` follows the paper's sample-anchor positive/negative sets, unassigned-pseudo negative set, and per-positive averaging form. |
| Incremental calibration `Lh`, `La`, and `Linc` | `incremental_calibration_loss`; input shape/class validation now rejects silent sample drops. |
| FSCIL metrics `A_bar`, `H_bar`, `F_bar` | `metrics.py`; `F_bar` defaults to the paper's incremental-task denominator and does not clamp negative forgetting. |
| Synthetic wiring verification | `configs/orthogonal_incremental_sei_smoke.json` and `train.py --dry-run` |
| Paper-named dry-run hyperparameters | Configs use `tau_s`, `tau_c`, `q`, and `lambda_a`; the entrypoint keeps compatibility aliases for older configs. |
| Incremental freeze boundary | Dry-run explicitly freezes the encoder and reports `encoder_trainable_after_increment=0`. |
| Claim boundary guard | Dry-run emits `claim_boundary=synthetic_dry_run_not_formal_reproduction` and reports unsupported formal protocol fields. |

## Remaining Gaps

| Gap | Reason |
|---|---|
| ADS-B table reproduction | The public ADS-B file path, 100-class filtering, and class order are not wired yet. |
| WiFi/WiSig table reproduction | The paper requires same-receiver filtering, at least 50 samples per transmitter, and a 130-class set; the loader and class list still need to be implemented. |
| Formal incremental training loop | The current entrypoint is a dry-run scaffold; it verifies losses and calibration wiring but does not run full sessions. |
| Figure 4-7 reproduction | Forgetting-rate curves and initial/incremental class-count sensitivity curves are not generated yet. |
| Exact backbone hyperparameters | The paper describes six Conv1D modules but does not provide channel sizes, kernel sizes, or strides. Current values are implementation choices. |
| Exact temperature/margin grid | `tau_fuse=0.01`, `top_k=60`, perturbation `0.01`, and `lambda_a=1.6` are paper-supported defaults; `tau_s`, `tau_c`, and `q` need author-code confirmation or a documented grid. |
| Source evidence publication | The root workspace keeps the detailed matrix and gap ledger in `paper_reproduction/paper_original_matrix.md` and `paper_reproduction/repro_gap.md`; this GitHub release keeps only a bounded summary. |

## Verification

Use the `ssr-gpu` environment:

```powershell
& 'C:\Users\lh594\.conda\envs\ssr-gpu\python.exe' -m pytest tests/test_orthogonal_incremental_sei.py -q
& 'C:\Users\lh594\.conda\envs\ssr-gpu\python.exe' -m paper_reproduction.orthogonal_incremental_sei.train --config paper_reproduction/configs/orthogonal_incremental_sei_smoke.json --dry-run --device cpu
```
