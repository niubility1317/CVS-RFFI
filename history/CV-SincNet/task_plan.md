# Paper Baseline Implementation Plan

## Goal
Implement four paper-derived RFFI comparison methods in this repository as isolated, runnable baselines:

1. TIFS2025 channel/receiver robust RFFI: spectrogram + contrastive pretraining + Siamese fine-tuning.
2. RIEI: receiver-agnostic feature disentanglement with CE, mutual-independence loss, and entropy maximization.
3. DRIFT: feature split + GRL domain adversarial training + center loss + negative MSE separation.
4. Receiver-agnostic collaborative RFFI: GRL receiver-agnostic training, fine-tuning, and soft/adaptive fusion.

## Implementation Strategy
- Keep existing `type*` experiment folders untouched.
- Add an isolated `baselines/` package with shared components and one subpackage per paper method.
- Support real CSV/indexed `.npy` IQ data and synthetic IQ data for smoke testing.
- Provide configs, scripts, README files, and tests/import checks.

## Phases
| Phase | Status | Notes |
|---|---|---|
| 1. Read provided paper plans and inspect repo | complete | Used provided Codex plans as approved specs. |
| 2. Create shared baseline utilities | complete | Config, datasets, metrics, GRL, ResNet1D/2D, spectrogram, augmentation. |
| 3. Implement TIFS2025 baseline | complete | Pretrain, Siamese train, single-branch eval, tests. |
| 4. Implement RIEI baseline | complete | Model, losses, alternating trainer, eval. |
| 5. Implement DRIFT baseline | complete | Model, GRL, losses, train/eval, ablation switches. |
| 6. Implement receiver-agnostic collaborative baseline | complete | Model, training, eval, fine-tuning alias, fusion. |
| 7. Add configs/scripts/docs | complete | YAML templates, shell launchers, README. |
| 8. Run smoke tests | complete | `compileall` and `unittest` passed; synthetic checkpoints/metrics produced. |

## Decisions
- Treat the four `*_codex_plan.md` files supplied by the user as the design/spec approval for implementation.
- Use synthetic data generation for verification because the actual experiment data path is not specified.
- Favor small, composable modules over modifying the current large training scripts.

## Errors Encountered
| Error | Resolution |
|---|---|
| `rg.exe` failed with access denied | Use PowerShell `Get-ChildItem` and `Select-String`. |
| `git` not in PATH | Skip commit/history-dependent steps. |
| Some plan output was mojibake in terminal | Re-read with `-Encoding UTF8`; rely on readable code blocks and English method names. |
| Base Python lacked torch | Used `D:\App\miniconda3\envs\rff_std\python.exe` where torch 2.8.0+cpu is installed. |
| `rff_std` lacked PyYAML | Added a simple YAML fallback parser for these configs. |
| TIFS eval ran before parallel Siamese training wrote checkpoint | Re-ran eval after checkpoint existed. |
