# Findings: Mitigating receiver impact DA

## Confirmed starting point

- Prior best strict final reproduction results were 70.97% for `14-7->3-19`, 83.47% for `1-1->1-19`, and 86.61% for `7-7->8-8`.
- The corresponding paper values previously transcribed were 92.42%, 95.44%, and 99.74%.
- `14-7->3-19` remained dominated by low-precision pseudo-labels for classes `20-15` and `20-19`.
- Target-label-based checkpoint selection is diagnostic/oracle and cannot support a strict UDA reproduction claim.

## Evidence to collect

- Exact paper text, equations, tables, and stated/unstated implementation details.
- Complete local and remote artifact inventory and full learning curves.
- Code paths for data construction, model, MINE, E/C steps, pseudo-labeling, class weighting, normalization, checkpointing, and evaluation.

## Confirmed paper-to-code mismatches

- Paper Section II requires subtracting the signal mean before power normalization. `code/dataset_wisig.py::_rms_normalize_iq` currently divides by RMS without centering.
- Paper Algorithm 1 resets `sigma_0(k)`, `sigma'_0(k)`, and `n^t_0` at the start of each outer iteration. The strict runner default is `pseudo_state_scope=global`; epoch-scoped state is only enabled by the official-compat switch.
- Paper Algorithm 1 pairs source and target for `min(Ns/b,Nt/b)` batches. The strict runner default is `cycle_target`; `zip_min` is only enabled by the official-compat switch.
- Paper Section IV-A starts from an initial model `h0`, for example a model learned from Rx-1. Most prior best-final runs adapted from random initialization; the paper does not state the required source-pretraining duration.
- The public trainer performs one source/target model forward before MINE and reuses those outputs/features. The reproduction E/C step separately calls `_estimate_outputs` and then `model(...)`, so BatchNorm running statistics are updated twice per domain in the E/C step.
- The public trainer uses `torch.nn.CrossEntropyLoss(weight=...)`, whose weighted mean is normalized by the selected sample-weight sum. The reproduction uses `mean(weight * CE)`. The latter matches the literal paper Eq. 10; the former matches the released trainer and may be necessary to reproduce its reported numbers.
- The public trainer resets MINE's moving average to its default inside each of the `m` estimate updates, then passes only the final returned value to the E/C KL call. The reproduction carries the moving average across all `m` updates.

## N607 data audit

- Direct preflight passed at 2026-07-10 10:53 CST; all eight RTX 3090 GPUs were idle and no related training process was active.
- For receivers `14-7`, `3-19`, `1-1`, `1-19`, `7-7`, and `8-8`, equalized ManySig contains exactly 4000 samples per TX and every sample is shaped `(256,2)`.
- Mean subtraction removes only 0.052%-0.061% of average signal power for these receivers. Centering is a real paper mismatch but cannot plausibly explain a 20-50pp gap by itself.

## Public-code boundary

- The authors' public repository exposes only `mine_pseudo_classweight_trainer.py` and points to `YannLeo/Pytorch-Template`.
- It does not publish the experiment TOML, WiSig dataset wrapper/split, model wrapper returning `(output, feature)`, MINE class, or exact ResNet18 constructor arguments. Exact author-code reproduction therefore remains underdetermined even after matching the exposed trainer.
