# Frozen branch support-only diagnostic

Verified 4800 episodes across 8 lanes: 1200 K1 numerical-only, 3600 physical OOF, 64800 Cholesky factorizations.

This is support evidence only. No arm is selected and no query generalization claim is made.

## Six fixed arms and paired changes

Values below are equal-episode descriptive means; new-present and old-only populations are separate. Accuracy deltas are fractions (multiply by 100 for percentage points).

| Population | Metric | N | Mean |
|---|---|---:|---:|
| new_present | z.accuracy | 2880 | 0.34080263 |
| new_present | z.h | 2880 | 0.25360286 |
| new_present | z.macro_accuracy | 2880 | 0.34080263 |
| new_present | z.macro_nll | 2880 | 2.4790271 |
| new_present | z.new_accuracy | 2880 | 0.20900608 |
| new_present | z.old_accuracy | 2880 | 0.46056134 |
| new_present | z_aux.accuracy | 2880 | 0.46107412 |
| new_present | z_aux.h | 2880 | 0.41569569 |
| new_present | z_aux.macro_accuracy | 2880 | 0.46107412 |
| new_present | z_aux.macro_nll | 2880 | 2.4254973 |
| new_present | z_aux.new_accuracy | 2880 | 0.36368316 |
| new_present | z_aux.old_accuracy | 2880 | 0.54756366 |
| new_present | z_duplicate.accuracy | 2880 | 0.34976852 |
| new_present | z_duplicate.h | 2880 | 0.26837485 |
| new_present | z_duplicate.macro_accuracy | 2880 | 0.34976852 |
| new_present | z_duplicate.macro_nll | 2880 | 2.4645767 |
| new_present | z_duplicate.new_accuracy | 2880 | 0.22171441 |
| new_present | z_duplicate.old_accuracy | 2880 | 0.4654022 |
| new_present | zfft.accuracy | 2880 | 0.4483573 |
| new_present | zfft.h | 2880 | 0.39764312 |
| new_present | zfft.macro_accuracy | 2880 | 0.4483573 |
| new_present | zfft.macro_nll | 2880 | 2.5214794 |
| new_present | zfft.new_accuracy | 2880 | 0.34014323 |
| new_present | zfft.old_accuracy | 2880 | 0.54271701 |
| new_present | zfft_aux.accuracy | 2880 | 0.52336779 |
| new_present | zfft_aux.h | 2880 | 0.49228334 |
| new_present | zfft_aux.macro_accuracy | 2880 | 0.52336779 |
| new_present | zfft_aux.macro_nll | 2880 | 2.4463003 |
| new_present | zfft_aux.new_accuracy | 2880 | 0.4423559 |
| new_present | zfft_aux.old_accuracy | 2880 | 0.59297164 |
| new_present | zfft_duplicate.accuracy | 2880 | 0.45766252 |
| new_present | zfft_duplicate.h | 2880 | 0.40997679 |
| new_present | zfft_duplicate.macro_accuracy | 2880 | 0.45766252 |
| new_present | zfft_duplicate.macro_nll | 2880 | 2.4858249 |
| new_present | zfft_duplicate.new_accuracy | 2880 | 0.35188108 |
| new_present | zfft_duplicate.old_accuracy | 2880 | 0.5495544 |
| new_present | z.aux_minus_base.accuracy | 2880 | 0.12027149 |
| new_present | z.aux_minus_base.h | 2880 | 0.16209283 |
| new_present | z.aux_minus_base.macro_accuracy | 2880 | 0.12027149 |
| new_present | z.aux_minus_base.macro_nll | 2880 | -0.053529751 |
| new_present | z.aux_minus_base.new_accuracy | 2880 | 0.15467708 |
| new_present | z.aux_minus_base.old_accuracy | 2880 | 0.087002315 |
| new_present | z.aux_minus_duplicate.accuracy | 2880 | 0.1113056 |
| new_present | z.aux_minus_duplicate.h | 2880 | 0.14732084 |
| new_present | z.aux_minus_duplicate.macro_accuracy | 2880 | 0.1113056 |
| new_present | z.aux_minus_duplicate.macro_nll | 2880 | -0.039079404 |
| new_present | z.aux_minus_duplicate.new_accuracy | 2880 | 0.14196875 |
| new_present | z.aux_minus_duplicate.old_accuracy | 2880 | 0.082161458 |
| new_present | z.duplicate_minus_base.accuracy | 2880 | 0.0089658909 |
| new_present | z.duplicate_minus_base.h | 2880 | 0.014771995 |
| new_present | z.duplicate_minus_base.macro_accuracy | 2880 | 0.0089658909 |
| new_present | z.duplicate_minus_base.macro_nll | 2880 | -0.014450347 |
| new_present | z.duplicate_minus_base.new_accuracy | 2880 | 0.012708333 |
| new_present | z.duplicate_minus_base.old_accuracy | 2880 | 0.0048408565 |
| new_present | zfft.aux_minus_base.accuracy | 2880 | 0.075010494 |
| new_present | zfft.aux_minus_base.h | 2880 | 0.094640216 |
| new_present | zfft.aux_minus_base.macro_accuracy | 2880 | 0.075010494 |
| new_present | zfft.aux_minus_base.macro_nll | 2880 | -0.075179117 |
| new_present | zfft.aux_minus_base.new_accuracy | 2880 | 0.10221267 |
| new_present | zfft.aux_minus_base.old_accuracy | 2880 | 0.05025463 |
| new_present | zfft.aux_minus_duplicate.accuracy | 2880 | 0.065705272 |
| new_present | zfft.aux_minus_duplicate.h | 2880 | 0.082306548 |
| new_present | zfft.aux_minus_duplicate.macro_accuracy | 2880 | 0.065705272 |
| new_present | zfft.aux_minus_duplicate.macro_nll | 2880 | -0.039524623 |
| new_present | zfft.aux_minus_duplicate.new_accuracy | 2880 | 0.090474826 |
| new_present | zfft.aux_minus_duplicate.old_accuracy | 2880 | 0.043417245 |
| new_present | zfft.duplicate_minus_base.accuracy | 2880 | 0.0093052217 |
| new_present | zfft.duplicate_minus_base.h | 2880 | 0.012333669 |
| new_present | zfft.duplicate_minus_base.macro_accuracy | 2880 | 0.0093052217 |
| new_present | zfft.duplicate_minus_base.macro_nll | 2880 | -0.035654494 |
| new_present | zfft.duplicate_minus_base.new_accuracy | 2880 | 0.011737847 |
| new_present | zfft.duplicate_minus_base.old_accuracy | 2880 | 0.0068373843 |
| old_only | z.accuracy | 720 | 0.6153588 |
| old_only | z.h | 0 | N/A |
| old_only | z.macro_accuracy | 720 | 0.6153588 |
| old_only | z.macro_nll | 720 | 1.5028666 |
| old_only | z.new_accuracy | 0 | N/A |
| old_only | z.old_accuracy | 720 | 0.6153588 |
| old_only | z_aux.accuracy | 720 | 0.65356481 |
| old_only | z_aux.h | 0 | N/A |
| old_only | z_aux.macro_accuracy | 720 | 0.65356481 |
| old_only | z_aux.macro_nll | 720 | 1.4749705 |
| old_only | z_aux.new_accuracy | 0 | N/A |
| old_only | z_aux.old_accuracy | 720 | 0.65356481 |
| old_only | z_duplicate.accuracy | 720 | 0.61868056 |
| old_only | z_duplicate.h | 0 | N/A |
| old_only | z_duplicate.macro_accuracy | 720 | 0.61868056 |
| old_only | z_duplicate.macro_nll | 720 | 1.4755687 |
| old_only | z_duplicate.new_accuracy | 0 | N/A |
| old_only | z_duplicate.old_accuracy | 720 | 0.61868056 |
| old_only | zfft.accuracy | 720 | 0.65121528 |
| old_only | zfft.h | 0 | N/A |
| old_only | zfft.macro_accuracy | 720 | 0.65121528 |
| old_only | zfft.macro_nll | 720 | 1.6485478 |
| old_only | zfft.new_accuracy | 0 | N/A |
| old_only | zfft.old_accuracy | 720 | 0.65121528 |
| old_only | zfft_aux.accuracy | 720 | 0.68565972 |
| old_only | zfft_aux.h | 0 | N/A |
| old_only | zfft_aux.macro_accuracy | 720 | 0.68565972 |
| old_only | zfft_aux.macro_nll | 720 | 1.5506913 |
| old_only | zfft_aux.new_accuracy | 0 | N/A |
| old_only | zfft_aux.old_accuracy | 720 | 0.68565972 |
| old_only | zfft_duplicate.accuracy | 720 | 0.66291667 |
| old_only | zfft_duplicate.h | 0 | N/A |
| old_only | zfft_duplicate.macro_accuracy | 720 | 0.66291667 |
| old_only | zfft_duplicate.macro_nll | 720 | 1.5942228 |
| old_only | zfft_duplicate.new_accuracy | 0 | N/A |
| old_only | zfft_duplicate.old_accuracy | 720 | 0.66291667 |
| old_only | z.aux_minus_base.accuracy | 720 | 0.038206019 |
| old_only | z.aux_minus_base.h | 0 | N/A |
| old_only | z.aux_minus_base.macro_accuracy | 720 | 0.038206019 |
| old_only | z.aux_minus_base.macro_nll | 720 | -0.027896112 |
| old_only | z.aux_minus_base.new_accuracy | 0 | N/A |
| old_only | z.aux_minus_base.old_accuracy | 720 | 0.038206019 |
| old_only | z.aux_minus_duplicate.accuracy | 720 | 0.034884259 |
| old_only | z.aux_minus_duplicate.h | 0 | N/A |
| old_only | z.aux_minus_duplicate.macro_accuracy | 720 | 0.034884259 |
| old_only | z.aux_minus_duplicate.macro_nll | 720 | -0.0005982243 |
| old_only | z.aux_minus_duplicate.new_accuracy | 0 | N/A |
| old_only | z.aux_minus_duplicate.old_accuracy | 720 | 0.034884259 |
| old_only | z.duplicate_minus_base.accuracy | 720 | 0.0033217593 |
| old_only | z.duplicate_minus_base.h | 0 | N/A |
| old_only | z.duplicate_minus_base.macro_accuracy | 720 | 0.0033217593 |
| old_only | z.duplicate_minus_base.macro_nll | 720 | -0.027297887 |
| old_only | z.duplicate_minus_base.new_accuracy | 0 | N/A |
| old_only | z.duplicate_minus_base.old_accuracy | 720 | 0.0033217593 |
| old_only | zfft.aux_minus_base.accuracy | 720 | 0.034444444 |
| old_only | zfft.aux_minus_base.h | 0 | N/A |
| old_only | zfft.aux_minus_base.macro_accuracy | 720 | 0.034444444 |
| old_only | zfft.aux_minus_base.macro_nll | 720 | -0.097856429 |
| old_only | zfft.aux_minus_base.new_accuracy | 0 | N/A |
| old_only | zfft.aux_minus_base.old_accuracy | 720 | 0.034444444 |
| old_only | zfft.aux_minus_duplicate.accuracy | 720 | 0.022743056 |
| old_only | zfft.aux_minus_duplicate.h | 0 | N/A |
| old_only | zfft.aux_minus_duplicate.macro_accuracy | 720 | 0.022743056 |
| old_only | zfft.aux_minus_duplicate.macro_nll | 720 | -0.043531506 |
| old_only | zfft.aux_minus_duplicate.new_accuracy | 0 | N/A |
| old_only | zfft.aux_minus_duplicate.old_accuracy | 720 | 0.022743056 |
| old_only | zfft.duplicate_minus_base.accuracy | 720 | 0.011701389 |
| old_only | zfft.duplicate_minus_base.h | 0 | N/A |
| old_only | zfft.duplicate_minus_base.macro_accuracy | 720 | 0.011701389 |
| old_only | zfft.duplicate_minus_base.macro_nll | 720 | -0.054324923 |
| old_only | zfft.duplicate_minus_base.new_accuracy | 0 | N/A |
| old_only | zfft.duplicate_minus_base.old_accuracy | 720 | 0.011701389 |

## Interpretation and costs

- Means are descriptive equal-episode aggregates; overlapping support draws and model/cohort reuse are correlated, not independent samples.
- No confidence intervals or p-values treat support draws as independent; no query performance claim.
- Old-only and new-present populations are separate; undefined new/H metrics remain null.
- Auxiliary classification increments can reflect changed regularization; duplicate controls and held reconstruction do not prove causal TX information.
- Numerical rank is arithmetic structure, not held prediction; K1 has no independent holdout.
- Completion/access/immutability checks audit recorded metadata, not a second data validation or independent execution replay.
- K1 trace omits physical IDs: physical binding was checked by the evaluator; this summary checks counts/nulls and producer bindings.
- Process-time sums are work totals, not wall-clock elapsed; RSS maxima are per-process high-water marks, not concurrent aggregate memory.
- Model package bytes are existing full checkpoint storage, not inferred new transmission; deployment status remains unknown.
- Raw trace/cache artifacts were preserved; cache/IQ arrays and query/source/score/index files were not read.

All measured extraction, fit, score, logging, cache/model byte and RSS statistics are in summary.json; fold loss/gradient minima, means and maxima are also in fit_stage_statistics.csv.
Stratified CSVs cover K, model seed, support seed, cohort, receiver×scenario and K×new-count. Missing conditional metrics remain empty; no row was selected by its diagnostic outcome.
