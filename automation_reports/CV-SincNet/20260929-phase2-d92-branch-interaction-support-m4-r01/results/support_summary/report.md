# Support-only branch interaction diagnostic

Verified 4800 episodes: 1200 K1 numerical-only, 3600 OOF, 32400 factorizations.

Descriptive equal-episode means. Deltas are fractions; multiply by 100 for percentage points.

| Population | Metric | N | Mean |
|---|---|---:|---:|
| allold | energy_control.accuracy | 720 | 0.69130787 |
| allold | energy_control.h | 0 | N/A |
| allold | energy_control.macro_accuracy | 720 | 0.69130787 |
| allold | energy_control.macro_nll | 720 | 1.5223583 |
| allold | energy_control.new_accuracy | 0 | N/A |
| allold | energy_control.old_accuracy | 720 | 0.69130787 |
| allold | interaction.accuracy | 720 | 0.69396991 |
| allold | interaction.h | 0 | N/A |
| allold | interaction.macro_accuracy | 720 | 0.69396991 |
| allold | interaction.macro_nll | 720 | 1.5097882 |
| allold | interaction.new_accuracy | 0 | N/A |
| allold | interaction.old_accuracy | 720 | 0.69396991 |
| allold | linear.accuracy | 720 | 0.68565972 |
| allold | linear.h | 0 | N/A |
| allold | linear.macro_accuracy | 720 | 0.68565972 |
| allold | linear.macro_nll | 720 | 1.5506913 |
| allold | linear.new_accuracy | 0 | N/A |
| allold | linear.old_accuracy | 720 | 0.68565972 |
| allold | energy_control_minus_linear.accuracy | 720 | 0.0056481481 |
| allold | energy_control_minus_linear.h | 0 | N/A |
| allold | energy_control_minus_linear.macro_accuracy | 720 | 0.0056481481 |
| allold | energy_control_minus_linear.macro_nll | 720 | -0.02833301 |
| allold | energy_control_minus_linear.new_accuracy | 0 | N/A |
| allold | energy_control_minus_linear.old_accuracy | 720 | 0.0056481481 |
| allold | interaction_minus_energy_control.accuracy | 720 | 0.002662037 |
| allold | interaction_minus_energy_control.h | 0 | N/A |
| allold | interaction_minus_energy_control.macro_accuracy | 720 | 0.002662037 |
| allold | interaction_minus_energy_control.macro_nll | 720 | -0.012570111 |
| allold | interaction_minus_energy_control.new_accuracy | 0 | N/A |
| allold | interaction_minus_energy_control.old_accuracy | 720 | 0.002662037 |
| allold | interaction_minus_linear.accuracy | 720 | 0.0083101852 |
| allold | interaction_minus_linear.h | 0 | N/A |
| allold | interaction_minus_linear.macro_accuracy | 720 | 0.0083101852 |
| allold | interaction_minus_linear.macro_nll | 720 | -0.040903121 |
| allold | interaction_minus_linear.new_accuracy | 0 | N/A |
| allold | interaction_minus_linear.old_accuracy | 720 | 0.0083101852 |
| new_present | energy_control.accuracy | 2880 | 0.52912531 |
| new_present | energy_control.h | 2880 | 0.49968125 |
| new_present | energy_control.macro_accuracy | 2880 | 0.52912531 |
| new_present | energy_control.macro_nll | 2880 | 2.4242514 |
| new_present | energy_control.new_accuracy | 2880 | 0.45117014 |
| new_present | energy_control.old_accuracy | 2880 | 0.59590278 |
| new_present | interaction.accuracy | 2880 | 0.53737903 |
| new_present | interaction.h | 2880 | 0.5097262 |
| new_present | interaction.macro_accuracy | 2880 | 0.53737903 |
| new_present | interaction.macro_nll | 2880 | 2.4095602 |
| new_present | interaction.new_accuracy | 2880 | 0.46311198 |
| new_present | interaction.old_accuracy | 2880 | 0.60062211 |
| new_present | linear.accuracy | 2880 | 0.52336779 |
| new_present | linear.h | 2880 | 0.49228334 |
| new_present | linear.macro_accuracy | 2880 | 0.52336779 |
| new_present | linear.macro_nll | 2880 | 2.4463003 |
| new_present | linear.new_accuracy | 2880 | 0.4423559 |
| new_present | linear.old_accuracy | 2880 | 0.59297164 |
| new_present | energy_control_minus_linear.accuracy | 2880 | 0.0057575151 |
| new_present | energy_control_minus_linear.h | 2880 | 0.0073979131 |
| new_present | energy_control_minus_linear.macro_accuracy | 2880 | 0.0057575151 |
| new_present | energy_control_minus_linear.macro_nll | 2880 | -0.022048877 |
| new_present | energy_control_minus_linear.new_accuracy | 2880 | 0.0088142361 |
| new_present | energy_control_minus_linear.old_accuracy | 2880 | 0.0029311343 |
| new_present | interaction_minus_energy_control.accuracy | 2880 | 0.0082537287 |
| new_present | interaction_minus_energy_control.h | 2880 | 0.010044949 |
| new_present | interaction_minus_energy_control.macro_accuracy | 2880 | 0.0082537287 |
| new_present | interaction_minus_energy_control.macro_nll | 2880 | -0.014691214 |
| new_present | interaction_minus_energy_control.new_accuracy | 2880 | 0.01194184 |
| new_present | interaction_minus_energy_control.old_accuracy | 2880 | 0.0047193287 |
| new_present | interaction_minus_linear.accuracy | 2880 | 0.014011244 |
| new_present | interaction_minus_linear.h | 2880 | 0.017442862 |
| new_present | interaction_minus_linear.macro_accuracy | 2880 | 0.014011244 |
| new_present | interaction_minus_linear.macro_nll | 2880 | -0.036740091 |
| new_present | interaction_minus_linear.new_accuracy | 2880 | 0.020756076 |
| new_present | interaction_minus_linear.old_accuracy | 2880 | 0.007650463 |

- Equal-episode descriptive means; repeated and overlapping support draws are correlated. No independent-sample confidence interval or p-value is claimed.
- Allold and new-present populations are separate; undefined new/H metrics remain null. Paired deltas compare the same physical OOF rows and episodes.
- K1 has numerical diagnostics only; no independent held prediction and no classifier fitting.
- All three predetermined kernels are reported. No automatic arm selection, query evaluation, or deployment is performed.
- Extraction and checkpoint reuse cost zero new forwards/transfers in this run; cache loading and analytical fitting costs are measured separately.
- Process time sums are work totals; RSS values are per-process high-water marks, not aggregate concurrent memory.
- Only current run traces/metadata and explicitly bound cache completion metadata were read; no IQ, cache arrays, historical result, or query/truth file was read.

Support-only candidate screen: PASS.
For each of K5/10/20 separately on joint new-present tasks, mean H and new accuracy strictly improve versus both controls, and old accuracy delta is at least -0.01. Cross-K means are descriptive only.
This screen does not authorize automatic promotion or establish query performance.
