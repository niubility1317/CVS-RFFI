# receiver_agnostic_fig8_classrx_opt_20260709_105935

Git-backed mirror for:

`E:\type10-7\automation_reports\CV-SincNet\receiver_agnostic_fig8_classrx_opt_20260709_105935\report.md`

Scope: Bao et al. receiver-agnostic two-stage UDA WiSig/ManySig closed-set Fig8 optimization. This is not CVS Stage2-C, satellite/LEO deployment, open-set, or new-class registration evidence.

## Objective

Optimize the incomplete Fig8 reproduction by testing whether joint class/receiver-balanced target labels and receiver replacement diagnostics improve absolute target accuracy while preserving the paper's fine-tuning trend.

## Local Change

Commit `351c155 feat: add joint Fig8 target balance diagnostics` adds:

| file | purpose |
|---|---|
| `sampling.py` | `class_receiver` target label selection mode |
| `steps.py` | selected label/receiver/pair diagnostics |
| `train.py` | CLI `--fig8-target-balance class_receiver` and Fig8 diagnostics output |
| `tests/test_receiver_agnostic_twostage_uda.py` | regression coverage |

Verification: `C:\Users\lh594\.conda\envs\ssr-gpu\python.exe -m pytest tests/test_receiver_agnostic_twostage_uda.py -q` -> `20 passed`.

## Planned Runs

| run | GPU | purpose |
|---|---|---|
| `receiver_agnostic_fig8_classrx_strict_r123_20260709_105935` | `cuda:3` | strict deterministic R=1/2/3 with `class_receiver` target labels |
| `receiver_agnostic_fig8_move_3_19_r1_20260709_105935` | `cuda:4` | R=1 sensitivity: move `3-19` to source, target all other receivers |
| `receiver_agnostic_fig8_move_18_2_3_19_r2_20260709_105935` | `cuda:5` | R=2 sensitivity: move `18-2,3-19` to source |
| `receiver_agnostic_fig8_move_18_2_3_19_r3_20260709_105935` | `cuda:6` | R=3 sensitivity: move `1-1,18-2,3-19` to source |

Key settings: full fine-tune, lr `0.0005`, source replay per class `20`, Fig8 strategies `random,entropy,margin,least_confidence`, iterations `0,25,50,75,100`, R-specific LMMD for R=3 where applicable.

## Comparison Targets

| baseline | result |
|---|---|
| strict tuned Fig8 | R=1 random@100 `72.93%`, R=2 `79.81%`, R=3 `78.16%` |
| no-`18-2` target diagnostic | R=1 random@100 `78.43%`, R=2 `84.53%`, R=3 `86.64%` |
| paper Fig8 | only visual read available; R=1:11 at 100 iterations roughly `~93%` |

## Status

Remote sync and verification completed. `py_compile` passed and the remote CLI exposes `--fig8-target-balance {none,class,receiver,class_receiver}`.

Launched at `2026-07-09 11:00 CST`:

| run | PID | GPU |
|---|---:|---|
| `receiver_agnostic_fig8_classrx_strict_r123_20260709_105935` | `1221716` | `cuda:3` |
| `receiver_agnostic_fig8_move_3_19_r1_20260709_105935` | `1221718` | `cuda:4` |
| `receiver_agnostic_fig8_move_18_2_3_19_r2_20260709_105935` | `1221720` | `cuda:5` |
| `receiver_agnostic_fig8_move_18_2_3_19_r3_20260709_105935` | `1221722` | `cuda:6` |

Full local report will track startup health, final results, and selected target diagnostics.

Startup health passed at about `2026-07-09 11:01 CST`: all four PIDs were active on GPUs 3-6, each log had written the intended receiver split and reached stage1 step 500, and no `Traceback`, argparse/unrecognized argument error, OOM, or empty-target split was observed.

## Final Results

Runs completed by about `2026-07-09 11:47 CST`. Local evidence was copied to:

`E:\type10-7\local_artifacts\receiver_agnostic_fig8_classrx_opt_20260709_105935\`

### Strict Split

Strict split keeps the deterministic source receiver order used by the current reproduction code. Because this round uses `class_receiver` target-label balancing and tuned Fig8/LMMD settings not explicitly specified by the paper, this table is an optimized strict-split diagnostic, not a fully paper-faithful headline.

| R | source receivers | base | random@100 | best active@100 | weakest target receivers at random@100 |
|---:|---|---:|---:|---:|---|
| 1 | `1-1` | `56.22%` | `80.50%` | margin `78.09%` | `3-19` `52.28%`; `18-2` `55.87%`; `20-1` `73.33%` |
| 2 | `1-1,1-19` | `64.50%` | `83.35%` | margin `83.61%` | `18-2` `59.90%`; `3-19` `74.10%`; `2-1` `83.28%` |
| 3 | `1-1,1-19,14-7` | `62.46%` | `88.11%` | random `88.11%` | `18-2` `66.96%`; `3-19` `74.58%`; `2-1` `88.31%` |

The selected target labels were exactly balanced: R=1 labels each `880` and receivers each `480`; R=2 labels each `800` and receivers each `480`; R=3 labels each `720` and receivers each `480`.

### Receiver-Replacement Diagnostics

| run | source receivers | base | random@100 | weakest target receivers at random@100 | verdict |
|---|---|---:|---:|---|---|
| R=1 move `3-19` | `3-19` | `52.59%` | `92.86%` | `18-2` `78.27%`; `7-7` `91.48%`; `2-1` `92.96%` | matches paper-level R=1 visual read, diagnostic only |
| R=2 move `18-2,3-19` | `18-2,3-19` | `53.58%` | `96.10%` | `7-7` `91.84%`; `1-1` `93.42%`; `2-1` `95.65%` | confirms hard-receiver bottleneck |
| R=3 move `18-2,3-19` | `1-1,18-2,3-19` | `66.94%` | `98.03%` | `7-7` `96.83%`; `19-2` `97.05%`; `1-19` `97.15%` | near-saturated Fig8 curve, diagnostic only |

### Delta Versus Earlier Runs

| line | R=1 random@100 | R=2 random@100 | R=3 random@100 |
|---|---:|---:|---:|
| previous strict tuned | `72.93%` | `79.81%` | `78.16%` |
| previous no-`18-2` diagnostic | `78.43%` | `84.53%` | `86.64%` |
| this strict class_receiver | `80.50%` | `83.35%` | `88.11%` |
| this best replacement | `92.86%` | `96.10%` | `98.03%` |

### Diagnosis and Boundary

The previous low absolute Fig8 was partly caused by target label selection. Joint class/receiver balancing improved the strict line while proving that every target receiver and every class is covered evenly during fine-tuning.

The remaining strict gap is dominated by target receiver composition. Even with balanced labels, `3-19` and `18-2` remain the weakest strict target receivers. Moving `3-19` into the R=1 source raises random@100 to `92.86%`, and moving both `18-2` and `3-19` into source raises R=2/R=3 to `96.10%`/`98.03%`.

Therefore the implementation can reach paper-level Fig8 values, but the replacement lines cannot be claimed as strict paper-faithful unless the original paper's hidden receiver ordering is recovered or justified. The strict current line should be reported as improved but still below paper in absolute value, especially for R=1.

### Method-Faithfulness Audit

Independent method review found that this run family should be reported as `Fig8 optimized diagnostic/sensitivity`, not as the strict paper-faithful headline:

| item | current implementation | boundary |
|---|---|---|
| target label selection | `class_receiver` uses target true class and receiver metadata to enforce joint quotas | useful diagnostic, but not explicitly described in Fig8 |
| R-specific LMMD | R=3 uses tuned `features_and_activations`, `lambda=0.01`, `250` steps, `lr=0.0001` | optimization choice, not confirmed paper protocol |
| Fig8 fine-tune | full fine-tune, `finetune_lr=0.0005`, source replay per class `20` | paper does not fully specify lr, replay, or scope |
| receiver replacement | manual source receiver changes such as `3-19` or `18-2,3-19` | receiver-order sensitivity only unless original paper order is recovered |

Future strict paper-faithful runs should keep a declared receiver split, avoid true-label joint quotas unless the paper split protocol is recovered, avoid R-specific tuning, and separate those results from the optimized diagnostic table above.
