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
| `receiver_agnostic_fig8_move_18_2_3_19_r23_20260709_105935` | `cuda:5` | R=2/3 sensitivity: move `18-2,3-19` to source |

Key settings: full fine-tune, lr `0.0005`, source replay per class `20`, Fig8 strategies `random,entropy,margin,least_confidence`, iterations `0,25,50,75,100`, R-specific LMMD for R=3 where applicable.

## Comparison Targets

| baseline | result |
|---|---|
| strict tuned Fig8 | R=1 random@100 `72.93%`, R=2 `79.81%`, R=3 `78.16%` |
| no-`18-2` target diagnostic | R=1 random@100 `78.43%`, R=2 `84.53%`, R=3 `86.64%` |
| paper Fig8 | only visual read available; R=1:11 at 100 iterations roughly `~93%` |

## Status

Created before sync/launch. Full local report will track exact N607 commands, PIDs, startup health, final results, and selected target diagnostics.
