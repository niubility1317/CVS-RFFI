# receiver_agnostic_repro_optimization_20260708_2240

This is the Git-backed handoff mirror for the local N607 experiment report:

`E:\type10-7\automation_reports\CV-SincNet\receiver_agnostic_repro_optimization_20260708_2240\report.md`

Scope: Bao et al. receiver-agnostic two-stage UDA WiSig/ManySig paper-faithful closed-set reproduction diagnostics. This is not CVS Stage2-C, satellite/LEO deployment, open-set, or new-class registration evidence.

## Local Version

| field | value |
|---|---|
| code commit | `a1bf8c8 feat: add receiver agnostic reproduction diagnostics` |
| local verification | `C:\Users\lh594\.conda\envs\ssr-gpu\python.exe -m pytest tests/test_receiver_agnostic_twostage_uda.py -q` -> 20 passed |
| remote verification | N607 `py_compile` PASS; CLI flags confirmed |
| sync | changed code/tests synced to `/home/szu2070436088/2510044040/CV-SincNet` |

## Runs

| run | status | GPU | PID | objective |
|---|---|---|---:|---|
| `receiver_agnostic_r6_replace_20_1_3_19_20260708_2240` | completed | `cuda:4` | `892055` | replace weak target receivers `20-1` and `3-19` |
| `receiver_agnostic_rspecific_confdiag_r1234_20260708_2240` | running | `cuda:5` | `893123` | R-specific LMMD, confidence threshold, temperature, detach target probabilities, source balance, GRL schedule |
| `receiver_agnostic_fig8_balanced_classifier_r123_20260708_2240` | running | `cuda:6` | `894243` | Fig8 class-balanced target labels and classifier-only fine-tuning |

## Interim Result

Run A completed with target accuracy `0.818451`, below the paper Table I R=6 reference `0.9083` by `0.089849`.

| target receiver | accuracy |
|---|---:|
| `18-2` | `0.478333` |
| `19-2` | `0.874792` |
| `2-19` | `0.929583` |
| `7-14` | `0.896000` |
| `7-7` | `0.859625` |
| `8-8` | `0.872375` |

The replacement did not improve R=6 because `18-2` became the dominant low point. The next receiver replacement pass should avoid treating `18-2` as paper-comparable unless the paper receiver mapping can be verified.

## Pending

Monitor B and C until `formal_training_summary` appears, then update the local report with final tables and compare R=1/2/3/4 and Fig8 curves against the previous reproduction and the paper.
