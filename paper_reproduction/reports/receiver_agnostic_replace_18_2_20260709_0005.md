# receiver_agnostic_replace_18_2_20260709_0005

Git-backed mirror for:

`E:\type10-7\automation_reports\CV-SincNet\receiver_agnostic_replace_18_2_20260709_0005\report.md`

Scope: Bao et al. receiver-agnostic two-stage UDA WiSig/ManySig paper-faithful closed-set reproduction diagnostics. This is not CVS Stage2-C, satellite/LEO deployment, open-set, or new-class registration evidence.

## Purpose

Previous R=6 replacement made `18-2` the dominant weak target receiver (`0.478333`). Previous Fig8 rows also showed `18-2` as a persistent low point. This run removes `18-2` from target-side evaluation for both R=6 and Fig8 diagnostics.

## Planned Runs

| run | GPU | source receivers | target receivers |
|---|---|---|---|
| `receiver_agnostic_r6_replace_18_2_with_2_1_20260709_0005` | `cuda:0` | inferred complement `1-1,1-19,14-7,18-2,20-1,3-19` | `19-2,2-1,2-19,7-14,7-7,8-8` |
| `receiver_agnostic_fig8_no18_r1_20260709_0005` | `cuda:1` | `18-2` | all other 11 receivers |
| `receiver_agnostic_fig8_no18_r2_20260709_0005` | `cuda:2` | `1-1,18-2` | all other 10 receivers |
| `receiver_agnostic_fig8_no18_r3_20260709_0005` | `cuda:3` | `1-1,1-19,18-2` | all other 9 receivers |

Fig8 is split into three runs because the runner accepts one explicit source/target receiver split per invocation. Moving `18-2` to source preserves the R=1:11, R=2:10, and R=3:9 receiver-count ratios while removing `18-2` from target evaluation.

## Local State

No code changes are introduced by this report. Current working tree had unrelated dirty files under `code/` and unrelated untracked `local_artifacts/` directories before this report was added.

## Launch

Launched on N607 at `2026-07-09 00:05-00:06 CST`.

| run | PID | GPU |
|---|---:|---|
| `receiver_agnostic_r6_replace_18_2_with_2_1_20260709_0005` | `930981` | `cuda:0` |
| `receiver_agnostic_fig8_no18_r1_20260709_0005` | `930983` | `cuda:1` |
| `receiver_agnostic_fig8_no18_r2_20260709_0005` | `930985` | `cuda:2` |
| `receiver_agnostic_fig8_no18_r3_20260709_0005` | `930987` | `cuda:3` |

The first launch attempt did not land any matching remote process because local PowerShell expanded remote shell variables. The successful launch used a base64-encoded remote bash script.

## Startup Health

All four runs were confirmed active. The explicit receiver splits are correct and `18-2` is absent from every target set:

| run | PID | latest observed progress |
|---|---:|---|
| R=6 replacement | `930981` | stage1 step `500/2000` |
| Fig8 R=1 no18 | `930983` | stage1 step `500/1880` |
| Fig8 R=2 no18 | `930985` | stage1 step `500/2000` |
| Fig8 R=3 no18 | `930987` | stage1 step `500/2000` |

No startup `Traceback`, argparse failure, OOM, or empty-target split was observed.
