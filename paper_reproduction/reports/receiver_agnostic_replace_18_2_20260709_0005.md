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

## Interim Results

`receiver_agnostic_r6_replace_18_2_with_2_1_20260709_0005` completed on N607.

| target receiver | accuracy | correct / total |
|---|---:|---:|
| `19-2` | `0.875333` | `21008 / 24000` |
| `2-1` | `0.834542` | `20029 / 24000` |
| `2-19` | `0.667625` | `16023 / 24000` |
| `7-14` | `0.908167` | `21796 / 24000` |
| `7-7` | `0.827208` | `19853 / 24000` |
| `8-8` | `0.808208` | `19397 / 24000` |
| mean | `0.820181` | `118106 / 144000` |

This is only `+0.001730` above the previous R=6 replacement mean `0.818451` and remains `-0.088153` below the paper Table I R=6 mean `0.908333`. The weak point moved from `18-2` to `2-19=0.667625`, so replacing `18-2` alone does not solve the R=6 gap.

The three Fig8 no-`18-2` target runs completed by `2026-07-09 00:23 CST`.

| R | source receivers | target count | base | random@100 | entropy@100 | margin@100 | least_conf@100 | best@100 |
|---:|---|---:|---:|---:|---:|---:|---:|---:|
| 1 | `18-2` | 11 | `0.611799` | `0.784295` | `0.537572` | `0.710617` | `0.587193` | `0.784295` |
| 2 | `1-1,18-2` | 10 | `0.738112` | `0.845325` | `0.730646` | `0.776046` | `0.759387` | `0.845325` |
| 3 | `1-1,1-19,18-2` | 9 | `0.807250` | `0.866356` | `0.817778` | `0.839875` | `0.833583` | `0.866356` |

Compared with the previous Fig8 run, current R=1 improves from base `0.562693` to `0.611799` and random@100 from `0.666939` to `0.784295`; R=2 base improves from `0.658425` to `0.738112`; R=3 improves from base `0.615611` to `0.807250` and random@100 from `0.732028` to `0.866356`.

Residual weak receivers after the best random@100 strategy are `3-19=0.631417` for R=1, `3-19=0.686958` for R=2, and `3-19=0.773083` for R=3. This confirms that removing `18-2` from target-side Fig8 materially improves the diagnostic curves, but it is a receiver-replacement sensitivity result rather than a strict paper-faithful Fig8 reproduction because `18-2` was moved into the source set to preserve receiver-count ratios.
