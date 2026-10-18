# Fixed FINAL50 own EMA source comparison

One immutable source-only run compares `model` and existing `ema_model` from
the original seed01 B00/B01 FINAL50 checkpoints. The pinned E50 bootstrap and
its unmodified `own_model` enforce the entire parent/source contract. Both
current states additionally require all 206 finite keys, shapes, dtypes and
the existing default 6×160 cosine head. There is no training or EMA update.

The first step is a real-checkpoint no-query 4-sample smoke. On PASS the same
process immediately evaluates all 27000 V packets with four readouts. It
reuses the original student V1800 panel predictions and evaluates each EMA on
that panel's clean and two draws of three fixed source observations.
Expected new network sample forwards: 108000 + 25200 + 4 = 133204.
Source IQ/checkpoint reads are counted separately; historical training and
student panel inference are referenced, not charged again.

`runner.py` accepts no model, epoch, EMA coefficient or seed arguments.
It reads private `runtime.json` at release root. Run `--checks` first for
synthetic protocol negatives. Runtime source evaluation writes `smoke.json`,
`predictions.npz`, `summary.json`, `metrics.csv`, `fullV_confusions.json`,
`paired_errors.json`, `costs.json`, four `readouts/*/summary.json`, and finally
`complete.json`. Technical failure preserves `failure.json` and all outputs;
there is no automatic retry or formal prediction path.

After the source-only preregistration and actual code/config commit, the root
owner can release with `python -m experiments.cvs_practical_ema_source.publish`.
It reads private `local_artifacts/F_ownEMA_source_E50_20261018/runtime.json`,
uses one ordinary-account preflight, one archive SHA transfer comparison,
remote compile and focused checks, then starts one process on its fixed GPU.
The owner independently checks the submitted PID/CWD/argv/GPU and log growth.

The source qualification has exactly three scientific criteria: AsEMA versus
NoAsEMA full-V clean change ≥ −0.1 absolute pp; AsEMA versus own student full-V
clean net repairs ≥ 0; and AsEMA versus own student equal-weight mid/low urban
combined net repairs > 0. Scene/draw results and difference interaction are
reported without further cell gates. A failed source qualification still
writes every complete output and does not trigger another seed or EMA choice.
