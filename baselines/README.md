# Paper Baselines

This package adds isolated paper reproduction baselines for CVS-RFFI experiments.

## Methods

- `riei_fd`: receiver-independent emitter identification via feature disentanglement; this is the method with CE, mutual-independence (MI), and information-entropy (IE) losses.
- `drift`: cross-receiver feature disentanglement with GRL, receiver center loss, and negative MSE separation.
- `ra_collab`: receiver-agnostic and collaborative RFFI; this is the GRL/CIS-style method with fine-tuning and soft/adaptive fusion, not the MI/IE feature-disentanglement method.
- `cvcnn_ce`: a plain complex-valued CNN baseline trained only with cross entropy.

## CVS-RFFI Training Commands

These entrypoints reuse the root `dataset_wisig.py` split used by CVS-RFFI:

- train/validation from `wisig_train_days x wisig_train_rxs`
- validation is the contiguous tail after `wisig_guard_gap`
- tests are the named CVS-RFFI OOD subsets:
  `test_unseen_day_seen_rx`, `test_seen_day_unseen_rx`, and
  `test_unseen_day_unseen_rx`
- named tests run when each method's validation criterion improves, and a final
  post-training named-test pass is written to `metrics.json["final"]`; most
  methods use transmitter accuracy, while paper baselines that specify
  validation-loss scheduling use validation loss.

Default split parameters mirror the current CVS-RFFI defaults:
`--wisig_train_ratio 0.2 --wisig_guard_gap 8 --wisig_train_days 0,1 --wisig_test_days 2,3 --wisig_train_rxs 0,1,2,3,4,5,6 --wisig_test_rxs 7,8,9,10,11`.

```bash
python -u -m baselines.cvcnn_ce.train --wisig_pkl ./Dataset_WigSig/ManySig.pkl
python -u -m baselines.riei_fd.train --wisig_pkl ./Dataset_WigSig/ManySig.pkl
python -u -m baselines.drift.train --wisig_pkl ./Dataset_WigSig/ManySig.pkl
python -u -m baselines.ra_collab.train --wisig_pkl ./Dataset_WigSig/ManySig.pkl
python -u -m baselines.ra_collab.finetune_cvs --checkpoint baseline_runs/ra_collab/best_by_val.pt --wisig_pkl ./Dataset_WigSig/ManySig.pkl
```

On the training server, activate the intended Conda environment first and prefer
`python` from that environment over a system `python3`:

```bash
conda activate ssr-gpu
python -u -m baselines.ra_collab.train --wisig_pkl ./Dataset_WigSig/ManySig.pkl
```

The method-local file entrypoints also bootstrap the repository root, so
`python baselines/ra_collab/train.py --help` is supported for quick
server sanity checks.

For queued multi-GPU runs, use the repo-level launcher. It starts methods in
parallel, keeps the selected WiSig protocol fixed across methods, records a
manifest TSV, and leaves paper-stated training hyperparameters at each method's
own defaults. Methods without paper-stated epoch counts use 200 epochs; CVCNN
uses 200 epochs.

The default queue uses the CVS day/receiver split with the 10% train / 90%
validation split used in the latest WISIG setup:
`--wisig_train_ratio 0.1 --wisig_val_ratio 0.9 --wisig_guard_gap 8 --seed 1337 --wisig_train_days 0,1 --wisig_test_days 2,3 --wisig_train_rxs 0,1,2,3,4,5,6 --wisig_test_rxs 7,8,9,10,11`.
Training-time satellite-channel view augmentation is default-off in the public
launcher. Enable it explicitly with `--sat-view-aug` and record the chosen
scenario as a stress/control or CVS extension condition, not as real in-orbit
evidence.

```bash
METHODS=cvcnn_ce,riei_fd,drift,ra_collab \
GPU_IDS=0,1,2,3 \
nohup bash scripts/launchers/run_cvs_baseline_queue.sh > logs/wisig_baselines_seed1337_ratio010_satview/nohup_$(date +%Y%m%d_%H%M%S).out 2>&1 &
```

The launcher enables CVS-RFFI-style satellite-channel OOD evaluation by
default and writes the exact command for each method to
`runs/.../manifest_<timestamp>.tsv`:

```bash
--eval_sat_channel
--eval_sat_on main
--eval_sat_scenarios clear_leo,low_elev_leo,rain_leo,storm_mp,mixed_orbit
```

This reports the three main named dimensions:
`test_unseen_day_seen_rx`, `test_seen_day_unseen_rx`, and
`test_unseen_day_unseen_rx`, plus satellite-channel aggregates for the same
dimensions.

## Paper-Audit Experiment Protocols

`baselines/PAPER_CODE_AUDIT.md` identifies two formal paper-alignment checks
that require real WiSig/ManySig data and long training rather than synthetic
smoke tests. Dry-run them first to inspect the exact commands:

```bash
bash scripts/launchers/run_cvs_baseline_queue.sh \
  --methods riei_fd \
  --wisig-protocol riei_original \
  --gpu-ids 0 \
  --dry-run

bash scripts/launchers/run_cvs_baseline_queue.sh \
  --methods drift \
  --wisig-protocol drift_day1 \
  --gpu-ids 1 \
  --dry-run

bash baselines/scripts/run_riei_original_table3_queue.sh --dry-run
```

When the dataset path and GPU allocation are verified, remove `--dry-run`.
The RIEI protocol uses receiver-holdout settings and writes
`paper_eval_window.name=riei_last10` under `metrics.json["final"]`; the DRIFT
Day1 protocol uses source/test receivers from Day 1 and writes
`paper_eval_window.name=drift_last5`. Both commands also preserve the
same-row named-test context under `metrics.json["final"]["test_named"]`.

For a CVS-aligned comparison across all four baselines, keep the default
`cvs_day_rx` protocol:

```bash
METHODS=cvcnn_ce,riei_fd,drift,ra_collab \
GPU_IDS=0,1,2,3 \
bash scripts/launchers/run_cvs_baseline_queue.sh --wisig-protocol cvs_day_rx --dry-run
```

These commands set up experiments; they do not by themselves constitute
deployment success. Any reported result must name the run directory, split,
protocol, seed, satellite/stress view, and full metric row.

## LEO Satellite-Channel View Augmentation

All CVS baseline comparison trainers can add a satellite-ground channel view
during training. The repo-level launcher and individual trainer entrypoints
require the switch explicitly:

```bash
python -u -m baselines.cvcnn_ce.train \
  --wisig_pkl ./Dataset_WigSig/ManySig.pkl \
  --use_sat_channel_view_aug \
  --sat_train_scenario clear_leo
```

For supervised baselines, each training batch is expanded with a matching LEO
satellite-channel view and duplicated transmitter/receiver labels.

Supported training-view switches:

- `--use_sat_channel_view_aug`: enable training-time satellite-channel views.
- `--sat_train_scenario`: single training scenario, default `clear_leo`.
- `--sat_train_scenarios`: comma-separated scenario cycle for ablations, e.g.
  `clear_leo,low_elev_leo,rain_leo`.
- `--sat_view_prob`: probability of applying the satellite view transform.
- `--sat_view_seed`: deterministic generator seed for the satellite view.

Paper-specific defaults used by the CVS entrypoints:

- `riei_fd`: FED feature split, EC/RC classifiers, signed cosine MI loss, information-entropy confusion loss, alternating classifier/FED updates.
- DRIFT: 1D ResNet-18-style encoder, transmitter/receiver feature split, GRL receiver discriminator, receiver style center regularization, negative MSE feature separation, Adam `lr=1e-4`, `lambda_grl=1`, `lambda_center=0.01`, `lambda_mse=0.02`, `batch_size=64`.
- `ra_collab`: spectrogram/CNN adversarial receiver training with GRL, SGD momentum `0.9`, `lr=1e-3`, `batch_size=64`, validation-loss LR decay factor `0.2` after 10 stagnant epochs, early stop after 20 stagnant epochs; formal CVS tests use receiver collaborative fusion. Few-shot target receiver fine-tuning uses `lr=1e-5`, `batch_size=32`, `epochs=20`.
- `cvcnn_ce`: `epochs=200`; no paper auxiliary loss; only cross entropy on CVS-RFFI training data.

| Method | Paper-aligned structure | Default optimizer / schedule | Paper-window metric |
|---|---|---|---|
| `riei_fd` | ResNet1D-18 FED, EC/RC 3-layer classifiers, `z_e/z_r` split, CE + MI - IE alternating updates | Adam for all parameters and FED-only step, `lr_all=1e-4`, `lr_fed=1e-4`, `lambda_mi=1.2`, `lambda_ie=1.2`, `epochs=200` | `riei_last10` for `riei_original` |
| `drift` | ResNet18-1D encoder, TX/RX split, GRL, receiver center loss, raw negative MSE separation | Adam `lr=1e-4`, `lambda_grl=1.0`, `lambda_center=0.01`, `lambda_mse=0.02`, constant GRL by default, `epochs=200` | `drift_last5` for `drift_day1` |
| `ra_collab` | Spectrogram CNN, GRL receiver adversary, OBS and collaborative fusion evaluation | SGD `lr=1e-3`, momentum `0.9`, validation-loss plateau factor `0.2`, patience `10`, early stop `20`, fine-tune `lr=1e-5` | `aligned_wisig_last5` for CVS comparisons |
| `cvcnn_ce` | Complex CNN or optional Sinc stem with CE-only objective | AdamW `lr=2e-4`, cosine annealing to `1e-6`, weight decay `1e-4`, `epochs=200` | `aligned_wisig_last5` for CVS comparisons |

## Opt-in Pseudo-Label Self-Training

All CVS baseline comparison trainers support the same default-off pseudo-label
module for SSDG-style ablations:

```bash
python -u -m baselines.cvcnn_ce.train \
  --wisig_pkl ./Dataset_WigSig/ManySig.pkl \
  --use_pseudo_labels \
  --pseudo_start_epoch 150 \
  --pseudo_threshold 0.90 \
  --pseudo_margin 0.0 \
  --lambda_pseudo 1.0
```

The switches are shared by `cvcnn_ce`, `riei_fd`, `drift`, and `ra_collab`.
When `--use_pseudo_labels` is absent, the original baseline training path is
unchanged.

## Smoke Commands

The older commands below remain synthetic smoke tests for API checks. Use an environment that has PyTorch installed:

```bash
python -m pytest tests/test_paper_reproduction_paper_parity.py tests/test_paper_reproduction_cvs_aligned.py -q
python -m baselines.riei_fd.train --help
python -m baselines.drift.train --help
python -m baselines.ra_collab.train --help
```

The default smoke configs use synthetic IQ data only for legacy API checks.
For real comparisons, use the method-local `train.py` entrypoints above.
