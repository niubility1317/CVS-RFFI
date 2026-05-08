# Paper Baselines

This package adds isolated paper reproduction baselines for CVS-RFFI experiments.

## Methods

- `tifs2025_channel_receiver_rffi`: spectrogram + online channel augmentation + NT-Xent pretraining + Siamese fine-tuning + single-branch inference.
- `riei`: receiver-independent emitter identification with emitter/receiver feature disentanglement, MI loss, and entropy maximization.
- `drift`: cross-receiver feature disentanglement with GRL, receiver center loss, and negative MSE separation.
- `receiver_agnostic_rffi`: GRL receiver-agnostic training plus fine-tuning and soft/adaptive collaborative fusion.
- `cvcnn`: a plain complex-valued CNN baseline trained only with cross entropy.

## CVS-RFFI Training Commands

These entrypoints reuse the root `dataset_wisig.py` split used by CVS-RFFI:

- train/validation from `wisig_train_days x wisig_train_rxs`
- validation is the contiguous tail after `wisig_guard_gap`
- tests are the named CVS-RFFI OOD subsets:
  `test_unseen_day_seen_rx`, `test_seen_day_unseen_rx`, and
  `test_unseen_day_unseen_rx`
- named tests run only when each method's validation criterion improves; most
  methods use transmitter accuracy, while paper baselines that specify
  validation-loss scheduling use validation loss.

Default split parameters mirror the current CVS-RFFI defaults:
`--wisig_train_ratio 0.2 --wisig_guard_gap 8 --wisig_train_days 0,1 --wisig_test_days 2,3 --wisig_train_rxs 0,1,2,3,4,5,6 --wisig_test_rxs 7,8,9,10,11`.

```bash
python -m baselines.cvcnn.train --wisig_pkl ./Dataset_WigSig/ManySig.pkl
python -m baselines.riei.train --wisig_pkl ./Dataset_WigSig/ManySig.pkl
python -m baselines.drift.train --wisig_pkl ./Dataset_WigSig/ManySig.pkl
python -m baselines.receiver_agnostic_rffi.train --wisig_pkl ./Dataset_WigSig/ManySig.pkl
python -m baselines.receiver_agnostic_rffi.finetune_cvs --checkpoint baseline_runs/receiver_agnostic_rffi/best_by_val.pt --wisig_pkl ./Dataset_WigSig/ManySig.pkl
python -m baselines.tifs2025_channel_receiver_rffi.train --wisig_pkl ./Dataset_WigSig/ManySig.pkl
```

For queued multi-GPU runs, use the repo-level launcher. It defaults to GPUs
`0,1,2,3,4,5,6,7`, starts methods in parallel, keeps the CVS-RFFI split fixed
across methods, and leaves paper-stated training hyperparameters at each
method's own defaults. Methods without paper-stated epoch counts use 200
epochs; CVCNN uses 200 epochs.

```bash
METHODS=cvcnn,riei,drift,receiver_agnostic,tifs2025 \
GPU_IDS=0,1,2,3,4,5,6,7 \
nohup bash run_cvs_baseline_queue.sh > baseline_logs/cvs_baselines_$(date +%Y%m%d_%H%M%S).nohup.log 2>&1 &
```

The launcher also enables CVS-RFFI-style satellite-channel OOD evaluation by
default:

```bash
--eval_sat_channel
--eval_sat_on main
--eval_sat_scenarios clear_leo,low_elev_leo,rain_leo,storm_mp,mixed_orbit
```

This reports the three main named dimensions:
`test_unseen_day_seen_rx`, `test_seen_day_unseen_rx`, and
`test_unseen_day_unseen_rx`, plus satellite-channel aggregates for the same
dimensions.

Paper-specific defaults used by the CVS entrypoints:

- TIFS 2025: spectrogram representation, online channel augmentation, NT-Xent pretraining, receiver-paired Siamese fine-tuning, single-branch inference, `temperature=0.05`, `batch_size=32`, Adam `lr=3e-4`, validation-loss LR decay factor `0.5` after 10 stagnant epochs, early stop after 30 stagnant epochs.
- RIEI: FED feature split, EC/RC classifiers, signed cosine MI loss, information-entropy confusion loss, alternating classifier/FED updates.
- DRIFT: 1D ResNet-18-style encoder, transmitter/receiver feature split, GRL receiver discriminator, receiver style center regularization, negative MSE feature separation, Adam `lr=1e-4`, `lambda_grl=1`, `lambda_center=0.01`, `lambda_mse=0.02`, `batch_size=64`.
- Receiver-agnostic RFFI: spectrogram/CNN adversarial receiver training with GRL, SGD momentum `0.9`, `lr=1e-3`, `batch_size=64`, validation-loss LR decay factor `0.2` after 10 stagnant epochs, early stop after 20 stagnant epochs; formal CVS tests use receiver collaborative fusion. Few-shot target receiver fine-tuning uses `lr=1e-5`, `batch_size=32`, `epochs=20`.
- CVCNN: `epochs=200`; no paper auxiliary loss; only cross entropy on CVS-RFFI training data.

## Smoke Commands

The older commands below remain synthetic smoke tests for API checks. Use an environment that has PyTorch installed:

```bash
python -m unittest tests.test_cvs_paper_baselines -v
python -m baselines.tifs2025_channel_receiver_rffi.train_pretrain --help
python -m baselines.riei.train --help
python -m baselines.drift.train --help
python -m baselines.receiver_agnostic_rffi.train --help
```

The default smoke configs use synthetic IQ data only for legacy API checks.
For real comparisons, use the method-local `train.py` entrypoints above.
