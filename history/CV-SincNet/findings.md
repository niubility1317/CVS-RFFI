# Paper Baseline Findings

## Repo Context
- Current project is a collection of experimental folders (`type1` through `type15`, plus `type10-*` branches).
- Latest and most complete code appears around `type10-7`, with WiSig loaders, satellite/channel augmentation, physical-aware CVSincNet, and training controls.
- There is no top-level package/config structure for reusable baselines yet.
- Existing WiSig dataset code already returns `(x, y, d, meta)` where `x` is `[2, L]`, `y` is transmitter/device, and `d` can represent receiver/day/domain.
- Existing model code is tightly coupled to CVSincNet experiments, so adding an isolated baseline package is safer than modifying it.

## Provided Paper Plans
- `tifs2025_channel_receiver_rffi_codex_plan.md`: spectrogram, online channel augmentation, ResNet-style 2D extractor, NT-Xent pretraining, Siamese fine-tuning, single-branch eval.
- `riei_receiver_agnostic_feature_disentanglement_codex_plan.md`: split features into emitter and receiver parts, train emitter/receiver classifiers, minimize cosine dependence, maximize entropy of cross predictions, default alternating training.
- `drift_cross_receiver_rffi_codex_plan.md`: ResNet1D feature extractor, split into transmitter/receiver halves, GRL on transmitter feature, receiver center loss, negative MSE separation, ERM/MTL/DANN/DRIFT ablation switches.
- `receiver_agnostic_collaborative_rffi_codex_plan.md`: GRL receiver-agnostic training, spectrogram/CIS-ready model, fine-tuning on few target samples, soft and SNR-weighted collaborative fusion.

## Integration Choice
- New top-level package: `baselines/`.
- Shared utilities: `baselines/common/`.
- Paper-specific packages:
  - `baselines/tifs2025_channel_receiver_rffi/`
  - `baselines/riei/`
  - `baselines/drift/`
  - `baselines/receiver_agnostic_rffi/`

## Verification Targets
- Import all baseline model classes.
- Run forward passes on random IQ/spec tensors.
- Verify loss values are finite and backward works.
- Verify GRL sign behavior.
- Verify synthetic dataset and pair sampler.
