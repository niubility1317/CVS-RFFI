# SGC-Adapter Design

## Goal

Restore and extend the repository's SGC-Adapter path so CVS-RFFI can train and evaluate a lightweight satellite-ground-channel processor in front of the existing dual CVSincNet model.

## Project Context

The root project is a PyTorch WiSig/CVS-RFFI training codebase. `train.py` owns data loading, WiSig day/RX splits, SSDG/domain-generalization losses, satellite-channel simulation hooks, evaluation, checkpointing, and logging. `model_dual_cvsincnet.py` owns the dual identity/domain backbone. `sat_channel.py` already provides physics-inspired satellite-ground channel simulation. The HEAD history already contains an SGC implementation, but the current worktree has removed the SGC module, loss helper, wrapper, and launcher scripts, and has removed SGC wiring from model/train code.

## Architecture

The SGC processor is a small `nn.Module` placed before both model backbones. It preserves IQ tensor shape `[B, 2, L]`, defaults to identity-like behavior where possible, and exposes auxiliary metrics for residual regularization and logging. The adapter is opt-in through presets or CLI flags so existing non-SGC training remains compatible.

The restored adapter consists of four bounded blocks: per-sample RMS amplitude normalization, CNN-based frequency-offset compensation, FFT-domain soft spectral masking, and a depthwise residual channel compensator. Training uses three stages: source training, satellite-channel augmentation pretraining, and adapter-only target adaptation with frozen backbone/classifier parameters.

## Components

- `sgc_adapter.py`: standalone SGC-Adapter implementation and submodule toggles.
- `sgc_losses.py`: residual regularization, source prototypes, feature consistency, pseudo-label, entropy, and consistency helpers.
- `model_dual_cvsincnet.py`: optional `sgc_adapter` constructor arguments and forward pass preprocessing.
- `train.py`: SGC CLI flags, preset table entries, checkpoint loading, adapter-only freezing, satellite-view source selection, and residual regularization.
- `train_sgc.py`: compatibility entry point to `train.py`.
- `run_sgc_experiments.sh`: single-GPU three-stage SGC launcher.
- `run_all_preset_experiments.sh`: unified launcher for SGC/slimming/SSDG preset groups.
- `docs/SGC_EXPERIMENTS.md` and `docs/ALL_PRESET_EXPERIMENTS.md`: experiment matrix and startup examples.
- `tests/test_sgc_adapter.py`: shape, gradient, loss, and model integration sanity checks.

## Experiment Coverage

The restored preset set includes the full SGC baseline, no-adapter control, and four core ablations: no amplitude normalization, no frequency compensation, no spectral suppressor, and no residual compensator. The launch scripts expose source, augment, and adapt stages; the unified launcher can schedule the full staged chain or individual stages across configurable GPU IDs.

## Compatibility

When `sgc_adapter=false`, model behavior remains unchanged. Old checkpoints can load with `strict=False` because SGC parameters are optional and absent in legacy weights. The adaptation stage freezes the dual backbones and heads and optimizes only adapter parameters.

## Verification

Minimum verification is import/syntax checks plus targeted `pytest tests/test_sgc_adapter.py`. Because the repository depends on local datasets for full training, launcher verification uses dry-run or help-style command validation where possible rather than launching long experiments.
