# SGC-Adapter Restore Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Restore and extend SGC-Adapter code, training integration, experiment presets, and launch scripts for satellite-ground-channel CVS-RFFI experiments.

**Architecture:** Add a lightweight shape-preserving IQ adapter before the dual CVSincNet backbones, wire it through `train.py` behind opt-in flags and presets, and restore the three-stage SGC launcher. Preserve existing non-SGC behavior when the adapter is disabled.

**Tech Stack:** Python, PyTorch, pytest, Bash launch scripts.

---

### Task 1: Restore Core SGC Files

**Files:**
- Restore: `sgc_adapter.py`
- Restore: `sgc_losses.py`
- Restore: `train_sgc.py`
- Test: `tests/test_sgc_adapter.py`

- [ ] Restore `sgc_adapter.py` from HEAD and keep the public API: `SGCAdapter`, `AmplitudeNormalizer`, `FrequencyOffsetCompensator`, `SpectralInterferenceSuppressor`, `ResidualChannelCompensator`.
- [ ] Restore `sgc_losses.py` from HEAD and keep safe scalar behavior for empty masks.
- [ ] Restore `train_sgc.py` as a compatibility wrapper around `train.main`.
- [ ] Run `pytest tests/test_sgc_adapter.py::test_sgc_adapter_preserves_shape_and_gradients -q`.

### Task 2: Rewire Model Integration

**Files:**
- Modify: `model_dual_cvsincnet.py`
- Test: `tests/test_sgc_adapter.py::test_dual_model_exposes_sgc_aux_when_enabled`

- [ ] Add `sgc_adapter: bool = False` and `sgc_adapter_kwargs: Optional[Dict] = None` to `DualCVSincNetDisentangle.__init__`.
- [ ] Instantiate `self.sgc_adapter = SGCAdapter(**kwargs)` when enabled, else `None`.
- [ ] In `forward`, pass `x_processed` into both backbones and `DomainFeatureEnhancer`.
- [ ] Include `sgc_aux` in the returned aux dict.
- [ ] Add the same arguments to `build_dual_model` and pass them through.
- [ ] Run the model integration test.

### Task 3: Restore Train.py SGC Wiring

**Files:**
- Modify: `train.py`
- Test: `python -m py_compile train.py model_dual_cvsincnet.py sgc_adapter.py sgc_losses.py train_sgc.py`

- [ ] Re-add `json` import and `sgc_losses.residual_regularization`.
- [ ] Re-add SGC presets inside `apply_slim_ablation_preset`.
- [ ] Re-add helpers: `parse_json_dict`, `load_checkpoint_model_state`, `configure_sgc_trainable_params`, and `sgc_residual_loss_from_output`.
- [ ] Re-add CLI flags: `--preset`, `--stage`, `--sgc_adapter`, `--sgc_adapter_kwargs`, `--source_ckpt`, `--pseudo_label_threshold`, `--lambda_feat`, `--lambda_ent`, `--lambda_res`, `--adapt_lr`, `--adapt_epochs`, `--train_sat_channel`, `--train_sat_scenario`, and `--sat_view_source`.
- [ ] Apply stage aliases before and after preset resolution so `sgc_augment` and `sgc_adapt` force the adapter on.
- [ ] Pass SGC options into `build_dual_model`, load `--source_ckpt` with `strict=False`, and restrict optimizer params to adapter-only in `sgc_adapt`.
- [ ] Add `sgc_res` to meters, loss, logging, and checkpoint stats.
- [ ] Use `x_main` or clean `x` for satellite training according to `--sat_view_source`.
- [ ] Run py_compile.

### Task 4: Restore and Extend Launchers

**Files:**
- Restore/modify: `run_sgc_experiments.sh`
- Restore/modify: `run_all_preset_experiments.sh`
- Modify: `docs/SGC_EXPERIMENTS.md`
- Modify: `docs/ALL_PRESET_EXPERIMENTS.md`

- [ ] Restore the single-GPU SGC launcher with `source`, `augment`, and `adapt` stages.
- [ ] Restore the unified preset launcher with `PRESET_GROUPS`, `SGC_STAGES`, `GPU_IDS`, `DRY_RUN`, and configurable epoch counts.
- [ ] Add experiment documentation for full, ablation, staged-chain, and dry-run commands.
- [ ] Validate Bash syntax with `bash -n` when Bash is available.

### Task 5: Final Verification

**Files:**
- Verify: modified Python files, launchers, docs

- [ ] Run `python -m py_compile train.py model_dual_cvsincnet.py sgc_adapter.py sgc_losses.py train_sgc.py`.
- [ ] Run `pytest tests/test_sgc_adapter.py -q`.
- [ ] Run `bash -n run_sgc_experiments.sh run_all_preset_experiments.sh` if Bash is available.
- [ ] Summarize changed files, preset groups, launch examples, and any verification limitations.
