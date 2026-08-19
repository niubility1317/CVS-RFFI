# ADVB02 CRRA Mixed-Orbit Enhancement Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 在当前Phase1数据协议下，为ADVB02加入CRRA（Channel–Receiver Robust Adapter）并在历史`mixed_orbit`星地信道上实现可测量的信道/接收机干扰抑制。

**Architecture:** CRRA位于身份分支共享Sinc/IQ与高频特征之后，只修正身份路径的时间/频率特征；域分支继续读取未稳健化的共享特征，PA分支保留直通路径。CRRA由选择性复数I/Q收缩白化、零初始化低秩深度卷积残差、stop-gradient条件向量、源域支持门和最小干预约束组成。Phase1只使用同一次`mixed_orbit`生成的卫星视图及其元数据，不引入目标接收机访问或Phase2在线校准。

**Tech Stack:** Python 3、PyTorch、pytest、现有ADVB02双CV-SincNet、现有`mixed_orbit`卫星仿真与SSDG训练器、`ssr-gpu`环境。

**Spec:** `E:/codex/home/attachments/2cf19e77-82bf-42ec-9cbe-3e24d0198789/pasted-text.txt`；逐条映射见`analysis/advb02_crra_traceability.md`。

## Global Constraints

- 历史星地信道固定为`mixed_orbit`；不得把本次实现替换为`leo_*_weak`。
- Phase1保持source-only弱标签域泛化、无目标接收机访问、无目标阈值/校准和query状态更新。
- 继续遵守`p2_min_v1`的数据权限边界；现有`VALIDATED_ONCE`数据不因方法、超参数或checkpoint变化而重验。
- 代码先在本地Git工作区编辑和验证，再同步N607；当前远端旧实验只读监控，不修改、不重启、不混用结果。
- 测试必须在`ssr-gpu`环境执行；Windows终端统一使用`C:\Program Files\Git\bin\bash.exe`并验证`MSYSTEM=MINGW64`。
- 实验前只执行项目允许的最小流程：聚焦协议负测、真实checkpoint无query smoke、一次P0/P1正确性审查、最小预登记、N607 preflight和启动后绑定检查。

---

### Task 1: CRRA核心模块与数学边界

**Files:**
- Create: `code/crra.py`
- Test: `code/tests/test_crra_adapter.py`
- Modify: `analysis/advb02_crra_traceability.md`

**Interfaces:**
- Consumes: `[B,2*S,T]` paired Sinc/IQ feature maps, `[B,C,T]` high-frequency cues, raw `[B,2,T]` IQ for RCN statistics.
- Produces: robustified time/frequency feature maps, stop-gradient condition `q`, gate/alpha/correction-energy telemetry, and nuisance prediction logits/regression values.

- [x] **Step 1: Write failing tests**

```python
def test_crra_is_identity_before_gate_warmup():
    adapter = CRRAAdapter(iq_channels=8, feature_channels=16, rank=8)
    x = torch.randn(3, 16, 32)
    out = adapter(x, raw_iq=torch.randn(3, 2, 32), epoch=1)
    assert torch.allclose(out.feature, x, atol=1e-6)
    assert out.gate.item() == 0.0

def test_crra_preserves_iq_pairing_and_bounds_intervention():
    adapter = CRRAAdapter(iq_channels=8, feature_channels=16, rank=8, alpha_max=0.25)
    out = adapter(torch.randn(4, 16, 32), raw_iq=torch.randn(4, 2, 32), epoch=80)
    assert out.feature.shape == (4, 16, 32)
    assert float(out.alpha.max()) <= 0.25 + 1e-6
    assert torch.isfinite(out.correction_energy).all()

def test_condition_q_does_not_backpropagate_into_condition_source():
    adapter = CRRAAdapter(iq_channels=8, feature_channels=16, rank=8)
    raw = torch.randn(2, 2, 32, requires_grad=True)
    out = adapter(torch.randn(2, 16, 32), raw_iq=raw, epoch=80)
    out.feature.sum().backward()
    assert raw.grad is None or torch.allclose(raw.grad, torch.zeros_like(raw.grad))
```

- [x] **Step 2: Run the focused tests and verify failure**

Run: `conda activate ssr-gpu && python -m pytest code/tests/test_crra_adapter.py -q`

Expected: FAIL because `code/crra.py` and `CRRAAdapter` do not yet exist.

- [x] **Step 3: Implement the minimal CRRA core**

Implement `ComplexIQShrinkageWhitening` with per-sample temporal I/Q mean and 2x2 covariance, covariance shrinkage toward the identity, stable inverse square root, paired-channel output, and a bounded residual coefficient. Implement `LowRankDepthwiseResidual` as depthwise `k=5` convolution followed by rank projection/up-projection with zero-initialized up projection. Implement `CRRAAdapter.forward(feature, raw_iq, epoch, ...)` with `E1–16=0`, linear ramp `E17–46`, and fixed gate from `E47`; build `q` only from `RCNStatEncoder`-compatible raw statistics plus feature GAP, detach `q` before gating, expose `alpha`, `gate`, `correction_energy`, `support_distance`, and `q` without using target labels or target data.

- [x] **Step 4: Run the focused tests and verify passage**

Run: `conda activate ssr-gpu && python -m pytest code/tests/test_crra_adapter.py -q`

Expected: all CRRA core tests PASS.

- [ ] **Step 5: Commit the isolated core**

```bash
git add code/crra.py code/tests/test_crra_adapter.py analysis/advb02_crra_traceability.md
git commit -m "feat: add ADVB02 CRRA robust adapter core"
```

### Task 2: ADVB02 identity-path wiring and PA/domain isolation

**Files:**
- Modify: `code/model.py`
- Modify: `code/model_dual_cvsincnet.py`
- Modify: `code/post_stage_common.py`
- Test: `code/tests/test_advb02_crra_model.py`

**Interfaces:**
- Consumes: `CRRAAdapter` from Task 1 and existing `CVSincNet` Sinc/HF stem.
- Produces: `aux_id["crra_*" ]`, `DualCVSincNetDisentangle` top-level CRRA telemetry, and builder arguments with CRRA disabled by default for backward compatibility.

- [x] **Step 1: Write failing model integration tests**

```python
def test_crra_is_only_enabled_on_identity_backbone():
    model = build_dual_model(..., use_crra=True, crra_rank=8, crra_alpha_max=0.25)
    out = model(torch.randn(2, 2, 256), return_aux=True)
    assert out["aux_id"]["crra_enabled"] is True
    assert out["aux_dom"].get("crra_enabled", False) is False
    assert "crra_correction_energy" in out

def test_crra_does_not_replace_pa_features():
    model = build_dual_model(..., use_crra=True)
    out = model(torch.randn(2, 2, 256), return_aux=True)
    assert out["id_feat_pa"].shape == out["id_feat_joint"].shape
    assert out["aux_id"]["crra_pa_bypass"] is True
```

- [x] **Step 2: Run tests and verify failure**

Run: `conda activate ssr-gpu && python -m pytest code/tests/test_advb02_crra_model.py -q`

Expected: FAIL because the builder and `CVSincNet` do not accept CRRA arguments.

- [x] **Step 3: Wire CRRA through the model**

Add explicit CRRA parameters to `CVSincNet`, `build_model`, `build_arch_backbone`, `DualCVSincNetDisentangle`, `build_dual_model`, and `post_stage_common.build_baseline_model`. Apply CRRA to the identity time feature after Sinc/HF construction and before `time_fuse`; use a separate frequency adapter on compressed frequency features when the frequency branch is active; leave raw input PA construction and domain-backbone features untouched. Preserve existing state-dict loading by making CRRA disabled by default and using `strict=False` for old checkpoints. Return CRRA telemetry in both `aux_id` and the dual-model top level.

- [x] **Step 4: Run model integration tests**

Run: `conda activate ssr-gpu && python -m pytest code/tests/test_advb02_crra_model.py code/tests/test_backbone_stability_options.py -q`

Expected: new CRRA tests and existing backbone plumbing tests PASS.

- [ ] **Step 5: Commit model wiring**

```bash
git add code/model.py code/model_dual_cvsincnet.py code/post_stage_common.py code/tests/test_advb02_crra_model.py
git commit -m "feat: wire CRRA into ADVB02 identity path"
```

### Task 3: Same-view `mixed_orbit` metadata propagation

**Files:**
- Modify: `code/baseline_origin_sat_view.py`
- Modify: `code/concat_sat_channel_aug.py`
- Modify: `code/cvsrffi/eval.py`
- Test: `code/tests/test_crra_mixed_orbit_metadata.py`

**Interfaces:**
- Consumes: metadata returned by the exact existing `apply_sat_gnd_channel_batch(..., return_meta=True)` call.
- Produces: optional per-sample nuisance tensor/mapping attached to `SatViewTransform` and `BaselineOriginSatViewBatch`; no regenerated or second satellite view.

- [x] **Step 1: Write failing metadata tests**

```python
def test_mixed_orbit_metadata_is_carried_with_the_same_satellite_view():
    aug = BaselineOriginSatViewAugment(scenarios=["mixed_orbit"], p=1.0, seed=7, apply_fn=fake_apply)
    view = aug.transform(torch.randn(3, 2, 32), args=SimpleNamespace(), epoch=1, batch_idx=1)
    assert view.applied is True
    assert view.meta["scenario"] == "mixed_orbit"
    assert view.meta["snr_db"].shape[0] == 3

def test_non_mixed_orbit_or_clean_duplicate_does_not_invent_nuisance_targets():
    view = ...
    assert view.meta is None or view.meta.get("valid", False) is False
```

- [x] **Step 2: Run tests and verify failure**

Run: `conda activate ssr-gpu && python -m pytest code/tests/test_crra_mixed_orbit_metadata.py -q`

Expected: FAIL because transform dataclasses currently discard `return_meta=True` output.

- [x] **Step 3: Implement metadata carriage**

Change only the view-carriage layer: call the existing channel function once with `return_meta=True`, retain CPU metadata without changing IQ bytes, normalize numeric fields (`snr_db`, `cfo_hz`, `residual_cfo_hz`, `fD_hz`, `pl_db`, `K_db`, `theta_deg`, `h_km`, `state`) into a finite `[B,D]` nuisance tensor plus a field map, and attach it to the transform/batch. Clean duplicates carry an invalid mask. The implementation must retain `scenario="mixed_orbit"` and must not access target receiver labels.

- [x] **Step 4: Run metadata and legacy view tests**

Run: `conda activate ssr-gpu && python -m pytest code/tests/test_crra_mixed_orbit_metadata.py code/tests/test_baseline_origin_sat_view.py code/tests/test_concat_sat_channel_aug.py -q`

Expected: all tests PASS.

- [ ] **Step 5: Commit metadata propagation**

```bash
git add code/baseline_origin_sat_view.py code/concat_sat_channel_aug.py code/cvsrffi/eval.py code/tests/test_crra_mixed_orbit_metadata.py
git commit -m "feat: preserve mixed-orbit nuisance metadata with satellite views"
```

### Task 4: CRRA loss terms, schedule, and telemetry in SSDG

**Files:**
- Modify: `code/SSDG/train_ssdg.py`
- Modify: `code/cvsrffi/logging.py`
- Modify: `code/cvsrffi/losses.py`
- Test: `code/tests/test_crra_training_plumbing.py`

**Interfaces:**
- Consumes: Task 2 model telemetry, Task 3 same-view nuisance metadata, existing clean/satellite pair ordering, and existing `mixed_orbit` KL/pair consistency.
- Produces: CRRA CLI controls, epoch schedule, nuisance Huber loss, correction-energy/gate penalties, TX-adversarial condition loss, and epoch telemetry.

- [ ] **Step 1: Write failing training-plumbing tests**

```python
def test_crra_schedule_has_identity_ramp_and_fixed_tail():
    assert crra_gate_scale(1) == 0.0
    assert crra_gate_scale(16) == 0.0
    assert 0.0 < crra_gate_scale(30) < 1.0
    assert crra_gate_scale(47) == 1.0

def test_phase1_crra_defaults_to_mixed_orbit_and_has_no_target_access():
    args = build_arg_parser().parse_args(["--output_dir", "x"])
    assert args.sat_train_scenario == "mixed_orbit"
    assert args.crra_scenario == "mixed_orbit"
    assert args.crra_target_adapter is False

def test_nuisance_loss_uses_valid_same_view_metadata_only():
    loss, info = crra_nuisance_huber_loss(pred, meta, valid_mask=torch.tensor([True, False]))
    assert info["valid_count"] == 1
    assert torch.isfinite(loss)
```

- [ ] **Step 2: Run tests and verify failure**

Run: `conda activate ssr-gpu && python -m pytest code/tests/test_crra_training_plumbing.py -q`

Expected: FAIL because CRRA parser/config/schedule/loss functions are not wired.

- [ ] **Step 3: Implement parser, schedule, losses, and model calls**

Add `--use_crra`, `--crra_rank 8`, `--crra_alpha_max 0.25`, `--crra_start_epoch 17`, `--crra_ramp_epochs 30`, `--crra_scenario mixed_orbit`, `--lambda_crra_pair`, `--lambda_crra_sat_kl`, `--lambda_crra_energy`, `--lambda_crra_gate_l1`, `--lambda_crra_nuisance`, `--lambda_crra_condition_tx_adv 0.02`, and `--crra_target_adapter` defaulting to false. Add an explicit schedule helper and include CRRA terms in the labeled clean/satellite path without changing existing pair ordering. Use `z_id` clean/satellite cosine consistency, existing satellite KL, correction energy/gate L1, finite same-view nuisance Huber regression, and optional condition TX adversary. CRRA losses must be zero before their scheduled epoch and must not create a second channel view. Add raw/weighted telemetry keys for pair cosine, correction energy, alpha, gate, nuisance loss, and condition TX accuracy.

- [ ] **Step 4: Run focused and adjacent training tests**

Run: `conda activate ssr-gpu && python -m pytest code/tests/test_crra_training_plumbing.py code/tests/test_phase1_advb02_cipg_launcher.py code/tests/test_federated_trainer_smoke.py -q`

Expected: all focused tests PASS; unrelated historical failures remain separately identified if present.

- [ ] **Step 5: Commit training integration**

```bash
git add code/SSDG/train_ssdg.py code/cvsrffi/logging.py code/cvsrffi/losses.py code/tests/test_crra_training_plumbing.py
git commit -m "feat: train ADVB02 CRRA with mixed-orbit consistency losses"
```

### Task 5: Checkpoint compatibility, no-query smoke, and protocol negatives

**Files:**
- Modify: `code/cvsrffi/checkpoint_loading.py`
- Modify: `code/tests/test_exact_ssdg_checkpoint_loading.py`
- Create: `code/tests/test_crra_protocol_negatives.py`
- Modify: `analysis/advb02_crra_traceability.md`

**Interfaces:**
- Consumes: the completed CRRA model/training implementation and an existing real ADVB02 checkpoint.
- Produces: backward-compatible checkpoint loading, explicit negative tests for target access and wrong channel name, and a reproducible no-query inference smoke.

- [ ] **Step 1: Write failing protocol-negative tests**

```python
def test_crra_rejects_wrong_phase1_channel_name():
    with pytest.raises(ValueError, match="mixed_orbit"):
        validate_crra_phase1_config(SimpleNamespace(crra_scenario="leo_weak"))

def test_crra_target_adapter_is_rejected_in_phase1():
    with pytest.raises(ValueError, match="target adapter"):
        validate_crra_phase1_config(SimpleNamespace(crra_scenario="mixed_orbit", crra_target_adapter=True))

def test_old_checkpoint_loads_with_crra_disabled():
    model = load_model_from_old_checkpoint(...)
    assert model.crra_enabled is False
```

- [ ] **Step 2: Run tests and verify failure**

Run: `conda activate ssr-gpu && python -m pytest code/tests/test_crra_protocol_negatives.py code/tests/test_exact_ssdg_checkpoint_loading.py -q`

Expected: the new negative tests FAIL before validation/compatibility code exists.

- [ ] **Step 3: Implement validation and compatibility**

Reject any Phase1 CRRA configuration whose channel is not `mixed_orbit` or whose target-adapter flag is enabled. Ensure old checkpoints instantiate with CRRA disabled and missing CRRA keys tolerated. Add a no-query smoke function that loads a real checkpoint, runs clean and `mixed_orbit` IQ only, and verifies that no query truth, target receiver label, or target calibration state is passed to the model.

- [ ] **Step 4: Run the protocol suite and real-checkpoint smoke**

Run: `conda activate ssr-gpu && python -m pytest code/tests/test_crra_protocol_negatives.py code/tests/test_exact_ssdg_checkpoint_loading.py code/tests/test_phase1_p1_protocol.py -q`

Then run the repository's existing real-checkpoint no-query smoke with CRRA disabled and enabled on one batch. Expected: PASS with finite logits, stable output shapes, `scenario=mixed_orbit`, and no query access.

- [ ] **Step 5: Commit compatibility and negatives**

```bash
git add code/cvsrffi/checkpoint_loading.py code/tests/test_exact_ssdg_checkpoint_loading.py code/tests/test_crra_protocol_negatives.py analysis/advb02_crra_traceability.md
git commit -m "test: enforce CRRA Phase1 mixed-orbit and checkpoint boundaries"
```

### Task 6: Local validation, release, and minimal mixed-orbit experiment

**Files:**
- Create: `code/scripts/launch_phase1_advb02_crra_mixed_orbit_20260819.sh`
- Create: `automation_reports/CV-SincNet/phase1_advb02_crra_mixed_orbit_20260819/report.md`
- Modify: `analysis/advb02_crra_traceability.md`

**Interfaces:**
- Consumes: all completed code and tests from Tasks 1–5.
- Produces: one immutable CRRA run root, one minimal pre-registered historical `mixed_orbit` screen, same-row results, and a decision for the next candidate.

- [ ] **Step 1: Run local verification in `ssr-gpu`**

Run:

```bash
conda activate ssr-gpu
python -m pytest code/tests/test_crra_adapter.py code/tests/test_advb02_crra_model.py code/tests/test_crra_mixed_orbit_metadata.py code/tests/test_crra_training_plumbing.py code/tests/test_crra_protocol_negatives.py -q
python -m compileall -q code/crra.py code/model.py code/model_dual_cvsincnet.py code/baseline_origin_sat_view.py code/concat_sat_channel_aug.py code/cvsrffi/eval.py code/SSDG/train_ssdg.py
```

Expected: focused suite PASS and compile succeeds.

- [ ] **Step 2: Fix the release commit and record the minimal report**

Record changed files, commit, exact command, CWD, environment, input/output paths, GPU, stopping rule, and expected artifacts. Do not add extra seal, authority, receipt, per-file hash, or report-only gate.

- [ ] **Step 3: Run N607 read-only preflight and sync the release**

Run the required direct N607 preflight first. Sync the local release archive with `scp` only after local verification. Compare the single release archive SHA once locally/remotely and run remote compile; do not edit remote-only files.

- [ ] **Step 4: Launch the minimal same-row screen**

Use the historical Phase1 split and `mixed_orbit` for every satellite row. Start with the existing control and one CRRA candidate, one paired seed, and the smallest registered Target5/Target25 screen. Verify PID/CWD/cmdline/GPU/log growth once. Stop only for protocol violation, wrong checkout, output collision, launcher fault, or missing prediction closure.

- [ ] **Step 5: Score and analyze without mixing rows**

After prediction closure, connect truth only in the independent scorer. Report clean/satellite TX accuracy, same-row source-val satellite harmonic score, receiver split, correction energy, gate/alpha, nuisance loss, and any old/new/unknown metrics that actually exist. Do not combine maxima across rows. Decide whether to expand to multi-seed/full confirmation based on the pre-registered scientific threshold; low performance is an analysis result, not a technical stop.

- [ ] **Step 6: Commit the launch/report handoff**

```bash
git add code/scripts/launch_phase1_advb02_crra_mixed_orbit_20260819.sh automation_reports/CV-SincNet/phase1_advb02_crra_mixed_orbit_20260819/report.md analysis/advb02_crra_traceability.md
git commit -m "exp: register ADVB02 CRRA mixed-orbit Phase1 screen"
```

## Self-Review Against the Spec

- Shared-stem identity robustification: Tasks 1–2.
- Selective complex I/Q whitening and bounded intervention: Task 1.
- Low-rank depthwise residual and zero initialization: Task 1.
- Stop-gradient condition vector and source support gate: Tasks 1 and 4.
- PA bypass and unmodified domain representation: Task 2.
- Clean/satellite pair cosine, satellite KL, energy/gate regularization: Task 4.
- Same-view `mixed_orbit` nuisance regression: Task 3 and Task 4.
- E1–16/E17–46/E47 schedule: Tasks 1 and 4.
- Phase1 target-access boundary and `mixed_orbit` lock: Task 5.
- Local verification, N607 sync, minimal experiment, same-row evidence: Task 6.
