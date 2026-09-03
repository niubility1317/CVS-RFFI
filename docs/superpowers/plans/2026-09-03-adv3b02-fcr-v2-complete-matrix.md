# ADV3B02 FCR-V2 Complete Matrix Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement the report-faithful FCR-V2 factorization path, run the complete C0–M6 matrix from one ADV3B02 E200 checkpoint, and keep the N607 run under read-only health monitoring until truth-last four-scenario scoring closes.

**Architecture:** Keep the historical FCR-V1 path intact and add seven focused V2 modules behind an explicit `--fcr_version v2` route. `model_dual_cvsincnet.py` composes the modules, while `train.py` handles metadata, loss aggregation, optimizer groups, diagnostics, and final-only artifact export. One umbrella launcher schedules eight first-wave rows and six second-wave rows without using target metrics between waves.

**Tech Stack:** Python 3, PyTorch, pytest, Bash launchers on N607, Git, SSH/SCP, ManySig equalized IQ data.

**Spec:** `docs/superpowers/specs/2026-09-03-adv3b02-fcr-v2-complete-matrix-design.md`

## Global Constraints

- Initialize every new row from `/home/szu2070436088/2510044040/CV-SincNet/runs/phase1_adv3b02_fcr_r1r8_s392005_equalized_20260903_v4/ADV3B02/ADV3B02_CORE90_SOFT_E200/final_ssdg.pth`.
- Fix split seed to `392005`, training length to 200 epochs, and checkpoint selection to the final epoch only.
- Keep target test labels unreachable during training, scheduling, capability gating, and checkpoint selection.
- Use one non-overwriting umbrella run ID; C0 is reused, while C1–M6 receive disjoint row roots.
- Preserve unrelated untracked `local_artifacts` files and stage only files owned by this implementation.
- Run local project tests in the `ssr-gpu` Conda environment.
- Publish with one release archive SHA comparison, remote compile, immediate launch binding check, and 30-minute read-only monitoring.

---

### Task 1: Freeze the V2 contracts and reproduce report defects

**Files:**
- Create: `code/tests/test_phase1_fcr_v2_contracts.py`
- Modify: `code/cvsrffi/phase1_fcr_types.py`

**Interfaces:**
- Produces: `FCRV2Metadata`, `FCRV2FactorOutput`, `FCRV2CapabilityState`, and `FCRV2LossOutput` dataclasses used by Tasks 2–6.

- [ ] **Step 1: Write failing contract tests**

```python
def test_v2_metadata_shape_mismatch_fails_closed():
    meta = valid_metadata(batch=2)
    meta["eta_valid_mask"] = torch.ones(1, 8, dtype=torch.bool)
    with pytest.raises(ValueError, match="eta_valid_mask"):
        FCRV2Metadata.from_mapping(meta, batch_size=2)

def test_cross_decode_uses_destination_fingerprint():
    out = cross_decode(source, destination, decoder)
    assert decoder.last_z_f is destination.factors.z_f_dev
    assert decoder.last_z_n is destination.factors.z_n
```

- [ ] **Step 2: Run the focused tests and retain the expected failures**

Run: `python -m pytest code/tests/test_phase1_fcr_v2_contracts.py -q`

Expected: FAIL because the V2 dataclasses and `cross_decode` function do not exist.

- [ ] **Step 3: Add typed V2 dataclasses without changing V1 defaults**

```python
@dataclass(frozen=True)
class FCRV2CapabilityState:
    eta_ready: bool
    decoder_ready: bool
    swap_ready: bool
    fingerprint_ready: bool
    reasons: dict[str, str]
```

- [ ] **Step 4: Run the contract tests**

Run: `python -m pytest code/tests/test_phase1_fcr_v2_contracts.py -q`

Expected: metadata contract tests PASS; cross-decode remains failing until Task 5.

- [ ] **Step 5: Commit the contracts and tests**

```text
git add code/cvsrffi/phase1_fcr_types.py code/tests/test_phase1_fcr_v2_contracts.py
git commit -m "test: freeze FCR-V2 contracts"
```

### Task 2: Add strict metadata and deterministic three-axis pairing

**Files:**
- Create: `code/cvsrffi/phase1_fcr_v2_metadata.py`
- Create: `code/cvsrffi/phase1_fcr_v2_pairing.py`
- Create: `code/tests/test_phase1_fcr_v2_metadata_pairing.py`
- Modify: `code/baseline_origin_sat_aug.py`

**Interfaces:**
- Produces: `build_fcr_v2_metadata(batch, augmentation) -> FCRV2Metadata`.
- Produces: `FCRV2PairBuilder.build(metadata) -> dict[str, Tensor]` with `nuisance`, `content`, and `fingerprint` directed pair indices.

- [ ] **Step 1: Write failing eta and pairing tests**

```python
def test_augmentation_exports_applied_eta_and_full_valid_mask():
    view = augment(iq, sample_keys=("p0", "p1"), epoch=7, seed=392005)
    assert view.eta.shape == (2, ETA_DIM)
    assert view.eta_valid_mask.shape == view.eta.shape
    assert view.eta_valid_mask.float().mean().item() >= 0.99

def test_pair_builder_is_stateless_and_cross_tx_strict():
    a = builder.build(metadata, epoch=8, seed=392005)
    b = builder.build(metadata.flip_batch(), epoch=8, seed=392005)
    assert directed_pair_keys(a["fingerprint"]) == directed_pair_keys(b["fingerprint"])
    assert all(metadata.tx_id[i] != metadata.tx_id[j] for i, j in a["fingerprint"].tolist())
```

- [ ] **Step 2: Confirm both tests fail**

Run: `python -m pytest code/tests/test_phase1_fcr_v2_metadata_pairing.py -q`

- [ ] **Step 3: Implement strict schema validation and keyed random streams**

Use `blake2b(f"{seed}:{epoch}:{physical_sample_id}:{view_type}".encode(), digest_size=8)` as the deterministic augmentation/pair key. Reject missing fields, wrong leading batch dimensions, unknown `eta_schema_version`, and non-finite valid eta values.

- [ ] **Step 4: Implement the three pairing predicates**

Nuisance pairs require equal `(tx_id,content_record_id,crop_offset)` and different nuisance; content pairs require equal `(tx_id,rx_i,day_i,link_condition)` and overlap at most 25%; fingerprint pairs require equal `(common_preamble_id,rx_i,day_i,link_condition,excitation_bin)` and different TX.

- [ ] **Step 5: Run metadata/pairing and existing augmentation tests**

Run: `python -m pytest code/tests/test_phase1_fcr_v2_metadata_pairing.py code/tests/test_phase1_fcr_pairing.py -q`

Expected: PASS.

- [ ] **Step 6: Commit metadata and pairing**

```text
git add code/cvsrffi/phase1_fcr_v2_metadata.py code/cvsrffi/phase1_fcr_v2_pairing.py code/baseline_origin_sat_aug.py code/tests/test_phase1_fcr_v2_metadata_pairing.py
git commit -m "feat: add strict FCR-V2 metadata pairing"
```

### Task 3: Implement restricted factors and corrected physics

**Files:**
- Create: `code/cvsrffi/phase1_fcr_v2_factors.py`
- Create: `code/cvsrffi/phase1_fcr_v2_physics.py`
- Create: `code/tests/test_phase1_fcr_v2_physics.py`

**Interfaces:**
- Produces: `FCRV2FactorEncoder.forward(canonical_iq, residual_iq, z_adv) -> FCRV2FactorOutput`.
- Produces: `IdentityInitializedPhysicsDecoder.forward(s_hat, delta_f, z_n) -> FCRDecodeOutput`.
- Produces: `complex_gram(x: Tensor) -> Tensor`.

- [ ] **Step 1: Write failing factor and physics tests**

```python
def test_content_code_is_capacity_limited():
    out = encoder(iq, residual, z_adv)
    assert out.z_s.shape[-1] <= 16
    assert out.z_f_dev.requires_grad

def test_identity_decoder_starts_as_identity_channel():
    decoded = decoder.identity_forward(iq)
    torch.testing.assert_close(decoded.mu_iq, iq, atol=1e-5, rtol=1e-5)

def test_sto_shifts_and_sfo_creates_phase_slope():
    assert peak_index(apply_sto(impulse, 3.0)) == peak_index(impulse) + 3
    phase = torch.angle(apply_sfo(tone, 0.01).complex())
    assert torch.diff(torch.unwrap(phase)).abs().mean() > 0
```

- [ ] **Step 2: Verify the tests fail before implementation**

Run: `python -m pytest code/tests/test_phase1_fcr_v2_physics.py -q`

- [ ] **Step 3: Implement the content bottleneck and coupled fingerprint response**

Project normalized `z_f_id=normalize(z_adv+delta_z_f)` into `z_f_dev`; feed `z_f_dev` and excitation-bin surface features into the orthogonalized response basis plus low-rank complex residual. Define residual as `canonical_iq-s_hat` in canonical coordinates.

- [ ] **Step 4: Implement effective nuisance dimensions and physics operators**

Represent IQ imbalance with complex `alpha,beta`, STO with differentiable temporal resampling, SFO with a sample-index phase ramp, and multipath with normalized taps. Do not expose decoder inputs that have no forward effect.

- [ ] **Step 5: Run new and existing physics tests**

Run: `python -m pytest code/tests/test_phase1_fcr_v2_physics.py code/tests/test_phase1_fcr_physics.py -q`

Expected: PASS.

- [ ] **Step 6: Commit factor and physics modules**

```text
git add code/cvsrffi/phase1_fcr_v2_factors.py code/cvsrffi/phase1_fcr_v2_physics.py code/tests/test_phase1_fcr_v2_physics.py
git commit -m "feat: implement FCR-V2 factors and physics"
```

### Task 4: Implement separated losses and capability-gated schedule

**Files:**
- Create: `code/cvsrffi/phase1_fcr_v2_losses.py`
- Create: `code/cvsrffi/phase1_fcr_v2_schedule.py`
- Create: `code/tests/test_phase1_fcr_v2_losses_schedule.py`

**Interfaces:**
- Produces: `compute_fcr_v2_losses(inputs, row, ema_normalizer) -> FCRV2LossOutput`.
- Produces: `FCRV2Schedule.state(epoch, row, capabilities) -> FCRStageState`.
- Produces: `cross_decode(source, destination, decoder) -> FCRDecodeOutput`.

- [ ] **Step 1: Write row-isolation, necessity, and schedule tests**

```python
@pytest.mark.parametrize("row,active", [("S1", {"self","shared_f"}), ("S2", {"self","shared_s"}), ("S4", {"self","shared_f","shared_s","swap"})])
def test_row_activates_only_registered_losses(row, active):
    assert schedule.state(epoch=120, row=row, capabilities=ready()).active_losses == active

def test_necessity_is_relative_drop_f_gap():
    loss = necessity_loss(full_error=torch.tensor(2.0), drop_error=torch.tensor(5.0))
    torch.testing.assert_close(loss, torch.tensor(1.5))
```

- [ ] **Step 2: Confirm expected failures**

Run: `python -m pytest code/tests/test_phase1_fcr_v2_losses_schedule.py -q`

- [ ] **Step 3: Implement EMA normalization and report weights**

Register CE 1.00, prototype 0.10, class-tail 0.075, self 0.10, shared-f 0.20, shared-s 0.05, response 0.05, eta 0.10, swap ramp 0→0.05, and U_s FCR weight 0.35. Keep cycle, need, transplant, physical, and factor at zero unless the row and capability state both enable them.

- [ ] **Step 4: Implement the six-stage schedule**

Encode E1–20 warm-up, E21–60 shared, E61–100 nuisance, E101–130 swap, E131–160 cycle/need, and E161–200 identity refinement. A failed capability produces a zero scale plus `MECHANISM_NOT_ACTIVATED:<reason>`; it never reads target state or terminates the row.

- [ ] **Step 5: Run the focused loss and schedule tests**

Run: `python -m pytest code/tests/test_phase1_fcr_v2_losses_schedule.py code/tests/test_phase1_fcr_schedule.py code/tests/test_phase1_fcr_cross_losses.py -q`

Expected: PASS.

- [ ] **Step 6: Commit losses and schedule**

```text
git add code/cvsrffi/phase1_fcr_v2_losses.py code/cvsrffi/phase1_fcr_v2_schedule.py code/tests/test_phase1_fcr_v2_losses_schedule.py
git commit -m "feat: add FCR-V2 objectives and schedule"
```

### Task 5: Add complete diagnostics before deferred target evaluation

**Files:**
- Create: `code/cvsrffi/phase1_fcr_v2_diagnostics.py`
- Create: `code/tests/test_phase1_fcr_v2_diagnostics.py`
- Modify: `code/train.py:404-506`
- Modify: `code/train.py:5277-5345`

**Interfaces:**
- Produces: `collect_fcr_v2_diagnostics(model, loader, resources) -> dict[str, Any]`.
- Produces: `write_fcr_v2_diagnostics(path, row_id, artifacts) -> None`.

- [ ] **Step 1: Write a failing defer-order test**

```python
def test_diagnostics_written_before_deferred_target_return(tmp_path, monkeypatch):
    args = training_args(defer_target_evaluation=True, fcr_diagnostics_path=tmp_path / "diag.json")
    run_finalization(args, model, source_loader)
    payload = json.loads((tmp_path / "diag.json").read_text())
    assert payload["schema"] == "adv3b02_fcr_diagnostics:v2"
    assert payload["eta_valid_coverage"] >= 0.99
```

- [ ] **Step 2: Confirm the test fails on current finalization order**

Run: `python -m pytest code/tests/test_phase1_fcr_v2_diagnostics.py -q`

- [ ] **Step 3: Implement V2 diagnostic schema**

Include pair counts/coverage, eta coverage and component error, decoder nuisance sensitivity, swap output delta, z-factor probes, gradient norm ratios/cosines, per-TX source metrics, peak VRAM, epoch time, activation state, and reasons. Missing required numeric artifacts are `N/A` with a reason, never fabricated zeros.

- [ ] **Step 4: Move diagnostic collection before the deferred return**

Reuse source loaders only. Ensure no call from this block opens target datasets or truth sidecars.

- [ ] **Step 5: Run diagnostic and truth-blind regression tests**

Run: `python -m pytest code/tests/test_phase1_fcr_v2_diagnostics.py code/tests/test_phase1_fcr_diagnostics.py code/tests/test_phase1_fcr_review_fix.py -q`

Expected: PASS.

- [ ] **Step 6: Commit diagnostics**

```text
git add code/cvsrffi/phase1_fcr_v2_diagnostics.py code/train.py code/tests/test_phase1_fcr_v2_diagnostics.py
git commit -m "feat: close FCR-V2 diagnostics"
```

### Task 6: Integrate V2 into the model and training loop

**Files:**
- Modify: `code/model_dual_cvsincnet.py:551-645`
- Modify: `code/model_dual_cvsincnet.py:902-996`
- Modify: `code/train.py:216-389`
- Modify: `code/train.py:676-852`
- Modify: `code/train.py:3512-3528`
- Modify: `code/train.py:3870-5189`
- Create: `code/tests/test_phase1_fcr_v2_training_integration.py`

**Interfaces:**
- Consumes all Task 1–5 interfaces.
- Produces: `--fcr_version {v1,v2}`, `--fcr_matrix_row`, `forward_identity_only`, row execution signatures, V2 optimizer groups, and final `final.pth` bundles.

- [ ] **Step 1: Write failing checkpoint, route, and source-only smoke tests**

```python
def test_v2_loads_mature_checkpoint_and_copies_identity_head(real_checkpoint):
    report = load_init_checkpoint_weights(model_v2, real_checkpoint, require_mature_base_complete=True)
    assert report["expected_seed"] == 392005
    assert model_v2.fcr_identity_head_matches_legacy()

def test_forward_identity_only_does_not_run_decoder():
    model.fcr.decoder.forward = Mock(side_effect=AssertionError("decoder called"))
    assert model.forward_identity_only(iq)["tx_logits"].shape == (2, 6)
```

- [ ] **Step 2: Verify the new integration tests fail**

Run: `python -m pytest code/tests/test_phase1_fcr_v2_training_integration.py -q`

- [ ] **Step 3: Add explicit V2 routing and optimizer parameter groups**

Keep V1 command behavior unchanged. Route C1 around the FCR model, route C2/S0 through identity-noop V2 with all auxiliary scales zero, and map every remaining row to the registry defined in Task 4. Assert no parameter belongs to two optimizer groups.

- [ ] **Step 4: Use source metadata and separated L_s/U_s weights in training**

Apply identity/prototype/tail only where labels are available. Restrict U_s to self, shared-f, shared-s, response-shared, and eta until the row-specific repaired mechanism is enabled. Log execution signature and activation reasons each epoch.

- [ ] **Step 5: Run integration and historical FCR regressions**

Run: `python -m pytest code/tests/test_phase1_fcr_v2_training_integration.py code/tests/test_phase1_fcr_forward.py code/tests/test_phase1_fcr_checkpoint.py code/tests/test_phase1_fcr_gradient_routing.py code/tests/test_phase1_fcr_unlabeled_boundary.py -q`

Expected: PASS.

- [ ] **Step 6: Commit integration**

```text
git add code/model_dual_cvsincnet.py code/train.py code/tests/test_phase1_fcr_v2_training_integration.py
git commit -m "feat: integrate FCR-V2 training path"
```

### Task 7: Build and validate the complete matrix launcher

**Files:**
- Create: `code/scripts/launch_phase1_adv3b02_fcr_v2_complete_s392005_20260903.sh`
- Create: `code/tests/test_phase1_fcr_v2_complete_launcher.py`
- Modify: `code/scripts/predict_phase1_truth_last.py`
- Modify: `code/scripts/score_phase1_truth_last.py`

**Interfaces:**
- Produces one umbrella launcher with rows `C1 C2 C3 S0 S1 S2 S3 S4 M1 M2 M3 M4 M5 M6` and C0 external baseline registration.
- Produces complete per-row `final.pth`, `fcr_diagnostics.json`, `predictions.json`, and `score.json` paths.

- [ ] **Step 1: Write failing launcher contract tests**

```python
def test_complete_matrix_has_all_rows_once_and_final_only():
    dry = run_launcher_dry_run()
    assert dry.rows == ["C1","C2","C3","S0","S1","S2","S3","S4","M1","M2","M3","M4","M5","M6"]
    assert all(row.epochs == 200 and row.checkpoint_selection == "final_only" for row in dry.rowspecs)
    assert all(row.init_checkpoint == EXPECTED_ADV3B02 for row in dry.rowspecs)

def test_target_evaluation_occurs_after_all_training_waits():
    text = LAUNCHER.read_text()
    assert text.index("wait_training_rows") < text.index("--mode prepare") < text.index("--mode predict")
```

- [ ] **Step 2: Confirm launcher tests fail**

Run: `python -m pytest code/tests/test_phase1_fcr_v2_complete_launcher.py -q`

- [ ] **Step 3: Implement two-wave, one-row-per-GPU scheduling**

Wave 1 assigns C1,C2,C3,S0,S1,S2,S3,S4 to GPUs 0–7. Wave 2 assigns M1–M6 to GPUs 0–5 after Wave 1 processes finish. Wave 2 dispatch never reads Wave 1 target scores; it reads only source diagnostic activation records.

- [ ] **Step 4: Close final-only truth-last scoring**

Prepare one immutable target input package and independent truth sidecar after training. Export one prediction file per final checkpoint, validate all four scenario counts and sample IDs, then score each row with the independent scorer.

- [ ] **Step 5: Run dry-run and prediction/scorer tests**

Run: `python -m pytest code/tests/test_phase1_fcr_v2_complete_launcher.py code/tests/test_phase1_fcr_review_fix.py code/tests/test_phase1_fcr_r1r8_s392005_release.py -q`

Expected: PASS and a dry-run containing 14 unique output roots.

- [ ] **Step 6: Commit launcher and scoring integration**

```text
git add code/scripts/launch_phase1_adv3b02_fcr_v2_complete_s392005_20260903.sh code/scripts/predict_phase1_truth_last.py code/scripts/score_phase1_truth_last.py code/tests/test_phase1_fcr_v2_complete_launcher.py
git commit -m "feat: add complete FCR-V2 experiment matrix"
```

### Task 8: Verify, review, publish, launch, and monitor

**Files:**
- Create: `automation_reports/CV-SincNet/phase1_adv3b02_fcr_v2_complete_s392005_e200_20260903_v1/report.md`
- Create: `E:\type10-7\release_archives\phase1_adv3b02_fcr_v2_complete_s392005_e200_20260903_v1.tar.gz`

**Interfaces:**
- Consumes the completed code, tests, launcher, fixed checkpoint, and N607 access.
- Produces the `RUNNING` experiment state and recurring 30-minute health tracking.

- [ ] **Step 1: Run the complete focused local test set in `ssr-gpu`**

Run the Task 1–7 test files plus all existing `code/tests/test_phase1_fcr_*.py`. Expected: PASS with no skipped V2 contract test.

- [ ] **Step 2: Run one real-checkpoint no-query smoke locally**

Load the fixed ADV3B02 checkpoint, execute one source batch through C1, C2, S4, and M6 routes, perform one optimizer step, write diagnostics, and assert no target path or truth sidecar is opened.

- [ ] **Step 3: Request one independent P0/P1 correctness review**

Provide the frozen diff, spec, test output, exact checkpoint, and launcher dry-run. Fix only direct run-breaking, protocol-breaking, output-collision, process-ownership, or prediction-closure issues; perform at most one targeted re-review of a repaired finding.

- [ ] **Step 4: Commit and verify remote Git state**

Stage only owned implementation, test, plan, and preregistration report files. Push the branch and verify `git ls-remote` branch OID equals local `HEAD`.

- [ ] **Step 5: Run the direct N607 preflight and build one release archive**

Run `powershell -ExecutionPolicy Bypass -File tools\n607_ssh_preflight.ps1`. Build one archive from the verified commit, compare its local and remote SHA once, unpack into a new release root, and run remote Python compilation once.

- [ ] **Step 6: Launch the immutable complete matrix**

Use the fixed checkpoint, `seed=392005`, 200 epochs, final-only selection, one experiment per GPU, and a new non-existing output root. Immediately verify PID, CWD, cmdline, GPU mapping, run-root ownership, and log growth.

- [ ] **Step 7: Install 30-minute read-only health tracking**

Track active row count, epoch/log growth, GPU allocation, deterministic exception fingerprints, checkpoint/output growth, and later truth-last artifact closure. Stay quiet while healthy; notify on completion, technical failure, or required user action. Never stop for low performance.

- [ ] **Step 8: Complete scoring and publish the final report**

After all final checkpoints exist, verify predictions before scorer truth access, score clean and the three LEO scenarios, aggregate LEO mean/four-scenario mean/worst scenario/per-TX metrics, compare every row with C0, append mechanism activation evidence, commit and push the completed report, and verify the remote OID.
