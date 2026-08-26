# SF-TAPFT R0 Clean Reference Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build an R0 SF-TAPFT clean reference that is jointly bound to the formal Phase1 deployment bundle and validated target-support capsule, preserves every non-trainable state tensor exactly, selects with 4-fold OOF evidence, and saves a model refit on all 60 support rows.

**Architecture:** Keep the V1 model capacity and loss unchanged. Add a Phase1 binding adapter around the existing formal deployment-bundle loader, replace full-state averaging with anchor-plus-trainable-delta restoration, separate OOF selection from full-support refit, and add a new strict `clean_single.v2` bundle. Query prediction code is implemented and protocol-tested in R0 but no real query is opened until R3.

**Tech Stack:** Python3.10、PyTorch2.1、NumPy、pytest、JSON/NPZ、existing ADV3B02 formal deployment-bundle loader.

**Spec:** `docs/superpowers/specs/2026-08-26-sf-tapft-staged-upper-bound-slimming-design.md`

## Global Constraints

- Phase2 updates may modify prototypes, target head, Adapter, original model layers and parameter tensors under the current conversation authorization.
- Phase2 must not read Phase1 raw/source samples, clean samples, sample-level source features, query truth, query role or query statistics.
- Target training rows come only from the existing `p2_min_v1/VALIDATED_ONCE` support capsule; `capsule_id` and `split_id` remain unchanged.
- The formal Phase1 deployment bundle fixes checkpoint lineage, runtime, ordered class registry and immutable int8 aggregate knowledge; it does not contain training samples.
- R0 keeps rank-16, full`t3`, persistent target head, phase steps500/1500/2500 and selective-KD weight0.
- Every production change follows RED→GREEN; only exact task files are staged and committed.
- R1, R2, R3 and all slimming stages remain deferred until R0 artifacts are analyzed.

---

### Task 1: Bind R0 to the formal Phase1 deployment bundle

**Files:**
- Create: `code/cvsrffi/sf_tapft_phase1_binding.py`
- Modify: `code/cvsrffi/target_only_progressive_runner.py`
- Test: `tests/test_sf_tapft_phase1_binding.py`
- Test: `tests/test_target_only_progressive_runner.py`

**Interfaces:**
- Consumes: the existing `load_formal_adv3b02_deployment_bundle()` and R0 config field `phase1_bundle` containing package/seal/envelope paths plus all expected SHA256 bindings.
- Produces: `SFTAPFTPhase1Binding` and `load_sf_tapft_phase1_binding(config, checkpoint_path)`.

- [ ] **Step 1: Write the failing binding tests**

```python
def test_phase1_binding_rejects_checkpoint_or_class_registry_drift(tmp_path):
    formal = _formal_fixture(
        checkpoint_lineage_sha256=sha256_file(tmp_path / "checkpoint.pth"),
        class_handles=("tx0", "tx1"),
    )
    binding = load_sf_tapft_phase1_binding(
        _binding_config(formal), tmp_path / "checkpoint.pth", formal_loader=lambda **_: formal
    )
    assert binding.class_handles == ("tx0", "tx1")
    with pytest.raises(ValueError, match="checkpoint lineage"):
        load_sf_tapft_phase1_binding(
            _binding_config(formal), tmp_path / "other.pth", formal_loader=lambda **_: formal
        )
```

Add a runner test whose support label exceeds the ordered Phase1 class registry and assert failure before `fit_sf_tapft()` is called. Add a test asserting the receipt carries the Phase1 outer root, checkpoint lineage and class binding SHA.

- [ ] **Step 2: Run the tests and verify RED**

Run: `C:\Users\lh594\.conda\envs\ssr-gpu\python.exe -m pytest tests/test_sf_tapft_phase1_binding.py tests/test_target_only_progressive_runner.py -q`

Expected: collection or assertion failure because `sf_tapft_phase1_binding` and the config contract do not exist.

- [ ] **Step 3: Implement the binding object and loader**

```python
@dataclass(frozen=True)
class SFTAPFTPhase1Binding:
    outer_content_root_sha256: str
    checkpoint_lineage_sha256: str
    runtime_sha256: str
    class_handle_binding_sha256: str
    class_handles: tuple[str, ...]
    component_pre_sign_content_root_sha256: str


def load_sf_tapft_phase1_binding(
    config: Mapping[str, Any],
    checkpoint_path: str | Path,
    *,
    formal_loader: Callable[..., VerifiedADV3B02DeploymentBundle] = load_formal_adv3b02_deployment_bundle,
) -> SFTAPFTPhase1Binding:
    verified = formal_loader(**_formal_loader_kwargs(config))
    actual_checkpoint_sha = sha256_file(Path(checkpoint_path))
    expected = str(verified.formal_phase2_context["checkpoint_lineage_sha256"])
    if actual_checkpoint_sha != expected:
        raise ValueError("SF-TAPFT checkpoint lineage does not match Phase1 bundle")
    rows = verified.class_binding["class_id_to_handle"]
    handles = tuple(str(row["class_handle"]) for row in rows)
    context = verified.formal_phase2_context
    return SFTAPFTPhase1Binding(
        outer_content_root_sha256=str(context["outer_content_root_sha256"]),
        checkpoint_lineage_sha256=expected,
        runtime_sha256=str(context["runtime_sha256"]),
        class_handle_binding_sha256=str(context["class_handle_binding_sha256"]),
        class_handles=handles,
        component_pre_sign_content_root_sha256=str(
            context["component_pre_sign_content_root_sha256"]
        ),
    )
```

Require the runner config to contain the complete `phase1_bundle` mapping. Validate support labels against `len(class_handles)` before model fitting. Do not read or expose component raw cells to the adapter; retain only immutable binding identifiers.

- [ ] **Step 4: Run focused tests and verify GREEN**

Run the same pytest command. Expected: all selected tests pass.

- [ ] **Step 5: Commit Task1**

```text
git add code/cvsrffi/sf_tapft_phase1_binding.py code/cvsrffi/target_only_progressive_runner.py tests/test_sf_tapft_phase1_binding.py tests/test_target_only_progressive_runner.py
git commit -m "feat: bind SF-TAPFT to Phase1 deployment bundle"
```

### Task 2: Replace full-state averaging with permitted delta restoration

**Files:**
- Modify: `code/cvsrffi/target_only_progressive_adapt.py`
- Test: `tests/test_target_only_progressive_adapt.py`

**Interfaces:**
- Consumes: initial student state after identity Adapter insertion, initial target-head state and the union of model parameter names permitted by A/B/C.
- Produces: `TrainableDeltaAverager.average(states, model_anchor, head_anchor, permitted_model_names)` and audit fields for permitted/non-permitted changes.

- [ ] **Step 1: Write failing exact-state tests**

```python
def test_delta_average_restores_nonpermitted_floating_state_exactly():
    anchor = {"allowed": torch.tensor([1.0]), "frozen": torch.tensor([10000001.0])}
    states = [
        ({"allowed": torch.tensor([2.0]), "frozen": anchor["frozen"].clone()}, (1.0,)),
        ({"allowed": torch.tensor([4.0]), "frozen": anchor["frozen"].clone()}, (0.0,)),
    ]
    result = TrainableDeltaAverager(top_k=2).average(
        states, anchor=anchor, permitted_names={"allowed"}
    )
    assert torch.equal(result["frozen"], anchor["frozen"])
    assert torch.equal(result["allowed"], torch.tensor([3.0]))
```

Add an end-to-end fit test with`checkpoint_average_top_k=3` asserting every model state outside the A/B/C union is exactly equal to the post-Adapter anchor, including Sinc-like large floating buffers.

- [ ] **Step 2: Run the focused test and verify RED**

Run: `C:\Users\lh594\.conda\envs\ssr-gpu\python.exe -m pytest tests/test_target_only_progressive_adapt.py -q`

Expected: failure because V1 averages every floating state tensor.

- [ ] **Step 3: Implement anchor-plus-delta averaging**

```python
if name in permitted_names:
    deltas = [state[name].to(torch.float64) - anchor[name].to(torch.float64) for state, _ in selected]
    averaged[name] = (anchor[name].to(torch.float64) + torch.stack(deltas).mean(0)).to(anchor[name].dtype)
else:
    averaged[name] = anchor[name].detach().clone()
```

Average target-head weights through their own anchor. Restore all non-permitted model parameters and buffers from the model anchor without arithmetic. Audit `permitted_changed_names`、`nonpermitted_changed_names` and require the last field to be empty.

- [ ] **Step 4: Run focused tests and verify GREEN**

Run the same pytest command. Expected: all target-only progressive adaptation tests pass.

- [ ] **Step 5: Commit Task2**

```text
git add code/cvsrffi/target_only_progressive_adapt.py tests/test_target_only_progressive_adapt.py
git commit -m "fix: preserve frozen SF-TAPFT state exactly"
```

### Task 3: Record stage metrics and select one unified refit schedule

**Files:**
- Modify: `code/cvsrffi/target_only_progressive_adapt.py`
- Test: `tests/test_target_only_progressive_adapt.py`

**Interfaces:**
- Consumes: per-step validation logits and the fixed A/B/C schedule.
- Produces: `StageValidationMetrics`, `selected_phase_steps` and per-fold stage rows.

- [ ] **Step 1: Write failing metric and schedule tests**

Create hand-derived logits for two classes and assert balanced accuracy、macro-F1、class floor、NLL、per-class recall、per-class margin and positive/negative flips. Add a deterministic schedule test where fold best steps are A=`[400,450,500,500]`、B=`[1000,1200,1100,1300]`、C=`[300,400,500,500]`; assert the lower median schedule is `(475,1150,450)`.

- [ ] **Step 2: Run focused tests and verify RED**

Run the adaptation test file. Expected: missing metric and schedule APIs.

- [ ] **Step 3: Implement stage telemetry and lower-median aggregation**

```python
@dataclass(frozen=True)
class StageValidationMetrics:
    balanced_accuracy: float
    macro_f1: float
    class_floor: float
    nll: float
    per_class_recall: tuple[float, ...]
    per_class_margin: tuple[float, ...]
    positive_flips: int
    negative_flips: int
    permitted_parameter_distance: float


def _lower_median(values: Sequence[int]) -> int:
    ordered = sorted(int(value) for value in values)
    return ordered[(len(ordered) - 1) // 2]
```

For each fold, retain the best checkpoint independently within A、B and C. Aggregate each phase length by lower median across folds; this avoids a single optimistic fold extending the full-support refit. Stage selection ordering is BA→floor→-NLL→macro-F1→margin→-permitted-distance.

- [ ] **Step 4: Run focused tests and verify GREEN**

Run the adaptation test file. Expected: all tests pass and existing V1 aggregate metrics remain available.

- [ ] **Step 5: Commit Task3**

```text
git add code/cvsrffi/target_only_progressive_adapt.py tests/test_target_only_progressive_adapt.py
git commit -m "feat: add SF-TAPFT stage selection metrics"
```

### Task 4: Separate OOF selection from all-support refit

**Files:**
- Modify: `code/cvsrffi/target_only_progressive_adapt.py`
- Test: `tests/test_target_only_progressive_adapt.py`

**Interfaces:**
- Consumes: `selected_phase_steps` from Task3 and the complete target-support dataset.
- Produces: `SFTAPFTSelectionResult.full_support_result` and `final_training_sample_count`.

- [ ] **Step 1: Write a failing regression test for fold0 deployment**

```python
def test_grouped_selection_refits_from_base_on_all_support_rows():
    dataset = _dataset()
    selection = select_sf_tapft_by_grouped_cv(
        _ToyModel(),
        dataset,
        SFTAPFTConfig(
            phase_steps=(1, 1, 1),
            warmup_ratio=0.0,
            checkpoint_average_top_k=1,
            adapter_rank=2,
            seed=29,
        ),
        folds=3,
    )
    assert selection.full_support_result is not None
    assert selection.final_training_sample_count == len(dataset.physical_ids)
    assert selection.fold0_as_final is False
    assert selection.full_support_result.audit.phase_steps == selection.selected_phase_steps
```

Instrument the toy model or fit audit so the test proves the final model consumed all rows, not merely that receipt text says60.

- [ ] **Step 2: Run the regression test and verify RED**

Run the adaptation test file. Expected: failure because V1 returns`fitted_folds[0]`.

- [ ] **Step 3: Implement all-support refit**

After OOF chooses`adapted`, create a new config using`selected_phase_steps`, copy the original checkpoint model, and call`fit_sf_tapft()` on the complete dataset with`checkpoint_average_top_k=1`. The final refit uses the fixed schedule and no inner-validation model selection. Preserve all OOF fold models only in selection memory; do not parameter-average them.

- [ ] **Step 4: Run focused tests and verify GREEN**

Run the adaptation test file. Expected: final sample count equals the dataset size and no fold model is used as final.

- [ ] **Step 5: Commit Task4**

```text
git add code/cvsrffi/target_only_progressive_adapt.py tests/test_target_only_progressive_adapt.py
git commit -m "feat: refit SF-TAPFT on full support"
```

### Task 5: Write and strictly reload the clean-single V2 bundle

**Files:**
- Modify: `code/cvsrffi/target_only_progressive_runner.py`
- Test: `tests/test_target_only_progressive_runner.py`

**Interfaces:**
- Consumes: full-support result, OOF receipt, Phase1 binding and target data binding.
- Produces: schema`cvs.sf_tapft.clean_single.v2` and `load_sf_tapft_clean_single_bundle_strict()`.

- [ ] **Step 1: Write failing bundle-schema tests**

Assert exact top-level fields include Phase1 outer root、checkpoint lineage、class binding、capsule/split、selected phase steps、support count、per-class counts、`fold0_as_final=false`、model/head state and state-change audit. Mutate each binding and assert strict loader rejection.

- [ ] **Step 2: Run runner tests and verify RED**

Run the runner test file. Expected: V2 schema and strict loader do not exist.

- [ ] **Step 3: Implement the V2 writer and loader**

Keep`load_sf_tapft_bundle_strict()`for V1 read-only compatibility. Add a separate V2 loader; do not broaden the V1 allowlist. Write the output directory with exclusive creation, then persist the bundle and`selection.json`containing OOF metrics and full-support identity.

- [ ] **Step 4: Run runner tests and verify GREEN**

Run the runner test file. Expected: V1 compatibility and V2 strict round-trip both pass.

- [ ] **Step 5: Commit Task5**

```text
git add code/cvsrffi/target_only_progressive_runner.py tests/test_target_only_progressive_runner.py
git commit -m "feat: add SF-TAPFT clean single bundle"
```

### Task 6: Add R0 query-read-only interface without opening real query

**Files:**
- Create: `code/cvsrffi/sf_tapft_prediction.py`
- Test: `tests/test_sf_tapft_prediction.py`

**Interfaces:**
- Consumes: one strict clean-single V2 model/head and query IQ tensor.
- Produces: independent per-row logits/predictions with no label/role/global-assignment input.

- [ ] **Step 1: Write failing prediction tests**

```python
def test_clean_single_prediction_is_independent_per_row_and_accepts_no_truth_role():
    parameters = inspect.signature(predict_sf_tapft_rows).parameters
    assert set(parameters) == {"model", "head", "received_iq"}
    first = predict_sf_tapft_rows(model, head, rows[:1])
    batched = predict_sf_tapft_rows(model, head, rows)
    assert torch.equal(first.predictions, batched.predictions[:1])
```

Also test that reordering query rows only reorders outputs and that the result exposes`query_truth_opened=false`and`query_role_opened=false`.

- [ ] **Step 2: Run prediction tests and verify RED**

Expected: module/function missing.

- [ ] **Step 3: Implement stateless prediction**

Run model/head in`eval()`under`torch.no_grad()`; compute each row against all registered head classes. Return detached CPU logits, predictions and audit flags. Do not write a scorer or load any real query in this task.

- [ ] **Step 4: Run prediction tests and verify GREEN**

Expected: all prediction protocol tests pass.

- [ ] **Step 5: Commit Task6**

```text
git add code/cvsrffi/sf_tapft_prediction.py tests/test_sf_tapft_prediction.py
git commit -m "feat: add read-only SF-TAPFT prediction"
```

### Task 7: Complete R0 local verification and release preparation

**Files:**
- Create: `configs/stage2_sf_tapft_v2_clean_r16_t3_rx20_1_s392002_20260826.json`
- Modify: `docs/experiments/sf_tapft_v2_staged_traceability_20260826.md`
- Create: `E:\type10-7\automation_reports\CV-SincNet\stage2_sf_tapft_v2_clean_r16_t3_rx20_1_s392002_20260826_r1\report.md`outside the Git root, mirrored as`docs/experiments/stage2_sf_tapft_v2_clean_r16_t3_rx20_1_s392002_20260826_r1_report.md`before launch.

**Interfaces:**
- Consumes: Tasks1–6.
- Produces: one Git-fixed R0 configuration, complete local verification evidence and an immutable N607 run contract.

- [ ] **Step 1: Run focused and regression tests**

Run:

```text
C:\Users\lh594\.conda\envs\ssr-gpu\python.exe -m pytest tests/test_sf_tapft_phase1_binding.py tests/test_target_only_progressive_adapt.py tests/test_target_only_progressive_runner.py tests/test_sf_tapft_prediction.py -q
C:\Users\lh594\.conda\envs\ssr-gpu\python.exe -m py_compile code/cvsrffi/sf_tapft_phase1_binding.py code/cvsrffi/target_only_progressive_adapt.py code/cvsrffi/target_only_progressive_runner.py code/cvsrffi/sf_tapft_prediction.py
```

Expected: zero failures and zero syntax errors.

- [ ] **Step 2: Reverse-audit traceability**

Update V2-02、V2-02A and V2-03–V2-09 with exact files and test evidence. Leave every R1+ item`deferred`; do not mark later stages implemented because interfaces exist.

- [ ] **Step 3: Commit and push the R0 implementation**

Stage only Tasks1–7 files and the mirrored report. Push the current branch and independently verify remote OID equals local HEAD.

- [ ] **Step 4: Run the project preflight and no-query smoke**

After local commit and report preparation, use the project N607 preflight. Release one archive, compare its local/remote SHA once, compile remotely once, then run the real ADV3B02 checkpoint no-query smoke with the formal Phase1 bundle and existing 60-row target-support capsule.

- [ ] **Step 5: Stop at the R0 evidence boundary**

Verify the smoke artifact reports`nonpermitted_changed_count=0`、`support_count=60`、`fold0_as_final=false`and no source/query/truth access. Do not launch R1 in the same step. R1 begins only after R0 performance artifacts are complete and analyzed.
