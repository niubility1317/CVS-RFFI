# SF-TAPFT P1 Compact Deployment Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 将P0C改造成可审计的紧凑suffix部署底座，并在新未暴露capsule上完成D0–D4最小性能矩阵。

**Architecture:** 保留现有研究入口和完整模型兼容路径；新增一次性计算cache与独立`CompactH6Suffix`部署路径。性能候选只通过统一配置扩展可训练Norm集合、support-only温度或H6后的head-only CVaR，不改变Query推理分支。

**Tech Stack:** Python、PyTorch、pytest、JSON、N607、Git

**Spec:** `docs/superpowers/specs/2026-08-28-sf-tapft-p1-compact-deploy-design.md`

## Global Constraints

- Phase2只读取匹配的`p2_min_v1/VALIDATED_ONCE/capsule_id/split_id`、合法support、冻结checkpoint和bundle。
- 禁止适配期读取query、query truth、query role、source或clean样本。
- D0–D4不注册新类，只报告`DA0_REG0`与`DA1_REG0`。
- 可训练元素不超过1584、delta不超过10KB、适配时间不超过20秒。
- 所有生产代码必须先有失败测试；只stage本轮文件。
- HardPair固定为0；CUDA Graph/AOT不是本轮发布门。

---

### Task 1: Correct Resource Measurement Semantics

**Files:**
- Modify: `code/cvsrffi/sf_tapft_deployment_benchmark.py`
- Modify: `code/scripts/run_sf_tapft_deployment_benchmark.py`
- Test: `tests/test_sf_tapft_deployment_benchmark.py`

**Interfaces:**
- Produces: `_current_rss_bytes() -> int`
- Produces: `_process_lifetime_maxrss_bytes() -> int`
- Produces: `benchmark_deployment_runs(..., execution_mode: str) -> dict[str, Any]`

- [ ] **Step 1: Write failing Linux RSS tests**

```python
def test_linux_current_rss_reads_vmrss_not_ru_maxrss(monkeypatch, tmp_path):
    status = tmp_path / "status"
    status.write_text("VmRSS:\t1234 kB\n", encoding="ascii")
    monkeypatch.setattr(benchmark, "_LINUX_PROC_STATUS", status)
    assert benchmark._current_rss_bytes() == 1234 * 1024

def test_lifetime_maxrss_has_a_distinct_field_name():
    result = benchmark.benchmark_deployment_runs(...)
    assert "process_lifetime_maxrss_bytes" in result
```

- [ ] **Step 2: Run focused tests and confirm the expected failure**

Run: `C:\Users\lh594\.conda\envs\ssr-gpu\python.exe -m pytest tests\test_sf_tapft_deployment_benchmark.py -q`

Expected: FAIL because Linux still uses`ru_maxrss`as current RSS and the new field is absent.

- [ ] **Step 3: Implement current/lifetime/CUDA free-memory sampling**

Read`VmRSS`from`/proc/self/status`; keep`ru_maxrss`in the distinct lifetime function. Add start/min/end CUDA free memory and execution mode to every sample and summary.

- [ ] **Step 4: Add cold-start runner mode**

The script accepts`--execution-mode resident_process|cold_start`; cold start invokes one fresh child process per repeat and aggregates immutable JSON receipts.

- [ ] **Step 5: Run focused and regression tests**

Expected: resource tests pass; existing benchmark schema remains readable.

### Task 2: Split Cache Storage and Compute Precision

**Files:**
- Modify: `code/cvsrffi/target_only_progressive_adapt.py`
- Modify: `code/cvsrffi/target_only_progressive_runner.py`
- Modify: `tests/test_target_only_progressive_adapt.py`
- Modify: `tests/test_target_only_progressive_runner.py`

**Interfaces:**
- Extend: `SFTAPFTConfig.cache_storage_dtype: str`
- Extend: `SFTAPFTConfig.suffix_compute_dtype: str`
- Extend: `SFTAPFTConfig.cache_device: str`
- Produces: `H6PrefixCache.materialize_once(device, dtype) -> H6PrefixCache`

- [ ] **Step 1: Write failing config compatibility tests**

```python
def test_legacy_prefix_cache_dtype_maps_to_storage_and_compute():
    config = SFTAPFTConfig(prefix_cache_dtype="float16")
    assert config.cache_storage_dtype == "float16"
    assert config.suffix_compute_dtype == "float32"

def test_invalid_cache_compute_dtype_is_rejected():
    with pytest.raises(ValueError):
        SFTAPFTConfig(suffix_compute_dtype="int8")
```

- [ ] **Step 2: Confirm RED**

Expected: new fields and mapping do not exist.

- [ ] **Step 3: Implement normalized dtype/device configuration**

Accepted dtypes are`off/float16/bfloat16/float32`; accepted devices are`model/cpu/cuda`. Historical`prefix_cache_dtype`remains readable and maps to the new fields when explicit new values are absent.

- [ ] **Step 4: Write and confirm failing one-time materialize test**

Patch`Tensor.to`or use a counting tensor fixture to prove suffix forward performs no storage→compute conversion after trainer construction.

- [ ] **Step 5: Implement one-time materialization**

Create a compute cache before the optimizer loop. `forward_h6_suffix`consumes tensors already on the correct device and dtype; it must not call`.to(...)`for each step.

- [ ] **Step 6: Run focused regressions**

Expected: legacy P0A/P0B/P0C configs parse unchanged and new cache tests pass.

### Task 3: Build Independent CompactH6Suffix

**Files:**
- Modify: `code/cvsrffi/target_only_progressive_adapt.py`
- Modify: `tests/test_target_only_progressive_deploy.py`
- Modify: `tests/test_target_only_progressive_adapt.py`

**Interfaces:**
- Produces: `CompactH6Suffix.from_model(model, head, cache) -> CompactH6Suffix`
- Produces: `CompactH6Suffix.embedding() -> Tensor`
- Produces: `CompactH6Suffix.logits() -> Tensor`
- Produces: `CompactH6Suffix.export_permitted_state() -> Mapping[str, Tensor]`

- [ ] **Step 1: Write failing ownership test**

```python
def test_compact_suffix_does_not_retain_full_model_reference(h6_model, head, cache):
    compact = CompactH6Suffix.from_model(h6_model, head, cache)
    assert all(value is not h6_model for value in vars(compact).values())
```

- [ ] **Step 2: Write failing forward/gradient equivalence test**

With fixed literal input and seed, compare reference and compact logits, target-head gradients and`t3.norm.weight/bias`gradients. The test fails before the class exists.

- [ ] **Step 3: Implement the minimal compact module**

Deep-copy only`t3.norm/act/pool/drop`、`t_pool/t_proj`、`meta_adapter_time`、`fuse`、`meta_adapter_fusion`and`cls_head`; freeze every parameter except`t3.norm.weight/bias`andtarget head.

- [ ] **Step 4: Wire the deployment fit path**

`fit_sf_tapft_inplace`selects the compact engine when prefix caching is enabled. After training, copy only permitted state back to the caller-owned model and preserve existing safety audit/fallback.

- [ ] **Step 5: Run focused tests**

Expected: no full model reference, permitted state only, reference/compact argmax and gradients agree within dtype-specific tolerance.

### Task 4: Make Delta Export and Apply Atomic

**Files:**
- Modify: `code/cvsrffi/target_only_progressive_runner.py`
- Modify: `tests/test_target_only_progressive_deploy.py`

**Interfaces:**
- Produces: `write_sf_tapft_delta_atomic(path, payload) -> Path`
- Produces: `apply_sf_tapft_delta_transactional(model, delta, ...) -> result`

- [ ] **Step 1: Write failing partial-write and apply-failure tests**

Use a temporary directory and injected serializer/apply exception. Assert the old final file remains byte-identical, no partial final file appears, and model permitted parameters are restored.

- [ ] **Step 2: Confirm RED**

Expected: direct write/apply leaves no transactional guarantee.

- [ ] **Step 3: Implement same-directory temporary write, self-load and `os.replace`**

The temporary file is uniquely named in the final directory. Self-load validates schema before atomic replace. Cleanup only the exact temporary file.

- [ ] **Step 4: Implement transactional apply rollback**

Capture only permitted anchors and target head; restore both on any exception.

- [ ] **Step 5: Run deploy regressions**

Expected: v1/v2 load compatibility and existing Q180 delta closure tests remain green.

### Task 5: Implement D1–D4 Performance Candidates

**Files:**
- Modify: `code/cvsrffi/target_only_progressive_adapt.py`
- Modify: `code/cvsrffi/target_only_progressive_runner.py`
- Modify: `tests/test_target_only_progressive_adapt.py`
- Modify: `tests/test_target_only_progressive_runner.py`

**Interfaces:**
- Extend: `SFTAPFTConfig.head_cvar_weight: float`
- Extend: `SFTAPFTConfig.head_cvar_top_k: int`
- Extend: `SFTAPFTConfig.head_cvar_steps: int`
- Reuse: `fit_positive_temperature(logits, labels)`with support OOF only

- [ ] **Step 1: Write failing CVaR objective test**

```python
def test_class_cvar_uses_top2_class_mean_losses():
    losses = torch.tensor([0.1, 0.2, 0.9, 0.8, 0.3, 0.4])
    assert class_cvar_from_class_losses(losses, top_k=2).item() == pytest.approx(0.85)
```

- [ ] **Step 2: Write failing head-only mutation test**

Run a small CVaR polish fixture and assert target head changes while all model parameters and buffers remain byte-equal.

- [ ] **Step 3: Implement class-generic head-only CVaR polish**

After base H6 adaptation, detach final support embeddings, run exactly30head-only steps with`lambda_t=0.03`andTop2 class means, then include the polished head in delta v2.

- [ ] **Step 4: Add Q2A/Q2B deployment profiles**

Map D1 to`t3 weight_bias+t2 weight`and D2 to`t3 weight_bias+t2/t1/time_fuse weight`, enforce expected trainable counts1248and1368, and use the report-frozen step caps.

- [ ] **Step 5: Add R1-T profile**

Keep R1 training unchanged, fit one positive temperature from support OOF logits, assert argmax preservation, and serialize calibrated scale in delta v2.

- [ ] **Step 6: Add HardPair/resource guards**

The P1 matrix validator rejects nonzeroHardPair, trainable elements above1584, delta estimate above10KB or addedquery branches.

- [ ] **Step 7: Run focused tests**

Expected: D0 behavior unchanged; D1–D4 differ only through their registered variable.

### Task 6: Freeze the New-Capsule Matrix and Launcher

**Files:**
- Create: `configs/stage2_sf_tapft_p1_compact_d0_d4_s392002_20260828.json`
- Create: `code/scripts/run_sf_tapft_p1_compact_matrix.py`
- Create: `tests/test_sf_tapft_p1_compact_matrix.py`
- Update: `docs/experiments/stage2_sf_tapft_p1_compact_deploy_20260828_traceability.md`

**Interfaces:**
- Consumes: one discovered, unused`VALIDATED_ONCE`capsule with K=10、6 old classes and maximum legalQuery split
- Produces: immutable D0–D4 run roots, selection/delta/prediction receipts

- [ ] **Step 1: Write failing matrix validation test**

The fixture asserts exactlyD0–D4, one unique purpose per row, shared protocol handles,HardPair zero,delta-only and resource caps.

- [ ] **Step 2: Confirm RED**

Expected: config and validator do not exist.

- [ ] **Step 3: Discover the new capsule read-only**

On N607 list only targeted capsule metadata and protocol handles. Do not read query arrays or truth. Select one never used for candidate development.

- [ ] **Step 4: Write config and launcher**

Bind the exact checkpoint/support/query opaque paths, receiver、scene、K、seed and immutable run ID. Launcher first executes real-checkpoint no-query smoke, then dispatches five support rows and later prediction closure.

- [ ] **Step 5: Run local config, help and protocol-negative tests**

Expected: invalid protocol handles/query access/HardPair fail; frozen legal matrix passes.

### Task 7: Local Verification, Review and Release

**Files:**
- Update: `docs/experiments/stage2_sf_tapft_p1_compact_deploy_20260828_traceability.md`
- Create: `automation_reports/CV-SincNet/<run-id>/report.md`outsideGit and mirror it under`docs/experiments/`

- [ ] **Step 1: Run the full directly relevant test set in`ssr-gpu`**

Run adapt、deploy、runner、benchmark、matrix andquery-closure tests; run`py_compile`for changed modules/scripts.

- [ ] **Step 2: Run one real CORE90 checkpoint no-query smoke locally or on N607**

Expected: support-only selection/delta,`query_opened=false`、`source_opened=false`.

- [ ] **Step 3: Perform one independent P0/P1 correctness review**

Review only defects that can directly misroute, violate protocol, overwrite output, fail launch or fail to produce legalprediction. If needed, fix and perform one scoped re-review.

- [ ] **Step 4: Complete reverse traceability audit**

Every requirement is`verified/deferred/rejected/blocked`; each nonverified item has an evidence-backed reason.

- [ ] **Step 5: Stage exact files, commit, push and compare remote OID toHEAD**

Do not stage pre-existing`conversation_index/`or`local_artifacts/`.

### Task 8: N607 Experiment and Truth-Last Analysis

**Files:**
- Update: root run report
- Update: Git mirror report and traceability

- [ ] **Step 1: Read the PowerShell/SSH failure catalog and run direct preflight**

Use the project-owned read-onlypreflight; use the lab bridge only if direct connectivity alone fails.

- [ ] **Step 2: Build one release archive and compare one local/remote SHA**

Compile remotely once. No member hashes or extra receipts.

- [ ] **Step 3: Launch D0–D4 with at most two training jobs per GPU**

Immediately verifyPID/CWD/cmdline/run-root/GPU/log growth. Never stop for low performance.

- [ ] **Step 4: Monitor to five complete prediction receipts**

Report only completed/active/error until all predictions close. Do not read truth.

- [ ] **Step 5: Run independent scorer once**

ReportD0–D4 BA、floor、per-class accuracy、NLL、ECE、prediction discordance、trainable/changed elements、delta bytes、wall clock、RSS andGPU peaks.

- [ ] **Step 6: Apply preregistered same-row promotion rules**

Choose the smallest candidate satisfying all performance/resource gates. Negative rows remain valid evidence and are not rerun.

- [ ] **Step 7: Publish final report**

Precisely stage report/traceability updates, commit、push and independently verify remoteOID equals localHEAD.
