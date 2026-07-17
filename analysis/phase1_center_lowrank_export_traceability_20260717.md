# Phase1 center-lowrank export traceability

| ID | Source section | Requirement | Target files | Status | Verification | Notes |
|----|----------------|-------------|--------------|--------|--------------|-------|
| EX-01 | User request | Export from normalized `z_id` in two no-gradient passes without sample retention | `code/scripts/export_adv3b02_center_lowrank_radius_component.py` | verified | synthetic two-pass loader test | Pure `export_from_loader` boundary |
| EX-02 | User request | Rebuild the exact SSDG checkpoint and reuse its Phase1 labeled-train split and batch/domain contract | same script | verified | CLI help, source reachability, shared codec regression | CLI uses checkpoint loader plus `_build_ssdg_wisig_data` and `probe_train_loader` |
| EX-03 | User request | Bind checkpoint, ManySig, class mapping, generation config, code, ordered two-pass stream and aggregate-radius proof before export | same script | verified | hash mismatch, class-order, stream-order and wrong-proof tests | Standalone component deliberately has no signature; it emits a pre-sign root for the later joint seal |
| EX-04 | User request | Construct v1 aggregate only in memory and save through the v2 codec | same script | verified | synthetic codec save and output allowlist test | No full-precision PT/JSON intermediate |
| EX-05 | AGENTS protocol | Bundle contains no sample feature, count, path, query, role, or quota fields | script and tests | verified | NPZ/manifest member audit | Final output equals codec three-file allowlist |
| EX-06 | Version workflow | Authoritative implementation is versioned in the Git release workspace before N607 sync | Git script/tests/traceability | verified | focused tests, compile, diff check and scoped Git commit | Root is not a Git repository; protocol/report surfaces are mirrored separately |

## 2026-07-17 P0 protocol repair

| ID | Source section | Requirement | Target files | Status | Verification | Notes |
|----|----------------|-------------|--------------|--------|--------------|-------|
| P0-01 | Independent protocol review | Remove the component-local detached-signature placeholder and signature/root cycle; generate a pre-sign content root and require an outer joint seal | codec, exporter, codec/exporter tests | verified | focused pytest and CLI parser negative test | Manifest has no detached-signature field; `pre_sign_content_root_sha256` is the outer signing input |
| P0-02 | Independent protocol review | Hash the ordered normalized `z_id` plus class/domain stream independently in both passes and fail unless both stream hashes match | exporter, exporter tests | verified | same-sum/different-stream negative test | Hash is row-order-sensitive and does not persist sample material |
| P0-03 | Independent protocol review | Bind a generation-proof hash, actual v1 payload hash, registry hash, checkpoint/class binding/code/config hashes in the component manifest | codec, exporter, tests | verified | wrong-proof negative test and manifest validation | Codec recomputes the actual v1 payload hash and expected radius proof |
| P0-04 | Independent protocol review | Mark the output `PHASE1_COMPONENT_PENDING_OUTER_JOINT_SEAL`; allow Phase1 offline generation but block the formal Phase2 loader | codec, exporter, tests | verified | formal-load negative test and CLI help | Outer checkpoint+component+registry seal remains intentionally deferred |

### P0 repair verification

- `python -m py_compile code/cvsrffi/phase1_center_lowrank_prototype_bundle.py code/scripts/export_adv3b02_center_lowrank_radius_component.py tests/test_phase1_center_lowrank_prototype_bundle.py tests/test_export_adv3b02_center_lowrank_radius_component.py`：通过。
- `PYTHONPATH=code python -m pytest -q -p no:cacheprovider tests/test_phase1_center_lowrank_prototype_bundle.py tests/test_phase1_geometry_streaming.py tests/test_export_adv3b02_center_lowrank_radius_component.py`（`ssr-gpu`）：`28 passed`；结束后的Windows pytest临时目录清理产生已知`PermissionError`，测试进程退出码为0。
- exporter `--help`：通过；CLI不再接收任何detached-signature参数。
- P0修复状态：`verified=4`、`deferred=0`、`rejected=0`、`blocked=0`。外层共同bundle签名是后续阶段，不属于本次standalone component实现。

## Approximation boundary

The exporter uses the streaming layer's fixed-bin upper-edge estimate of empirical nearest-rank P90 cosine distance. Its deterministic overestimate is bounded by `2 / radius_histogram_bins`; all other requested bindings and allowlists are strict.

## Verification record

- `PYTHONPATH=code conda run -n ssr-gpu --no-capture-output python -m pytest -q -p no:cacheprovider tests/test_export_adv3b02_center_lowrank_radius_component.py tests/test_phase1_geometry_streaming.py tests/test_phase1_center_lowrank_prototype_bundle.py`: `26 passed`.
- `conda run -n ssr-gpu --no-capture-output python -m py_compile code/scripts/export_adv3b02_center_lowrank_radius_component.py tests/test_export_adv3b02_center_lowrank_radius_component.py`: passed.
- CLI `--help`: passed with all required provenance bindings visible.
- Root/mirror SHA256 comparison: script, tests, and this record are byte-identical.
- Real checkpoint plus ManySig execution is intentionally left for the authorized N607 CLI run; no remote action was performed here.
