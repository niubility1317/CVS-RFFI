from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from cvsrffi import stage2_next_r5_fa_target125_matrix as matrix
from cvsrffi import stage2_next_r5_fa_target125_core as core
from cvsrffi import stage2_next_r5_fa_target125_runtime as runtime
from cvsrffi.stage2_next_r5_fa_target125_runtime import (
    NextR5FATarget125RuntimeError,
    Target125ConditionInput,
    Target125RegistrationInput,
    build_target125_runtime_bindings,
    query_isolation_receipt,
)


_PR160_BINDING = {
    "schema": runtime.PR160_RUNTIME_BINDING_SCHEMA,
    "source_runtime_sha256": runtime.PR160_SOURCE_RUNTIME_SHA256,
    "extractor_runtime_sha256": runtime.PR160_EXTRACTOR_RUNTIME_SHA256,
    "extractor_runtime_size_bytes": runtime.PR160_EXTRACTOR_RUNTIME_SIZE_BYTES,
}


def _rows(count: int, *, offset: int = 0) -> np.ndarray:
    value = np.zeros((count, matrix.FEATURE_DIM), dtype=np.float32)
    for index in range(count):
        value[index, (offset + index) % matrix.FEATURE_DIM] = 1.0
    return value


def _totalize(
    pre_relu: np.ndarray, physical_ids: tuple[str, ...], *, scope: str
) -> tuple[np.ndarray, object]:
    return runtime._totalize_same_iq_zid160(  # noqa: SLF001 - direct boundary test.
        pre_relu,
        np.maximum(pre_relu, np.float32(0.0)),
        physical_ids,
        scope=scope,
        pr160_runtime_binding=_PR160_BINDING,
    )


def _condition() -> Target125ConditionInput:
    outer = next(row for row in matrix.freeze_next_r5_fa_target125_matrix().outer_rows if row.k_shot == 1)
    old = tuple(f"old-{index}" for index in range(matrix.OLD_CLASS_COUNT))
    new = tuple(f"new-{index}" for index in range(outer.new_count))
    old_ids = tuple(f"old-support-{index}" for index in range(len(old)))
    reg0_support = _rows(len(old))
    reg0_query = _rows(len(old), offset=40)
    reg0_query_ids = tuple(f"old-query-{index}" for index in range(len(old)))
    reg0_support, reg0_support_receipt = _totalize(
        reg0_support, old_ids, scope="REG0_support"
    )
    reg0_query, reg0_query_receipt = _totalize(
        reg0_query, reg0_query_ids, scope="REG0_query"
    )
    reg0 = Target125RegistrationInput(
        registration_phase="REG0",
        registered_classes=old,
        registered_class_indices=tuple(range(len(old))),
        support_zid160=reg0_support,
        support_labels=old,
        support_physical_ids=old_ids,
        query_zid160=reg0_query,
        query_physical_ids=reg0_query_ids,
        support_totalization_receipt=reg0_support_receipt,
        query_totalization_receipt=reg0_query_receipt,
    )
    reg1_support_ids = old_ids + tuple(f"new-support-{index}" for index in range(len(new)))
    reg1_query_ids = tuple(f"reg1-query-{index}" for index in range(len(old) + len(new)))
    reg1_support, reg1_support_receipt = _totalize(
        np.vstack((_rows(len(old)), _rows(len(new), offset=20))).astype(np.float32),
        reg1_support_ids,
        scope="REG1_support",
    )
    reg1_query, reg1_query_receipt = _totalize(
        _rows(len(old) + len(new), offset=80), reg1_query_ids, scope="REG1_query"
    )
    reg1 = Target125RegistrationInput(
        registration_phase="REG1",
        registered_classes=old + new,
        registered_class_indices=tuple(range(len(old) + len(new))),
        support_zid160=reg1_support,
        support_labels=old + new,
        support_physical_ids=reg1_support_ids,
        query_zid160=reg1_query,
        query_physical_ids=reg1_query_ids,
        support_totalization_receipt=reg1_support_receipt,
        query_totalization_receipt=reg1_query_receipt,
    )
    source_row = {
        "outer_id": "d108-source-row",
        "receiver": outer.receiver,
        "seed": outer.seed,
        "k_shot": outer.k_shot,
        "active_k": outer.k_shot,
        "new_count": outer.new_count,
        "source_pool_k": outer.source_pool_k,
        "packages": {},
        "authority_bundle": {},
    }
    return Target125ConditionInput(
        outer_row=outer,
        scene=matrix.SCENES[0],
        source_row=source_row,
        reg0=reg0,
        reg1=reg1,
    )


def test_condition_preserves_reg0_old_support_order_and_builds_core_bindings() -> None:
    condition = _condition()
    token = hashlib.sha256(b"binding").hexdigest()
    source_plan = {
        "identity": {
            "d92_matrix_manifest": {"sha256": token},
            "checkpoint": {"sha256": token},
            "d92_sealed_runtime_sha256": token,
        },
        "plan_receipt_sha256": token,
    }
    reg0, reg1 = build_target125_runtime_bindings(source_plan=source_plan, condition=condition)
    assert reg0.support_physical_ids == condition.reg0.support_physical_ids
    assert reg0.registered_class_indices == tuple(range(matrix.OLD_CLASS_COUNT))
    assert reg1.registered_class_indices == tuple(range(matrix.OLD_CLASS_COUNT + condition.outer_row.new_count))
    assert reg1.support_physical_ids == condition.reg0.support_physical_ids + tuple(
        condition.reg1.support_physical_ids[matrix.OLD_CLASS_COUNT :]
    )
    assert reg0.query_physical_ids == condition.reg0.query_physical_ids
    assert reg1.query_physical_ids == condition.reg1.query_physical_ids


def test_query_isolation_and_reg1_old_support_reorder_fail_closed() -> None:
    condition = _condition()
    assert query_isolation_receipt()["query_truth_access"] is False
    assert query_isolation_receipt()["phase2_optimizer_steps"] == 0
    reordered_ids = (
        condition.reg1.support_physical_ids[1],
        condition.reg1.support_physical_ids[0],
        *condition.reg1.support_physical_ids[2:],
    )
    _rows_bound, reordered_receipt = _totalize(
        np.asarray(condition.reg1.support_zid160), reordered_ids, scope="REG1_support"
    )
    reordered = Target125RegistrationInput(
        registration_phase="REG1",
        registered_classes=condition.reg1.registered_classes,
        registered_class_indices=condition.reg1.registered_class_indices,
        support_zid160=condition.reg1.support_zid160,
        support_labels=(condition.reg1.support_labels[1], condition.reg1.support_labels[0], *condition.reg1.support_labels[2:]),
        support_physical_ids=reordered_ids,
        query_zid160=condition.reg1.query_zid160,
        query_physical_ids=condition.reg1.query_physical_ids,
        support_totalization_receipt=reordered_receipt,
        query_totalization_receipt=condition.reg1.query_totalization_receipt,
    )
    with pytest.raises(NextR5FATarget125RuntimeError, match="byte-preserve"):
        Target125ConditionInput(
            outer_row=condition.outer_row,
            scene=condition.scene,
            source_row=condition.source_row,
            reg0=condition.reg0,
            reg1=reordered,
        )


def test_real_relu_zero_row_uses_only_same_iq_signed_pre_relu_direction() -> None:
    pre_relu = np.zeros((2, matrix.FEATURE_DIM), dtype=np.float32)
    pre_relu[0, 0] = -2.0
    pre_relu[0, 1] = 3.0
    pre_relu[1, :] = -1.0
    graph_post_relu = np.maximum(pre_relu, np.float32(0.0))
    result, receipt = runtime._totalize_same_iq_zid160(  # noqa: SLF001
        pre_relu,
        graph_post_relu,
        ("physical-positive", "physical-relu-zero"),
        scope="REG1_support",
        pr160_runtime_binding=_PR160_BINDING,
    )
    from cvsrffi.stage2_zid_student_t_qknn import normalize_zid_rows

    # Every nonzero row follows the canonical R0 ReLU-normalization path; only
    # the exact same-graph zero uses the signed pre-ReLU direction.
    assert np.array_equal(result[0], normalize_zid_rows(graph_post_relu[:1])[0])
    assert result[0, 0] == 0.0
    assert result[0, 1] == pytest.approx(1.0)
    assert np.allclose(
        result[1], np.full(matrix.FEATURE_DIM, -1.0 / np.sqrt(matrix.FEATURE_DIM), dtype=np.float32)
    )
    assert receipt["replaced_count"] == 1
    assert receipt["same_fixed_received_iq"] is True
    assert receipt["query_truth_access"] is False
    assert receipt["ordered_physical_ids_sha256"] == matrix.canonical_sha256(
        ["physical-positive", "physical-relu-zero"]
    )
    assert receipt["physical_id_to_pr160_graph_relu_row_root_sha256"]
    assert receipt["pre_relu_to_pr160_graph_relu_binding"]["all_rows_bound"] is True
    assert receipt["pr160_runtime_binding"] == _PR160_BINDING
    qknn = core.fit_qknn(
        result,
        ("class-positive", "class-relu-zero"),
        ("class-positive", "class-relu-zero"),
        support_physical_ids=("physical-positive", "physical-relu-zero"),
        representation=core.R0_REPRESENTATION,
    )
    logits = core.score_qknn(qknn, result)
    assert logits.shape == (2, 2)
    assert np.isfinite(logits).all()

    exact_zero = np.zeros((1, matrix.FEATURE_DIM), dtype=np.float32)
    with pytest.raises(NextR5FATarget125RuntimeError, match="exact-zero pre-ReLU"):
        runtime._totalize_same_iq_zid160(  # noqa: SLF001
            exact_zero,
            exact_zero,
            ("physical-exact-zero",),
            scope="REG1_query",
            pr160_runtime_binding=_PR160_BINDING,
        )


def test_same_graph_relu_binding_drift_fails_closed() -> None:
    pre_relu = _rows(2)
    sealed_post_relu = np.maximum(pre_relu, np.float32(0.0))
    sealed_post_relu[0, 0] += np.float32(1.0e-3)
    with pytest.raises(NextR5FATarget125RuntimeError, match="ReLU binding drift"):
        runtime._totalize_same_iq_zid160(  # noqa: SLF001 - direct boundary test.
            pre_relu,
            sealed_post_relu,
            ("physical-0", "physical-1"),
            scope="REG0_query",
            pr160_runtime_binding=_PR160_BINDING,
        )

    # A nonzero pre-ReLU coordinate cannot be converted into a graph zero to
    # unlock the signed fallback.
    almost_zero_pre = np.full((1, matrix.FEATURE_DIM), -1.0, dtype=np.float32)
    almost_zero_pre[0, 0] = np.float32(1.0e-7)
    with pytest.raises(NextR5FATarget125RuntimeError, match="same-graph pre-ReLU / ReLU binding drift"):
        runtime._totalize_same_iq_zid160(  # noqa: SLF001 - direct boundary test.
            almost_zero_pre,
            np.zeros_like(almost_zero_pre),
            ("physical-near-zero",),
            scope="REG1_query",
            pr160_runtime_binding=_PR160_BINDING,
        )


def test_materializer_uses_one_pr160_graph_for_normal_and_zero_rows(monkeypatch) -> None:
    from cvsrffi import stage2_diag_cosine_exploration as diag

    pre_relu = np.zeros((2, matrix.FEATURE_DIM), dtype=np.float32)
    pre_relu[0, 0] = 2.0
    pre_relu[1, :] = -1.0
    graph_post_relu = np.maximum(pre_relu, np.float32(0.0))
    calls: list[tuple[str, object]] = []

    def pr160_forward(model, rows, **_kwargs):
        calls.append(("pr160", model))
        assert np.array_equal(rows, np.zeros((2, 2, 8), dtype=np.float32))
        return pre_relu

    monkeypatch.setattr(diag, "forward_zid160", pr160_forward)
    materializer = object.__new__(runtime.D108ZID160Materializer)
    materializer._device = object()  # noqa: SLF001 - isolated runtime boundary.
    materializer._pr160_extractor_model = "pr160-model"  # noqa: SLF001
    materializer._pr160_runtime_binding = _PR160_BINDING  # noqa: SLF001

    result, receipt = materializer._zid160(  # noqa: SLF001 - direct materializer boundary.
        iq=np.zeros((2, 2, 8), dtype=np.float32),
        physical_ids=("normal", "exact-graph-zero"),
        scope="REG1_query",
        batch_size=1,
    )
    from cvsrffi.stage2_zid_student_t_qknn import normalize_zid_rows

    assert calls == [("pr160", "pr160-model")]
    assert np.array_equal(result[0], normalize_zid_rows(graph_post_relu[:1])[0])
    assert receipt["replaced_count"] == 1
    assert receipt["nonzero_rows_reuse_original_r0_semantics"] is True


def test_support_cache_reuses_reg0_prefix_across_reg1_batch_shapes(monkeypatch) -> None:
    """REG1 must forward only its new support suffix, never its old prefix."""

    from cvsrffi import stage2_diag_cosine_exploration as diag

    outer = next(
        row
        for row in matrix.freeze_next_r5_fa_target125_matrix().outer_rows
        if row.k_shot == 10 and row.new_count == 5
    )
    scene = matrix.SCENES[0]
    old = tuple(f"old-{index}" for index in range(matrix.OLD_CLASS_COUNT))
    new = tuple(f"new-{index}" for index in range(outer.new_count))
    old_count = len(old) * outer.k_shot
    new_count = len(new) * outer.k_shot
    old_ids = tuple(f"old-support-{index}" for index in range(old_count))
    new_ids = tuple(f"new-support-{index}" for index in range(new_count))
    before_support = np.zeros((old_count, 2, 8), dtype=np.float32)
    after_support = np.zeros((old_count + new_count, 2, 8), dtype=np.float32)
    for index in range(old_count):
        before_support[index, 0, 0] = np.float32(index + 1)
    after_support[:old_count] = before_support
    for index in range(new_count):
        after_support[old_count + index, 0, 0] = np.float32(1000 + index)
    shared_query = np.zeros((2, 2, 8), dtype=np.float32)
    shared_query[:, 0, 0] = np.asarray((2001.0, 2002.0), dtype=np.float32)
    query_ids = ("query-0", "query-1")
    after_ids = list(old_ids + new_ids)
    before_labels = tuple(label for label in old for _ in range(outer.k_shot))
    after_labels = before_labels + tuple(
        label for label in new for _ in range(outer.k_shot)
    )

    def reference(name: str) -> dict[str, str]:
        return {
            "package_root": f"/sealed/{name}",
            "detached_seal_path": f"/sealed/{name}.seal.json",
            "expected_seal_sha256": hashlib.sha256(name.encode("ascii")).hexdigest(),
        }

    packages = {
        "before_enrollment": reference("before-enrollment"),
        "before_apply": reference("before-apply"),
        "after_enrollment": reference("after-enrollment"),
        "after_apply": reference("after-apply"),
    }

    def manifest(classes: tuple[str, ...]) -> dict[str, object]:
        return {
            "registered_classes": [
                {"class_index": index, "class_handle": label}
                for index, label in enumerate(classes)
            ],
            "receiver": outer.receiver,
            "seed": outer.seed,
            "k_shot": outer.source_pool_k,
        }

    payloads = {
        "/sealed/before-enrollment": ({scene: "before-support"}, manifest(old)),
        "/sealed/before-apply": ({scene: "shared-query"}, manifest(old)),
        "/sealed/after-enrollment": ({scene: "after-support"}, manifest(old + new)),
        "/sealed/after-apply": ({scene: "shared-query"}, manifest(old + new)),
    }

    class FakeD108:
        @staticmethod
        def _support_rows(payload, *, registered_classes, active_k):
            assert active_k == outer.k_shot
            if payload == "before-support":
                assert tuple(registered_classes) == old
                return before_support.copy(), before_labels, old_ids
            assert payload == "after-support"
            assert tuple(registered_classes) == old + new
            return after_support.copy(), after_labels, tuple(after_ids)

        @staticmethod
        def _query_rows(payload):
            assert payload == "shared-query"
            return shared_query.copy(), query_ids

    calls: list[tuple[int, int]] = []

    def pr160_forward(_model, rows, *, batch_size, **_kwargs):
        rows = np.asarray(rows, dtype=np.float32)
        calls.append((len(rows), batch_size))
        output = np.zeros((len(rows), matrix.FEATURE_DIM), dtype=np.float32)
        for index, row in enumerate(rows):
            token = int(row[0, 0])
            output[index, token % matrix.FEATURE_DIM] = 1.0
            output[index, (token + 1) % matrix.FEATURE_DIM] = np.float32(0.25)
            # Deliberately make an otherwise repeated GPU forward depend on
            # its outer batch shape, reproducing the r7 failure mechanism.
            output[index, (token + 2) % matrix.FEATURE_DIM] = np.float32(
                len(rows) * 1.0e-4
            )
        return output

    def fake_package(_self, reference_value):
        return payloads[reference_value["package_root"]]

    monkeypatch.setattr(diag, "forward_zid160", pr160_forward)
    monkeypatch.setattr(diag, "_validate_matched_packages", lambda *_args: None)
    monkeypatch.setattr(runtime.D108ZID160Materializer, "_package", fake_package)
    monkeypatch.setattr(
        runtime.D108ZID160Materializer,
        "_require_package_source_runtime",
        lambda _self, _manifest: None,
    )
    materializer = object.__new__(runtime.D108ZID160Materializer)
    materializer._d108 = FakeD108()  # noqa: SLF001 - isolated sealed-package adapter.
    materializer._device = object()  # noqa: SLF001
    materializer._pr160_extractor_model = "pr160-model"  # noqa: SLF001
    materializer._pr160_runtime_binding = _PR160_BINDING  # noqa: SLF001
    materializer._support_batch_size = 64  # noqa: SLF001
    materializer._support_feature_cache = {}  # noqa: SLF001
    materializer._source_plan = {  # noqa: SLF001
        "plan_receipt_sha256": hashlib.sha256(b"synthetic-source-plan").hexdigest()
    }
    source_row = {
        "outer_id": outer.outer_id,
        "receiver": outer.receiver,
        "seed": outer.seed,
        "k_shot": outer.k_shot,
        "active_k": outer.k_shot,
        "new_count": outer.new_count,
        "source_pool_k": outer.source_pool_k,
        "packages": packages,
        "authority_bundle": {},
    }

    condition = materializer.materialize_condition(  # noqa: SLF001 - exact phase boundary.
        outer_row=outer,
        source_row=source_row,
        scene=scene,
    )
    assert np.array_equal(
        condition.reg1.support_zid160[:old_count], condition.reg0.support_zid160
    )
    assert calls == [(old_count, 64), (2, 1), (new_count, 64), (2, 1)]
    assert len(materializer._support_feature_cache) == old_count + new_count  # noqa: SLF001

    after_ids[0], after_ids[1] = after_ids[1], after_ids[0]
    with pytest.raises(NextR5FATarget125RuntimeError, match="physical-ID prefix drift"):
        materializer.materialize_condition(  # noqa: SLF001 - fail-closed cache binding.
            outer_row=outer,
            source_row=source_row,
            scene=scene,
        )
    after_ids[0], after_ids[1] = after_ids[1], after_ids[0]
    after_support[0, 0, 0] += np.float32(0.5)
    with pytest.raises(NextR5FATarget125RuntimeError, match="received-IQ/cache binding drift"):
        materializer.materialize_condition(  # noqa: SLF001 - fail-closed byte binding.
            outer_row=outer,
            source_row=source_row,
            scene=scene,
        )


def test_totalization_receipt_rejects_physical_binding_drift() -> None:
    pre_relu = _rows(1)
    result, receipt = _totalize(pre_relu, ("physical-original",), scope="REG0_support")
    with pytest.raises(NextR5FATarget125RuntimeError, match="receipt binding drift"):
        runtime._validate_totalization_receipt(  # noqa: SLF001 - direct boundary test.
            receipt,
            rows=result,
            physical_ids=("physical-swapped",),
            scope="REG0_support",
        )


def test_prepared_pr160_runtime_requires_exact_source_plan_binding(monkeypatch) -> None:
    source_sha = runtime.PR160_SOURCE_RUNTIME_SHA256
    extractor_sha = runtime.PR160_EXTRACTOR_RUNTIME_SHA256
    descriptor = {
        "path": "C:/sealed/pr160.pt",
        "sha256": extractor_sha,
        "size_bytes": runtime.PR160_EXTRACTOR_RUNTIME_SIZE_BYTES,
        "source_runtime_sha256": source_sha,
    }
    source_plan = {"identity": {"d92_sealed_runtime_sha256": source_sha}}
    plan = {"identity": {"d92_sealed_runtime_sha256": source_sha, "pr160_extractor_runtime": descriptor}}
    calls: list[dict[str, object]] = []

    def bind(**kwargs: object) -> dict[str, object]:
        calls.append(dict(kwargs))
        return dict(descriptor)

    monkeypatch.setattr(runtime, "_bind_pr160_extractor_runtime", bind)
    assert runtime._prepared_pr160_extractor_runtime(  # noqa: SLF001
        plan=plan, source_plan=source_plan
    ) == descriptor
    assert calls == [
        {
            "extractor_runtime_path": descriptor["path"],
            "expected_extractor_runtime_sha256": extractor_sha,
            "source_runtime_sha256": source_sha,
            "expected_size_bytes": runtime.PR160_EXTRACTOR_RUNTIME_SIZE_BYTES,
        }
    ]
    tampered = {"identity": {**plan["identity"], "d92_sealed_runtime_sha256": "0" * 64}}
    with pytest.raises(NextR5FATarget125RuntimeError, match="source runtime binding drift"):
        runtime._prepared_pr160_extractor_runtime(  # noqa: SLF001
            plan=tampered, source_plan=source_plan
        )


def test_pr160_extractor_file_sha_and_size_are_fail_closed(tmp_path, monkeypatch) -> None:
    extractor = tmp_path / "pr160.pt"
    extractor.write_bytes(b"sealed-pr160-graph")
    sha256 = hashlib.sha256(extractor.read_bytes()).hexdigest()
    monkeypatch.setattr(runtime, "PR160_EXTRACTOR_RUNTIME_SHA256", sha256)
    monkeypatch.setattr(
        runtime, "PR160_EXTRACTOR_RUNTIME_SIZE_BYTES", extractor.stat().st_size
    )
    bound = runtime._bind_pr160_extractor_runtime(  # noqa: SLF001
        extractor_runtime_path=extractor,
        expected_extractor_runtime_sha256=sha256,
        source_runtime_sha256=runtime.PR160_SOURCE_RUNTIME_SHA256,
        expected_size_bytes=extractor.stat().st_size,
    )
    assert bound["path"] == str(extractor.resolve())
    extractor.write_bytes(b"tampered-pr160-graph")
    with pytest.raises(NextR5FATarget125RuntimeError, match="file binding drift"):
        runtime._bind_pr160_extractor_runtime(  # noqa: SLF001
            extractor_runtime_path=extractor,
            expected_extractor_runtime_sha256=sha256,
            source_runtime_sha256=runtime.PR160_SOURCE_RUNTIME_SHA256,
            expected_size_bytes=len(b"sealed-pr160-graph"),
        )


def test_sealed_package_class_records_require_exact_continuous_bridge() -> None:
    records = [
        {"class_index": index, "class_handle": f"cls_{index}"}
        for index in range(matrix.OLD_CLASS_COUNT)
    ]
    assert runtime._registered_class_records(  # noqa: SLF001 - exact package boundary test.
        {"registered_classes": records}, name="test package"
    ) == (tuple(range(6)), tuple(f"cls_{index}" for index in range(6)))
    bad = [dict(item) for item in records]
    bad[-1]["class_index"] = 7
    with pytest.raises(NextR5FATarget125RuntimeError, match="record/index drift"):
        runtime._registered_class_records(  # noqa: SLF001 - exact package boundary test.
            {"registered_classes": bad}, name="test package"
        )


def test_fa_asset_method_lock_mismatch_fails_closed_on_reload(tmp_path) -> None:
    def sha(value: str) -> str:
        return hashlib.sha256(value.encode("ascii")).hexdigest()

    basis = np.zeros((core.FA_RANK, core.Z_DIM), dtype=np.float32)
    basis[0, 0] = basis[1, 1] = basis[2, 2] = 1.0
    asset = core.build_target_fa_asset(
        old_classes=tuple(f"old-{index}" for index in range(core.OLD_CLASS_COUNT)),
        aggregate_samples_per_class=(98,) * core.OLD_CLASS_COUNT,
        class_centers_3d=np.zeros((core.OLD_CLASS_COUNT, core.FA_RANK), dtype=np.float32),
        fisher_precision_3d=np.ones(core.FA_RANK, dtype=np.float32),
        residual_variance_3d=np.ones(core.FA_RANK, dtype=np.float32),
        fisher_radius=np.asarray([np.sqrt(3.0)], dtype=np.float32),
        rdce_kappa_3d=np.asarray([0.2, 0.1, 0.05], dtype=np.float32),
        basis_3x160=basis,
        checkpoint_sha256=sha("checkpoint"),
        phase1_bundle_sha256=sha("phase1-bundle"),
        phase1_aggregate_receipt_sha256=sha("aggregate"),
        method_lock_sha256=sha("asset-method-lock"),
    )
    wire = core.serialize_target_fa_asset(asset)
    path = tmp_path / "fa.wire"
    path.write_bytes(wire)
    plan = {
        "identity": {
            "fa_asset": {"path": str(path), "sha256": hashlib.sha256(wire).hexdigest()},
            "checkpoint_sha256": sha("checkpoint"),
            "method_lock": {"sha256": sha("different-method-lock")},
        }
    }
    with pytest.raises(NextR5FATarget125RuntimeError, match="method-lock binding drift"):
        runtime._load_target_asset(plan)  # noqa: SLF001 - negative release-boundary test.


def test_method_lock_requires_same_iq_signed_totalization_contract(tmp_path) -> None:
    source = (
        Path(__file__).resolve().parents[1]
        / "configs"
        / "next_r5_fa_rdce3_q_target125_20260805.json"
    )
    raw = source.read_bytes()
    verified = runtime._validate_method_lock(  # noqa: SLF001 - exact release boundary.
        source, hashlib.sha256(raw).hexdigest()
    )
    assert verified["representation"]["r0"] == core.R0_REPRESENTATION

    tampered = json.loads(raw.decode("utf-8"))
    tampered["representation"]["r0"] = "d106_canonical_normalized_relu_zid160"
    path = tmp_path / "tampered-method-lock.json"
    path.write_text(json.dumps(tampered), encoding="utf-8")
    with pytest.raises(NextR5FATarget125RuntimeError, match="method-lock identity/count drift"):
        runtime._validate_method_lock(  # noqa: SLF001 - exact release boundary.
            path, hashlib.sha256(path.read_bytes()).hexdigest()
        )
