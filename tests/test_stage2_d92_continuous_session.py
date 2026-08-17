from __future__ import annotations

import importlib
import importlib.util
import inspect

import numpy as np
import pytest
from sklearn.covariance import ledoit_wolf
from sklearn.preprocessing import StandardScaler

from cvsrffi import stage2_d42_unified_shrinkage_lda as d42
from cvsrffi import stage2_d92_registration_balanced_covariance as d92


_TEST_LOG_DIAG = np.zeros(d42.FEATURE_DIM, dtype=np.float32)


def _module():
    return importlib.import_module("cvsrffi.stage2_d92_continuous_session")


def _identity_transform(
    rows: np.ndarray, labels: np.ndarray, class_count: int, k_shot: int
) -> np.ndarray:
    assert rows.shape == (class_count * k_shot, d42.FEATURE_DIM)
    assert labels.shape == (len(rows),)
    # Fixture-only identity D81 translation after the explicit frozen D42 map.
    return d42._transform(np.asarray(rows, dtype=np.float32), _TEST_LOG_DIAG)


def _packet(
    module,
    handle: str,
    *,
    arrival_session: int,
    seed: int,
    permutation: np.ndarray | None = None,
    tokens: tuple[str, ...] | None = None,
):
    rng = np.random.default_rng(seed)
    center = rng.normal(size=d42.FEATURE_DIM).astype(np.float32)
    center[(seed * 17) % d42.FEATURE_DIM] += np.float32(7.0)
    rows = center[None, :] + np.float32(0.08) * rng.normal(
        size=(10, d42.FEATURE_DIM)
    ).astype(np.float32)
    physical_tokens = tokens or tuple(f"{handle}:token:{index}" for index in range(10))
    if permutation is not None:
        rows = rows[permutation]
        physical_tokens = tuple(physical_tokens[index] for index in permutation)
    return module.SupportPacket(
        handle=handle,
        rows=rows,
        physical_tokens=physical_tokens,
        package_id=f"pkg:{handle}",
        arrival_session=arrival_session,
    )


def _ledger(module, *, old_permutation: tuple[int, ...] | None = None):
    handles = [f"old_{index}" for index in range(6)]
    if old_permutation is not None:
        handles = [handles[index] for index in old_permutation]
    old_packets = [
        _packet(module, handle, arrival_session=0, seed=100 + int(handle[-1]))
        for handle in handles
    ]
    anchor = module.FrozenDAAnchor.from_old_support(
        old_packets,
        da_anchor_id="frozen-da-anchor-v1",
        support_transform=_identity_transform,
        log_diag_fp32=_TEST_LOG_DIAG,
    )
    return module.SessionLedger.start(anchor)


def _advance(module, ledger, new_count: int):
    packets = [
        _packet(
            module,
            f"new_{index}",
            arrival_session=ledger.next_session,
            seed=200 + index,
        )
        for index in range(new_count)
    ]
    return module.advance_session(ledger, packets)


def _d42_unit_support(
    prefix: str, class_count: int, *, offset: int
) -> tuple[np.ndarray, np.ndarray, tuple[str, ...]]:
    classes = tuple(f"{prefix}_{index}" for index in range(class_count))
    rows = np.zeros((class_count, d42.FEATURE_DIM), dtype=np.float32)
    for index in range(class_count):
        rows[index, offset + index] = np.float32(1.0)
    return rows, np.asarray(classes), classes


def test_continuous_session_public_module_is_available() -> None:
    """This fails before Task1 production code exists."""

    assert importlib.util.find_spec("cvsrffi.stage2_d92_continuous_session") is not None


def test_s1_singleton_uses_standard_scaler_ledoit_wolf_bridge() -> None:
    module = _module()
    ledger = _ledger(module)
    packet = _packet(module, "new_0", arrival_session=1, seed=200)

    result = module.advance_session(ledger, [packet])

    singleton_rows = np.asarray(
        result.transformed_rows[result.targets == 6], dtype=np.float64
    )
    centered = singleton_rows - singleton_rows.mean(axis=0, keepdims=True)
    scaler = StandardScaler()
    standardized = scaler.fit_transform(centered)
    standardized_covariance, _ = ledoit_wolf(standardized)
    expected_new = (
        scaler.scale_[:, None]
        * standardized_covariance
        * scaler.scale_[None, :]
    )
    np.testing.assert_allclose(
        result.statistics.new_covariance, expected_new, rtol=0.0, atol=1e-12
    )
    np.testing.assert_allclose(
        result.statistics.covariance,
        0.5 * result.statistics.old_covariance
        + 0.5 * result.statistics.new_covariance,
        rtol=0.0,
        atol=0.0,
    )
    assert result.audit["d92_continuous_bridge_active"] is True
    assert result.audit["d92_continuous_bridge_policy"] == (
        "standard_scaler_ledoit_wolf_singleton"
    )
    assert result.statistics.covariance_audit[
        "d92_group_local_shrinkage_estimation_count"
    ] == 2
    assert result.audit["d92_continuous_original_e0_equivalent"] is False


@pytest.mark.parametrize("new_count", [2, 3, 4])
def test_s2_to_s4_use_the_d92_group_formula_prefix(new_count: int) -> None:
    module = _module()
    result = _advance(module, _ledger(module), new_count)
    rows = result.transformed_rows.astype(np.float64)
    targets = result.targets
    expected_old = d92._group_covariance(
        d42, rows, targets, np.arange(6, dtype=np.int64)
    )
    expected_new = d92._group_covariance(
        d42, rows, targets, np.arange(6, 6 + new_count, dtype=np.int64)
    )
    np.testing.assert_array_equal(result.statistics.old_covariance, expected_old)
    np.testing.assert_array_equal(result.statistics.new_covariance, expected_new)
    np.testing.assert_array_equal(
        result.statistics.covariance, 0.5 * expected_old + 0.5 * expected_new
    )
    assert result.audit["d92_continuous_bridge_active"] is False
    assert result.audit["d92_continuous_original_e0_equivalent"] is False


@pytest.mark.parametrize("new_count", [1, 3, 4])
def test_intermediate_continuous_state_converts_to_original_d42_score_and_predict(
    new_count: int,
) -> None:
    module = _module()
    result = _advance(module, _ledger(module), new_count)
    state = module.to_d42_unified_state(result.state)
    raw_rows = np.concatenate(
        [
            record.rows
            for record in result.ledger.anchor.old_records
            + result.ledger.arrived_records
        ],
        axis=0,
    ).astype(np.float32)

    scores = d42.score_d42_unified_shrinkage_lda(state, raw_rows)
    decoded = d42.decode_d42_coefficients(state)
    transformed = d42._transform(raw_rows, state.log_diag_fp32)
    expected_scores = np.stack(
        [
            np.asarray(
                row @ decoded.T + state.intercept_fp16.astype(np.float32),
                dtype=np.float32,
            )
            for row in transformed
        ]
    )
    np.testing.assert_array_equal(scores, expected_scores)
    np.testing.assert_array_equal(
        d42.predict_d42_unified_shrinkage_lda(state, raw_rows),
        np.asarray(state.classes)[np.argmax(expected_scores, axis=1)],
    )
    assert state.classes == result.state.classes
    assert state.old_class_count == result.state.old_class_count
    np.testing.assert_array_equal(state.log_diag_fp32, result.state.log_diag_fp32)
    for name in (
        "coef1_qint8",
        "coef2_qint8",
        "scale1_fp16",
        "scale2_fp16",
        "intercept_fp16",
    ):
        assert getattr(state, name).tobytes() == getattr(result.state, name).tobytes()
    assert state.covariance_policy == result.state.covariance_policy


@pytest.mark.parametrize("new_count", [1, 3, 4])
def test_original_d42_fit_keeps_intermediate_new_counts_rejected(
    monkeypatch: pytest.MonkeyPatch, new_count: int
) -> None:
    old_rows, old_labels, old_classes = _d42_unit_support("old", 2, offset=0)
    new_rows, new_labels, new_classes = _d42_unit_support(
        "new", new_count, offset=16
    )

    def frozen_old_metric(*args, **kwargs):
        del args, kwargs
        return (
            np.zeros(d42.FEATURE_DIM, dtype=np.float32),
            tuple(
                {"optimizer_step": step}
                for step in range(1, d42.METRIC_EPOCHS + 1)
            ),
            {"estimated_adaptation_macs": 0, "peak_cuda_memory_bytes": 0},
        )

    monkeypatch.setattr(d42, "_fit_old_only_b3_metric", frozen_old_metric)
    assert d42.ALLOWED_NEW_CLASS_COUNTS == (2, 5, 10, 20)
    with pytest.raises(d42.D42UnifiedShrinkageLDAError, match="class/K closure"):
        d42.fit_d42_unified_shrinkage_lda(
            old_rows,
            old_labels,
            old_classes,
            new_rows,
            new_labels,
            new_classes,
            seed=19,
        )


def test_s5_calls_original_d92_builder_with_byte_exact_statistics_and_affine() -> None:
    module = _module()
    batch = _advance(module, _ledger(module), 5)

    ledger = _ledger(module)
    result = None
    for index in range(5):
        result = module.advance_session(
            ledger,
            [
                _packet(
                    module,
                    f"new_{index}",
                    arrival_session=ledger.next_session,
                    seed=200 + index,
                )
            ],
        )
        ledger = result.ledger
    assert result is not None

    expected_statistics = d92.build_registration_balanced_statistics(
        d42, result.transformed_rows, result.targets, 11, 10
    )
    expected_coefficient, expected_intercept, _ = d92.compile_registration_balanced_affine(
        d42, expected_statistics, arm="full"
    )

    np.testing.assert_array_equal(result.statistics.means, expected_statistics.means)
    np.testing.assert_array_equal(result.statistics.old_covariance, expected_statistics.old_covariance)
    np.testing.assert_array_equal(result.statistics.new_covariance, expected_statistics.new_covariance)
    np.testing.assert_array_equal(result.statistics.covariance, expected_statistics.covariance)
    np.testing.assert_array_equal(result.coefficient, expected_coefficient)
    np.testing.assert_array_equal(result.intercept, expected_intercept)
    assert result.audit["d92_continuous_s5_original_builder_used"] is True
    assert result.audit["d92_continuous_original_e0_equivalent"] is True

    assert result.state.classes == batch.state.classes
    np.testing.assert_array_equal(result.coefficient, batch.coefficient)
    np.testing.assert_array_equal(result.intercept, batch.intercept)
    for name in (
        "log_diag_fp32",
        "coef1_qint8",
        "coef2_qint8",
        "scale1_fp16",
        "scale2_fp16",
        "intercept_fp16",
    ):
        assert getattr(result.state, name).tobytes() == getattr(batch.state, name).tobytes()


def test_canonical_class_and_row_order_make_equivalent_packets_identical() -> None:
    module = _module()
    first_ledger = _ledger(module)
    second_ledger = _ledger(module, old_permutation=(5, 2, 4, 0, 3, 1))
    forward = [
        _packet(module, "new_0", arrival_session=1, seed=200),
        _packet(module, "new_1", arrival_session=1, seed=201),
    ]
    reverse_rows = np.arange(9, -1, -1)
    reordered = [
        _packet(
            module,
            "new_1",
            arrival_session=1,
            seed=201,
            permutation=reverse_rows,
        ),
        _packet(
            module,
            "new_0",
            arrival_session=1,
            seed=200,
            permutation=reverse_rows,
        ),
    ]

    first = module.advance_session(first_ledger, forward)
    second = module.advance_session(second_ledger, reordered)

    assert first.state.classes == second.state.classes
    for name in (
        "coef1_qint8",
        "coef2_qint8",
        "scale1_fp16",
        "scale2_fp16",
        "intercept_fp16",
    ):
        np.testing.assert_array_equal(getattr(first.state, name), getattr(second.state, name))


def test_future_support_and_repeated_tokens_fail_before_state_transition() -> None:
    module = _module()
    ledger = _ledger(module)

    with pytest.raises(module.D92ContinuousSessionError, match="metadata drift"):
        module.SupportPacket(
            handle="new_bad_session",
            rows=np.zeros((10, d42.FEATURE_DIM), dtype=np.float32),
            physical_tokens=tuple(f"bad:{index}" for index in range(10)),
            package_id="pkg:bad-session",
            arrival_session="1",
        )
    with pytest.raises(module.D92ContinuousSessionError, match="metadata drift"):
        module.SupportPacket(
            handle="new_bad_tokens",
            rows=np.zeros((10, d42.FEATURE_DIM), dtype=np.float32),
            physical_tokens="0123456789",
            package_id="pkg:bad-tokens",
            arrival_session=1,
        )

    class PoisonRows:
        def __array__(self, *args, **kwargs):
            raise AssertionError("future support content was opened")

    future = module.SupportPacket(
        handle="new_future",
        rows=PoisonRows(),
        physical_tokens=("future",) * 10,
        package_id="pkg:future",
        arrival_session=2,
    )
    with pytest.raises(module.D92ContinuousSessionError, match="future support"):
        module.advance_session(ledger, [future])

    repeated = _packet(
        module,
        "new_0",
        arrival_session=1,
        seed=200,
        tokens=tuple("old_0:token:0" for _ in range(10)),
    )
    with pytest.raises(module.D92ContinuousSessionError, match="duplicate physical support token"):
        module.advance_session(ledger, [repeated])


def test_each_session_emits_one_full_solve_one_d42_codec_and_zero_query_access(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _module()
    calls = 0
    original = d42._quantize_coefficients

    def counted(coefficients: np.ndarray):
        nonlocal calls
        calls += 1
        return original(coefficients)

    monkeypatch.setattr(d42, "_quantize_coefficients", counted)
    result = _advance(module, _ledger(module), 2)

    assert calls == 1
    assert result.audit["d92_continuous_full_solve_count"] == 1
    assert result.audit["d92_continuous_d42_codec_count"] == 1
    assert result.audit["future_support_open_sentinel"] == 0
    assert result.audit["past_token_duplicate_count"] == 0
    np.testing.assert_array_equal(result.state.log_diag_fp32, _TEST_LOG_DIAG)
    assert result.state.log_diag_fp32.flags.writeable is False
    assert result.audit["d92_continuous_query_state_bytes"] == (
        result.state.persistent_state_bytes
    )
    assert result.audit["d92_continuous_query_state_sha256"] == (
        result.state.persistent_state_sha256
    )
    assert result.audit["d92_continuous_peak_budget_bytes"] == 4 * 1024 * 1024
    assert result.audit["d92_continuous_wall_budget_ms"] == 300
    assert result.ledger.anchor.da_anchor_id == "frozen-da-anchor-v1"
    assert result.ledger.next_session == 2
    assert set(inspect.signature(module.advance_session).parameters).isdisjoint(
        {"query", "truth", "role", "quota", "global_assignment"}
    )
    for key in (
        "query_fit_access",
        "query_update_access",
        "query_selection_access",
        "query_truth_access",
        "query_role_oracle_access",
        "query_class_quota_access",
        "query_global_reassignment",
    ):
        assert result.audit[key] is False
