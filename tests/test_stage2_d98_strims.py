from __future__ import annotations

from dataclasses import replace
import inspect

import numpy as np
import pytest

import cvsrffi.stage2_d98_strims as d98
from cvsrffi.stage2_d81_phase1_episode_scorer import D81Phase1EpisodeScorer
from cvsrffi.stage2_qk_d81_lgf import Phase1LockedConfig as D97Lock


CLASSES = ("tx-a", "tx-b", "tx-c")


def _d81_scorer() -> D81Phase1EpisodeScorer:
    return D81Phase1EpisodeScorer(
        nuisance_basis_fp64=np.eye(160, 3, dtype=np.float64),
        spectral_weights_fp64=np.asarray([0.5, 0.3, 0.2]),
        ground_manifest_sha256="1" * 64,
        ground_component_npz_sha256="2" * 64,
        ground_audit={"ground_component_input_count": 84},
    )


def _d97_lock() -> D97Lock:
    return D97Lock(
        beta=8.0,
        temp_base=2.0,
        temp_qk=0.5,
        eta_max=0.5,
        phase1_receipt_sha256="3" * 64,
        margin_audit_sha256="4" * 64,
        k1_eta_prior=0.0,
    )


def _lock(
    scorer: D81Phase1EpisodeScorer,
    d97_config: D97Lock,
    *,
    k1_alpha: float = 0.0,
    retention: float = 0.25,
) -> d98.Phase1STRIMSLock:
    return d98.Phase1STRIMSLock(
        temp_base=d97_config.temp_base,
        temp_qk=d97_config.temp_qk,
        gain_prior_mean=0.0,
        gain_prior_variance=0.04,
        prior_strength=2.0,
        lcb_kappa=1.0,
        reliability_temperature=0.2,
        alpha_max=0.8,
        lambda_tail=0.5,
        lambda_intrusion=0.1,
        lambda_retention=retention,
        cvar_rho=1.0 / 3.0,
        intrusion_margin=0.25,
        retention_delta=0.05,
        k1_reliability=0.0,
        k1_alpha=k1_alpha,
        solver_iterations=48,
        phase1_receipt_sha256="5" * 64,
        d81_scorer_receipt_sha256=scorer.scorer_id,
        d97_lock_receipt_sha256=d97_config.lock_digest,
    )


def _support(k: int, classes=CLASSES):
    rng = np.random.default_rng(9800 + k)
    rows = []
    labels = []
    physical = []
    for class_index, class_name in enumerate(classes):
        for shot in range(k):
            row = rng.normal(size=288).astype(np.float32)
            row[class_index] += 4.0
            row[160 + class_index] += 3.0
            row[256 + class_index] += 2.0
            rows.append(row)
            labels.append(class_name)
            physical.append(f"p-{class_name}-{shot}")
    return (
        np.stack(rows).astype(np.float32),
        np.asarray(labels),
        np.asarray(physical),
    )


def _row_hashes(value: np.ndarray) -> set[bytes]:
    return {np.ascontiguousarray(row).tobytes() for row in value}


def _patch_exact_d81(
    monkeypatch: pytest.MonkeyPatch, calls: list[dict[str, object]]
) -> None:
    def fake_call(self, support_features, support_labels, query_features, class_ids):
        support = np.asarray(support_features, dtype=np.float32)
        query = np.asarray(query_features, dtype=np.float32)
        labels = np.asarray(support_labels).astype(str)
        classes = tuple(np.asarray(class_ids).astype(str).tolist())
        calls.append(
            {
                "support": support.copy(),
                "query": query.copy(),
                "labels": labels.copy(),
                "classes": classes,
            }
        )
        return np.zeros((len(query), len(classes)), dtype=np.float32)

    monkeypatch.setattr(D81Phase1EpisodeScorer, "__call__", fake_call)


def _artifact(
    monkeypatch: pytest.MonkeyPatch,
    *,
    k: int = 5,
    classes=CLASSES,
    order: np.ndarray | None = None,
):
    scorer = _d81_scorer()
    d97_config = _d97_lock()
    calls: list[dict[str, object]] = []
    _patch_exact_d81(monkeypatch, calls)
    features, labels, physical = _support(k, classes)
    if order is not None:
        features, labels, physical = features[order], labels[order], physical[order]
    artifact = d98.produce_strims_support_artifact(
        support_features=features,
        support_labels=labels,
        physical_ids=physical,
        class_ids=classes,
        d81_scorer=scorer,
        d97_config=d97_config,
    )
    return artifact, scorer, d97_config, calls, (features, labels, physical)


def test_k1_alpha_nonzero_is_rejected_at_lock_construction() -> None:
    scorer = _d81_scorer()
    config = _d97_lock()
    with pytest.raises(d98.D98STRIMSError, match="k1_alpha must be 0"):
        _lock(scorer, config, k1_alpha=0.1)


def test_k1_artifact_never_calls_d81_or_d97_and_fits_exact_fallback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    scorer = _d81_scorer()
    config = _d97_lock()

    def forbidden_d81(*_args, **_kwargs):
        raise AssertionError("K1 must not call D81 OOF scorer")

    monkeypatch.setattr(D81Phase1EpisodeScorer, "__call__", forbidden_d81)
    import cvsrffi.stage2_qk_d81_lgf as d97

    def forbidden_bank(*_args, **_kwargs):
        raise AssertionError("K1 must not construct a qK bank")

    monkeypatch.setattr(d97, "build_support_bank", forbidden_bank)
    features, labels, physical = _support(1)
    artifact = d98.produce_strims_support_artifact(
        support_features=features,
        support_labels=labels,
        physical_ids=physical,
        class_ids=CLASSES,
        d81_scorer=scorer,
        d97_config=config,
    )
    state = d98.fit_strims_state(artifact=artifact, lock=_lock(scorer, config))
    assert state.k_shot == 1
    assert float(state.alpha_fp16) == 0.0
    assert np.all(state.reliability_fp16 == 0.0)
    assert state.fit_audit["qk_head_used"] is False
    assert state.resource_audit["support_oof_rows_consumed"] == 0
    assert d98.verify_state_receipt(state, _lock(scorer, config))


@pytest.mark.parametrize("k", [5, 10])
def test_internal_oof_producer_uses_exact_physical_complements_and_raw_d97(
    monkeypatch: pytest.MonkeyPatch, k: int
) -> None:
    artifact, scorer, config, calls, values = _artifact(monkeypatch, k=k)
    features, _labels, _physical = values
    assert len(calls) == k
    full = _row_hashes(features)
    held_union: set[bytes] = set()
    for call in calls:
        train = _row_hashes(call["support"])
        held = _row_hashes(call["query"])
        assert not train & held
        assert train | held == full
        assert len(train) == len(CLASSES) * (k - 1)
        assert len(held) == len(CLASSES)
        assert not held_union & held
        held_union |= held
    assert held_union == full
    assert artifact.raw_unfused_logits is True
    assert len(artifact.fold_records_json) == k
    assert all('"raw_unfused_logits":true' in row for row in artifact.fold_records_json)

    lock = _lock(scorer, config, retention=0.0)
    state = d98.fit_strims_state(artifact=artifact, lock=lock)
    assert state.k_shot == k
    assert state.fit_audit["fit_source"] == "typed_internal_D81_D97_raw_OOF_artifact"
    assert state.fit_audit["query_rows_used"] == 0
    assert state.resource_audit["alpha_objective_evaluations_upper_bound"] == (
        2 + lock.solver_iterations + 5 + 2
    )
    assert state.resource_audit["deployment_inference_exposed"] is False
    assert d98.verify_state_receipt(state, lock)


def test_support_order_is_canonicalized_before_internal_head_calls(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    first, scorer, config, _calls, values = _artifact(monkeypatch, k=5)
    order = np.arange(len(values[0]))[::-1]
    second, _, _, _, _ = _artifact(monkeypatch, k=5, order=order)
    assert second.artifact_receipt_sha256 == first.artifact_receipt_sha256
    first_state = d98.fit_strims_state(artifact=first, lock=_lock(scorer, config))
    second_state = d98.fit_strims_state(artifact=second, lock=_lock(scorer, config))
    assert second_state.fit_receipt_sha256 == first_state.fit_receipt_sha256
    np.testing.assert_array_equal(
        second_state.reliability_fp16, first_state.reliability_fp16
    )


def test_class_permutation_is_equivariant(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    scorer = _d81_scorer()
    config = _d97_lock()
    calls: list[dict[str, object]] = []
    _patch_exact_d81(monkeypatch, calls)
    features, labels, physical = _support(5)
    first = d98.produce_strims_support_artifact(
        support_features=features,
        support_labels=labels,
        physical_ids=physical,
        class_ids=CLASSES,
        d81_scorer=scorer,
        d97_config=config,
    )
    permutation = np.asarray([2, 0, 1])
    permuted_classes = tuple(CLASSES[index] for index in permutation)
    second = d98.produce_strims_support_artifact(
        support_features=features,
        support_labels=labels,
        physical_ids=physical,
        class_ids=permuted_classes,
        d81_scorer=scorer,
        d97_config=config,
    )
    lock = _lock(scorer, config, retention=0.0)
    first_state = d98.fit_strims_state(artifact=first, lock=lock)
    second_state = d98.fit_strims_state(artifact=second, lock=lock)
    np.testing.assert_array_equal(
        second_state.reliability_fp16,
        first_state.reliability_fp16[permutation],
    )
    assert second_state.alpha_fp16 == first_state.alpha_fp16


def test_artifact_capability_blocks_manual_sha_or_logits(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    artifact, scorer, config, _calls, _ = _artifact(monkeypatch, k=5)
    lock = _lock(scorer, config)
    with pytest.raises(d98.D98STRIMSError, match="internally produced typed"):
        d98.fit_strims_state(
            artifact={"artifact_receipt_sha256": "a" * 64}, lock=lock
        )
    with pytest.raises(d98.D98STRIMSError, match="module capability"):
        d98._STRIMSSupportArtifact(
            _capability=object(),
            classes=CLASSES,
            k_shot=1,
            truth_int16=np.arange(3, dtype=np.int16),
            base_logits_oof=None,
            qk_logits_oof=None,
            support_input_sha256="a" * 64,
            d81_scorer_receipt_sha256="b" * 64,
            d97_lock_receipt_sha256="c" * 64,
            fold_records_json=(),
            raw_unfused_logits=False,
            artifact_receipt_sha256="d" * 64,
        )
    assert artifact.artifact_receipt_sha256 != "a" * 64


def test_exact_scorer_and_config_types_are_required() -> None:
    features, labels, physical = _support(1)

    class FakeScorer:
        scorer_id = "a" * 64

        def __call__(self, *_args):
            return np.zeros((3, 3), dtype=np.float32)

    with pytest.raises(d98.D98STRIMSError, match="exact D81"):
        d98.produce_strims_support_artifact(
            support_features=features,
            support_labels=labels,
            physical_ids=physical,
            class_ids=CLASSES,
            d81_scorer=FakeScorer(),
            d97_config=_d97_lock(),
        )
    with pytest.raises(d98.D98STRIMSError, match="exact D97"):
        d98.produce_strims_support_artifact(
            support_features=features,
            support_labels=labels,
            physical_ids=physical,
            class_ids=CLASSES,
            d81_scorer=_d81_scorer(),
            d97_config=object(),
        )


def test_lock_and_state_receipt_tamper_fail_closed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    artifact, scorer, config, _calls, _ = _artifact(monkeypatch, k=5)
    lock = _lock(scorer, config)
    state = d98.fit_strims_state(artifact=artifact, lock=lock)
    tampered_temperature = replace(state, temp_base=100.0)
    assert not d98.verify_state_receipt(tampered_temperature, lock)
    mismatched_lock = replace(lock, d81_scorer_receipt_sha256="e" * 64)
    with pytest.raises(d98.D98STRIMSError, match="head receipt"):
        d98.fit_strims_state(artifact=artifact, lock=mismatched_lock)
    assert not d98.verify_state_receipt(state, mismatched_lock)


def test_public_surface_has_no_generic_logits_or_deployable_fuse() -> None:
    assert d98.DEPLOYMENT_STATUS == "LOCAL_CORE_PENDING_TYPED_D81_INTEGRATION"
    assert "produce_strims_support_artifact" in d98.__all__
    assert "fit_strims_state" in d98.__all__
    assert not any(
        token in name
        for name in d98.__all__
        for token in ("fuse", "provenance", "receipt_builder", "logits")
    )
    assert not hasattr(d98, "fuse_strims_logits")
    assert not hasattr(d98, "build_inference_head_provenance")
    assert not hasattr(d98, "compute_oof_receipt_sha256")
    public_functions = {
        name: getattr(d98, name)
        for name in d98.__all__
        if inspect.isfunction(getattr(d98, name, None))
    }
    for function in public_functions.values():
        parameters = set(inspect.signature(function).parameters)
        assert not parameters & {
            "base_logits",
            "qk_logits",
            "base_probabilities",
            "fused_probabilities",
            "query_labels",
            "query_truth",
            "receiver",
            "scenario",
            "old_classes",
            "new_classes",
        }


def test_private_gauge_invariant_math_and_convex_solver(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    artifact, scorer, config, _calls, _ = _artifact(monkeypatch, k=5)
    lock = _lock(scorer, config, retention=0.0)
    state = d98.fit_strims_state(artifact=artifact, lock=lock)
    if float(state.alpha_fp16) == 0.0:
        pytest.skip("synthetic typed OOF selected exact zero; deployment remains disabled")
    base = np.asarray([[0.2, -0.1, 0.4]], dtype=np.float32)
    qk = np.asarray([[0.1, 0.3, 0.8]], dtype=np.float32)
    fused = d98._fuse_local_core_coordinates(state, base, qk, lock=lock)
    shifted = d98._fuse_local_core_coordinates(
        state,
        base + np.float32(17.0),
        qk - np.float32(9.0),
        lock=lock,
    )
    np.testing.assert_allclose(shifted, fused, rtol=0.0, atol=3e-6)

    rng = np.random.default_rng(44)
    truth = np.repeat(np.arange(len(CLASSES)), 5)
    base_coordinate = rng.normal(size=(len(truth), len(CLASSES)))
    delta = rng.normal(size=(len(truth), len(CLASSES)))
    grid = np.linspace(0.0, lock.alpha_max, 1001)
    objective = np.asarray(
        [
            d98._objective_components(
                float(alpha), base_coordinate, delta, truth, lock
            )["objective"]
            for alpha in grid
        ]
    )
    second = objective[:-2] - 2.0 * objective[1:-1] + objective[2:]
    assert float(np.min(second)) >= -1e-9
    alpha, metrics = d98._solve_alpha(base_coordinate, delta, truth, lock)
    assert 0.0 <= float(alpha) <= lock.alpha_max
    assert metrics["objective"] <= float(np.min(objective)) + 2e-5
