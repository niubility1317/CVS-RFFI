import inspect

import numpy as np
import pytest

from cvsrffi import stage2_d42_unified_shrinkage_lda as d42
from cvsrffi.stage2_d81_phase1_episode_scorer import (
    D81Phase1EpisodeScorer,
    D81Phase1EpisodeScorerError,
    raw_concat_to_d81_registered_feature,
)
from cvsrffi.stage2_diag_cosine_exploration import (
    registered_feature,
    rf_statistics,
    spectral_logmag_sketch,
)


def _scorer(*, device: str = "cpu") -> D81Phase1EpisodeScorer:
    basis = np.eye(160, 3, dtype=np.float64)
    return D81Phase1EpisodeScorer(
        nuisance_basis_fp64=basis,
        spectral_weights_fp64=np.asarray([0.5, 0.3, 0.2]),
        ground_manifest_sha256="1" * 64,
        ground_component_npz_sha256="2" * 64,
        ground_audit={"ground_component_input_count": 84},
        device=device,
    )


def test_receipt_is_stable_and_binds_ground_spectrum() -> None:
    first = _scorer()
    second = _scorer()
    assert first.scorer_id == second.scorer_id
    assert len(first.scorer_id) == 64
    assert first.receipt["query_labels_input"] is False
    assert first.receipt["mutable_fit_cache"] is False
    assert first.receipt["phase1_checkpoint_sha256"] == first.phase1_checkpoint_sha256
    assert first.receipt["sklearn_runtime_version"] == d42.sklearn.__version__
    assert first.receipt["numpy_runtime_version"] == np.__version__
    assert first.receipt["metric_seed"] == 713101
    assert set(first.receipt["dependency_code_sha256"]) >= {
        "scorer",
        "d42_core",
        "d81_core",
        "d81_probe",
        "d62_probe",
    }
    assert len(first.receipt["dependency_closure_sha256"]) == 64
    assert first.nuisance_basis_fp64.flags.writeable is False


def test_rejects_nonorthogonal_basis_or_bad_hash() -> None:
    with pytest.raises(D81Phase1EpisodeScorerError, match="spectrum drift"):
        D81Phase1EpisodeScorer(
            np.ones((160, 2)), np.asarray([0.5, 0.5]), "1" * 64, "2" * 64, {"x": 1}
        )
    with pytest.raises(D81Phase1EpisodeScorerError, match="SHA256"):
        D81Phase1EpisodeScorer(
            np.eye(160, 2), np.asarray([0.5, 0.5]), "bad", "2" * 64, {"x": 1}
        )


def test_rejects_unbalanced_or_invalid_episode_before_fit() -> None:
    scorer = _scorer()
    features = np.ones((3, 288), dtype=np.float32)
    with pytest.raises(D81Phase1EpisodeScorerError, match="balanced K-shot"):
        scorer(features, np.asarray(["a", "a", "b"]), features[:1], np.asarray(["a", "b"]))
    with pytest.raises(D81Phase1EpisodeScorerError, match="shape drift"):
        scorer(features[:, :20], np.asarray(["a", "b", "c"]), features[:1], np.asarray(["a", "b", "c"]))


def test_ground_audit_is_deeply_immutable_and_receipt_cannot_drift() -> None:
    source = {"ground_component_input_count": 84, "nested": {"values": [1, 2]}}
    scorer = D81Phase1EpisodeScorer(
        nuisance_basis_fp64=np.eye(160, 2),
        spectral_weights_fp64=np.asarray([0.6, 0.4]),
        ground_manifest_sha256="1" * 64,
        ground_component_npz_sha256="2" * 64,
        ground_audit=source,
    )
    locked = scorer.scorer_id
    source["nested"]["values"].append(3)
    assert scorer.scorer_id == locked
    with pytest.raises(TypeError):
        scorer.ground_audit["new"] = True


def test_public_call_surface_has_no_query_labels_receiver_or_role() -> None:
    parameters = inspect.signature(D81Phase1EpisodeScorer.__call__).parameters
    assert tuple(parameters) == (
        "self",
        "support_features",
        "support_labels",
        "query_features",
        "class_ids",
    )
    assert not any(
        token in name
        for name in parameters
        for token in ("query_label", "receiver", "old", "new", "role")
    )


def test_exact_support_fit_compiles_int8_and_scores_query_only(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from scripts import probe_d81_ground_nuisance_cauchy_center as probe

    scorer = _scorer(device="cuda")
    support = np.zeros((4, 288), dtype=np.float32)
    support[:, 0] = [1.0, 0.9, -1.0, -0.9]
    support[:, 160] = 1.0
    support[:, 256] = 1.0
    labels = np.asarray(["a", "a", "b", "b"])
    query = np.zeros((3, 288), dtype=np.float32)
    query[:, 1] = [1.0, 2.0, 3.0]
    query[:, 160] = 1.0
    query[:, 256] = 1.0
    captures = {}

    def fake_build(module, basis, weights, audit):
        assert module is d42

        def fit(rows, targets, class_count, k_shot):
            captures["fit_rows"] = np.array(rows, copy=True)
            captures["fit_targets"] = np.array(targets, copy=True)
            assert class_count == 2 and k_shot == 2
            return (
                np.zeros((2, 288), dtype=np.float32),
                np.zeros(2, dtype=np.float32),
                {"covariance_policy": "test"},
            )

        return fit, [], []

    def fake_metric(rows, targets, class_count, *, seed, device):
        captures["metric_rows"] = np.array(rows, copy=True)
        captures["seed"] = seed
        captures["device"] = str(device)
        trace = tuple({"epoch": index} for index in range(d42.METRIC_EPOCHS))
        return np.zeros(288, dtype=np.float32), trace, {}

    state = object()

    def fake_compile(classes, old_count, logdiag, coef, intercept, policy, *, precision):
        captures["precision"] = precision
        captures["classes"] = classes
        return state, {"quantized": True}

    def fake_score(received_state, rows):
        assert received_state is state
        captures["score_rows"] = np.array(rows, copy=True)
        return np.arange(len(rows) * 2, dtype=np.float32).reshape(len(rows), 2)

    monkeypatch.setattr(probe, "build_d81_fit", fake_build)
    monkeypatch.setattr(d42, "_fit_old_only_b3_metric", fake_metric)
    monkeypatch.setattr(d42, "_compile_state", fake_compile)
    monkeypatch.setattr(d42, "score_d42_unified_shrinkage_lda", fake_score)
    scores = scorer(support, labels, query, np.asarray(["a", "b"]))
    assert scores.shape == (3, 2)
    assert captures["precision"] == "int8"
    assert captures["seed"] == scorer.metric_seed
    assert captures["device"] == "cuda:0"
    np.testing.assert_allclose(
        captures["metric_rows"], raw_concat_to_d81_registered_feature(support)
    )
    np.testing.assert_allclose(
        captures["score_rows"], raw_concat_to_d81_registered_feature(query)
    )
    assert not np.array_equal(captures["fit_rows"][:3], query)


def test_runtime_version_drift_fails_before_fit(monkeypatch: pytest.MonkeyPatch) -> None:
    scorer = _scorer()
    monkeypatch.setattr(d42.sklearn, "__version__", "9.9.9")
    support = np.ones((2, 288), dtype=np.float32)
    with pytest.raises(D81Phase1EpisodeScorerError, match="changed after lock"):
        scorer(
            support,
            np.asarray(["a", "b"]),
            support[:1],
            np.asarray(["a", "b"]),
        )


def test_raw_concat_conversion_matches_historical_registered_feature() -> None:
    rng = np.random.default_rng(81097)
    iq = rng.normal(size=(5, 2, 256)).astype(np.float32)
    z160 = rng.normal(size=(5, 160)).astype(np.float32)
    raw = np.concatenate(
        [z160, spectral_logmag_sketch(iq), rf_statistics(iq)], axis=1
    ).astype(np.float32)
    expected = registered_feature(iq, z160)
    actual = raw_concat_to_d81_registered_feature(raw)
    np.testing.assert_allclose(actual, expected, atol=2e-7, rtol=0)
