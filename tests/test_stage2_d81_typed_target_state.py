from __future__ import annotations

from dataclasses import replace
import hashlib
import inspect
import json

import numpy as np
import pytest

from cvsrffi.stage2_d81_phase1_episode_scorer import D81Phase1EpisodeScorer
import cvsrffi.stage2_d81_typed_target_state as typed


CLASSES = ("registered-a", "registered-b")


def _sha(path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _scorer(tmp_path) -> D81Phase1EpisodeScorer:
    from scripts import probe_d81_ground_nuisance_cauchy_center as probe

    rng = np.random.default_rng(81099)
    residual = rng.normal(size=(24, 160))
    covariance = residual.T @ residual / len(residual) + 1.0e-6 * np.eye(160)
    basis, weights, basis_audit = probe.core.ground_nuisance_basis(
        covariance, 1.0e-6
    )
    component = tmp_path / "ground_component.npz"
    component.write_bytes(b"sealed-int8-ground-component-for-typed-d81")
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps({"schema": "test-sealed-ground-manifest"}), encoding="utf-8"
    )
    audit = {
        "component_path": str(component.resolve()),
        "component_npz_sha256": _sha(component),
        "ground_component_input_count": 84,
        "ground_statistic_semantics": (
            "class_centered_cross_domain_centroid_drift_eigenspectrum"
        ),
        "ground_int8_component_logical_state_bytes": 5816,
        "ground_covariance_statistics_mac_upper_bound": 123456,
        "transient_dequantized_ground_bytes": 84 * 160 * 4,
        "d81_basis_transient_fp64_bytes": int(basis.nbytes + weights.nbytes),
        "d81_basis_sha256": basis_audit["basis_sha256"],
        "d81_spectral_weight_sha256": basis_audit["spectral_weight_sha256"],
        "d81_participation_ratio_effective_rank": basis_audit[
            "participation_ratio_effective_rank"
        ],
        "d81_retained_rank": basis_audit["retained_rank"],
        "d81_rank_policy": basis_audit["rank_policy"],
    }
    return D81Phase1EpisodeScorer(
        nuisance_basis_fp64=basis,
        spectral_weights_fp64=weights,
        ground_manifest_sha256=_sha(manifest),
        ground_component_npz_sha256=_sha(component),
        ground_audit=audit,
        device="cpu",
    )


def _support(k: int = 1):
    rng = np.random.default_rng(81100 + k)
    rows, labels, physical = [], [], []
    for class_index, class_id in enumerate(CLASSES):
        primary = np.zeros(160, dtype=np.float32)
        primary[class_index] = 1.0
        for shot in range(k):
            z = primary + np.float32(0.01) * rng.normal(size=160).astype(np.float32)
            fft = rng.normal(size=96).astype(np.float32)
            rf = rng.normal(size=32).astype(np.float32)
            fft[class_index] += 3.0
            rf[class_index] += 2.0
            rows.append(np.concatenate([z, fft, rf]).astype(np.float32))
            labels.append(class_id)
            physical.append(f"physical-{class_id}-{shot}")
    return np.stack(rows), np.asarray(labels), np.asarray(physical)


@pytest.fixture(scope="module")
def exact_row(tmp_path_factory):
    tmp_path = tmp_path_factory.mktemp("typed-d81-ground")
    scorer = _scorer(tmp_path)
    config = typed.D81TypedTargetConfig.from_scorer(scorer)
    support = _support(1)
    state = typed.fit_d81_typed_target_state(
        *support,
        CLASSES,
        d81_scorer=scorer,
        config=config,
    )
    return scorer, config, support, state


def test_public_surface_has_no_arbitrary_logit_probability_or_role_input() -> None:
    assert typed.DEPLOYMENT_STATUS == "LOCAL_CORE_PENDING_INDEPENDENT_REVIEW"
    for name in typed.__all__:
        function = getattr(typed, name, None)
        if inspect.isfunction(function):
            parameters = set(inspect.signature(function).parameters)
            assert not parameters & {
                "base_logits",
                "logits",
                "probabilities",
                "query_labels",
                "query_truth",
                "receiver",
                "scenario",
                "old_classes",
                "new_classes",
                "role",
            }
    assert tuple(inspect.signature(typed.score_d81_typed_target_raw_logits).parameters) == (
        "state",
        "query_features",
    )


def test_real_d81_fit_runs_exactly_one_twenty_step_metric_and_typed_score_matches(
    exact_row,
) -> None:
    scorer, _config, support, state = exact_row
    features, labels, _physical = support
    query = np.concatenate([features, features[::-1]], axis=0).astype(np.float32)
    expected = scorer(features, labels, query, np.asarray(CLASSES))
    actual = typed.score_d81_typed_target_raw_logits(state, query)
    np.testing.assert_allclose(actual, expected, atol=2e-6, rtol=0)
    assert state.fit_audit["metric_fit_execution_count"] == 1
    assert state.fit_audit["metric_optimizer_steps"] == 20
    assert state.resource_audit["optimizer_steps"] == 20
    assert state.resource_audit["metric_fit_execution_count"] == 1
    assert state.resource_audit["d81_head_fit_execution_count"] == 1
    assert state.fit_audit["all_registered_classes_same_formula"] is True
    assert state.fit_audit["old_new_role_input"] is False
    assert state.fit_audit["physical_ids_persisted"] is False
    assert state.fit_audit["d81_transform_audit"][
        "per_support_energy_or_weight_vectors_persisted"
    ] is False
    audit_text = json.dumps(dict(state.fit_audit), default=list)
    assert all(value not in audit_text for value in support[2].tolist())


def test_resource_receipt_counts_ground_head_serialized_peak_fit_and_query(exact_row) -> None:
    _scorer_value, config, _support_value, state = exact_row
    resource = state.resource_audit
    assert resource["ground_bundle_logical_state_bytes"] == 5816
    assert resource["ground_bundle_serialized_state_bytes"] == (
        config.ground_component_npz_serialized_bytes
        + config.ground_manifest_serialized_bytes
    )
    assert resource["head_numeric_logical_state_bytes"] > 0
    assert resource["head_serialized_state_bytes"] > resource[
        "head_numeric_logical_state_bytes"
    ]
    assert resource["peak_state_bytes_upper_bound_including_ground"] > resource[
        "head_serialized_state_bytes"
    ]
    assert resource["fit_mac_upper_bound"] > 0
    assert resource["query_mac_upper_bound_per_sample"] > 0
    assert resource["fit_latency_measurement_protocol"].startswith("external_wall_clock")
    assert resource["query_latency_measurement_protocol"].startswith("external_wall_clock")
    assert typed.verify_d81_typed_target_state(state)


def test_support_permutation_is_exactly_invariant(exact_row) -> None:
    scorer, config, support, state = exact_row
    features, labels, physical = support
    order = np.arange(len(features))[::-1]
    permuted = typed.fit_d81_typed_target_state(
        features[order],
        labels[order],
        physical[order],
        CLASSES,
        d81_scorer=scorer,
        config=config,
    )
    assert permuted.support_receipt_sha256 == state.support_receipt_sha256
    assert permuted.state_receipt_sha256 == state.state_receipt_sha256
    np.testing.assert_array_equal(permuted.coef1_qint8, state.coef1_qint8)


def test_class_permutation_is_equivariant(exact_row) -> None:
    scorer, config, support, state = exact_row
    features, labels, physical = support
    classes = tuple(reversed(CLASSES))
    permuted = typed.fit_d81_typed_target_state(
        features,
        labels,
        physical,
        classes,
        d81_scorer=scorer,
        config=config,
    )
    query = np.concatenate([features, features[::-1]], axis=0).astype(np.float32)
    original_logits = typed.score_d81_typed_target_raw_logits(state, query)
    permuted_logits = typed.score_d81_typed_target_raw_logits(permuted, query)
    np.testing.assert_allclose(permuted_logits, original_logits[:, ::-1], atol=2e-6)


def test_query_batch_equals_individual_and_cannot_update_state(exact_row) -> None:
    _scorer_value, _config, support, state = exact_row
    query = np.concatenate([support[0], support[0][::-1]], axis=0).astype(np.float32)
    receipt = state.state_receipt_sha256
    together = typed.score_d81_typed_target_raw_logits(state, query)
    separate = np.concatenate(
        [
            typed.score_d81_typed_target_raw_logits(state, query[index : index + 1])
            for index in range(len(query))
        ],
        axis=0,
    )
    np.testing.assert_array_equal(together, separate)
    assert state.state_receipt_sha256 == receipt
    assert state.log_diag_fp32.flags.writeable is False
    assert state.coef1_qint8.flags.writeable is False
    assert state.resource_audit["query_state_updates"] == 0


@pytest.mark.parametrize("failure", ["unbalanced", "duplicate", "missing"])
def test_unbalanced_duplicate_or_missing_support_fails_before_metric(
    exact_row, failure: str
) -> None:
    scorer, config, _support_value, _state = exact_row
    features, labels, physical = _support(5)
    if failure == "unbalanced":
        features, labels, physical = features[:-1], labels[:-1], physical[:-1]
    elif failure == "duplicate":
        physical[1] = physical[0]
    else:
        keep = labels != CLASSES[-1]
        features, labels, physical = features[keep], labels[keep], physical[keep]
    with pytest.raises(typed.D81TypedTargetStateError, match="support"):
        typed.fit_d81_typed_target_state(
            features,
            labels,
            physical,
            CLASSES,
            d81_scorer=scorer,
            config=config,
        )


def test_config_dependency_and_component_receipt_drift_fail_closed(exact_row) -> None:
    scorer, config, support, _state = exact_row
    changed = replace(
        config,
        ground_component_npz_serialized_bytes=(
            config.ground_component_npz_serialized_bytes + 1
        ),
    )
    with pytest.raises(typed.D81TypedTargetStateError, match="scorer/config"):
        typed.fit_d81_typed_target_state(
            *support, CLASSES, d81_scorer=scorer, config=changed
        )
    dependency_rows = list(config.dependency_code_sha256)
    dependency_rows[0] = (dependency_rows[0][0], "a" * 64)
    with pytest.raises(typed.D81TypedTargetStateError, match="dependency"):
        replace(config, dependency_code_sha256=tuple(dependency_rows))


def test_state_array_resource_registry_and_receipt_tamper_are_rejected(exact_row) -> None:
    _scorer_value, _config, _support_value, state = exact_row
    coefficient = np.array(state.coef1_qint8, copy=True)
    coefficient[0, 0] = np.int8(
        int(coefficient[0, 0]) - 1
        if int(coefficient[0, 0]) == 127
        else int(coefficient[0, 0]) + 1
    )
    with pytest.raises(typed.D81TypedTargetStateError, match="receipt"):
        replace(state, coef1_qint8=coefficient)
    resource = dict(state.resource_audit)
    resource["optimizer_steps"] = 19
    with pytest.raises(typed.D81TypedTargetStateError, match="receipt"):
        replace(state, resource_audit=resource)
    with pytest.raises(typed.D81TypedTargetStateError):
        replace(state, classes=(state.classes[0], state.classes[0]))
    with pytest.raises(typed.D81TypedTargetStateError, match="receipt"):
        replace(state, state_receipt_sha256="b" * 64)


def test_query_shape_and_dtype_drift_fail_closed(exact_row) -> None:
    _scorer_value, _config, support, state = exact_row
    with pytest.raises(typed.D81TypedTargetStateError, match="float32"):
        typed.score_d81_typed_target_raw_logits(state, support[0].astype(np.float64))
    with pytest.raises(typed.D81TypedTargetStateError, match="float32"):
        typed.score_d81_typed_target_raw_logits(state, support[0][:, :20])
