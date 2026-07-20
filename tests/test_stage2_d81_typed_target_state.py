from __future__ import annotations

import copy
from dataclasses import replace
import hashlib
import inspect
import json
from pathlib import Path

import numpy as np
import pytest
import torch

from cvsrffi import stage2_d42_unified_shrinkage_lda as d42
from cvsrffi.stage2_d81_phase1_episode_scorer import D81Phase1EpisodeScorer
import cvsrffi.stage2_d81_typed_target_state as typed


OLD2 = ("old-a", "old-b")
NEW2 = ("new-a", "new-b")


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _scorer(tmp_path: Path) -> D81Phase1EpisodeScorer:
    from scripts import probe_d81_ground_nuisance_cauchy_center as probe

    tmp_path.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(81099)
    residual = rng.normal(size=(24, 160))
    covariance = residual.T @ residual / len(residual) + 1.0e-6 * np.eye(160)
    basis, weights, basis_audit = probe.core.ground_nuisance_basis(covariance, 1.0e-6)
    component = tmp_path / "ground_component.npz"
    component.write_bytes(b"sealed-int8-ground-component-for-typed-d81")
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({"schema": "test-sealed-ground-manifest"}), encoding="utf-8")
    audit = {
        "component_path": str(component.resolve()),
        "component_npz_sha256": _sha(component),
        "ground_component_input_count": 84,
        "ground_statistic_semantics": "class_centered_cross_domain_centroid_drift_eigenspectrum",
        "ground_int8_component_logical_state_bytes": 5816,
        "ground_covariance_statistics_mac_upper_bound": 123456,
        "transient_dequantized_ground_bytes": 84 * 160 * 4,
        "d81_basis_transient_fp64_bytes": int(basis.nbytes + weights.nbytes),
        "d81_basis_sha256": basis_audit["basis_sha256"],
        "d81_spectral_weight_sha256": basis_audit["spectral_weight_sha256"],
        "d81_participation_ratio_effective_rank": basis_audit["participation_ratio_effective_rank"],
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


def _canonical_artifact(value: dict) -> tuple[bytes, str]:
    raw = typed._canonical_bytes(value)
    return raw, hashlib.sha256(raw).hexdigest()


def _phase1_authority(scorer: D81Phase1EpisodeScorer) -> tuple[typed.D81Phase1Authority, bytes]:
    dependencies = typed._current_dependency_hashes()
    value = {
        "schema": typed.PHASE1_AUTHORITY_SCHEMA,
        "bundle_status": "IMMUTABLE_SEALED",
        "method_lock_sha256": "1" * 64,
        "phase1_bundle_receipt_sha256": "2" * 64,
        "d81_scorer_receipt_sha256": scorer.scorer_id,
        "phase1_checkpoint_sha256": scorer.phase1_checkpoint_sha256,
        "ground_manifest_sha256": scorer.ground_manifest_sha256,
        "ground_component_npz_sha256": scorer.ground_component_npz_sha256,
        "dependency_closure_sha256": typed._canonical_sha256(dict(dependencies)),
        "metric_seed": int(scorer.metric_seed),
        "ground_component_npz_serialized_bytes": int(
            Path(str(scorer.ground_audit["component_path"])).stat().st_size
        ),
        "ground_manifest_serialized_bytes": int(
            (Path(str(scorer.ground_audit["component_path"])).parent / "manifest.json").stat().st_size
        ),
        "ground_bundle_logical_state_bytes": int(
            scorer.ground_audit["ground_int8_component_logical_state_bytes"]
        ),
        "ground_retained_rank": int(scorer.ground_audit["d81_retained_rank"]),
        "ground_covariance_statistics_mac_upper_bound": int(
            scorer.ground_audit["ground_covariance_statistics_mac_upper_bound"]
        ),
        "ground_transient_dequantized_bytes": int(
            scorer.ground_audit["transient_dequantized_ground_bytes"]
        ),
    }
    raw, sha = _canonical_artifact(value)
    return typed.load_d81_phase1_authority(raw, expected_artifact_sha256=sha), raw


def _support(classes: tuple[str, ...], k: int, *, seed: int, shift: float = 0.0):
    rng = np.random.default_rng(seed)
    rows, labels, physical = [], [], []
    for class_index, class_id in enumerate(classes):
        for shot in range(k):
            z = np.float32(0.01) * rng.normal(size=160).astype(np.float32)
            fft = rng.normal(size=96).astype(np.float32)
            rf = rng.normal(size=32).astype(np.float32)
            z[class_index % 160] += np.float32(1.0 + shift)
            fft[class_index % 96] += np.float32(3.0 + shift)
            rf[class_index % 32] += np.float32(2.0 + shift)
            rows.append(np.concatenate([z, fft, rf]).astype(np.float32))
            labels.append(class_id)
            physical.append(f"physical-{class_id}-{shot}")
    return np.stack(rows), np.asarray(labels), np.asarray(physical)


def _row_authority(
    phase1: typed.D81Phase1Authority,
    old_support,
    all_support,
    old_classes: tuple[str, ...],
    final_classes: tuple[str, ...],
) -> tuple[typed.D81TargetRowAuthority, bytes]:
    old_receipt = typed._support_closure(*old_support, old_classes, "old_support")[-1]
    all_receipt = typed._support_closure(*all_support, final_classes, "all_registered_support")[-1]
    value = {
        "schema": typed.ROW_AUTHORITY_SCHEMA,
        "protocol_schema": "p2_min_v1",
        "phase2_data_status": "VALIDATED_ONCE",
        "single_leo_observation": True,
        "clean_source_runtime_access": False,
        "query_fit_access": False,
        "query_decision_policy": "per_sample_all_registered_classes",
        "capsule_id": "3" * 64,
        "split_id": "4" * 64,
        "opaque_row_receipt_sha256": "5" * 64,
        "method_lock_sha256": phase1.method_lock_sha256,
        "phase1_bundle_receipt_sha256": phase1.phase1_bundle_receipt_sha256,
        "phase1_authority_artifact_sha256": phase1.artifact_sha256,
        "k_shot": int(old_receipt["k_shot"]),
        "old_registry": list(old_classes),
        "final_registry": list(final_classes),
        "old_support": old_receipt,
        "all_registered_support": all_receipt,
    }
    raw, sha = _canonical_artifact(value)
    return typed.load_d81_target_row_authority(
        raw, expected_artifact_sha256=sha, phase1_authority=phase1
    ), raw


def _fit_fixture(tmp_path: Path, *, k: int = 1, new_shift: float = 0.0):
    scorer = _scorer(tmp_path)
    phase1, phase1_raw = _phase1_authority(scorer)
    config = typed.D81TypedTargetConfig.from_scorer(scorer, phase1)
    old = _support(OLD2, k, seed=81100 + k)
    new = _support(NEW2, k, seed=81200 + k, shift=new_shift)
    all_support = (
        np.concatenate([old[0], new[0]]),
        np.concatenate([old[1], new[1]]),
        np.concatenate([old[2], new[2]]),
    )
    row, row_raw = _row_authority(phase1, old, all_support, OLD2, OLD2 + NEW2)
    state = typed.fit_d81_typed_target_state(
        *old, *all_support, d81_scorer=scorer, config=config,
        phase1_authority=phase1, row_authority=row,
    )
    return scorer, phase1, phase1_raw, config, old, all_support, row, row_raw, state


@pytest.fixture(scope="module")
def exact_row(tmp_path_factory):
    return _fit_fixture(tmp_path_factory.mktemp("typed-d81-v2"), k=1)


def _independent_oracle(scorer, old, all_support, old_classes, final_classes):
    from scripts import probe_d81_ground_nuisance_cauchy_center as probe

    old_registered = typed.raw_concat_to_d81_registered_feature(old[0])
    all_registered = typed.raw_concat_to_d81_registered_feature(all_support[0])
    old_lookup = {value: index for index, value in enumerate(old_classes)}
    final_lookup = {value: index for index, value in enumerate(final_classes)}
    old_targets = np.asarray([old_lookup[str(value)] for value in old[1]], dtype=np.int64)
    final_targets = np.asarray([final_lookup[str(value)] for value in all_support[1]], dtype=np.int64)
    d81_fit, _calls, _transforms = probe.build_d81_fit(
        d42, scorer.nuisance_basis_fp64, scorer.spectral_weights_fp64,
        typed._json_safe(scorer.ground_audit),
    )
    log_diag, trace, _resource = d42._fit_old_only_b3_metric(
        old_registered, old_targets, len(old_classes), seed=scorer.metric_seed,
        device=torch.device("cpu"),
    )
    before_coef, before_intercept, before_audit = d81_fit(
        d42._transform(old_registered, log_diag), old_targets, len(old_classes), len(old[0]) // len(old_classes)
    )
    final_coef, final_intercept, final_audit = d81_fit(
        d42._transform(all_registered, log_diag), final_targets, len(final_classes), len(all_support[0]) // len(final_classes)
    )
    before, _ = d42._compile_state(
        old_classes, len(old_classes), log_diag, before_coef, before_intercept,
        str(before_audit["covariance_policy"]), precision="int8",
    )
    final, _ = d42._compile_state(
        final_classes, len(old_classes), log_diag, final_coef, final_intercept,
        str(final_audit["covariance_policy"]), precision="int8",
    )
    assert len(trace) == 20
    return log_diag, before, final


def test_public_query_surface_has_no_role_truth_or_caller_logits(exact_row) -> None:
    assert typed.DEPLOYMENT_STATUS.endswith("EXTERNAL_CAPSULE_PRODUCER_AND_REVIEW")
    for function in (
        typed.score_d81_typed_old_before_raw_logits,
        typed.score_d81_typed_target_raw_logits,
    ):
        assert tuple(inspect.signature(function).parameters) == ("state", "query_features")
    assert not {
        "base_logits", "logits", "probabilities", "query_labels", "query_truth",
        "receiver", "scenario", "role",
    } & set(inspect.signature(typed.score_d81_typed_target_raw_logits).parameters)
    with pytest.raises(typed.D81TypedTargetStateError, match="formal query unavailable"):
        typed.score_d81_typed_target_raw_logits(exact_row[-1], exact_row[4][0])


def test_formal_query_is_unconditional_fail_closed_against_copy_and_field_attacks(
    exact_row,
) -> None:
    state = exact_row[-1]
    query = exact_row[4][0]
    assert not hasattr(state, "__dict__")
    assert not hasattr(state.config, "__dict__")
    assert not hasattr(exact_row[1], "__dict__")
    assert not hasattr(exact_row[6], "__dict__")

    attacks = []
    flag_attack = copy.copy(state)
    object.__setattr__(flag_attack, "formal_query_authorized", True)
    attacks.append(flag_attack)

    config_attack = copy.copy(state)
    changed_config = replace(
        state.config, metric_seed=int(state.config.metric_seed) + 1
    )
    object.__setattr__(config_attack, "config", changed_config)
    attacks.append(config_attack)

    registry_attack = copy.copy(state)
    object.__setattr__(registry_attack, "classes", tuple(reversed(state.classes)))
    attacks.append(registry_attack)

    array_attack = copy.copy(state)
    changed_array = np.array(state.final_coef1_qint8, copy=True)
    changed_array[0, 0] = np.int8(0)
    object.__setattr__(array_attack, "final_coef1_qint8", changed_array)
    attacks.append(array_attack)

    for attacked in attacks:
        with pytest.raises(typed.D81TypedTargetStateError, match="formal query unavailable"):
            typed.score_d81_typed_target_raw_logits(attacked, query)
        with pytest.raises(typed.D81TypedTargetStateError, match="formal query unavailable"):
            typed.score_d81_typed_old_before_raw_logits(attacked, query)

    with pytest.raises(typed.D81TypedTargetStateError, match="registry/lifecycle drift"):
        replace(state, formal_query_authorized=True)
    local_replacement = replace(state)
    with pytest.raises(typed.D81TypedTargetStateError, match="formal query unavailable"):
        typed.score_d81_typed_target_raw_logits(local_replacement, query)
    with pytest.raises(typed.D81TypedTargetStateError, match="formal query unavailable"):
        typed.score_d81_typed_target_raw_logits(object(), None)


def test_exact_d81_d42_lifecycle_matches_independent_oracle(exact_row) -> None:
    scorer, _p1, _p1raw, _config, old, all_support, _row, _rowraw, state = exact_row
    log_diag, before, final = _independent_oracle(scorer, old, all_support, OLD2, OLD2 + NEW2)
    query = np.concatenate([old[0], all_support[0][::-1]], axis=0).astype(np.float32)
    np.testing.assert_array_equal(state.log_diag_fp32, log_diag)
    np.testing.assert_allclose(
        typed.score_d81_typed_local_diagnostic_old_before_logits(state, query),
        d42.score_d42_unified_shrinkage_lda(before, typed.raw_concat_to_d81_registered_feature(query)),
        atol=2e-6, rtol=0,
    )
    np.testing.assert_allclose(
        typed.score_d81_typed_local_diagnostic_target_logits(state, query),
        d42.score_d42_unified_shrinkage_lda(final, typed.raw_concat_to_d81_registered_feature(query)),
        atol=2e-6, rtol=0,
    )
    assert state.fit_audit["metric_fit_scope"] == "old_support_only"
    assert state.fit_audit["metric_fit_execution_count"] == 1
    assert state.fit_audit["before_head_fit_count"] == 1
    assert state.fit_audit["final_head_fit_count"] == 1


@pytest.mark.parametrize("k", [5, 10])
def test_k5_k10_old_metric_and_both_heads_match_independent_oracle(tmp_path: Path, k: int) -> None:
    scorer, _p1, _p1raw, _config, old, all_support, _row, _rowraw, state = _fit_fixture(
        tmp_path / f"oracle-k{k}", k=k
    )
    log_diag, before, final = _independent_oracle(scorer, old, all_support, OLD2, OLD2 + NEW2)
    query = all_support[0]
    np.testing.assert_array_equal(state.log_diag_fp32, log_diag)
    np.testing.assert_allclose(
        typed.score_d81_typed_local_diagnostic_old_before_logits(state, query),
        d42.score_d42_unified_shrinkage_lda(before, typed.raw_concat_to_d81_registered_feature(query)),
        atol=2e-6, rtol=0,
    )
    np.testing.assert_allclose(
        typed.score_d81_typed_local_diagnostic_target_logits(state, query),
        d42.score_d42_unified_shrinkage_lda(final, typed.raw_concat_to_d81_registered_feature(query)),
        atol=2e-6, rtol=0,
    )


@pytest.mark.parametrize("k", [1, 5, 10])
def test_two_old_two_new_k1_k5_k10_complete_lifecycle(tmp_path: Path, k: int) -> None:
    *_, state = _fit_fixture(tmp_path / f"k{k}", k=k)
    assert state.k_shot == k
    assert state.old_classes == OLD2
    assert state.classes == OLD2 + NEW2
    assert state.resource_audit["optimizer_steps"] == 20
    if k == 1:
        assert state.before_covariance_policy == "unit_covariance_equal_prior_nearest_centroid"
        assert state.final_covariance_policy == "unit_covariance_equal_prior_nearest_centroid"


@pytest.mark.parametrize("new_count", [5, 10, 20])
def test_six_old_new5_new10_new20_authority_shapes(tmp_path: Path, new_count: int) -> None:
    scorer = _scorer(tmp_path / f"shape-{new_count}")
    phase1, _ = _phase1_authority(scorer)
    config = typed.D81TypedTargetConfig.from_scorer(scorer, phase1)
    old_classes = tuple(f"old-{index}" for index in range(6))
    new_classes = tuple(f"new-{index}" for index in range(new_count))
    old = _support(old_classes, 1, seed=82000 + new_count)
    new = _support(new_classes, 1, seed=83000 + new_count)
    all_support = tuple(np.concatenate([old[index], new[index]]) for index in range(3))
    row, _ = _row_authority(phase1, old, all_support, old_classes, old_classes + new_classes)
    state = typed.fit_d81_typed_target_state(
        *old, *all_support, d81_scorer=scorer, config=config,
        phase1_authority=phase1, row_authority=row,
    )
    assert len(row.old_registry) == 6
    assert len(row.final_registry) == 6 + new_count
    assert row.final_registry[:6] == row.old_registry
    assert row.all_registered_support["row_count"] == 6 + new_count
    assert state.before_coef1_qint8.shape == (6, 288)
    assert state.final_coef1_qint8.shape == (6 + new_count, 288)
    assert len(typed.serialize_d81_typed_target_state(state)) == state.resource_audit[
        "total_wire_serialized_bytes"
    ]


def test_changing_only_new_support_cannot_change_metric_or_before_head(tmp_path: Path) -> None:
    first = _fit_fixture(tmp_path / "first", k=1, new_shift=0.0)[-1]
    second = _fit_fixture(tmp_path / "second", k=1, new_shift=0.75)[-1]
    np.testing.assert_array_equal(first.log_diag_fp32, second.log_diag_fp32)
    for name in (
        "before_coef1_qint8", "before_coef2_qint8", "before_scale1_fp16",
        "before_scale2_fp16", "before_intercept_fp16",
    ):
        np.testing.assert_array_equal(getattr(first, name), getattr(second, name))
    assert first.fit_audit["metric_trace_sha256"] == second.fit_audit["metric_trace_sha256"]
    assert first.all_registered_support_receipt_sha256 != second.all_registered_support_receipt_sha256


def test_raw_self_signed_authority_and_fake_scorer_are_rejected(exact_row) -> None:
    scorer, phase1, phase1_raw, _config, _old, _all, _row, _rowraw, _state = exact_row
    phase1_value = json.loads(phase1_raw)
    with pytest.raises(typed.D81TypedTargetStateError, match="external loader"):
        typed.D81Phase1Authority(artifact_sha256=hashlib.sha256(phase1_raw).hexdigest(), **phase1_value)
    changed = dict(phase1_value)
    changed["d81_scorer_receipt_sha256"] = "9" * 64
    changed_raw, changed_sha = _canonical_artifact(changed)
    changed_authority = typed.load_d81_phase1_authority(changed_raw, expected_artifact_sha256=changed_sha)
    with pytest.raises(typed.D81TypedTargetStateError, match="scorer/Phase1"):
        typed.D81TypedTargetConfig.from_scorer(scorer, changed_authority)
    with pytest.raises(typed.D81TypedTargetStateError, match="SHA mismatch"):
        typed.load_d81_phase1_authority(changed_raw, expected_artifact_sha256=phase1.artifact_sha256)


def test_row_receipt_resigning_and_old_prefix_attacks_fail(exact_row) -> None:
    _scorer, phase1, _p1raw, _config, _old, _all, row, row_raw, _state = exact_row
    value = json.loads(row_raw)
    value["old_registry"] = list(reversed(value["old_registry"]))
    modified_raw, _modified_sha = _canonical_artifact(value)
    with pytest.raises(typed.D81TypedTargetStateError, match="SHA mismatch"):
        typed.load_d81_target_row_authority(
            modified_raw, expected_artifact_sha256=row.artifact_sha256, phase1_authority=phase1
        )
    modified_raw, modified_sha = _canonical_artifact(value)
    with pytest.raises(typed.D81TypedTargetStateError, match="prefix"):
        typed.load_d81_target_row_authority(
            modified_raw, expected_artifact_sha256=modified_sha, phase1_authority=phase1
        )


def test_old_and_all_support_receipts_are_distinct_and_raw_members_absent(exact_row) -> None:
    _scorer, _phase1, _p1raw, _config, old, _all, _row, _rowraw, state = exact_row
    assert state.old_support_receipt_sha256 != state.all_registered_support_receipt_sha256
    audit_text = json.dumps(typed._json_safe(state.fit_audit), sort_keys=True)
    assert all(str(value) not in audit_text for value in old[2].tolist())
    assert state.fit_audit["physical_ids_persisted"] is False
    assert state.fit_audit["raw_support_features_persisted"] is False
    binding = state.row_authority_binding
    assert binding["capsule_id"] == "3" * 64
    assert binding["split_id"] == "4" * 64
    assert binding["opaque_row_receipt_sha256"] == "5" * 64
    for scope in ("old_support", "all_registered_support"):
        assert len(binding[scope]["ordered_physical_ids_root_sha256"]) == 64
        assert len(binding[scope]["ordered_feature_root_sha256"]) == 64
        assert len(binding[scope]["ordered_row_root_sha256"]) == 64


def test_actual_wire_round_trip_and_exact_deploy_audit_total_sizes(tmp_path: Path, exact_row) -> None:
    _scorer, phase1, _p1raw, _config, _old, _all, row, _rowraw, state = exact_row
    raw = typed.serialize_d81_typed_target_state(state)
    resource = state.resource_audit
    assert resource["deploy_state_serialized_bytes"] > resource["head_numeric_logical_state_bytes"]
    assert resource["audit_serialized_bytes"] > 8
    assert resource["total_wire_serialized_bytes"] == len(raw)
    assert len(raw) == 17_743
    assert resource["deploy_state_serialized_bytes"] + resource["audit_serialized_bytes"] == len(raw)
    assert resource["total_deployment_serialized_bytes_including_ground"] == (
        len(raw) + resource["ground_bundle_serialized_state_bytes"]
    )
    path = tmp_path / "state.d81"
    artifact_sha = typed.save_d81_typed_target_state(state, path)
    assert artifact_sha == hashlib.sha256(raw).hexdigest()
    loaded = typed.load_d81_typed_target_state(
        path, expected_artifact_sha256=artifact_sha,
        phase1_authority=phase1, row_authority=row,
    )
    assert typed.serialize_d81_typed_target_state(loaded) == raw
    with pytest.raises(typed.D81TypedTargetStateError, match="formal query unavailable"):
        typed.score_d81_typed_target_raw_logits(loaded, exact_row[4][0])
    with pytest.raises(typed.D81TypedTargetStateError, match="already exists"):
        typed.save_d81_typed_target_state(state, path)


def test_load_requires_external_expected_state_and_matching_authorities(tmp_path: Path, exact_row) -> None:
    _scorer, phase1, _p1raw, _config, old, all_support, _row, _rowraw, state = exact_row
    path = tmp_path / "state.d81"
    actual = typed.save_d81_typed_target_state(state, path)
    with pytest.raises(typed.D81TypedTargetStateError, match="SHA mismatch"):
        typed.load_d81_typed_target_state(
            path, expected_artifact_sha256="a" * 64,
            phase1_authority=phase1, row_authority=exact_row[6],
        )
    altered_all = (all_support[0].copy(), all_support[1].copy(), all_support[2].copy())
    altered_all[0][-1, 0] += np.float32(0.5)
    other_row, _ = _row_authority(phase1, old, altered_all, OLD2, OLD2 + NEW2)
    with pytest.raises(typed.D81TypedTargetStateError, match="authority"):
        typed.load_d81_typed_target_state(
            path, expected_artifact_sha256=actual,
            phase1_authority=phase1, row_authority=other_row,
        )


def test_load_rechecks_current_dependency_closure(monkeypatch, tmp_path: Path, exact_row) -> None:
    _scorer, phase1, _p1raw, _config, _old, _all, row, _rowraw, state = exact_row
    path = tmp_path / "state.d81"
    sha = typed.save_d81_typed_target_state(state, path)
    current = typed._current_dependency_hashes()
    changed = list(current)
    changed[0] = (changed[0][0], "a" * 64)
    monkeypatch.setattr(typed, "_current_dependency_hashes", lambda: tuple(changed))
    with pytest.raises(typed.D81TypedTargetStateError, match="dependency"):
        typed.load_d81_typed_target_state(
            path, expected_artifact_sha256=sha,
            phase1_authority=phase1, row_authority=row,
        )


def test_after_load_query_needs_no_open_or_external_hash(monkeypatch, tmp_path: Path, exact_row) -> None:
    _scorer, phase1, _p1raw, _config, old, _all, row, _rowraw, state = exact_row
    path = tmp_path / "state.d81"
    sha = typed.save_d81_typed_target_state(state, path)
    loaded = typed.load_d81_typed_target_state(
        path, expected_artifact_sha256=sha, phase1_authority=phase1, row_authority=row
    )
    expected = typed.score_d81_typed_local_diagnostic_target_logits(loaded, old[0])
    monkeypatch.setattr(Path, "open", lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("open forbidden")))
    monkeypatch.setattr(typed, "_sha256_file", lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("hash forbidden")))
    monkeypatch.setattr(typed, "_sha256_bytes", lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("full hash forbidden")))
    actual = typed.score_d81_typed_local_diagnostic_target_logits(loaded, old[0])
    np.testing.assert_array_equal(actual, expected)


def test_resource_bound_includes_raw_geometry_decode_and_complete_peak(exact_row) -> None:
    _scorer, _p1, _p1raw, _config, old, all_support, _row, _rowraw, state = exact_row
    resource = state.resource_audit
    classes = len(state.classes)
    analytic_query_lower = 1856 + 4 * 288 + 3 * 288 * classes + (2 * 288 + 1) * classes
    assert resource["query_mac_upper_bound_per_sample"] >= analytic_query_lower
    assert resource["query_raw288_geometry_macs_upper_bound"] == 1856
    assert resource["query_int8_decode_macs_upper_bound"] == 3 * 288 * classes
    minimum_peak = old[0].nbytes + all_support[0].nbytes + resource["head_numeric_logical_state_bytes"]
    assert resource["complete_fit_lifecycle_peak_bytes_upper_bound"] > minimum_peak
    assert resource["peak_includes_optimizer_gradient_and_activation_bound"] is True
    assert resource["peak_includes_ground_decode_and_two_fp32_heads"] is True
    assert resource["d46_d62_component_fit_count"] == 4
    assert resource["exact_lda_component_fit_macs"] == 97_542_144
    assert resource["fit_mac_upper_bound"] >= 97_731_960
    assert resource["complete_fit_lifecycle_peak_bytes_upper_bound"] >= 24_272_503


def test_six_old_twenty_new_k10_exact_component_inventory_and_lower_bound(
    tmp_path: Path,
) -> None:
    scorer = _scorer(tmp_path / "resource-max")
    phase1, _ = _phase1_authority(scorer)
    config = typed.D81TypedTargetConfig.from_scorer(scorer, phase1)
    old_classes = tuple(f"old-{index}" for index in range(6))
    new_classes = tuple(f"new-{index}" for index in range(20))
    old = _support(old_classes, 10, seed=89101)
    new = _support(new_classes, 10, seed=89102)
    all_support = tuple(np.concatenate([old[index], new[index]]) for index in range(3))
    row, _ = _row_authority(
        phase1, old, all_support, old_classes, old_classes + new_classes
    )
    state = typed.fit_d81_typed_target_state(
        *old, *all_support, d81_scorer=scorer, config=config,
        phase1_authority=phase1, row_authority=row,
    )
    resource = state.resource_audit
    assert resource["d46_base_component_fit_count"] == 44
    assert resource["d62_additional_component_fit_count"] == 44
    assert resource["d46_d62_component_fit_count"] == 88
    assert resource["exact_lda_component_fit_macs"] == 3_280_601_088
    assert resource["fisher_dense_algebra_macs"] == 8_408_530_944
    assert resource["support_center_translation_macs_upper_bound"] == 93_504_000
    assert resource["fit_mac_upper_bound"] >= 11_835_007_168
    assert resource["complete_fit_lifecycle_peak_bytes_upper_bound"] >= 142_162_891
    raw = typed.serialize_d81_typed_target_state(state)
    assert len(raw) == 35_706
    assert resource["total_wire_serialized_bytes"] == 35_706


def test_resource_zero_then_internal_resign_is_rejected_on_construct_and_load(
    tmp_path: Path, exact_row
) -> None:
    _scorer, phase1, _p1raw, _config, _old, _all, row, _rowraw, state = exact_row
    tampered = dict(state.resource_audit)
    tampered["fit_mac_upper_bound"] = 0
    tampered["complete_fit_lifecycle_peak_bytes_upper_bound"] = 0
    tampered, receipt = typed._resource_with_exact_wire_sizes(
        tampered, typed._state_core(state), state.fit_audit, state.arrays
    )
    wire, receipt = typed._wire_from_parts(
        typed._state_core(state), state.fit_audit, tampered, state.arrays
    )
    with pytest.raises(typed.D81TypedTargetStateError, match="receipt/resource"):
        replace(
            state, resource_audit=tampered, state_receipt_sha256=receipt
        )
    path = tmp_path / "resource-resigned.d81"
    path.write_bytes(wire)
    with pytest.raises(typed.D81TypedTargetStateError, match="receipt/resource"):
        typed.load_d81_typed_target_state(
            path, expected_artifact_sha256=hashlib.sha256(wire).hexdigest(),
            phase1_authority=phase1, row_authority=row,
        )


def test_metric_seed_changed_then_internal_resign_is_rejected_by_phase1_lock(
    tmp_path: Path, exact_row
) -> None:
    _scorer, phase1, _p1raw, config, _old, _all, row, _rowraw, state = exact_row
    changed_config = replace(config, metric_seed=config.metric_seed + 1)
    core = typed._core_payload(
        old_classes=state.old_classes, classes=state.classes, k_shot=state.k_shot,
        before_covariance_policy=state.before_covariance_policy,
        final_covariance_policy=state.final_covariance_policy, arrays=state.arrays,
        config=changed_config,
        phase1_authority_sha256=state.phase1_authority_artifact_sha256,
        row_authority_sha256=state.row_authority_artifact_sha256,
        old_support_receipt_sha256=state.old_support_receipt_sha256,
        all_support_receipt_sha256=state.all_registered_support_receipt_sha256,
        row_authority_binding=state.row_authority_binding,
        phase1_lock_binding=state.phase1_lock_binding,
        formal_query_authorized=False,
    )
    resource, _ = typed._resource_with_exact_wire_sizes(
        state.resource_audit, core, state.fit_audit, state.arrays
    )
    wire, _ = typed._wire_from_parts(core, state.fit_audit, resource, state.arrays)
    path = tmp_path / "seed-resigned.d81"
    path.write_bytes(wire)
    with pytest.raises(typed.D81TypedTargetStateError, match="config/Phase1|external lock"):
        typed.load_d81_typed_target_state(
            path, expected_artifact_sha256=hashlib.sha256(wire).hexdigest(),
            phase1_authority=phase1, row_authority=row,
        )


def test_support_permutation_is_rejected_by_external_ordered_row_receipt(exact_row) -> None:
    scorer, phase1, _p1raw, config, old, all_support, row, _rowraw, _state = exact_row
    old_order = np.arange(len(old[0]))[::-1]
    all_order = np.concatenate(
        [old_order, np.arange(len(old[0]), len(all_support[0]))]
    )
    with pytest.raises(typed.D81TypedTargetStateError, match="external row authority"):
        typed.fit_d81_typed_target_state(
            *(value[old_order] for value in old),
            *(value[all_order] for value in all_support),
            d81_scorer=scorer, config=config,
            phase1_authority=phase1, row_authority=row,
        )


@pytest.mark.parametrize("k", [1, 5, 10, 20])
def test_non_sorted_external_payload_order_matches_independent_oracle(
    tmp_path: Path, k: int
) -> None:
    scorer = _scorer(tmp_path / f"nonsorted-k{k}")
    phase1, _ = _phase1_authority(scorer)
    config = typed.D81TypedTargetConfig.from_scorer(scorer, phase1)
    old_source = _support(OLD2, k, seed=87100 + k)
    new_source = _support(NEW2, k, seed=87200 + k)
    old_classes = tuple(reversed(OLD2))
    new_classes = tuple(reversed(NEW2))

    def ordered_payload(source, registry):
        indices = np.concatenate(
            [
                np.flatnonzero(source[1] == class_id)[::-1]
                for class_id in registry
            ]
        )
        return tuple(value[indices] for value in source)

    old = ordered_payload(old_source, old_classes)
    new = ordered_payload(new_source, new_classes)
    all_support = tuple(np.concatenate([old[index], new[index]]) for index in range(3))
    final_classes = old_classes + new_classes
    row, _ = _row_authority(phase1, old, all_support, old_classes, final_classes)
    state = typed.fit_d81_typed_target_state(
        *old, *all_support, d81_scorer=scorer, config=config,
        phase1_authority=phase1, row_authority=row,
    )
    log_diag, before, final = _independent_oracle(
        scorer, old, all_support, old_classes, final_classes
    )
    query = all_support[0]
    np.testing.assert_array_equal(state.log_diag_fp32, log_diag)
    actual_before = typed.score_d81_typed_local_diagnostic_old_before_logits(
        state, query
    )
    expected_before = d42.score_d42_unified_shrinkage_lda(
        before, typed.raw_concat_to_d81_registered_feature(query)
    )
    actual_final = typed.score_d81_typed_local_diagnostic_target_logits(state, query)
    expected_final = d42.score_d42_unified_shrinkage_lda(
        final, typed.raw_concat_to_d81_registered_feature(query)
    )
    np.testing.assert_allclose(actual_before, expected_before, atol=2e-6, rtol=0)
    np.testing.assert_allclose(actual_final, expected_final, atol=2e-6, rtol=0)
    if k == 20:
        # The new K20 path must not merely fall under the historical tolerance:
        # it is the same ordered D81 numerical lifecycle bit-for-bit.
        np.testing.assert_array_equal(actual_before, expected_before)
        np.testing.assert_array_equal(actual_final, expected_final)


def test_six_old_twenty_new_k20_exact_inventory_wire_and_formal_block(
    tmp_path: Path,
) -> None:
    scorer = _scorer(tmp_path / "resource-k20-max")
    phase1, _ = _phase1_authority(scorer)
    config = typed.D81TypedTargetConfig.from_scorer(scorer, phase1)
    old_classes = tuple(f"old-{index}" for index in range(6))
    new_classes = tuple(f"new-{index}" for index in range(20))
    old = _support(old_classes, 20, seed=89201)
    new = _support(new_classes, 20, seed=89202)
    all_support = tuple(np.concatenate([old[index], new[index]]) for index in range(3))
    row, _ = _row_authority(
        phase1, old, all_support, old_classes, old_classes + new_classes
    )
    state = typed.fit_d81_typed_target_state(
        *old, *all_support, d81_scorer=scorer, config=config,
        phase1_authority=phase1, row_authority=row,
    )
    resource = state.resource_audit
    assert typed.ALLOWED_K_SHOT == (1, 5, 10, 20)
    assert resource["d46_base_component_fit_count"] == 84
    assert resource["d62_additional_component_fit_count"] == 84
    assert resource["d46_d62_component_fit_count"] == 168
    assert resource["exact_lda_component_fit_macs"] == 8_482_848_768
    assert resource["fisher_dense_algebra_macs"] == 16_052_649_984
    assert resource["support_center_translation_macs_upper_bound"] == 374_016_000
    assert resource["fit_mac_upper_bound"] == 25_096_476_544
    assert resource["complete_fit_lifecycle_peak_bytes_upper_bound"] == 352_748_491
    assert resource["query_mac_upper_bound_per_sample"] == 40_474
    assert resource["head_numeric_logical_state_bytes"] == 20_032
    raw = typed.serialize_d81_typed_target_state(state)
    assert len(raw) == resource["total_wire_serialized_bytes"] == 35_746
    assert resource["total_deployment_logical_bytes_including_ground"] == 25_848
    for formal in (
        typed.score_d81_typed_old_before_raw_logits,
        typed.score_d81_typed_target_raw_logits,
    ):
        with pytest.raises(typed.D81TypedTargetStateError, match="formal query unavailable"):
            formal(state, all_support[0][:1])


def test_query_batch_equals_individual_and_state_tamper_fails(exact_row) -> None:
    _scorer, _p1, _p1raw, _config, old, _all, _row, _rowraw, state = exact_row
    together = typed.score_d81_typed_local_diagnostic_target_logits(state, old[0])
    separate = np.concatenate([
        typed.score_d81_typed_local_diagnostic_target_logits(state, old[0][index:index + 1])
        for index in range(len(old[0]))
    ])
    np.testing.assert_array_equal(together, separate)
    changed = np.array(state.final_coef1_qint8, copy=True)
    changed[0, 0] = np.int8(int(changed[0, 0]) + 1 if changed[0, 0] < 127 else 126)
    with pytest.raises(typed.D81TypedTargetStateError, match="receipt"):
        replace(state, final_coef1_qint8=changed)
    with pytest.raises(typed.D81TypedTargetStateError, match="fit or external"):
        replace(state, _token=None)


def test_unbalanced_duplicate_missing_and_old_all_mismatch_fail_before_metric(exact_row) -> None:
    scorer, phase1, _p1raw, config, old, all_support, row, _rowraw, _state = exact_row
    duplicate_old = (old[0], old[1], old[2].copy())
    duplicate_old[2][1] = duplicate_old[2][0]
    with pytest.raises(typed.D81TypedTargetStateError, match="support"):
        typed.fit_d81_typed_target_state(
            *duplicate_old, *all_support, d81_scorer=scorer, config=config,
            phase1_authority=phase1, row_authority=row,
        )
    changed_all = (all_support[0].copy(), all_support[1], all_support[2])
    changed_all[0][0, 0] += np.float32(0.25)
    with pytest.raises(typed.D81TypedTargetStateError, match="authority"):
        typed.fit_d81_typed_target_state(
            *old, *changed_all, d81_scorer=scorer, config=config,
            phase1_authority=phase1, row_authority=row,
        )


def test_query_dtype_shape_and_nonfinite_fail_closed(exact_row) -> None:
    old = exact_row[4]
    state = exact_row[-1]
    with pytest.raises(typed.D81TypedTargetStateError, match="float32"):
        typed.score_d81_typed_local_diagnostic_target_logits(state, old[0].astype(np.float64))
    with pytest.raises(typed.D81TypedTargetStateError, match="float32"):
        typed.score_d81_typed_local_diagnostic_target_logits(state, old[0][:, :20])
    bad = old[0].copy()
    bad[0, 0] = np.nan
    with pytest.raises(typed.D81TypedTargetStateError, match="float32"):
        typed.score_d81_typed_local_diagnostic_target_logits(state, bad)
