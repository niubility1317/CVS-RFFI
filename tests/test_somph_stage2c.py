from __future__ import annotations

import json
import hashlib
from pathlib import Path

import numpy as np
import pytest

from paper_reproduction.cvs_aligned.somph_stage2c import (
    FORMAL_NEW20,
    FORMAL_OLD_CLASSES,
    FORMAL_SCENARIOS,
    ROW_MANIFEST_SCHEMA,
    REGISTRATION_PAIR_SCHEMA,
    STAGE_INPUT_SCHEMA,
    K_FAMILY_SCHEMA,
    apply_stage_head_capsule,
    array_sha256,
    build_stage_head_capsule,
    canonical_sha256,
    expected_method_lock,
    ordered_values_sha256,
    stage_head_resource_audit,
    validate_method_lock,
    validate_k_family,
    validate_registration_pair,
    validate_row_manifest,
    validate_stage_head_capsule,
)


SHA = "1" * 64


def _token(prefix: str, value: str) -> str:
    return f"{prefix}_{hashlib.sha256(value.encode('utf-8')).hexdigest()}"


def _row(*, k_shot: int = 5, new_count: int = 5) -> tuple[dict, dict]:
    method = expected_method_lock()
    row = {
        "schema": ROW_MANIFEST_SCHEMA,
        "method_lock_sha256": canonical_sha256(method),
        "split_role": "confirmation",
        "receiver": "8-8",
        "seed": 713106,
        "k_shot": k_shot,
        "new_class_count": new_count,
        "old_class_handles": list(FORMAL_OLD_CLASSES),
        "new_class_handles": list(FORMAL_NEW20[:new_count]),
        "support_pool_max_k": 20,
        "query_per_tx": 20,
        "scenarios": list(FORMAL_SCENARIOS),
    }
    return method, row


def _pool(class_count: int, k_shot: int) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray], np.ndarray, np.ndarray, np.ndarray]:
    pool_labels = np.repeat(np.arange(class_count, dtype=np.int64), 20)
    pool_ranks = np.tile(np.arange(20, dtype=np.int64), class_count)
    selected = pool_ranks < k_shot
    physical = np.asarray(
        [_token("sid", f"c{label}-r{rank}") for label, rank in zip(pool_labels, pool_ranks)]
    )
    pool_tokens = {scenario: physical.copy() for scenario in FORMAL_SCENARIOS}
    support_tokens = {scenario: values[selected] for scenario, values in pool_tokens.items()}
    return support_tokens, pool_tokens, pool_labels, pool_ranks, pool_labels[selected]


def _binding(
    row: dict,
    *,
    stage: str,
    labels: np.ndarray,
    support_tokens: dict[str, np.ndarray],
    pool_tokens: dict[str, np.ndarray],
    features: dict[str, np.ndarray],
) -> dict:
    handles = list(FORMAL_OLD_CLASSES)
    if stage == "after_registration":
        handles.extend(row["new_class_handles"])
    sequence = [handles[int(index)] for index in labels.tolist()]
    tokens = [
        _token("qid", f"{stage}-q-{index}")
        for index in range(len(handles) * int(row["query_per_tx"]))
    ]
    return {
        "schema": STAGE_INPUT_SCHEMA,
        "stage": stage,
        "row_manifest_sha256": canonical_sha256(row),
        "sealed_package_sha256": SHA,
        "preopen_audit_sha256": "2" * 64,
        "runtime_access_audit_policy_sha256": "3" * 64,
        "feature_runtime_sha256": "4" * 64,
        "registered_class_order_sha256": ordered_values_sha256(handles),
        "support_class_handles": handles,
        "support_pool_ids_sha256_by_scenario": {
            scenario: ordered_values_sha256(pool_tokens[scenario].tolist())
            for scenario in FORMAL_SCENARIOS
        },
        "support_ids_sha256_by_scenario": {
            scenario: ordered_values_sha256(support_tokens[scenario].tolist())
            for scenario in FORMAL_SCENARIOS
        },
        "support_label_sequence_sha256_by_scenario": {
            scenario: ordered_values_sha256(sequence) for scenario in FORMAL_SCENARIOS
        },
        "support_feature_sha256_by_scenario": {
            scenario: array_sha256(features[scenario]) for scenario in FORMAL_SCENARIOS
        },
        "query_ids_sha256_by_scenario": {
            scenario: ordered_values_sha256(tokens) for scenario in FORMAL_SCENARIOS
        },
        "satellite_seed_by_scenario": {
            scenario: 713100 + index for index, scenario in enumerate(FORMAL_SCENARIOS)
        },
        "support_prefix_policy": "rank_lt_k_from_locked_k20_pool",
        "support_query_overlap_count": 0,
    }


def _features(class_count: int, k_shot: int) -> tuple[dict[str, np.ndarray], np.ndarray]:
    rng = np.random.default_rng(19)
    labels = np.repeat(np.arange(class_count, dtype=np.int64), k_shot)
    centers = rng.normal(size=(class_count, 160)).astype(np.float32)
    rows = centers[labels] + 0.02 * rng.normal(size=(len(labels), 160)).astype(np.float32)
    return {scenario: rows.copy() for scenario in FORMAL_SCENARIOS}, labels


def _capsule(stage: str = "after_registration") -> tuple[dict, dict, dict, dict]:
    method, row = _row()
    class_count = 6 if stage == "before_registration" else 11
    support_tokens, pool_tokens, pool_labels, pool_ranks, labels = _pool(
        class_count, row["k_shot"]
    )
    features, _ = _features(class_count, row["k_shot"])
    binding = _binding(
        row,
        stage=stage,
        labels=labels,
        support_tokens=support_tokens,
        pool_tokens=pool_tokens,
        features=features,
    )
    capsule = build_stage_head_capsule(
        features_by_scenario=features,
        support_tokens_by_scenario=support_tokens,
        support_pool_tokens_by_scenario=pool_tokens,
        support_pool_labels=pool_labels,
        support_pool_ranks=pool_ranks,
        method_lock=method,
        row_manifest=row,
        stage_input_binding=binding,
    )
    return method, row, binding, capsule


def test_method_lock_nested_protocol_mutation_is_rejected() -> None:
    method = expected_method_lock()
    method["phase2_contract"]["clean_sample_access"] = True
    with pytest.raises(ValueError, match="method lock drift"):
        validate_method_lock(method)
    assert expected_method_lock()["phase2_contract"]["clean_sample_access"] is False


def test_committed_method_lock_matches_code_exactly() -> None:
    root = Path(__file__).resolve().parents[1]
    payload = json.loads(
        (root / "paper_reproduction" / "configs" / "somph_v1_method_lock_20260716.json")
        .read_text(encoding="utf-8")
    )
    assert validate_method_lock(payload) == expected_method_lock()


def test_method_and_row_locks_are_separate_and_cover_formal_k_stress() -> None:
    method, row = _row(k_shot=1, new_count=10)
    validate_method_lock(method)
    validate_row_manifest(row, method_lock_sha256=canonical_sha256(method))
    for k_shot in (1, 5, 10, 20):
        _, selected = _row(k_shot=k_shot)
        validate_row_manifest(selected, method_lock_sha256=canonical_sha256(method))
    drift = dict(row)
    drift["seed"] = 713101
    with pytest.raises(ValueError, match="confirmation row seed/K"):
        validate_row_manifest(drift, method_lock_sha256=canonical_sha256(method))


def test_before_stage_rejects_any_new_class_support_access() -> None:
    method, row = _row()
    features, _ = _features(7, row["k_shot"])
    support_tokens, pool_tokens, pool_labels, pool_ranks, labels = _pool(
        7, row["k_shot"]
    )
    old_only_labels = np.repeat(np.arange(6, dtype=np.int64), row["k_shot"])
    binding = _binding(
        row,
        stage="before_registration",
        labels=old_only_labels,
        support_tokens=support_tokens,
        pool_tokens=pool_tokens,
        features=features,
    )
    binding["support_class_handles"] = [*FORMAL_OLD_CLASSES, FORMAL_NEW20[0]]
    binding["registered_class_order_sha256"] = ordered_values_sha256(
        binding["support_class_handles"]
    )
    with pytest.raises(ValueError, match="cannot access classes outside"):
        build_stage_head_capsule(
            features_by_scenario=features,
            support_tokens_by_scenario=support_tokens,
            support_pool_tokens_by_scenario=pool_tokens,
            support_pool_labels=pool_labels,
            support_pool_ranks=pool_ranks,
            method_lock=method,
            row_manifest=row,
            stage_input_binding=binding,
        )


def test_stage_support_requires_exact_physical_k_per_class() -> None:
    method, row = _row()
    support_tokens, pool_tokens, pool_labels, pool_ranks, labels = _pool(
        11, row["k_shot"]
    )
    features, _ = _features(11, row["k_shot"])
    broken_features = {scenario: rows[:-1] for scenario, rows in features.items()}
    broken_support = {scenario: rows[:-1] for scenario, rows in support_tokens.items()}
    binding = _binding(
        row,
        stage="after_registration",
        labels=labels,
        support_tokens=support_tokens,
        pool_tokens=pool_tokens,
        features=features,
    )
    with pytest.raises(ValueError, match="selected support token layout drift"):
        build_stage_head_capsule(
            features_by_scenario=broken_features,
            support_tokens_by_scenario=broken_support,
            support_pool_tokens_by_scenario=pool_tokens,
            support_pool_labels=pool_labels,
            support_pool_ranks=pool_ranks,
            method_lock=method,
            row_manifest=row,
            stage_input_binding=binding,
        )


def test_stage_rejects_broken_k20_pool_or_nonprefix_selection() -> None:
    method, row = _row()
    support_tokens, pool_tokens, pool_labels, pool_ranks, labels = _pool(
        11, row["k_shot"]
    )
    features, _ = _features(11, row["k_shot"])
    binding = _binding(
        row,
        stage="after_registration",
        labels=labels,
        support_tokens=support_tokens,
        pool_tokens=pool_tokens,
        features=features,
    )
    bad_ranks = pool_ranks.copy()
    bad_ranks[19] = 0
    with pytest.raises(ValueError, match="ranks 0..19"):
        build_stage_head_capsule(
            features_by_scenario=features,
            support_tokens_by_scenario=support_tokens,
            support_pool_tokens_by_scenario=pool_tokens,
            support_pool_labels=pool_labels,
            support_pool_ranks=bad_ranks,
            method_lock=method,
            row_manifest=row,
            stage_input_binding=binding,
        )
    reordered = {scenario: values.copy() for scenario, values in support_tokens.items()}
    reordered[FORMAL_SCENARIOS[0]][[0, 1]] = reordered[FORMAL_SCENARIOS[0]][[1, 0]]
    with pytest.raises(ValueError, match="not the locked rank<K prefix"):
        build_stage_head_capsule(
            features_by_scenario=features,
            support_tokens_by_scenario=reordered,
            support_pool_tokens_by_scenario=pool_tokens,
            support_pool_labels=pool_labels,
            support_pool_ranks=pool_ranks,
            method_lock=method,
            row_manifest=row,
            stage_input_binding=binding,
        )


def test_apply_uses_fp16_capsule_and_scores_all_registered_classes() -> None:
    method, row, binding, capsule = _capsule("after_registration")
    query_count = 11 * row["query_per_tx"]
    tokens = np.asarray(
        [_token("qid", f"after_registration-q-{index}") for index in range(query_count)]
    )
    query = np.random.default_rng(5).normal(size=(query_count, 160)).astype(np.float32)
    output = apply_stage_head_capsule(
        scenario=FORMAL_SCENARIOS[0],
        query_features=query,
        query_tokens=tokens,
        capsule=capsule,
        method_lock=method,
        row_manifest=row,
        stage_input_binding=binding,
    )
    registered = {*FORMAL_OLD_CLASSES, *FORMAL_NEW20[:5]}
    assert set(output["prediction"]).issubset(registered)
    np.testing.assert_array_equal(output["query_tokens"], tokens)


def test_apply_rejects_query_token_row_or_order_misalignment() -> None:
    method, row, binding, capsule = _capsule("after_registration")
    with pytest.raises(ValueError, match="token layout drift"):
        apply_stage_head_capsule(
            scenario=FORMAL_SCENARIOS[0],
            query_features=np.zeros((3, 160), dtype=np.float32),
            query_tokens=np.asarray([_token("qid", "a"), _token("qid", "b")]),
            capsule=capsule,
            method_lock=method,
            row_manifest=row,
            stage_input_binding=binding,
        )
    with pytest.raises(ValueError, match="digest/order drift"):
        apply_stage_head_capsule(
            scenario=FORMAL_SCENARIOS[0],
            query_features=np.zeros((220, 160), dtype=np.float32),
            query_tokens=np.asarray(
                [_token("qid", f"after_registration-q-{index}") for index in [1, 0, *range(2, 220)]]
            ),
            capsule=capsule,
            method_lock=method,
            row_manifest=row,
            stage_input_binding=binding,
        )


def test_capsule_rejects_extra_truth_or_out_of_lock_scale() -> None:
    method, row, binding, capsule = _capsule("after_registration")
    with_truth = dict(capsule)
    with_truth["query_truth"] = np.asarray([0], dtype=np.int64)
    with pytest.raises(ValueError, match="exact member/order drift"):
        validate_stage_head_capsule(
            with_truth,
            method_lock=method,
            row_manifest=row,
            stage_input_binding=binding,
        )
    bad_scale = dict(capsule)
    key = f"head__{FORMAL_SCENARIOS[0]}__residual_scale_fp16"
    bad_scale[key] = np.full_like(bad_scale[key], 100.0)
    with pytest.raises(ValueError, match="head payload digest drift"):
        validate_stage_head_capsule(
            bad_scale,
            method_lock=method,
            row_manifest=row,
            stage_input_binding=binding,
        )


def test_resource_audit_counts_registry_but_claims_candidate_only() -> None:
    method, row, binding, capsule = _capsule("after_registration")
    audit = stage_head_resource_audit(
        capsule,
        method_lock=method,
        row_manifest=row,
        stage_input_binding=binding,
    )
    assert audit["trainable_parameters"] == 0
    assert audit["optimizer_steps"] == 0
    assert audit["candidate_state_within_cap"] is True
    assert audit["capsule_array_bytes_including_registry_and_audit"] > audit[
        "candidate_state_bytes_fp16"
    ]


def test_support_feature_tensor_is_bound_to_stage_input() -> None:
    method, row = _row()
    support_tokens, pool_tokens, pool_labels, pool_ranks, labels = _pool(11, row["k_shot"])
    features, _ = _features(11, row["k_shot"])
    binding = _binding(
        row,
        stage="after_registration",
        labels=labels,
        support_tokens=support_tokens,
        pool_tokens=pool_tokens,
        features=features,
    )
    changed = {scenario: rows.copy() for scenario, rows in features.items()}
    changed[FORMAL_SCENARIOS[0]][0, 0] += 1.0
    with pytest.raises(ValueError, match="feature tensor digest drift"):
        build_stage_head_capsule(
            features_by_scenario=changed,
            support_tokens_by_scenario=support_tokens,
            support_pool_tokens_by_scenario=pool_tokens,
            support_pool_labels=pool_labels,
            support_pool_ranks=pool_ranks,
            method_lock=method,
            row_manifest=row,
            stage_input_binding=binding,
        )


def test_stage2b_new_count_zero_is_supported_only_before_registration() -> None:
    method, row = _row(new_count=0)
    support_tokens, pool_tokens, pool_labels, pool_ranks, labels = _pool(6, row["k_shot"])
    features, _ = _features(6, row["k_shot"])
    before = _binding(
        row,
        stage="before_registration",
        labels=labels,
        support_tokens=support_tokens,
        pool_tokens=pool_tokens,
        features=features,
    )
    build_stage_head_capsule(
        features_by_scenario=features,
        support_tokens_by_scenario=support_tokens,
        support_pool_tokens_by_scenario=pool_tokens,
        support_pool_labels=pool_labels,
        support_pool_ranks=pool_ranks,
        method_lock=method,
        row_manifest=row,
        stage_input_binding=before,
    )
    after = dict(before)
    after["stage"] = "after_registration"
    with pytest.raises(ValueError, match="Stage2-B row only supports"):
        build_stage_head_capsule(
            features_by_scenario=features,
            support_tokens_by_scenario=support_tokens,
            support_pool_tokens_by_scenario=pool_tokens,
            support_pool_labels=pool_labels,
            support_pool_ranks=pool_ranks,
            method_lock=method,
            row_manifest=row,
            stage_input_binding=after,
        )


def test_query_tokens_must_be_opaque_and_have_exact_count() -> None:
    method, row, binding, capsule = _capsule("after_registration")
    with pytest.raises(ValueError, match="not opaque"):
        apply_stage_head_capsule(
            scenario=FORMAL_SCENARIOS[0],
            query_features=np.zeros((220, 160), dtype=np.float32),
            query_tokens=np.asarray([f"target_old|14-10|{index}" for index in range(220)]),
            capsule=capsule,
            method_lock=method,
            row_manifest=row,
            stage_input_binding=binding,
        )
    with pytest.raises(ValueError, match="token layout drift"):
        apply_stage_head_capsule(
            scenario=FORMAL_SCENARIOS[0],
            query_features=np.zeros((1, 160), dtype=np.float32),
            query_tokens=np.asarray([_token("qid", "one")]),
            capsule=capsule,
            method_lock=method,
            row_manifest=row,
            stage_input_binding=binding,
        )


def test_registration_pair_locks_matched_old_support_query_and_runtime() -> None:
    method, row = _row()
    before_support, before_pool, _pl, _pr, before_labels = _pool(6, row["k_shot"])
    before_features, _ = _features(6, row["k_shot"])
    before = _binding(
        row,
        stage="before_registration",
        labels=before_labels,
        support_tokens=before_support,
        pool_tokens=before_pool,
        features=before_features,
    )
    after_support, after_pool, _apl, _apr, after_labels = _pool(11, row["k_shot"])
    after_features, _ = _features(11, row["k_shot"])
    after = _binding(
        row,
        stage="after_registration",
        labels=after_labels,
        support_tokens=after_support,
        pool_tokens=after_pool,
        features=after_features,
    )
    pair = {
        "schema": REGISTRATION_PAIR_SCHEMA,
        "row_manifest_sha256": canonical_sha256(row),
        "before_binding_sha256": canonical_sha256(before),
        "after_binding_sha256": canonical_sha256(after),
        "old_support_physical_ids_sha256_before": "7" * 64,
        "old_support_physical_ids_sha256_after": "7" * 64,
        "old_query_physical_ids_sha256_before": "8" * 64,
        "old_query_physical_ids_sha256_after": "8" * 64,
    }
    validate_registration_pair(
        pair,
        method_lock=method,
        row_manifest=row,
        before_binding=before,
        after_binding=after,
    )
    broken = dict(pair)
    broken["old_query_physical_ids_sha256_after"] = "9" * 64
    with pytest.raises(ValueError, match="old query mismatch"):
        validate_registration_pair(
            broken,
            method_lock=method,
            row_manifest=row,
            before_binding=before,
            after_binding=after,
        )


def test_k_family_shares_pool_query_runtime_and_satellite_seeds() -> None:
    method = expected_method_lock()
    rows: dict[int, dict] = {}
    bindings: dict[int, dict] = {}
    for k_shot in (1, 5, 10, 20):
        _, row = _row(k_shot=k_shot)
        support, pool, _pl, _pr, labels = _pool(11, k_shot)
        features, _ = _features(11, k_shot)
        binding = _binding(
            row,
            stage="after_registration",
            labels=labels,
            support_tokens=support,
            pool_tokens=pool,
            features=features,
        )
        rows[k_shot] = row
        bindings[k_shot] = binding
    family = {
        "schema": K_FAMILY_SCHEMA,
        "stage": "after_registration",
        "row_manifest_sha256_by_k": {
            str(k): canonical_sha256(rows[k]) for k in (1, 5, 10, 20)
        },
        "binding_sha256_by_k": {
            str(k): canonical_sha256(bindings[k]) for k in (1, 5, 10, 20)
        },
    }
    validate_k_family(
        family,
        method_lock=method,
        row_manifests_by_k=rows,
        bindings_by_k=bindings,
    )
    broken_bindings = dict(bindings)
    broken_bindings[5] = json.loads(json.dumps(bindings[5]))
    broken_bindings[5]["query_ids_sha256_by_scenario"] = {
        scenario: "a" * 64 for scenario in FORMAL_SCENARIOS
    }
    broken_family = json.loads(json.dumps(family))
    broken_family["binding_sha256_by_k"]["5"] = canonical_sha256(broken_bindings[5])
    with pytest.raises(ValueError, match="binding mismatch: query_ids"):
        validate_k_family(
            broken_family,
            method_lock=method,
            row_manifests_by_k=rows,
            bindings_by_k=broken_bindings,
        )
