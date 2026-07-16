from __future__ import annotations

import hashlib
import inspect

import numpy as np
import pytest
import torch

from cvsrffi.somph_predictor_runtime import (
    ADV3B02_CHECKPOINT_SHA256,
    SOMPH_ENROLLMENT_BINDING_SCHEMA,
    SomphPredictorRuntimeError,
    apply_somph_heads,
    assert_role_oracle_free_public_api,
    canonical_sha256,
    enroll_somph_heads,
    expected_somph_method_lock,
    somph_head_capsule_members,
    validate_somph_head_capsule,
)
from cvsrffi.stage2_predictor_bundle import FORMAL_LEO_WEAK_SCENARIOS
from paper_reproduction.cvs_aligned.somph_stage2c import expected_method_lock


def _token(prefix: str, value: str) -> str:
    return f"{prefix}_{hashlib.sha256(value.encode('utf-8')).hexdigest()}"


class _Runtime(torch.nn.Module):
    def __init__(self, feature_dim: int = 160) -> None:
        super().__init__()
        self.feature_dim = feature_dim

    def forward(self, rows: torch.Tensor):
        flat = rows.flatten(1)
        repeats = (self.feature_dim + flat.shape[1] - 1) // flat.shape[1]
        features = flat.repeat(1, repeats)[:, : self.feature_dim]
        logits = features[:, :6]
        return features, logits


class _Fp16Runtime(_Runtime):
    def forward(self, rows: torch.Tensor):
        features, logits = super().forward(rows)
        return features.to(torch.float16), logits.to(torch.float16)


def _support(class_count: int = 3, max_k: int = 20) -> dict[str, dict[str, np.ndarray]]:
    labels = np.repeat(np.arange(class_count, dtype=np.int64), max_k)
    ranks = np.tile(np.arange(max_k, dtype=np.int64), class_count)
    result = {}
    for scenario_index, scenario in enumerate(FORMAL_LEO_WEAK_SCENARIOS):
        rows = []
        for class_index, rank in zip(labels, ranks):
            base = float(class_index * 10 + rank + 1)
            rows.append(
                np.asarray(
                    [
                        [base + scenario_index, class_index + 0.1, rank + 0.2, 1.0],
                        [rank + 0.3, base * 0.1, scenario_index + 0.4, 0.5],
                    ],
                    dtype=np.float32,
                )
            )
        result[scenario] = {
            "support_leo_weak_iq": np.stack(rows),
            "support_class_indices": labels,
            "support_rank_within_class": ranks,
            "support_tokens": np.asarray(
                [_token("sid", f"{class_index}-{rank}") for class_index, rank in zip(labels, ranks)]
            ),
            "support_overlay_tokens": np.asarray(
                [_token("oid", f"{scenario}-{index}") for index in range(len(labels))]
            ),
            "support_satellite_seeds": np.arange(
                len(labels), dtype=np.int64
            ),
            "support_post_channel_iq_sha256": np.asarray(
                ["a" * 64] * len(labels)
            ),
        }
    return result


def _query(order: tuple[int, ...] = (0, 1, 2, 3)) -> dict[str, dict[str, np.ndarray]]:
    result = {}
    base_rows = np.asarray(
        [
            [[1.0, 0.1, 0.2, 1.0], [0.3, 0.1, 0.4, 0.5]],
            [[11.0, 1.1, 0.2, 1.0], [0.3, 1.1, 0.4, 0.5]],
            [[21.0, 2.1, 0.2, 1.0], [0.3, 2.1, 0.4, 0.5]],
            [[12.0, 1.2, 0.4, 1.0], [0.5, 1.2, 0.4, 0.5]],
        ],
        dtype=np.float32,
    )
    tokens = np.asarray([_token("qid", f"q-{index}") for index in range(len(base_rows))])
    indices = np.asarray(order, dtype=np.int64)
    for scenario_index, scenario in enumerate(FORMAL_LEO_WEAK_SCENARIOS):
        rows = base_rows.copy()
        rows[:, 0, 0] += scenario_index
        result[scenario] = {
            "query_leo_weak_iq": rows[indices],
            "query_tokens": tokens[indices],
            "query_overlay_tokens": np.asarray(
                [_token("oid", f"{scenario}-q-{index}") for index in indices]
            ),
            "query_satellite_seeds": np.arange(len(indices), dtype=np.int64),
            "query_post_channel_iq_sha256": np.asarray(
                ["b" * 64] * len(indices)
            ),
        }
    return result


def _handles(class_count: int) -> list[str]:
    return [_token("cls", f"class-{index}") for index in range(class_count)]


def _enrollment_input(class_count: int, *, k_shot: int) -> dict:
    lock = expected_somph_method_lock()
    return {
        "schema": SOMPH_ENROLLMENT_BINDING_SCHEMA,
        "stage": "stage2c",
        "registration_state": "after",
        "receiver": "20-1",
        "seed": 713101,
        "k_shot": k_shot,
        "registered_class_handles": _handles(class_count),
        "enrollment_package_root_sha256": "1" * 64,
        "enrollment_package_seal_sha256": "2" * 64,
        "checkpoint_sha256": ADV3B02_CHECKPOINT_SHA256,
        "method_lock_sha256": canonical_sha256(lock),
    }


def test_runtime_method_lock_matches_the_locked_algorithm_definition() -> None:
    assert expected_somph_method_lock() == expected_method_lock()


def test_public_runtime_api_has_no_old_new_or_query_oracle_controls() -> None:
    assert_role_oracle_free_public_api()
    forbidden = {
        "old_class_count",
        "new_class_count",
        "query_labels",
        "query_roles",
        "query_per_tx",
        "class_quota",
    }
    for function in (enroll_somph_heads, apply_somph_heads):
        assert not (forbidden & set(inspect.signature(function).parameters))


def test_enrollment_and_apply_apis_are_separate_single_view_paths() -> None:
    model = _Runtime().eval()
    capsule, enrollment = enroll_somph_heads(
        model,
        _support(),
        enrollment_binding=_enrollment_input(3, k_shot=5),
        method_lock=expected_somph_method_lock(),
        device=torch.device("cpu"),
        batch_size=16,
    )
    assert tuple(capsule) == somph_head_capsule_members()
    assert not any("query" in name or "truth" in name or "role" in name for name in capsule)
    assert enrollment["query_rows_used_for_fit"] == 0
    assert enrollment["trainable_parameters"] == 0
    payload, apply_receipt = apply_somph_heads(
        model,
        _query(),
        capsule,
        registered_class_handles=enrollment["registered_class_handles"],
        expected_enrollment_binding_sha256=enrollment[
            "enrollment_binding_sha256"
        ],
        method_lock=expected_somph_method_lock(),
        device=torch.device("cpu"),
        batch_size=1,
    )
    assert set(payload) == {
        "query_tokens",
        "scenarios",
        "predicted_class_indices",
        "backbone_forward_counts",
    }
    assert np.all(payload["backbone_forward_counts"] == 1)
    assert apply_receipt["mean_backbone_forward_count"] == 1.0


def test_per_sample_predictions_are_query_order_and_batch_composition_invariant() -> None:
    model = _Runtime().eval()
    lock = expected_somph_method_lock()
    capsule, _receipt = enroll_somph_heads(
        model,
        _support(),
        enrollment_binding=_enrollment_input(3, k_shot=10),
        method_lock=lock,
        device=torch.device("cpu"),
    )
    full, _ = apply_somph_heads(
        model,
        _query((0, 1, 2, 3)),
        capsule,
        registered_class_handles=_handles(3),
        expected_enrollment_binding_sha256=_receipt[
            "enrollment_binding_sha256"
        ],
        method_lock=lock,
        device=torch.device("cpu"),
    )
    shuffled, _ = apply_somph_heads(
        model,
        _query((3, 1, 0, 2)),
        capsule,
        registered_class_handles=_handles(3),
        expected_enrollment_binding_sha256=_receipt[
            "enrollment_binding_sha256"
        ],
        method_lock=lock,
        device=torch.device("cpu"),
    )
    full_map = dict(
        zip(
            zip(full["scenarios"].tolist(), full["query_tokens"].tolist()),
            full["predicted_class_indices"].tolist(),
        )
    )
    shuffled_map = dict(
        zip(
            zip(shuffled["scenarios"].tolist(), shuffled["query_tokens"].tolist()),
            shuffled["predicted_class_indices"].tolist(),
        )
    )
    assert full_map == shuffled_map

    singleton = {}
    for scenario in FORMAL_LEO_WEAK_SCENARIOS:
        values = _query((1,))[scenario]
        singleton[scenario] = values
    one, _ = apply_somph_heads(
        model,
        singleton,
        capsule,
        registered_class_handles=_handles(3),
        expected_enrollment_binding_sha256=_receipt[
            "enrollment_binding_sha256"
        ],
        method_lock=lock,
        device=torch.device("cpu"),
    )
    for scenario, token, prediction in zip(
        one["scenarios"].tolist(),
        one["query_tokens"].tolist(),
        one["predicted_class_indices"].tolist(),
    ):
        assert full_map[(scenario, token)] == prediction


def test_runtime_rejects_non_zid160_backbone_output() -> None:
    with pytest.raises(SomphPredictorRuntimeError, match="z_id160"):
        enroll_somph_heads(
            _Runtime(feature_dim=32).eval(),
            _support(),
            enrollment_binding=_enrollment_input(3, k_shot=5),
            method_lock=expected_somph_method_lock(),
            device=torch.device("cpu"),
        )


def test_runtime_rejects_fp16_backbone_output_before_numpy_conversion() -> None:
    with pytest.raises(SomphPredictorRuntimeError, match="z_id160_fp32"):
        enroll_somph_heads(
            _Fp16Runtime().eval(),
            _support(),
            enrollment_binding=_enrollment_input(3, k_shot=5),
            method_lock=expected_somph_method_lock(),
            device=torch.device("cpu"),
        )


def test_method_lock_and_capsule_tamper_fail_closed() -> None:
    drift = expected_somph_method_lock()
    drift["hubness_weight"] = 0.5
    with pytest.raises(SomphPredictorRuntimeError, match="method lock drift"):
        enroll_somph_heads(
            _Runtime().eval(),
            _support(),
            enrollment_binding=_enrollment_input(3, k_shot=5),
            method_lock=drift,
            device=torch.device("cpu"),
        )
    capsule, _ = enroll_somph_heads(
        _Runtime().eval(),
        _support(),
        enrollment_binding=_enrollment_input(3, k_shot=5),
        method_lock=expected_somph_method_lock(),
        device=torch.device("cpu"),
    )
    tampered = dict(capsule)
    tampered["forbidden_query_labels"] = np.asarray([0], dtype=np.int64)
    with pytest.raises(SomphPredictorRuntimeError, match="exact member"):
        validate_somph_head_capsule(
            tampered,
            method_lock=expected_somph_method_lock(),
        )
    scalar_tamper = dict(capsule)
    scalar_key = f"{FORMAL_LEO_WEAK_SCENARIOS[0]}__scalars_fp16"
    scalar_tamper[scalar_key] = np.asarray([0.5, 0.25], dtype=np.float16)
    with pytest.raises(SomphPredictorRuntimeError, match="scoring scalar"):
        validate_somph_head_capsule(
            scalar_tamper,
            method_lock=expected_somph_method_lock(),
        )
    prototype_tamper = dict(capsule)
    id_key = f"{FORMAL_LEO_WEAK_SCENARIOS[0]}__prototype_class_ids_uint16"
    prototype_tamper[id_key] = np.asarray(
        prototype_tamper[id_key][:-1], dtype=np.uint16
    )
    prototype_key = f"{FORMAL_LEO_WEAK_SCENARIOS[0]}__prototypes_fp16"
    prototype_tamper[prototype_key] = np.asarray(
        prototype_tamper[prototype_key][:-1], dtype=np.float16
    )
    with pytest.raises(SomphPredictorRuntimeError, match="prototype count"):
        validate_somph_head_capsule(
            prototype_tamper,
            method_lock=expected_somph_method_lock(),
        )


def test_26_class_state_and_macs_are_recomputed_from_actual_capsule() -> None:
    capsule, receipt = enroll_somph_heads(
        _Runtime().eval(),
        _support(class_count=26),
        enrollment_binding=_enrollment_input(26, k_shot=10),
        method_lock=expected_somph_method_lock(),
        device=torch.device("cpu"),
        batch_size=128,
    )
    audited = validate_somph_head_capsule(
        capsule,
        method_lock=expected_somph_method_lock(),
    )
    assert audited["candidate_state_bytes_fp16"] == 76_320
    assert audited["active_scenario_state_bytes_fp16"] == 25_440
    assert audited["candidate_extra_macs_per_query"] == 13_142
    assert receipt["candidate_state_bytes_fp16"] == audited["candidate_state_bytes_fp16"]


def test_query_extra_oracle_field_and_registry_replay_are_rejected() -> None:
    lock = expected_somph_method_lock()
    capsule, receipt = enroll_somph_heads(
        _Runtime().eval(),
        _support(),
        enrollment_binding=_enrollment_input(3, k_shot=5),
        method_lock=lock,
        device=torch.device("cpu"),
    )
    query = _query()
    query[FORMAL_LEO_WEAK_SCENARIOS[0]]["query_roles"] = np.asarray(
        ["target_old"] * 4
    )
    with pytest.raises(SomphPredictorRuntimeError, match="exact schema"):
        apply_somph_heads(
            _Runtime().eval(),
            query,
            capsule,
            registered_class_handles=_handles(3),
            expected_enrollment_binding_sha256=receipt[
                "enrollment_binding_sha256"
            ],
            method_lock=lock,
            device=torch.device("cpu"),
        )
    reordered = list(reversed(_handles(3)))
    with pytest.raises(SomphPredictorRuntimeError, match="registry"):
        apply_somph_heads(
            _Runtime().eval(),
            _query(),
            capsule,
            registered_class_handles=reordered,
            expected_enrollment_binding_sha256=receipt[
                "enrollment_binding_sha256"
            ],
            method_lock=lock,
            device=torch.device("cpu"),
        )


def test_apply_rejects_non_singleton_backbone_batching() -> None:
    lock = expected_somph_method_lock()
    capsule, receipt = enroll_somph_heads(
        _Runtime().eval(),
        _support(),
        enrollment_binding=_enrollment_input(3, k_shot=5),
        method_lock=lock,
        device=torch.device("cpu"),
    )
    with pytest.raises(SomphPredictorRuntimeError, match="singleton"):
        apply_somph_heads(
            _Runtime().eval(),
            _query(),
            capsule,
            registered_class_handles=_handles(3),
            expected_enrollment_binding_sha256=receipt[
                "enrollment_binding_sha256"
            ],
            method_lock=lock,
            device=torch.device("cpu"),
            batch_size=4,
        )
