from __future__ import annotations

import inspect
import json
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

from cvsrffi.stage2_ciaf import FEATURE_DIM, Int8DomainClassComponent

SCRIPTS = Path(__file__).resolve().parents[1] / "code" / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import run_d19_support_only_ciaf as runner


def test_cli_has_no_query_truth_role_quota_or_global_assignment_surface() -> None:
    destinations = {action.dest for action in runner.build_parser()._actions}
    forbidden = ("query", "truth", "role", "quota", "assignment")
    assert not any(token in name.lower() for name in destinations for token in forbidden)


def test_run_requires_preopen_component_and_class_binding_inputs() -> None:
    parameters = inspect.signature(runner.run).parameters
    for name in (
        "component_dir",
        "expected_component_manifest_sha256",
        "class_binding_path",
        "expected_class_binding_sha256",
    ):
        assert name in parameters
    assert runner.MODE == "development_select_unverified_component"
    assert runner.SUPPORT_QUERY_DISJOINTNESS_STATUS == "SUPPORT_ONLY_NO_QUERY_CLAIM"


def test_d20_candidate_set_is_small_mechanism_separated_and_pair_free() -> None:
    candidates = runner.preregistered_candidates()
    assert tuple(candidates) == (
        runner.IDENTITY_CANDIDATE,
        runner.GROUND_CANDIDATE,
        runner.GROUND_DIRECT_CANDIDATE,
        runner.DIAG_CANDIDATE,
        runner.DIAG_MAXOLD_CANDIDATE,
    )
    assert len(candidates) == 5
    assert all(config.pair_weight == 0.0 for config in candidates.values())
    assert candidates[runner.IDENTITY_CANDIDATE].ground_weight == 0.0
    assert candidates[runner.IDENTITY_CANDIDATE].direct_weight == 0.0
    assert candidates[runner.GROUND_CANDIDATE].ground_weight > 0.0
    assert candidates[runner.GROUND_CANDIDATE].direct_weight == 0.0
    assert candidates[runner.GROUND_DIRECT_CANDIDATE].direct_weight > 0.0
    assert candidates[runner.DIAG_MAXOLD_CANDIDATE].ground_weight > 0.0


def test_signal_extraction_binds_same_forward_and_no_legacy_zid_call() -> None:
    source = inspect.getsource(runner.run)
    assert "_extract_scene_zid" not in source
    assert "_extract_scene_signals" in source
    assert source.index("_verify_runtime_direct_logit_binding") < source.index(
        "materialize_somph_enrollment_with_signed_authority"
    )
    extraction = inspect.getsource(runner._extract_scene_signals)
    assert "feature_value" in extraction
    assert "logit_value" in extraction
    assert "direct_logit_indices" in extraction


def test_dlpack_bridge_avoids_numpy_c_api_and_restores_as_tensor(monkeypatch) -> None:
    float_rows = np.arange(6, dtype=np.float32).reshape(2, 3)
    tensor = runner._tensor_from_numpy_dlpack(
        float_rows, dtype=torch.float32, device=torch.device("cpu")
    )
    assert tensor.tolist() == float_rows.tolist()

    def blocked(*args, **kwargs):
        raise RuntimeError("legacy NumPy C-API bridge called")

    monkeypatch.setattr(torch, "as_tensor", blocked)
    with runner._numpy2_torch21_as_tensor_compatibility():
        values = torch.as_tensor(
            np.asarray([1, 2, 3], dtype=np.int64),
            dtype=torch.long,
            device=torch.device("cpu"),
        )
        assert values.tolist() == [1, 2, 3]
        with pytest.raises(RuntimeError, match="legacy NumPy"):
            torch.as_tensor([1, 2, 3])
    with pytest.raises(RuntimeError, match="legacy NumPy"):
        torch.as_tensor(np.asarray([1], dtype=np.int64))


def test_signal_extraction_does_not_call_torch_from_numpy(monkeypatch) -> None:
    class FakeRuntime(torch.nn.Module):
        def forward(self, batch):
            feature = torch.zeros((len(batch), FEATURE_DIM), dtype=torch.float32)
            feature[:, 0] = batch[:, 0, 0]
            logits = torch.stack([feature[:, 0], -feature[:, 0]], dim=1)
            return feature, logits

    def blocked(*args, **kwargs):
        raise RuntimeError("torch.from_numpy is forbidden")

    monkeypatch.setattr(torch, "from_numpy", blocked)
    rows = {"iq": np.ones((2, 2, 8), dtype=np.float32)}
    features, logits, audit = runner._extract_scene_signals(
        FakeRuntime(), torch.device("cpu"), rows, (0, 1)
    )
    assert features.shape == (2, FEATURE_DIM)
    assert logits.tolist() == [[1.0, -1.0], [1.0, -1.0]]
    assert audit["backbone_forwards"] == 2


def test_binding_explicitly_maps_every_old_class_to_direct_logit_column() -> None:
    path = Path(__file__).resolve().parents[1] / "analysis" / (
        "d19_adv3b02_class_binding_20260717.json"
    )
    value = json.loads(path.read_text(encoding="utf-8"))
    assert value["schema"] == "cvs.phase2.d20_adv3b02_class_binding.v2"
    assert [row["class_index"] for row in value["entries"]] == list(range(6))
    assert [row["direct_logit_index"] for row in value["entries"]] == list(
        range(6)
    )
    assert all(
        len(row["direct_logit_weight_row_sha256"]) == 64
        for row in value["entries"]
    )


def test_runtime_direct_logit_row_hash_binding_rejects_tampering() -> None:
    key = "model.id_backbone.cls_head.head.weight"
    weight = torch.arange(12, dtype=torch.float32).reshape(3, 4)

    class FakeRuntime:
        def state_dict(self):
            return {key: weight}

    matrix = np.ascontiguousarray(weight.numpy(), dtype=np.float32)
    binding = {
        "feature_runtime_sha256": "a" * 64,
        "direct_logit_head_state_key": key,
        "direct_logit_head_tensor_sha256": runner.hashlib.sha256(
            matrix.tobytes()
        ).hexdigest(),
        "direct_logit_weight_row_sha256": [
            runner.hashlib.sha256(np.ascontiguousarray(row).tobytes()).hexdigest()
            for row in matrix
        ],
    }
    audit = runner._verify_runtime_direct_logit_binding(
        FakeRuntime(), {"feature_runtime_sha256": "a" * 64}, binding
    )
    assert audit["verified_before_support_open"] is True
    tampered = dict(binding)
    tampered["direct_logit_weight_row_sha256"] = list(
        binding["direct_logit_weight_row_sha256"]
    )
    tampered["direct_logit_weight_row_sha256"][1] = "0" * 64
    with pytest.raises(runner.D19RunnerError, match="row binding drift"):
        runner._verify_runtime_direct_logit_binding(
            FakeRuntime(), {"feature_runtime_sha256": "a" * 64}, tampered
        )


def test_post_reception_view_lineage_keeps_one_parent_and_zero_extra_overlay() -> None:
    rows = {
        "tokens": np.asarray(["sample-a", "sample-b"]),
        "hashes": np.asarray(["1" * 64, "2" * 64]),
    }
    lineage = runner._post_reception_view_lineage(rows)
    assert len(lineage) == 2
    for index, row in enumerate(lineage):
        assert row["parent_received_iq_sha256"] == str(index + 1) * 64
        assert row["post_reception_view_count"] == 3
        assert row["additional_physical_sample_count"] == 0
        assert row["additional_leo_overlay_count"] == 0
        assert [item["view_seed"] for item in row["operators"]] == [0, 0, 0]


def test_diag_deployment_resource_counts_combined_state_and_no_unbound_after() -> None:
    old_classes = ("old-a", "old-b")
    new_classes = ("new-a",)
    q = np.zeros((2, 2, FEATURE_DIM), dtype=np.int8)
    q[:, 0, 0] = 127
    q[:, 1, 1] = 127
    component = Int8DomainClassComponent(
        q,
        np.full((2, 2), 1.0 / 127.0, dtype=np.float16),
        np.ones((2, 2), dtype=np.uint8),
        old_classes,
    )
    labels = np.asarray(["old-a", "old-b", "new-a"])
    z_id = np.zeros((3, FEATURE_DIM), dtype=np.float32)
    z_id[0, 0] = 1.0
    z_id[1, 1] = 1.0
    z_id[2, 2] = 1.0
    direct = z_id[:, :2] * 4.0
    diag = np.zeros((3, 288), dtype=np.float32)
    diag[:, :FEATURE_DIM] = z_id
    diag[0, 160] = 1.0
    diag[1, 161] = 1.0
    diag[2, 162] = 1.0
    rows = {"labels": labels}

    b3 = runner._deployment_state_audit(
        component,
        rows,
        z_id,
        direct,
        diag,
        old_classes=old_classes,
        new_classes=new_classes,
        candidate_id=runner.DIAG_CANDIDATE,
        config=runner.preregistered_candidates()[runner.DIAG_CANDIDATE],
        fit_seed=713101,
        device=torch.device("cpu"),
    )
    b0 = runner._deployment_state_audit(
        component,
        rows,
        z_id,
        direct,
        diag,
        old_classes=old_classes,
        new_classes=new_classes,
        candidate_id=runner.IDENTITY_CANDIDATE,
        config=runner.preregistered_candidates()[runner.IDENTITY_CANDIDATE],
        fit_seed=713101,
        device=torch.device("cpu"),
    )
    b4 = runner._deployment_state_audit(
        component,
        rows,
        z_id,
        direct,
        diag,
        old_classes=old_classes,
        new_classes=new_classes,
        candidate_id=runner.DIAG_MAXOLD_CANDIDATE,
        config=runner.preregistered_candidates()[runner.DIAG_MAXOLD_CANDIDATE],
        fit_seed=713101,
        device=torch.device("cpu"),
    )
    assert b0["int8_component_used_for_prediction"] is False
    assert b0["int8_component_state_bytes"] == 0
    assert b0["schema"] == "cvs.phase2.d20_target_centroid_baseline.resource.v1"
    assert b3["int8_component_used_for_prediction"] is False
    assert b3["dali_rerank_state_bytes"] == 0
    assert b3["stage2c_class_balanced_loss"] is True
    assert b3["stage2c_worst_class_surrogate_weight"] > 0.0
    assert b3["stage2c_old_raw_score_columns_bitwise_unchanged"] is True
    assert b3["registered_class_count"] == 3
    assert b3["class_count"] == 3
    assert b3["trainable_parameters"] < 50_000
    assert b3["max_adaptation_epochs_per_event"] <= 20
    assert b3["stage2c_registration_epochs"] == 0
    assert b3["stage2c_trainable_parameters"] == 0
    assert b3["trainable_parameters"] == b3["stage2b_trainable_parameters"]
    assert b3["stage2c_estimated_adaptation_macs"] == 0
    assert b3["estimated_adaptation_macs"] == (
        b3["stage2b_estimated_adaptation_macs"]
    )
    assert b4["int8_component_used_for_prediction"] is True
    assert b4["dali_rerank_state_bytes"] > 0
    assert b4["persistent_state_bytes"] == (
        b4["diag_registered_state_bytes"] + b4["dali_rerank_state_bytes"]
    )
    assert b4["persistent_state_limit_pass"] is True


def test_stage2c_resource_adds_registration_macs_for_k_greater_than_one() -> None:
    old_classes = ("old-a", "old-b")
    new_classes = ("new-a",)
    labels = np.asarray(old_classes + new_classes).repeat(2)
    feature_dim = 288
    features = np.zeros((len(labels), feature_dim), dtype=np.float32)
    for index in range(len(labels)):
        features[index, index % 3] = 1.0
        features[index, 160 + (index % 3)] = 0.5
    train = np.ones(len(labels), dtype=bool)
    old = np.isin(labels, old_classes)
    new = np.isin(labels, new_classes)
    state = runner._fit_diag_registered_state(
        features,
        labels,
        train,
        old,
        new,
        old_classes=old_classes,
        new_classes=new_classes,
        fit_seed=713101,
        device=torch.device("cpu"),
    )
    resource = state["resource"]
    expected_stage2c_macs = (
        resource["stage2c_registration_epochs"]
        * len(labels)
        * len(old_classes + new_classes)
        * feature_dim
        * 3
    )
    assert resource["stage2c_registration_epochs"] == 5
    assert resource["stage2c_trainable_parameters"] == feature_dim
    assert resource["trainable_parameters"] == (
        resource["stage2b_trainable_parameters"]
        + resource["stage2c_trainable_parameters"]
    )
    assert resource["stage2c_estimated_adaptation_macs"] == expected_stage2c_macs
    assert resource["estimated_adaptation_macs"] == (
        resource["stage2b_estimated_adaptation_macs"] + expected_stage2c_macs
    )


def _synthetic_fold(
    candidate_id: str,
    *,
    old_overall: float = 0.70,
    old_floor: float = 0.50,
    old_class: float = 0.50,
) -> dict:
    return {
        "candidate_id": candidate_id,
        "before_old": {
            "overall_accuracy": 0.75,
            "class_floor_accuracy": 0.50,
            "per_class_accuracy": {"old": 0.50},
        },
        "after_old": {
            "overall_accuracy": old_overall,
            "class_floor_accuracy": old_floor,
            "per_class_accuracy": {"old": old_class},
        },
        "after_new": {
            "overall_accuracy": 0.80,
            "class_floor_accuracy": 0.70,
            "per_class_accuracy": {"new": 0.70},
        },
        "H_old_new": 0.74,
        "joint_floor": min(old_floor, 0.70),
        "forgetting": 0.05,
        "old_score_columns_bitwise_unchanged": True,
    }


def test_selection_requires_strict_worst_old_floor_not_only_mean_old_gain() -> None:
    candidates = runner.preregistered_candidates()
    matrix = {
        candidate_id: [
            _synthetic_fold(candidate_id) for _ in range(15)
        ]
        for candidate_id in candidates
    }
    matrix[runner.GROUND_DIRECT_CANDIDATE] = [
        _synthetic_fold(
            runner.GROUND_DIRECT_CANDIDATE,
            old_overall=0.80,
            old_floor=0.50,
            old_class=0.50,
        )
        for _ in range(15)
    ]
    matrix[runner.GROUND_CANDIDATE] = [
        _synthetic_fold(
            runner.GROUND_CANDIDATE,
            old_overall=0.80,
            old_floor=0.60,
            old_class=0.60,
        )
        for _ in range(15)
    ]
    selected, decisions = runner._select_candidate(matrix)
    by_id = {row["candidate_id"]: row for row in decisions}
    assert selected == runner.GROUND_CANDIDATE
    assert (
        by_id[runner.GROUND_DIRECT_CANDIDATE]["eligible_positive_route"]
        is False
    )
    assert by_id[runner.GROUND_CANDIDATE]["eligible_positive_route"] is True
