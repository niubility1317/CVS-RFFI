from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest


SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "code"
    / "scripts"
    / "run_d9_support_only_enrollment.py"
)
SPEC = importlib.util.spec_from_file_location("d9_support_runner", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
runner = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(runner)


def _manifest(state: str = "before") -> dict[str, object]:
    scenarios = list(runner.FORMAL_LEO_WEAK_SCENARIOS)
    members = [
        {"kind": "feature_runtime"},
        {"kind": "method_lock"},
        {"kind": "overlay_provenance"},
        *[{"kind": f"support:{scenario}"} for scenario in scenarios],
    ]
    payload: dict[str, object] = {
        "profile": "enrollment_only",
        "registration_state": state,
        "k_shot": 10,
        "phase2_sample_view_policy": "leo_weak_only_no_clean_access",
        "phase2_physical_sample_observation_policy": (
            "single_leo_weak_observation_per_physical_sample"
        ),
        "phase2_pretrained_artifact_policy": (
            "sealed_phase1_checkpoint_only"
        ),
        "phase2_query_decision_policy": (
            "per_sample_all_registered_classes"
        ),
        "phase2_post_reception_view_from_fixed_received_iq_only": True,
        "target_channel_scenarios": scenarios,
        "members": members,
    }
    for field in (
        "clean_sample_access",
        "clean_derived_signal_access",
        "phase2_clean_dataset_reachable",
        "phase2_clean_cache_reachable",
        "phase2_clean_control_flow_reachable",
        "phase2_source_sample_access",
        "phase2_source_derived_signal_access",
        "phase2_source_cache_access",
        "phase2_source_label_access",
        "phase2_source_replay",
        "phase2_additional_leo_channel_state_generation",
        "phase2_post_reception_view_counts_as_additional_physical_sample",
        "phase2_query_post_reception_view_fit_access",
        "phase2_query_role_oracle_access",
        "phase2_query_true_batch_class_count_access",
        "phase2_query_class_quota_access",
        "phase2_query_batch_global_assignment",
    ):
        payload[field] = False
    return payload


def _support_payload() -> tuple[dict[str, np.ndarray], dict[str, object]]:
    labels = ("cls_a", "cls_b")
    iq = np.arange(20 * 2 * 8, dtype=np.float32).reshape(20, 2, 8)
    hashes = np.asarray(
        [
            runner.hashlib.sha256(
                np.ascontiguousarray(row).tobytes()
            ).hexdigest()
            for row in iq
        ]
    )
    payload = {
        "support_leo_weak_iq": iq,
        "support_class_indices": np.repeat(
            np.arange(2, dtype=np.int64), 10
        ),
        "support_rank_within_class": np.tile(
            np.arange(10, dtype=np.int64), 2
        ),
        "support_tokens": np.asarray(
            [f"sid_{index:064x}" for index in range(20)]
        ),
        "support_overlay_tokens": np.asarray(
            [f"oid_{index:064x}" for index in range(20)]
        ),
        "support_satellite_seeds": np.full(
            20, 71310100, dtype=np.int64
        ),
        "support_post_channel_iq_sha256": hashes,
    }
    manifest = {
        "registered_classes": [
            {"class_handle": label, "class_index": index}
            for index, label in enumerate(labels)
        ]
    }
    return payload, manifest


def test_parser_has_no_query_truth_prediction_score_or_scorer_argument():
    destinations = {
        action.dest for action in runner.build_parser()._actions
    }
    assert destinations == {
        "help",
        "before_root",
        "before_seal",
        "after_root",
        "after_seal",
        "output",
        "device",
    }
    assert not destinations.intersection(
        {"query", "truth", "prediction", "score", "scorer"}
    )


def test_manifest_is_exact_enrollment_only_and_rejects_query_member():
    runner._validate_manifest(
        _manifest(), registration_state="before"
    )
    drift = _manifest()
    drift["members"] = list(drift["members"]) + [
        {"kind": "query:leo_clear_weak"}
    ]
    with pytest.raises(
        runner.D9SupportRunnerError, match="allowlist"
    ):
        runner._validate_manifest(
            drift, registration_state="before"
        )


def test_support_payload_verifies_iq_hash_unique_lineage_and_uniform_k10():
    payload, manifest = _support_payload()
    rows = runner._payload_rows(
        payload,
        manifest,
        scenario="leo_clear_weak",
    )
    assert len(rows["iq"]) == 20
    assert set(np.unique(rows["ranks"]).tolist()) == set(range(10))
    bad = dict(payload)
    bad["support_post_channel_iq_sha256"] = (
        payload["support_post_channel_iq_sha256"].copy()
    )
    bad["support_post_channel_iq_sha256"][0] = "0" * 64
    with pytest.raises(
        runner.D9SupportRunnerError, match="payload drift"
    ):
        runner._payload_rows(
            bad,
            manifest,
            scenario="leo_clear_weak",
        )
    doubled = {
        key: np.concatenate([value, value], axis=0)
        for key, value in payload.items()
    }
    doubled["support_rank_within_class"] = np.tile(
        np.arange(20, dtype=np.int64), 2
    )
    doubled["support_tokens"] = np.asarray(
        [f"sid_{1000 + index:064x}" for index in range(40)]
    )
    doubled["support_overlay_tokens"] = np.asarray(
        [f"oid_{1000 + index:064x}" for index in range(40)]
    )
    doubled["support_post_channel_iq_sha256"] = np.asarray(
        [
            runner.hashlib.sha256(
                np.ascontiguousarray(row).tobytes()
            ).hexdigest()
            for row in doubled["support_leo_weak_iq"]
        ]
    )
    with pytest.raises(
        runner.D9SupportRunnerError, match="strict K10-only"
    ):
        runner._payload_rows(
            doubled,
            manifest,
            scenario="leo_clear_weak",
        )


def test_state_writer_is_create_only_and_hash_bound(tmp_path: Path):
    import cvsrffi.stage2_class_conditional_iq_head as d7a
    import cvsrffi.stage2_floor_sparse_operator_fusion as d9

    rng = np.random.default_rng(9)
    labels = np.repeat(np.asarray(["a", "b"]), 10)
    features = {
        operator: rng.normal(size=(20, 5)).astype(np.float32)
        for operator in d7a.OPERATORS
    }
    hashes = tuple(f"{index:064x}" for index in range(20))
    state = d9.fit_floor_sparse_operator_fusion(
        features,
        d9.build_operator_feature_provenance(hashes, view_seed=0),
        labels,
        physical_sample_ids=tuple(
            f"sid_{index:064x}" for index in range(20)
        ),
        parent_received_iq_sha256=hashes,
        base_resource_audit={
            "persistent_state_bytes": 0,
            "estimated_head_macs_per_query": 0,
        },
        floor_priority_classes=(),
    )
    result = runner._write_state_new(
        tmp_path, stem="state", state=state
    )
    assert set(result) == {"npz_sha256", "metadata_sha256"}
    metadata = json.loads(
        (tmp_path / "state.json").read_text(encoding="utf-8")
    )
    assert metadata["selection_lock_k"] == 10
    with pytest.raises(FileExistsError):
        runner._write_state_new(
            tmp_path, stem="state", state=state
        )
