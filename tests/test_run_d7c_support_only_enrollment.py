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
    / "run_d7c_support_only_enrollment.py"
)
SPEC = importlib.util.spec_from_file_location("d7c_support_runner", SCRIPT)
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
            for index, label in enumerate(("cls_a", "cls_b"))
        ]
    }
    return payload, manifest


def _artifact_and_state():
    import cvsrffi.stage2_class_conditional_iq_head as d7a
    import cvsrffi.stage2_class_conditional_local_boundary as d7c

    rng = np.random.default_rng(17)
    iq = rng.normal(size=(20, 2, 32)).astype(np.float32)
    ids = tuple(f"sid_{index:064x}" for index in range(20))
    hashes = tuple(
        runner.hashlib.sha256(
            np.ascontiguousarray(row).tobytes()
        ).hexdigest()
        for row in iq
    )
    labels = np.repeat(np.asarray(["a", "b"]), 10)

    def extractor(view: np.ndarray) -> np.ndarray:
        flat = view.reshape(len(view), -1)
        return np.stack(
            (
                flat.mean(axis=1),
                flat.std(axis=1),
                flat[:, 0],
                flat[:, -1],
            ),
            axis=1,
        ).astype(np.float32)

    artifact = d7a.build_validated_operator_feature_artifact(
        iq,
        feature_extractor=extractor,
        physical_sample_ids=ids,
        parent_received_iq_sha256=hashes,
    )
    base = d7a.fit_class_conditional_head(artifact, labels)
    state = d7c.lock_k10_class_conditional_local_boundary_strategy(
        d7c.fit_class_conditional_local_boundary(base, artifact, labels)
    )
    rows = {
        "iq": iq,
        "labels": labels,
        "ranks": np.tile(np.arange(10, dtype=np.int64), 2),
        "tokens": np.asarray(ids),
        "hashes": np.asarray(hashes),
    }
    return state, rows, extractor


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
        runner.D7cSupportRunnerError, match="allowlist"
    ):
        runner._validate_manifest(
            drift, registration_state="before"
        )


def test_support_payload_verifies_iq_hash_lineage_and_uniform_k10():
    payload, manifest = _support_payload()
    rows = runner._payload_rows(
        payload, manifest, scenario="leo_clear_weak"
    )
    assert len(rows["iq"]) == 20
    assert set(np.unique(rows["ranks"]).tolist()) == set(range(10))
    bad = dict(payload)
    bad["support_post_channel_iq_sha256"] = (
        payload["support_post_channel_iq_sha256"].copy()
    )
    bad["support_post_channel_iq_sha256"][0] = "0" * 64
    with pytest.raises(
        runner.D7cSupportRunnerError, match="payload drift"
    ):
        runner._payload_rows(
            bad, manifest, scenario="leo_clear_weak"
        )


def test_samplewise_feature_cache_reuses_bitwise_result_without_second_call():
    calls = []

    def unstable(view: np.ndarray) -> np.ndarray:
        calls.append(1)
        return np.asarray(
            [[float(len(calls)), float(view.mean())]], dtype=np.float32
        )

    cached = runner._SamplewiseFeatureCache(unstable)
    view = np.arange(16, dtype=np.float32).reshape(1, 2, 8)
    first = cached(view)
    second = cached(view.copy())
    assert len(calls) == 1
    assert np.array_equal(first, second)
    assert cached.hits == 1
    assert cached.misses == 1
    assert cached.entry_count == 1
    with pytest.raises(
        runner.D7cSupportRunnerError, match="samplewise"
    ):
        cached(np.concatenate([view, view], axis=0))


def test_k1_k5_nested_proof_locks_all_policy_and_writes_create_only(
    tmp_path: Path,
):
    state, rows, extractor = _artifact_and_state()
    output = tmp_path / "out"
    output.mkdir()
    for k in (1, 5):
        proof = runner._nested_proof(
            state,
            rows,
            extractor,
            scenario="leo_clear_weak",
            registration_state="before",
            k=k,
            output=output,
        )
        assert proof["support_count_per_class"] == k
        assert proof["operator_policy_locked"]
        assert proof["calibrations_locked"]
        assert proof["rivals_bitwise_locked"]
        assert proof["beta_bitwise_locked"]
        assert proof["only_prototypes_rebuilt"]
        assert not proof["formal_lower_k_package_opened"]
    metadata = json.loads(
        (output / "state_leo_clear_weak_before_k1.json").read_text(
            encoding="utf-8"
        )
    )
    assert metadata["strategy_locked_k"] == 10
    with pytest.raises(FileExistsError):
        runner._write_state_new(
            output,
            stem="state_leo_clear_weak_before_k1",
            state=state,
        )
