from __future__ import annotations

import hashlib
import inspect
import json
from pathlib import Path

import numpy as np
import pytest

import cvsrffi.stage2_ablation_feature_cache as feature_cache_module
from cvsrffi.stage2_ablation_feature_cache import (
    Stage2AblationFeatureCacheError,
    load_feature_cache,
    publish_feature_cache,
)
from cvsrffi.stage2_ablation_row_executor import execute_feature_row
from cvsrffi.stage2_ablation_factory import resolved_stage2_config_hash
from cvsrffi.somph_predictor_bundle import FORMAL_LEO_WEAK_SCENARIOS
from scripts.run_full_ablation_stage2_row import (
    REQUEST_SCHEMA,
    Stage2AblationRowRequestError,
    run_request,
)


def _opaque(prefix: str, value: str) -> str:
    return prefix + hashlib.sha256(value.encode("utf-8")).hexdigest()


def _payload():
    rng = np.random.default_rng(771)
    old_classes = tuple(_opaque("cls_", f"old-{index}") for index in range(6))
    new_classes = tuple(_opaque("cls_", f"new-{index}") for index in range(5))
    old_targets = np.repeat(np.arange(6), 2)
    new_targets = np.repeat(np.arange(5), 2)
    means = rng.normal(size=(11, 288)).astype(np.float32)
    scenarios = {}
    for index, scenario in enumerate(FORMAL_LEO_WEAK_SCENARIOS):
        tokens = np.asarray(
            [
                _opaque("qid_", f"{scenario}-query-{position}")
                for position in range(6)
            ]
        )
        old = (
            means[old_targets]
            + 0.01 * index
            + rng.normal(scale=0.01, size=(12, 288))
        ).astype(np.float32)
        new = (
            means[6 + new_targets]
            + 0.01 * index
            + rng.normal(scale=0.01, size=(10, 288))
        ).astype(np.float32)
        scenarios[scenario] = {
            "old_support_features": old,
            "old_support_labels": np.asarray(old_classes)[old_targets],
            "new_support_features": new,
            "new_support_labels": np.asarray(new_classes)[new_targets],
            "query_features": np.concatenate([old[:3], new[:3]]),
            "query_tokens": tokens,
        }
    labels = scenarios[FORMAL_LEO_WEAK_SCENARIOS[0]][
        "old_support_labels"
    ]
    source = scenarios[FORMAL_LEO_WEAK_SCENARIOS[0]][
        "old_support_features"
    ]
    prototypes = np.stack(
        [source[labels == value].mean(axis=0) for value in old_classes]
    )
    basis, _ = np.linalg.qr(rng.normal(size=(160, 3)))
    return {
        "receiver": "20-1",
        "method_seed": 840001,
        "support_seed": 840002,
        "query_seed": 840003,
        "new_class_draw_seed": 850001,
        "package_root_sha256": "a" * 64,
        "package_seal_sha256": "b" * 64,
        "phase1_bundle_sha256": "c" * 64,
        "phase1_prototype_sha256": "f" * 64,
        "stage_scope": "stage2c",
        "phase2_data_status": "VALIDATED_ONCE",
        "capsule_id": "capsule-test",
        "split_id": "split-test",
        "old_classes": old_classes,
        "new_classes": new_classes,
        "scenario_payloads": scenarios,
        "deployment_prototypes": prototypes,
        "ground_basis": basis,
        "ground_spectral_weights": np.asarray([0.5, 0.3, 0.2]),
        "ground_audit": {
            "d81_basis_sha256": "d" * 64,
            "d81_spectral_weight_sha256": "e" * 64,
            "d81_participation_ratio_effective_rank": 2.6,
            "d81_retained_rank": 3,
            "d81_rank_policy": "ceil_participation_ratio_effective_rank",
            "ground_component_input_count": 84,
            "ground_statistic_semantics": (
                "class_centered_cross_domain_centroid_drift_eigenspectrum"
            ),
        },
    }


def test_feature_cache_public_api_has_no_query_truth_surface() -> None:
    for function in (publish_feature_cache, load_feature_cache):
        assert not any(
            name in inspect.signature(function).parameters
            for name in (
                "query_truth",
                "truth",
                "truth_sidecar",
                "query_labels",
                "query_roles",
                "dataset",
            )
        )


def test_feature_cache_is_immutable_reusable_and_row_executor_compatible(
    tmp_path: Path,
) -> None:
    payload_path = tmp_path / "features.npz"
    manifest_path = tmp_path / "features.manifest.json"
    published = publish_feature_cache(
        payload_path,
        manifest_path,
        **_payload(),
    )
    loaded = load_feature_cache(
        payload_path,
        manifest_path,
        expected_payload_sha256=published["payload_sha256"],
        expected_manifest_sha256=published["manifest_sha256"],
    )
    manifest_text = manifest_path.read_text(encoding="utf-8").lower()
    assert '"query_truth_present":false' in manifest_text
    assert '"query_role_present":false' in manifest_text
    assert loaded["manifest"]["protocol_schema"] == "p2_min_v1"
    assert loaded["manifest"]["k_shot"] == 2

    receipt = execute_feature_row(
        ablation_id="P2-BASE-COSINE",
        row_id="row_" + "f" * 64,
        receiver="20-1",
        candidate_lock_sha256="1" * 64,
        package_root_sha256="a" * 64,
        package_seal_sha256="b" * 64,
        input_identity={
            "stage_scope": "stage2c",
            "k_shot": int(loaded["manifest"]["k_shot"]),
            "new_class_count": len(loaded["new_classes"]),
            "method_seed": int(loaded["manifest"]["method_seed"]),
            "support_seed": int(loaded["manifest"]["support_seed"]),
            "query_seed": int(loaded["manifest"]["query_seed"]),
            "new_class_draw_seed": int(
                loaded["manifest"]["new_class_draw_seed"]
            ),
            "phase2_data_status": str(
                loaded["manifest"]["phase2_data_status"]
            ),
            "capsule_id": str(loaded["manifest"]["capsule_id"]),
            "split_id": str(loaded["manifest"]["split_id"]),
            "phase1_bundle_sha256": str(
                loaded["manifest"]["phase1_bundle_sha256"]
            ),
            "phase1_prototype_sha256": str(
                loaded["manifest"]["phase1_prototype_sha256"]
            ),
            "feature_cache_payload_sha256": str(
                published["payload_sha256"]
            ),
            "feature_cache_manifest_sha256": str(
                published["manifest_sha256"]
            ),
        },
        output_root=tmp_path / "row",
        seed=840001,
        device="cpu",
        feature_cache_bytes=payload_path.stat().st_size,
        deployment_state_bytes=loaded["manifest"][
            "deployment_state_bytes"
        ],
        peak_rss_bytes=1,
        peak_vram_bytes=0,
        **{
            key: loaded[key]
            for key in (
                "old_classes",
                "new_classes",
                "scenario_payloads",
                "deployment_prototypes_by_scenario",
                "ground_basis",
                "ground_spectral_weights",
                "ground_audit",
            )
        },
    )
    assert receipt["status"] == "PREDICTIONS_COMPLETE_TRUTH_UNOPENED"
    assert receipt["prediction"]["row_count"] == 18

    with pytest.raises(
        Stage2AblationFeatureCacheError,
        match="overwrite",
    ):
        publish_feature_cache(
            payload_path,
            manifest_path,
            **_payload(),
        )


def test_repair_legacy_stage2b_manifest_adds_only_builder_protocol_field(
    tmp_path: Path,
) -> None:
    payload = _payload()
    payload["stage_scope"] = "stage2b"
    payload["new_classes"] = ()
    for scenario in payload["scenario_payloads"].values():
        scenario.pop("new_support_features")
        scenario.pop("new_support_labels")
    payload["capsule_id"] = (
        "d18-reuse-validated-once-rx20-1-seed713101-m840001-k2-new20"
    )
    payload["split_id"] = (
        "p2_min_v1-rx20-1-m840001-s840002-q840003-d850001-k2-new20"
    )
    payload_path = tmp_path / "features.npz"
    current_manifest_path = tmp_path / "current.manifest.json"
    published = publish_feature_cache(
        payload_path,
        current_manifest_path,
        **payload,
    )
    legacy = json.loads(current_manifest_path.read_text(encoding="utf-8"))
    legacy.pop("protocol_schema")
    legacy_bytes = json.dumps(
        legacy,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    legacy_path = tmp_path / "legacy.manifest.json"
    legacy_path.write_bytes(legacy_bytes + b"\n")
    legacy_path.chmod(0o444)
    legacy_sha256 = hashlib.sha256(legacy_bytes).hexdigest()
    repaired_path = tmp_path / "features.manifest.json"

    loaded_legacy = load_feature_cache(
        payload_path,
        legacy_path,
        expected_payload_sha256=published["payload_sha256"],
        expected_manifest_sha256=legacy_sha256,
    )
    assert "protocol_schema" not in loaded_legacy["manifest"]

    result = feature_cache_module.repair_legacy_stage2b_manifest_protocol_schema(
        legacy_path,
        repaired_path,
        expected_source_manifest_sha256=legacy_sha256,
    )

    repaired = json.loads(repaired_path.read_text(encoding="utf-8"))
    assert repaired == {**legacy, "protocol_schema": "p2_min_v1"}
    assert result["source_manifest_sha256"] == legacy_sha256
    assert result["manifest_sha256"] == hashlib.sha256(
        repaired_path.read_bytes().rstrip(b"\n")
    ).hexdigest()


def test_feature_cache_rejects_truth_bearing_ground_audit(
    tmp_path: Path,
) -> None:
    payload = _payload()
    payload["ground_audit"]["query_truth"] = ["forbidden"]
    with pytest.raises(
        Stage2AblationFeatureCacheError,
        match="forbidden truth-side",
    ):
        publish_feature_cache(
            tmp_path / "features.npz",
            tmp_path / "features.manifest.json",
            **payload,
        )


def test_row_request_binds_cache_and_effective_config(tmp_path: Path) -> None:
    payload_path = tmp_path / "features.npz"
    manifest_path = tmp_path / "features.manifest.json"
    published = publish_feature_cache(
        payload_path,
        manifest_path,
        **_payload(),
    )
    request = {
        "schema": REQUEST_SCHEMA,
        "ablation_id": "P2-BASE-EUCLIDEAN",
        "row_id": "row_" + "9" * 64,
        "receiver": "20-1",
        "candidate_lock_sha256": "8" * 64,
        "stage_scope": "stage2c",
        "k_shot": 2,
        "new_class_count": 5,
        "support_seed": 840002,
        "query_seed": 840003,
        "new_class_draw_seed": 850001,
        "phase2_data_status": "VALIDATED_ONCE",
        "capsule_id": "capsule-test",
        "split_id": "split-test",
        "phase1_bundle_sha256": "c" * 64,
        "phase1_prototype_sha256": "f" * 64,
        "effective_config_hash": resolved_stage2_config_hash(
            "P2-BASE-EUCLIDEAN"
        ),
        "feature_cache_payload": str(payload_path),
        "feature_cache_payload_sha256": published["payload_sha256"],
        "feature_cache_manifest": str(manifest_path),
        "feature_cache_manifest_sha256": published["manifest_sha256"],
        "output_root": str(tmp_path / "row_request_output"),
        "seed": 840001,
        "device": "cpu",
        "shared_view_count": 1,
    }
    request_path = tmp_path / "row_request.json"
    request_path.write_text(
        json.dumps(request, sort_keys=True),
        encoding="utf-8",
    )
    receipt = run_request(request_path)
    assert receipt["ablation_id"] == "P2-BASE-EUCLIDEAN"
    assert receipt["query_truth_opened"] is False

    drift = dict(request)
    drift["effective_config_hash"] = "7" * 64
    drift["output_root"] = str(tmp_path / "drift_output")
    drift_path = tmp_path / "drift_request.json"
    drift_path.write_text(
        json.dumps(drift, sort_keys=True),
        encoding="utf-8",
    )
    with pytest.raises(
        Stage2AblationRowRequestError,
        match="config hash drift",
    ):
        run_request(drift_path)
