from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from cvsrffi.stage2_m23_overlay_cache import (
    M23OverlayCacheError,
    load_m23_overlay_cache,
    publish_m23_overlay_cache,
)


SCENARIOS = ("leo_clear_weak", "leo_low_elev_weak", "leo_rain_weak")
OLD = tuple(f"cls_{index:032x}" for index in range(6))
NEW = tuple(f"cls_{index + 100:032x}" for index in range(5))


def _unit(rng: np.random.Generator, rows: int, dim: int) -> np.ndarray:
    value = rng.normal(size=(rows, dim))
    return (value / np.linalg.norm(value, axis=1, keepdims=True)).astype(np.float32)


def _payloads() -> dict[str, dict[str, np.ndarray]]:
    rng = np.random.default_rng(101)
    result = {}
    for scenario_index, scenario in enumerate(SCENARIOS):
        old = np.concatenate([_unit(rng, 12, 160), _unit(rng, 12, 96), _unit(rng, 12, 10)], axis=1)
        new = np.concatenate([_unit(rng, 10, 160), _unit(rng, 10, 96), _unit(rng, 10, 10)], axis=1)
        query = np.concatenate([_unit(rng, 9, 160), _unit(rng, 9, 96), _unit(rng, 9, 10)], axis=1)
        result[scenario] = {
            "old_support_blocks": old,
            "old_support_labels": np.repeat(np.asarray(OLD), 2),
            "old_support_quality": np.linspace(0.2, 1.0, len(old), dtype=np.float32),
            "new_support_blocks": new,
            "new_support_labels": np.repeat(np.asarray(NEW), 2),
            "new_support_quality": np.linspace(0.3, 1.0, len(new), dtype=np.float32),
            "query_blocks": query,
            "query_tokens": np.asarray(
                [f"qid_{scenario_index:02x}{index:062x}" for index in range(len(query))]
            ),
        }
    return result


def _component() -> dict[str, np.ndarray]:
    rng = np.random.default_rng(102)
    return {
        "core_q": rng.integers(-20, 21, size=(6, 160), dtype=np.int8),
        "core_scale": np.full(6, 0.01, dtype=np.float16),
        "residual_basis_q": rng.integers(-20, 21, size=(6, 3, 160), dtype=np.int8),
        "residual_basis_scale": np.full((6, 3), 0.005, dtype=np.float16),
        "residual_coeff_q": rng.integers(-20, 21, size=(4, 6, 3), dtype=np.int8),
        "residual_coeff_scale": np.full((4, 6), 0.004, dtype=np.float16),
        "domain_registry": np.asarray([f"domain_{index}" for index in range(5)]),
        "residual_domain_registry": np.asarray([f"domain_{index}" for index in range(1, 5)]),
        "class_registry": np.asarray(OLD),
        "center_domain_handle": np.asarray("domain_0"),
    }


def test_overlay_cache_round_trip_keeps_compact_blocks_and_int8_ground_only(tmp_path: Path) -> None:
    payload = tmp_path / "overlay.npz"
    manifest = tmp_path / "overlay.manifest.json"
    published = publish_m23_overlay_cache(
        payload,
        manifest,
        receiver="3-19",
        k_shot=2,
        method_seed=7282101,
        support_seed=7282201,
        query_seed=7282301,
        new_class_draw_seed=7282401,
        phase2_data_status="VALIDATED_ONCE",
        capsule_id="capsule-fixed",
        split_id="split-fixed",
        base_feature_cache_payload_sha256="1" * 64,
        base_feature_cache_manifest_sha256="2" * 64,
        predictor_package_root_sha256="3" * 64,
        predictor_package_seal_sha256="4" * 64,
        phase1_bundle_sha256="5" * 64,
        phase1_component_manifest_sha256="6" * 64,
        old_classes=OLD,
        new_classes=NEW,
        scenario_payloads=_payloads(),
        ground_component=_component(),
    )
    loaded = load_m23_overlay_cache(
        payload,
        manifest,
        expected_payload_sha256=published["payload_sha256"],
        expected_manifest_sha256=published["manifest_sha256"],
    )
    assert loaded["manifest"]["query_truth_present"] is False
    assert loaded["manifest"]["clean_source_samples_present"] is False
    assert loaded["old_classes"] == OLD
    assert loaded["new_classes"] == NEW
    assert loaded["ground_component"]["core_q"].dtype == np.int8
    assert loaded["ground_component"]["core_scale"].dtype == np.float16
    assert loaded["scenario_payloads"][SCENARIOS[0]]["query_blocks"].shape[1] == 266
    assert not payload.stat().st_mode & 0o222
    assert not manifest.stat().st_mode & 0o222


def test_overlay_cache_rejects_query_labels_and_bad_quality(tmp_path: Path) -> None:
    payloads = _payloads()
    payloads[SCENARIOS[0]]["query_labels"] = np.asarray(["forbidden"] * 9)
    with pytest.raises(M23OverlayCacheError, match="schema"):
        publish_m23_overlay_cache(
            tmp_path / "bad.npz",
            tmp_path / "bad.json",
            receiver="3-19",
            k_shot=2,
            method_seed=1,
            support_seed=2,
            query_seed=3,
            new_class_draw_seed=4,
            phase2_data_status="VALIDATED_ONCE",
            capsule_id="c",
            split_id="s",
            base_feature_cache_payload_sha256="1" * 64,
            base_feature_cache_manifest_sha256="2" * 64,
            predictor_package_root_sha256="3" * 64,
            predictor_package_seal_sha256="4" * 64,
            phase1_bundle_sha256="5" * 64,
            phase1_component_manifest_sha256="6" * 64,
            old_classes=OLD,
            new_classes=NEW,
            scenario_payloads=payloads,
            ground_component=_component(),
        )

    payloads = _payloads()
    payloads[SCENARIOS[1]]["old_support_quality"][0] = 0.0
    with pytest.raises(M23OverlayCacheError, match="quality"):
        publish_m23_overlay_cache(
            tmp_path / "bad_quality.npz",
            tmp_path / "bad_quality.json",
            receiver="3-19",
            k_shot=2,
            method_seed=1,
            support_seed=2,
            query_seed=3,
            new_class_draw_seed=4,
            phase2_data_status="VALIDATED_ONCE",
            capsule_id="c",
            split_id="s",
            base_feature_cache_payload_sha256="1" * 64,
            base_feature_cache_manifest_sha256="2" * 64,
            predictor_package_root_sha256="3" * 64,
            predictor_package_seal_sha256="4" * 64,
            phase1_bundle_sha256="5" * 64,
            phase1_component_manifest_sha256="6" * 64,
            old_classes=OLD,
            new_classes=NEW,
            scenario_payloads=payloads,
            ground_component=_component(),
        )
