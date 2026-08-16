from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "code") not in sys.path:
    sys.path.insert(0, str(ROOT / "code"))

from cvsrffi import stage2_d92_ccoc_hard9_k1 as matrix  # noqa: E402


CONFIG = ROOT / "configs" / "stage2_d92_ccoc_hard9_k1_v2.json"
G0_OUTER = "rx_7_7__seed_713106__k_10__new_5"


def test_hard9_k1_selection_is_ccoc_only_disjoint_from_g0() -> None:
    assert matrix.ARM_ID == "E0_FULL_CROSS_CLASS_OFFBLOCK_CONSENSUS"
    assert matrix.CANDIDATE_ID == "d92_e0_full_cross_class_offblock_consensus"
    assert matrix.REGISTERED_MODE == "ccoc_full"
    assert matrix.CLAIM_SCOPE == "DEVELOPMENT_ONLY_DISJOINT_FROM_G0_HARD_SCREEN"
    assert matrix.SCENES == (
        "leo_clear_weak",
        "leo_low_elev_weak",
        "leo_rain_weak",
    )
    assert matrix.SHARD_COUNT == 8
    assert G0_OUTER in matrix.EXCLUDED_OUTER_KEYS
    assert all(row["outer_key"] != G0_OUTER for row in matrix.HARD9_K1_ROWS)
    assert sum(row["role"] == "performance" for row in matrix.HARD9_K1_ROWS) == 9
    assert matrix.HARD9_K1_ROWS[-1] == {
        "outer_key": matrix.LIVENESS_OUTER_KEY,
        "role": "liveness",
        "hard_score": None,
    }
    assert re.fullmatch(r"[0-9a-f]{64}", matrix.canonical_selection_sha256())
    assert matrix.canonical_selection_sha256() == matrix.CANONICAL_SELECTION_SHA256


def test_manifest_binds_all_sealed_inputs_and_exact_hard9_k1_jobs() -> None:
    manifest = matrix.build_hard9_k1_manifest(
        CONFIG,
        require_package_files=False,
    )

    assert manifest["schema"] == "cvs.phase2.d92_ccoc_hard9_k1.matrix.v1"
    assert manifest["job_count"] == 10
    assert manifest["outer_count"] == 10
    assert manifest["performance_outer_count"] == 9
    assert manifest["liveness_outer_count"] == 1
    assert manifest["scene_count"] == 3
    assert manifest["scene_arm_count"] == 30
    assert manifest["shard_count"] == 8
    assert manifest["selection_sha256"] == matrix.CANONICAL_SELECTION_SHA256
    assert manifest["candidate_ids"] == {
        matrix.ARM_ID: matrix.CANDIDATE_ID,
    }

    for job in manifest["jobs"]:
        assert job["arm_id"] == matrix.ARM_ID
        assert job["candidate"] == matrix.CANDIDATE_ID
        assert job["scenarios"] == list(matrix.SCENES)
        assert job["method_lock_sha256"] == manifest["method_lock_sha256"]
        assert job["output_root"].endswith(
            "/".join(("jobs", job["outer_key"], matrix.ARM_ID))
        )
        assert set(job["packages"]) == {
            "before_enrollment",
            "before_apply",
            "after_enrollment",
            "after_apply",
        }
        for package in job["packages"].values():
            assert set(package) == {
                "package_root",
                "detached_seal_path",
                "expected_seal_sha256",
            }
            assert package["package_root"]
            assert package["detached_seal_path"]
            assert re.fullmatch(
                r"[0-9a-f]{64}",
                str(package["expected_seal_sha256"]),
            )
        assert re.fullmatch(r"[0-9a-f]{64}", job["truth_sidecar_sha256"])

    checked = matrix.validate_hard9_k1_manifest(
        manifest,
        expected_method_lock_sha256=manifest["method_lock_sha256"],
        require_package_hashes=False,
    )
    assert checked["job_count"] == 10


def test_method_lock_freezes_ccoc_fit_query_and_absolute_peak_contract() -> None:
    lock = json.loads(CONFIG.read_text(encoding="utf-8"))

    assert lock["schema"] == "cvs.phase2.d92_ccoc_hard9_k1.method_lock.v1"
    assert lock["selection_sha256"] == matrix.CANONICAL_SELECTION_SHA256
    assert lock["fit_gate"] == {
        "k_gt_2_total": 2,
        "k_gt_2_actual": 1,
        "postprocess_fit": 0,
        "k1_alias": "K1_K2_EXACT_D92_FULL_ALIAS",
        "k1_total": 3,
        "k1_actual": 3,
    }
    assert lock["query_contract"] == {
        "decision": "per_sample_all_registered_classes",
        "truth_access": False,
        "fit_access": False,
        "update_access": False,
        "selection_access": False,
        "role_oracle_access": False,
        "class_quota_access": False,
        "global_reassignment": False,
    }
    assert lock["resource_gate"]["candidate_peak_hard_max_bytes"] == 1_048_576
    assert lock["resource_gate"]["candidate_peak_target_max_bytes"] == 524_288
    assert matrix.validate_method_lock(lock)["claim_scope"] == matrix.CLAIM_SCOPE

    with pytest.raises(ValueError, match="method lock"):
        matrix.validate_method_lock(
            {
                **lock,
                "fit_gate": {**lock["fit_gate"], "k_gt_2_actual": 2},
            }
        )


def test_manifest_rejects_g0_or_single_job_drift() -> None:
    manifest = matrix.build_hard9_k1_manifest(
        CONFIG,
        require_package_files=False,
    )
    jobs = list(manifest["jobs"])
    jobs[0] = {**jobs[0], "outer_key": G0_OUTER}

    with pytest.raises(ValueError):
        matrix.validate_hard9_k1_manifest(
            {**manifest, "jobs": jobs},
            expected_method_lock_sha256=manifest["method_lock_sha256"],
            require_package_hashes=False,
        )
    with pytest.raises(ValueError):
        matrix.validate_hard9_k1_manifest(
            {**manifest, "job_count": 11},
            expected_method_lock_sha256=manifest["method_lock_sha256"],
            require_package_hashes=False,
        )


def test_manifest_requires_non_placeholder_package_and_truth_hashes() -> None:
    manifest = matrix.build_hard9_k1_manifest(
        CONFIG,
        require_package_files=False,
    )

    with pytest.raises(ValueError, match="hash"):
        matrix.validate_hard9_k1_manifest(
            manifest,
            expected_method_lock_sha256=manifest["method_lock_sha256"],
            require_package_hashes=True,
        )


def test_manifest_binds_one_sealed_e0_fit_audit_with_per_scene_resources() -> None:
    manifest = matrix.build_hard9_k1_manifest(
        CONFIG,
        require_package_files=False,
    )
    resource = manifest["jobs"][0]["e0_resource"]

    assert set(resource) == {"fit_audit", "scenes"}
    assert set(resource["fit_audit"]) == {"path", "sha256"}
    assert re.fullmatch(r"[0-9a-f]{64}", resource["fit_audit"]["sha256"])
    assert set(resource["scenes"]) == set(matrix.SCENES)
    assert resource["scenes"]["leo_clear_weak"]["registration_wall_time_ns"] != (
        resource["scenes"]["leo_low_elev_weak"]["registration_wall_time_ns"]
    )


def test_manifest_build_uses_preregistered_truth_hashes_without_truth_files(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Reopening a truth sidecar during build must make this test fail."""

    def sealed_packages(
        source_job_root: object,
        *,
        require_files: bool,
    ) -> dict[str, dict[str, str]]:
        assert require_files is True
        return {
            name: {
                "package_root": str(source_job_root.joinpath(*package_parts)),
                "detached_seal_path": str(source_job_root.joinpath(*seal_parts)),
                "expected_seal_sha256": "1" * 64,
            }
            for name, (package_parts, seal_parts) in matrix.PACKAGE_LAYOUT.items()
        }

    monkeypatch.setattr(matrix, "_package_entries", sealed_packages)

    # The configured source paths are N607-only and deliberately do not exist
    # on this host. Building with require_package_files=True therefore proves
    # truth identity came from pre-registered metadata rather than a truth file.
    manifest = matrix.build_hard9_k1_manifest(
        CONFIG,
        require_package_files=True,
    )

    expected = {
        "rx_7_7__seed_713104__k_5__new_20": (
            "0ea2f8471e3632545cda52f3e0879fc276237f263885ba8a14d74b45b4b84237"
        ),
        "rx_20_1__seed_713106__k_1__new_20": (
            "b6fc53dc3a02b0867084a1146e4f23fc40ca543b726da3cb54db587f59ec621d"
        ),
    }
    by_outer = {
        str(job["outer_key"]): str(job["truth_sidecar_sha256"])
        for job in manifest["jobs"]
    }
    assert by_outer["rx_7_7__seed_713104__k_5__new_20"] == expected[
        "rx_7_7__seed_713104__k_5__new_20"
    ]
    assert by_outer["rx_20_1__seed_713106__k_1__new_20"] == expected[
        "rx_20_1__seed_713106__k_1__new_20"
    ]
    assert len(by_outer) == 10
    assert all(value != "0" * 64 for value in by_outer.values())
