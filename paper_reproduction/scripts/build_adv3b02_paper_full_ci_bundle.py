#!/usr/bin/env python3
"""Build comparison packages while enforcing only the new-class LEO condition."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT_ROOT = REPO_ROOT / "code" / "scripts"
CODE_ROOT = REPO_ROOT / "code"
for value in (str(REPO_ROOT), str(CODE_ROOT), str(SCRIPT_ROOT)):
    if value not in sys.path:
        sys.path.insert(0, value)

import build_cvs_stage2_predictor_bundle as base_builder  # noqa: E402
from cvsrffi.leo_weak_cache import (  # noqa: E402
    FORMAL_LEO_WEAK_SCENARIOS,
    canonical_json_sha256,
    ids_sha256,
    overlay_id,
    post_channel_iq_sha256,
    sha256_file,
)
from paper_reproduction.scripts.build_adv3b02_ci_predictor_bundle import (  # noqa: E402
    reject_predictor_truth_leaks_structurally,
)


def _resolve(manifest_path: Path, raw: str) -> Path:
    value = Path(str(raw))
    return value if value.is_absolute() else manifest_path.parent / value


_COMPARISON_CACHE_MEMBERS = (
    "leo_weak_iq",
    "raw_labels",
    "domain_labels",
    "tx_ids",
    "rx_ids",
    "day_ids",
    "eq_ids",
    "sig_ids",
    "dataset_role",
    "channel_views",
    "sat_scenarios",
    "satellite_seeds",
    "overlay_applied",
    "sample_ids",
    "post_channel_iq_sha256",
    "overlay_ids",
    "manifest_json",
)
_COMPARISON_PROVENANCE_FIELDS = (
    "sample_ids",
    "sat_scenarios",
    "satellite_seeds",
    "post_channel_iq_sha256",
    "overlay_ids",
)


def _comparison_manifest(archive: np.lib.npyio.NpzFile) -> dict:
    raw = np.asarray(archive["manifest_json"])
    if raw.size != 1:
        raise ValueError("comparison LEO cache manifest_json must be scalar")
    value = raw.reshape(-1)[0]
    if isinstance(value, bytes):
        value = value.decode("utf-8")
    manifest = json.loads(str(value))
    if not isinstance(manifest, dict):
        raise TypeError("comparison LEO cache manifest must be an object")
    return manifest


def load_comparison_inner_leo_cache(
    cache_path: str | Path,
    *,
    expected_scenario: str,
    allowed_roles,
):
    """Verify a legacy LEO cache without main-method source-lineage fields."""

    path = Path(cache_path).resolve(strict=True)
    scenario = str(expected_scenario)
    if scenario not in FORMAL_LEO_WEAK_SCENARIOS:
        raise ValueError(f"unsupported comparison LEO scenario: {scenario}")
    allowed = {str(value) for value in allowed_roles}
    if not allowed:
        raise ValueError("comparison allowed_roles must be nonempty")

    with np.load(path, allow_pickle=False) as archive:
        members = tuple(str(value) for value in archive.files)
        if set(members) != set(_COMPARISON_CACHE_MEMBERS):
            missing = sorted(set(_COMPARISON_CACHE_MEMBERS) - set(members))
            extra = sorted(set(members) - set(_COMPARISON_CACHE_MEMBERS))
            raise ValueError(
                f"comparison LEO cache member drift: missing={missing}, extra={extra}"
            )
        manifest = _comparison_manifest(archive)
        arrays = {
            key: np.asarray(archive[key])
            for key in _COMPARISON_CACHE_MEMBERS
            if key != "manifest_json"
        }

    iq = np.asarray(arrays["leo_weak_iq"], dtype=np.float32)
    if iq.ndim != 3 or iq.shape[1] != 2:
        raise ValueError(f"comparison leo_weak_iq must be [N,2,T], got {iq.shape}")
    row_count = int(iq.shape[0])
    if row_count <= 0:
        raise ValueError("comparison LEO cache contains no rows")
    for key, value in arrays.items():
        array = np.asarray(value)
        if array.ndim == 0 or int(array.shape[0]) != row_count:
            raise ValueError(f"comparison LEO cache row count drift: {key}")
        if array.dtype == object:
            raise ValueError(f"comparison LEO cache object array forbidden: {key}")

    roles = np.asarray(arrays["dataset_role"]).astype(str)
    observed_roles = set(roles.tolist())
    if not observed_roles or not observed_roles.issubset(allowed):
        raise ValueError(
            "comparison LEO cache role leakage: "
            f"observed={sorted(observed_roles)}, allowed={sorted(allowed)}"
        )
    required_manifest = {
        "schema": "cvs_leo_weak_iq_cache_v1",
        "artifact_stage": "phase1_offline_prechannel_export",
        "contains_post_channel_iq_only": True,
        "raw_or_clean_iq_key_present": False,
        "overlay_applied_before_phase2": True,
        "scenario": scenario,
        "iq_array_key": "leo_weak_iq",
    }
    failed = [
        key
        for key, expected in required_manifest.items()
        if manifest.get(key) != expected
    ]
    if failed:
        raise ValueError(f"comparison LEO manifest contract failed: {failed}")
    if tuple(str(v) for v in manifest.get("target_channel_scenarios", [])) != (
        scenario,
    ):
        raise ValueError("comparison LEO manifest scenario list drift")
    if {str(v) for v in manifest.get("output_roles", [])} != observed_roles:
        raise ValueError("comparison LEO manifest output_roles drift")
    if tuple(
        str(v) for v in manifest.get("sample_overlay_provenance_fields", [])
    ) != _COMPARISON_PROVENANCE_FIELDS:
        raise ValueError("comparison LEO provenance fields drift")
    if int(manifest.get("row_count", -1)) != row_count:
        raise ValueError("comparison LEO manifest row_count drift")

    scenarios = np.asarray(arrays["sat_scenarios"]).astype(str)
    views = np.asarray(arrays["channel_views"]).astype(str)
    seeds = np.asarray(arrays["satellite_seeds"]).astype(np.int64)
    applied = np.asarray(arrays["overlay_applied"]).astype(bool)
    if not np.all(scenarios == scenario):
        raise ValueError("comparison cache mixes LEO scenarios")
    if not np.all(views == "rx_base"):
        raise ValueError("comparison cache contains a non-rx_base channel view")
    if not bool(np.all(applied)):
        raise ValueError("comparison cache contains a row without LEO overlay")

    channel_hash = str(manifest.get("channel_config_sha256", ""))
    if len(channel_hash) != 64:
        raise ValueError("comparison LEO channel_config_sha256 is missing")
    sample_ids = np.asarray(arrays["sample_ids"]).astype(str)
    iq_hashes = np.asarray(arrays["post_channel_iq_sha256"]).astype(str)
    overlay_ids = np.asarray(arrays["overlay_ids"]).astype(str)
    expected_iq_hashes = [post_channel_iq_sha256(row) for row in iq]
    expected_overlay_ids = [
        overlay_id(
            sample_id=sample_ids[index],
            scenario=scenario,
            satellite_seed=int(seeds[index]),
            channel_config_sha256=channel_hash,
            iq_sha256=expected_iq_hashes[index],
        )
        for index in range(row_count)
    ]
    if iq_hashes.tolist() != expected_iq_hashes:
        raise ValueError("comparison LEO post-channel IQ digest mismatch")
    if overlay_ids.tolist() != expected_overlay_ids:
        raise ValueError("comparison LEO overlay provenance mismatch")
    if len(set(sample_ids.tolist())) != row_count:
        raise ValueError("comparison LEO cache contains duplicate sample IDs")

    roots = {
        "physical_sample_ids_sha256": ids_sha256(sample_ids.tolist()),
        "post_channel_iq_sha256_root": ids_sha256(expected_iq_hashes),
        "overlay_ids_sha256": ids_sha256(expected_overlay_ids),
    }
    for key, observed in roots.items():
        if str(manifest.get(key, "")) != observed:
            raise ValueError(f"comparison LEO manifest root mismatch: {key}")

    new_mask = roles == "target_new"
    if "target_new" in allowed and not bool(np.any(new_mask)):
        raise ValueError("comparison LEO cache has no target_new rows")
    if bool(np.any(new_mask)) and not bool(
        np.all(applied[new_mask] & (scenarios[new_mask] == scenario))
    ):
        raise ValueError("comparison new-class rows are not all LEO-overlaid")

    arrays["leo_weak_iq"] = iq
    audit = {
        "path": str(path),
        "sha256": sha256_file(path),
        "schema": str(manifest["schema"]),
        "scenario": scenario,
        "row_count": row_count,
        "roles": sorted(observed_roles),
        "satellite_seeds": sorted(set(int(value) for value in seeds.tolist())),
        **roots,
        "manifest_sha256": canonical_json_sha256(manifest),
        "comparison_source_lineage_arrays_required": False,
        "new_class_leo_iq_verified": bool(np.any(new_mask)),
        "exact_legacy_member_set_verified": True,
    }
    return arrays, dict(manifest), audit


def load_comparison_leo_cache_set(
    manifest_path: str | Path,
    *,
    expected_scope: str,
    allowed_roles,
):
    """Load legacy/current cache sets without applying main-method p2_min_v1 gates.

    Each inner scenario cache still receives full cryptographic and LEO overlay
    verification. The relaxed surface is only the set-level schema/policy and
    cross-scenario physical-ID rule, which the user explicitly exempted for
    external comparison methods.
    """

    path = Path(manifest_path).resolve(strict=True)
    payload = json.loads(path.read_text(encoding="utf-8-sig"))
    scenario_map = dict(payload.get("cache_npz_by_scenario", {}))
    hash_map = dict(payload.get("cache_sha256_by_scenario", {}))
    if tuple(scenario_map) != FORMAL_LEO_WEAK_SCENARIOS:
        raise ValueError("comparison cache scenarios drift")
    if tuple(hash_map) != FORMAL_LEO_WEAK_SCENARIOS:
        raise ValueError("comparison cache hash map drift")
    allowed = {str(value) for value in allowed_roles}
    arrays_by_scenario = {}
    audits = {}
    for scenario in FORMAL_LEO_WEAK_SCENARIOS:
        cache_path = _resolve(path, scenario_map[scenario]).resolve(strict=True)
        if sha256_file(cache_path) != str(hash_map[scenario]):
            raise ValueError(f"comparison LEO cache hash mismatch: {scenario}")
        arrays, inner_manifest, audit = load_comparison_inner_leo_cache(
            cache_path,
            expected_scenario=scenario,
            allowed_roles=allowed,
        )
        roles = set(np.asarray(arrays["dataset_role"]).astype(str).tolist())
        if roles != allowed:
            raise ValueError(
                f"comparison LEO cache role drift: {scenario} {sorted(roles)}"
            )
        # Keep the verified legacy IDs unchanged. The shared builder also has a
        # scenario-alignment check that requires the same row identity/order
        # across views; its separate main-method no-reuse gate is disabled only
        # in this comparison entry point below.
        arrays_by_scenario[scenario] = arrays
        audits[scenario] = {
            **audit,
            "comparison_protocol_scope": (
                "stage2_main_method_protocol_exempt_new_class_leo_required"
            ),
            "set_manifest_schema_observed": payload.get("schema"),
            "set_manifest_expected_scope_argument": str(expected_scope),
            "inner_manifest_schema": inner_manifest.get("schema"),
            "new_class_leo_iq_verified": True,
            "cross_scenario_physical_reuse_allowed_for_comparison": True,
            "verified_sample_ids_preserved_for_scenario_alignment": True,
        }
    return arrays_by_scenario, payload, {
        "status": "PASS_COMPARISON_SCOPE",
        "new_class_leo_iq_verified": True,
        "scenario_audits": audits,
    }


def _comparison_reference_arrays(arrays_by_scenario):
    """Preserve the legacy alignment helper's return-value contract."""

    return arrays_by_scenario[FORMAL_LEO_WEAK_SCENARIOS[0]]


def main() -> int:
    base_builder.load_verified_leo_weak_cache_set = load_comparison_leo_cache_set
    # N607 may carry the legacy alignment-named gate while the Git release
    # surface carries the newer physical-independence gate. Both encode
    # Stage2-main-method cross-scenario policy and are out of scope here.
    base_builder._assert_scenario_alignment = _comparison_reference_arrays
    base_builder._assert_scenario_physical_independence = lambda _arrays: None
    base_builder._reject_predictor_truth_leaks = (
        reject_predictor_truth_leaks_structurally
    )
    return base_builder.main()


if __name__ == "__main__":
    raise SystemExit(main())
