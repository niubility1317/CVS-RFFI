from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np


FORMAL_LEO_WEAK_SCENARIOS = (
    "leo_clear_weak",
    "leo_low_elev_weak",
    "leo_rain_weak",
)
PHASE2_SAMPLE_VIEW_POLICY = "leo_weak_only_no_clean_access"
LEO_WEAK_CACHE_SCHEMA = "cvs_leo_weak_iq_cache_v1"
LEO_WEAK_CACHE_SET_SCHEMA = "cvs_leo_weak_iq_cache_set_v1"
LEO_WEAK_CACHE_STAGE = "phase1_offline_prechannel_export"

_REQUIRED_ARRAY_KEYS = (
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

_FORBIDDEN_EXACT_MEMBERS = {
    "raw_iq",
    "features",
    "tx_logits",
    "logits",
    "prototypes",
    "fft_logmag_features",
    "rf_stat_features",
    "fft_rf_features",
}
_OPTIONAL_OFFLINE_SPLIT_MEMBERS = ("split_partition", "split_rank")


def canonical_json_sha256(value: Any) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def ids_sha256(values: Sequence[str]) -> str:
    payload = "\n".join(str(value) for value in values).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def post_channel_iq_sha256(row: np.ndarray) -> str:
    iq = np.ascontiguousarray(np.asarray(row, dtype="<f4"))
    if iq.ndim != 2 or iq.shape[0] != 2:
        raise ValueError(f"LEO IQ row must have shape [2,T], got {iq.shape}")
    return hashlib.sha256(iq.tobytes(order="C")).hexdigest()


def physical_sample_id_from_values(
    *,
    role: str,
    tx_id: str,
    rx_id: str,
    day_id: str,
    eq_id: str,
    sig_id: str,
) -> str:
    return "|".join(
        (str(role), str(tx_id), str(rx_id), str(day_id), str(eq_id), str(sig_id))
    )


def physical_sample_id(arrays: Mapping[str, np.ndarray], index: int) -> str:
    return physical_sample_id_from_values(
        role=str(arrays["dataset_role"][index]),
        tx_id=str(arrays["tx_ids"][index]),
        rx_id=str(arrays["rx_ids"][index]),
        day_id=str(arrays["day_ids"][index]),
        eq_id=str(arrays["eq_ids"][index]),
        sig_id=str(arrays["sig_ids"][index]),
    )


def overlay_id(
    *,
    sample_id: str,
    scenario: str,
    satellite_seed: int,
    channel_config_sha256: str,
    iq_sha256: str,
) -> str:
    return canonical_json_sha256(
        {
            "channel_config_sha256": str(channel_config_sha256),
            "iq_sha256": str(iq_sha256),
            "phase2_sample_view_policy": PHASE2_SAMPLE_VIEW_POLICY,
            "sample_id": str(sample_id),
            "satellite_seed": int(satellite_seed),
            "scenario": str(scenario),
        }
    )


def _is_forbidden_member(name: str) -> bool:
    normalized = str(name).strip().lower()
    return (
        normalized in _FORBIDDEN_EXACT_MEMBERS
        or normalized.startswith("clean")
        or "clean_iq" in normalized
        or "clean_feature" in normalized
        or "clean_logit" in normalized
        or "clean_proto" in normalized
    )


def _manifest_from_archive(archive: np.lib.npyio.NpzFile) -> dict[str, Any]:
    raw = np.asarray(archive["manifest_json"])
    if raw.size != 1:
        raise ValueError("LEO cache manifest_json must be a scalar")
    value = raw.reshape(-1)[0]
    if isinstance(value, bytes):
        value = value.decode("utf-8")
    manifest = json.loads(str(value))
    if not isinstance(manifest, dict):
        raise TypeError("LEO cache manifest_json must decode to an object")
    return manifest


def _require_manifest_contract(
    manifest: Mapping[str, Any],
    *,
    expected_scenario: str,
    observed_roles: set[str],
) -> None:
    required = {
        "schema": LEO_WEAK_CACHE_SCHEMA,
        "artifact_stage": LEO_WEAK_CACHE_STAGE,
        "phase2_sample_view_policy": PHASE2_SAMPLE_VIEW_POLICY,
        "clean_sample_access": False,
        "clean_derived_signal_access": False,
        "contains_post_channel_iq_only": True,
        "contains_clean_rows": False,
        "target_channel_view": "leo_weak_only",
        "scenario": str(expected_scenario),
        "iq_array_key": "leo_weak_iq",
        "raw_or_clean_iq_key_present": False,
        "overlay_applied_before_phase2": True,
        "star_ground_channel_impl": "simplified_leo_residual",
        "channel_model": "leo_residual",
    }
    failed = [
        key
        for key, expected in required.items()
        if manifest.get(key) != expected
    ]
    if failed:
        raise ValueError(f"LEO cache manifest contract failed: {failed}")
    scenarios = tuple(str(value) for value in manifest.get("target_channel_scenarios", []))
    if scenarios != (str(expected_scenario),):
        raise ValueError("LEO cache manifest must expose exactly its one scenario")
    output_roles = {str(value) for value in manifest.get("output_roles", [])}
    if output_roles != observed_roles:
        raise ValueError(
            f"LEO cache output_roles drift: {sorted(output_roles)} != {sorted(observed_roles)}"
        )
    provenance_fields = tuple(
        str(value) for value in manifest.get("sample_overlay_provenance_fields", [])
    )
    required_fields = (
        "sample_ids",
        "sat_scenarios",
        "satellite_seeds",
        "post_channel_iq_sha256",
        "overlay_ids",
    )
    if provenance_fields != required_fields:
        raise ValueError("LEO cache sample overlay provenance field order drift")
    channel_hash = str(manifest.get("channel_config_sha256", ""))
    if len(channel_hash) != 64:
        raise ValueError("LEO cache channel_config_sha256 is missing")
    builder_hash = str(manifest.get("builder_sha256", ""))
    if len(builder_hash) != 64:
        raise ValueError("LEO cache builder_sha256 is missing")


def load_verified_leo_weak_cache(
    path: str | Path,
    *,
    expected_scenario: str,
    allowed_roles: Iterable[str],
) -> tuple[dict[str, np.ndarray], dict[str, Any], dict[str, Any]]:
    """Load one sealed cache after member-list and sample-level verification.

    The ZIP member list is inspected before any IQ array is materialized.  A
    cache that exposes raw/clean IQ or feature/logit/prototype members fails
    closed even if a downstream row mask would otherwise ignore those arrays.
    """

    cache_path = Path(path)
    if not cache_path.is_file():
        raise FileNotFoundError(f"LEO cache is missing: {cache_path}")
    scenario = str(expected_scenario)
    if scenario not in FORMAL_LEO_WEAK_SCENARIOS:
        raise ValueError(f"unsupported formal LEO scenario: {scenario}")
    allowed = {str(value) for value in allowed_roles}
    if not allowed:
        raise ValueError("allowed_roles must be nonempty")

    with np.load(cache_path, allow_pickle=False) as archive:
        members = tuple(str(value) for value in archive.files)
        forbidden = sorted(name for name in members if _is_forbidden_member(name))
        if forbidden:
            raise ValueError(
                "LEO cache exposes forbidden raw/clean/derived members before array read: "
                f"{forbidden}"
            )
        missing = [key for key in _REQUIRED_ARRAY_KEYS if key not in members]
        if missing:
            raise ValueError(f"LEO cache is missing required members: {missing}")
        manifest = _manifest_from_archive(archive)
        arrays = {
            key: np.asarray(archive[key])
            for key in _REQUIRED_ARRAY_KEYS
            if key != "manifest_json"
        }
        optional_present = [
            key for key in _OPTIONAL_OFFLINE_SPLIT_MEMBERS if key in members
        ]
        if optional_present and len(optional_present) != len(
            _OPTIONAL_OFFLINE_SPLIT_MEMBERS
        ):
            raise ValueError(
                "LEO cache offline split members must be present as an exact pair"
            )
        for key in optional_present:
            arrays[key] = np.asarray(archive[key])

    iq = np.asarray(arrays["leo_weak_iq"], dtype=np.float32)
    if iq.ndim != 3 or iq.shape[1] != 2:
        raise ValueError(f"leo_weak_iq must have shape [N,2,T], got {iq.shape}")
    row_count = int(iq.shape[0])
    if row_count <= 0:
        raise ValueError("LEO cache contains no rows")
    for key, value in arrays.items():
        if int(np.asarray(value).shape[0]) != row_count:
            raise ValueError(f"LEO cache row count drift for {key}")
        if np.asarray(value).dtype == object:
            raise ValueError(f"LEO cache object arrays are forbidden: {key}")
    if "split_partition" in arrays:
        partitions = np.asarray(arrays["split_partition"]).astype(str)
        ranks = np.asarray(arrays["split_rank"]).astype(np.int64)
        if set(partitions.tolist()) != {"support_pool", "query"}:
            raise ValueError("offline split partition values drift")
        if np.any(ranks < 0) or manifest.get(
            "offline_split_partition_policy"
        ) != "legacy_seeded_nested_exact":
            raise ValueError("offline split rank/policy drift")
    roles = np.asarray(arrays["dataset_role"]).astype(str)
    observed_roles = set(roles.tolist())
    if not observed_roles or not observed_roles.issubset(allowed):
        raise ValueError(
            f"LEO cache role leakage: observed={sorted(observed_roles)}, allowed={sorted(allowed)}"
        )
    _require_manifest_contract(
        manifest,
        expected_scenario=scenario,
        observed_roles=observed_roles,
    )
    if int(manifest.get("row_count", -1)) != row_count:
        raise ValueError("LEO cache manifest row_count drift")

    scenarios = np.asarray(arrays["sat_scenarios"]).astype(str)
    views = np.asarray(arrays["channel_views"]).astype(str)
    seeds = np.asarray(arrays["satellite_seeds"]).astype(np.int64)
    applied = np.asarray(arrays["overlay_applied"]).astype(bool)
    if not np.all(scenarios == scenario):
        raise ValueError("LEO cache contains a row from another scenario")
    if not np.all(views == "rx_base"):
        raise ValueError("LEO cache channel_views must be post-channel rx_base")
    if not bool(np.all(applied)):
        raise ValueError("LEO cache contains a row without overlay_applied=true")

    channel_config_hash = str(manifest["channel_config_sha256"])
    sample_ids = np.asarray(arrays["sample_ids"]).astype(str)
    iq_hashes = np.asarray(arrays["post_channel_iq_sha256"]).astype(str)
    overlay_ids = np.asarray(arrays["overlay_ids"]).astype(str)
    expected_sample_ids: list[str] = []
    expected_iq_hashes: list[str] = []
    expected_overlay_ids: list[str] = []
    for index in range(row_count):
        sid = physical_sample_id(arrays, index)
        iq_hash = post_channel_iq_sha256(iq[index])
        oid = overlay_id(
            sample_id=sid,
            scenario=scenario,
            satellite_seed=int(seeds[index]),
            channel_config_sha256=channel_config_hash,
            iq_sha256=iq_hash,
        )
        expected_sample_ids.append(sid)
        expected_iq_hashes.append(iq_hash)
        expected_overlay_ids.append(oid)
    if sample_ids.tolist() != expected_sample_ids:
        raise ValueError("LEO cache sample_ids do not match physical metadata")
    if iq_hashes.tolist() != expected_iq_hashes:
        raise ValueError("LEO cache post-channel IQ digest mismatch")
    if overlay_ids.tolist() != expected_overlay_ids:
        raise ValueError("LEO cache overlay_ids do not match sample provenance")
    if len(set(expected_sample_ids)) != row_count:
        raise ValueError("LEO cache contains duplicate physical sample IDs")

    arrays["leo_weak_iq"] = iq
    audit = {
        "path": str(cache_path),
        "sha256": sha256_file(cache_path),
        "schema": LEO_WEAK_CACHE_SCHEMA,
        "scenario": scenario,
        "row_count": row_count,
        "roles": sorted(observed_roles),
        "satellite_seeds": sorted(set(int(value) for value in seeds.tolist())),
        "physical_sample_ids_sha256": ids_sha256(expected_sample_ids),
        "post_channel_iq_sha256_root": ids_sha256(expected_iq_hashes),
        "overlay_ids_sha256": ids_sha256(expected_overlay_ids),
        "manifest_sha256": canonical_json_sha256(manifest),
        "forbidden_members_checked_before_iq_read": True,
        "clean_sample_access": False,
        "phase2_sample_view_policy": PHASE2_SAMPLE_VIEW_POLICY,
    }
    return arrays, dict(manifest), audit


def _resolve_from_manifest(manifest_path: Path, raw_path: str) -> Path:
    candidate = Path(str(raw_path))
    return candidate if candidate.is_absolute() else manifest_path.parent / candidate


def load_verified_leo_weak_cache_set(
    manifest_path: str | Path,
    *,
    expected_scope: str,
    allowed_roles: Iterable[str],
) -> tuple[
    dict[str, dict[str, np.ndarray]],
    dict[str, Any],
    dict[str, Any],
]:
    path = Path(manifest_path)
    payload = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(payload, dict):
        raise TypeError("LEO cache-set manifest must be a JSON object")
    required = {
        "schema": LEO_WEAK_CACHE_SET_SCHEMA,
        "artifact_stage": LEO_WEAK_CACHE_STAGE,
        "phase2_sample_view_policy": PHASE2_SAMPLE_VIEW_POLICY,
        "clean_sample_access": False,
        "clean_derived_signal_access": False,
        "target_channel_view": "leo_weak_only",
        "cache_scope": str(expected_scope),
    }
    failed = [key for key, expected in required.items() if payload.get(key) != expected]
    if failed:
        raise ValueError(f"LEO cache-set manifest contract failed: {failed}")
    scenario_map = dict(payload.get("cache_npz_by_scenario", {}))
    hash_map = dict(payload.get("cache_sha256_by_scenario", {}))
    if tuple(scenario_map) != FORMAL_LEO_WEAK_SCENARIOS:
        raise ValueError("LEO cache-set scenario mapping must use the formal ordered tuple")
    if tuple(hash_map) != FORMAL_LEO_WEAK_SCENARIOS:
        raise ValueError("LEO cache-set SHA mapping must use the formal ordered tuple")

    arrays_by_scenario: dict[str, dict[str, np.ndarray]] = {}
    cache_audits: dict[str, Any] = {}
    reference_ids: list[str] | None = None
    reference_roles: list[str] | None = None
    allowed = {str(value) for value in allowed_roles}
    for scenario in FORMAL_LEO_WEAK_SCENARIOS:
        cache_path = _resolve_from_manifest(path, str(scenario_map[scenario])).resolve()
        expected_hash = str(hash_map[scenario])
        if len(expected_hash) != 64 or sha256_file(cache_path) != expected_hash:
            raise ValueError(f"LEO cache-set file hash mismatch for {scenario}")
        arrays, _manifest, audit = load_verified_leo_weak_cache(
            cache_path,
            expected_scenario=scenario,
            allowed_roles=allowed,
        )
        current_ids = np.asarray(arrays["sample_ids"]).astype(str).tolist()
        current_roles = np.asarray(arrays["dataset_role"]).astype(str).tolist()
        if reference_ids is None:
            reference_ids = current_ids
            reference_roles = current_roles
        elif current_ids != reference_ids or current_roles != reference_roles:
            raise ValueError(
                "LEO cache-set physical sample ordering drifts across scenarios"
            )
        arrays_by_scenario[scenario] = arrays
        cache_audits[scenario] = audit

    observed_role_set = set(reference_roles or [])
    if observed_role_set != allowed:
        raise ValueError(
            "LEO cache-set must contain the exact registered role set: "
            f"observed={sorted(observed_role_set)}, expected={sorted(allowed)}"
        )
    declared_roles = {str(value) for value in payload.get("output_roles", [])}
    if declared_roles != allowed:
        raise ValueError("LEO cache-set output_roles do not match the required role set")

    physical_ids_hash = ids_sha256(reference_ids or [])
    if str(payload.get("physical_sample_ids_sha256", "")) != physical_ids_hash:
        raise ValueError("LEO cache-set physical sample ID root mismatch")
    audit = {
        "path": str(path),
        "sha256": sha256_file(path),
        "scope": str(expected_scope),
        "scenario_order": list(FORMAL_LEO_WEAK_SCENARIOS),
        "physical_sample_count": len(reference_ids or []),
        "physical_sample_ids_sha256": physical_ids_hash,
        "cache_audits": cache_audits,
        "clean_sample_access": False,
        "phase2_sample_view_policy": PHASE2_SAMPLE_VIEW_POLICY,
    }
    return arrays_by_scenario, payload, audit
