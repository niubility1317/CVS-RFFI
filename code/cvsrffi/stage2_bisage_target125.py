"""Historical D92 E0 Target125 binding for BiSAGE-D92.

This module only binds already validated Phase2 packages to new immutable
output roots.  It never opens query payloads or truth.
"""

from __future__ import annotations

from copy import deepcopy
from pathlib import PurePosixPath
from typing import Any, Mapping, Sequence


class BiSAGETarget125Error(ValueError):
    """Raised when the historical Target125 identity drifts."""


RECEIVERS = ("20-1", "3-19", "7-14", "7-7", "8-8")
SEEDS = (713102, 713103, 713104, 713105, 713106)
SLICES = ((1, 20), (5, 20), (10, 5), (10, 10), (10, 20))
SCENARIOS = ("leo_clear_weak", "leo_low_elev_weak", "leo_rain_weak")
QUERY_DENIAL_FIELDS = (
    "truth_access",
    "fit_access",
    "update_access",
    "selection_access",
    "role_oracle_access",
    "class_quota_access",
    "global_reassignment",
)
HISTORICAL_RUN_ID = "d92_e0_full_only_target125_20260812_v1"
HISTORICAL_MANIFEST_SHA256 = (
    "5910674066e8bbf93684fddd6af6fd2cef7e8f208d64e403ac7e58030a2a8cc5"
)


def _outer_key(receiver: str, seed: int, k_shot: int, new_count: int) -> str:
    return (
        f"rx_{receiver.replace('-', '_')}__seed_{int(seed)}"
        f"__k_{int(k_shot)}__new_{int(new_count)}"
    )


def canonical_target125_rows() -> tuple[dict[str, Any], ...]:
    rows: list[dict[str, Any]] = []
    for receiver in RECEIVERS:
        for seed in SEEDS:
            for k_shot, new_count in SLICES:
                rows.append(
                    {
                        "outer_key": _outer_key(receiver, seed, k_shot, new_count),
                        "receiver": receiver,
                        "seed": seed,
                        "k_shot": k_shot,
                        "new_class_count": new_count,
                        "scenarios": list(SCENARIOS),
                    }
                )
    return tuple(rows)


def _exact_sequence(actual: Any, expected: Sequence[Any], field: str) -> None:
    if tuple(actual or ()) != tuple(expected):
        raise BiSAGETarget125Error(f"historical Target125 {field} drift")


def validate_target125_config(config: Mapping[str, Any]) -> dict[str, int]:
    if config.get("schema") != "cvs.phase2.bisage_d92_target125.method_lock.v1":
        raise BiSAGETarget125Error("method lock schema drift")
    if config.get("protocol_schema") != "p2_min_v1":
        raise BiSAGETarget125Error("protocol schema drift")
    if config.get("phase2_data_status") != "VALIDATED_ONCE":
        raise BiSAGETarget125Error("phase2 data status drift")
    _exact_sequence(config.get("receivers"), RECEIVERS, "receiver")
    _exact_sequence(config.get("seeds"), SEEDS, "seed")
    actual_slices = tuple(
        (int(row["k_shot"]), int(row["new_class_count"]))
        for row in config.get("slices", ())
    )
    if actual_slices != SLICES:
        raise BiSAGETarget125Error("historical Target125 slice drift")
    _exact_sequence(config.get("scenarios"), SCENARIOS, "scenario")
    matrix = config.get("matrix", {})
    if matrix != {"outer_count": 125, "scene_count": 3, "scene_unit_count": 375}:
        raise BiSAGETarget125Error("historical Target125 matrix count drift")
    query_contract = config.get("query_contract", {})
    if any(query_contract.get(field) is not False for field in QUERY_DENIAL_FIELDS):
        raise BiSAGETarget125Error("query contract must deny every mutable or truth path")
    historical = config.get("historical_source", {})
    if historical.get("run_id") != HISTORICAL_RUN_ID:
        raise BiSAGETarget125Error("historical source run drift")
    if historical.get("matrix_manifest_sha256") != HISTORICAL_MANIFEST_SHA256:
        raise BiSAGETarget125Error("historical source manifest digest drift")
    return {"outer_count": 125, "scene_unit_count": 375}


def _source_jobs(source_manifest: Mapping[str, Any]) -> dict[str, Mapping[str, Any]]:
    jobs = source_manifest.get("jobs")
    if not isinstance(jobs, list):
        raise BiSAGETarget125Error("historical source manifest jobs missing")
    indexed: dict[str, Mapping[str, Any]] = {}
    for job in jobs:
        if not isinstance(job, Mapping) or not isinstance(job.get("outer_key"), str):
            raise BiSAGETarget125Error("historical source job identity missing")
        if job["outer_key"] in indexed:
            raise BiSAGETarget125Error("historical source outer coverage duplicated")
        indexed[job["outer_key"]] = job
    return indexed


def build_bisage_target125_manifest(
    config: Mapping[str, Any],
    source_manifest: Mapping[str, Any],
    output_root: str,
) -> dict[str, Any]:
    validate_target125_config(config)
    root = PurePosixPath(str(output_root))
    if str(root) in ("", ".", "/"):
        raise BiSAGETarget125Error("output root must be a dedicated immutable path")
    sources = _source_jobs(source_manifest)
    if source_manifest.get("protocol_schema") != "p2_min_v1":
        raise BiSAGETarget125Error("historical source protocol drift")
    if source_manifest.get("schema") != "cvs.phase2.d92_e0_full_only_target125.matrix.v1":
        raise BiSAGETarget125Error("historical source matrix schema drift")
    capsule_id = f"d92-e0-full-target125:{HISTORICAL_MANIFEST_SHA256}"
    jobs: list[dict[str, Any]] = []
    for index, row in enumerate(canonical_target125_rows()):
        source = sources.get(row["outer_key"])
        if source is None:
            raise BiSAGETarget125Error(f"historical source outer missing: {row['outer_key']}")
        for field in ("receiver", "seed", "k_shot", "new_class_count"):
            if source.get(field) != row[field]:
                raise BiSAGETarget125Error(f"historical source {field} drift")
        if tuple(source.get("scenarios", ())) != SCENARIOS:
            raise BiSAGETarget125Error("historical source scenario drift")
        split_id = f"d92-e0-full-target125:{row['outer_key']}"
        packages = source.get("packages")
        truth_sidecar = source.get("truth_sidecar")
        if not isinstance(packages, Mapping) or not packages:
            raise BiSAGETarget125Error("historical sealed packages missing")
        if not isinstance(truth_sidecar, str) or not truth_sidecar:
            raise BiSAGETarget125Error("historical truth sidecar missing")
        job = deepcopy(row)
        job.update(
            {
                "planned_shard_index": index % 8,
                "protocol_schema": "p2_min_v1",
                "phase2_data_status": "VALIDATED_ONCE",
                "capsule_id": capsule_id,
                "split_id": split_id,
                "source_capsule_id": capsule_id,
                "source_split_id": split_id,
                "source_job_root": source.get("source_job_root"),
                "packages": deepcopy(dict(packages)),
                "truth_sidecar": truth_sidecar,
                "output_root": str(root / "output" / "jobs" / row["outer_key"]),
                "selected_mode_policy": (
                    "S0_K1_FALLBACK" if row["k_shot"] == 1 else "SUPPORT_ONLY_S2_S1_S0"
                ),
            }
        )
        jobs.append(job)
    manifest = {
        "schema": "cvs.phase2.bisage_d92_target125.manifest.v1",
        "protocol_schema": "p2_min_v1",
        "phase2_data_status": "VALIDATED_ONCE",
        "historical_source": deepcopy(config["historical_source"]),
        "query_contract": deepcopy(config["query_contract"]),
        "jobs": jobs,
    }
    validate_bisage_target125_manifest(manifest, config)
    return manifest


def validate_bisage_target125_manifest(
    manifest: Mapping[str, Any], config: Mapping[str, Any]
) -> dict[str, int]:
    validate_target125_config(config)
    if manifest.get("schema") != "cvs.phase2.bisage_d92_target125.manifest.v1":
        raise BiSAGETarget125Error("Target125 manifest schema drift")
    if manifest.get("protocol_schema") != "p2_min_v1":
        raise BiSAGETarget125Error("Target125 manifest protocol drift")
    if manifest.get("phase2_data_status") != "VALIDATED_ONCE":
        raise BiSAGETarget125Error("Target125 manifest data status drift")
    if manifest.get("historical_source") != config.get("historical_source"):
        raise BiSAGETarget125Error("historical source binding drift")
    if manifest.get("query_contract") != config.get("query_contract"):
        raise BiSAGETarget125Error("query contract drift")
    jobs = manifest.get("jobs")
    if not isinstance(jobs, list) or len(jobs) != 125:
        raise BiSAGETarget125Error("outer coverage must contain 125 jobs")
    expected = {row["outer_key"]: row for row in canonical_target125_rows()}
    actual_keys = [job.get("outer_key") for job in jobs if isinstance(job, Mapping)]
    if len(actual_keys) != 125 or set(actual_keys) != set(expected):
        raise BiSAGETarget125Error("outer coverage drift")
    output_roots: list[str] = []
    truth_sidecars: list[str] = []
    k1_count = 0
    for job in jobs:
        row = expected[job["outer_key"]]
        for field in ("receiver", "seed", "k_shot", "new_class_count", "scenarios"):
            if job.get(field) != row[field]:
                raise BiSAGETarget125Error(f"Target125 {field} drift")
        if job.get("capsule_id") != job.get("source_capsule_id"):
            raise BiSAGETarget125Error("capsule_id drift from historical source")
        if job.get("split_id") != job.get("source_split_id"):
            raise BiSAGETarget125Error("split_id drift from historical source")
        if job.get("protocol_schema") != "p2_min_v1" or job.get("phase2_data_status") != "VALIDATED_ONCE":
            raise BiSAGETarget125Error("job protocol/data status drift")
        expected_policy = "S0_K1_FALLBACK" if row["k_shot"] == 1 else "SUPPORT_ONLY_S2_S1_S0"
        if job.get("selected_mode_policy") != expected_policy:
            raise BiSAGETarget125Error("support-only selection policy drift")
        expected_index = canonical_target125_rows().index(row) % 8
        if job.get("planned_shard_index") != expected_index:
            raise BiSAGETarget125Error("planned shard index drift")
        if not isinstance(job.get("packages"), Mapping) or not job["packages"]:
            raise BiSAGETarget125Error("sealed packages missing")
        truth_sidecar = job.get("truth_sidecar")
        if not isinstance(truth_sidecar, str) or not truth_sidecar:
            raise BiSAGETarget125Error("truth sidecar missing")
        truth_sidecars.append(truth_sidecar)
        if row["k_shot"] == 1:
            k1_count += 1
        output_root = job.get("output_root")
        if not isinstance(output_root, str):
            raise BiSAGETarget125Error("output root missing")
        output_roots.append(output_root)
    if len(set(output_roots)) != 125:
        raise BiSAGETarget125Error("output root collision")
    if len(set(truth_sidecars)) != 125:
        raise BiSAGETarget125Error("truth sidecar collision")
    return {"outer_count": 125, "scene_unit_count": 375, "k1_fallback_count": k1_count}
