"""Seal and validate immutable Phase2 full-ablation release plans.

The release surface deliberately binds reusable feature caches instead of
rebuilding or re-auditing the underlying dataset for every launch.  Different
launches may bind different already-valid caches.  Within one logical row,
the cache and truth-side scoring manifest remain immutable so prediction and
same-row scoring cannot silently separate.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

from cvsrffi.full_ablation_spec import (
    GPU_COUNT,
    SLOTS_PER_GPU,
    validate_plan_rows,
)
from cvsrffi.stage2_ablation_factory import (
    get_stage2_arm,
    resolved_stage2_config_hash,
)


BINDING_REGISTRY_SCHEMA = (
    "cvs.full_ablation.phase2.binding_registry.v1"
)
SEALED_PLAN_SCHEMA = "cvs.full_ablation.phase2.sealed_plan.v1"
SCORE_REQUEST_SCHEMA = "cvs.full_ablation.phase2.score_request.v1"
SCORE_COMPLETION_SCHEMA = (
    "cvs.full_ablation.phase2.score_completion.v1"
)
TERMINAL_ROW_SCHEMA = "cvs.full_ablation.phase2.terminal_row.v1"
RUNNER_SUMMARY_SCHEMA = "cvs.full_ablation.phase2.runner_summary.v1"

_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_PHYSICAL_ID_RE = re.compile(r"^[A-Za-z0-9_.-]{1,128}$")
_BINDING_KEYS = {
    "row_key",
    "mode",
    "feature_cache_payload",
    "feature_cache_payload_sha256",
    "feature_cache_manifest",
    "feature_cache_manifest_sha256",
    "phase2_data_status",
    "capsule_id",
    "split_id",
    "phase1_bundle_sha256",
    "phase1_prototype_sha256",
    "scoring_manifest",
    "scoring_manifest_sha256",
    "reuse_row_execution_receipt",
    "reuse_row_execution_receipt_sha256",
    "reuse_physical_execution_id",
}
_MODES = {"execute", "reuse_prediction"}


class Stage2AblationReleaseError(ValueError):
    """Raised when a Phase2 release would be incomplete or ambiguous."""


def canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def sha256_object(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def _sha256(value: Any, *, name: str) -> str:
    text = str(value)
    if _SHA256_RE.fullmatch(text) is None:
        raise Stage2AblationReleaseError(
            f"{name} must be a lowercase SHA256"
        )
    return text


def _text(value: Any, *, name: str) -> str:
    if (
        not isinstance(value, str)
        or not value
        or value.strip() != value
        or "\x00" in value
    ):
        raise Stage2AblationReleaseError(
            f"{name} must be nonempty trimmed text"
        )
    return value


def _path_text(value: Any, *, name: str) -> str:
    text = _text(value, name=name)
    if "\n" in text or "\r" in text:
        raise Stage2AblationReleaseError(f"{name} contains a newline")
    return text


def _exclusive_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    data = canonical_json_bytes(payload) + b"\n"
    descriptor = os.open(
        path,
        os.O_WRONLY | os.O_CREAT | os.O_EXCL
        | getattr(os, "O_BINARY", 0),
        0o444,
    )
    try:
        view = memoryview(data)
        while view:
            written = os.write(descriptor, view)
            if written <= 0:
                raise OSError("short write while sealing Phase2 release")
            view = view[written:]
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _validate_binding(
    raw: Mapping[str, Any],
) -> dict[str, Any]:
    if not isinstance(raw, Mapping) or set(raw) != _BINDING_KEYS:
        raise Stage2AblationReleaseError(
            "binding exact schema drift"
        )
    binding = dict(raw)
    binding["row_key"] = _text(
        binding["row_key"], name="binding.row_key"
    )
    mode = _text(binding["mode"], name="binding.mode")
    if mode not in _MODES:
        raise Stage2AblationReleaseError(
            f"unsupported binding mode: {mode}"
        )
    binding["mode"] = mode
    for field in (
        "feature_cache_payload",
        "feature_cache_manifest",
        "scoring_manifest",
    ):
        binding[field] = _path_text(
            binding[field], name=f"binding.{field}"
        )
    for field in (
        "feature_cache_payload_sha256",
        "feature_cache_manifest_sha256",
        "scoring_manifest_sha256",
        "phase1_bundle_sha256",
        "phase1_prototype_sha256",
    ):
        binding[field] = _sha256(
            binding[field], name=f"binding.{field}"
        )
    if binding["phase2_data_status"] != "VALIDATED_ONCE":
        raise Stage2AblationReleaseError(
            "binding Phase2 data status is not VALIDATED_ONCE"
        )
    binding["capsule_id"] = _text(
        binding["capsule_id"], name="binding.capsule_id"
    )
    binding["split_id"] = _text(
        binding["split_id"], name="binding.split_id"
    )
    reuse = binding["reuse_row_execution_receipt"]
    if mode == "execute":
        if (
            reuse not in {"", None}
            or binding["reuse_row_execution_receipt_sha256"]
            not in {"", None}
            or binding["reuse_physical_execution_id"] not in {"", None}
        ):
            raise Stage2AblationReleaseError(
                "execute binding cannot provide reuse identity"
            )
        binding["reuse_row_execution_receipt"] = ""
        binding["reuse_row_execution_receipt_sha256"] = ""
        binding["reuse_physical_execution_id"] = ""
    else:
        binding["reuse_row_execution_receipt"] = _path_text(
            reuse,
            name="binding.reuse_row_execution_receipt",
        )
        binding["reuse_row_execution_receipt_sha256"] = _sha256(
            binding["reuse_row_execution_receipt_sha256"],
            name="binding.reuse_row_execution_receipt_sha256",
        )
        reuse_physical_id = _text(
            binding["reuse_physical_execution_id"],
            name="binding.reuse_physical_execution_id",
        )
        if _PHYSICAL_ID_RE.fullmatch(reuse_physical_id) is None:
            raise Stage2AblationReleaseError(
                "reuse physical execution ID is unsafe"
            )
        binding["reuse_physical_execution_id"] = reuse_physical_id
    return binding


def validate_binding_registry(
    registry: Mapping[str, Any],
    *,
    expected_row_keys: Sequence[str],
) -> dict[str, dict[str, Any]]:
    if (
        not isinstance(registry, Mapping)
        or set(registry)
        != {"schema", "candidate_lock_sha256", "bindings"}
        or registry.get("schema") != BINDING_REGISTRY_SCHEMA
    ):
        raise Stage2AblationReleaseError(
            "binding registry exact schema drift"
        )
    _sha256(
        registry["candidate_lock_sha256"],
        name="candidate_lock_sha256",
    )
    raw_bindings = registry.get("bindings")
    if not isinstance(raw_bindings, list):
        raise Stage2AblationReleaseError(
            "binding registry bindings must be a list"
        )
    bindings = [_validate_binding(item) for item in raw_bindings]
    by_key = {item["row_key"]: item for item in bindings}
    expected = set(str(value) for value in expected_row_keys)
    if (
        len(by_key) != len(bindings)
        or set(by_key) != expected
    ):
        raise Stage2AblationReleaseError(
            "binding registry must bind every logical row exactly once"
        )
    return by_key


def _row_stage(row: Mapping[str, Any]) -> str:
    phase = str(row.get("phase", ""))
    if phase not in {"stage2a", "stage2b", "stage2c"}:
        raise Stage2AblationReleaseError(
            "sealed Phase2 row has unsupported stage"
        )
    return phase


def _binding_identity_key(
    row: Mapping[str, Any],
    binding: Mapping[str, Any],
) -> tuple[Any, ...]:
    """Identity for physical work; no cross-launch data equality is assumed."""

    spec = get_stage2_arm(str(row["ablation_id"]))
    physical_config_id = str(
        row.get("physical_config_id") or spec.alias_of
        or spec.ablation_id
    )
    source = (
        str(binding["reuse_row_execution_receipt"])
        if binding["mode"] == "reuse_prediction"
        else ""
    )
    return (
        physical_config_id,
        str(binding["feature_cache_payload_sha256"]),
        str(binding["feature_cache_manifest_sha256"]),
        str(row["receiver_id"]),
        int(row["method_seed"]),
        _row_stage(row),
        0 if _row_stage(row) == "stage2a" else int(row["k_shot"]),
        int(row.get("new_class_count") or 0),
        int(row.get("support_seed") or 0),
        int(row["query_seed"]),
        int(row.get("new_class_draw_seed") or 0),
        str(binding["phase2_data_status"]),
        str(binding["capsule_id"]),
        str(binding["split_id"]),
        str(binding["phase1_bundle_sha256"]),
        str(binding["phase1_prototype_sha256"]),
        source,
        str(binding["reuse_row_execution_receipt_sha256"]),
        str(binding["reuse_physical_execution_id"]),
    )


def _input_identity(
    row: Mapping[str, Any],
    binding: Mapping[str, Any],
) -> dict[str, Any]:
    stage = _row_stage(row)
    return {
        "stage_scope": stage,
        "k_shot": 0 if stage == "stage2a" else int(row["k_shot"]),
        "new_class_count": int(row.get("new_class_count") or 0),
        "method_seed": int(row["method_seed"]),
        "support_seed": int(row.get("support_seed") or 0),
        "query_seed": int(row["query_seed"]),
        "new_class_draw_seed": int(
            row.get("new_class_draw_seed") or 0
        ),
        "phase2_data_status": str(binding["phase2_data_status"]),
        "capsule_id": str(binding["capsule_id"]),
        "split_id": str(binding["split_id"]),
        "phase1_bundle_sha256": str(
            binding["phase1_bundle_sha256"]
        ),
        "phase1_prototype_sha256": str(
            binding["phase1_prototype_sha256"]
        ),
        "feature_cache_payload_sha256": str(
            binding["feature_cache_payload_sha256"]
        ),
        "feature_cache_manifest_sha256": str(
            binding["feature_cache_manifest_sha256"]
        ),
    }


def _safe_row_name(row_key: str) -> str:
    prefix = re.sub(r"[^A-Za-z0-9_.-]+", "_", row_key)[:80]
    return f"{prefix}__{hashlib.sha256(row_key.encode()).hexdigest()[:16]}"


def _make_request(
    *,
    representative: Mapping[str, Any],
    binding: Mapping[str, Any],
    candidate_lock_sha256: str,
    physical_execution_id: str,
    output_root: Path,
    device: str,
    shared_view_count: int,
) -> dict[str, Any]:
    return {
        "schema": "cvs.full_ablation.phase2.row_request.v1",
        "ablation_id": str(representative["ablation_id"]),
        "row_id": physical_execution_id,
        "receiver": str(representative["receiver_id"]),
        "stage_scope": _row_stage(representative),
        "k_shot": (
            0
            if _row_stage(representative) == "stage2a"
            else int(representative["k_shot"])
        ),
        "new_class_count": int(
            representative.get("new_class_count") or 0
        ),
        "support_seed": int(
            representative.get("support_seed") or 0
        ),
        "query_seed": int(representative["query_seed"]),
        "new_class_draw_seed": int(
            representative.get("new_class_draw_seed") or 0
        ),
        "phase2_data_status": str(binding["phase2_data_status"]),
        "capsule_id": str(binding["capsule_id"]),
        "split_id": str(binding["split_id"]),
        "phase1_bundle_sha256": str(
            binding["phase1_bundle_sha256"]
        ),
        "phase1_prototype_sha256": str(
            binding["phase1_prototype_sha256"]
        ),
        "candidate_lock_sha256": candidate_lock_sha256,
        "effective_config_hash": resolved_stage2_config_hash(
            str(representative["ablation_id"])
        ),
        "feature_cache_payload": str(
            binding["feature_cache_payload"]
        ),
        "feature_cache_payload_sha256": str(
            binding["feature_cache_payload_sha256"]
        ),
        "feature_cache_manifest": str(
            binding["feature_cache_manifest"]
        ),
        "feature_cache_manifest_sha256": str(
            binding["feature_cache_manifest_sha256"]
        ),
        "output_root": str(output_root),
        "seed": int(representative["method_seed"]),
        "device": str(device),
        "shared_view_count": int(shared_view_count),
    }


def _score_request(
    *,
    logical_row: Mapping[str, Any],
    physical_execution_id: str,
    representative_row_key: str,
    effective_config_hash: str,
    receipt_path: str,
    binding: Mapping[str, Any],
    output_path: Path,
    completion_receipt_path: Path,
) -> dict[str, Any]:
    alias_of = (
        None
        if str(logical_row["row_key"]) == representative_row_key
        else representative_row_key
    )
    return {
        "schema": SCORE_REQUEST_SCHEMA,
        "logical_row_key": str(logical_row["row_key"]),
        "ablation_id": str(logical_row["ablation_id"]),
        "physical_execution_id": physical_execution_id,
        "effective_config_hash": effective_config_hash,
        "alias_of": alias_of,
        "row_execution_receipt": receipt_path,
        "scoring_manifest": str(binding["scoring_manifest"]),
        "scoring_manifest_sha256": str(
            binding["scoring_manifest_sha256"]
        ),
        "output_path": str(output_path),
        "completion_receipt_path": str(completion_receipt_path),
    }


def seal_stage2_plan(
    plan: Mapping[str, Any],
    registry: Mapping[str, Any],
    *,
    run_id: str,
    request_root: str | Path,
    run_root: str | Path,
    log_root: str | Path,
    python_environment_id: str,
    review_p0_count: int,
    review_p1_count: int,
    device: str = "cuda:0",
    shared_view_count: int = 1,
    write_requests: bool = True,
) -> dict[str, Any]:
    """Bind one plan to reusable caches and freeze 8x2 physical queues."""

    if (
        plan.get("schema") != "cvs.full_ablation.plan.v1"
        or plan.get("phase") != "phase2"
    ):
        raise Stage2AblationReleaseError(
            "source plan is not a Phase2 full-ablation plan"
        )
    rows = list(plan.get("rows") or [])
    validate_plan_rows(rows)
    if int(review_p0_count) != 0 or int(review_p1_count) != 0:
        raise Stage2AblationReleaseError(
            "formal release requires independent P0=0 and P1=0"
        )
    if int(shared_view_count) not in {1, 3, 5}:
        raise Stage2AblationReleaseError(
            "shared_view_count must be 1, 3, or 5"
        )
    environment = _text(
        python_environment_id,
        name="python_environment_id",
    )
    if environment != "CVS-RFFI":
        raise Stage2AblationReleaseError(
            "N607 Phase2 release must use CVS-RFFI"
        )
    run_name = _text(run_id, name="run_id")
    candidate_lock = _sha256(
        registry.get("candidate_lock_sha256"),
        name="candidate_lock_sha256",
    )
    bindings = validate_binding_registry(
        registry,
        expected_row_keys=[str(row["row_key"]) for row in rows],
    )
    request_dir = Path(request_root)
    run_dir = Path(run_root)
    log_dir = Path(log_root)

    groups: dict[tuple[Any, ...], list[Mapping[str, Any]]] = (
        defaultdict(list)
    )
    for row in rows:
        get_stage2_arm(str(row["ablation_id"]))
        groups[_binding_identity_key(row, bindings[str(row["row_key"])])].append(
            row
        )

    physical_rows: list[dict[str, Any]] = []
    sorted_groups = sorted(
        groups.values(),
        key=lambda group: min(str(row["row_key"]) for row in group),
    )
    for index, group in enumerate(sorted_groups):
        # Prefer the non-alias arm as the actual executor when P2-F3 and
        # P2-FULL collapse to one physical configuration.
        representative = sorted(
            group,
            key=lambda row: (
                get_stage2_arm(str(row["ablation_id"])).alias_of
                is not None,
                str(row["row_key"]),
            ),
        )[0]
        representative_key = str(representative["row_key"])
        binding = bindings[representative_key]
        for logical in group:
            current = bindings[str(logical["row_key"])]
            for field in (
                "mode",
                "feature_cache_payload_sha256",
                "feature_cache_manifest_sha256",
                "feature_cache_payload",
                "feature_cache_manifest",
                "scoring_manifest_sha256",
                "scoring_manifest",
                "phase2_data_status",
                "capsule_id",
                "split_id",
                "phase1_bundle_sha256",
                "phase1_prototype_sha256",
                "reuse_row_execution_receipt",
                "reuse_row_execution_receipt_sha256",
                "reuse_physical_execution_id",
            ):
                if current[field] != binding[field]:
                    raise Stage2AblationReleaseError(
                        "physical alias bindings are not identical"
                    )
        physical_hash_input = {
            "run_id": run_name,
            "representative_row_key": representative_key,
            "physical_config_id": str(
                representative.get("physical_config_id")
                or representative["ablation_id"]
            ),
            "feature_cache_payload_sha256": binding[
                "feature_cache_payload_sha256"
            ],
            "feature_cache_manifest_sha256": binding[
                "feature_cache_manifest_sha256"
            ],
        }
        physical_execution_id = (
            str(binding["reuse_physical_execution_id"])
            if binding["mode"] == "reuse_prediction"
            else "phys_" + sha256_object(physical_hash_input)[:24]
        )
        output_dir = run_dir / "physical" / physical_execution_id
        request_path = (
            request_dir / "predict" / f"{physical_execution_id}.json"
        )
        if binding["mode"] == "execute":
            receipt_path = str(
                output_dir / "row_execution_receipt.json"
            )
            request = _make_request(
                representative=representative,
                binding=binding,
                candidate_lock_sha256=candidate_lock,
                physical_execution_id=physical_execution_id,
                output_root=output_dir,
                device=device,
                shared_view_count=shared_view_count,
            )
        else:
            receipt_path = str(
                binding["reuse_row_execution_receipt"]
            )
            request = None
        worker_index = index % (GPU_COUNT * SLOTS_PER_GPU)
        gpu = worker_index // SLOTS_PER_GPU
        slot = worker_index % SLOTS_PER_GPU
        logical_records: list[dict[str, Any]] = []
        for logical in sorted(
            group, key=lambda row: str(row["row_key"])
        ):
            logical_key = str(logical["row_key"])
            logical_binding = bindings[logical_key]
            score_request_path = (
                request_dir
                / "score"
                / f"{_safe_row_name(logical_key)}.json"
            )
            score_output_path = (
                run_dir
                / "logical"
                / f"{_safe_row_name(logical_key)}.json"
            )
            score_completion_path = (
                run_dir
                / "logical"
                / f"{_safe_row_name(logical_key)}.completion.json"
            )
            score_request = _score_request(
                logical_row=logical,
                physical_execution_id=physical_execution_id,
                representative_row_key=representative_key,
                effective_config_hash=resolved_stage2_config_hash(
                    str(logical["ablation_id"])
                ),
                receipt_path=receipt_path,
                binding=logical_binding,
                output_path=score_output_path,
                completion_receipt_path=score_completion_path,
            )
            if write_requests:
                _exclusive_json(score_request_path, score_request)
            logical_records.append(
                {
                    "logical_row_key": logical_key,
                    "ablation_id": str(logical["ablation_id"]),
                    "effective_config_hash": score_request[
                        "effective_config_hash"
                    ],
                    "alias_of": score_request["alias_of"],
                    "score_request_path": str(score_request_path),
                    "score_request_sha256": sha256_object(
                        score_request
                    ),
                    "score_output_path": str(score_output_path),
                    "score_completion_path": str(
                        score_completion_path
                    ),
                }
            )
        if write_requests and request is not None:
            _exclusive_json(request_path, request)
        physical_rows.append(
            {
                "physical_execution_id": physical_execution_id,
                "representative_logical_row_key": representative_key,
                "representative_ablation_id": str(
                    representative["ablation_id"]
                ),
                "stage": _row_stage(representative),
                "receiver": str(representative["receiver_id"]),
                "k_shot": (
                    0
                    if _row_stage(representative) == "stage2a"
                    else int(representative["k_shot"])
                ),
                "effective_config_hash": resolved_stage2_config_hash(
                    str(representative["ablation_id"])
                ),
                "mode": str(binding["mode"]),
                "worker": {"gpu": gpu, "slot": slot},
                "prediction_request_path": (
                    str(request_path) if request is not None else ""
                ),
                "prediction_request_sha256": (
                    sha256_object(request)
                    if request is not None
                    else ""
                ),
                "row_execution_receipt": receipt_path,
                "reuse_row_execution_receipt_sha256": str(
                    binding["reuse_row_execution_receipt_sha256"]
                ),
                "input_identity": _input_identity(
                    representative, binding
                ),
                "log_path": str(
                    log_dir / f"{physical_execution_id}.out"
                ),
                "status_path": str(
                    log_dir
                    / "status"
                    / f"{physical_execution_id}.json"
                ),
                "logical_rows": logical_records,
            }
        )

    source_plan_sha256 = sha256_object(plan)
    sealed = {
        "schema": SEALED_PLAN_SCHEMA,
        "run_id": run_name,
        "design_id": str(plan["design_id"]),
        "source_plan_sha256": source_plan_sha256,
        "git_commit": str(plan["git_commit"]),
        "candidate_lock_sha256": candidate_lock,
        "python_environment_id": environment,
        "review_p0_count": 0,
        "review_p1_count": 0,
        "formal_launch_authority": True,
        "gpu_count": GPU_COUNT,
        "slots_per_gpu": SLOTS_PER_GPU,
        "logical_row_count": len(rows),
        "physical_execution_count": len(physical_rows),
        "reused_physical_count": sum(
            row["mode"] == "reuse_prediction"
            for row in physical_rows
        ),
        "alias_logical_count": sum(
            logical["alias_of"] is not None
            for physical in physical_rows
            for logical in physical["logical_rows"]
        ),
        "request_root": str(request_dir),
        "run_root": str(run_dir),
        "log_root": str(log_dir),
        "physical_rows": physical_rows,
    }
    sealed["sealed_content_sha256"] = sha256_object(sealed)
    validate_sealed_stage2_plan(sealed)
    return sealed


def validate_sealed_stage2_plan(
    plan: Mapping[str, Any],
) -> None:
    if plan.get("schema") != SEALED_PLAN_SCHEMA:
        raise Stage2AblationReleaseError(
            "sealed Phase2 plan schema drift"
        )
    content = dict(plan)
    sealed_hash = content.pop("sealed_content_sha256", None)
    if sealed_hash != sha256_object(content):
        raise Stage2AblationReleaseError(
            "sealed Phase2 plan content hash drift"
        )
    if (
        plan.get("formal_launch_authority") is not True
        or int(plan.get("review_p0_count", -1)) != 0
        or int(plan.get("review_p1_count", -1)) != 0
        or plan.get("python_environment_id") != "CVS-RFFI"
        or int(plan.get("gpu_count", -1)) != GPU_COUNT
        or int(plan.get("slots_per_gpu", -1)) != SLOTS_PER_GPU
    ):
        raise Stage2AblationReleaseError(
            "sealed Phase2 launch authority is incomplete"
        )
    physical_rows = list(plan.get("physical_rows") or [])
    request_root = Path(str(plan.get("request_root", "")))
    run_root = Path(str(plan.get("run_root", "")))
    log_root = Path(str(plan.get("log_root", "")))
    if not all(
        path.is_absolute()
        for path in (request_root, run_root, log_root)
    ):
        raise Stage2AblationReleaseError(
            "sealed release roots must be absolute"
        )

    def require_within(
        raw_path: Any,
        root: Path,
        *,
        name: str,
    ) -> None:
        path = Path(str(raw_path))
        if not path.is_absolute():
            raise Stage2AblationReleaseError(
                f"{name} must be absolute"
            )
        try:
            path.resolve(strict=False).relative_to(
                root.resolve(strict=False)
            )
        except ValueError as exc:
            raise Stage2AblationReleaseError(
                f"{name} escapes its sealed root"
            ) from exc

    physical_ids = [
        str(item.get("physical_execution_id", ""))
        for item in physical_rows
    ]
    if (
        len(physical_rows)
        != int(plan.get("physical_execution_count", -1))
        or len(set(physical_ids)) != len(physical_ids)
        or any(not value for value in physical_ids)
    ):
        raise Stage2AblationReleaseError(
            "sealed physical execution identities drift"
        )
    logical_rows = [
        logical
        for physical in physical_rows
        for logical in list(physical.get("logical_rows") or [])
    ]
    logical_keys = [
        str(item.get("logical_row_key", ""))
        for item in logical_rows
    ]
    if (
        len(logical_rows) != int(plan.get("logical_row_count", -1))
        or len(set(logical_keys)) != len(logical_keys)
        or any(not value for value in logical_keys)
    ):
        raise Stage2AblationReleaseError(
            "sealed logical row identities drift"
        )
    for physical in physical_rows:
        worker = physical.get("worker") or {}
        gpu = int(worker.get("gpu", -1))
        slot = int(worker.get("slot", -1))
        if (
            gpu < 0
            or gpu >= GPU_COUNT
            or slot < 0
            or slot >= SLOTS_PER_GPU
            or physical.get("mode") not in _MODES
        ):
            raise Stage2AblationReleaseError(
                "sealed physical worker or mode drift"
            )
        if (
            physical.get("stage")
            not in {"stage2a", "stage2b", "stage2c"}
            or not str(physical.get("receiver", ""))
            or isinstance(physical.get("k_shot"), bool)
            or not isinstance(physical.get("k_shot"), int)
            or (
                physical["stage"] == "stage2a"
                and physical["k_shot"] != 0
            )
            or (
                physical["stage"] != "stage2a"
                and physical["k_shot"] <= 0
            )
        ):
            raise Stage2AblationReleaseError(
                "sealed physical stage identity drift"
            )
        if (
            physical["mode"] == "execute"
            and not str(physical.get("prediction_request_path", ""))
        ):
            raise Stage2AblationReleaseError(
                "execute physical row lacks a prediction request"
            )
        require_within(
            physical.get("log_path"),
            log_root,
            name="physical log path",
        )
        require_within(
            physical.get("status_path"),
            log_root,
            name="physical status path",
        )
        if physical["mode"] == "execute":
            require_within(
                physical.get("prediction_request_path"),
                request_root,
                name="prediction request path",
            )
            _sha256(
                physical.get("prediction_request_sha256"),
                name="prediction request SHA256",
            )
            require_within(
                physical.get("row_execution_receipt"),
                run_root,
                name="row execution receipt",
            )
        elif physical.get("prediction_request_sha256") != "":
            raise Stage2AblationReleaseError(
                "reuse physical row cannot bind a predictor request hash"
            )
        if not list(physical.get("logical_rows") or []):
            raise Stage2AblationReleaseError(
                "physical row has no logical result"
            )
        for logical in physical["logical_rows"]:
            _sha256(
                logical.get("effective_config_hash"),
                name="logical effective_config_hash",
            )
            if not str(logical.get("score_request_path", "")) or not str(
                logical.get("score_output_path", "")
            ) or not str(logical.get("score_completion_path", "")):
                raise Stage2AblationReleaseError(
                    "logical row lacks score paths"
                )
            require_within(
                logical["score_request_path"],
                request_root,
                name="score request path",
            )
            require_within(
                logical["score_output_path"],
                run_root,
                name="score output path",
            )
            require_within(
                logical["score_completion_path"],
                run_root,
                name="score completion path",
            )
            _sha256(
                logical.get("score_request_sha256"),
                name="score request SHA256",
            )


__all__ = [
    "BINDING_REGISTRY_SCHEMA",
    "RUNNER_SUMMARY_SCHEMA",
    "SCORE_COMPLETION_SCHEMA",
    "SCORE_REQUEST_SCHEMA",
    "SEALED_PLAN_SCHEMA",
    "TERMINAL_ROW_SCHEMA",
    "Stage2AblationReleaseError",
    "canonical_json_bytes",
    "seal_stage2_plan",
    "sha256_object",
    "validate_binding_registry",
    "validate_sealed_stage2_plan",
]
