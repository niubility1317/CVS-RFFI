#!/usr/bin/env python3
"""Bind every Phase2 logical row to one verified immutable cache package."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Any, Mapping


CODE_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = CODE_ROOT.parent
for value in (str(CODE_ROOT), str(REPO_ROOT)):
    while value in sys.path:
        sys.path.remove(value)
for value in (str(REPO_ROOT), str(CODE_ROOT)):
    sys.path.insert(0, value)

from cvsrffi.full_ablation_spec import validate_plan_rows  # noqa: E402
from cvsrffi.stage2_ablation_feature_cache import (  # noqa: E402
    load_feature_cache,
)
from cvsrffi.stage2_ablation_release import (  # noqa: E402
    BINDING_REGISTRY_SCHEMA,
    validate_binding_registry,
)
from cvsrffi.stage2_predictor_bundle import (  # noqa: E402
    preflight_stage2_predictor_package,
    sha256_file,
)
from cvsrffi.stage2_metric_scorer import (  # noqa: E402
    load_verified_scoring_sidecar,
)


CACHE_BINDING_INDEX_SCHEMA = (
    "cvs.full_ablation.phase2.cache_binding_index.v1"
)
_INDEX_KEYS = {"schema", "candidate_lock_sha256", "entries"}
_ENTRY_KEYS = {
    "stage_scope",
    "receiver",
    "method_seed",
    "support_seed",
    "query_seed",
    "new_class_draw_seed",
    "k_shot",
    "new_class_count",
    "feature_cache_payload",
    "feature_cache_manifest",
    "predictor_package_root",
    "predictor_detached_seal",
    "predictor_detached_seal_sha256",
    "scoring_manifest",
}


class Stage2BindingRegistryBuildError(ValueError):
    """Raised when a complete exact binding registry cannot be produced."""


def _load_json(path: str | Path) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8-sig"))
    if not isinstance(payload, dict):
        raise Stage2BindingRegistryBuildError(
            f"JSON object required: {path}"
        )
    return payload


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _canonical_manifest_sha256(path: Path) -> str:
    payload = path.read_bytes()
    if payload.endswith(b"\n"):
        payload = payload[:-1]
    return _sha256_bytes(payload)


def _require_sha256(value: Any, *, name: str) -> str:
    text = str(value)
    if len(text) != 64 or any(char not in "0123456789abcdef" for char in text):
        raise Stage2BindingRegistryBuildError(
            f"{name} must be a lowercase SHA256"
        )
    return text


def _require_nonnegative_int(value: Any, *, name: str) -> int:
    if isinstance(value, bool):
        raise Stage2BindingRegistryBuildError(f"{name} must be an integer")
    try:
        result = int(value)
    except (TypeError, ValueError) as exc:
        raise Stage2BindingRegistryBuildError(
            f"{name} must be an integer"
        ) from exc
    if result < 0:
        raise Stage2BindingRegistryBuildError(
            f"{name} must be nonnegative"
        )
    return result


def _identity(
    *,
    stage_scope: str,
    receiver: str,
    method_seed: int,
    support_seed: int,
    query_seed: int,
    new_class_draw_seed: int,
    k_shot: int,
    new_class_count: int,
) -> tuple[Any, ...]:
    return (
        stage_scope,
        receiver,
        method_seed,
        0 if stage_scope == "stage2a" else support_seed,
        query_seed,
        0 if stage_scope != "stage2c" else new_class_draw_seed,
        0 if stage_scope == "stage2a" else k_shot,
        0 if stage_scope != "stage2c" else new_class_count,
    )


def _validate_stage_fields(
    *,
    stage_scope: str,
    support_seed: int,
    new_class_draw_seed: int,
    k_shot: int,
    new_class_count: int,
    context: str,
) -> None:
    if stage_scope == "stage2a":
        if any(
            value != 0
            for value in (
                support_seed,
                new_class_draw_seed,
                k_shot,
                new_class_count,
            )
        ):
            raise Stage2BindingRegistryBuildError(
                f"{context} Stage2-A must bind zero support metadata"
            )
    elif stage_scope == "stage2b":
        if (
            support_seed <= 0
            or k_shot not in {1, 2, 5, 10}
            or new_class_draw_seed != 0
            or new_class_count != 0
        ):
            raise Stage2BindingRegistryBuildError(
                f"{context} Stage2-B support/draw metadata drift"
            )
    elif stage_scope == "stage2c":
        if (
            support_seed <= 0
            or k_shot not in {1, 2, 5, 10}
            or new_class_draw_seed <= 0
            or new_class_count <= 0
        ):
            raise Stage2BindingRegistryBuildError(
                f"{context} Stage2-C support/draw metadata drift"
            )
    else:
        raise Stage2BindingRegistryBuildError(
            f"{context} stage_scope is invalid"
        )


def _row_identity(row: Mapping[str, Any]) -> tuple[Any, ...]:
    stage_scope = str(row.get("phase", ""))
    return _identity(
        stage_scope=stage_scope,
        receiver=str(row.get("receiver_id", "")),
        method_seed=_require_nonnegative_int(
            row.get("method_seed"), name="row.method_seed"
        ),
        support_seed=_require_nonnegative_int(
            row.get("support_seed") or 0, name="row.support_seed"
        ),
        query_seed=_require_nonnegative_int(
            row.get("query_seed"), name="row.query_seed"
        ),
        new_class_draw_seed=_require_nonnegative_int(
            row.get("new_class_draw_seed") or 0,
            name="row.new_class_draw_seed",
        ),
        k_shot=_require_nonnegative_int(
            row.get("k_shot") or 0, name="row.k_shot"
        ),
        new_class_count=_require_nonnegative_int(
            row.get("new_class_count") or 0,
            name="row.new_class_count",
        ),
    )


def _entry_identity(entry: Mapping[str, Any]) -> tuple[Any, ...]:
    stage_scope = str(entry["stage_scope"])
    support_seed = _require_nonnegative_int(
        entry["support_seed"], name="entry.support_seed"
    )
    new_class_draw_seed = _require_nonnegative_int(
        entry["new_class_draw_seed"],
        name="entry.new_class_draw_seed",
    )
    k_shot = _require_nonnegative_int(
        entry["k_shot"], name="entry.k_shot"
    )
    new_class_count = _require_nonnegative_int(
        entry["new_class_count"], name="entry.new_class_count"
    )
    _validate_stage_fields(
        stage_scope=stage_scope,
        support_seed=support_seed,
        new_class_draw_seed=new_class_draw_seed,
        k_shot=k_shot,
        new_class_count=new_class_count,
        context="entry",
    )
    return _identity(
        stage_scope=stage_scope,
        receiver=str(entry["receiver"]),
        method_seed=_require_nonnegative_int(
            entry["method_seed"], name="entry.method_seed"
        ),
        support_seed=support_seed,
        query_seed=_require_nonnegative_int(
            entry["query_seed"], name="entry.query_seed"
        ),
        new_class_draw_seed=new_class_draw_seed,
        k_shot=k_shot,
        new_class_count=new_class_count,
    )


def _validate_entry(
    raw: Mapping[str, Any],
    *,
    candidate_lock_sha256: str,
) -> dict[str, Any]:
    if not isinstance(raw, Mapping) or set(raw) != _ENTRY_KEYS:
        raise Stage2BindingRegistryBuildError(
            "cache-binding index entry exact schema drift"
        )
    entry = dict(raw)
    identity = _entry_identity(entry)
    stage_scope = str(entry["stage_scope"])
    if stage_scope not in {"stage2a", "stage2b", "stage2c"}:
        raise Stage2BindingRegistryBuildError(
            "cache-binding entry stage_scope is invalid"
        )
    for field in (
        "feature_cache_payload",
        "feature_cache_manifest",
        "predictor_package_root",
        "predictor_detached_seal",
        "scoring_manifest",
    ):
        path = Path(str(entry[field]))
        if not path.is_absolute():
            raise Stage2BindingRegistryBuildError(
                f"{field} must be an absolute path"
            )
        entry[field] = str(path)

    payload_path = Path(entry["feature_cache_payload"])
    feature_manifest_path = Path(entry["feature_cache_manifest"])
    payload_sha256 = sha256_file(payload_path)
    manifest_sha256 = _canonical_manifest_sha256(feature_manifest_path)
    loaded_cache = load_feature_cache(
        payload_path,
        feature_manifest_path,
        expected_payload_sha256=payload_sha256,
        expected_manifest_sha256=manifest_sha256,
    )
    feature_manifest = dict(loaded_cache["manifest"])
    manifest_support_seed = int(feature_manifest["support_seed"])
    manifest_draw_seed = int(feature_manifest["new_class_draw_seed"])
    manifest_k_shot = int(feature_manifest["k_shot"])
    manifest_new_count = len(feature_manifest["new_classes"])
    _validate_stage_fields(
        stage_scope=str(feature_manifest["stage_scope"]),
        support_seed=manifest_support_seed,
        new_class_draw_seed=manifest_draw_seed,
        k_shot=manifest_k_shot,
        new_class_count=manifest_new_count,
        context="feature manifest",
    )
    expected_from_manifest = _identity(
        stage_scope=str(feature_manifest["stage_scope"]),
        receiver=str(feature_manifest["receiver"]),
        method_seed=int(feature_manifest["method_seed"]),
        support_seed=manifest_support_seed,
        query_seed=int(feature_manifest["query_seed"]),
        new_class_draw_seed=manifest_draw_seed,
        k_shot=manifest_k_shot,
        new_class_count=manifest_new_count,
    )
    if identity != expected_from_manifest:
        raise Stage2BindingRegistryBuildError(
            "cache-binding identity does not match feature manifest"
        )

    seal_sha256 = _require_sha256(
        entry["predictor_detached_seal_sha256"],
        name="entry.predictor_detached_seal_sha256",
    )
    predictor_manifest, predictor_seal, _ = (
        preflight_stage2_predictor_package(
            entry["predictor_package_root"],
            detached_seal_path=entry["predictor_detached_seal"],
            expected_seal_sha256=seal_sha256,
        )
    )
    expected_predictor_stage = (
        "stage2c" if stage_scope == "stage2c" else "stage2b"
    )
    if (
        predictor_manifest["stage"] != expected_predictor_stage
        or predictor_manifest["receiver"] != entry["receiver"]
        or predictor_manifest["candidate_lock_sha256"]
        != candidate_lock_sha256
        or predictor_manifest["package_root_sha256"]
        != feature_manifest["package_root_sha256"]
        or predictor_seal["package_root_sha256"]
        != feature_manifest["package_root_sha256"]
        or seal_sha256 != feature_manifest["package_seal_sha256"]
    ):
        raise Stage2BindingRegistryBuildError(
            "predictor package does not match the feature-cache binding"
        )

    scoring_manifest_sha256 = sha256_file(entry["scoring_manifest"])
    truth, scoring_payload, scoring_audit = load_verified_scoring_sidecar(
        entry["scoring_manifest"],
        expected_scoring_manifest_sha256=scoring_manifest_sha256,
    )
    if (
        truth["stage"] != stage_scope
        or truth["receiver"] != entry["receiver"]
        or
        scoring_payload["predictor_package_root_sha256"]
        != feature_manifest["package_root_sha256"]
        or scoring_payload["predictor_package_seal_sha256"]
        != feature_manifest["package_seal_sha256"]
    ):
        raise Stage2BindingRegistryBuildError(
            "scoring manifest does not match the feature-cache package"
        )

    return {
        **entry,
        "_identity": identity,
        "_feature_cache_payload_sha256": payload_sha256,
        "_feature_cache_manifest_sha256": manifest_sha256,
        "_phase2_data_status": feature_manifest["phase2_data_status"],
        "_capsule_id": feature_manifest["capsule_id"],
        "_split_id": feature_manifest["split_id"],
        "_phase1_bundle_sha256": feature_manifest[
            "phase1_bundle_sha256"
        ],
        "_phase1_prototype_sha256": feature_manifest[
            "phase1_prototype_sha256"
        ],
        "_scoring_manifest_sha256": scoring_audit[
            "scoring_manifest_sha256"
        ],
    }


def build_registry(
    plan: Mapping[str, Any],
    cache_index: Mapping[str, Any],
) -> dict[str, Any]:
    if (
        not isinstance(plan, Mapping)
        or plan.get("schema") != "cvs.full_ablation.plan.v1"
        or plan.get("phase") != "phase2"
    ):
        raise Stage2BindingRegistryBuildError(
            "source plan is not a Phase2 full-ablation plan"
        )
    rows = list(plan.get("rows") or [])
    validate_plan_rows(rows)
    if (
        not isinstance(cache_index, Mapping)
        or set(cache_index) != _INDEX_KEYS
        or cache_index.get("schema") != CACHE_BINDING_INDEX_SCHEMA
        or not isinstance(cache_index.get("entries"), list)
    ):
        raise Stage2BindingRegistryBuildError(
            "cache-binding index exact schema drift"
        )
    candidate_lock_sha256 = _require_sha256(
        cache_index["candidate_lock_sha256"],
        name="candidate_lock_sha256",
    )
    entries = [
        _validate_entry(
            item,
            candidate_lock_sha256=candidate_lock_sha256,
        )
        for item in cache_index["entries"]
    ]
    by_identity = {item["_identity"]: item for item in entries}
    if len(by_identity) != len(entries):
        raise Stage2BindingRegistryBuildError(
            "cache-binding index contains duplicate identities"
        )
    expected_identities = {_row_identity(row) for row in rows}
    if set(by_identity) != expected_identities:
        missing = len(expected_identities - set(by_identity))
        extra = len(set(by_identity) - expected_identities)
        raise Stage2BindingRegistryBuildError(
            "cache-binding index coverage mismatch: "
            f"missing={missing}, extra={extra}"
        )

    bindings = []
    for row in rows:
        entry = by_identity[_row_identity(row)]
        bindings.append(
            {
                "row_key": str(row["row_key"]),
                "mode": "execute",
                "feature_cache_payload": entry[
                    "feature_cache_payload"
                ],
                "feature_cache_payload_sha256": entry[
                    "_feature_cache_payload_sha256"
                ],
                "feature_cache_manifest": entry[
                    "feature_cache_manifest"
                ],
                "feature_cache_manifest_sha256": entry[
                    "_feature_cache_manifest_sha256"
                ],
                "phase2_data_status": entry["_phase2_data_status"],
                "capsule_id": entry["_capsule_id"],
                "split_id": entry["_split_id"],
                "phase1_bundle_sha256": entry[
                    "_phase1_bundle_sha256"
                ],
                "phase1_prototype_sha256": entry[
                    "_phase1_prototype_sha256"
                ],
                "scoring_manifest": entry["scoring_manifest"],
                "scoring_manifest_sha256": entry[
                    "_scoring_manifest_sha256"
                ],
                "reuse_row_execution_receipt": "",
                "reuse_row_execution_receipt_sha256": "",
                "reuse_physical_execution_id": "",
            }
        )
    registry = {
        "schema": BINDING_REGISTRY_SCHEMA,
        "candidate_lock_sha256": candidate_lock_sha256,
        "bindings": bindings,
    }
    validate_binding_registry(
        registry,
        expected_row_keys=[str(row["row_key"]) for row in rows],
    )
    return registry


def _write_new(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    data = (
        json.dumps(
            payload,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
        + b"\n"
    )
    descriptor = os.open(
        path,
        os.O_WRONLY
        | os.O_CREAT
        | os.O_EXCL
        | getattr(os, "O_BINARY", 0),
        0o444,
    )
    try:
        view = memoryview(data)
        while view:
            written = os.write(descriptor, view)
            if written <= 0:
                raise OSError("short write while publishing binding registry")
            view = view[written:]
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", required=True)
    parser.add_argument("--cache-binding-index", required=True)
    parser.add_argument("--output", required=True)
    return parser


def main() -> int:
    args = _parser().parse_args()
    registry = build_registry(
        _load_json(args.plan),
        _load_json(args.cache_binding_index),
    )
    output = Path(args.output)
    _write_new(output, registry)
    print(
        json.dumps(
            {
                "output": str(output),
                "logical_binding_count": len(registry["bindings"]),
                "candidate_lock_sha256": registry[
                    "candidate_lock_sha256"
                ],
                "cross_launch_data_identity_required": False,
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
