"""Build truth-free Target5/Target25 inputs from validated received-IQ packages."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Mapping, Sequence

from .stage2_target_row_export import export_target_row


_PLAN_SCHEMA = "cvs.stage2.meta_adapter.target_factory_plan.v1"
_PLAN_KEYS = frozenset(
    {
        "schema",
        "target",
        "candidate_id",
        "bundle_id",
        "checkpoint_path",
        "prototype_path",
        "protocol_schema",
        "phase2_data_status",
        "seed",
        "steps",
        "entries",
    }
)
_ENTRY_KEYS = frozenset(
    {"receiver", "operating_point", "k_shot", "manifest_path", "scenario_inputs"}
)
_SCENARIO_INPUT_KEYS = frozenset({"support_input", "query_input"})
_SCENARIOS = (
    "leo_clear_weak",
    "leo_low_elev_weak",
    "leo_rain_weak",
)
_OPERATING_POINTS = {
    "K10/new5": 10,
    "K10/new10": 10,
    "K10/new20": 10,
    "K5/new20": 5,
    "K1/new20": 1,
}
_TARGET_RECEIVERS = {
    "Target5": ("20-1",),
    "Target25": ("20-1", "3-19", "7-14", "7-7", "8-8"),
}
_MANIFEST_SCHEMA = "cvs.full_ablation.phase2.feature_cache_manifest.v2"


class MetaAdapterTargetFactoryError(ValueError):
    """Raised when target input preparation violates the frozen contract."""


def _exact_keys(
    payload: Mapping[str, Any], expected: frozenset[str], *, label: str
) -> None:
    actual = frozenset(payload)
    if actual != expected:
        raise MetaAdapterTargetFactoryError(
            f"{label} allowlist mismatch: "
            f"missing={sorted(expected - actual)} extra={sorted(actual - expected)}"
        )


def _read_manifest(path: str | Path) -> dict[str, Any]:
    resolved = Path(path)
    if resolved.is_symlink() or not resolved.is_file():
        raise MetaAdapterTargetFactoryError(
            f"validated feature manifest is missing or invalid: {resolved}"
        )
    try:
        payload = json.loads(resolved.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise MetaAdapterTargetFactoryError(
            f"cannot read validated feature manifest: {resolved}"
        ) from exc
    if not isinstance(payload, dict):
        raise MetaAdapterTargetFactoryError("validated feature manifest must be an object")
    return payload


def _validated_entries(plan: Mapping[str, Any]) -> tuple[str, list[dict[str, Any]]]:
    if not isinstance(plan, Mapping):
        raise MetaAdapterTargetFactoryError("target factory plan must be a mapping")
    _exact_keys(plan, _PLAN_KEYS, label="target factory plan")
    if plan["schema"] != _PLAN_SCHEMA:
        raise MetaAdapterTargetFactoryError(f"plan schema must be {_PLAN_SCHEMA}")
    target = str(plan["target"])
    if target not in _TARGET_RECEIVERS:
        raise MetaAdapterTargetFactoryError("target must be Target5 or Target25")
    if plan["protocol_schema"] != "p2_min_v1":
        raise MetaAdapterTargetFactoryError("protocol_schema must be p2_min_v1")
    if plan["phase2_data_status"] != "VALIDATED_ONCE":
        raise MetaAdapterTargetFactoryError(
            "phase2_data_status must be VALIDATED_ONCE"
        )
    if plan["steps"] != 3:
        raise MetaAdapterTargetFactoryError("formal target factory steps must be 3")
    if isinstance(plan["seed"], bool) or not isinstance(plan["seed"], int):
        raise MetaAdapterTargetFactoryError("seed must be an integer")
    for key in ("candidate_id", "bundle_id", "checkpoint_path", "prototype_path"):
        if not isinstance(plan[key], str) or not plan[key].strip():
            raise MetaAdapterTargetFactoryError(f"{key} must be a nonempty string")

    raw_entries = plan["entries"]
    if not isinstance(raw_entries, Sequence) or isinstance(raw_entries, (str, bytes)):
        raise MetaAdapterTargetFactoryError("entries must be a sequence")
    expected_entry_count = len(_TARGET_RECEIVERS[target]) * len(_OPERATING_POINTS)
    if len(raw_entries) != expected_entry_count:
        if target == "Target5":
            raise MetaAdapterTargetFactoryError(
                "Target5 requires exactly five operating points"
            )
        raise MetaAdapterTargetFactoryError(
            f"Target25 requires exactly {expected_entry_count} operating-point entries"
        )

    entries: list[dict[str, Any]] = []
    identities: set[tuple[str, str]] = set()
    for index, raw in enumerate(raw_entries):
        if not isinstance(raw, Mapping):
            raise MetaAdapterTargetFactoryError(f"entry {index} must be a mapping")
        _exact_keys(raw, _ENTRY_KEYS, label=f"entry {index}")
        receiver = str(raw["receiver"])
        operating_point = str(raw["operating_point"])
        if operating_point not in _OPERATING_POINTS:
            raise MetaAdapterTargetFactoryError(
                f"entry {index} has a non-frozen operating-point"
            )
        k_shot = raw["k_shot"]
        if (
            isinstance(k_shot, bool)
            or not isinstance(k_shot, int)
            or k_shot != _OPERATING_POINTS[operating_point]
        ):
            raise MetaAdapterTargetFactoryError(
                f"entry {index} k_shot does not match its operating-point"
            )
        identity = (receiver, operating_point)
        if identity in identities:
            raise MetaAdapterTargetFactoryError(
                f"duplicate receiver/operating-point entry: {identity}"
            )
        identities.add(identity)

        scenario_inputs = raw["scenario_inputs"]
        if not isinstance(scenario_inputs, Mapping) or set(scenario_inputs) != set(
            _SCENARIOS
        ):
            raise MetaAdapterTargetFactoryError(
                f"entry {index} must expose exactly the three LEO weak scenarios"
            )
        normalized_inputs: dict[str, dict[str, str]] = {}
        for scenario in _SCENARIOS:
            item = scenario_inputs[scenario]
            if not isinstance(item, Mapping):
                raise MetaAdapterTargetFactoryError(
                    f"entry {index} scenario {scenario} must be a mapping"
                )
            _exact_keys(
                item,
                _SCENARIO_INPUT_KEYS,
                label=f"entry {index} scenario {scenario}",
            )
            support_input = str(item["support_input"])
            query_input = str(item["query_input"])
            if not support_input or not query_input or support_input == query_input:
                raise MetaAdapterTargetFactoryError(
                    f"entry {index} scenario {scenario} has invalid input paths"
                )
            normalized_inputs[scenario] = {
                "support_input": support_input,
                "query_input": query_input,
            }

        manifest = _read_manifest(raw["manifest_path"])
        if manifest.get("schema") != _MANIFEST_SCHEMA:
            raise MetaAdapterTargetFactoryError("feature manifest schema mismatch")
        expected_values = {
            "stage_scope": "stage2b",
            "receiver": receiver,
            "k_shot": k_shot,
            "phase2_data_status": "VALIDATED_ONCE",
            "scenarios": list(_SCENARIOS),
            "query_truth_present": False,
            "query_role_present": False,
            "phase2_source_sample_access": False,
            "phase2_source_cache_access": False,
            "phase2_source_label_access": False,
            "phase2_source_replay": False,
        }
        for key, expected in expected_values.items():
            actual = manifest.get(key)
            if actual != expected:
                if key == "query_truth_present":
                    raise MetaAdapterTargetFactoryError(
                        "feature manifest exposes query truth"
                    )
                if key == "query_role_present":
                    raise MetaAdapterTargetFactoryError(
                        "feature manifest exposes query role"
                    )
                raise MetaAdapterTargetFactoryError(
                    f"feature manifest {key} mismatch: {actual!r} != {expected!r}"
                )
        capsule_id = str(manifest.get("capsule_id", ""))
        split_id = str(manifest.get("split_id", ""))
        if not capsule_id or not split_id.startswith("p2_min_v1-"):
            raise MetaAdapterTargetFactoryError(
                "feature manifest capsule_id/split_id is not p2_min_v1-bound"
            )
        expected_operating_tag = (
            f"k{k_shot}-new{operating_point.rsplit('new', 1)[1]}"
        )
        if expected_operating_tag not in capsule_id or expected_operating_tag not in split_id:
            raise MetaAdapterTargetFactoryError(
                "feature manifest operating-point identity mismatch"
            )
        entries.append(
            {
                "receiver": receiver,
                "operating_point": operating_point,
                "k_shot": k_shot,
                "capsule_id": capsule_id,
                "split_id": split_id,
                "scenario_inputs": normalized_inputs,
            }
        )

    expected_identities = {
        (receiver, operating_point)
        for receiver in _TARGET_RECEIVERS[target]
        for operating_point in _OPERATING_POINTS
    }
    if identities != expected_identities:
        raise MetaAdapterTargetFactoryError(
            f"{target} entries do not match the frozen receiver/operating-point product"
        )
    return target, entries


def _write_json_exclusive(path: Path, payload: Mapping[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    if path.exists() or path.is_symlink() or temporary.exists():
        raise FileExistsError(f"target factory artifact already exists: {path}")
    try:
        with temporary.open("x", encoding="utf-8", newline="\n") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
            handle.write("\n")
        if path.exists() or path.is_symlink():
            raise FileExistsError(f"target factory artifact appeared: {path}")
        os.replace(temporary, path)
    except Exception:
        if temporary.exists():
            temporary.unlink()
        raise


def _row_id(receiver: str, operating_point: str, scenario: str) -> str:
    return (
        f"rx{receiver.replace('-', '_')}-"
        f"{operating_point.lower().replace('/', '_')}-{scenario}"
    )


def build_meta_adapter_target_matrix(
    plan: Mapping[str, Any], output_root: str | Path
) -> Mapping[str, Any]:
    """Materialize fixed IQ carriers and a runner-ready truth-free matrix."""

    target, entries = _validated_entries(plan)
    destination = Path(output_root)
    if destination.exists() or destination.is_symlink():
        raise FileExistsError(f"target factory output root exists: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.mkdir(parents=False, exist_ok=False)
    rows: list[dict[str, Any]] = []
    active_row_id = ""
    try:
        for entry in entries:
            for scenario in _SCENARIOS:
                active_row_id = _row_id(
                    entry["receiver"], entry["operating_point"], scenario
                )
                row_root = destination / "rows" / active_row_id
                support_path = row_root / "support.npz"
                query_path = row_root / "query.npz"
                audit_path = row_root / "export_audit.json"
                inputs = entry["scenario_inputs"][scenario]
                export_target_row(
                    support_input=inputs["support_input"],
                    support_output=support_path,
                    audit_output=audit_path,
                    k_shot=entry["k_shot"],
                    query_input=inputs["query_input"],
                    query_output=query_path,
                )
                rows.append(
                    {
                        "row_id": active_row_id,
                        "config": {
                            "candidate_id": plan["candidate_id"],
                            "bundle_id": plan["bundle_id"],
                            "protocol_schema": "p2_min_v1",
                            "phase2_data_status": "VALIDATED_ONCE",
                            "capsule_id": entry["capsule_id"],
                            "split_id": entry["split_id"],
                            "checkpoint_path": plan["checkpoint_path"],
                            "support_path": str(support_path.resolve()),
                            "query_path": str(query_path.resolve()),
                            "prototype_path": plan["prototype_path"],
                            "receiver": entry["receiver"],
                            "scenario": scenario,
                            "operating_point": entry["operating_point"],
                            "seed": int(plan["seed"]),
                            "k_shot": int(entry["k_shot"]),
                            "steps": 3,
                        },
                    }
                )
        matrix = {
            "schema": "cvs.stage2.meta_adapter.matrix.v1",
            "target": target,
            "rows": rows,
        }
        matrix_path = destination / "matrix_config.json"
        _write_json_exclusive(matrix_path, matrix)
        receipt = {
            "schema": "cvs.stage2.meta_adapter.target_factory_receipt.v1",
            "status": "TARGET_INPUTS_COMPLETE",
            "target": target,
            "row_count": len(rows),
            "matrix_config_path": str(matrix_path.resolve()),
            "query_truth_opened": False,
            "query_role_opened": False,
            "source_opened": False,
        }
        _write_json_exclusive(destination / "factory_receipt.json", receipt)
        return receipt
    except Exception as exc:
        try:
            _write_json_exclusive(
                destination / "factory_failure.json",
                {
                    "schema": "cvs.stage2.meta_adapter.target_factory_failure.v1",
                    "status": "FAILED",
                    "target": target,
                    "completed_row_count": len(rows),
                    "failed_row_id": active_row_id,
                    "error_type": type(exc).__name__,
                    "error_message": str(exc),
                    "query_truth_opened": False,
                    "query_role_opened": False,
                },
            )
        except Exception:
            pass
        raise


__all__ = [
    "MetaAdapterTargetFactoryError",
    "build_meta_adapter_target_matrix",
]
