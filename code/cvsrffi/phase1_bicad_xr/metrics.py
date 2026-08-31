"""Diagnostics and artifact-closure helpers for source-only BiCAD-XR rows.

The module deliberately keeps the row identity and final-evaluation checks
small and explicit.  It does not load target or Phase2 data and it does not
make a performance claim from a partially reconstructed checkpoint.
"""

from __future__ import annotations

import copy
import json
import math
import re
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any


LEO_WEAK_SCENARIOS: tuple[str, str, str] = (
    "leo_clear_weak",
    "leo_low_elev_weak",
    "leo_rain_weak",
)
FORMAL_EVAL_SCENARIOS: tuple[str, str, str, str] = (
    "clean",
    *LEO_WEAK_SCENARIOS,
)


DIAGNOSTIC_SCHEMA: tuple[str, ...] = (
    "conditional_receiver_probe",
    "zdom_tx_probe",
    "domain_classifier_accuracy",
    "xdc_donor_query_matrix",
    "paired_satellite",
    "margin_q0_1",
    "worst_tx_receiver",
    "worst_tx_receiver_day",
    "worst_tx_receiver_channel",
    "gradient_ratios",
    "projection_trigger_rate",
    "effective_xdc_donors",
    "ridge_condition_numbers",
    "throughput_samples_per_second",
    "peak_gpu_memory_bytes",
    "gpu_hours",
    "extra_forward_ratio",
    "inference_parameter_count",
    "receiver_floor",
    "receiver_std",
    "negative_margin_rate",
)

_DEFAULT_DIAGNOSTICS: dict[str, Any] = {
    name: "N/A" for name in DIAGNOSTIC_SCHEMA
}


class BiCADXRMetricStore:
    """Mutable collector with a fixed, JSON-friendly diagnostic schema.

    Diagnostics that are not produced by a candidate remain ``"N/A"``.  An
    unknown key is rejected so a typo cannot silently create a second schema.
    """

    schema = DIAGNOSTIC_SCHEMA

    def __init__(self, initial: Mapping[str, Any] | None = None) -> None:
        self._values = copy.deepcopy(_DEFAULT_DIAGNOSTICS)
        if initial is not None:
            self.update(initial)

    def update(
        self,
        values: Mapping[str, Any] | None = None,
        **kwargs: Any,
    ) -> "BiCADXRMetricStore":
        merged: dict[str, Any] = {}
        if values is not None:
            if not isinstance(values, Mapping):
                raise TypeError("diagnostics must be a mapping")
            merged.update(values)
        merged.update(kwargs)
        unknown = sorted(set(merged) - set(DIAGNOSTIC_SCHEMA))
        if unknown:
            raise KeyError("unknown BiCAD-XR diagnostic(s): " + ", ".join(unknown))
        self._values.update(copy.deepcopy(merged))
        return self

    def set(self, name: str, value: Any) -> "BiCADXRMetricStore":
        return self.update({name: value})

    def snapshot(self) -> dict[str, Any]:
        return copy.deepcopy(self._values)

    def to_dict(self) -> dict[str, Any]:
        return self.snapshot()

    def __getitem__(self, key: str) -> Any:
        return self._values[key]

    def get(self, key: str, default: Any = None) -> Any:
        return self._values.get(key, default)

    def keys(self):
        return self._values.keys()

    def items(self):
        return self._values.items()


_SDG_ALIASES: dict[str, tuple[str, ...]] = {
    "clean_bal": ("clean_bal", "clean_accuracy", "clean_floor_accuracy"),
    "leo_scene_floor_bal": (
        "leo_scene_floor_bal",
        "leo_scenario_floor",
        "leo_scene_floor",
        "leo_floor",
    ),
    "receiver_floor": ("receiver_floor", "receiver_floor_accuracy"),
    "receiver_std": ("receiver_std", "receiver_population_std"),
    "negative_margin_rate": ("negative_margin_rate", "negative_margin"),
}


def _finite_metric(value: Any, *, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a finite number")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be a finite number")
    return result


def _sdg_mapping_value(mapping: Mapping[str, Any], name: str) -> Any:
    nested = mapping.get("metrics")
    sources: tuple[Mapping[str, Any], ...]
    if isinstance(nested, Mapping):
        sources = (mapping, nested)
    else:
        sources = (mapping,)
    for source in sources:
        for alias in _SDG_ALIASES[name]:
            if alias in source:
                return source[alias]
    raise ValueError(f"missing S_DG input: {name}")


def compute_s_dg(
    clean_bal: float | Mapping[str, Any],
    leo_scene_floor_bal: float | None = None,
    receiver_floor: float | None = None,
    receiver_std: float | None = None,
    negative_margin_rate: float | None = None,
) -> float:
    """Compute the frozen source-domain generalization score.

    The score is defined on the 0--1 scale as the three-way harmonic mean of
    clean, LEO-scene-floor, and receiver-floor accuracy, with the specified
    receiver-variance and negative-margin penalties.  A mapping input accepts
    the row/artifact field aliases used by the analyzer.
    """

    if isinstance(clean_bal, Mapping):
        values = {
            name: _finite_metric(_sdg_mapping_value(clean_bal, name), name=name)
            for name in _SDG_ALIASES
        }
    else:
        raw_values = {
            "clean_bal": clean_bal,
            "leo_scene_floor_bal": leo_scene_floor_bal,
            "receiver_floor": receiver_floor,
            "receiver_std": receiver_std,
            "negative_margin_rate": negative_margin_rate,
        }
        missing = [name for name, value in raw_values.items() if value is None]
        if missing:
            raise ValueError("missing S_DG input: " + ", ".join(missing))
        values = {
            name: _finite_metric(value, name=name)
            for name, value in raw_values.items()
        }

    for name in ("clean_bal", "leo_scene_floor_bal", "receiver_floor"):
        if not 0.0 <= values[name] <= 1.0:
            raise ValueError(f"{name} must be between 0 and 1")
    if values["receiver_std"] < 0.0:
        raise ValueError("receiver_std must be non-negative")
    if not 0.0 <= values["negative_margin_rate"] <= 1.0:
        raise ValueError("negative_margin_rate must be between 0 and 1")

    accuracy_values = tuple(
        values[name] for name in ("clean_bal", "leo_scene_floor_bal", "receiver_floor")
    )
    harmonic_mean = 0.0
    if all(value > 0.0 for value in accuracy_values):
        denominator = sum(1.0 / value for value in accuracy_values)
        harmonic_mean = 3.0 / denominator
    return harmonic_mean - 0.10 * values["receiver_std"] - 0.05 * values[
        "negative_margin_rate"
    ]


S_DG = compute_s_dg
s_dg = compute_s_dg


def aggregate_s_dg(rows: Sequence[Mapping[str, Any]]) -> dict[str, float]:
    """Aggregate already-scored rows without mixing unrelated metrics."""

    if isinstance(rows, (str, bytes)) or not isinstance(rows, Sequence) or not rows:
        raise ValueError("rows must be a non-empty sequence")
    values = []
    for row in rows:
        if not isinstance(row, Mapping):
            raise ValueError("each S_DG row must be a mapping")
        value = row.get("s_dg", row.get("S_DG"))
        values.append(_finite_metric(value, name="s_dg"))
    mean = sum(values) / len(values)
    return {
        "mean": mean,
        "population_std": (sum((value - mean) ** 2 for value in values) / len(values))
        ** 0.5,
        "minimum": min(values),
    }


_GATE_METRIC_ALIASES: dict[str, tuple[str, ...]] = {
    "s_dg": ("s_dg", "S_DG"),
    "leo_mean": ("leo_mean", "leo_accuracy_mean"),
    "leo_class_floor": ("leo_class_floor", "leo_class_floor_accuracy"),
    "clean_accuracy": ("clean_accuracy", "clean_bal"),
    "h": ("source_sat_hmean", "hmean", "H", "h"),
    "receiver_floor": ("receiver_floor", "receiver_floor_accuracy"),
}


def _gate_metric(row: Mapping[str, Any], name: str) -> float:
    for alias in _GATE_METRIC_ALIASES[name]:
        if alias in row:
            return _finite_metric(row[alias], name=name)
    raise ValueError(f"missing gate metric: {name}")


_SAME_ROW_IDENTITY_FIELDS = (
    "fold",
    "seed",
    "same_row_key",
    "split_id",
    "k_shot",
    "source_receivers",
    "train_days",
)


def _assert_same_row(candidate: Mapping[str, Any], control: Mapping[str, Any]) -> None:
    required_fields = ("fold", "seed")
    missing = [
        name
        for name in required_fields
        if name not in candidate or name not in control
    ]
    if missing:
        raise ValueError("same-row identity missing: " + ", ".join(missing))

    mismatches = [
        name
        for name in _SAME_ROW_IDENTITY_FIELDS
        if name in candidate and name in control and candidate[name] != control[name]
    ]
    if mismatches:
        raise ValueError("same-row identity mismatch: " + ", ".join(mismatches))


def _gate_result(
    candidate: Mapping[str, Any],
    control: Mapping[str, Any],
    *,
    kind: str,
    checks: Mapping[str, bool],
    deltas: Mapping[str, float],
) -> dict[str, Any]:
    _assert_same_row(candidate, control)
    failed = [name for name, passed in checks.items() if not passed]
    passed = not failed
    return {
        "gate": kind,
        "same_row": True,
        "candidate_id": candidate.get("candidate_id"),
        "control_candidate_id": control.get("candidate_id"),
        "passed": passed,
        "status": "PASS" if passed else "FAIL",
        "scientific_result": (
            "SCIENTIFIC_GATE_PASS" if passed else "NEGATIVE_SCIENTIFIC_RESULT"
        ),
        "technical_failure": False,
        "failed_conditions": failed,
        "deltas": dict(deltas),
    }


def evaluate_mainline_gate(
    candidate: Mapping[str, Any], control: Mapping[str, Any]
) -> dict[str, Any]:
    """Evaluate the mainline criteria against a same-row control."""

    _assert_same_row(candidate, control)
    deltas = {
        name: _gate_metric(candidate, name) - _gate_metric(control, name)
        for name in ("s_dg", "leo_mean", "leo_class_floor", "clean_accuracy", "h")
    }
    checks = {
        "s_dg_delta": deltas["s_dg"] >= 0.005,
        "leo_mean_delta": deltas["leo_mean"] > 0.0,
        "leo_class_floor_delta": deltas["leo_class_floor"] >= 0.0,
        "clean_accuracy_delta": deltas["clean_accuracy"] >= -0.005,
        "h_delta": deltas["h"] >= -0.03,
    }
    return _gate_result(candidate, control, kind="mainline", checks=checks, deltas=deltas)


def evaluate_tailguard_gate(
    candidate: Mapping[str, Any], control: Mapping[str, Any]
) -> dict[str, Any]:
    """Evaluate the dedicated TailGuard criteria against a same-row control."""

    _assert_same_row(candidate, control)
    deltas = {
        name: _gate_metric(candidate, name) - _gate_metric(control, name)
        for name in (
            "leo_class_floor",
            "clean_accuracy",
            "h",
            "receiver_floor",
        )
    }
    checks = {
        "leo_class_floor_delta": deltas["leo_class_floor"] >= 0.01,
        "h_delta": deltas["h"] >= -0.003,
        "clean_accuracy_delta": deltas["clean_accuracy"] >= -0.005,
        "receiver_floor_delta": deltas["receiver_floor"] >= 0.0,
    }
    return _gate_result(candidate, control, kind="tailguard", checks=checks, deltas=deltas)


def evaluate_candidate_gate(
    candidate: Mapping[str, Any],
    control: Mapping[str, Any],
    *,
    kind: str = "mainline",
) -> dict[str, Any]:
    if kind == "mainline":
        return evaluate_mainline_gate(candidate, control)
    if kind == "tailguard":
        return evaluate_tailguard_gate(candidate, control)
    raise ValueError("kind must be mainline or tailguard")


def validate_four_scenario_closure(row_root: str | Path) -> dict[str, Any]:
    """Validate clean plus all three LEO artifacts without scoring a gate."""

    result = validate_artifact_closure(row_root)
    result["four_scenario_complete"] = result["complete"]
    result["technical_failure"] = not result["complete"]
    return result


validate_cv2_artifact_closure = validate_four_scenario_closure


def _is_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _as_exact_sequence(value: Any, *, name: str) -> tuple[Any, ...] | None:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        return None
    return tuple(value)


def _same_sequence(actual: Any, expected: Any, *, name: str) -> bool:
    actual_values = _as_exact_sequence(actual, name=name)
    expected_values = _as_exact_sequence(expected, name=name)
    if actual_values is None or expected_values is None:
        return actual == expected
    return actual_values == expected_values


def validate_checkpoint_runtime(
    runtime: Mapping[str, Any],
    expectation: Mapping[str, Any],
) -> dict[str, Any]:
    """Validate the complete source-only identity of one final checkpoint.

    ``expectation["optimizer_updates"]`` is the terminal update count.  The
    runtime records that value both as ``optimizer_update`` and
    ``total_updates``; accepting the spelling ``optimizer_updates`` for the
    latter keeps manually-created JSON fixtures readable without weakening
    the value check.
    """

    if not isinstance(runtime, Mapping):
        return {"valid": False, "missing": ["runtime"], "mismatches": []}
    if not isinstance(expectation, Mapping):
        raise TypeError("runtime expectation must be a mapping")

    missing: list[str] = []
    mismatches: list[str] = []

    def compare(
        name: str,
        actual_name: str,
        expected_value: Any,
        *,
        sequence: bool = False,
    ) -> None:
        if actual_name not in runtime:
            missing.append(name)
            return
        actual_value = runtime[actual_name]
        if sequence:
            matches = _same_sequence(actual_value, expected_value, name=name)
        elif _is_int(expected_value):
            matches = _is_int(actual_value) and actual_value == expected_value
        else:
            matches = type(actual_value) is type(expected_value) and actual_value == expected_value
        if not matches:
            mismatches.append(name)

    compare("phase1_method", "phase1_method", "bicad_xr")
    for name in ("candidate_id", "fold", "seed"):
        if name not in expectation:
            missing.append(f"expectation.{name}")
            continue
        compare(name, name, expectation[name])

    if "optimizer_updates" not in expectation:
        missing.append("expectation.optimizer_updates")
    else:
        updates = expectation["optimizer_updates"]
        compare("optimizer_update", "optimizer_update", updates)
        if "total_updates" in runtime:
            if not (_is_int(runtime["total_updates"]) and runtime["total_updates"] == updates):
                mismatches.append("optimizer_updates")
        elif "optimizer_updates" in runtime:
            if not (_is_int(runtime["optimizer_updates"]) and runtime["optimizer_updates"] == updates):
                mismatches.append("optimizer_updates")
        else:
            missing.append("optimizer_updates")

    if "planned_optimizer_updates" in expectation:
        compare(
            "planned_optimizer_updates",
            "planned_total_updates",
            expectation["planned_optimizer_updates"],
        )

    for name in ("source_receivers", "train_days"):
        if name not in expectation:
            missing.append(f"expectation.{name}")
            continue
        compare(name, name, expectation[name], sequence=True)

    required_flags = {
        "source_only": True,
        "target_access": False,
        "phase2_access": False,
        "support_access": False,
        "query_access": False,
        "truth_access": False,
    }
    for name, expected_value in required_flags.items():
        compare(name, name, expected_value)

    # Keep the return order stable for reports and exact negative tests.
    missing = list(dict.fromkeys(missing))
    mismatches = list(dict.fromkeys(mismatches))
    return {
        "valid": not missing and not mismatches,
        "missing": missing,
        "mismatches": mismatches,
    }


def _as_list(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return copy.deepcopy(value)
    if isinstance(value, tuple):
        return list(copy.deepcopy(value))
    if isinstance(value, (str, bytes)):
        return [value]
    try:
        return list(copy.deepcopy(value))
    except TypeError:
        return [copy.deepcopy(value)]


def _normalise_reconstruction(value: Any) -> dict[str, list[Any]]:
    if not isinstance(value, Mapping):
        return {"missing": [], "unexpected": [], "shape_mismatch": []}
    aliases = {
        "missing": ("missing", "missing_keys"),
        "unexpected": ("unexpected", "unexpected_keys"),
        "shape_mismatch": ("shape_mismatch", "shape_mismatches", "shape_mismatch_keys"),
    }
    result: dict[str, list[Any]] = {}
    for canonical, names in aliases.items():
        selected = []
        for name in names:
            if name in value:
                selected = _as_list(value[name])
                break
        result[canonical] = selected
    return result


def _failure_result(
    *,
    missing: list[str] | None = None,
    reconstruction: Mapping[str, Any] | None = None,
    evaluations: Mapping[str, Any] | None = None,
    reason: str = "INCOMPLETE",
) -> dict[str, Any]:
    normalized = _normalise_reconstruction(reconstruction)
    return {
        "complete": False,
        "status": reason,
        "missing": list(missing or []),
        "reconstruction": normalized,
        "evaluations": dict(evaluations or {}),
    }


def _read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _checkpoint_from_metadata(root: Path, metadata: Mapping[str, Any] | None) -> Path | None:
    if metadata is not None:
        name = metadata.get("checkpoint_path")
        if isinstance(name, str) and name.strip():
            candidate = Path(name)
            if not candidate.is_absolute():
                candidate = root / candidate
            return candidate
    for name in ("final_checkpoint.pt", "bicad_xr_final.pth", "final_bicad_xr.pt"):
        candidate = root / name
        if candidate.is_file():
            return candidate
    return None


def _metric_payload_is_valid(payload: Mapping[str, Any], scenario: str) -> bool:
    if payload.get("scenario", payload.get("scene")) != scenario:
        return False
    if payload.get("checkpoint_load_strict") is not True:
        return False
    for name in ("missing_keys", "unexpected_keys", "shape_mismatches"):
        if name not in payload or not isinstance(payload[name], list) or payload[name]:
            return False
    if not isinstance(payload.get("per_class_accuracy"), Mapping):
        return False
    for name in ("accuracy", "floor_accuracy"):
        if name not in payload:
            return False
    return True


def validate_artifact_closure(row_root: str | Path) -> dict[str, Any]:
    """Return a structured closure result without ever marking partial output complete."""

    root = Path(row_root)
    missing: list[str] = []
    evaluations: dict[str, Any] = {}
    metadata: Mapping[str, Any] | None = None

    runtime_path = root / "checkpoint_runtime.json"
    if not runtime_path.is_file():
        missing.append("checkpoint_runtime")
    else:
        try:
            loaded_metadata = _read_json(runtime_path)
        except (OSError, json.JSONDecodeError):
            loaded_metadata = None
        if isinstance(loaded_metadata, Mapping):
            metadata = loaded_metadata
        else:
            missing.append("checkpoint_runtime")

    checkpoint = _checkpoint_from_metadata(root, metadata)
    if checkpoint is None or not checkpoint.is_file() or checkpoint.stat().st_size <= 0:
        missing.append("final_checkpoint")
    elif metadata is not None:
        declared = metadata.get("checkpoint_path")
        if isinstance(declared, str) and Path(declared).name != checkpoint.name:
            missing.append("checkpoint_path")

    reconstruction = _normalise_reconstruction(
        metadata.get("reconstruction") if metadata is not None else None
    )
    if metadata is not None:
        if metadata.get("strict_reconstruction") is not True:
            missing.append("strict_reconstruction")
        if metadata.get("trainer_runtime_strict") is not True:
            missing.append("trainer_runtime_strict")
        for name in ("missing_keys", "unexpected_keys", "shape_mismatches"):
            if name not in metadata or not isinstance(metadata[name], list):
                missing.append(name)
            elif metadata[name]:
                reconstruction = _normalise_reconstruction(metadata)
    if any(reconstruction.values()):
        return _failure_result(
            missing=missing,
            reconstruction=reconstruction,
            reason="CHECKPOINT_RECONSTRUCTION_FAILED",
        )

    diagnostics_path = root / "diagnostics.json"
    if not diagnostics_path.is_file():
        missing.append("diagnostics")
    else:
        try:
            diagnostics = _read_json(diagnostics_path)
        except (OSError, json.JSONDecodeError):
            diagnostics = None
        if not isinstance(diagnostics, Mapping) or not set(DIAGNOSTIC_SCHEMA).issubset(diagnostics):
            missing.append("diagnostics")

    for scenario in FORMAL_EVAL_SCENARIOS:
        metric_path = root / "evaluations" / f"{scenario}.json"
        log_path = root / "evaluations" / f"{scenario}.log"
        if not metric_path.is_file() or not log_path.is_file():
            missing.append(scenario)
            continue
        try:
            payload = _read_json(metric_path)
        except (OSError, json.JSONDecodeError):
            missing.append(scenario)
            continue
        if not isinstance(payload, Mapping) or not _metric_payload_is_valid(payload, scenario):
            missing.append(scenario)
            continue
        try:
            if log_path.stat().st_size <= 0:
                missing.append(scenario)
                continue
        except OSError:
            missing.append(scenario)
            continue
        evaluations[scenario] = dict(payload)

    missing = list(dict.fromkeys(missing))
    if missing:
        return _failure_result(
            missing=missing,
            reconstruction=reconstruction,
            evaluations=evaluations,
        )
    return {
        "complete": True,
        "status": "ARTIFACTS_COMPLETE",
        "missing": [],
        "reconstruction": reconstruction,
        "evaluations": evaluations,
        "checkpoint": str(checkpoint),
    }


def _jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if hasattr(value, "item"):
        try:
            return _jsonable(value.item())
        except (AttributeError, TypeError, ValueError, RuntimeError):
            if hasattr(value, "tolist"):
                return _jsonable(value.tolist())
    return str(value)


def _write_json_once(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    serialized = json.dumps(_jsonable(payload), ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    if path.exists():
        try:
            existing = path.read_text(encoding="utf-8")
        except OSError as exc:
            raise FileExistsError(f"cannot read existing artifact: {path}") from exc
        if existing != serialized:
            raise FileExistsError(f"refusing to overwrite artifact: {path}")
        return
    with path.open("x", encoding="utf-8", newline="\n") as handle:
        handle.write(serialized)


def _strict_reconstruction(
    model: Any,
    state: Any,
) -> dict[str, list[Any]]:
    try:
        loaded = model.load_state_dict(state, strict=True)
    except Exception as exc:  # strict loading must fail closed, including shape errors.
        message = str(exc)
        if re.search(r"size mismatch|shape", message, flags=re.IGNORECASE):
            return {"missing": [], "unexpected": [], "shape_mismatch": [message]}
        return {"missing": [message], "unexpected": [], "shape_mismatch": []}

    if isinstance(loaded, Mapping):
        return _normalise_reconstruction(loaded)
    return _normalise_reconstruction(
        {
            "missing": getattr(loaded, "missing_keys", []),
            "unexpected": getattr(loaded, "unexpected_keys", []),
            "shape_mismatch": getattr(loaded, "shape_mismatches", []),
        }
    )


def _default_checkpoint_loader(path: Path) -> Mapping[str, Any]:
    import torch

    loaded = torch.load(path, map_location="cpu")
    if not isinstance(loaded, Mapping):
        raise TypeError("BiCAD-XR checkpoint must be a mapping")
    return loaded


def _parsed_csv(value: Any) -> list[Any]:
    output: list[Any] = []
    for item in str(value or "").split(","):
        token = item.strip()
        if not token:
            continue
        try:
            output.append(int(token))
        except ValueError:
            output.append(token)
    return output


def _runtime_with_recorded_args(payload: Mapping[str, Any]) -> Mapping[str, Any] | None:
    raw_runtime = payload.get("bicad_xr_runtime", payload.get("runtime"))
    if not isinstance(raw_runtime, Mapping):
        return None
    runtime = dict(raw_runtime)
    args = payload.get("args")
    if not isinstance(args, Mapping):
        return runtime
    row_record: Mapping[str, Any] = {}
    raw_row_key = args.get("row_key")
    if isinstance(raw_row_key, str) and raw_row_key.strip():
        try:
            loaded_row_record = json.loads(raw_row_key)
        except json.JSONDecodeError:
            loaded_row_record = None
        if isinstance(loaded_row_record, Mapping):
            row_record = loaded_row_record
    for name in ("fold", "optimizer_updates"):
        if name in row_record:
            runtime[name] = row_record[name]
    if "seed" in args:
        runtime["seed"] = args["seed"]
    runtime["source_receivers"] = _parsed_csv(args.get("wisig_train_rxs"))
    runtime["train_days"] = _parsed_csv(args.get("wisig_train_days"))
    runtime["run_id"] = args.get("run_id")
    runtime["row_id"] = row_record.get("row_id")
    runtime["output_dir"] = args.get("output_dir")
    return runtime


def evaluate_final_checkpoint(
    checkpoint_path: str | Path,
    *,
    expected_runtime: Mapping[str, Any],
    output_dir: str | Path | None = None,
    checkpoint_loader: Callable[[Path], Mapping[str, Any]] | None = None,
    model_builder: Callable[[Mapping[str, Any]], Any] | None = None,
    trainer_runtime_restorer: Callable[[Any, Mapping[str, Any]], Any] | None = None,
    evaluator: Callable[[Any, str], Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    """Strictly reconstruct one checkpoint and independently evaluate four scenes.

    The callbacks keep the helper usable by both the real launcher and small
    local tests.  ``evaluator`` is called once per formal scenario, in order,
    and receives no query truth or target-role information.
    """

    checkpoint = Path(checkpoint_path)
    root = Path(output_dir) if output_dir is not None else checkpoint.parent
    root.mkdir(parents=True, exist_ok=True)
    if not checkpoint.is_file() or checkpoint.stat().st_size <= 0:
        return _failure_result(missing=["final_checkpoint"])
    if checkpoint_loader is None:
        checkpoint_loader = _default_checkpoint_loader
    if model_builder is None or evaluator is None:
        raise TypeError("model_builder and evaluator are required")

    try:
        payload = checkpoint_loader(checkpoint)
    except Exception as exc:
        return _failure_result(reason=f"CHECKPOINT_LOAD_FAILED: {exc}")
    if not isinstance(payload, Mapping):
        return _failure_result(reason="CHECKPOINT_LOAD_FAILED")

    runtime = _runtime_with_recorded_args(payload)
    runtime_check = validate_checkpoint_runtime(runtime, expected_runtime)
    if not runtime_check["valid"]:
        return {
            **_failure_result(
                missing=list(runtime_check["missing"]),
                reason="CHECKPOINT_RUNTIME_MISMATCH",
            ),
            "runtime": runtime_check,
        }

    state = payload.get("model", payload.get("state_dict"))
    if state is None:
        return _failure_result(reason="CHECKPOINT_RECONSTRUCTION_FAILED")
    try:
        model = model_builder(payload)
    except Exception as exc:
        return _failure_result(reason=f"CHECKPOINT_MODEL_BUILD_FAILED: {exc}")
    reconstruction = _strict_reconstruction(model, state)
    if any(reconstruction.values()):
        return {
            **_failure_result(
                reconstruction=reconstruction,
                reason="CHECKPOINT_RECONSTRUCTION_FAILED",
            ),
            "runtime": runtime_check,
        }
    if trainer_runtime_restorer is None:
        return _failure_result(reason="TRAINER_RUNTIME_RESTORER_REQUIRED")
    try:
        trainer_runtime_restorer(model, payload)
    except Exception as exc:
        return {
            **_failure_result(
                reconstruction=reconstruction,
                reason=f"TRAINER_RUNTIME_RESTORE_FAILED: {exc}",
            ),
            "runtime": runtime_check,
        }

    diagnostics = BiCADXRMetricStore().snapshot()
    _write_json_once(
        root / "checkpoint_runtime.json",
        {
            "checkpoint_path": checkpoint.name,
            "runtime": runtime,
            "reconstruction": reconstruction,
            "strict_reconstruction": True,
            "trainer_runtime_strict": True,
            "missing_keys": [],
            "unexpected_keys": [],
            "shape_mismatches": [],
        },
    )
    _write_json_once(root / "diagnostics.json", diagnostics)

    evaluations: dict[str, Any] = {}
    for scenario in FORMAL_EVAL_SCENARIOS:
        try:
            metrics = evaluator(model, scenario)
            if not isinstance(metrics, Mapping):
                raise TypeError("evaluator must return a mapping")
            metrics_payload = dict(metrics)
            log_value = metrics_payload.pop("log", None)
            result_payload = {
                **metrics_payload,
                "scenario": scenario,
                "checkpoint": checkpoint.name,
                "checkpoint_load_strict": True,
                "missing_keys": [],
                "unexpected_keys": [],
                "shape_mismatches": [],
            }
            _write_json_once(root / "evaluations" / f"{scenario}.json", result_payload)
            if log_value is None:
                log_value = f"scenario={scenario} checkpoint={checkpoint.name}\n"
            log_path = root / "evaluations" / f"{scenario}.log"
            if log_path.exists():
                existing = log_path.read_text(encoding="utf-8")
                if existing != str(log_value):
                    raise FileExistsError(f"refusing to overwrite artifact: {log_path}")
            else:
                log_path.parent.mkdir(parents=True, exist_ok=True)
                with log_path.open("x", encoding="utf-8", newline="\n") as handle:
                    handle.write(str(log_value))
            evaluations[scenario] = result_payload
        except Exception as exc:
            remaining = list(FORMAL_EVAL_SCENARIOS[FORMAL_EVAL_SCENARIOS.index(scenario):])
            return {
                **_failure_result(
                    missing=remaining,
                    reconstruction=reconstruction,
                    evaluations=evaluations,
                    reason=f"EVALUATION_FAILED: {exc}",
                ),
                "runtime": runtime_check,
            }

    result = validate_artifact_closure(root)
    result["runtime"] = runtime_check
    return result


__all__ = [
    "DIAGNOSTIC_SCHEMA",
    "FORMAL_EVAL_SCENARIOS",
    "LEO_WEAK_SCENARIOS",
    "BiCADXRMetricStore",
    "S_DG",
    "aggregate_s_dg",
    "compute_s_dg",
    "evaluate_final_checkpoint",
    "evaluate_candidate_gate",
    "evaluate_mainline_gate",
    "evaluate_tailguard_gate",
    "validate_artifact_closure",
    "validate_cv2_artifact_closure",
    "validate_checkpoint_runtime",
    "validate_four_scenario_closure",
    "s_dg",
]
