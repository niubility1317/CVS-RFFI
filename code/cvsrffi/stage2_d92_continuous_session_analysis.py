"""Truth-last audit for immutable D92 continuous-session predictions.

The predictor never imports this module.  This reader first closes every
prediction/receipt/commit binding, and only then opens the scorer-side truth
surface to construct the per-session trajectory.
"""

from __future__ import annotations

import hashlib
import json
import os
import stat
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np


SCHEMA = "cvs.phase2.d92_e0_continuous_session.truth_last_analysis.v1"
PREDICTION_SCHEMA = "cvs.phase2.d92_e0_continuous_session.truth_free_prediction.v1"
SCENES = ("leo_clear_weak", "leo_low_elev_weak", "leo_rain_weak")
SCHEDULES = ("batch_5", "singleton_forward", "singleton_reverse", "chunk_2_2_1")
EIGHT_METRICS = (
    "h_old_new",
    "old_balanced_accuracy",
    "c_old_acc",
    "old_floor",
    "seen_new_acc",
    "average_forgetting",
    "new_to_old_rate",
    "old_to_new_rate",
)
_WRITE_BITS = stat.S_IWUSR | stat.S_IWGRP | stat.S_IWOTH


class ContinuousSessionAnalysisError(ValueError):
    """Raised when immutable prediction or truth-last closure drifts."""


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value, ensure_ascii=True, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _fail(message: str) -> ContinuousSessionAnalysisError:
    return ContinuousSessionAnalysisError(message)


def _read_json(path: Path, *, label: str) -> dict[str, Any]:
    try:
        info = path.lstat()
    except FileNotFoundError as error:
        raise _fail(f"{label} is missing") from error
    if path.is_symlink() or not stat.S_ISREG(info.st_mode):
        raise _fail(f"{label} must be a regular file")
    try:
        payload = json.loads(path.read_bytes().decode("utf-8-sig"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise _fail(f"{label} is invalid") from error
    if not isinstance(payload, dict):
        raise _fail(f"{label} must be an object")
    return payload


def _text_vector(value: Any, *, label: str) -> tuple[str, ...]:
    array = np.asarray(value)
    if array.ndim != 1 or len(array) == 0 or array.dtype.kind not in {"U", "S", "O"}:
        raise _fail(f"{label} must be a nonempty text vector")
    result = tuple(str(item) for item in array.tolist())
    if any(not item or "\x00" in item for item in result):
        raise _fail(f"{label} contains an invalid value")
    return result


def _digest_values(values: Sequence[str]) -> str:
    return hashlib.sha256(_canonical_bytes(list(values))).hexdigest()


def _readonly(path: Path) -> bool:
    info = path.lstat()
    return path.is_file() and not path.is_symlink() and not (stat.S_IMODE(info.st_mode) & _WRITE_BITS)


def _artifact_state(root: Path, *, label: str) -> dict[str, Any]:
    if root.is_symlink() or not root.is_dir():
        raise _fail(f"{label} state root drift")
    commit = _read_json(root / "COMMIT.json", label=f"{label} commit")
    members = commit.get("members")
    if not isinstance(members, list) or not members:
        raise _fail(f"{label} commit member drift")
    if commit.get("artifact_root_sha256") != hashlib.sha256(_canonical_bytes(members)).hexdigest():
        raise _fail(f"{label} commit root SHA drift")
    index: dict[str, Mapping[str, Any]] = {}
    for item in members:
        if not isinstance(item, Mapping):
            raise _fail(f"{label} commit member schema drift")
        name = item.get("relative_path")
        if not isinstance(name, str) or Path(name).name != name or name in index:
            raise _fail(f"{label} commit member path drift")
        index[name] = item
    required = {
        "prediction_artifact.npz",
        "fit_audit.json",
        "resource_audit.json",
        "execution_receipt.json",
    }
    if not required.issubset(index):
        raise _fail(f"{label} commit closure drift")
    for name, item in index.items():
        path = root / name
        if path.parent != root or not _readonly(path):
            raise _fail(f"{label} commit member is mutable or unsafe")
        if int(item.get("size_bytes", -1)) != path.stat().st_size or item.get("sha256") != _sha(path):
            raise _fail(f"{label} commit member digest drift")
    artifact = root / "prediction_artifact.npz"
    try:
        with np.load(artifact, allow_pickle=False) as archive:
            if set(archive.files) != {"query_tokens", "scenarios", "predicted_class_handles"}:
                raise _fail(f"{label} prediction schema drift")
            query_tokens = _text_vector(archive["query_tokens"], label=f"{label} query tokens")
            scenarios = _text_vector(archive["scenarios"], label=f"{label} query scenarios")
            predictions = _text_vector(
                archive["predicted_class_handles"], label=f"{label} predictions"
            )
    except ContinuousSessionAnalysisError:
        raise
    except Exception as error:
        raise _fail(f"{label} prediction artifact is unreadable") from error
    if len(query_tokens) != len(scenarios) or len(query_tokens) != len(predictions):
        raise _fail(f"{label} prediction alignment drift")
    if len(set(zip(scenarios, query_tokens))) != len(query_tokens):
        raise _fail(f"{label} duplicate query key")
    fit = _read_json(root / "fit_audit.json", label=f"{label} fit audit")
    resource = _read_json(root / "resource_audit.json", label=f"{label} resource audit")
    receipt = _read_json(root / "execution_receipt.json", label=f"{label} execution receipt")
    state_sha = receipt.get("state_sha256")
    if not isinstance(state_sha, str) or len(state_sha) != 64 or state_sha.lower() != state_sha:
        raise _fail(f"{label} state SHA drift")
    classes = tuple(str(item) for item in receipt.get("registered_classes", ()))
    old_count = receipt.get("registered_class_count")
    if (
        len(classes) < 6
        or len(set(classes)) != len(classes)
        or any(not item for item in classes)
        or int(old_count) != len(classes)
        or fit.get("state_sha256") != state_sha
        or tuple(str(item) for item in fit.get("registered_classes", ())) != classes
        or int(fit.get("old_class_count", -1)) != 6
        or receipt.get("prediction_artifact_sha256") != _sha(artifact)
        or receipt.get("fit_audit_sha256") != _sha(root / "fit_audit.json")
        or receipt.get("resource_audit_sha256") != _sha(root / "resource_audit.json")
        or receipt.get("query_token_sha256") != _digest_values(query_tokens)
        or receipt.get("query_scenario_sha256") != _digest_values(scenarios)
        or commit.get("prediction_artifact_sha256") != _sha(artifact)
        or commit.get("execution_receipt_sha256") != _sha(root / "execution_receipt.json")
    ):
        raise _fail(f"{label} receipt/token binding drift")
    if any(prediction not in classes for prediction in predictions):
        raise _fail(f"{label} prediction class closure drift")
    for field in (
        "registration_wall_time_ns",
        "registration_incremental_peak_working_set_bytes",
        "support_bytes",
        "state_bytes",
        "query_macs",
        "head_latency_ns",
    ):
        try:
            value = int(resource[field])
        except (KeyError, TypeError, ValueError) as error:
            raise _fail(f"{label} resource field drift: {field}") from error
        if value < 0:
            raise _fail(f"{label} resource field is negative: {field}")
    if (
        int(resource["registration_wall_time_ns"]) > 150_000_000
        or int(resource["registration_incremental_peak_working_set_bytes"]) > 4 * 1024 * 1024
        or int(resource["query_macs"]) != len(classes) * 288
    ):
        raise _fail(f"{label} resource hard gate drift")
    return {
        "root": root,
        "query_tokens": query_tokens,
        "scenarios": scenarios,
        "predictions": predictions,
        "state_sha256": state_sha,
        "prediction_artifact_sha256": _sha(artifact),
        "lifecycle_state": receipt.get("lifecycle_state"),
        "session_index": int(receipt.get("session_index", -1)),
        "registered_classes": classes,
        "resource": resource,
    }


def _prediction_manifest(job: Mapping[str, Any], output_root: Path) -> Path:
    outer = str(job.get("outer_key", ""))
    candidates = (
        Path(str(job.get("output_root", ""))) / "full" / "prediction_manifest.json",
        output_root / "jobs" / outer / "full" / "prediction_manifest.json",
        output_root / "full" / "prediction_manifest.json",
        output_root / "prediction_manifest.json",
    )
    found = {
        path.resolve() for path in candidates if path.is_file() and not path.is_symlink()
    }
    if len(found) != 1:
        raise _fail(f"prediction manifest location drift for {outer}")
    return next(iter(found))


def _collect_prediction(job: Mapping[str, Any], output_root: Path) -> dict[str, Any]:
    manifest_path = _prediction_manifest(job, output_root)
    manifest = _read_json(manifest_path, label="prediction manifest")
    if manifest.get("schema") != PREDICTION_SCHEMA or not isinstance(manifest.get("scenes"), Mapping):
        raise _fail("prediction manifest schema drift")
    scenes = manifest["scenes"]
    if set(scenes) != set(SCENES):
        raise _fail("prediction scene closure drift")
    result: dict[str, Any] = {"manifest_path": str(manifest_path), "scenes": {}}
    for scene in SCENES:
        surface = scenes[scene]
        if not isinstance(surface, Mapping) or not isinstance(surface.get("DA1_REG0"), Mapping):
            raise _fail(f"prediction baseline closure drift: {scene}")
        baseline_ref = surface["DA1_REG0"]
        baseline = _artifact_state(Path(str(baseline_ref.get("output_root", ""))), label=f"{scene}/DA1_REG0")
        if baseline["lifecycle_state"] != "DA1_REG0" or baseline["session_index"] != 0:
            raise _fail(f"prediction baseline lifecycle drift: {scene}")
        schedules = surface.get("schedules")
        if not isinstance(schedules, Mapping) or set(schedules) != set(SCHEDULES):
            raise _fail(f"prediction schedule closure drift: {scene}")
        schedule_rows: dict[str, list[dict[str, Any]]] = {}
        for schedule in SCHEDULES:
            payload = schedules[schedule]
            sessions = payload.get("sessions") if isinstance(payload, Mapping) else None
            if not isinstance(sessions, list) or not sessions:
                raise _fail(f"prediction session closure drift: {scene}/{schedule}")
            rows: list[dict[str, Any]] = []
            for index, reference in enumerate(sessions, start=1):
                if not isinstance(reference, Mapping):
                    raise _fail(f"prediction session reference drift: {scene}/{schedule}")
                row = _artifact_state(
                    Path(str(reference.get("output_root", ""))),
                    label=f"{scene}/{schedule}/S{index}",
                )
                if row["lifecycle_state"] != f"DA1_REG1_S{index}" or row["session_index"] != index:
                    raise _fail(f"prediction session lifecycle drift: {scene}/{schedule}/S{index}")
                for field in ("state_sha256", "prediction_artifact_sha256"):
                    if reference.get(field) != row[field]:
                        raise _fail(f"prediction manifest/receipt drift: {scene}/{schedule}/{field}")
                rows.append(row)
            schedule_rows[schedule] = rows
        result["scenes"][scene] = {"baseline": baseline, "schedules": schedule_rows}
    return result


def _truth_path(root: Path, outer: str) -> Path:
    candidates = (
        root / "jobs" / outer / "offline" / "scorer" / "truth_sidecar.json",
        root / outer / "truth_sidecar.json",
        root / "truth_sidecar.json",
    )
    found = {
        path.resolve() for path in candidates if path.is_file() and not path.is_symlink()
    }
    if len(found) != 1:
        raise _fail(f"truth sidecar location drift for {outer}")
    return next(iter(found))


def _truth_rows(path: Path) -> Mapping[str, Mapping[str, str]]:
    payload = _read_json(path, label="truth sidecar")
    rows = payload.get("rows")
    if payload.get("schema") != "cvs.phase2.query_truth_sidecar.v2" or not isinstance(rows, list):
        raise _fail("truth sidecar schema drift")
    result: dict[str, Mapping[str, str]] = {}
    for row in rows:
        if not isinstance(row, Mapping):
            raise _fail("truth sidecar row drift")
        token = str(row.get("query_token", ""))
        handle = str(row.get("true_class_handle", ""))
        if not token or not handle or token in result:
            raise _fail("truth token/handle drift")
        result[token] = {"true_class_handle": handle}
    if not result:
        raise _fail("truth sidecar is empty")
    return result


def _mean(values: Sequence[float]) -> float:
    if not values:
        raise _fail("metric denominator is empty")
    return float(sum(values) / len(values))


def _harmonic(left: float, right: float) -> float:
    return 0.0 if left + right <= 0.0 else float(2.0 * left * right / (left + right))


def _score_state(
    state: Mapping[str, Any], truth: Mapping[str, Mapping[str, str]], *, baseline_old_ba: float | None
) -> dict[str, Any]:
    classes = tuple(state["registered_classes"])
    old = set(classes[:6])
    registered_new = set(classes[6:])
    old_correct: list[int] = []
    new_correct: list[int] = []
    old_to_new: list[int] = []
    new_to_old: list[int] = []
    by_old: dict[str, list[int]] = defaultdict(list)
    unregistered: list[str] = []
    for token, predicted in zip(state["query_tokens"], state["predictions"]):
        truth_row = truth.get(token)
        if truth_row is None:
            raise _fail("prediction token is absent from truth sidecar")
        actual = truth_row["true_class_handle"]
        if actual not in classes:
            unregistered.append(token)
            continue
        correct = int(predicted == actual)
        if actual in old:
            old_correct.append(correct)
            old_to_new.append(int(predicted in registered_new))
            by_old[actual].append(correct)
        elif actual in registered_new:
            new_correct.append(correct)
            new_to_old.append(int(predicted in old))
        else:  # classes have exactly old then registered-new
            raise _fail("registered truth role drift")
    old_class_accuracy = {name: _mean(values) for name, values in sorted(by_old.items())}
    old_acc = _mean([float(value) for value in old_correct])
    old_ba = _mean(list(old_class_accuracy.values()))
    seen_new = _mean([float(value) for value in new_correct]) if new_correct else None
    metrics: dict[str, float | None] = {
        "h_old_new": _harmonic(old_acc, seen_new) if seen_new is not None else None,
        "old_balanced_accuracy": old_ba,
        "c_old_acc": old_acc,
        "old_floor": min(old_class_accuracy.values()),
        "seen_new_acc": seen_new,
        "average_forgetting": 0.0 if baseline_old_ba is None else baseline_old_ba - old_ba,
        "new_to_old_rate": _mean([float(value) for value in new_to_old]) if new_to_old else None,
        "old_to_new_rate": _mean([float(value) for value in old_to_new]),
    }
    return {
        **metrics,
        "registered_new_accuracy": seen_new,
        "registered_new_query_count": len(new_correct),
        "unregistered_truth_count": len(unregistered),
        "unregistered_truth_status": (
            "UNREGISTERED_NOT_SCORED" if unregistered else "ALL_REGISTERED_SCORED"
        ),
        "unregistered_truth_tokens": sorted(unregistered),
        "old_class_accuracy": old_class_accuracy,
        "resource": dict(state["resource"]),
        "state_sha256": state["state_sha256"],
        "prediction_artifact_sha256": state["prediction_artifact_sha256"],
    }


def _terminal_equivalence(rows: Mapping[str, Sequence[Mapping[str, Any]]]) -> dict[str, Any]:
    reference = rows["batch_5"][-1]
    for schedule in SCHEDULES:
        candidate = rows[schedule][-1]
        if (
            candidate["state_sha256"] != reference["state_sha256"]
            or candidate["prediction_artifact_sha256"] != reference["prediction_artifact_sha256"]
            or any(candidate[name] != reference[name] for name in EIGHT_METRICS)
        ):
            raise _fail("terminal state/prediction/eight-metric equivalence drift")
    return {
        "status": "STRICT_EQUAL",
        "state_sha256": reference["state_sha256"],
        "prediction_artifact_sha256": reference["prediction_artifact_sha256"],
        "metrics": {name: reference[name] for name in EIGHT_METRICS},
    }


def _write_analysis(path: Path, value: Mapping[str, Any]) -> str:
    raw = _canonical_bytes(value) + b"\n"
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_BINARY", 0)
    descriptor = os.open(path, flags, 0o444)
    try:
        os.write(descriptor, raw)
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    try:
        os.chmod(path, stat.S_IREAD)
    except OSError:
        pass
    return hashlib.sha256(raw).hexdigest()


def analyze_continuous_session_run(
    manifest_path: str | Path,
    output_root: str | Path,
    truth_root: str | Path,
    analysis_root: str | Path,
) -> dict[str, Any]:
    """Validate predictions first, then join truth in the offline process only."""

    matrix = _read_json(Path(manifest_path), label="matrix manifest")
    jobs = matrix.get("jobs")
    if not isinstance(jobs, list) or not jobs or any(not isinstance(job, Mapping) for job in jobs):
        raise _fail("matrix job closure drift")
    output = Path(output_root)
    truth_base = Path(truth_root)
    destination = Path(analysis_root)
    if destination.is_symlink() or (destination.exists() and (not destination.is_dir() or any(destination.iterdir()))):
        raise FileExistsError("analysis output already exists")
    if not destination.exists():
        destination.mkdir(parents=True, exist_ok=False)

    # No truth is opened until this entire immutable-prediction collection closes.
    collected: list[tuple[Mapping[str, Any], dict[str, Any]]] = [
        (job, _collect_prediction(job, output)) for job in jobs
    ]
    trajectories: dict[str, Any] = {}
    terminal: dict[str, Any] = {}
    for job, prediction in collected:
        outer = str(job.get("outer_key", ""))
        if not outer or outer in trajectories:
            raise _fail("matrix outer identity drift")
        truth = _truth_rows(_truth_path(truth_base, outer))
        trajectories[outer] = {}
        terminal[outer] = {}
        for scene in SCENES:
            surface = prediction["scenes"][scene]
            baseline_score = _score_state(surface["baseline"], truth, baseline_old_ba=None)
            schedule_scores = {
                name: [
                    _score_state(row, truth, baseline_old_ba=baseline_score["old_balanced_accuracy"])
                    for row in rows
                ]
                for name, rows in surface["schedules"].items()
            }
            trajectories[outer][scene] = {"DA1_REG0": baseline_score, **schedule_scores}
            terminal[outer][scene] = _terminal_equivalence(schedule_scores)
    result = {
        "schema": SCHEMA,
        "status": "ANALYZED_TRUTH_LAST",
        "prediction_validation_complete_before_truth_open": True,
        "truth_sidecar_exposed_to_predictor": False,
        "trajectories": trajectories,
        "terminal_equivalence": terminal,
    }
    analysis_sha = _write_analysis(destination / "continuous_session_analysis.json", result)
    return {**result, "analysis_sha256": analysis_sha}


__all__ = [
    "ContinuousSessionAnalysisError",
    "EIGHT_METRICS",
    "SCHEMA",
    "analyze_continuous_session_run",
]
