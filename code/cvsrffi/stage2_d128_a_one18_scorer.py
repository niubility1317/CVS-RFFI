"""Independent scorer-side closure for the A-only D128-A-ONE18 falsifier.

The predictor remains entirely truth-free.  This module first validates the
sealed D128 prediction and writes/validates a durable truth-open event.  Only
after that boundary may it open the D92 retry2 scorer-side truth and formal
reference files.  It reports the three frozen direction criteria but never
promotes, reruns, or changes the candidate.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any, Mapping, Sequence

from . import stage2_d127_s0_package_adapter as adapter
from . import stage2_d127_s0_scorer as d127_scorer
from . import stage2_d127_s0_truth_assets as d92_assets
from . import stage2_d128_a_one18 as one


NORMALIZED_SCHEMA = "cvs.stage2.d128.a.one18.normalized_prediction.v1"
TRUTH_OPEN_EVENT_SCHEMA = "cvs.stage2.d128.a.one18.truth_open_event.v1"
TRUTH_CATALOG_SCHEMA = "cvs.stage2.d128.a.one18.truth_catalog.v1"
FORMAL_D92_REFERENCE_SCHEMA = "cvs.stage2.d128.a.one18.formal_d92_reference.v1"
SCORE_SCHEMA = "cvs.stage2.d128.a.one18.score_manifest.v1"
TRUTH_ASSETS_RECEIPT_SCHEMA = "cvs.stage2.d128.a.one18.truth_assets_receipt.v1"
D92_PIPELINE_ROOT_SCHEMA = "cvs.stage2.d128.a.one18.d92_pipeline_root.v1"


class D128AOne18ScorerError(ValueError):
    """Raised when a D128 scorer-side artifact is incomplete or mismatched."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise D128AOne18ScorerError(message)


def canonical_sha256(value: Any) -> str:
    return one.canonical_sha256(value)


def _sha(value: Any, name: str) -> str:
    try:
        return one._sha(value, name)
    except Exception as exc:
        raise D128AOne18ScorerError(str(exc)) from exc


def _text(value: Any, name: str) -> str:
    _require(type(value) is str and bool(value), f"{name} must be nonempty text")
    return value


def _integer(value: Any, name: str, *, minimum: int = 0) -> int:
    _require(type(value) is int and value >= minimum, f"{name} must be an integer >= {minimum}")
    return value


def _strings(value: Any, name: str, *, unique: bool = True) -> tuple[str, ...]:
    _require(isinstance(value, list) and bool(value), f"{name} must be a nonempty list")
    result = tuple(_text(item, f"{name}[{index}]") for index, item in enumerate(value))
    _require(not unique or len(result) == len(set(result)), f"{name} contains duplicate values")
    return result


def _exact(value: Any, expected: set[str], name: str) -> Mapping[str, Any]:
    _require(isinstance(value, Mapping) and set(value) == expected, f"{name} field closure drift")
    return value


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1_048_576), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_pinned_json(path: str | Path, *, expected_sha256: str, name: str) -> tuple[dict[str, Any], str]:
    source = Path(path)
    _require(source.is_file() and not source.is_symlink(), f"{name} must be a regular file")
    expected = _sha(expected_sha256, f"expected {name} file SHA256")
    observed = _sha256_file(source)
    _require(observed == expected, f"{name} file SHA256 mismatch")
    try:
        value = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise D128AOne18ScorerError(f"{name} is not valid UTF-8 JSON") from exc
    _require(type(value) is dict, f"{name} must contain a JSON object")
    return value, observed


def _write_exclusive_json(path: str | Path, payload: Mapping[str, Any], *, name: str) -> Path:
    target = Path(path)
    _require(not target.is_symlink(), f"{name} output cannot be a symlink")
    target.parent.mkdir(parents=True, exist_ok=True)
    try:
        with target.open("xb") as stream:
            stream.write(one.canonical_bytes(dict(payload)) + b"\n")
            stream.flush()
            os.fsync(stream.fileno())
    except FileExistsError as exc:
        raise D128AOne18ScorerError(f"{name} output already exists") from exc
    return target


def _assert_output_absent(path: str | Path, name: str) -> Path:
    target = Path(path)
    _require(not target.is_symlink(), f"{name} output cannot be a symlink")
    _require(not target.exists(), f"{name} output already exists")
    return target


def _validate_normalized(value: Any) -> dict[str, Any]:
    expected_fields = {
        "schema", "prediction_sha256", "prepared_plan_sha256", "method_lock_sha256", "checkpoint_sha256",
        "pair_manifest_sha256", "candidate_id", "arm_ids", "row_count", "after_query_id_root_sha256", "rows",
        "normalized_prediction_sha256",
    }
    document = _exact(value, expected_fields, "D128 normalized prediction")
    unsigned = dict(document)
    receipt = _sha(unsigned.pop("normalized_prediction_sha256"), "D128 normalized prediction receipt")
    _require(canonical_sha256(unsigned) == receipt, "D128 normalized prediction digest drift")
    _require(
        document["schema"] == NORMALIZED_SCHEMA
        and document["candidate_id"] == one.CANDIDATE_ID
        and document["arm_ids"] == list(one.ARM_IDS)
        and document["row_count"] == one.ROW_COUNT,
        "D128 normalized prediction closure drift",
    )
    for field in ("prediction_sha256", "prepared_plan_sha256", "method_lock_sha256", "checkpoint_sha256", "pair_manifest_sha256", "after_query_id_root_sha256"):
        _sha(document[field], f"D128 normalized prediction {field}")
    rows = document["rows"]
    _require(isinstance(rows, list) and len(rows) == one.ROW_COUNT, "D128 normalized row coverage drift")
    expected_row_fields = {
        "row_id", "receiver_id", "k_shot", "scene", "old_classes", "new_classes", "before_query_ids",
        "after_query_ids", "before_arms", "after_arms", "formal_d92_source_job_id", "formal_d92_retry2_manifest_sha256",
    }
    all_after: list[str] = []
    coverage: set[tuple[str, int, str]] = set()
    for index, raw in enumerate(rows):
        row = _exact(raw, expected_row_fields, f"D128 normalized row[{index}]")
        identity = (_text(row["receiver_id"], "normalized receiver"), _integer(row["k_shot"], "normalized K-shot", minimum=1), _text(row["scene"], "normalized scene"))
        _require(identity not in coverage and identity[1] in {1, 5}, f"D128 normalized row[{index}] identity drift")
        coverage.add(identity)
        _text(row["row_id"], "normalized row ID")
        old_classes = _strings(row["old_classes"], f"normalized row[{index}] old classes")
        new_classes = _strings(row["new_classes"], f"normalized row[{index}] new classes")
        _require(not set(old_classes).intersection(new_classes), f"normalized row[{index}] old/new class overlap")
        before_ids = _strings(row["before_query_ids"], f"normalized row[{index}] before query IDs")
        after_ids = _strings(row["after_query_ids"], f"normalized row[{index}] after query IDs")
        _require(one._ordered_subset(before_ids, after_ids), f"normalized row[{index}] before-query subset drift")
        classes_by_state = {"before": old_classes, "after": (*old_classes, *new_classes)}
        for state, arms in (("before", row["before_arms"]), ("after", row["after_arms"])):
            _require(isinstance(arms, Mapping) and set(arms) == set(one.ARM_IDS), f"normalized row[{index}] {state} arm closure drift")
            query_count = len(before_ids) if state == "before" else len(after_ids)
            for arm_id in one.ARM_IDS:
                predictions = _strings(arms[arm_id], f"normalized row[{index}] {state}.{arm_id}", unique=False)
                _require(len(predictions) == query_count and all(item in classes_by_state[state] for item in predictions), f"normalized row[{index}] {state}.{arm_id} prediction closure drift")
        _text(row["formal_d92_source_job_id"], f"normalized row[{index}] D92 source job")
        _sha(row["formal_d92_retry2_manifest_sha256"], f"normalized row[{index}] D92 manifest SHA256")
        all_after.extend(after_ids)
    _require(len(all_after) == len(set(all_after)), "D128 normalized after query IDs are duplicated")
    _require(one._opaque_root(all_after) == document["after_query_id_root_sha256"], "D128 normalized after query root drift")
    expected_coverage = {(receiver, k, scene) for receiver in {item[0] for item in coverage} for k in (1, 5) for scene in {item[2] for item in coverage}}
    _require(len({item[0] for item in coverage}) == 3 and len({item[2] for item in coverage}) == 3 and coverage == expected_coverage, "D128 normalized 18-row coverage drift")
    return dict(document)


def _normalize_prediction(prediction: Mapping[str, Any], *, prepared_plan: Mapping[str, Any]) -> dict[str, Any]:
    one.validate_d128_a_one18_prediction(prediction, prepared_plan=prepared_plan)
    pair_manifest = prediction["pair_manifest"]
    rows: list[dict[str, Any]] = []
    after_ids_all: list[str] = []
    for manifest_row, before, after in zip(pair_manifest["rows"], prediction["states"]["before"]["rows"], prediction["states"]["after"]["rows"], strict=True):
        before_ids = list(before["opaque_query_ids"])
        after_ids = list(after["opaque_query_ids"])
        _require(
            canonical_sha256(before_ids) == manifest_row["before_query_ids_sha256"]
            and canonical_sha256(after_ids) == manifest_row["after_query_ids_sha256"],
            "D128 scorer normalization query-hash drift",
        )
        rows.append(
            {
                "row_id": manifest_row["row_id"],
                "receiver_id": manifest_row["receiver_id"],
                "k_shot": manifest_row["k_shot"],
                "scene": manifest_row["scene"],
                "old_classes": list(manifest_row["old_classes"]),
                "new_classes": list(manifest_row["new_classes"]),
                "before_query_ids": before_ids,
                "after_query_ids": after_ids,
                "before_arms": {arm_id: list(before["arms"][arm_id]["predictions"]) for arm_id in one.ARM_IDS},
                "after_arms": {arm_id: list(after["arms"][arm_id]["predictions"]) for arm_id in one.ARM_IDS},
                "formal_d92_source_job_id": manifest_row["formal_d92_source_job_id"],
                "formal_d92_retry2_manifest_sha256": manifest_row["formal_d92_retry2_manifest_sha256"],
            }
        )
        after_ids_all.extend(after_ids)
    normalized: dict[str, Any] = {
        "schema": NORMALIZED_SCHEMA,
        "prediction_sha256": prediction["prediction_sha256"],
        "prepared_plan_sha256": prediction["prepared_plan_sha256"],
        "method_lock_sha256": prediction["method_lock_sha256"],
        "checkpoint_sha256": prediction["checkpoint_sha256"],
        "pair_manifest_sha256": pair_manifest["pair_manifest_sha256"],
        "candidate_id": one.CANDIDATE_ID,
        "arm_ids": list(one.ARM_IDS),
        "row_count": one.ROW_COUNT,
        "after_query_id_root_sha256": one._opaque_root(after_ids_all),
        "rows": rows,
    }
    normalized["normalized_prediction_sha256"] = canonical_sha256(normalized)
    return _validate_normalized(normalized)


def prepare_d128_a_one18_scoring_inputs(
    *,
    prediction_path: str | Path,
    expected_prediction_sha256: str,
    prepared_plan_path: str | Path,
    expected_prepared_plan_sha256: str,
    method_lock_path: str | Path,
    expected_method_lock_sha256: str,
) -> dict[str, Any]:
    """Perform every truth-free check required before a truth-open event."""

    try:
        plan, _plan_file_sha = adapter.load_d127_s0_prepared_plan(
            prepared_plan_path, expected_sha256=expected_prepared_plan_sha256
        )
        prediction, prediction_file_sha = one.load_d128_a_one18_prediction(
            prediction_path, expected_sha256=expected_prediction_sha256, prepared_plan=plan
        )
        method_lock, method_lock_file_sha, _locks = adapter.load_d127_s0_method_lock(
            method_lock_path, expected_sha256=expected_method_lock_sha256
        )
    except Exception as exc:
        raise D128AOne18ScorerError("D128 truth-free scoring input closure failed") from exc
    _require(
        method_lock_file_sha == plan["method_lock_sha256"] == prediction["method_lock_sha256"],
        "D128 method-lock/prediction/plan drift",
    )
    _require(method_lock.get("checkpoint", {}).get("sha256") == prediction["checkpoint_sha256"], "D128 method-lock checkpoint drift")
    normalized = _normalize_prediction(prediction, prepared_plan=plan)
    return {
        "status": "D128_A_ONE18_TRUTH_FREE_SCORING_INPUTS_PREPARED",
        "prediction_file_sha256": prediction_file_sha,
        "normalized_prediction": normalized,
        "method_lock": method_lock,
    }


def build_d128_a_one18_truth_open_event(normalized_prediction: Mapping[str, Any]) -> dict[str, Any]:
    normalized = _validate_normalized(normalized_prediction)
    event: dict[str, Any] = {
        "schema": TRUTH_OPEN_EVENT_SCHEMA,
        "truth_open": True,
        "candidate_id": one.CANDIDATE_ID,
        "prediction_sha256": normalized["prediction_sha256"],
        "prepared_plan_sha256": normalized["prepared_plan_sha256"],
        "method_lock_sha256": normalized["method_lock_sha256"],
        "checkpoint_sha256": normalized["checkpoint_sha256"],
        "pair_manifest_sha256": normalized["pair_manifest_sha256"],
        "normalized_prediction_sha256": normalized["normalized_prediction_sha256"],
        "row_count": one.ROW_COUNT,
    }
    event["truth_open_event_sha256"] = canonical_sha256(event)
    return event


def _validate_truth_open_event(event: Any, *, normalized_prediction: Mapping[str, Any]) -> dict[str, Any]:
    normalized = _validate_normalized(normalized_prediction)
    expected_fields = {
        "schema", "truth_open", "candidate_id", "prediction_sha256", "prepared_plan_sha256", "method_lock_sha256",
        "checkpoint_sha256", "pair_manifest_sha256", "normalized_prediction_sha256", "row_count", "truth_open_event_sha256",
    }
    document = _exact(event, expected_fields, "D128 truth-open event")
    unsigned = dict(document)
    receipt = _sha(unsigned.pop("truth_open_event_sha256"), "D128 truth-open event receipt")
    _require(canonical_sha256(unsigned) == receipt, "D128 truth-open event digest drift")
    _require(
        document["schema"] == TRUTH_OPEN_EVENT_SCHEMA
        and document["truth_open"] is True
        and document["candidate_id"] == one.CANDIDATE_ID
        and document["row_count"] == one.ROW_COUNT
        and all(document[field] == normalized[field] for field in ("prediction_sha256", "prepared_plan_sha256", "method_lock_sha256", "checkpoint_sha256", "pair_manifest_sha256", "normalized_prediction_sha256")),
        "D128 truth-open event/prediction binding drift",
    )
    return dict(document)


def write_d128_a_one18_truth_open_event_exclusive(
    path: str | Path, event: Mapping[str, Any], *, normalized_prediction: Mapping[str, Any]
) -> Path:
    """Persist the open event only after it is bound to a complete normalized prediction."""

    _validate_truth_open_event(event, normalized_prediction=normalized_prediction)
    return _write_exclusive_json(path, event, name="D128 truth-open event")


def _write_truth_open_event(path: str | Path, event: Mapping[str, Any], *, normalized_prediction: Mapping[str, Any]) -> Path:
    _validate_truth_open_event(event, normalized_prediction=normalized_prediction)
    return _write_exclusive_json(path, event, name="D128 truth-open event")


def _read_opened_context(
    *,
    prediction_path: str | Path,
    expected_prediction_sha256: str,
    prepared_plan_path: str | Path,
    expected_prepared_plan_sha256: str,
    method_lock_path: str | Path,
    expected_method_lock_sha256: str,
    truth_open_event_path: str | Path,
    expected_truth_open_event_sha256: str,
) -> tuple[dict[str, Any], dict[str, Any], str, str]:
    prepared = prepare_d128_a_one18_scoring_inputs(
        prediction_path=prediction_path,
        expected_prediction_sha256=expected_prediction_sha256,
        prepared_plan_path=prepared_plan_path,
        expected_prepared_plan_sha256=expected_prepared_plan_sha256,
        method_lock_path=method_lock_path,
        expected_method_lock_sha256=expected_method_lock_sha256,
    )
    normalized = prepared["normalized_prediction"]
    event, event_file_sha = _read_pinned_json(
        truth_open_event_path,
        expected_sha256=expected_truth_open_event_sha256,
        name="D128 truth-open event",
    )
    validated_event = _validate_truth_open_event(event, normalized_prediction=normalized)
    return normalized, dict(prepared["method_lock"]), event_file_sha, validated_event["truth_open_event_sha256"]


def _truth_catalog(
    *, normalized: Mapping[str, Any], assets_by_job: Mapping[str, Any]
) -> dict[str, Any]:
    queries: list[dict[str, str]] = []
    seen: set[str] = set()
    for row in normalized["rows"]:
        job_id = _text(row["formal_d92_source_job_id"], "D128 normalized source D92 job")
        assets = assets_by_job[job_id]
        after_ids = tuple(str(item) for item in row["after_query_ids"])
        before_ids = tuple(str(item) for item in row["before_query_ids"])
        old_classes = set(str(item) for item in row["old_classes"])
        new_classes = set(str(item) for item in row["new_classes"])
        expected_before = tuple(
            query_id for query_id in after_ids if assets.truth_by_query_id.get(query_id, {}).get("role") == "target_old"
        )
        _require(before_ids == expected_before, "D128 before query IDs/D92 truth role drift")
        row_items: list[dict[str, str]] = []
        for query_id in after_ids:
            _require(query_id not in seen, "D128 truth catalog has duplicate opaque query IDs")
            source_truth = assets.truth_by_query_id.get(query_id)
            _require(source_truth is not None, "D128 opaque query ID is absent from D92 truth")
            role = source_truth["role"]
            label = source_truth["label"]
            if role == "target_old":
                _require(label in old_classes, "D128 truth old label is outside the old registry")
                output_role = "old"
            else:
                _require(label in new_classes, "D128 truth new label is outside the new registry")
                output_role = "new"
            item = {"opaque_query_id": query_id, "label": label, "role": output_role}
            seen.add(query_id)
            row_items.append(item)
            queries.append(item)
        _require(old_classes.issubset({item["label"] for item in row_items if item["opaque_query_id"] in before_ids}), "D128 before truth old-class coverage drift")
        _require(new_classes.issubset({item["label"] for item in row_items if item["role"] == "new"}), "D128 after truth new-class coverage drift")
    catalog: dict[str, Any] = {
        "schema": TRUTH_CATALOG_SCHEMA,
        "truth_open": True,
        "candidate_id": one.CANDIDATE_ID,
        "prediction_sha256": normalized["prediction_sha256"],
        "prepared_plan_sha256": normalized["prepared_plan_sha256"],
        "method_lock_sha256": normalized["method_lock_sha256"],
        "checkpoint_sha256": normalized["checkpoint_sha256"],
        "pair_manifest_sha256": normalized["pair_manifest_sha256"],
        "normalized_prediction_sha256": normalized["normalized_prediction_sha256"],
        "query_count": len(queries),
        "queries": queries,
    }
    catalog["truth_catalog_sha256"] = canonical_sha256(catalog)
    return catalog


def _formal_reference(
    *, normalized: Mapping[str, Any], assets_by_job: Mapping[str, Any], retry2_manifest_sha256: str
) -> tuple[dict[str, Any], dict[str, Any]]:
    jobs = [
        {"source_d92_job_id": job_id, "pipeline_receipt_sha256": assets.pipeline_receipt_sha256}
        for job_id, assets in sorted(assets_by_job.items())
    ]
    root: dict[str, Any] = {
        "schema": D92_PIPELINE_ROOT_SCHEMA,
        "d92_retry2_manifest_sha256": retry2_manifest_sha256,
        "jobs": jobs,
    }
    root_sha = canonical_sha256(root)
    rows: list[dict[str, Any]] = []
    for row in normalized["rows"]:
        job_id = row["formal_d92_source_job_id"]
        assets = assets_by_job[job_id]
        score_slice = assets.score_by_scene[row["scene"]]
        rows.append(
            {
                "row_id": row["row_id"],
                "receiver_id": row["receiver_id"],
                "k_shot": row["k_shot"],
                "scene": row["scene"],
                "source_d92_job_id": job_id,
                "d92_retry2_manifest_sha256": retry2_manifest_sha256,
                "formal_d92_score_row_key": score_slice["key"],
                "formal_d92_score_row_sha256": score_slice["sha256"],
            }
        )
    reference: dict[str, Any] = {
        "schema": FORMAL_D92_REFERENCE_SCHEMA,
        "candidate_id": one.CANDIDATE_ID,
        "prediction_sha256": normalized["prediction_sha256"],
        "prepared_plan_sha256": normalized["prepared_plan_sha256"],
        "method_lock_sha256": normalized["method_lock_sha256"],
        "checkpoint_sha256": normalized["checkpoint_sha256"],
        "pair_manifest_sha256": normalized["pair_manifest_sha256"],
        "normalized_prediction_sha256": normalized["normalized_prediction_sha256"],
        "pipeline_receipt_sha256": root_sha,
        "row_count": one.ROW_COUNT,
        "rows": rows,
    }
    reference["formal_d92_reference_sha256"] = canonical_sha256(reference)
    return reference, root


def build_d128_a_one18_truth_assets(
    *,
    prediction_path: str | Path,
    expected_prediction_sha256: str,
    prepared_plan_path: str | Path,
    expected_prepared_plan_sha256: str,
    method_lock_path: str | Path,
    expected_method_lock_sha256: str,
    truth_open_event_path: str | Path,
    expected_truth_open_event_sha256: str,
    d92_retry2_root: str | Path,
    d92_retry2_manifest_path: str | Path,
    expected_d92_retry2_manifest_sha256: str,
    truth_catalog_output: str | Path,
    formal_d92_reference_output: str | Path,
    build_receipt_output: str | Path,
) -> dict[str, Any]:
    """Build D128-owned truth/formal assets only after durable truth opening."""

    normalized, method_lock, event_file_sha, event_receipt_sha = _read_opened_context(
        prediction_path=prediction_path,
        expected_prediction_sha256=expected_prediction_sha256,
        prepared_plan_path=prepared_plan_path,
        expected_prepared_plan_sha256=expected_prepared_plan_sha256,
        method_lock_path=method_lock_path,
        expected_method_lock_sha256=expected_method_lock_sha256,
        truth_open_event_path=truth_open_event_path,
        expected_truth_open_event_sha256=expected_truth_open_event_sha256,
    )
    truth_target = _assert_output_absent(truth_catalog_output, "D128 truth catalog")
    formal_target = _assert_output_absent(formal_d92_reference_output, "D128 formal D92 reference")
    receipt_target = _assert_output_absent(build_receipt_output, "D128 truth-assets receipt")

    # Everything above is truth-free.  The D92 read boundary starts here.
    try:
        retry_root = d92_assets._regular_directory(d92_retry2_root, "D128 D92 retry2 root")
        matrix_path = d92_assets._regular_file(d92_retry2_manifest_path, "D128 D92 retry2 matrix manifest")
        _require(matrix_path.parent == retry_root, "D128 D92 retry2 matrix manifest/root drift")
        matrix, matrix_sha, _matrix_path = d92_assets._read_json(
            matrix_path,
            name="D128 D92 retry2 matrix manifest",
            expected_sha256=expected_d92_retry2_manifest_sha256,
        )
        _require(
            matrix_sha == method_lock["s0_matrix"]["d92_retry2_manifest_sha256"],
            "D128 method-lock/D92 retry2 manifest drift",
        )
        seed, _source_k5, scenes, routes = d92_assets._expected_routes(method_lock, normalized)
        jobs = d92_assets._index_d92_jobs(matrix)
        assets_by_job: dict[str, Any] = {}
        for row in normalized["rows"]:
            source_k, job_id = routes[row["row_id"]]
            _require(row["formal_d92_retry2_manifest_sha256"] == matrix_sha, "D128 normalized D92 manifest drift")
            job = jobs.get(job_id)
            _require(job is not None and job.get("receiver") == row["receiver_id"], "D128/D92 receiver route drift")
            if job_id not in assets_by_job:
                assets_by_job[job_id] = d92_assets._load_d92_job_assets(
                    retry2_root=retry_root,
                    job=job,
                    seed=seed,
                    source_k_shot=source_k,
                    scenes=scenes,
                )
    except D128AOne18ScorerError:
        raise
    except Exception as exc:
        raise D128AOne18ScorerError("D128 D92 scorer-side truth/formal asset load failed") from exc
    _require(len(assets_by_job) == 6, "D128 must bind exactly six D92 source jobs")
    truth_catalog = _truth_catalog(normalized=normalized, assets_by_job=assets_by_job)
    formal_reference, pipeline_root = _formal_reference(
        normalized=normalized,
        assets_by_job=assets_by_job,
        retry2_manifest_sha256=matrix_sha,
    )
    _write_exclusive_json(truth_target, truth_catalog, name="D128 truth catalog")
    _write_exclusive_json(formal_target, formal_reference, name="D128 formal D92 reference")
    source_jobs = [
        {
            "source_d92_job_id": assets.job_id,
            "receiver_id": assets.receiver,
            "seed": assets.seed,
            "source_k_shot": assets.source_k_shot,
            "new_class_count": assets.new_class_count,
            "pipeline_receipt_sha256": assets.pipeline_receipt_sha256,
            "row_manifest_sha256": assets.row_manifest_sha256,
            "registration_pair_sha256": assets.registration_pair_sha256,
            "truth_sidecar_sha256": assets.truth_sidecar_sha256,
            "formal_score_sha256": assets.score_artifact_sha256,
            "truth_query_count_used": sum(len(row["after_query_ids"]) for row in normalized["rows"] if row["formal_d92_source_job_id"] == assets.job_id),
        }
        for assets in sorted(assets_by_job.values(), key=lambda item: item.job_id)
    ]
    receipt: dict[str, Any] = {
        "schema": TRUTH_ASSETS_RECEIPT_SCHEMA,
        "candidate_id": one.CANDIDATE_ID,
        "truth_open_event_file_sha256": event_file_sha,
        "truth_open_event_sha256": event_receipt_sha,
        "prediction_sha256": normalized["prediction_sha256"],
        "prepared_plan_sha256": normalized["prepared_plan_sha256"],
        "method_lock_sha256": normalized["method_lock_sha256"],
        "pair_manifest_sha256": normalized["pair_manifest_sha256"],
        "normalized_prediction_sha256": normalized["normalized_prediction_sha256"],
        "d92_retry2_manifest_sha256": matrix_sha,
        "pipeline_receipt_root": pipeline_root,
        "source_jobs": source_jobs,
        "row_count": one.ROW_COUNT,
        "query_count": truth_catalog["query_count"],
        "truth_catalog_sha256": _sha256_file(truth_target),
        "formal_d92_reference_sha256": _sha256_file(formal_target),
        "prediction_values_read": False,
        "query_truth_opened_after_durable_event": True,
    }
    receipt["truth_assets_receipt_sha256"] = canonical_sha256(receipt)
    _write_exclusive_json(receipt_target, receipt, name="D128 truth-assets receipt")
    return {
        "status": "D128_A_ONE18_TRUTH_ASSETS_BUILT",
        "candidate_id": one.CANDIDATE_ID,
        "row_count": one.ROW_COUNT,
        "query_count": truth_catalog["query_count"],
        "truth_catalog": str(truth_target.resolve()),
        "truth_catalog_sha256": _sha256_file(truth_target),
        "formal_d92_reference": str(formal_target.resolve()),
        "formal_d92_reference_sha256": _sha256_file(formal_target),
        "build_receipt": str(receipt_target.resolve()),
        "build_receipt_sha256": _sha256_file(receipt_target),
        "prediction_values_read": False,
    }


def _open_truth_catalog(catalog: Any, *, normalized_prediction: Mapping[str, Any]) -> tuple[dict[str, dict[str, str]], str]:
    normalized = _validate_normalized(normalized_prediction)
    expected_fields = {
        "schema", "truth_open", "candidate_id", "prediction_sha256", "prepared_plan_sha256", "method_lock_sha256",
        "checkpoint_sha256", "pair_manifest_sha256", "normalized_prediction_sha256", "query_count", "queries",
        "truth_catalog_sha256",
    }
    document = _exact(catalog, expected_fields, "D128 truth catalog")
    unsigned = dict(document)
    receipt = _sha(unsigned.pop("truth_catalog_sha256"), "D128 truth catalog receipt")
    _require(canonical_sha256(unsigned) == receipt, "D128 truth catalog digest drift")
    _require(
        document["schema"] == TRUTH_CATALOG_SCHEMA
        and document["truth_open"] is True
        and document["candidate_id"] == one.CANDIDATE_ID
        and all(document[field] == normalized[field] for field in ("prediction_sha256", "prepared_plan_sha256", "method_lock_sha256", "checkpoint_sha256", "pair_manifest_sha256", "normalized_prediction_sha256")),
        "D128 truth catalog binding drift",
    )
    queries = document["queries"]
    _require(isinstance(queries, list) and len(queries) == document["query_count"] and bool(queries), "D128 truth catalog query count drift")
    truth: dict[str, dict[str, str]] = {}
    for index, raw in enumerate(queries):
        item = _exact(raw, {"opaque_query_id", "label", "role"}, f"D128 truth query[{index}]")
        query_id = _text(item["opaque_query_id"], "D128 truth query ID")
        _require(query_id not in truth, "D128 truth catalog duplicate query ID")
        role = _text(item["role"], "D128 truth role")
        _require(role in {"old", "new"}, "D128 truth role drift")
        truth[query_id] = {"label": _text(item["label"], "D128 truth label"), "role": role}
    expected_after = {query_id for row in normalized["rows"] for query_id in row["after_query_ids"]}
    _require(set(truth) == expected_after, "D128 truth catalog query-root drift")
    for row in normalized["rows"]:
        before_ids = tuple(row["before_query_ids"])
        after_ids = tuple(row["after_query_ids"])
        old = set(row["old_classes"])
        new = set(row["new_classes"])
        _require(before_ids == tuple(query_id for query_id in after_ids if truth[query_id]["role"] == "old"), "D128 before/after old query role drift")
        _require(all((truth[qid]["role"] == "old" and truth[qid]["label"] in old) or (truth[qid]["role"] == "new" and truth[qid]["label"] in new) for qid in after_ids), "D128 truth label/role registry drift")
    return truth, receipt


def _open_formal_reference(reference: Any, *, normalized_prediction: Mapping[str, Any]) -> tuple[dict[str, Mapping[str, Any]], str]:
    normalized = _validate_normalized(normalized_prediction)
    expected_fields = {
        "schema", "candidate_id", "prediction_sha256", "prepared_plan_sha256", "method_lock_sha256", "checkpoint_sha256",
        "pair_manifest_sha256", "normalized_prediction_sha256", "pipeline_receipt_sha256", "row_count", "rows",
        "formal_d92_reference_sha256",
    }
    document = _exact(reference, expected_fields, "D128 formal D92 reference")
    unsigned = dict(document)
    receipt = _sha(unsigned.pop("formal_d92_reference_sha256"), "D128 formal D92 reference receipt")
    _require(canonical_sha256(unsigned) == receipt, "D128 formal D92 reference digest drift")
    _require(
        document["schema"] == FORMAL_D92_REFERENCE_SCHEMA
        and document["candidate_id"] == one.CANDIDATE_ID
        and document["row_count"] == one.ROW_COUNT
        and all(document[field] == normalized[field] for field in ("prediction_sha256", "prepared_plan_sha256", "method_lock_sha256", "checkpoint_sha256", "pair_manifest_sha256", "normalized_prediction_sha256")),
        "D128 formal D92 reference binding drift",
    )
    _sha(document["pipeline_receipt_sha256"], "D128 formal D92 pipeline receipt")
    rows = document["rows"]
    _require(isinstance(rows, list) and len(rows) == one.ROW_COUNT, "D128 formal D92 row coverage drift")
    expected_row_fields = {
        "row_id", "receiver_id", "k_shot", "scene", "source_d92_job_id", "d92_retry2_manifest_sha256",
        "formal_d92_score_row_key", "formal_d92_score_row_sha256",
    }
    by_id: dict[str, Mapping[str, Any]] = {}
    for index, raw in enumerate(rows):
        row = _exact(raw, expected_row_fields, f"D128 formal D92 row[{index}]")
        row_id = _text(row["row_id"], "D128 formal D92 row ID")
        _require(row_id not in by_id, "D128 formal D92 duplicate row")
        _sha(row["d92_retry2_manifest_sha256"], "D128 formal D92 retry2 manifest SHA256")
        _text(row["formal_d92_score_row_key"], "D128 formal D92 score row key")
        _sha(row["formal_d92_score_row_sha256"], "D128 formal D92 score row SHA256")
        by_id[row_id] = row
    _require(set(by_id) == {row["row_id"] for row in normalized["rows"]}, "D128 formal D92 row IDs drift")
    for row in normalized["rows"]:
        formal = by_id[row["row_id"]]
        _require(
            formal["receiver_id"] == row["receiver_id"]
            and formal["k_shot"] == row["k_shot"]
            and formal["scene"] == row["scene"]
            and formal["source_d92_job_id"] == row["formal_d92_source_job_id"]
            and formal["d92_retry2_manifest_sha256"] == row["formal_d92_retry2_manifest_sha256"],
            "D128 formal D92 same-row locator/hash drift",
        )
    return by_id, receipt


def _direction_decision(metric_rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    by_arm = {arm_id: [row for row in metric_rows if row["arm_id"] == arm_id] for arm_id in one.ARM_IDS}
    all_rows = {
        arm_id: d127_scorer._aggregate(
            by_arm[arm_id], group_key="scope", group_value="D128_A_ONE18", candidate_id=one.CANDIDATE_ID, arm_id=arm_id
        )
        for arm_id in one.ARM_IDS
    }
    k5 = {
        arm_id: d127_scorer._aggregate(
            [row for row in by_arm[arm_id] if row["k_shot"] == 5],
            group_key="k_shot", group_value=5, candidate_id=one.CANDIDATE_ID, arm_id=arm_id,
        )
        for arm_id in ("M_DA", "M_JOINT")
    }
    g1_delta = all_rows["M_DA"]["H_old_new"] - all_rows["M0"]["H_old_new"]
    g2_delta = k5["M_JOINT"]["H_old_new"] - k5["M_DA"]["H_old_new"]
    g3_delta = all_rows["M_JOINT"]["H_old_new"] - all_rows["M0"]["H_old_new"]
    g3_total = all_rows["M_JOINT"]["total_correct_count"] - all_rows["M0"]["total_correct_count"]
    result: dict[str, Any] = {
        "candidate_id": one.CANDIDATE_ID,
        "G1_M_DA_over_M0_delta_H_old_new": g1_delta,
        "G1_M_DA_over_M0_pass": g1_delta > 0.0,
        "G2_K5_M_JOINT_over_M_DA_delta_H_old_new": g2_delta,
        "G2_K5_M_JOINT_over_M_DA_pass": g2_delta > 0.0,
        "G3_M_JOINT_over_M0_delta_H_old_new": g3_delta,
        "G3_M_JOINT_over_M0_total_correct_delta": g3_total,
        "G3_M_JOINT_over_M0_pass": g3_delta > 0.0 and g3_total > 0,
        "promotion_action": "NONE_REPORT_ONLY",
    }
    result["all_three_direction_pass"] = bool(
        result["G1_M_DA_over_M0_pass"]
        and result["G2_K5_M_JOINT_over_M_DA_pass"]
        and result["G3_M_JOINT_over_M0_pass"]
    )
    result["direction_receipt_sha256"] = canonical_sha256(result)
    return result


def score_d128_a_one18(
    *,
    normalized_prediction: Mapping[str, Any],
    truth_open_event: Mapping[str, Any],
    truth_catalog: Mapping[str, Any],
    formal_d92_reference: Mapping[str, Any],
) -> dict[str, Any]:
    """Score one sealed A artifact; all returned gates are report-only."""

    normalized = _validate_normalized(normalized_prediction)
    _validate_truth_open_event(truth_open_event, normalized_prediction=normalized)
    # Truth/formal content is deliberately opened only after the durable event.
    truth, truth_receipt = _open_truth_catalog(truth_catalog, normalized_prediction=normalized)
    formal_by_id, formal_receipt = _open_formal_reference(formal_d92_reference, normalized_prediction=normalized)
    metric_rows: list[dict[str, Any]] = []
    same_rows: list[dict[str, Any]] = []
    for row in normalized["rows"]:
        formal = formal_by_id[row["row_id"]]
        metric_manifest = {
            "row_id": row["row_id"],
            "receiver_id": row["receiver_id"],
            "k_shot": row["k_shot"],
            "scene": row["scene"],
            "old_classes": row["old_classes"],
            "new_classes": row["new_classes"],
            "formal_d92_row_key": formal["formal_d92_score_row_key"],
            "formal_d92_score_row_sha256": formal["formal_d92_score_row_sha256"],
        }
        arms: list[dict[str, Any]] = []
        for arm_id in one.ARM_IDS:
            metric = d127_scorer._metric_row(
                manifest_row=metric_manifest,
                candidate_id=one.CANDIDATE_ID,
                arm_id=arm_id,
                before_predictions=row["before_arms"][arm_id],
                after_predictions=row["after_arms"][arm_id],
                before_ids=row["before_query_ids"],
                after_ids=row["after_query_ids"],
                truth=truth,
            )
            arms.append(metric)
            metric_rows.append(metric)
        same: dict[str, Any] = {
            "row_id": row["row_id"],
            "receiver_id": row["receiver_id"],
            "k_shot": row["k_shot"],
            "scene": row["scene"],
            "formal_d92_row_key": formal["formal_d92_score_row_key"],
            "formal_d92_score_row_sha256": formal["formal_d92_score_row_sha256"],
            "candidate_arm_metrics": arms,
        }
        same["same_row_sha256"] = canonical_sha256(same)
        same_rows.append(same)
    _require(len(same_rows) == one.ROW_COUNT and len(metric_rows) == one.ROW_COUNT * len(one.ARM_IDS), "D128 score metric row coverage drift")
    aggregates: dict[str, list[dict[str, Any]]] = {"scope": [], "receiver_id": [], "scene": [], "k_shot": []}
    for arm_id in one.ARM_IDS:
        aggregates["scope"].append(
            d127_scorer._aggregate(
                [row for row in metric_rows if row["arm_id"] == arm_id],
                group_key="scope",
                group_value="D128_A_ONE18",
                candidate_id=one.CANDIDATE_ID,
                arm_id=arm_id,
            )
        )
    for group_key in ("receiver_id", "scene", "k_shot"):
        for group_value in sorted({row[group_key] for row in metric_rows}, key=str):
            for arm_id in one.ARM_IDS:
                aggregates[group_key].append(
                    d127_scorer._aggregate(
                        [row for row in metric_rows if row[group_key] == group_value and row["arm_id"] == arm_id],
                        group_key=group_key,
                        group_value=group_value,
                        candidate_id=one.CANDIDATE_ID,
                        arm_id=arm_id,
                    )
                )
    score: dict[str, Any] = {
        "schema": SCORE_SCHEMA,
        "candidate_id": one.CANDIDATE_ID,
        "prediction_sha256": normalized["prediction_sha256"],
        "prepared_plan_sha256": normalized["prepared_plan_sha256"],
        "method_lock_sha256": normalized["method_lock_sha256"],
        "checkpoint_sha256": normalized["checkpoint_sha256"],
        "pair_manifest_sha256": normalized["pair_manifest_sha256"],
        "normalized_prediction_sha256": normalized["normalized_prediction_sha256"],
        "truth_catalog_sha256": truth_receipt,
        "formal_d92_reference_sha256": formal_receipt,
        "row_count": one.ROW_COUNT,
        "metric_row_count": len(metric_rows),
        "same_row_results": same_rows,
        "aggregates": aggregates,
        "one_shot_direction_decision": _direction_decision(metric_rows),
        "truth_never_returned_to_predictor": True,
        "formal_d92_is_same_row_reference_only": True,
        "promotion_action": "NONE_REPORT_ONLY",
    }
    score["score_manifest_sha256"] = canonical_sha256(score)
    return score


def write_d128_a_one18_score_exclusive(path: str | Path, score: Mapping[str, Any]) -> Path:
    _require(score.get("schema") == SCORE_SCHEMA and score.get("candidate_id") == one.CANDIDATE_ID, "D128 score schema/candidate drift")
    unsigned = dict(score)
    receipt = _sha(unsigned.pop("score_manifest_sha256", None), "D128 score receipt")
    _require(canonical_sha256(unsigned) == receipt, "D128 score digest drift")
    return _write_exclusive_json(path, score, name="D128 score")


__all__ = [
    "D128AOne18ScorerError", "D92_PIPELINE_ROOT_SCHEMA", "FORMAL_D92_REFERENCE_SCHEMA", "NORMALIZED_SCHEMA",
    "SCORE_SCHEMA", "TRUTH_ASSETS_RECEIPT_SCHEMA", "TRUTH_CATALOG_SCHEMA", "TRUTH_OPEN_EVENT_SCHEMA",
    "build_d128_a_one18_truth_assets", "build_d128_a_one18_truth_open_event", "canonical_sha256",
    "prepare_d128_a_one18_scoring_inputs", "score_d128_a_one18", "write_d128_a_one18_score_exclusive",
    "write_d128_a_one18_truth_open_event_exclusive",
]
