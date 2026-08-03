"""Build the scorer-only truth and same-row D92 reference for D127 S0.

This module is deliberately downstream of the immutable paired prediction.
It validates every truth-free D127 binding and a durable truth-open event
*before* it resolves a D92 scorer-side path.  It never opens a predictor
package, never reads a prediction value, and only joins labels by opaque query
ID from D92's scorer-side ``truth_sidecar.json``.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Mapping, Sequence

from . import stage2_d127_s0_package_adapter as adapter
from . import stage2_d127_s0_scorer as scorer


TRUTH_ASSETS_BUILD_RECEIPT_SCHEMA = "cvs.stage2.d127.s0.truth_assets_build_receipt.v1"
D92_PIPELINE_RECEIPT_ROOT_SCHEMA = "cvs.stage2.d127.s0.d92_pipeline_receipt_root.v1"
D92_SCENE_SCORE_SLICE_SCHEMA = "cvs.stage2.d127.s0.d92_scene_score_slice.v1"


class D127S0TruthAssetsError(ValueError):
    """Raised when an opened D127 S0 truth asset has provenance drift."""


@dataclass(frozen=True, slots=True)
class _D92JobAssets:
    """Verified scorer-side material from one immutable D92 retry2 job."""

    job_id: str
    receiver: str
    seed: int
    source_k_shot: int
    new_class_count: int
    pipeline_receipt_sha256: str
    row_manifest_sha256: str
    registration_pair_sha256: str
    truth_sidecar_sha256: str
    score_artifact_sha256: str
    truth_by_query_id: Mapping[str, Mapping[str, str]]
    score_by_scene: Mapping[str, Mapping[str, Any]]
    source_paths: Mapping[str, str]


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise D127S0TruthAssetsError(message)


def _canonical_bytes(value: Mapping[str, Any]) -> bytes:
    return json.dumps(
        dict(value), ensure_ascii=True, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")


def _canonical_sha256(value: Mapping[str, Any]) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1_048_576), b""):
            digest.update(block)
    return digest.hexdigest()


def _sha(value: Any, name: str) -> str:
    text = str(value).lower()
    _require(len(text) == 64 and all(char in "0123456789abcdef" for char in text), f"{name} SHA256 drift")
    return text


def _text(value: Any, name: str) -> str:
    result = str(value)
    _require(bool(result), f"{name} is empty")
    return result


def _integer(value: Any, name: str, *, minimum: int = 0) -> int:
    _require(type(value) is int and value >= minimum, f"{name} is invalid")
    return int(value)


def _regular_file(path: str | Path, name: str) -> Path:
    candidate = Path(path)
    _require(candidate.is_file() and not candidate.is_symlink(), f"{name} must be a regular non-symlink file")
    return candidate.resolve(strict=True)


def _regular_directory(path: str | Path, name: str) -> Path:
    candidate = Path(path)
    _require(candidate.is_dir() and not candidate.is_symlink(), f"{name} must be a regular non-symlink directory")
    return candidate.resolve(strict=True)


def _read_json(
    path: str | Path,
    *,
    name: str,
    expected_sha256: str | None = None,
) -> tuple[dict[str, Any], str, Path]:
    source = _regular_file(path, name)
    digest = _sha256_file(source)
    if expected_sha256 is not None:
        _require(digest == _sha(expected_sha256, f"expected {name}"), f"{name} SHA mismatch")
    try:
        value = json.loads(source.read_text(encoding="utf-8-sig"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise D127S0TruthAssetsError(f"{name} is not valid UTF-8 JSON") from exc
    _require(isinstance(value, dict), f"{name} must contain a JSON object")
    return value, digest, source


def _assert_output_absent(path: str | Path, name: str) -> Path:
    target = Path(path)
    _require(not target.is_symlink(), f"{name} output cannot be a symlink")
    _require(not target.exists(), f"{name} output already exists")
    return target


def _write_exclusive_json(path: Path, payload: Mapping[str, Any], *, name: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with path.open("xb") as stream:
            stream.write(_canonical_bytes(payload) + b"\n")
            stream.flush()
            os.fsync(stream.fileno())
    except FileExistsError as exc:
        raise D127S0TruthAssetsError(f"{name} output already exists") from exc
    return path


def _read_opened_context(
    *,
    paired_prediction_path: str | Path,
    expected_paired_prediction_sha256: str,
    prepared_plan_path: str | Path,
    expected_prepared_plan_sha256: str,
    method_lock_path: str | Path,
    expected_method_lock_sha256: str,
    truth_open_event_path: str | Path,
    expected_truth_open_event_sha256: str,
) -> tuple[dict[str, Any], dict[str, Any], str, str]:
    """Run all truth-free checks, then authenticate the already-written open event."""

    prepared = scorer.prepare_d127_s0_scoring_inputs(
        paired_prediction_path=paired_prediction_path,
        expected_paired_prediction_sha256=expected_paired_prediction_sha256,
        prepared_plan_path=prepared_plan_path,
        expected_prepared_plan_sha256=expected_prepared_plan_sha256,
        method_lock_path=method_lock_path,
        expected_method_lock_sha256=expected_method_lock_sha256,
    )
    normalized = prepared["normalized_prediction"]
    event, event_file_sha, _event_path = _read_json(
        truth_open_event_path,
        name="D127 truth-open event",
        expected_sha256=expected_truth_open_event_sha256,
    )
    expected_event = scorer.build_d127_s0_truth_open_event(normalized)
    _require(event == expected_event, "D127 truth-open event/prediction binding drift")
    lock, lock_file_sha, _locks = adapter.load_d127_s0_method_lock(
        method_lock_path,
        expected_sha256=expected_method_lock_sha256,
    )
    _require(lock_file_sha == normalized["method_lock_sha256"], "D127 method-lock/normalized prediction drift")
    return normalized, lock, event_file_sha, str(event["truth_open_event_sha256"])


def _index_d92_jobs(manifest: Mapping[str, Any]) -> dict[str, Mapping[str, Any]]:
    _require(manifest.get("schema") == "cvs.phase2.somph_diag_125_stability.v1", "D92 retry2 matrix schema drift")
    jobs = manifest.get("jobs")
    _require(isinstance(jobs, list) and bool(jobs), "D92 retry2 matrix jobs are missing")
    result: dict[str, Mapping[str, Any]] = {}
    for index, value in enumerate(jobs):
        _require(isinstance(value, Mapping), f"D92 matrix job[{index}] is invalid")
        job_id = _text(value.get("job_id"), f"D92 matrix job[{index}].job_id")
        _require(job_id not in result, "D92 retry2 matrix has duplicate job IDs")
        _text(value.get("receiver"), f"D92 matrix job[{index}].receiver")
        _integer(value.get("seed"), f"D92 matrix job[{index}].seed", minimum=1)
        _integer(value.get("k_shot"), f"D92 matrix job[{index}].k_shot", minimum=1)
        _integer(value.get("new_class_count"), f"D92 matrix job[{index}].new_class_count", minimum=1)
        _text(value.get("output_root"), f"D92 matrix job[{index}].output_root")
        result[job_id] = value
    return result


def _expected_routes(method_lock: Mapping[str, Any], normalized: Mapping[str, Any]) -> tuple[int, int, tuple[str, ...], dict[str, tuple[int, str]]]:
    matrix = method_lock.get("s0_matrix")
    _require(isinstance(matrix, Mapping), "D127 method lock S0 matrix is missing")
    seed = _integer(matrix.get("seed"), "D127 S0 seed", minimum=1)
    source_k5 = _integer(matrix.get("k5_source_pool_k"), "D127 S0 K5 source pool", minimum=5)
    scenes_raw = matrix.get("scenes")
    _require(isinstance(scenes_raw, list) and all(isinstance(item, str) and item for item in scenes_raw), "D127 S0 scene lock drift")
    scenes = tuple(scenes_raw)
    _require(len(scenes) == len(set(scenes)), "D127 S0 scene lock is duplicated")
    routes: dict[str, tuple[int, str]] = {}
    for row in normalized["rows"]:
        receiver = _text(row.get("receiver_id"), "normalized receiver")
        k_shot = _integer(row.get("k_shot"), "normalized K-shot", minimum=1)
        scene = _text(row.get("scene"), "normalized scene")
        _require(scene in scenes, "normalized scene is outside the frozen S0 matrix")
        source_k = source_k5 if k_shot == 5 else k_shot
        expected_job_id = f"rx_{receiver.replace('-', '_')}__seed_{seed}__k_{source_k}__new_20"
        _require(row.get("formal_d92_source_job_id") == expected_job_id, "D127 normalized source D92 job drift")
        routes[_text(row.get("row_id"), "normalized row ID")] = (source_k, expected_job_id)
    _require(len(routes) == scorer.ROW_COUNT, "D127 normalized route coverage drift")
    return seed, source_k5, scenes, routes


def _validate_row_manifest(
    document: Mapping[str, Any],
    *,
    receiver: str,
    seed: int,
    source_k_shot: int,
    new_class_count: int,
    scenes: Sequence[str],
) -> None:
    _require(document.get("schema") == "cvs.phase2.somph_row_manifest.v2", "D92 scorer row-manifest schema drift")
    _require(
        document.get("receiver") == receiver
        and document.get("seed") == seed
        and document.get("k_shot") == source_k_shot
        and document.get("new_class_count") == new_class_count,
        "D92 scorer row-manifest identity drift",
    )
    observed = document.get("scenarios")
    _require(isinstance(observed, list) and tuple(observed) == tuple(scenes), "D92 scorer row-manifest scene drift")


def _truth_map(document: Mapping[str, Any]) -> dict[str, dict[str, str]]:
    _require(document.get("schema") == "cvs.phase2.query_truth_sidecar.v2", "D92 truth-sidecar schema drift")
    rows = document.get("rows")
    _require(isinstance(rows, list) and bool(rows), "D92 truth-sidecar rows are missing")
    result: dict[str, dict[str, str]] = {}
    for index, value in enumerate(rows):
        _require(isinstance(value, Mapping), f"D92 truth row[{index}] is invalid")
        query_id = _text(value.get("query_token"), f"D92 truth row[{index}].query_token")
        _require(query_id not in result, "D92 truth-sidecar has duplicate opaque query IDs")
        role = _text(value.get("evaluation_role"), f"D92 truth row[{index}].evaluation_role")
        _require(role in {"target_old", "target_new"}, "D92 truth-sidecar role drift")
        result[query_id] = {
            "label": _text(value.get("true_class_handle"), f"D92 truth row[{index}].true_class_handle"),
            "role": role,
            "transmitter_label": _text(value.get("transmitter_label"), f"D92 truth row[{index}].transmitter_label"),
        }
    return result


def _score_slices(document: Mapping[str, Any], *, score_sha256: str, job_id: str, receiver: str, seed: int, source_k_shot: int, scenes: Sequence[str]) -> dict[str, dict[str, Any]]:
    _require(document.get("schema") == "cvs.phase2.diag_cosine_dev_pair_score.v1", "D92 formal score schema drift")
    before = document.get("before")
    after = document.get("after")
    _require(isinstance(before, Mapping) and isinstance(after, Mapping), "D92 formal score state drift")
    before_scenes = before.get("by_scenario")
    after_scenes = after.get("by_scenario")
    _require(isinstance(before_scenes, Mapping) and isinstance(after_scenes, Mapping), "D92 formal score scene table drift")
    result: dict[str, dict[str, Any]] = {}
    for scene in scenes:
        _require(scene in before_scenes and scene in after_scenes, "D92 formal score is missing a frozen scene")
        _require(isinstance(before_scenes[scene], Mapping) and isinstance(after_scenes[scene], Mapping), "D92 formal score scene payload drift")
        slice_payload: dict[str, Any] = {
            "schema": D92_SCENE_SCORE_SLICE_SCHEMA,
            "source_d92_job_id": job_id,
            "receiver_id": receiver,
            "seed": seed,
            "source_k_shot": source_k_shot,
            "new_class_count": 20,
            "scene": scene,
            "score_artifact_sha256": score_sha256,
            "before_by_scenario": dict(before_scenes[scene]),
            "after_by_scenario": dict(after_scenes[scene]),
        }
        result[scene] = {
            "key": f"{job_id}::{scene}",
            "sha256": _canonical_sha256(slice_payload),
        }
    return result


def _load_d92_job_assets(
    *,
    retry2_root: Path,
    job: Mapping[str, Any],
    seed: int,
    source_k_shot: int,
    scenes: Sequence[str],
) -> _D92JobAssets:
    job_id = _text(job.get("job_id"), "D92 source job ID")
    receiver = _text(job.get("receiver"), "D92 source receiver")
    _require(_integer(job.get("seed"), "D92 source seed", minimum=1) == seed, "D92 source job seed drift")
    _require(_integer(job.get("k_shot"), "D92 source K-shot", minimum=1) == source_k_shot, "D92 source job K-shot drift")
    _require(_integer(job.get("new_class_count"), "D92 source new-class count", minimum=1) == 20, "D92 source job new-class count drift")
    job_root = _regular_directory(retry2_root / "jobs" / job_id, "D92 source job root")
    _require(job_root.parent == (retry2_root / "jobs").resolve(strict=True), "D92 source job root escape")
    _require(Path(str(job.get("output_root"))).resolve(strict=False) == job_root, "D92 matrix output-root/job-root drift")

    pipeline, pipeline_sha, pipeline_path = _read_json(job_root / "pipeline_receipt.json", name="D92 pipeline receipt")
    row_manifest, row_manifest_sha, row_manifest_path = _read_json(job_root / "offline" / "scorer" / "row_manifest.json", name="D92 scorer row manifest")
    pair, pair_sha, pair_path = _read_json(job_root / "scorer" / "registration_pair.final.json", name="D92 registration pair")
    truth, truth_sha, truth_path = _read_json(job_root / "offline" / "scorer" / "truth_sidecar.json", name="D92 scorer truth sidecar")
    score, score_sha, score_path = _read_json(job_root / "scorer" / "diag_cosine_score.json", name="D92 formal score artifact")

    _require(pipeline.get("schema") == "cvs.phase2.somph_diag_row_pipeline.v1", "D92 pipeline receipt schema drift")
    _require(
        pipeline.get("receiver") == receiver
        and pipeline.get("seed") == seed
        and pipeline.get("k_shot") == source_k_shot
        and pipeline.get("new_class_count") == 20,
        "D92 pipeline receipt identity drift",
    )
    _require(pipeline.get("score_artifact_sha256") == score_sha, "D92 pipeline/formal-score hash drift")
    _require(pipeline.get("row_manifest_sha256") == row_manifest_sha, "D92 pipeline/row-manifest hash drift")
    _require(pipeline.get("registration_pair_final_sha256") == pair_sha, "D92 pipeline/registration-pair hash drift")
    _validate_row_manifest(
        row_manifest,
        receiver=receiver,
        seed=seed,
        source_k_shot=source_k_shot,
        new_class_count=20,
        scenes=scenes,
    )
    _require(pair.get("schema") == "cvs.phase2.somph_registration_pair.v1", "D92 registration-pair schema drift")
    _require(
        pair.get("old_query_physical_ids_sha256_before") == pair.get("old_query_physical_ids_sha256_after")
        and pair.get("old_support_physical_ids_sha256_before") == pair.get("old_support_physical_ids_sha256_after"),
        "D92 registration-pair old support/query reuse drift",
    )
    _require(score.get("truth_sidecar_sha256") == truth_sha, "D92 formal-score/truth-sidecar hash drift")
    return _D92JobAssets(
        job_id=job_id,
        receiver=receiver,
        seed=seed,
        source_k_shot=source_k_shot,
        new_class_count=20,
        pipeline_receipt_sha256=pipeline_sha,
        row_manifest_sha256=row_manifest_sha,
        registration_pair_sha256=pair_sha,
        truth_sidecar_sha256=truth_sha,
        score_artifact_sha256=score_sha,
        truth_by_query_id=_truth_map(truth),
        score_by_scene=_score_slices(
            score,
            score_sha256=score_sha,
            job_id=job_id,
            receiver=receiver,
            seed=seed,
            source_k_shot=source_k_shot,
            scenes=scenes,
        ),
        source_paths={
            "pipeline_receipt": str(pipeline_path),
            "row_manifest": str(row_manifest_path),
            "registration_pair": str(pair_path),
            "truth_sidecar": str(truth_path),
            "formal_score": str(score_path),
        },
    )


def _truth_catalog(
    *,
    normalized: Mapping[str, Any],
    assets_by_job: Mapping[str, _D92JobAssets],
) -> dict[str, Any]:
    queries: list[dict[str, str]] = []
    seen: set[str] = set()
    for row in normalized["rows"]:
        job_id = _text(row.get("formal_d92_source_job_id"), "normalized source D92 job")
        assets = assets_by_job[job_id]
        after_ids = tuple(str(item) for item in row["after_query_ids"])
        before_ids = tuple(str(item) for item in row["before_query_ids"])
        old_classes = set(str(item) for item in row["old_classes"])
        new_classes = set(str(item) for item in row["new_classes"])
        expected_before = tuple(query_id for query_id in after_ids if assets.truth_by_query_id.get(query_id, {}).get("role") == "target_old")
        _require(before_ids == expected_before, "D127 before query IDs/D92 truth role drift")
        for query_id in after_ids:
            _require(query_id not in seen, "D127 truth catalog has duplicate opaque query IDs")
            source_truth = assets.truth_by_query_id.get(query_id)
            _require(source_truth is not None, "D127 opaque query ID is absent from D92 scorer truth")
            role = source_truth["role"]
            label = source_truth["label"]
            if role == "target_old":
                _require(label in old_classes, "D92 truth old label is outside D127 old registry")
                output_role = "old"
            else:
                _require(label in new_classes, "D92 truth new label is outside D127 new registry")
                output_role = "new"
            seen.add(query_id)
            queries.append({"opaque_query_id": query_id, "label": label, "role": output_role})
        _require(old_classes.issubset({item["label"] for item in queries if item["opaque_query_id"] in before_ids}), "D127 before old-class truth coverage drift")
        _require(new_classes.issubset({item["label"] for item in queries if item["opaque_query_id"] in after_ids and item["role"] == "new"}), "D127 after new-class truth coverage drift")
    catalog: dict[str, Any] = {
        "schema": scorer.PAIRED_TRUTH_CATALOG_SCHEMA,
        "truth_open": True,
        "paired_prediction_sha256": normalized["paired_prediction_sha256"],
        "prepared_plan_sha256": normalized["prepared_plan_sha256"],
        "method_lock_sha256": normalized["method_lock_sha256"],
        "pair_manifest_sha256": normalized["pair_manifest_sha256"],
        "normalized_prediction_sha256": normalized["normalized_prediction_sha256"],
        "query_count": len(queries),
        "queries": queries,
    }
    catalog["truth_catalog_sha256"] = _canonical_sha256(catalog)
    return catalog


def _formal_reference(
    *,
    normalized: Mapping[str, Any],
    assets_by_job: Mapping[str, _D92JobAssets],
    retry2_manifest_sha256: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    receipt_jobs: list[dict[str, Any]] = []
    for job_id, assets in sorted(assets_by_job.items()):
        receipt_jobs.append({"source_d92_job_id": job_id, "pipeline_receipt_sha256": assets.pipeline_receipt_sha256})
    pipeline_root_payload: dict[str, Any] = {
        "schema": D92_PIPELINE_RECEIPT_ROOT_SCHEMA,
        "d92_retry2_manifest_sha256": retry2_manifest_sha256,
        "jobs": receipt_jobs,
    }
    pipeline_root_sha = _canonical_sha256(pipeline_root_payload)
    rows: list[dict[str, Any]] = []
    for row in normalized["rows"]:
        job_id = _text(row.get("formal_d92_source_job_id"), "normalized source D92 job")
        assets = assets_by_job[job_id]
        scene = _text(row.get("scene"), "normalized scene")
        slice_receipt = assets.score_by_scene[scene]
        rows.append(
            {
                "row_id": row["row_id"],
                "receiver_id": row["receiver_id"],
                "k_shot": row["k_shot"],
                "scene": scene,
                "source_d92_job_id": job_id,
                "d92_retry2_manifest_sha256": retry2_manifest_sha256,
                "formal_d92_score_row_key": slice_receipt["key"],
                "formal_d92_score_row_sha256": slice_receipt["sha256"],
            }
        )
    reference: dict[str, Any] = {
        "schema": scorer.PAIRED_FORMAL_D92_REFERENCE_SCHEMA,
        "paired_prediction_sha256": normalized["paired_prediction_sha256"],
        "prepared_plan_sha256": normalized["prepared_plan_sha256"],
        "method_lock_sha256": normalized["method_lock_sha256"],
        "pair_manifest_sha256": normalized["pair_manifest_sha256"],
        "normalized_prediction_sha256": normalized["normalized_prediction_sha256"],
        "pipeline_receipt_sha256": pipeline_root_sha,
        "row_count": scorer.ROW_COUNT,
        "rows": rows,
    }
    reference["formal_d92_reference_sha256"] = _canonical_sha256(reference)
    return reference, pipeline_root_payload


def build_d127_s0_truth_assets(
    *,
    paired_prediction_path: str | Path,
    expected_paired_prediction_sha256: str,
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
    """Build three exclusive scorer-side artifacts after a frozen open event.

    The first function call opens only D127 truth-free artifacts and validates
    the durable event.  D92 scorer-side files are deliberately untouched until
    after that return, which keeps query truth and roles out of inference.
    """

    normalized, method_lock, event_file_sha, event_receipt_sha = _read_opened_context(
        paired_prediction_path=paired_prediction_path,
        expected_paired_prediction_sha256=expected_paired_prediction_sha256,
        prepared_plan_path=prepared_plan_path,
        expected_prepared_plan_sha256=expected_prepared_plan_sha256,
        method_lock_path=method_lock_path,
        expected_method_lock_sha256=expected_method_lock_sha256,
        truth_open_event_path=truth_open_event_path,
        expected_truth_open_event_sha256=expected_truth_open_event_sha256,
    )
    truth_target = _assert_output_absent(truth_catalog_output, "D127 truth catalog")
    formal_target = _assert_output_absent(formal_d92_reference_output, "D127 formal D92 reference")
    receipt_target = _assert_output_absent(build_receipt_output, "D127 truth-assets receipt")

    # The following D92 reads are intentionally below the durable-open check.
    retry2_root = _regular_directory(d92_retry2_root, "D92 retry2 root")
    matrix_path = _regular_file(d92_retry2_manifest_path, "D92 retry2 matrix manifest")
    _require(matrix_path.parent == retry2_root, "D92 retry2 matrix manifest/root drift")
    matrix, matrix_sha, _matrix_path = _read_json(
        matrix_path,
        name="D92 retry2 matrix manifest",
        expected_sha256=expected_d92_retry2_manifest_sha256,
    )
    expected_lock_manifest_sha = _sha(method_lock["s0_matrix"]["d92_retry2_manifest_sha256"], "D127 method-lock D92 retry2 manifest")
    _require(matrix_sha == expected_lock_manifest_sha, "D127 method-lock/D92 retry2 manifest drift")
    seed, _source_k5, scenes, routes = _expected_routes(method_lock, normalized)
    jobs = _index_d92_jobs(matrix)
    assets_by_job: dict[str, _D92JobAssets] = {}
    for row in normalized["rows"]:
        row_id = _text(row.get("row_id"), "normalized row ID")
        source_k, job_id = routes[row_id]
        _require(row.get("formal_d92_retry2_manifest_sha256") == matrix_sha, "D127 normalized D92 retry2 manifest drift")
        job = jobs.get(job_id)
        _require(job is not None, "D127 source D92 job is absent from retry2 matrix")
        _require(job.get("receiver") == row.get("receiver_id"), "D127/D92 receiver route drift")
        if job_id not in assets_by_job:
            assets_by_job[job_id] = _load_d92_job_assets(
                retry2_root=retry2_root,
                job=job,
                seed=seed,
                source_k_shot=source_k,
                scenes=scenes,
            )
    _require(len(assets_by_job) == 6, "D127 S0 must bind exactly six D92 source jobs")

    truth_catalog = _truth_catalog(normalized=normalized, assets_by_job=assets_by_job)
    formal_reference, pipeline_root_payload = _formal_reference(
        normalized=normalized,
        assets_by_job=assets_by_job,
        retry2_manifest_sha256=matrix_sha,
    )
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
            "source_paths": dict(assets.source_paths),
            "truth_query_count_used": sum(
                len(row["after_query_ids"])
                for row in normalized["rows"]
                if row["formal_d92_source_job_id"] == assets.job_id
            ),
        }
        for assets in sorted(assets_by_job.values(), key=lambda item: item.job_id)
    ]

    _write_exclusive_json(truth_target, truth_catalog, name="D127 truth catalog")
    _write_exclusive_json(formal_target, formal_reference, name="D127 formal D92 reference")
    receipt: dict[str, Any] = {
        "schema": TRUTH_ASSETS_BUILD_RECEIPT_SCHEMA,
        "truth_open_event_file_sha256": event_file_sha,
        "truth_open_event_sha256": event_receipt_sha,
        "paired_prediction_sha256": normalized["paired_prediction_sha256"],
        "prepared_plan_sha256": normalized["prepared_plan_sha256"],
        "method_lock_sha256": normalized["method_lock_sha256"],
        "pair_manifest_sha256": normalized["pair_manifest_sha256"],
        "normalized_prediction_sha256": normalized["normalized_prediction_sha256"],
        "d92_retry2_root": str(retry2_root),
        "d92_retry2_matrix_manifest": str(matrix_path),
        "d92_retry2_manifest_sha256": matrix_sha,
        "pipeline_receipt_root": pipeline_root_payload,
        "source_jobs": source_jobs,
        "row_count": scorer.ROW_COUNT,
        "query_count": truth_catalog["query_count"],
        "truth_catalog_path": str(truth_target),
        "truth_catalog_sha256": _sha256_file(truth_target),
        "formal_d92_reference_path": str(formal_target),
        "formal_d92_reference_sha256": _sha256_file(formal_target),
        "predictor_package_read": False,
        "prediction_values_read": False,
        "query_truth_opened_after_durable_event": True,
    }
    receipt["truth_assets_build_receipt_sha256"] = _canonical_sha256(receipt)
    _write_exclusive_json(receipt_target, receipt, name="D127 truth-assets receipt")
    return {
        "status": "D127_S0_TRUTH_ASSETS_BUILT",
        "row_count": scorer.ROW_COUNT,
        "query_count": truth_catalog["query_count"],
        "truth_catalog": str(truth_target.resolve()),
        "truth_catalog_sha256": _sha256_file(truth_target),
        "formal_d92_reference": str(formal_target.resolve()),
        "formal_d92_reference_sha256": _sha256_file(formal_target),
        "build_receipt": str(receipt_target.resolve()),
        "build_receipt_sha256": _sha256_file(receipt_target),
        "truth_open_event_sha256": event_receipt_sha,
        "prediction_values_read": False,
    }


__all__ = [
    "D127S0TruthAssetsError",
    "D92_PIPELINE_RECEIPT_ROOT_SCHEMA",
    "D92_SCENE_SCORE_SLICE_SCHEMA",
    "TRUTH_ASSETS_BUILD_RECEIPT_SCHEMA",
    "build_d127_s0_truth_assets",
]
