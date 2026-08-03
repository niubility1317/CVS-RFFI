from __future__ import annotations

from copy import deepcopy
import hashlib
import json
from pathlib import Path

import pytest

from cvsrffi import stage2_d127_s0_package_adapter as adapter
from cvsrffi import stage2_d127_s0_scorer as scorer
from cvsrffi import stage2_d127_s0_truth_assets as assets


RECEIVERS = ("20-1", "3-19", "7-14")
SCENES = ("leo_clear_weak", "leo_low_elev_weak", "leo_rain_weak")


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write(path: Path, value: dict) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=True, sort_keys=True), encoding="utf-8")
    return _sha(path)


def _job_id(receiver: str, source_k: int) -> str:
    return f"rx_{receiver.replace('-', '_')}__seed_713102__k_{source_k}__new_20"


def _normalized(matrix_sha: str) -> dict:
    rows: list[dict] = []
    all_after: list[str] = []
    for receiver in RECEIVERS:
        for k_shot in (1, 5):
            source_k = 10 if k_shot == 5 else 1
            job_id = _job_id(receiver, source_k)
            for scene in SCENES:
                after = [
                    f"qid-{job_id}-{scene}-old0",
                    f"qid-{job_id}-{scene}-old1",
                    f"qid-{job_id}-{scene}-new0",
                ]
                all_after.extend(after)
                rows.append(
                    {
                        "row_id": f"d127-{receiver}-k{k_shot}-{scene}",
                        "receiver_id": receiver,
                        "k_shot": k_shot,
                        "scene": scene,
                        "old_classes": ["old0", "old1"],
                        "new_classes": ["new0"],
                        "before_query_ids": after[:2],
                        "after_query_ids": after,
                        "before_query_ids_sha256": scorer.canonical_sha256(after[:2]),
                        "after_query_ids_sha256": scorer.canonical_sha256(after),
                        "formal_d92_source_job_id": job_id,
                        "formal_d92_retry2_manifest_sha256": matrix_sha,
                    }
                )
    rows.sort(key=lambda item: (item["receiver_id"], item["k_shot"], item["scene"], item["row_id"]))
    value: dict = {
        "schema": scorer.PAIRED_NORMALIZED_SCHEMA,
        "paired_prediction_sha256": "b" * 64,
        "prepared_plan_sha256": "c" * 64,
        "method_lock_sha256": "a" * 64,
        "pair_manifest_sha256": "d" * 64,
        "row_count": 18,
        "candidate_ids": list(scorer.CANDIDATE_IDS),
        "arm_ids": list(scorer.ARM_IDS),
        "after_query_id_root_sha256": scorer.canonical_sha256(sorted(all_after)),
        "rows": rows,
    }
    value["normalized_prediction_sha256"] = scorer.canonical_sha256(value)
    return value


def _write_d92_job(root: Path, *, receiver: str, source_k: int) -> dict:
    job_id = _job_id(receiver, source_k)
    job_root = root / "jobs" / job_id
    truth_rows: list[dict] = []
    for scene in SCENES:
        for label, role in (("old0", "target_old"), ("old1", "target_old"), ("new0", "target_new")):
            truth_rows.append(
                {
                    "query_token": f"qid-{job_id}-{scene}-{label}",
                    "true_class_handle": label,
                    "transmitter_label": f"tx-{label}",
                    "evaluation_role": role,
                }
            )
    truth_path = job_root / "offline" / "scorer" / "truth_sidecar.json"
    truth_sha = _write(truth_path, {"schema": "cvs.phase2.query_truth_sidecar.v2", "rows": truth_rows})
    score_path = job_root / "scorer" / "diag_cosine_score.json"
    score_sha = _write(
        score_path,
        {
            "schema": "cvs.phase2.diag_cosine_dev_pair_score.v1",
            "truth_sidecar_sha256": truth_sha,
            "before": {"by_scenario": {scene: {"query_count": 2, "old_acc": 0.5} for scene in SCENES}},
            "after": {"by_scenario": {scene: {"query_count": 3, "old_acc": 0.5, "seen_new_acc": 0.5} for scene in SCENES}},
        },
    )
    row_manifest_path = job_root / "offline" / "scorer" / "row_manifest.json"
    row_manifest_sha = _write(
        row_manifest_path,
        {
            "schema": "cvs.phase2.somph_row_manifest.v2",
            "receiver": receiver,
            "seed": 713102,
            "k_shot": source_k,
            "new_class_count": 20,
            "scenarios": list(SCENES),
        },
    )
    pair_path = job_root / "scorer" / "registration_pair.final.json"
    pair_sha = _write(
        pair_path,
        {
            "schema": "cvs.phase2.somph_registration_pair.v1",
            "old_query_physical_ids_sha256_before": "1" * 64,
            "old_query_physical_ids_sha256_after": "1" * 64,
            "old_support_physical_ids_sha256_before": "2" * 64,
            "old_support_physical_ids_sha256_after": "2" * 64,
        },
    )
    _write(
        job_root / "pipeline_receipt.json",
        {
            "schema": "cvs.phase2.somph_diag_row_pipeline.v1",
            "receiver": receiver,
            "seed": 713102,
            "k_shot": source_k,
            "new_class_count": 20,
            "score_artifact_sha256": score_sha,
            "row_manifest_sha256": row_manifest_sha,
            "registration_pair_final_sha256": pair_sha,
        },
    )
    return {"job_id": job_id, "receiver": receiver, "seed": 713102, "k_shot": source_k, "new_class_count": 20, "output_root": str(job_root.resolve())}


def _environment(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> dict:
    retry_root = tmp_path / "d92_retry2"
    jobs = [_write_d92_job(retry_root, receiver=receiver, source_k=source_k) for receiver in RECEIVERS for source_k in (1, 10)]
    matrix_path = retry_root / "matrix_manifest.json"
    matrix_sha = _write(matrix_path, {"schema": "cvs.phase2.somph_diag_125_stability.v1", "jobs": jobs})
    normalized = _normalized(matrix_sha)
    event = scorer.build_d127_s0_truth_open_event(normalized)
    event_path = tmp_path / "truth_open_event.json"
    event_sha = _write(event_path, event)
    lock = {
        "s0_matrix": {
            "seed": 713102,
            "k5_source_pool_k": 10,
            "scenes": list(SCENES),
            "d92_retry2_manifest_sha256": matrix_sha,
        }
    }
    monkeypatch.setattr(scorer, "prepare_d127_s0_scoring_inputs", lambda **_: {"normalized_prediction": normalized})
    monkeypatch.setattr(adapter, "load_d127_s0_method_lock", lambda *_args, **_kwargs: (lock, "a" * 64, {}))
    return {
        "retry_root": retry_root,
        "matrix_path": matrix_path,
        "matrix_sha": matrix_sha,
        "normalized": normalized,
        "event_path": event_path,
        "event_sha": event_sha,
    }


def _run(env: dict, tmp_path: Path) -> dict:
    return assets.build_d127_s0_truth_assets(
        paired_prediction_path=tmp_path / "paired_prediction.json",
        expected_paired_prediction_sha256="b" * 64,
        prepared_plan_path=tmp_path / "prepared_plan.json",
        expected_prepared_plan_sha256="c" * 64,
        method_lock_path=tmp_path / "method_lock.json",
        expected_method_lock_sha256="a" * 64,
        truth_open_event_path=env["event_path"],
        expected_truth_open_event_sha256=env["event_sha"],
        d92_retry2_root=env["retry_root"],
        d92_retry2_manifest_path=env["matrix_path"],
        expected_d92_retry2_manifest_sha256=env["matrix_sha"],
        truth_catalog_output=tmp_path / "out" / "truth.json",
        formal_d92_reference_output=tmp_path / "out" / "formal.json",
        build_receipt_output=tmp_path / "out" / "receipt.json",
    )


def test_builds_18_same_row_truth_and_formal_assets_from_scorer_side_sources(tmp_path, monkeypatch) -> None:
    env = _environment(tmp_path, monkeypatch)
    result = _run(env, tmp_path)
    truth = json.loads(Path(result["truth_catalog"]).read_text(encoding="utf-8"))
    formal = json.loads(Path(result["formal_d92_reference"]).read_text(encoding="utf-8"))
    receipt = json.loads(Path(result["build_receipt"]).read_text(encoding="utf-8"))
    assert result["row_count"] == 18 and result["query_count"] == 54
    assert truth["schema"] == scorer.PAIRED_TRUTH_CATALOG_SCHEMA
    assert {(item["label"], item["role"]) for item in truth["queries"]} == {("old0", "old"), ("old1", "old"), ("new0", "new")}
    assert formal["schema"] == scorer.PAIRED_FORMAL_D92_REFERENCE_SCHEMA
    assert len(formal["rows"]) == 18
    assert all("__k_10__" in row["source_d92_job_id"] for row in formal["rows"] if row["k_shot"] == 5)
    assert len(receipt["source_jobs"]) == 6
    assert receipt["predictor_package_read"] is False and receipt["prediction_values_read"] is False
    scorer._open_paired_truth_catalog(truth, normalized_prediction=env["normalized"])
    scorer._open_paired_formal_d92_reference(formal, normalized_prediction=env["normalized"])


def test_rejects_source_score_hash_drift(tmp_path, monkeypatch) -> None:
    env = _environment(tmp_path, monkeypatch)
    score = env["retry_root"] / "jobs" / _job_id("20-1", 1) / "scorer" / "diag_cosine_score.json"
    score.write_text("{}", encoding="utf-8")
    with pytest.raises(assets.D127S0TruthAssetsError, match="pipeline/formal-score hash drift"):
        _run(env, tmp_path)


def test_rejects_k5_route_that_does_not_use_k10_source_pool(tmp_path, monkeypatch) -> None:
    env = _environment(tmp_path, monkeypatch)
    broken = env["normalized"]
    row = next(item for item in broken["rows"] if item["k_shot"] == 5)
    row["formal_d92_source_job_id"] = _job_id(row["receiver_id"], 5)
    unsigned = dict(broken)
    unsigned.pop("normalized_prediction_sha256")
    broken["normalized_prediction_sha256"] = scorer.canonical_sha256(unsigned)
    event = scorer.build_d127_s0_truth_open_event(broken)
    env["event_sha"] = _write(env["event_path"], event)
    with pytest.raises(assets.D127S0TruthAssetsError, match="source D92 job drift"):
        _run(env, tmp_path)


def test_refuses_to_overwrite_truth_output_before_opening_d92_sources(tmp_path, monkeypatch) -> None:
    env = _environment(tmp_path, monkeypatch)
    target = tmp_path / "out" / "truth.json"
    target.parent.mkdir(parents=True)
    target.write_text("occupied", encoding="utf-8")
    with pytest.raises(assets.D127S0TruthAssetsError, match="truth catalog output already exists"):
        _run(env, tmp_path)
