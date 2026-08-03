from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import hashlib
import json

import pytest

from cvsrffi import stage2_d128_a_one18 as one
from cvsrffi import stage2_d128_a_one18_scorer as scorer


RECEIVERS = ("20-1", "3-19", "7-14")
SCENES = ("leo_clear_weak", "leo_low_elev_weak", "leo_rain_weak")


def _sha(index: int) -> str:
    return f"{index:064x}"


def _normalized() -> dict:
    rows: list[dict] = []
    after_ids_all: list[str] = []
    for receiver in RECEIVERS:
        for k_shot in (1, 5):
            for scene in SCENES:
                row_id = f"d128-{receiver}-k{k_shot}-{scene}"
                before_ids = [f"{row_id}-old0", f"{row_id}-old1"]
                after_ids = [*before_ids, f"{row_id}-new0"]
                before_arms = {arm: ["old0", "old1"] for arm in one.ARM_IDS}
                base = ["old0", "old0", "old0"]
                if k_shot == 1:
                    adapted = ["old0", "old1", "new0"]
                    joint = adapted
                else:
                    adapted = ["old0", "old1", "old0"]
                    joint = ["old0", "old1", "new0"]
                rows.append(
                    {
                        "row_id": row_id,
                        "receiver_id": receiver,
                        "k_shot": k_shot,
                        "scene": scene,
                        "old_classes": ["old0", "old1"],
                        "new_classes": ["new0"],
                        "before_query_ids": before_ids,
                        "after_query_ids": after_ids,
                        "before_arms": before_arms,
                        "after_arms": {"M0": base, "M_DA": adapted, "M_L92": base, "M_JOINT": joint},
                        "formal_d92_source_job_id": f"rx_{receiver.replace('-', '_')}__seed_713102__k_{10 if k_shot == 5 else k_shot}__new_20",
                        "formal_d92_retry2_manifest_sha256": _sha(900),
                    }
                )
                after_ids_all.extend(after_ids)
    document = {
        "schema": scorer.NORMALIZED_SCHEMA,
        "prediction_sha256": _sha(100),
        "prepared_plan_sha256": _sha(101),
        "method_lock_sha256": _sha(102),
        "checkpoint_sha256": _sha(103),
        "pair_manifest_sha256": _sha(104),
        "candidate_id": one.CANDIDATE_ID,
        "arm_ids": list(one.ARM_IDS),
        "row_count": 18,
        "after_query_id_root_sha256": one._opaque_root(after_ids_all),
        "rows": rows,
    }
    document["normalized_prediction_sha256"] = scorer.canonical_sha256(document)
    return document


def _sign(value: dict, field: str) -> dict:
    unsigned = dict(value)
    unsigned.pop(field, None)
    value[field] = scorer.canonical_sha256(unsigned)
    return value


def _truth(normalized: dict) -> dict:
    queries: list[dict] = []
    for row in normalized["rows"]:
        queries.extend(
            [
                {"opaque_query_id": row["after_query_ids"][0], "label": "old0", "role": "old"},
                {"opaque_query_id": row["after_query_ids"][1], "label": "old1", "role": "old"},
                {"opaque_query_id": row["after_query_ids"][2], "label": "new0", "role": "new"},
            ]
        )
    value = {
        "schema": scorer.TRUTH_CATALOG_SCHEMA,
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
    return _sign(value, "truth_catalog_sha256")


def _formal(normalized: dict) -> dict:
    rows = [
        {
            "row_id": row["row_id"],
            "receiver_id": row["receiver_id"],
            "k_shot": row["k_shot"],
            "scene": row["scene"],
            "source_d92_job_id": row["formal_d92_source_job_id"],
            "d92_retry2_manifest_sha256": row["formal_d92_retry2_manifest_sha256"],
            "formal_d92_score_row_key": f"formal::{row['row_id']}",
            "formal_d92_score_row_sha256": _sha(index + 500),
        }
        for index, row in enumerate(normalized["rows"])
    ]
    value = {
        "schema": scorer.FORMAL_D92_REFERENCE_SCHEMA,
        "candidate_id": one.CANDIDATE_ID,
        "prediction_sha256": normalized["prediction_sha256"],
        "prepared_plan_sha256": normalized["prepared_plan_sha256"],
        "method_lock_sha256": normalized["method_lock_sha256"],
        "checkpoint_sha256": normalized["checkpoint_sha256"],
        "pair_manifest_sha256": normalized["pair_manifest_sha256"],
        "normalized_prediction_sha256": normalized["normalized_prediction_sha256"],
        "pipeline_receipt_sha256": _sha(901),
        "row_count": 18,
        "rows": rows,
    }
    return _sign(value, "formal_d92_reference_sha256")


def _file_sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_same_row_score_reports_all_three_direction_gates_without_promotion() -> None:
    normalized = _normalized()
    event = scorer.build_d128_a_one18_truth_open_event(normalized)
    result = scorer.score_d128_a_one18(
        normalized_prediction=normalized,
        truth_open_event=event,
        truth_catalog=_truth(normalized),
        formal_d92_reference=_formal(normalized),
    )
    assert result["row_count"] == 18 and result["metric_row_count"] == 72
    decision = result["one_shot_direction_decision"]
    assert decision["G1_M_DA_over_M0_pass"]
    assert decision["G2_K5_M_JOINT_over_M_DA_pass"]
    assert decision["G3_M_JOINT_over_M0_pass"]
    assert decision["all_three_direction_pass"]
    assert decision["promotion_action"] == "NONE_REPORT_ONLY"
    assert len(result["aggregates"]["scope"]) == 4
    metric = result["same_row_results"][0]["candidate_arm_metrics"][0]
    assert {"B_old", "A_old", "seen_new", "H_old_new", "old_per_class_floor", "forgetting", "total_correct_count"}.issubset(metric)


def test_truth_is_not_opened_when_event_fails_first() -> None:
    normalized = _normalized()
    event = scorer.build_d128_a_one18_truth_open_event(normalized)
    event["truth_open"] = False
    _sign(event, "truth_open_event_sha256")
    with pytest.raises(scorer.D128AOne18ScorerError, match="truth-open event"):
        scorer.score_d128_a_one18(
            normalized_prediction=normalized,
            truth_open_event=event,
            truth_catalog={"must_not": "be opened"},
            formal_d92_reference={"must_not": "be opened"},
        )


def test_truth_asset_builder_creates_d128_owned_assets_after_open(tmp_path, monkeypatch) -> None:
    normalized = _normalized()
    retry_root = tmp_path / "retry2"
    retry_root.mkdir()
    matrix_path = retry_root / "matrix_manifest.json"
    matrix_path.write_text("{}", encoding="utf-8")
    jobs: dict[str, dict] = {}
    assets_by_job: dict[str, SimpleNamespace] = {}
    for row in normalized["rows"]:
        job_id = row["formal_d92_source_job_id"]
        if job_id in jobs:
            continue
        jobs[job_id] = {"job_id": job_id, "receiver": row["receiver_id"]}
        query_truth = {}
        for matched in normalized["rows"]:
            if matched["formal_d92_source_job_id"] != job_id:
                continue
            query_truth[matched["after_query_ids"][0]] = {"label": "old0", "role": "target_old"}
            query_truth[matched["after_query_ids"][1]] = {"label": "old1", "role": "target_old"}
            query_truth[matched["after_query_ids"][2]] = {"label": "new0", "role": "target_new"}
        assets_by_job[job_id] = SimpleNamespace(
            job_id=job_id,
            receiver=row["receiver_id"],
            seed=713102,
            source_k_shot=10 if row["k_shot"] == 5 else 1,
            new_class_count=20,
            pipeline_receipt_sha256=_sha(len(assets_by_job) + 600),
            row_manifest_sha256=_sha(len(assets_by_job) + 620),
            registration_pair_sha256=_sha(len(assets_by_job) + 640),
            truth_sidecar_sha256=_sha(len(assets_by_job) + 660),
            score_artifact_sha256=_sha(len(assets_by_job) + 680),
            truth_by_query_id=query_truth,
            score_by_scene={scene: {"key": f"{job_id}::{scene}", "sha256": _sha(700 + index)} for index, scene in enumerate(SCENES)},
        )
    routes = {
        row["row_id"]: (10 if row["k_shot"] == 5 else 1, row["formal_d92_source_job_id"])
        for row in normalized["rows"]
    }
    method_lock = {"s0_matrix": {"d92_retry2_manifest_sha256": _sha(900)}}
    monkeypatch.setattr(
        scorer,
        "_read_opened_context",
        lambda **_kwargs: (normalized, method_lock, _sha(920), _sha(921)),
    )
    monkeypatch.setattr(
        scorer.d92_assets,
        "_read_json",
        lambda *_args, **_kwargs: ({"schema": "cvs.phase2.somph_diag_125_stability.v1", "jobs": list(jobs.values())}, _sha(900), matrix_path),
    )
    monkeypatch.setattr(scorer.d92_assets, "_expected_routes", lambda *_args, **_kwargs: (713102, 10, SCENES, routes))
    monkeypatch.setattr(scorer.d92_assets, "_index_d92_jobs", lambda _matrix: jobs)
    monkeypatch.setattr(scorer.d92_assets, "_load_d92_job_assets", lambda **kwargs: assets_by_job[kwargs["job"]["job_id"]])
    result = scorer.build_d128_a_one18_truth_assets(
        prediction_path=tmp_path / "prediction.json",
        expected_prediction_sha256=_sha(100),
        prepared_plan_path=tmp_path / "plan.json",
        expected_prepared_plan_sha256=_sha(101),
        method_lock_path=tmp_path / "method_lock.json",
        expected_method_lock_sha256=_sha(102),
        truth_open_event_path=tmp_path / "open.json",
        expected_truth_open_event_sha256=_sha(103),
        d92_retry2_root=retry_root,
        d92_retry2_manifest_path=matrix_path,
        expected_d92_retry2_manifest_sha256=_sha(900),
        truth_catalog_output=tmp_path / "out" / "truth.json",
        formal_d92_reference_output=tmp_path / "out" / "formal.json",
        build_receipt_output=tmp_path / "out" / "receipt.json",
    )
    truth = json.loads(Path(result["truth_catalog"]).read_text(encoding="utf-8"))
    formal = json.loads(Path(result["formal_d92_reference"]).read_text(encoding="utf-8"))
    assert result["row_count"] == 18 and truth["schema"] == scorer.TRUTH_CATALOG_SCHEMA
    assert formal["schema"] == scorer.FORMAL_D92_REFERENCE_SCHEMA and len(formal["rows"]) == 18
    assert _file_sha(Path(result["truth_catalog"])) == result["truth_catalog_sha256"]
