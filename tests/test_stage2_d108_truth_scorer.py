"""Focused truth-after-seal tests for the D108 Target125 scorer."""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
from typing import Any

import pytest

from cvsrffi import stage2_d108_truth_scorer as scorer


SCENES = ["leo_clear_weak", "leo_low_elev_weak", "leo_rain_weak"]
ACCESS_LEDGER = {
    "clean_source_runtime_access": False,
    "query_fit_access": False,
    "query_update_access": False,
    "query_truth_access": False,
    "query_role_access": False,
    "query_selection_access": False,
}


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_json(path: Path, value: object) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(
        json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode(
            "utf-8"
        )
    )
    return _sha256_file(path)


def _resign(value: dict[str, object], field: str) -> None:
    value.pop(field, None)
    value[field] = scorer.canonical_sha256(value)


def _outer(index: int) -> dict[str, object]:
    receiver = f"rx-{index:03d}"
    seed = 90_000 + index
    return {
        "outer_id": f"d108-rx-{receiver}__seed-{seed}__k-1__new-1",
        "receiver": receiver,
        "seed": seed,
        "k_shot": 1,
        "new_count": 1,
        "old_classes": ["old-a", "old-b"],
        "new_classes": ["new-a"],
    }


def _predictions(arm: str, phase: str) -> list[str]:
    if phase == "before":
        return ["old-a", "old-a"] if arm in {"M0", "M_HEAD"} else ["old-a", "old-b"]
    if arm == "M0":
        return ["old-a", "old-a", "old-a"]
    if arm == "M_DA":
        return ["old-a", "old-b", "old-a"]
    if arm == "M_HEAD":
        return ["old-a", "old-a", "new-a"]
    if arm == "M_JOINT":
        return ["old-a", "old-b", "new-a"]
    raise AssertionError(arm)


def _build_case(root: Path) -> dict[str, object]:
    prediction_root = root / "prediction"
    artifact_root = prediction_root / "predictions"
    outer_rows = [_outer(index) for index in range(scorer.OUTER_JOB_COUNT)]
    surfaces: list[dict[str, object]] = []
    truth_surfaces: list[dict[str, object]] = []
    for outer in outer_rows:
        for scene in SCENES:
            query_ids = [
                f"{outer['outer_id']}::{scene}::old-a",
                f"{outer['outer_id']}::{scene}::old-b",
                f"{outer['outer_id']}::{scene}::new-a",
            ]
            for phase in scorer.PHASES:
                phase_ids = query_ids[:2] if phase == "before" else query_ids
                phase_truth = ["old-a", "old-b"] if phase == "before" else ["old-a", "old-b", "new-a"]
                truth_surfaces.append(
                    {
                        "outer_id": outer["outer_id"], "receiver": outer["receiver"],
                        "seed": outer["seed"], "k_shot": outer["k_shot"],
                        "new_count": outer["new_count"], "scene": scene, "phase": phase,
                        "ordered_query_physical_ids": phase_ids, "labels": phase_truth,
                    }
                )
                for arm in scorer.ARMS:
                    surface_id = f"{outer['outer_id']}__scene-{scene}__arm-{arm}__phase-{phase}"
                    registered = outer["old_classes"] if phase == "before" else [*outer["old_classes"], *outer["new_classes"]]
                    labels = _predictions(arm, phase)
                    surface: dict[str, object] = {
                        "surface_id": surface_id, "outer_id": outer["outer_id"],
                        "receiver": outer["receiver"], "seed": outer["seed"],
                        "k_shot": outer["k_shot"], "new_count": outer["new_count"],
                        "scene": scene, "arm": arm, "phase": phase,
                        "registered_classes": registered,
                        "prediction_artifact": f"predictions/{surface_id}.json",
                        "prediction_artifact_sha256": "",
                        "ordered_query_physical_ids": phase_ids,
                        "ordered_query_physical_ids_sha256": scorer.canonical_sha256(phase_ids),
                        "predicted_labels": labels,
                        "predicted_labels_sha256": scorer.canonical_sha256(labels),
                        "access_ledger": copy.deepcopy(ACCESS_LEDGER),
                        "truth_open": False, "immutable": True,
                    }
                    artifact = {
                        key: copy.deepcopy(value)
                        for key, value in surface.items()
                        if key not in {"prediction_artifact", "prediction_artifact_sha256"}
                    }
                    artifact["schema"] = scorer.PREDICTION_ARTIFACT_SCHEMA
                    _resign(artifact, "artifact_receipt_sha256")
                    surface["prediction_artifact_sha256"] = _write_json(
                        artifact_root / f"{surface_id}.json", artifact
                    )
                    surfaces.append(surface)
    manifest: dict[str, object] = {
        "schema": scorer.PREDICTION_MANIFEST_SCHEMA,
        "candidate_id": "D108-CB-RRC-SMME/r1", "protocol_schema": scorer.PROTOCOL_SCHEMA,
        "manifest_sealed": True, "truth_open": False,
        "outer_job_count": scorer.OUTER_JOB_COUNT, "scene_row_count": scorer.SCENE_ROW_COUNT,
        "arm_pair_count": scorer.ARM_PAIR_COUNT, "surface_count": scorer.SURFACE_COUNT,
        "scenes": SCENES, "arms": list(scorer.ARMS), "phases": list(scorer.PHASES),
        "outer_rows": outer_rows, "access_ledger": copy.deepcopy(ACCESS_LEDGER), "surfaces": surfaces,
    }
    _resign(manifest, "manifest_sha256")
    manifest_path = prediction_root / "prediction_manifest.json"
    manifest_sha = _write_json(manifest_path, manifest)
    truth: dict[str, object] = {
        "schema": scorer.TRUTH_CATALOG_SCHEMA, "truth_open": True,
        "prediction_manifest_sha256": manifest["manifest_sha256"],
        "outer_job_count": scorer.OUTER_JOB_COUNT, "scene_row_count": scorer.SCENE_ROW_COUNT,
        "truth_surface_count": scorer.TRUTH_SURFACE_COUNT, "scenes": SCENES,
        "phases": list(scorer.PHASES), "surfaces": truth_surfaces,
    }
    _resign(truth, "truth_catalog_sha256")
    truth_path = root / "truth_catalog.json"
    truth_sha = _write_json(truth_path, truth)
    return {
        "manifest_path": manifest_path, "manifest_sha": manifest_sha,
        "truth_path": truth_path, "truth_sha": truth_sha, "score_root": root / "score",
    }


def _score(case: dict[str, object], output_dir: Path | None = None) -> dict[str, object]:
    return scorer.score_d108_target125(
        prediction_manifest_path=Path(case["manifest_path"]),
        expected_prediction_manifest_file_sha256=str(case["manifest_sha"]),
        truth_catalog_path=Path(case["truth_path"]),
        expected_truth_catalog_file_sha256=str(case["truth_sha"]),
        output_dir=Path(case["score_root"]) if output_dir is None else output_dir,
    )


def test_truth_opens_only_after_complete_prediction_closure_and_scores_same_rows(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    case = _build_case(tmp_path / "case")
    original = scorer._read_json_regular

    def _spy(path, **kwargs):  # type: ignore[no-untyped-def]
        if Path(path) == Path(case["truth_path"]):
            assert (Path(case["score_root"]) / "truth_open_event.json").is_file()
        return original(path, **kwargs)

    monkeypatch.setattr(scorer, "_read_json_regular", _spy)
    result = _score(case)
    score = json.loads(Path(result["score_manifest"]).read_text(encoding="utf-8"))
    assert score["scene_same_row_count"] == 375
    assert score["scene_arm_metric_row_count"] == 1500
    assert score["outer_arm_aggregate_row_count"] == 500
    arms = {row["arm"]: row for row in score["scene_same_rows"][0]["arms"]}
    assert arms["M_HEAD"]["H_old_new"] == pytest.approx(200.0 * 50.0 / 150.0)
    assert arms["M_JOINT"]["H_old_new"] == pytest.approx(100.0)
    assert score["target_verdict_summary"]["performance_early_stop_or_selection_performed"] is False


def test_prediction_leak_or_missing_surface_is_rejected_before_truth_open(tmp_path: Path) -> None:
    missing = _build_case(tmp_path / "missing")
    manifest = json.loads(Path(missing["manifest_path"]).read_text(encoding="utf-8"))
    manifest["surfaces"] = manifest["surfaces"][:-1]
    _resign(manifest, "manifest_sha256")
    missing["manifest_sha"] = _write_json(Path(missing["manifest_path"]), manifest)
    with pytest.raises(scorer.D108TruthScorerError, match="3000 surfaces"):
        _score(missing)
    assert not (Path(missing["score_root"]) / "truth_open_event.json").exists()

    leaked = _build_case(tmp_path / "leaked")
    manifest = json.loads(Path(leaked["manifest_path"]).read_text(encoding="utf-8"))
    manifest["surfaces"][0]["truth_labels"] = ["old-a", "old-b"]
    _resign(manifest, "manifest_sha256")
    leaked["manifest_sha"] = _write_json(Path(leaked["manifest_path"]), manifest)
    with pytest.raises(scorer.D108TruthScorerError, match="forbidden truth/role field"):
        _score(leaked)
    assert not (Path(leaked["score_root"]) / "truth_open_event.json").exists()


def test_build_truth_joins_sealed_tokens_not_day_labels(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    case = _build_case(tmp_path / "build")
    prediction = scorer.validate_d108_prediction_manifest(
        prediction_manifest_path=Path(case["manifest_path"]),
        expected_prediction_manifest_file_sha256=str(case["manifest_sha"]),
    )
    prepared_rows = [
        {
            "outer_id": row["outer_id"], "receiver": row["receiver"], "seed": row["seed"],
            "k_shot": row["k_shot"], "new_count": row["new_count"],
        }
        for row in prediction["outer_rows"]
    ]

    def _prepared(**_kwargs):  # type: ignore[no-untyped-def]
        return {"identity": {}}, {"rows": prepared_rows}

    def _source_truth(*, outer, **_kwargs):  # type: ignore[no-untyped-def]
        result = {}
        for scene in SCENES:
            for label, role in (("old-a", "target_old"), ("old-b", "target_old"), ("new-a", "target_new")):
                token = f"{outer['outer_id']}::{scene}::{label}"
                result[token] = {"label": label, "role": role, "physical_sample_id": token + "::physical"}
        return result

    monkeypatch.setattr(scorer, "_load_prepared_d108_truth_inputs", _prepared)
    monkeypatch.setattr(scorer, "_load_d92_truth_sidecar_for_outer", _source_truth)
    built_path = tmp_path / "build" / "built_truth.json"
    built = scorer.build_d108_target125_truth_catalog(
        prediction_manifest_path=Path(case["manifest_path"]),
        expected_prediction_manifest_file_sha256=str(case["manifest_sha"]),
        plan_manifest_path=tmp_path / "unused-plan.json", expected_plan_file_sha256="a" * 64,
        context_manifest_path=tmp_path / "unused-context.json", expected_context_file_sha256="b" * 64,
        output_path=built_path,
    )
    catalog = json.loads(built_path.read_text(encoding="utf-8"))
    assert built["truth_surface_count"] == 750
    assert catalog["surfaces"][0]["labels"] == ["old-a", "old-b"]
    assert catalog["surfaces"][1]["labels"] == ["old-a", "old-b", "new-a"]


def test_scorer_never_imports_a_k_router() -> None:
    source = Path(scorer.__file__).read_text(encoding="utf-8")
    assert "stage2_d106_k_conditioned_router" not in source
    assert "ROUTE_BY_K" not in source
