from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import pytest

from cvsrffi import stage2_d107_truth_scorer as scorer


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
    row = {
        "receiver": receiver,
        "seed": seed,
        "k_shot": 1,
        "new_count": 1,
        "old_classes": ["old-a", "old-b"],
        "new_classes": ["new-a"],
    }
    row["outer_id"] = (
        f"d107-rx-{receiver}__seed-{seed}__k-{row['k_shot']}__new-{row['new_count']}"
    )
    return {
        "outer_id": row["outer_id"],
        "receiver": receiver,
        "seed": seed,
        "k_shot": 1,
        "new_count": 1,
        "old_classes": ["old-a", "old-b"],
        "new_classes": ["new-a"],
    }


def _predictions(arm: str, phase: str) -> list[str]:
    if phase == "before":
        if arm in {"M0", "M_HEAD"}:
            return ["old-a", "old-a"]
        if arm in {"M_DA", "M_JOINT"}:
            return ["old-a", "old-b"]
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
                phase_query_ids = query_ids[:2] if phase == "before" else query_ids
                phase_truth = ["old-a", "old-b"] if phase == "before" else ["old-a", "old-b", "new-a"]
                truth_surfaces.append(
                    {
                        "outer_id": outer["outer_id"],
                        "receiver": outer["receiver"],
                        "seed": outer["seed"],
                        "k_shot": outer["k_shot"],
                        "new_count": outer["new_count"],
                        "scene": scene,
                        "phase": phase,
                        "ordered_query_physical_ids": phase_query_ids,
                        "labels": phase_truth,
                    }
                )
                for arm in scorer.ARMS:
                    surface_id = (
                        f"{outer['outer_id']}__scene-{scene}__arm-{arm}__phase-{phase}"
                    )
                    registered = (
                        outer["old_classes"]
                        if phase == "before"
                        else [*outer["old_classes"], *outer["new_classes"]]
                    )
                    labels = _predictions(arm, phase)
                    surface: dict[str, object] = {
                        "surface_id": surface_id,
                        "outer_id": outer["outer_id"],
                        "receiver": outer["receiver"],
                        "seed": outer["seed"],
                        "k_shot": outer["k_shot"],
                        "new_count": outer["new_count"],
                        "scene": scene,
                        "arm": arm,
                        "phase": phase,
                        "registered_classes": registered,
                        "prediction_artifact": f"predictions/{surface_id}.json",
                        "prediction_artifact_sha256": "",
                        "ordered_query_physical_ids": phase_query_ids,
                        "ordered_query_physical_ids_sha256": scorer.canonical_sha256(phase_query_ids),
                        "predicted_labels": labels,
                        "predicted_labels_sha256": scorer.canonical_sha256(labels),
                        "access_ledger": copy.deepcopy(ACCESS_LEDGER),
                        "truth_open": False,
                        "immutable": True,
                    }
                    artifact = {
                        key: copy.deepcopy(value)
                        for key, value in surface.items()
                        if key
                        not in {"prediction_artifact", "prediction_artifact_sha256"}
                    }
                    artifact["schema"] = scorer.PREDICTION_ARTIFACT_SCHEMA
                    _resign(artifact, "artifact_receipt_sha256")
                    artifact_path = artifact_root / f"{surface_id}.json"
                    surface["prediction_artifact_sha256"] = _write_json(artifact_path, artifact)
                    surfaces.append(surface)
    manifest: dict[str, object] = {
        "schema": scorer.PREDICTION_MANIFEST_SCHEMA,
        "candidate_id": "D107-SCMKRR/r1",
        "protocol_schema": scorer.PROTOCOL_SCHEMA,
        "manifest_sealed": True,
        "truth_open": False,
        "outer_job_count": scorer.OUTER_JOB_COUNT,
        "scene_row_count": scorer.SCENE_ROW_COUNT,
        "arm_pair_count": scorer.ARM_PAIR_COUNT,
        "surface_count": scorer.SURFACE_COUNT,
        "scenes": SCENES,
        "arms": list(scorer.ARMS),
        "phases": list(scorer.PHASES),
        "outer_rows": outer_rows,
        "access_ledger": copy.deepcopy(ACCESS_LEDGER),
        "surfaces": surfaces,
    }
    _resign(manifest, "manifest_sha256")
    manifest_path = prediction_root / "prediction_manifest.json"
    manifest_file_sha = _write_json(manifest_path, manifest)
    truth: dict[str, object] = {
        "schema": scorer.TRUTH_CATALOG_SCHEMA,
        "truth_open": True,
        "prediction_manifest_sha256": manifest["manifest_sha256"],
        "outer_job_count": scorer.OUTER_JOB_COUNT,
        "scene_row_count": scorer.SCENE_ROW_COUNT,
        "truth_surface_count": scorer.TRUTH_SURFACE_COUNT,
        "scenes": SCENES,
        "phases": list(scorer.PHASES),
        "surfaces": truth_surfaces,
    }
    _resign(truth, "truth_catalog_sha256")
    truth_path = root / "truth_catalog.json"
    truth_file_sha = _write_json(truth_path, truth)
    return {
        "manifest_path": manifest_path,
        "manifest_file_sha": manifest_file_sha,
        "truth_path": truth_path,
        "truth_file_sha": truth_file_sha,
        "score_root": root / "score",
    }


def _load(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def _rewrite_manifest(case: dict[str, object], value: dict[str, object]) -> None:
    _resign(value, "manifest_sha256")
    case["manifest_file_sha"] = _write_json(Path(case["manifest_path"]), value)


def _rewrite_truth(case: dict[str, object], value: dict[str, object]) -> None:
    _resign(value, "truth_catalog_sha256")
    case["truth_file_sha"] = _write_json(Path(case["truth_path"]), value)


def _score(case: dict[str, object]) -> dict[str, object]:
    return scorer.score_d107_target125(
        prediction_manifest_path=Path(case["manifest_path"]),
        expected_prediction_manifest_file_sha256=str(case["manifest_file_sha"]),
        truth_catalog_path=Path(case["truth_path"]),
        expected_truth_catalog_file_sha256=str(case["truth_file_sha"]),
        output_dir=Path(case["score_root"]),
    )


def test_truth_is_opened_only_after_complete_prediction_closure(
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
    assert Path(result["truth_open_event"]).is_file()
    assert Path(result["score_manifest"]).is_file()


def test_rejects_missing_or_duplicate_prediction_surface_before_truth_open(
    tmp_path: Path,
) -> None:
    missing = _build_case(tmp_path / "missing")
    manifest = _load(Path(missing["manifest_path"]))
    manifest["surfaces"] = manifest["surfaces"][:-1]
    _rewrite_manifest(missing, manifest)
    with pytest.raises(scorer.D107TruthScorerError, match="3000 surfaces"):
        _score(missing)
    assert not (Path(missing["score_root"]) / "truth_open_event.json").exists()

    duplicate = _build_case(tmp_path / "duplicate")
    manifest = _load(Path(duplicate["manifest_path"]))
    manifest["surfaces"][1] = copy.deepcopy(manifest["surfaces"][0])
    _rewrite_manifest(duplicate, manifest)
    with pytest.raises(scorer.D107TruthScorerError, match="duplicate|reused"):
        _score(duplicate)
    assert not (Path(duplicate["score_root"]) / "truth_open_event.json").exists()


def test_rejects_manifest_or_prediction_artifact_tamper(tmp_path: Path) -> None:
    manifest_tamper = _build_case(tmp_path / "manifest-tamper")
    manifest_path = Path(manifest_tamper["manifest_path"])
    manifest_path.write_bytes(manifest_path.read_bytes() + b" ")
    with pytest.raises(scorer.D107TruthScorerError, match="prediction manifest SHA mismatch"):
        _score(manifest_tamper)

    artifact_tamper = _build_case(tmp_path / "artifact-tamper")
    manifest = _load(Path(artifact_tamper["manifest_path"]))
    artifact = Path(artifact_tamper["manifest_path"]).parent / manifest["surfaces"][0][
        "prediction_artifact"
    ]
    artifact.write_bytes(artifact.read_bytes() + b" ")
    with pytest.raises(scorer.D107TruthScorerError, match="prediction artifact.*SHA mismatch"):
        _score(artifact_tamper)


def test_rejects_artifact_and_truth_ordered_physical_id_mismatch(tmp_path: Path) -> None:
    artifact_case = _build_case(tmp_path / "artifact-id-mismatch")
    manifest = _load(Path(artifact_case["manifest_path"]))
    surface = manifest["surfaces"][0]
    artifact_path = Path(artifact_case["manifest_path"]).parent / surface["prediction_artifact"]
    artifact = _load(artifact_path)
    artifact["ordered_query_physical_ids"][0] = "wrong-physical-id"
    artifact["ordered_query_physical_ids_sha256"] = scorer.canonical_sha256(
        artifact["ordered_query_physical_ids"]
    )
    _resign(artifact, "artifact_receipt_sha256")
    surface["prediction_artifact_sha256"] = _write_json(artifact_path, artifact)
    _rewrite_manifest(artifact_case, manifest)
    with pytest.raises(scorer.D107TruthScorerError, match="artifact/surface binding"):
        _score(artifact_case)

    truth_case = _build_case(tmp_path / "truth-id-mismatch")
    truth = _load(Path(truth_case["truth_path"]))
    truth["surfaces"][0]["ordered_query_physical_ids"][0] = "wrong-truth-physical-id"
    _rewrite_truth(truth_case, truth)
    with pytest.raises(scorer.D107TruthScorerError, match="truth/prediction ordered query-ID mismatch"):
        _score(truth_case)


def test_rejects_extra_role_or_truth_fields_in_prediction_payload(tmp_path: Path) -> None:
    role_case = _build_case(tmp_path / "role")
    manifest = _load(Path(role_case["manifest_path"]))
    manifest["surfaces"][0]["query_roles"] = ["old", "old"]
    _rewrite_manifest(role_case, manifest)
    with pytest.raises(scorer.D107TruthScorerError, match="forbidden truth/role field"):
        _score(role_case)

    truth_case = _build_case(tmp_path / "truth")
    manifest = _load(Path(truth_case["manifest_path"]))
    manifest["surfaces"][0]["truth_labels"] = ["old-a", "old-b"]
    _rewrite_manifest(truth_case, manifest)
    with pytest.raises(scorer.D107TruthScorerError, match="forbidden truth/role field"):
        _score(truth_case)


def test_same_row_metrics_harmonic_and_outer_micro_average(tmp_path: Path) -> None:
    case = _build_case(tmp_path / "metrics")
    result = _score(case)
    score = _load(Path(result["score_manifest"]))
    assert score["scene_same_row_count"] == 375
    assert score["scene_arm_metric_row_count"] == 1_500
    assert score["outer_arm_aggregate_row_count"] == 500
    scene_row = score["scene_same_rows"][0]
    arms = {row["arm"]: row for row in scene_row["arms"]}
    head = arms["M_HEAD"]
    assert head["before_old"] == pytest.approx(50.0)
    assert head["after_old"] == pytest.approx(50.0)
    assert head["before_old_floor"] == pytest.approx(0.0)
    assert head["after_old_floor"] == pytest.approx(0.0)
    assert head["seen_new"] == pytest.approx(100.0)
    assert head["H_old_new"] == pytest.approx(200.0 * 50.0 / 150.0)
    assert head["forgetting"] == pytest.approx(0.0)
    assert head["before_old_correct_count"] == 1
    assert head["after_old_correct_count"] == 1
    assert head["new_correct_count"] == 1
    assert head["total_correct_count"] == 3
    assert head["total_query_count"] == 5
    joint = arms["M_JOINT"]
    assert joint["H_old_new"] == pytest.approx(100.0)
    aggregate = next(
        row
        for row in score["outer_arm_aggregate_rows"]
        if row["outer_id"] == scene_row["outer_id"] and row["arm"] == "M_HEAD"
    )
    assert aggregate["aggregation"] == "micro_average_across_three_scenes"
    assert aggregate["after_old"] == pytest.approx(50.0)
    assert aggregate["seen_new"] == pytest.approx(100.0)
    assert aggregate["H_old_new"] == pytest.approx(200.0 * 50.0 / 150.0)
    assert aggregate["total_correct_count"] == 9
    assert aggregate["total_query_count"] == 15
    assert len(aggregate["source_scene_metric_row_receipt_sha256s"]) == 3
    assert score["target_verdict_summary"] == {
        "primary_candidate_arm": "M_JOINT",
        "causal_arms": list(scorer.ARMS),
        "causal_table_preserved": True,
        "coverage_verdict": "COMPLETE_125_TRUTH_OPEN_AND_SCORED",
        "target_thresholds_declared": False,
        "target_verdict": "NO_TARGET_THRESHOLD_DECLARED",
        "performance_early_stop_or_selection_performed": False,
        "D91_status": "D91_DEVELOPMENT_ONLY_15_ROWS_NON_PROMOTABLE",
    }


def test_comparator_pairing_fences_d91_to_fifteen_development_rows(tmp_path: Path) -> None:
    case = _build_case(tmp_path / "pairing")
    result = _score(case)
    score = _load(Path(result["score_manifest"]))
    comparator_rows = [
        {
            "receiver": row["receiver"],
            "seed": row["seed"],
            "k_shot": row["k_shot"],
            "new_count": row["new_count"],
            "scene": row["scene"],
            "development_metric": index,
        }
        for index, row in enumerate(score["scene_same_rows"][:15])
    ]
    paired = scorer.pair_d107_same_row(
        scene_same_rows=score["scene_same_rows"],
        comparator_id="D91",
        comparator_rows=comparator_rows,
    )
    assert paired["pairing_status"] == "D91_DEVELOPMENT_ONLY_15_ROWS_NON_PROMOTABLE"
    assert paired["matched_row_count"] == 15
    assert all(
        item["pairing_status"] == "D91_DEVELOPMENT_ONLY_15_ROWS_NON_PROMOTABLE"
        for item in paired["pairs"]
    )


def _d92_source_sidecar(
    *, root: Path, outer: dict[str, object]
) -> tuple[dict[str, object], dict[str, list[str]]]:
    job_root = root / "jobs" / "source-job"
    apply_root = job_root / "offline" / "predictor" / "after" / "apply_only_staging"
    apply_root.mkdir(parents=True)
    tokens_by_scene: dict[str, list[str]] = {}
    rows: list[dict[str, object]] = []
    for scene_index, scene in enumerate(SCENES):
        tokens = [
            f"qid_{scene_index + 1:02x}{class_index + 1:02x}" + "a" * 60
            for class_index in range(3)
        ]
        tokens_by_scene[scene] = tokens
        for class_index, (label, role) in enumerate(
            (("old-a", "target_old"), ("old-b", "target_old"), ("new-a", "target_new"))
        ):
            rows.append(
                {
                    "query_token": tokens[class_index],
                    "true_class_index": class_index,
                    "true_class_handle": label,
                    "transmitter_label": f"tx-{class_index}",
                    "evaluation_role": role,
                    "receiver_label": outer["receiver"],
                    # Every row deliberately shares this day: a scene cannot be
                    # inferred from it and must be recovered from query tokens.
                    "day_label": "day-not-a-scene",
                    "signal_label": "wifi",
                    "physical_sample_id": f"physical-{scene_index}-{class_index}",
                }
            )
    sidecar = {
        "schema": scorer.D92_TRUTH_SIDECAR_SCHEMA,
        "stage": "stage2c",
        "receiver": outer["receiver"],
        "seed": outer["seed"],
        "rows": list(reversed(rows)),
    }
    sidecar_path = job_root / "offline" / "scorer" / "truth_sidecar.json"
    sidecar_sha = _write_json(sidecar_path, sidecar)
    receipt = {
        "schema": scorer.D92_OFFLINE_BUILD_SCHEMA,
        "receiver": outer["receiver"],
        "seed": outer["seed"],
        "k_shot": outer["k_shot"],
        "new_class_count": outer["new_count"],
        "states": {"after": {"apply_staging_root": str(apply_root)}},
        "truth_sidecar": str(sidecar_path),
        "truth_sidecar_sha256": sidecar_sha,
    }
    _write_json(job_root / "offline" / "offline_build_receipt.json", receipt)
    plan = {"identity": {"d92_output_root": str(root)}}
    row = {
        "source_d92_job_id": "source-job",
        "source_pool_k": outer["k_shot"],
        "packages": {"after_apply": {"package_root": str(apply_root)}},
    }
    return {"plan": plan, "row": row}, tokens_by_scene


def test_build_truth_uses_sealed_query_tokens_not_day_labels(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    outer = _outer(0)
    source, tokens_by_scene = _d92_source_sidecar(root=tmp_path / "d92", outer=outer)
    sidecar_by_token = scorer._load_d92_truth_sidecar_for_outer(
        plan=source["plan"], row=source["row"], outer=outer
    )
    surfaces: dict[tuple[str, str, str, str], dict[str, object]] = {}
    for scene in SCENES:
        after_ids = tokens_by_scene[scene]
        before_ids = after_ids[:2]
        surfaces[(str(outer["outer_id"]), scene, "M0", "before")] = {
            "query_ids": tuple(before_ids)
        }
        surfaces[(str(outer["outer_id"]), scene, "M0", "after")] = {
            "query_ids": tuple(after_ids)
        }
    prediction = {"scenes": tuple(SCENES), "surfaces": surfaces}
    truth_surfaces = scorer._build_outer_truth_surfaces(
        prediction=prediction, outer=outer, sidecar_by_token=sidecar_by_token
    )
    first_before, first_after = truth_surfaces[:2]
    assert first_before["ordered_query_physical_ids"] == tokens_by_scene[SCENES[0]][:2]
    assert first_before["labels"] == ["old-a", "old-b"]
    assert first_after["ordered_query_physical_ids"] == tokens_by_scene[SCENES[0]]
    assert first_after["labels"] == ["old-a", "old-b", "new-a"]

    case = _build_case(tmp_path / "full")
    calls: list[str] = []
    manifest = scorer.validate_d107_prediction_manifest(
        prediction_manifest_path=Path(case["manifest_path"]),
        expected_prediction_manifest_file_sha256=str(case["manifest_file_sha"]),
    )
    prepared_rows = [
        {
            "outer_id": row["outer_id"],
            "receiver": row["receiver"],
            "seed": row["seed"],
            "k_shot": row["k_shot"],
            "new_count": row["new_count"],
        }
        for row in manifest["outer_rows"]
    ]

    def _prepared(**_kwargs):  # type: ignore[no-untyped-def]
        return {"identity": {}}, {"rows": prepared_rows}

    def _source_truth(*, outer, **_kwargs):  # type: ignore[no-untyped-def]
        calls.append(outer["outer_id"])
        result = {}
        for scene in SCENES:
            for label, role in (
                ("old-a", "target_old"),
                ("old-b", "target_old"),
                ("new-a", "target_new"),
            ):
                token = f"{outer['outer_id']}::{scene}::{label}"
                result[token] = {
                    "label": label,
                    "role": role,
                    "physical_sample_id": token + "::physical",
                }
        return result

    monkeypatch.setattr(scorer, "_load_prepared_d107_truth_inputs", _prepared)
    monkeypatch.setattr(scorer, "_load_d92_truth_sidecar_for_outer", _source_truth)
    built_path = tmp_path / "full" / "built_truth_catalog.json"
    built = scorer.build_d107_target125_truth_catalog(
        prediction_manifest_path=Path(case["manifest_path"]),
        expected_prediction_manifest_file_sha256=str(case["manifest_file_sha"]),
        plan_manifest_path=tmp_path / "unused-plan.json",
        expected_plan_file_sha256="a" * 64,
        context_manifest_path=tmp_path / "unused-context.json",
        expected_context_file_sha256="b" * 64,
        output_path=built_path,
    )
    assert len(calls) == scorer.OUTER_JOB_COUNT
    built_catalog = _load(built_path)
    assert built_catalog["truth_surface_count"] == scorer.TRUTH_SURFACE_COUNT
    assert built_catalog["surfaces"][0]["labels"] == ["old-a", "old-b"]
    assert built_catalog["surfaces"][1]["labels"] == ["old-a", "old-b", "new-a"]
    scored = scorer.score_d107_target125(
        prediction_manifest_path=Path(case["manifest_path"]),
        expected_prediction_manifest_file_sha256=str(case["manifest_file_sha"]),
        truth_catalog_path=built_path,
        expected_truth_catalog_file_sha256=str(built["truth_catalog_file_sha256"]),
        output_dir=tmp_path / "full" / "built_score",
    )
    assert Path(scored["score_manifest"]).is_file()


def test_scorer_never_imports_d106_router() -> None:
    source = Path(scorer.__file__).read_text(encoding="utf-8")
    assert "stage2_d106_k_conditioned_router" not in source
    assert "ROUTE_BY_K" not in source
