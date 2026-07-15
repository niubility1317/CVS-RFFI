from __future__ import annotations

import ast
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from cvsrffi import stage2_metric_scorer as scorer
from cvsrffi.stage2_prediction_artifact import (
    PredictionArtifactError,
    publish_prediction_artifact,
)


SCENARIOS = (
    "leo_clear_weak",
    "leo_low_elev_weak",
    "leo_rain_weak",
)
TOKENS = [f"qid_{index:064x}" for index in range(1, 7)]
HANDLES = [f"cls_{index:064x}" for index in range(101, 104)]


def _json(path: Path, payload: object) -> None:
    path.write_bytes(scorer.canonical_json_bytes(payload) + b"\n")


def _truth_rows(stage: str) -> list[dict[str, object]]:
    definitions = [
        (TOKENS[0], 0, HANDLES[0], "old-a", "target_old"),
        (TOKENS[1], 0, HANDLES[0], "old-a", "target_old"),
        (TOKENS[2], 1, HANDLES[1], "old-b", "target_old"),
        (TOKENS[3], 1, HANDLES[1], "old-b", "target_old"),
        (
            TOKENS[4],
            2 if stage == "stage2c" else None,
            HANDLES[2] if stage == "stage2c" else None,
            "new-a",
            "target_new",
        ),
        (
            TOKENS[5],
            2 if stage == "stage2c" else None,
            HANDLES[2] if stage == "stage2c" else None,
            "new-a",
            "target_new",
        ),
    ]
    return [
        {
            "query_token": token,
            "true_class_index": class_index,
            "true_class_handle": class_handle,
            "transmitter_label": transmitter,
            "evaluation_role": role,
            "receiver_label": "20-1",
            "day_label": "day-1",
            "signal_label": f"sig-{position}",
            "physical_sample_id": f"sample-{position}",
        }
        for position, (token, class_index, class_handle, transmitter, role) in enumerate(definitions)
    ]


def _make_case(
    root: Path,
    *,
    stage: str = "stage2c",
    k_shot: int = 1,
    prediction_tokens: list[str] | None = None,
    truth_rows: list[dict[str, object]] | None = None,
    stream_overrides: dict[str, list[str]] | None = None,
    scenario_order: tuple[str, ...] = SCENARIOS,
    tokens_by_scenario: dict[str, list[str]] | None = None,
) -> dict[str, object]:
    root.mkdir()
    tokens = prediction_tokens or list(TOKENS)
    count = len(tokens)
    prediction_path = root / "prediction_artifact.cvspred"
    base_streams = {
        "candidate_after": [
            HANDLES[0], HANDLES[0], HANDLES[1], HANDLES[0], HANDLES[2], HANDLES[0]
        ][:count],
        "candidate_before": [
            HANDLES[0], HANDLES[1], HANDLES[0], HANDLES[1], HANDLES[0], HANDLES[1]
        ][:count],
        "identity_after": [
            HANDLES[0], HANDLES[1], HANDLES[1], HANDLES[0], HANDLES[2], HANDLES[1]
        ][:count],
        "identity_before": [
            HANDLES[0], HANDLES[0], HANDLES[1], HANDLES[0], HANDLES[0], HANDLES[1]
        ][:count],
        "direct": [
            HANDLES[1], HANDLES[1], HANDLES[1], HANDLES[0], HANDLES[0], HANDLES[1]
        ][:count],
    }
    for name, values in (stream_overrides or {}).items():
        base_streams[name] = values[:count]
    flat_tokens: list[str] = []
    flat_scenarios: list[str] = []
    flat_streams = {name: [] for name in base_streams}
    flat_view_counts: list[int] = []
    base_view_counts = [1, 3, 5, 1, 3, 5][:count]
    for scenario in scenario_order:
        scenario_tokens = (tokens_by_scenario or {}).get(scenario, tokens)
        if len(scenario_tokens) != count:
            raise ValueError("test fixture scenario token count drift")
        flat_tokens.extend(scenario_tokens)
        flat_scenarios.extend([scenario] * count)
        for name, values in base_streams.items():
            flat_streams[name].extend(values)
        flat_view_counts.extend(base_view_counts)
    streams = {name: np.asarray(values) for name, values in flat_streams.items()}
    streams["shared_view_counts"] = np.asarray(flat_view_counts, dtype=np.int64)
    package_root = "a" * 64
    package_seal = "b" * 64
    publication = publish_prediction_artifact(
        prediction_path,
        stage="Stage2-C" if stage == "stage2c" else "Stage2-B",
        row_id="row-001",
        receiver="20-1",
        k_shot=k_shot,
        candidate_lock_sha256="c" * 64,
        package_root_sha256=package_root,
        package_seal_sha256=package_seal,
        query_tokens=np.asarray(flat_tokens),
        scenarios=np.asarray(flat_scenarios),
        **streams,
    )
    truth_path = root / "truth_sidecar.json"
    _json(
        truth_path,
        {
            "schema": scorer.TRUTH_SIDECAR_SCHEMA,
            "stage": stage,
            "receiver": "20-1",
            "seed": 701,
            "rows": truth_rows if truth_rows is not None else _truth_rows(stage),
        },
    )
    scoring_manifest_path = root / "scoring_manifest.json"
    _json(
        scoring_manifest_path,
        {
            "schema": scorer.SCORING_MANIFEST_SCHEMA,
            "predictor_package_root_sha256": package_root,
            "predictor_package_seal_sha256": package_seal,
            "truth_sidecar_json": truth_path.name,
            "truth_sidecar_sha256": scorer.sha256_file(truth_path),
            "scorer_output_must_not_feed_predictor": True,
        },
    )
    return {
        "prediction": prediction_path,
        "prediction_artifact_sha256": publication["artifact_sha256"],
        "prediction_seal_sha256": publication["seal_sha256"],
        "truth": truth_path,
        "scoring_manifest": scoring_manifest_path,
        "scoring_manifest_sha256": scorer.sha256_file(scoring_manifest_path),
    }


def _score(case: dict[str, object]):
    return scorer.score_sealed_prediction(
        case["prediction"],
        case["scoring_manifest"],
        expected_prediction_artifact_sha256=case["prediction_artifact_sha256"],
        expected_prediction_seal_sha256=case["prediction_seal_sha256"],
        expected_scoring_manifest_sha256=case["scoring_manifest_sha256"],
    )


def _resign_truth(case: dict[str, object], payload: dict[str, object]) -> None:
    truth_path = Path(case["truth"])
    _json(truth_path, payload)
    manifest_path = Path(case["scoring_manifest"])
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["truth_sidecar_sha256"] = scorer.sha256_file(truth_path)
    _json(manifest_path, manifest)
    case["scoring_manifest_sha256"] = scorer.sha256_file(manifest_path)


def test_stage2c_scores_five_streams_and_k1_deltas(tmp_path: Path) -> None:
    case = _make_case(tmp_path / "case")
    rows_payload, predictions_payload, receipt = _score(case)

    assert rows_payload["schema"] == scorer.FORMAL_ROWS_SCHEMA
    assert len(rows_payload["rows"]) == 3
    row = rows_payload["rows"][0]
    assert row["old_acc"] == pytest.approx(0.75)
    assert row["old_acc_before_increment"] == pytest.approx(0.5)
    assert row["min_old_class_acc"] == pytest.approx(0.5)
    assert row["min_old_class_acc_before_increment"] == pytest.approx(0.5)
    assert row["min_old_class_acc_after_increment"] == pytest.approx(0.5)
    assert row["seen_new_acc"] == pytest.approx(0.5)
    assert row["H_old_new"] == pytest.approx(0.6)
    assert row["seen_new_acc_before_increment"] is None
    assert row["seen_new_acc_after_increment"] == pytest.approx(0.5)
    assert row["H_old_new_before_increment"] is None
    assert row["H_old_new_after_increment"] == pytest.approx(0.6)
    assert row["pre_increment_new_class_state"] == "NEW_CLASSES_NOT_REGISTERED"
    assert row["candidate_average_forgetting"] == pytest.approx(-0.25)
    assert row["identity_average_forgetting"] == pytest.approx(0.25)
    assert row["delta_vs_direct_ADV3B02_K1"] == pytest.approx(0.5)
    assert row["delta_vs_identity_K1"] == pytest.approx(0.25)
    assert row["identity_delta_vs_direct_ADV3B02_K1"] == pytest.approx(0.25)
    assert row["candidate_old_class_acc"] == {"old-a": 1.0, "old-b": 0.5}
    assert row["candidate_old_class_acc_before_increment"] == {
        "old-a": 0.5, "old-b": 0.5,
    }
    assert row["candidate_old_class_acc_after_increment"] == {
        "old-a": 1.0, "old-b": 0.5,
    }
    assert row["candidate_old_class_forgetting"] == {
        "old-a": pytest.approx(-0.5), "old-b": pytest.approx(0.0),
    }
    assert row["candidate_old_class_adaptation_gain"] == {
        "old-a": pytest.approx(0.5), "old-b": pytest.approx(0.0),
    }
    assert row["identity_old_class_acc_before_increment"] == {
        "old-a": 1.0, "old-b": 0.5,
    }
    assert row["identity_old_class_acc_after_increment"] == {
        "old-a": 0.5, "old-b": 0.5,
    }
    assert row["identity_old_class_forgetting"] == {
        "old-a": pytest.approx(0.5), "old-b": pytest.approx(0.0),
    }
    assert row["view1_count"] == 2
    assert row["view3_count"] == 2
    assert row["view5_count"] == 2
    assert len(predictions_payload["predictions"]) == 18
    assert {value["scenario"] for value in rows_payload["rows"]} == set(SCENARIOS)
    assert all(value["old_acc"] == pytest.approx(0.75) for value in rows_payload["rows"])
    assert receipt["join_policy"] == "exact_scenario_query_token"
    assert receipt["truth_join_after_prediction_only"] is True
    assert receipt["scenario_count"] == 3
    assert receipt["formal_row_count"] == 3
    assert receipt["prediction_immutable_state"] == "SEALED_READ_ONLY_ATOMIC_NOREPLACE"
    assert receipt["formal_adapter_resource_claim_allowed"] is False
    assert receipt["adapter_resource_verification"] == {
        "status": "NOT_PROVABLE_FROM_PREDICTION_ARTIFACT",
        "reason_code": "ADAPTER_MATRIX_NOT_EMBEDDED",
        "adapter_matrix_embedded": False,
        "trainable_parameter_count_verified": False,
        "persistent_state_bytes_verified": False,
        "formal_adapter_resource_claim_allowed": False,
    }


def test_stage2b_excludes_target_new_reference_from_identity_metrics(
    tmp_path: Path,
) -> None:
    case = _make_case(tmp_path / "case", stage="stage2b", k_shot=5)
    rows_payload, predictions_payload, _receipt = _score(case)
    row = rows_payload["rows"][0]
    assert row["stage"] == "stage2b"
    assert row["target_new_query_count"] == 2
    assert row["seen_new_acc"] is None
    assert row["H_old_new"] is None
    assert row["delta_vs_direct_ADV3B02_K1"] is None
    new_rows = [
        value
        for value in predictions_payload["predictions"]
        if value["evaluation_role"] == "target_new"
    ]
    assert all(value["true_class_index"] is None for value in new_rows)
    assert all(value["true_class_handle"] is None for value in new_rows)
    assert all(value["candidate_after_correct"] is None for value in new_rows)


def test_rejects_duplicate_prediction_key(tmp_path: Path) -> None:
    tokens = list(TOKENS)
    tokens[-1] = tokens[-2]
    with pytest.raises(PredictionArtifactError, match="scenario/query_token"):
        _make_case(tmp_path / "case", prediction_tokens=tokens)


@pytest.mark.parametrize(
    "scenario_order",
    [SCENARIOS[:2], (SCENARIOS[1], SCENARIOS[0], SCENARIOS[2])],
)
def test_rejects_missing_or_reordered_formal_scenarios(
    tmp_path: Path, scenario_order: tuple[str, ...]
) -> None:
    case = _make_case(tmp_path / "case", scenario_order=scenario_order)
    with pytest.raises(scorer.Stage2ScoringError, match="scenario sequence"):
        _score(case)


def test_rejects_query_token_set_drift_across_formal_scenarios(
    tmp_path: Path,
) -> None:
    rain_tokens = list(TOKENS)
    rain_tokens[-1] = f"qid_{999:064x}"
    case = _make_case(
        tmp_path / "case",
        tokens_by_scenario={SCENARIOS[2]: rain_tokens},
    )
    with pytest.raises(scorer.Stage2ScoringError, match="identical query_token sets"):
        _score(case)


@pytest.mark.parametrize(
    "tokens",
    [TOKENS[:-1], [*TOKENS[:-1], f"qid_{999:064x}"]],
)
def test_rejects_missing_or_unmatched_prediction_tokens(
    tmp_path: Path, tokens: list[str]
) -> None:
    case = _make_case(tmp_path / "case", prediction_tokens=list(tokens))
    with pytest.raises(scorer.Stage2ScoringError, match="token mismatch"):
        _score(case)


def test_rejects_duplicate_truth_token(tmp_path: Path) -> None:
    truth_rows = _truth_rows("stage2c")
    truth_rows[-1]["query_token"] = truth_rows[-2]["query_token"]
    case = _make_case(tmp_path / "case", truth_rows=truth_rows)
    with pytest.raises(scorer.Stage2ScoringError, match="duplicate truth"):
        _score(case)


def test_rejects_role_contamination_even_when_truth_is_resigned(tmp_path: Path) -> None:
    case = _make_case(tmp_path / "case")
    truth = json.loads(Path(case["truth"]).read_text(encoding="utf-8"))
    truth["rows"][0]["evaluation_role"] = "target_unknown"
    _resign_truth(case, truth)
    with pytest.raises(scorer.Stage2ScoringError, match="role contamination"):
        _score(case)


def test_rejects_cross_role_transmitter_contamination(tmp_path: Path) -> None:
    case = _make_case(tmp_path / "case")
    truth = json.loads(Path(case["truth"]).read_text(encoding="utf-8"))
    truth["rows"][-1]["transmitter_label"] = "old-a"
    _resign_truth(case, truth)
    with pytest.raises(scorer.Stage2ScoringError, match="role contamination"):
        _score(case)


def test_rejects_stage2b_registered_target_new_truth(tmp_path: Path) -> None:
    case = _make_case(tmp_path / "case", stage="stage2b")
    truth = json.loads(Path(case["truth"]).read_text(encoding="utf-8"))
    truth["rows"][-1]["true_class_index"] = 2
    _resign_truth(case, truth)
    with pytest.raises(scorer.Stage2ScoringError, match="Stage2-B target-new"):
        _score(case)


def test_rejects_truth_sidecar_hash_tamper(tmp_path: Path) -> None:
    case = _make_case(tmp_path / "case")
    truth_path = Path(case["truth"])
    truth_path.write_bytes(truth_path.read_bytes() + b" ")
    with pytest.raises(scorer.Stage2ScoringError, match="truth sidecar detached hash"):
        _score(case)


def test_rejects_scoring_manifest_tamper_at_external_trust_root(
    tmp_path: Path,
) -> None:
    case = _make_case(tmp_path / "case")
    manifest_path = Path(case["scoring_manifest"])
    manifest_path.write_bytes(manifest_path.read_bytes() + b" ")
    with pytest.raises(scorer.Stage2ScoringError, match="scoring manifest detached hash"):
        _score(case)


def test_rejects_wrong_external_prediction_seal_sha256(tmp_path: Path) -> None:
    case = _make_case(tmp_path / "case")
    with pytest.raises(scorer.Stage2ScoringError, match="verification failed"):
        scorer.score_sealed_prediction(
            case["prediction"],
            case["scoring_manifest"],
            expected_prediction_artifact_sha256=case[
                "prediction_artifact_sha256"
            ],
            expected_prediction_seal_sha256="f" * 64,
            expected_scoring_manifest_sha256=case["scoring_manifest_sha256"],
        )


def test_rejects_tampered_cvspred_container(tmp_path: Path) -> None:
    case = _make_case(tmp_path / "case")
    artifact = Path(case["prediction"])
    data = bytearray(artifact.read_bytes())
    data[-1] ^= 0x01
    os.chmod(artifact, 0o600)
    artifact.write_bytes(data)
    os.chmod(artifact, 0o444)
    with pytest.raises(scorer.Stage2ScoringError, match="verification failed"):
        _score(case)


def test_rejects_nonopaque_prediction_handle_inside_valid_container(
    tmp_path: Path,
) -> None:
    case = _make_case(
        tmp_path / "case",
        stream_overrides={
            "candidate_after": [
                "old-a",
                HANDLES[0],
                HANDLES[1],
                HANDLES[0],
                HANDLES[2],
                HANDLES[0],
            ]
        },
    )
    with pytest.raises(scorer.Stage2ScoringError, match="non-opaque class handle"):
        _score(case)


def test_exclusive_outputs_refuse_overwrite(tmp_path: Path) -> None:
    case = _make_case(tmp_path / "case")
    rows, predictions, receipt = _score(case)
    output = tmp_path / "output"
    paths = {
        "formal_rows_path": output / "formal_rows.json",
        "formal_predictions_path": output / "formal_predictions.json",
        "scoring_receipt_path": output / "scoring_receipt.json",
    }
    scorer.write_scoring_outputs_exclusive(
        **paths,
        formal_rows=rows,
        formal_predictions=predictions,
        scoring_receipt=receipt,
    )
    assert scorer.sha256_file(paths["formal_rows_path"]) == receipt["formal_rows_sha256"]
    assert (
        scorer.sha256_file(paths["formal_predictions_path"])
        == receipt["formal_predictions_sha256"]
    )
    with pytest.raises(FileExistsError, match="overwrite is forbidden"):
        scorer.write_scoring_outputs_exclusive(
            **paths,
            formal_rows=rows,
            formal_predictions=predictions,
            scoring_receipt=receipt,
        )


def test_cli_executes_separate_scoring_flow(tmp_path: Path) -> None:
    case = _make_case(tmp_path / "case")
    output = tmp_path / "cli-output"
    command = [
        sys.executable,
        str(
            Path(__file__).resolve().parents[1]
            / "code"
            / "scripts"
            / "score_cvs_stage2_sealed_prediction.py"
        ),
        "--prediction-artifact",
        str(case["prediction"]),
        "--expected-prediction-artifact-sha256",
        str(case["prediction_artifact_sha256"]),
        "--expected-prediction-seal-sha256",
        str(case["prediction_seal_sha256"]),
        "--scoring-manifest",
        str(case["scoring_manifest"]),
        "--expected-scoring-manifest-sha256",
        str(case["scoring_manifest_sha256"]),
        "--formal-rows",
        str(output / "formal_rows.json"),
        "--formal-predictions",
        str(output / "formal_predictions.json"),
        "--scoring-receipt",
        str(output / "scoring_receipt.json"),
    ]
    result = subprocess.run(command, check=False, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert (output / "formal_rows.json").is_file()
    assert (output / "formal_predictions.json").is_file()
    receipt = json.loads((output / "scoring_receipt.json").read_text(encoding="utf-8"))
    assert receipt["status"] == "PASS"


def test_scorer_module_imports_no_project_training_dataset_or_legacy_code() -> None:
    source = Path(scorer.__file__).read_text(encoding="utf-8")
    tree = ast.parse(source)
    imported_roots: set[str] = set()
    project_modules: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported_roots.update(alias.name.split(".", 1)[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported_roots.add(node.module.split(".", 1)[0])
            if node.module.startswith("cvsrffi"):
                project_modules.add(node.module)
    assert imported_roots <= {
        "__future__",
        "contextlib",
        "hashlib",
        "json",
        "math",
        "os",
        "re",
        "stat",
        "zipfile",
        "pathlib",
        "typing",
        "numpy",
        "cvsrffi",
    }
    assert project_modules == {"cvsrffi.stage2_prediction_artifact"}
