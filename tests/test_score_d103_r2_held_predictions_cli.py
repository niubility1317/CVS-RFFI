import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import pytest


def test_truth_side_scorer_cli_help() -> None:
    script = (
        Path(__file__).resolve().parents[1]
        / "code"
        / "scripts"
        / "score_d103_r2_held_predictions.py"
    )
    result = subprocess.run(
        [sys.executable, str(script), "--help"],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0
    assert "--truth-json" in result.stdout
    assert "--prediction-root" in result.stdout


def _load_scorer():
    script = (
        Path(__file__).resolve().parents[1]
        / "code"
        / "scripts"
        / "score_d103_r2_held_predictions.py"
    )
    spec = importlib.util.spec_from_file_location(
        "d103_truth_scorer_test", script
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _prediction_fixture(tmp_path: Path, module):
    prediction_root = tmp_path / "predictions"
    rows_root = prediction_root / "rows"
    rows_root.mkdir(parents=True)
    manifest_rows = []
    receiver_ids = [f"r{receiver}" for receiver in range(7)]
    class_ids = [f"c{class_id}" for class_id in range(6)]
    row_specs = [
        (receiver, None, k_shot)
        for receiver in receiver_ids
        for k_shot in (1, 5, 10)
    ] + [
        (receiver, class_id, 1)
        for receiver in receiver_ids
        for class_id in class_ids
    ]
    package_specs = [
        (receiver, k_shot)
        for receiver in receiver_ids
        for k_shot in (1, 5, 10)
    ]
    package_ids = [
        module.package_id(receiver, k_shot)
        for receiver, k_shot in package_specs
    ]
    for index, (receiver, held_class, k_shot) in enumerate(row_specs):
        package = module.package_id(receiver, k_shot)
        artifact_path = rows_root / f"row-{index:02d}.json"
        artifact_path.write_text(
            json.dumps(
                {
                    "held_receiver": receiver,
                    "held_class": held_class,
                    "K": k_shot,
                    "query_physical_ids": [f"{package}-query"],
                    "int8_audit": {
                        "top1_agreement": 1.0,
                        "large_margin_flip_count": 0,
                    },
                    "resource_audit": {
                        "actual_serialized_state_bytes": 1024,
                        "numeric_bundle_state_bytes": 1024,
                        "post_backbone_mac_per_query": 2048,
                    },
                }
            ),
            encoding="utf-8",
        )
        manifest_rows.append(
            {
                "held_receiver": receiver,
                "held_class": held_class,
                "K": k_shot,
                "package_id": package,
                "path": f"rows/{artifact_path.name}",
                "sha256": module.sha256_file(artifact_path),
            }
        )
    time.sleep(0.01)
    manifest_path = prediction_root / "prediction_manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "schema": (
                    "cvs.d103_r2.rxid_crossreceiver.held_predictions.v1"
                ),
                "row_count": 63,
                "rows": manifest_rows,
                "day_stability_rows": [
                    {"row": index} for index in range(49)
                ],
                "query_truth_access": False,
                "target_access": False,
                "formal_query_access": False,
                "sealed_at_unix_ns": time.time_ns(),
            }
        ),
        encoding="utf-8",
    )
    truth_path = tmp_path / "truth.json"
    truth_path.write_text(
        json.dumps(
            {
                "schema": "cvs.d103_r2.rxid_crossreceiver.held_truth.v1",
                "package_count": 21,
                "predictor_access": False,
                "packages": [
                    {
                        "package_id": package,
                        "query_physical_ids": [f"{package}-query"],
                        "query_truth_labels": ["c0"],
                    }
                    for package in package_ids
                ],
            }
        ),
        encoding="utf-8",
    )
    return prediction_root, manifest_path, truth_path


def test_truth_is_first_read_only_after_all_predictions_are_prevalidated(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = _load_scorer()
    prediction_root, _, truth_path = _prediction_fixture(tmp_path, module)
    output = tmp_path / "scores" / "held_scores.json"
    event = tmp_path / "scores" / "truth_first_open.json"
    original_read = module._read_json
    artifact_reads = []

    def tracked_read(path: Path):
        resolved = Path(path).resolve()
        if resolved == truth_path.resolve():
            assert event.is_file()
            assert len(artifact_reads) == 63
        elif resolved.parent == (prediction_root / "rows").resolve():
            artifact_reads.append(resolved)
        return original_read(path)

    monkeypatch.setattr(module, "_read_json", tracked_read)
    monkeypatch.setattr(
        module,
        "score_prediction_artifact",
        lambda artifact, truth: {
            "held_receiver": artifact["held_receiver"]
        },
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "score",
            "--prediction-root",
            str(prediction_root),
            "--truth-json",
            str(truth_path),
            "--output-json",
            str(output),
            "--truth-open-event-json",
            str(event),
        ],
    )
    assert module.main() == 0
    result = json.loads(output.read_text(encoding="utf-8"))
    event_sha = module.sha256_file(event)
    assert len(result["performance_rows"]) == 63
    assert {
        row["truth_open_event_sha256"]
        for row in result["performance_rows"]
    } == {event_sha}


def test_drifted_prediction_aborts_before_truth_is_opened(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = _load_scorer()
    prediction_root, manifest_path, truth_path = _prediction_fixture(
        tmp_path, module
    )
    drifted = prediction_root / "rows" / "row-62.json"
    drifted.write_text(
        drifted.read_text(encoding="utf-8") + "\n", encoding="utf-8"
    )
    drifted_time = manifest_path.stat().st_mtime_ns + 1_000_000_000
    os.utime(drifted, ns=(drifted_time, drifted_time))
    assert drifted.stat().st_mtime_ns > manifest_path.stat().st_mtime_ns
    event = tmp_path / "scores" / "truth_first_open.json"
    original_read = module._read_json
    truth_opened = False

    def tracked_read(path: Path):
        nonlocal truth_opened
        if Path(path).resolve() == truth_path.resolve():
            truth_opened = True
        return original_read(path)

    monkeypatch.setattr(module, "_read_json", tracked_read)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "score",
            "--prediction-root",
            str(prediction_root),
            "--truth-json",
            str(truth_path),
            "--output-json",
            str(tmp_path / "scores" / "held_scores.json"),
            "--truth-open-event-json",
            str(event),
        ],
    )
    with pytest.raises(ValueError, match="sealed before truth open"):
        module.main()
    assert truth_opened is False
    assert event.exists() is False


def test_nonformal_63_row_identity_matrix_aborts_before_truth(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = _load_scorer()
    prediction_root, manifest_path, truth_path = _prediction_fixture(
        tmp_path, module
    )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    row = manifest["rows"][-1]
    row["K"] = 5
    row["package_id"] = module.package_id(row["held_receiver"], 5)
    artifact_path = prediction_root / row["path"]
    artifact = json.loads(artifact_path.read_text(encoding="utf-8"))
    artifact["K"] = 5
    artifact_path.write_text(json.dumps(artifact), encoding="utf-8")
    row["sha256"] = module.sha256_file(artifact_path)
    time.sleep(0.01)
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    event = tmp_path / "scores" / "truth_first_open.json"
    original_read = module._read_json
    truth_opened = False

    def tracked_read(path: Path):
        nonlocal truth_opened
        if Path(path).resolve() == truth_path.resolve():
            truth_opened = True
        return original_read(path)

    monkeypatch.setattr(module, "_read_json", tracked_read)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "score",
            "--prediction-root",
            str(prediction_root),
            "--truth-json",
            str(truth_path),
            "--output-json",
            str(tmp_path / "scores" / "held_scores.json"),
            "--truth-open-event-json",
            str(event),
        ],
    )
    with pytest.raises(ValueError, match="identity matrix drift"):
        module.main()
    assert truth_opened is False
    assert event.exists() is False


def test_resolved_artifact_path_alias_aborts_before_truth(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = _load_scorer()
    prediction_root, manifest_path, truth_path = _prediction_fixture(
        tmp_path, module
    )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["rows"][2]["path"] = "rows/../rows/row-01.json"
    manifest["rows"][2]["sha256"] = manifest["rows"][1]["sha256"]
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    event = tmp_path / "scores" / "truth_first_open.json"
    original_read = module._read_json
    truth_opened = False

    def tracked_read(path: Path):
        nonlocal truth_opened
        if Path(path).resolve() == truth_path.resolve():
            truth_opened = True
        return original_read(path)

    monkeypatch.setattr(module, "_read_json", tracked_read)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "score",
            "--prediction-root",
            str(prediction_root),
            "--truth-json",
            str(truth_path),
            "--output-json",
            str(tmp_path / "scores" / "held_scores.json"),
            "--truth-open-event-json",
            str(event),
        ],
    )
    with pytest.raises(ValueError, match="sealed before truth open"):
        module.main()
    assert truth_opened is False
    assert event.exists() is False
