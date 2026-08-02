from __future__ import annotations

import hashlib
import importlib.util
import inspect
import json
from pathlib import Path
import sys

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "code" / "scripts" / "run_d121_g1_sourceheld_one_shot.py"
SPEC = importlib.util.spec_from_file_location("d121_g1_sourceheld", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
runner = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = runner
SPEC.loader.exec_module(runner)


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, sort_keys=True), encoding="utf-8")


def _write_d104_packages(tmp_path: Path) -> tuple[Path, Path, Path]:
    root = tmp_path / "d104_packages"
    package_dir = root / "predictor_packages"
    scorer_dir = root / "scorer_only"
    package_dir.mkdir(parents=True)
    scorer_dir.mkdir()
    receivers = tuple(f"rx-{index}" for index in range(7))
    classes = tuple(f"tx-{index}" for index in range(6))
    package_rows = []
    truth_rows = []
    for receiver_index, receiver in enumerate(receivers):
        for k_shot in runner.K_VALUES:
            support, labels, support_ids = [], [], []
            # Reverse the input class traversal: LBR must later use bank order,
            # rather than this package order, to attach physical IDs.
            for class_index in reversed(range(6)):
                for sample_index in range(k_shot):
                    row = np.zeros(160, dtype=np.float32)
                    row[0] = np.float32(1.0)
                    row[8 + class_index] = np.float32(0.62)
                    row[48 + receiver_index] = np.float32(0.04)
                    row[96 + sample_index] = np.float32(0.011 * (sample_index + 1))
                    row[150] = np.float32(0.0001 * (class_index + 1) * (sample_index + 1))
                    support.append(row)
                    labels.append(classes[class_index])
                    support_ids.append(
                        f"support-{receiver_index}-{k_shot}-{class_index}-{sample_index}"
                    )
            query, query_ids, truth = [], [], []
            for class_index, class_id in enumerate(classes):
                row = np.zeros(160, dtype=np.float32)
                row[0] = np.float32(1.0)
                row[8 + class_index] = np.float32(0.62)
                row[48 + receiver_index] = np.float32(0.04)
                row[151] = np.float32(0.003 * (class_index + 1))
                query.append(row)
                query_ids.append(f"query-{receiver_index}-{k_shot}-{class_index}")
                truth.append(class_id)
            package_id = f"package-{receiver_index}-{k_shot}"
            package_path = package_dir / f"{package_id}.npz"
            np.savez(
                package_path,
                support_pre_relu=np.asarray(support, dtype=np.float32),
                support_zdom=np.zeros((len(support), 160), dtype=np.float32),
                support_labels=np.asarray(labels, dtype=str),
                support_physical_ids=np.asarray(support_ids, dtype=str),
                query_pre_relu=np.asarray(query, dtype=np.float32),
                query_physical_ids=np.asarray(query_ids, dtype=str),
                registered_classes=np.asarray(classes, dtype=str),
            )
            package_rows.append(
                {
                    "package_id": package_id,
                    "held_receiver": receiver,
                    "K": k_shot,
                    "path": str(Path("predictor_packages") / package_path.name),
                    "sha256": _sha(package_path),
                    "query_truth_present": False,
                }
            )
            truth_rows.append(
                {
                    "package_id": package_id,
                    "query_physical_ids": query_ids,
                    "query_truth_labels": truth,
                }
            )
    truth_path = scorer_dir / "truth.json"
    _write(
        truth_path,
        {
            "schema": "cvs.d104_r1.rxid_angq.held_truth.v2",
            "split_id": runner.SPLIT_ID,
            "package_count": 21,
            "predictor_access": False,
            "packages": truth_rows,
        },
    )
    seal_path = scorer_dir / "truth_input_seal.json"
    _write(
        seal_path,
        {
            "schema": "cvs.d104_r1.rxid_angq.truth_input_seal.v1",
            "split_id": runner.SPLIT_ID,
            "package_count": 21,
            "package_ids": [row["package_id"] for row in truth_rows],
            "truth_package_root_sha256": runner.d106.canonical_sha256(truth_rows),
            "predictor_truth_access": False,
        },
    )
    manifest_path = root / "package_manifest.json"
    _write(
        manifest_path,
        {
            "schema": runner.PACKAGE_SCHEMA,
            "candidate_id": runner.d106.D104_CANDIDATE_ID,
            "split_id": runner.SPLIT_ID,
            "receiver_ids": list(receivers),
            "class_ids": list(classes),
            "package_count": 21,
            "packages": package_rows,
            "truth_input_seal_sha256": _sha(seal_path),
            "query_truth_present": False,
            "target_access": False,
        },
    )
    return root, truth_path, seal_path


def _write_complete_prediction_root(
    tmp_path: Path,
    package_root: Path,
    *,
    name: str,
    candidate_better: bool,
) -> Path:
    manifest = json.loads((package_root / "package_manifest.json").read_text(encoding="utf-8"))
    rows_by_key = {
        (row["held_receiver"], int(row["K"])): row for row in manifest["packages"]
    }
    root = tmp_path / name
    row_root = root / "rows"
    row_root.mkdir(parents=True)
    rows = []
    classes = tuple(manifest["class_ids"])
    for index, (receiver, held_class, k_shot) in enumerate(
        runner.fixed_row_specs(manifest["receiver_ids"], classes)
    ):
        package = rows_by_key[(receiver, k_shot)]
        with np.load(package_root / package["path"], allow_pickle=False) as archive:
            query_ids = archive["query_physical_ids"].astype(str).tolist()
        correct = list(classes)
        wrong = list(classes[1:] + classes[:1])
        if candidate_better:
            arms = {"M0": wrong, "M_DA": wrong, "M_HEAD": correct, "M_JOINT": correct}
        else:
            arms = {"M0": correct, "M_DA": correct, "M_HEAD": wrong, "M_JOINT": wrong}
        artifact = {
            "schema": runner.PREDICTION_SCHEMA + ".row",
            "candidate_id": runner.CANDIDATE_ID,
            "split_id": runner.SPLIT_ID,
            "run_id": "fixture-run",
            "held_receiver": receiver,
            "held_class": held_class,
            "K": k_shot,
            "package_id": package["package_id"],
            "registered_classes": list(classes),
            "query_physical_ids": query_ids,
            "arm_predictions": arms,
            "shared_component_receipts": {},
            "query_truth_access": False,
            "target_access": False,
            "formal_p2_authority": False,
            "query_rows_used_for_fit": 0,
            "query_state_updates": 0,
            "query_selection_count": 0,
        }
        artifact["prediction_receipt_sha256"] = runner._sha(artifact)
        path = row_root / f"{index:02d}.json"
        _write(path, artifact)
        rows.append(
            {
                "held_receiver": receiver,
                "held_class": held_class,
                "K": k_shot,
                "package_id": package["package_id"],
                "path": str(Path("rows") / path.name),
                "sha256": _sha(path),
                "prediction_receipt_sha256": artifact["prediction_receipt_sha256"],
            }
        )
    prediction_manifest = {
        "schema": runner.PREDICTION_SCHEMA,
        "candidate_id": runner.CANDIDATE_ID,
        "split_id": runner.SPLIT_ID,
        "run_id": "fixture-run",
        "arms": list(runner.ARMS),
        "row_count": 63,
        "arm_row_prediction_unit_count": 252,
        "rows": rows,
        "package_manifest_sha256": _sha(package_root / "package_manifest.json"),
        "truth_input_seal_sha256": manifest["truth_input_seal_sha256"],
        "rdce_asset_wire_sha256": "a" * 64,
        "query_truth_access": False,
        "target_access": False,
        "query_rows_used_for_fit": 0,
        "query_state_updates": 0,
        "query_selection_count": 0,
        "sourceheld_non_target": True,
        "formal_p2_authority": False,
        "sealed_at_unix_ns": 1,
    }
    prediction_manifest["prediction_set_receipt_sha256"] = runner._sha(prediction_manifest)
    _write(root / "prediction_manifest.json", prediction_manifest)
    return root


def test_fixed_matrix_and_predict_cli_have_four_arms_and_no_truth_surface() -> None:
    receivers = tuple(f"r{index}" for index in range(7))
    classes = tuple(f"c{index}" for index in range(6))
    rows = runner.fixed_row_specs(receivers, classes)
    assert runner.ARMS == ("M0", "M_DA", "M_HEAD", "M_JOINT")
    assert len(rows) == 63
    assert sum(held is None for _receiver, held, _k in rows) == 21
    assert sum(k_shot == 1 for _receiver, _held, k_shot in rows) == 49
    parsed = runner.parse_args(
        [
            "predict",
            "--package-root",
            "packages",
            "--rdce-asset-wire",
            "asset.bin",
            "--rdce-wire-sha256",
            "a" * 64,
            "--run-id",
            "d121",
            "--output-dir",
            "out",
        ]
    )
    assert not hasattr(parsed, "truth_json")
    parser_source = inspect.getsource(runner.parse_args)
    predict_block = parser_source.split('commands.add_parser("predict")', 1)[1].split(
        'commands.add_parser("score")', 1
    )[0]
    assert "truth" not in predict_block.lower()
    assert "receiver" not in predict_block.lower()
    assert set(inspect.signature(runner.predict).parameters) == {"args"}


def test_predict_rebuilds_lbr_for_identity_and_rdce_canonical_banks(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    package_root, _truth, _seal = _write_d104_packages(tmp_path)
    asset_path = tmp_path / "rdce.asset"
    asset_path.write_bytes(b"fixture")
    fake_asset = type("Asset", (), {"split_id": runner.SPLIT_ID})()
    monkeypatch.setattr(runner.d106, "_parse_asset_wire", lambda _path, _sha: fake_asset)
    monkeypatch.setattr(
        runner.d106,
        "fit_rdce_sourceheld_state",
        lambda _asset, _support, _labels, _k: {"receipt": "b" * 64},
    )

    def fake_apply(_state, values: np.ndarray) -> np.ndarray:
        changed = np.asarray(values, dtype=np.float32).copy()
        changed[:, 1] += np.float32(0.17) * changed[:, 0]
        return runner.d106._normalized(changed).astype(np.float32)

    monkeypatch.setattr(runner.d106, "apply_rdce_state", fake_apply)
    original_build = runner.build_lbr_qknn_state
    canonical_id_calls: list[tuple[str, ...]] = []

    def counted_build(bank, physical_ids, *, metric):
        canonical_id_calls.append(tuple(physical_ids))
        return original_build(bank, physical_ids, metric=metric)

    monkeypatch.setattr(runner, "build_lbr_qknn_state", counted_build)
    output = tmp_path / "predictions"
    assert runner.main(
        [
            "predict",
            "--package-root",
            str(package_root),
            "--rdce-asset-wire",
            str(asset_path),
            "--rdce-wire-sha256",
            _sha(asset_path),
            "--run-id",
            "d121-fixture",
            "--output-dir",
            str(output),
        ]
    ) == 0
    prediction_manifest = json.loads((output / "prediction_manifest.json").read_text(encoding="utf-8"))
    assert prediction_manifest["row_count"] == 63
    assert prediction_manifest["arm_row_prediction_unit_count"] == 252
    assert len(canonical_id_calls) == 42  # 21 packages x identity/RDCE rebuilds.
    package_manifest = json.loads((package_root / "package_manifest.json").read_text(encoding="utf-8"))
    first_package = package_manifest["packages"][0]
    with np.load(package_root / first_package["path"], allow_pickle=False) as archive:
        support = np.maximum(np.array(archive["support_pre_relu"], copy=True), np.float32(0.0))
        labels = tuple(archive["support_labels"].astype(str).tolist())
        ids = tuple(archive["support_physical_ids"].astype(str).tolist())
        classes = tuple(archive["registered_classes"].astype(str).tolist())
    expected_identity = runner._canonical_bank_physical_ids(support, labels, ids, classes)
    expected_rdce = runner._canonical_bank_physical_ids(fake_apply({}, support), labels, ids, classes)
    assert canonical_id_calls[0] == expected_identity
    assert canonical_id_calls[1] == expected_rdce
    assert canonical_id_calls[0] != ids
    first_artifact = json.loads(
        (output / prediction_manifest["rows"][0]["path"]).read_text(encoding="utf-8")
    )
    assert set(first_artifact["arm_predictions"]) == set(runner.ARMS)
    for audit in first_artifact["shared_component_receipts"]["lbr_bank_rebuilds"].values():
        assert audit["state_audit"]["query_rows_used_for_fit"] == 0
        assert audit["state_audit"]["query_state_updates"] == 0
        assert audit["state_audit"]["query_selection_count"] == 0
    with pytest.raises(FileExistsError):
        runner.main(
            [
                "predict",
                "--package-root",
                str(package_root),
                "--rdce-asset-wire",
                str(asset_path),
                "--rdce-wire-sha256",
                _sha(asset_path),
                "--run-id",
                "d121-fixture-again",
                "--output-dir",
                str(output),
            ]
        )


def test_score_requires_complete_seal_and_uses_matrix_aggregate_promotion(
    tmp_path: Path,
) -> None:
    package_root, truth_path, seal_path = _write_d104_packages(tmp_path)
    incomplete = _write_complete_prediction_root(
        tmp_path, package_root, name="incomplete", candidate_better=True
    )
    incomplete_manifest_path = incomplete / "prediction_manifest.json"
    incomplete_manifest = json.loads(incomplete_manifest_path.read_text(encoding="utf-8"))
    incomplete_manifest["rows"].pop()
    incomplete_manifest["prediction_set_receipt_sha256"] = runner._sha(
        {
            key: value
            for key, value in incomplete_manifest.items()
            if key != "prediction_set_receipt_sha256"
        }
    )
    _write(incomplete_manifest_path, incomplete_manifest)
    no_event = tmp_path / "no-event.json"
    with pytest.raises(runner.D121G1Error, match="prediction row"):
        runner.main(
            [
                "score",
                "--prediction-root",
                str(incomplete),
                "--truth-json",
                str(truth_path),
                "--truth-input-seal-json",
                str(seal_path),
                "--truth-open-event-json",
                str(no_event),
                "--output-json",
                str(tmp_path / "no-score.json"),
            ]
        )
    assert not no_event.exists()

    passed = _write_complete_prediction_root(
        tmp_path, package_root, name="passed", candidate_better=True
    )
    event, score = tmp_path / "truth-open.json", tmp_path / "scores.json"
    assert runner.main(
        [
            "score",
            "--prediction-root",
            str(passed),
            "--truth-json",
            str(truth_path),
            "--truth-input-seal-json",
            str(seal_path),
            "--truth-open-event-json",
            str(event),
            "--output-json",
            str(score),
        ]
    ) == 0
    result = json.loads(score.read_text(encoding="utf-8"))
    assert len(result["performance_rows"]) == 63
    assert "progression_summary" in result
    assert result["progression_summary"]["promotion_allowed"] is True
    assert result["progression_summary"]["effects"]["HEAD_AT_ID"]["k1_total_correct_count_net"] > 0
    none_row = next(row for row in result["performance_rows"] if row["held_class"] is None)
    assert none_row["arm_metrics"]["M0"]["seen_new_accuracy"] is None
    assert none_row["arm_metrics"]["M0"]["H_old_new"] is None
    assert none_row["same_row_effects"]["HEAD_AT_ID"]["seen_new_correct_count"] is None
    assert json.loads(event.read_text(encoding="utf-8"))["truth_opened_after_all_predictions_committed"] is True
    with pytest.raises(FileExistsError):
        runner.main(
            [
                "score",
                "--prediction-root",
                str(passed),
                "--truth-json",
                str(truth_path),
                "--truth-input-seal-json",
                str(seal_path),
                "--truth-open-event-json",
                str(event),
                "--output-json",
                str(score),
            ]
        )

    failed = _write_complete_prediction_root(
        tmp_path, package_root, name="failed", candidate_better=False
    )
    failed_event, failed_score = tmp_path / "failed-event.json", tmp_path / "failed-score.json"
    assert runner.main(
        [
            "score",
            "--prediction-root",
            str(failed),
            "--truth-json",
            str(truth_path),
            "--truth-input-seal-json",
            str(seal_path),
            "--truth-open-event-json",
            str(failed_event),
            "--output-json",
            str(failed_score),
        ]
    ) == 0
    failed_result = json.loads(failed_score.read_text(encoding="utf-8"))
    assert failed_result["progression_summary"]["promotion_allowed"] is False
    assert (
        failed_result["progression_summary"]["effects"]["HEAD_AT_DA"]
        ["old_correct_count_net"]
        < 0
    )
