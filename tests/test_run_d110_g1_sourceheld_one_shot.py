from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
import sys

import numpy as np
import pytest

from cvsrffi import stage2_zid_student_t_qknn as qknn
from cvsrffi.stage2_d110_sourceheld_split import SCORER_MEMBERS


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "code" / "scripts" / "run_d110_g1_sourceheld_one_shot.py"


def _module():
    spec = importlib.util.spec_from_file_location("run_d110_g1_sourceheld_one_shot", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, sort_keys=True), encoding="utf-8")


def _source_arrays() -> dict[str, np.ndarray]:
    rng = np.random.default_rng(110813)
    values: dict[str, list] = {
        "z_id": [], "z_dom": [], "pre_relu": [], "labels": [],
        "receiver_ids": [], "day_ids": [], "physical_ids": [],
        "scenario_names": [], "observation_ids": [],
    }
    for receiver in range(7):
        for class_index in range(6):
            for day in range(4):
                for sample in range(7):
                    row = rng.normal(0.0, 0.012, size=160).astype(np.float32)
                    row[class_index] += np.float32(1.0)
                    row[12 + receiver] += np.float32(0.18)
                    row[30 + day] += np.float32(0.09)
                    row[80 + sample] += np.float32(0.02)
                    values["z_id"].append(np.maximum(row, np.float32(0.0)))
                    values["z_dom"].append(rng.normal(0.0, 0.01, size=160).astype(np.float32))
                    values["pre_relu"].append(row.copy())
                    values["labels"].append(f"tx{class_index}")
                    values["receiver_ids"].append(f"rx{receiver}")
                    values["day_ids"].append(f"day{day}")
                    values["physical_ids"].append(f"p-{receiver:02d}-{class_index:02d}-{day:02d}-{sample:02d}")
                    values["scenario_names"].append("leo_weak")
                    values["observation_ids"].append(f"o-{len(values['observation_ids']):04d}")
    result = {
        "z_id": np.stack(values["z_id"]).astype(np.float32),
        "z_dom": np.stack(values["z_dom"]).astype(np.float32),
        "pre_relu": np.stack(values["pre_relu"]).astype(np.float32),
        "labels": np.asarray(values["labels"]),
        "receiver_ids": np.asarray(values["receiver_ids"]),
        "day_ids": np.asarray(values["day_ids"]),
        "physical_ids": np.asarray(values["physical_ids"]),
        "scenario_names": np.asarray(values["scenario_names"]),
        "observation_ids": np.asarray(values["observation_ids"]),
        "class_ids": np.asarray([f"tx{index}" for index in range(6)]),
    }
    assert len(result["z_id"]) == 1176 and tuple(result) == SCORER_MEMBERS
    return result


def _write_source(tmp_path: Path, module) -> tuple[Path, Path]:
    archive = tmp_path / "d110_source.npz"
    np.savez(archive, **_source_arrays())
    manifest = tmp_path / "d110_source_manifest.json"
    _write_json(manifest, {
        "schema": module.SCORER_SCHEMA, "candidate_id": module.CANDIDATE_ID,
        "split_id": module.SPLIT_ID, "role": "source_val_scorer_only",
        "archive": {"sha256": _sha(archive)},
        "exact_member_allowlist": list(SCORER_MEMBERS), "row_count": 1176,
        "asset_access": False, "gradient_access": False, "selection_access": False,
        "target_access": False, "formal_query_access": False,
        "performance_computed": False, "d106_prepare_member_compatible": True,
    })
    return archive, manifest


def _write_tap(tmp_path: Path) -> Path:
    rng = np.random.default_rng(110)
    rows, labels, receivers, days, physical, observations = [], [], [], [], [], []
    for receiver in range(7):
        for day in range(4):
            cell = receiver * 4 + day
            for class_index in range(6):
                for sample in range(4 if (cell + class_index) % 2 == 0 else 3):
                    row = rng.normal(0.0, 0.012, size=160).astype(np.float32)
                    row[0] = np.float32(1.0 + 0.02 * receiver)
                    row[10 + class_index] += np.float32(0.42)
                    row[40 + day] += np.float32(0.08)
                    row[80 + receiver] += np.float32(0.06)
                    row[120 + sample] += np.float32(0.015)
                    rows.append(row); labels.append(f"tx{class_index}")
                    receivers.append(f"rx{receiver}"); days.append(f"day{day}")
                    physical.append(f"tap-{receiver:02d}-{day:02d}-{class_index:02d}-{sample:02d}")
                    observations.append(f"tap-o-{len(observations):04d}")
    pre_relu = np.stack(rows).astype(np.float32)
    assert pre_relu.shape == (588, 160)
    path = tmp_path / "d106_tap.npz"
    np.savez(path, pre_relu=pre_relu, z_dom=np.zeros_like(pre_relu),
             tx_labels=np.asarray(labels), receiver_ids=np.asarray(receivers),
             day_ids=np.asarray(days), physical_ids=np.asarray(physical),
             scenario_names=np.full(588, "leo_weak"), observation_ids=np.asarray(observations))
    return path


def _prepare_and_predict(tmp_path: Path, module, *, run_id: str = "d110-test") -> tuple[Path, Path, Path]:
    archive, source_manifest = _write_source(tmp_path, module)
    packages = tmp_path / "packages"
    assert module.main(["prepare", "--source-val-archive", str(archive),
                        "--source-val-manifest", str(source_manifest),
                        "--output-dir", str(packages)]) == 0
    tap = _write_tap(tmp_path)
    predictions = tmp_path / f"predictions-{run_id}"
    assert module.main(["predict", "--package-root", str(packages),
                        "--d106-tap-archive", str(tap),
                        "--d106-tap-archive-sha256", _sha(tap), "--run-id", run_id,
                        "--output-dir", str(predictions)]) == 0
    return packages, tap, predictions


def test_minimal_d110_g1_prepare_predict_score_end_to_end(tmp_path: Path) -> None:
    module = _module()
    packages, tap, predictions = _prepare_and_predict(tmp_path, module)
    package_manifest = json.loads((packages / "package_manifest.json").read_text(encoding="utf-8"))
    prediction_manifest = json.loads((predictions / "prediction_manifest.json").read_text(encoding="utf-8"))
    assert package_manifest["schema"] == module.PACKAGE_SCHEMA
    assert package_manifest["candidate_id"] == module.CANDIDATE_ID
    assert package_manifest["split_id"] == module.SPLIT_ID
    assert package_manifest["package_count"] == 21
    assert prediction_manifest["row_count"] == 63
    assert prediction_manifest["arm_row_prediction_unit_count"] == 252
    assert prediction_manifest["package_manifest_sha256"] == _sha(packages / "package_manifest.json")
    assert prediction_manifest["truth_input_seal_sha256"] == package_manifest["truth_input_seal_sha256"]
    assert prediction_manifest["d106_tap_archive_sha256"] == _sha(tap)

    first = next(row for row in prediction_manifest["rows"] if row["K"] == 5)
    artifact = json.loads((predictions / first["path"]).read_text(encoding="utf-8"))
    package_row = next(row for row in package_manifest["packages"] if row["package_id"] == first["package_id"])
    with np.load(packages / package_row["path"], allow_pickle=False) as package:
        support = np.maximum(np.array(package["support_pre_relu"], copy=True), np.float32(0.0))
        labels = package["support_labels"].astype(str).tolist()
        query = np.maximum(np.array(package["query_pre_relu"], copy=True), np.float32(0.0))
        classes = tuple(package["registered_classes"].astype(str).tolist())
    lock = module.d106_g1._lock(5, package_row["sha256"])
    bank = qknn.build_typed_zid_support_bank(support, labels, classes, config=lock)
    baseline = qknn.score_zid_student_t_logits(bank, query, metric=qknn.identity_shared_psd_metric(config=lock))
    assert artifact["arm_predictions"]["M0"] == [classes[int(np.argmax(row))] for row in baseline]
    for row in prediction_manifest["rows"]:
        artifact = json.loads((predictions / row["path"]).read_text(encoding="utf-8"))
        if artifact["K"] == 1:
            assert artifact["arm_predictions"]["M_HEAD"] == artifact["arm_predictions"]["M0"]
            assert artifact["arm_predictions"]["M_JOINT"] == artifact["arm_predictions"]["M_DA"]
        for audit in artifact["shared_component_receipts"]["arm_state_audits"].values():
            assert audit["query_rows_used_for_fit"] == audit["query_state_updates"] == 0

    score, event = tmp_path / "scores.json", tmp_path / "truth_open_event.json"
    assert module.main(["score", "--prediction-root", str(predictions),
                        "--truth-json", str(packages / "scorer_only" / "truth.json"),
                        "--truth-input-seal-json", str(packages / "scorer_only" / "truth_input_seal.json"),
                        "--truth-open-event-json", str(event), "--output-json", str(score)]) == 0
    result = json.loads(score.read_text(encoding="utf-8"))
    opened = json.loads(event.read_text(encoding="utf-8"))
    assert len(result["performance_rows"]) == 63
    assert opened["truth_opened_after_all_predictions_committed"] is True


def test_fail_closed_before_truth_for_bad_split_hash_or_incomplete_matrix(tmp_path: Path) -> None:
    module = _module()
    archive, source_manifest = _write_source(tmp_path, module)
    bad_source = json.loads(source_manifest.read_text(encoding="utf-8"))
    bad_source["split_id"] = "d104_source_seed104713_v2"
    bad_source_path = tmp_path / "bad-source.json"
    _write_json(bad_source_path, bad_source)
    with pytest.raises(module.D110G1Error, match="manifest"):
        module.main(["prepare", "--source-val-archive", str(archive),
                     "--source-val-manifest", str(bad_source_path),
                     "--output-dir", str(tmp_path / "bad-packages")])

    packages, tap, predictions = _prepare_and_predict(tmp_path, module)
    with pytest.raises(module.d110_g0.base.OneShotG0Error, match="SHA256"):
        module.main(["predict", "--package-root", str(packages),
                     "--d106-tap-archive", str(tap), "--d106-tap-archive-sha256", "0" * 64,
                     "--run-id", "bad-hash", "--output-dir", str(tmp_path / "bad-hash")])
    assert not (tmp_path / "bad-hash").exists()

    truth_seal = packages / "scorer_only" / "truth_input_seal.json"
    original_truth_seal = truth_seal.read_bytes()
    truth_seal.write_bytes(original_truth_seal + b"\n")
    seal_score, seal_event = tmp_path / "seal-score.json", tmp_path / "seal-event.json"
    with pytest.raises(module.D110G1Error, match="truth-input seal SHA drift"):
        module.main(["score", "--prediction-root", str(predictions),
                     "--truth-json", str(packages / "scorer_only" / "truth.json"),
                     "--truth-input-seal-json", str(truth_seal),
                     "--truth-open-event-json", str(seal_event), "--output-json", str(seal_score)])
    assert not seal_event.exists() and not seal_score.exists()
    truth_seal.write_bytes(original_truth_seal)

    manifest_path = predictions / "prediction_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["rows"].pop()
    manifest["prediction_set_receipt_sha256"] = module._sha({key: value for key, value in manifest.items() if key != "prediction_set_receipt_sha256"})
    _write_json(manifest_path, manifest)
    score, event = tmp_path / "broken-score.json", tmp_path / "broken-event.json"
    with pytest.raises((module.D110G1Error, module.d106_g1.D106G1Error), match="closure"):
        module.main(["score", "--prediction-root", str(predictions),
                     "--truth-json", str(packages / "scorer_only" / "truth.json"),
                     "--truth-input-seal-json", str(packages / "scorer_only" / "truth_input_seal.json"),
                     "--truth-open-event-json", str(event), "--output-json", str(score)])
    assert not event.exists() and not score.exists()
