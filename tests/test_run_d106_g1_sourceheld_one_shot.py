from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import copy

import numpy as np
import pytest


SCRIPT = Path(__file__).resolve().parents[1] / "code" / "scripts" / "run_d106_g1_sourceheld_one_shot.py"
SPEC = importlib.util.spec_from_file_location("run_d106_g1_sourceheld_one_shot", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
g1 = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(g1)


def _write(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, sort_keys=True), encoding="utf-8")


def test_fixed_matrix_and_predict_cli_have_no_truth_or_slice_selector() -> None:
    receivers = tuple(f"r{i}" for i in range(7))
    classes = tuple(f"c{i}" for i in range(6))
    rows = g1.fixed_row_specs(receivers, classes)
    assert len(rows) == 63
    assert sum(held is None for _, held, _ in rows) == 21
    assert sum(k == 1 for _, _, k in rows) == 49
    predict = g1.parse_args([
        "predict", "--package-root", ".", "--rdce-asset-wire", "asset.bin",
        "--rdce-wire-sha256", "0" * 64, "--rcmr-method-lock", "lock.json",
        "--rcmr-method-lock-sha256", g1.RCMR_LOCK_SHA256, "--output-dir", "out",
    ])
    assert not hasattr(predict, "truth_json")
    for forbidden in ("receiver", "held_class", "K", "scene", "row"):
        assert not hasattr(predict, forbidden)


def test_rdce_state_is_support_only_and_reusable(monkeypatch: pytest.MonkeyPatch) -> None:
    basis = np.zeros((3, 160), dtype=np.float64)
    basis[0, 0] = basis[1, 1] = basis[2, 2] = 1.0
    monkeypatch.setattr(g1, "decode_d106_rdce_basis", lambda asset: basis)
    monkeypatch.setattr(g1, "decode_d106_rdce_tau", lambda asset: np.full(3, 1.0e-30))
    asset = type("Asset", (), {"asset_receipt_sha256": "a" * 64})()
    rng = np.random.default_rng(106)
    support = rng.normal(size=(30, 160)).astype(np.float32)
    labels = tuple(f"c{index // 5}" for index in range(30))
    state = g1.fit_rdce_sourceheld_state(asset, support, labels, 5)
    first = g1.apply_rdce_state(state, support[:4])
    second = g1.apply_rdce_state(state, support[:4])
    assert np.array_equal(first, second)
    assert state["payload"]["query_rows_used_for_fit"] == 0
    assert state["payload"]["query_state_updates"] == 0
    assert state["payload"]["scope"] == "SOURCE_HELD_NON_TARGET_NO_P2_AUTHORITY"
    assert np.all(state["attenuation"] <= float(g1.MAX_ATTENUATION_FP16))
    assert np.any(state["attenuation"] == float(g1.MAX_ATTENUATION_FP16))


def test_prepare_builds_21_truth_separated_packages_without_fits(tmp_path: Path) -> None:
    receivers = tuple(f"r{i}" for i in range(7))
    classes = tuple(f"c{i}" for i in range(6))
    receiver_ids = []
    labels = []
    physical_ids = []
    day_ids = []
    for receiver in receivers:
        for class_id in classes:
            for index in range(11):
                receiver_ids.append(receiver)
                labels.append(class_id)
                physical_ids.append(f"{receiver}-{class_id}-{index:02d}")
                day_ids.append(f"d{index % 4}")
    count = len(physical_ids)
    rng = np.random.default_rng(104)
    archive_path = tmp_path / "source_val.npz"
    np.savez(
        archive_path,
        z_id=rng.normal(size=(count, 160)).astype(np.float32),
        z_dom=rng.normal(size=(count, 160)).astype(np.float32),
        pre_relu=rng.normal(size=(count, 160)).astype(np.float32),
        labels=np.asarray(labels), receiver_ids=np.asarray(receiver_ids),
        day_ids=np.asarray(day_ids), physical_ids=np.asarray(physical_ids),
        scenario_names=np.asarray(["source-held"] * count),
        observation_ids=np.asarray([f"o{i}" for i in range(count)]),
        class_ids=np.asarray(classes),
    )
    manifest_path = tmp_path / "source_val.manifest.json"
    _write(manifest_path, {
        "candidate_id": g1.D104_CANDIDATE_ID, "split_id": g1.SPLIT_ID,
        "role": "source_val_scorer_only", "archive": {"sha256": g1._file_sha(archive_path)},
        "asset_access": False, "gradient_access": False, "selection_access": False,
        "target_access": False, "formal_query_access": False,
    })
    output = tmp_path / "packages"
    assert g1.main([
        "prepare", "--source-val-archive", str(archive_path),
        "--source-val-manifest", str(manifest_path), "--output-dir", str(output),
    ]) == 0
    package_manifest = json.loads((output / "package_manifest.json").read_text(encoding="utf-8"))
    truth = json.loads((output / "scorer_only" / "truth.json").read_text(encoding="utf-8"))
    assert package_manifest["package_count"] == 21
    assert truth["package_count"] == 21
    assert package_manifest["query_truth_present"] is False
    with np.load(output / package_manifest["packages"][0]["path"], allow_pickle=False) as package:
        assert set(package.files) == g1.PACKAGE_KEYS
        assert not any("truth" in name for name in package.files)


def test_independent_scorer_requires_sealed_complete_predictions(tmp_path: Path) -> None:
    receivers = tuple(f"r{i}" for i in range(7))
    classes = tuple(f"c{i}" for i in range(6))
    prediction_root = tmp_path / "predictions"
    row_root = prediction_root / "rows"
    row_root.mkdir(parents=True)
    truth_packages = []
    query_ids_by_package = {}
    for receiver in receivers:
        for k_shot in g1.K_VALUES:
            query_ids = [f"{receiver}-k{k_shot}-{class_id}" for class_id in classes]
            query_ids_by_package[(receiver, k_shot)] = query_ids
            truth_packages.append({
                "package_id": f"{receiver}-k{k_shot}",
                "query_physical_ids": query_ids,
                "query_truth_labels": list(classes),
            })
    manifest_rows = []
    for index, (receiver, held_class, k_shot) in enumerate(g1.fixed_row_specs(receivers, classes)):
        predictions = list(classes)
        artifact = {
            "schema": g1.PREDICTION_SCHEMA + ".row",
            "held_receiver": receiver, "held_class": held_class, "K": k_shot,
            "package_id": f"{receiver}-k{k_shot}",
            "registered_classes": list(classes),
            "query_physical_ids": query_ids_by_package[(receiver, k_shot)],
            "arm_predictions": {arm: predictions for arm in g1.ARMS},
            "shared_component_receipts": {}, "query_truth_access": False,
            "target_access": False, "formal_p2_authority": False,
            "query_state_updates": 0,
        }
        artifact["prediction_receipt_sha256"] = g1._sha(artifact)
        path = row_root / f"{index:02d}.json"
        _write(path, artifact)
        manifest_rows.append({
            "held_receiver": receiver, "held_class": held_class, "K": k_shot,
            "package_id": f"{receiver}-k{k_shot}",
            "path": str(Path("rows") / path.name), "sha256": g1._file_sha(path),
            "prediction_receipt_sha256": artifact["prediction_receipt_sha256"],
        })
    manifest = {
        "schema": g1.PREDICTION_SCHEMA, "split_id": g1.SPLIT_ID,
        "row_count": 63, "arm_row_prediction_unit_count": 252,
        "rows": manifest_rows, "query_truth_access": False,
    }
    manifest["prediction_set_receipt_sha256"] = g1._sha(manifest)
    _write(prediction_root / "prediction_manifest.json", manifest)
    truth_path = tmp_path / "truth.json"
    _write(truth_path, {
        "schema": "cvs.d104_r1.rxid_angq.held_truth.v2", "split_id": g1.SPLIT_ID,
        "package_count": 21, "predictor_access": False, "packages": truth_packages,
    })
    seal_path = tmp_path / "truth_input_seal.json"
    _write(seal_path, {
        "schema": "cvs.d104_r1.rxid_angq.truth_input_seal.v1", "split_id": g1.SPLIT_ID,
        "package_count": 21, "package_ids": [row["package_id"] for row in truth_packages],
        "truth_package_root_sha256": g1.canonical_sha256(truth_packages),
        "predictor_truth_access": False,
    })
    output = tmp_path / "scores.json"
    event = tmp_path / "truth_open.json"

    def rejected_before_truth(index: int, mutate) -> None:
        original_artifact = json.loads(
            (prediction_root / manifest_rows[index]["path"]).read_text(encoding="utf-8")
        )
        changed = copy.deepcopy(original_artifact)
        mutate(changed)
        changed["prediction_receipt_sha256"] = g1._sha({
            key: value for key, value in changed.items()
            if key != "prediction_receipt_sha256"
        })
        artifact_path = prediction_root / manifest_rows[index]["path"]
        _write(artifact_path, changed)
        changed_manifest = copy.deepcopy(manifest)
        for name in ("held_receiver", "held_class", "K", "package_id", "prediction_receipt_sha256"):
            changed_manifest["rows"][index][name] = changed[name]
        changed_manifest["rows"][index]["sha256"] = g1._file_sha(artifact_path)
        changed_manifest["prediction_set_receipt_sha256"] = g1._sha({
            key: value for key, value in changed_manifest.items()
            if key != "prediction_set_receipt_sha256"
        })
        _write(prediction_root / "prediction_manifest.json", changed_manifest)
        with pytest.raises(g1.D106G1Error):
            g1.main([
                "score", "--prediction-root", str(prediction_root),
                "--truth-json", str(truth_path),
                "--truth-input-seal-json", str(seal_path),
                "--truth-open-event-json", str(event), "--output-json", str(output),
            ])
        assert not event.exists() and not output.exists()
        _write(artifact_path, original_artifact)
        _write(prediction_root / "prediction_manifest.json", manifest)

    rejected_before_truth(0, lambda row: row["arm_predictions"].pop("M_JOINT"))
    rejected_before_truth(0, lambda row: row["arm_predictions"]["M0"].pop())
    first_coordinate = (manifest_rows[0]["held_receiver"], manifest_rows[0]["held_class"], manifest_rows[0]["K"])
    rejected_before_truth(
        62,
        lambda row: row.update(
            held_receiver=first_coordinate[0], held_class=first_coordinate[1], K=first_coordinate[2]
        ),
    )

    assert g1.main([
        "score", "--prediction-root", str(prediction_root), "--truth-json", str(truth_path),
        "--truth-input-seal-json", str(seal_path),
        "--truth-open-event-json", str(event), "--output-json", str(output),
    ]) == 0
    scored = json.loads(output.read_text(encoding="utf-8"))
    assert scored["schema"] == g1.SCORE_SCHEMA
    assert len(scored["performance_rows"]) == 63
    assert scored["prediction_artifact_committed_before_truth"] is True
    with pytest.raises(FileExistsError):
        g1.main([
            "score", "--prediction-root", str(prediction_root), "--truth-json", str(truth_path),
            "--truth-input-seal-json", str(seal_path),
            "--truth-open-event-json", str(event), "--output-json", str(output),
        ])
