from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from cvsrffi import stage2_d122_rdce_ground_head as d122
from cvsrffi.stage2_d112_seam_bundle import FEATURE_DIM, build_d112_source_held_g1_bundle
from scripts import run_d122_g1_sourceheld_one_shot as runner


CLASSES = tuple(f"tx-{index}" for index in range(6))
RECEIVERS = tuple(f"rx-{index}" for index in range(7))


def _sha_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, ensure_ascii=True, sort_keys=True), encoding="utf-8")


def _bundle(*, checkpoint_sha256: str = "1" * 64, phase1_seal_sha256: str = "3" * 64):
    ground = np.zeros((6, FEATURE_DIM), dtype=np.float64)
    for index in range(6):
        ground[index, 10 + index] = 1.0
    q0 = np.sum(ground, axis=0)
    q0 /= np.linalg.norm(q0)
    U = np.zeros((3, FEATURE_DIM), dtype=np.float64)
    U[np.arange(3), np.arange(3)] = 1.0
    return build_d112_source_held_g1_bundle(
        class_registry=CLASSES,
        g=ground,
        q0=q0,
        U=U,
        sigma0_r=np.linspace(0.0020, 0.0025, 6),
        sigma0_amb=np.linspace(0.0020, 0.0025, 6),
        v_g_r=np.linspace(0.0010, 0.0015, 6),
        v_g_amb=np.linspace(0.0010, 0.0015, 6),
        tau_h_r=0.004,
        checkpoint_sha256=checkpoint_sha256,
        source_aggregate_sha256="2" * 64,
        phase1_seal_sha256=phase1_seal_sha256,
        source_held_split_sha256="4" * 64,
    )


def _basis() -> np.ndarray:
    basis = np.zeros((3, FEATURE_DIM), dtype=np.float64)
    basis[0, 0] = 0.99
    basis[0, 1] = 0.10
    basis[1, 1] = 0.93
    basis[1, 2] = -0.16
    basis[2, 0] = -0.09
    basis[2, 2] = 0.91
    return basis


def _fake_rdce_state(support: np.ndarray, k_shot: int) -> dict[str, object]:
    normalized = np.asarray(support, dtype=np.float64)
    normalized /= np.linalg.norm(normalized, axis=1, keepdims=True)
    attenuation = np.asarray([0.300048828125, 0.349853515625, 0.39990234375], dtype=np.float64)
    payload = {
        "scope": "SOURCE_HELD_NON_TARGET_NO_P2_AUTHORITY",
        "asset_receipt_sha256": "9" * 64,
        "K": k_shot,
        "attenuation_fp16": [float(value) for value in attenuation],
        "support_root_sha256": hashlib.sha256(
            np.ascontiguousarray(normalized, dtype=np.float64).tobytes()
        ).hexdigest(),
        "query_rows_used_for_fit": 0,
        "query_state_updates": 0,
    }
    receipt = hashlib.sha256(
        json.dumps(payload, ensure_ascii=True, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    return {
        "basis": _basis(),
        "attenuation": attenuation,
        "payload": payload,
        "receipt": receipt,
    }


def _fake_asset(*, checkpoint_sha256: str, tap_sha256: str, tap_receipt_sha256: str):
    return SimpleNamespace(
        split_id=runner.SPLIT_ID,
        checkpoint_sha256=checkpoint_sha256,
        tap_sha256=tap_sha256,
        tap_receipt_sha256=tap_receipt_sha256,
    )


def _package_arrays(k_shot: int) -> tuple[np.ndarray, list[str], np.ndarray, list[str]]:
    rows: list[np.ndarray] = []
    labels: list[str] = []
    for class_index, class_id in enumerate(CLASSES):
        for sample in range(k_shot):
            row = np.zeros(FEATURE_DIM, dtype=np.float32)
            row[10 + class_index] = 1.0
            row[0] = np.float32(0.012 * (sample + 1))
            row[2] = np.float32(-0.004 * (class_index + 1) * (sample + 1))
            row /= np.linalg.norm(row)
            rows.append(row)
            labels.append(class_id)
    support = np.asarray(rows, dtype=np.float32)
    query = np.asarray(
        [support[class_index * k_shot] for class_index in range(len(CLASSES))],
        dtype=np.float32,
    )
    return support, labels, query, list(CLASSES)


def _make_frozen_packages(root: Path) -> tuple[Path, Path, Path]:
    root.mkdir()
    packages: list[dict[str, object]] = []
    truth_packages: list[dict[str, object]] = []
    for receiver in RECEIVERS:
        for k_shot in runner.K_VALUES:
            support, labels, query, truth_labels = _package_arrays(k_shot)
            package_id = f"{receiver}-k{k_shot}"
            query_ids = [f"{package_id}-q{index}" for index in range(len(query))]
            support_ids = [f"{package_id}-s{index}" for index in range(len(support))]
            path = root / f"{package_id}.npz"
            np.savez(
                path,
                support_pre_relu=support,
                support_zdom=np.zeros_like(support),
                support_labels=np.asarray(labels),
                support_physical_ids=np.asarray(support_ids),
                query_pre_relu=query,
                query_physical_ids=np.asarray(query_ids),
                registered_classes=np.asarray(CLASSES),
            )
            packages.append(
                {
                    "held_receiver": receiver,
                    "K": k_shot,
                    "package_id": package_id,
                    "path": path.name,
                    "sha256": _sha_file(path),
                }
            )
            truth_packages.append(
                {
                    "package_id": package_id,
                    "query_physical_ids": query_ids,
                    "query_truth_labels": truth_labels,
                }
            )
    truth_dir = root / "truth"
    truth_dir.mkdir()
    truth = {
        "schema": "cvs.d104_r1.rxid_angq.held_truth.v2",
        "split_id": runner.SPLIT_ID,
        "package_count": 21,
        "predictor_access": False,
        "packages": truth_packages,
    }
    truth_path = truth_dir / "truth.json"
    _write_json(truth_path, truth)
    seal = {
        "split_id": runner.SPLIT_ID,
        "package_count": 21,
        "predictor_truth_access": False,
        "package_ids": [str(row["package_id"]) for row in packages],
        "truth_package_root_sha256": runner.d106.canonical_sha256(truth_packages),
    }
    seal_path = truth_dir / "truth_input_seal.json"
    _write_json(seal_path, seal)
    manifest = {
        "schema": runner.PACKAGE_SCHEMA,
        "candidate_id": runner.d106.D104_CANDIDATE_ID,
        "split_id": runner.SPLIT_ID,
        "receiver_ids": list(RECEIVERS),
        "class_ids": list(CLASSES),
        "query_truth_present": False,
        "target_access": False,
        "truth_input_seal_sha256": _sha_file(seal_path),
        "packages": packages,
    }
    _write_json(root / "package_manifest.json", manifest)
    return root, truth_path, seal_path


def test_predict_then_separate_score_closes_full_63_by_4_surface(tmp_path, monkeypatch) -> None:
    package_root, truth_path, seal_path = _make_frozen_packages(tmp_path / "packages")
    wire = tmp_path / "rdce.asset.wire"
    wire.write_bytes(b"placeholder")
    tap_sha256 = _sha_file(wire)

    def fake_fit(_asset, support, _labels, k_shot):
        return _fake_rdce_state(np.asarray(support, dtype=np.float32), k_shot)

    def fake_apply(state, rows):
        return d122._d106_like_transform(
            np.asarray(rows, dtype=np.float32),
            np.asarray(state["basis"], dtype=np.float64),
            np.asarray(state["attenuation"], dtype=np.float64),
        )[0]

    monkeypatch.setattr(
        runner.d106,
        "_parse_asset_wire",
        lambda *_args: _fake_asset(
            checkpoint_sha256="c" * 64,
            tap_sha256=tap_sha256,
            tap_receipt_sha256=tap_sha256,
        ),
    )
    monkeypatch.setattr(runner.d106, "fit_rdce_sourceheld_state", fake_fit)
    monkeypatch.setattr(runner.d106, "apply_rdce_state", fake_apply)
    monkeypatch.setattr(
        runner.d112,
        "_g1_bundle",
        lambda *_args: _bundle(
            checkpoint_sha256="c" * 64, phase1_seal_sha256=tap_sha256
        ),
    )
    prediction_root = tmp_path / "prediction"
    predict_args = argparse.Namespace(
        package_root=package_root,
        rdce_asset_wire=wire,
        rdce_wire_sha256="a" * 64,
        d106_tap_archive=wire,
        d106_tap_receipt=wire,
        d106_tap_archive_sha256=tap_sha256,
        checkpoint_sha256="c" * 64,
        run_id="d122-test",
        output_dir=prediction_root,
    )
    assert runner.predict(predict_args) == 0
    prediction = json.loads((prediction_root / "prediction_manifest.json").read_text(encoding="utf-8"))
    assert prediction["row_count"] == 63
    assert prediction["arm_row_prediction_unit_count"] == 252
    assert prediction["query_truth_access"] is False
    assert len(list((prediction_root / "rows").glob("*.json"))) == 63
    sample = json.loads(next((prediction_root / "rows").glob("*.json")).read_text(encoding="utf-8"))
    assert set(sample["arm_predictions"]) == set(runner.ARMS)
    assert sample["shared_component_receipts"]["new_class_logit_boundary_bit_exact"] is True
    assert sample["shared_component_receipts"]["M_JOINT_state_audit"]["query_selection_count"] == 0

    score_path = tmp_path / "scores.json"
    event_path = tmp_path / "truth-open.json"
    score_args = argparse.Namespace(
        prediction_root=prediction_root,
        truth_json=truth_path,
        truth_input_seal_json=seal_path,
        truth_open_event_json=event_path,
        output_json=score_path,
    )
    assert runner.score(score_args) == 0
    scores = json.loads(score_path.read_text(encoding="utf-8"))
    assert len(scores["performance_rows"]) == 63
    assert scores["interaction_definition"] == "(M_JOINT-M_DA)-(M_HEAD-M0)"
    assert event_path.exists()


def test_predict_cli_has_no_truth_input() -> None:
    args = runner.parse_args(
        [
            "predict",
            "--package-root",
            "packages",
            "--rdce-asset-wire",
            "rdce.wire",
            "--rdce-wire-sha256",
            "a" * 64,
            "--d106-tap-archive",
            "tap.npz",
            "--d106-tap-receipt",
            "tap.json",
            "--d106-tap-archive-sha256",
            "b" * 64,
            "--checkpoint-sha256",
            "c" * 64,
            "--run-id",
            "d122-cli",
            "--output-dir",
            "out",
        ]
    )
    assert args.command == "predict"
    assert "truth_json" not in vars(args)
    assert "truth_input_seal_json" not in vars(args)


def test_global_joint_asset_binding_failure_cannot_seal_prediction(tmp_path, monkeypatch) -> None:
    package_root, _truth_path, _seal_path = _make_frozen_packages(tmp_path / "packages")
    wire = tmp_path / "rdce.asset.wire"
    wire.write_bytes(b"placeholder")
    tap_sha256 = _sha_file(wire)

    def invalid_fit(_asset, support, _labels, k_shot):
        state = _fake_rdce_state(np.asarray(support, dtype=np.float32), k_shot)
        state["receipt"] = "0" * 64
        return state

    def fake_apply(state, rows):
        return d122._d106_like_transform(
            np.asarray(rows, dtype=np.float32),
            np.asarray(state["basis"], dtype=np.float64),
            np.asarray(state["attenuation"], dtype=np.float64),
        )[0]

    monkeypatch.setattr(
        runner.d106,
        "_parse_asset_wire",
        lambda *_args: _fake_asset(
            checkpoint_sha256="c" * 64,
            tap_sha256=tap_sha256,
            tap_receipt_sha256=tap_sha256,
        ),
    )
    monkeypatch.setattr(runner.d106, "fit_rdce_sourceheld_state", invalid_fit)
    monkeypatch.setattr(runner.d106, "apply_rdce_state", fake_apply)
    monkeypatch.setattr(
        runner.d112,
        "_g1_bundle",
        lambda *_args: _bundle(
            checkpoint_sha256="c" * 64, phase1_seal_sha256=tap_sha256
        ),
    )
    prediction_root = tmp_path / "prediction-invalid"
    args = argparse.Namespace(
        package_root=package_root,
        rdce_asset_wire=wire,
        rdce_wire_sha256="a" * 64,
        d106_tap_archive=wire,
        d106_tap_receipt=wire,
        d106_tap_archive_sha256=tap_sha256,
        checkpoint_sha256="c" * 64,
        run_id="d122-invalid",
        output_dir=prediction_root,
    )
    with pytest.raises(runner.D122G1Error, match="global joint component binding"):
        runner.predict(args)
    assert not (prediction_root / "prediction_manifest.json").exists()


def test_valid_but_different_rdce_lineage_cannot_seal_prediction(tmp_path, monkeypatch) -> None:
    package_root, _truth_path, _seal_path = _make_frozen_packages(tmp_path / "packages")
    wire = tmp_path / "rdce.asset.wire"
    wire.write_bytes(b"placeholder")
    tap_sha256 = _sha_file(wire)
    # Every individual identifier is syntactically valid, but the checkpoint
    # identity is from a different lineage and must fail before any row output.
    monkeypatch.setattr(
        runner.d106,
        "_parse_asset_wire",
        lambda *_args: _fake_asset(
            checkpoint_sha256="d" * 64,
            tap_sha256=tap_sha256,
            tap_receipt_sha256=tap_sha256,
        ),
    )
    prediction_root = tmp_path / "prediction-lineage-invalid"
    args = argparse.Namespace(
        package_root=package_root,
        rdce_asset_wire=wire,
        rdce_wire_sha256="a" * 64,
        d106_tap_archive=wire,
        d106_tap_receipt=wire,
        d106_tap_archive_sha256=tap_sha256,
        checkpoint_sha256="c" * 64,
        run_id="d122-lineage-invalid",
        output_dir=prediction_root,
    )
    with pytest.raises(runner.D122G1Error, match="RDCE asset lineage"):
        runner.predict(args)
    assert not (prediction_root / "prediction_manifest.json").exists()
