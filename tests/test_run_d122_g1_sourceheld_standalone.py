from __future__ import annotations

import hashlib
import json
import struct
import sys
from dataclasses import replace
from argparse import Namespace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from cvsrffi import stage2_d122_rdce_ground_head as d122
from cvsrffi.stage2_d112_seam_bundle import FEATURE_DIM, build_d112_source_held_g1_bundle
from scripts import run_d122_g1_sourceheld_standalone as runner


def _valid_asset_and_wire(tmp_path):
    basis_codes = np.zeros((runner.RDCE_RANK, runner.Z_DIM), dtype=np.int8)
    basis_codes[np.arange(runner.RDCE_RANK), np.arange(runner.RDCE_RANK)] = 127
    basis_scales = np.full(runner.RDCE_RANK, np.float16(1.0 / 127.0), dtype="<f2")
    tau_codes = np.asarray([1, 2, 3], dtype=np.int8)
    tau_scales = np.full(runner.RDCE_RANK, np.float16(0.01), dtype="<f2")
    spectrum_codes = np.asarray([3, 2, 1], dtype=np.int8)
    spectrum_scales = np.full(runner.RDCE_RANK, np.float16(0.02), dtype="<f2")
    hashes = {name: "a" * 64 for name in (
        "checkpoint_sha256", "runtime_sha256", "method_lock_sha256",
        "tap_sha256", "construction_code_sha256", "content_root_sha256",
        "source_receipt_sha256", "tap_receipt_sha256",
    )}
    authority = hashlib.sha256(
        runner._canonical_bytes(
            {
                "schema": runner.TAP_AUTHORITY_SCHEMA,
                "loader": "load_d106_phase1_ls_tap",
                "archive_sha256": hashes["tap_sha256"],
                "receipt_sha256": hashes["tap_receipt_sha256"],
            }
        )
    ).hexdigest()
    asset = runner._StandaloneRDCEAsset(
        **hashes,
        split_id=runner.SPLIT_ID,
        tap_authority_sha256=authority,
        source_row_count=runner.SOURCE_ROW_COUNT,
        source_class_count=runner.SOURCE_CLASS_COUNT,
        basis_codes_qint8=basis_codes,
        basis_scales_fp16=basis_scales,
        tau_codes_qint8=tau_codes,
        tau_scales_fp16=tau_scales,
        spectrum_codes_qint8=spectrum_codes,
        spectrum_scales_fp16=spectrum_scales,
        asset_receipt_sha256="0" * 64,
    )
    payload = runner._asset_payload(asset)
    asset_receipt = hashlib.sha256(runner._canonical_bytes(payload)).hexdigest()
    asset = replace(asset, asset_receipt_sha256=asset_receipt)
    header = runner._canonical_bytes(
        {
            "schema": runner.WIRE_SCHEMA,
            "asset": runner._asset_payload(asset),
            "asset_receipt_sha256": asset_receipt,
        }
    )
    wire = runner.WIRE_MAGIC + struct.pack(">I", len(header)) + header + b"".join(
        np.ascontiguousarray(value).tobytes(order="C")
        for value in (
            basis_codes,
            basis_scales,
            tau_codes,
            tau_scales,
            spectrum_codes,
            spectrum_scales,
        )
    )
    path = tmp_path / "rdce.asset.wire"
    path.write_bytes(wire)
    return asset, path, wire


def test_import_does_not_load_legacy_d106_modules() -> None:
    assert "scripts.run_d106_g1_sourceheld_one_shot" not in sys.modules
    assert not any(name.startswith("cvsrffi.stage2_d106_") for name in sys.modules)


def test_wire_parser_reconstructs_canonical_payload_and_receipts(tmp_path) -> None:
    expected, path, wire = _valid_asset_and_wire(tmp_path)
    parsed = runner._parse_asset_wire(path, hashlib.sha256(wire).hexdigest())
    assert parsed.split_id == runner.SPLIT_ID
    assert parsed.asset_receipt_sha256 == expected.asset_receipt_sha256
    assert np.array_equal(parsed.basis_codes_qint8, expected.basis_codes_qint8)
    assert np.array_equal(parsed.tau_codes_qint8, expected.tau_codes_qint8)

    path.write_bytes(wire + b"trailing")
    with pytest.raises(runner.D122G1Error, match="trailing"):
        runner._parse_asset_wire(path, hashlib.sha256(path.read_bytes()).hexdigest())


def test_wire_math_guards_raw_gram_and_positive_decode() -> None:
    codes = np.zeros((runner.RDCE_RANK, runner.Z_DIM), dtype=np.int8)
    codes[:, 0] = 127
    scales = np.full(runner.RDCE_RANK, np.float16(1.0 / 127.0), dtype="<f2")
    with pytest.raises(runner.D122G1Error, match="raw Gram"):
        runner._orthogonal_closure(codes, scales)
    with pytest.raises(runner.D122G1Error, match="positive"):
        runner._decode_positive(np.asarray([0, 1, 2], dtype=np.int8), scales, "tau")


def test_rdce_fit_is_support_only_and_numeric_lock_is_frozen(tmp_path) -> None:
    asset, _path, _wire = _valid_asset_and_wire(tmp_path)
    support = np.eye(6, runner.Z_DIM, dtype=np.float32)
    labels = tuple(f"c{index}" for index in range(6))
    state = runner.fit_rdce_sourceheld_state(asset, support, labels, 1)
    assert np.array_equal(state["attenuation"], np.full(3, np.float16(0.3), dtype=np.float64))
    assert state["payload"]["query_rows_used_for_fit"] == 0
    assert state["payload"]["query_state_updates"] == 0
    lock = runner._lock(1, "b" * 64)
    assert lock.student_nu == 3.0
    assert lock.kernel_effective_dim == 12
    assert lock.shared_h0 == 0.35
    assert lock.temperature == 0.85
    transformed = runner.apply_rdce_state(state, support)
    assert transformed.shape == support.shape
    assert np.isfinite(transformed).all()


def test_fixed_cli_predict_surface_has_no_truth_inputs() -> None:
    args = runner.parse_args(
        [
            "predict",
            "--package-root", "packages",
            "--rdce-asset-wire", "rdce.wire",
            "--rdce-wire-sha256", "a" * 64,
            "--d106-tap-archive", "tap.npz",
            "--d106-tap-receipt", "tap.json",
            "--d106-tap-archive-sha256", "b" * 64,
            "--checkpoint-sha256", "c" * 64,
            "--run-id", "d122-standalone-test",
            "--output-dir", "out",
        ]
    )
    assert args.command == "predict"
    assert "truth_json" not in vars(args)
    assert "truth_input_seal_json" not in vars(args)


def test_canonical_truth_sha_uses_utf8_non_ascii_contract() -> None:
    value = [{"label": "类"}]
    expected = hashlib.sha256(
        json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
    ).hexdigest()
    assert runner._canonical_sha256(value) == expected


def _write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, ensure_ascii=True, sort_keys=True), encoding="utf-8")


def _fake_bundle(*, checkpoint_sha256: str = "c" * 64, phase1_seal_sha256: str = "3" * 64):
    ground = np.zeros((6, FEATURE_DIM), dtype=np.float64)
    for index in range(6):
        ground[index, 10 + index] = 1.0
    q0 = np.sum(ground, axis=0)
    q0 /= np.linalg.norm(q0)
    U = np.zeros((3, FEATURE_DIM), dtype=np.float64)
    U[np.arange(3), np.arange(3)] = 1.0
    return build_d112_source_held_g1_bundle(
        class_registry=tuple(f"tx-{index}" for index in range(6)),
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


def _make_package_root(root: Path) -> tuple[Path, Path, Path]:
    receivers = tuple(f"rx-{index}" for index in range(7))
    classes = tuple(f"tx-{index}" for index in range(6))
    root.mkdir()
    package_rows = []
    truth_rows = []
    for receiver in receivers:
        for k_shot in runner.K_VALUES:
            support = []
            labels = []
            for class_index, class_id in enumerate(classes):
                for sample_index in range(k_shot):
                    row = np.zeros(FEATURE_DIM, dtype=np.float32)
                    row[10 + class_index] = 1.0
                    row[0] = 0.01 * (sample_index + 1)
                    row /= np.linalg.norm(row)
                    support.append(row)
                    labels.append(class_id)
            support = np.asarray(support, dtype=np.float32)
            query = support[np.arange(0, len(support), k_shot)].copy()
            package_id = f"{receiver}-k{k_shot}"
            query_ids = [f"{package_id}-q{index}" for index in range(len(query))]
            support_ids = [f"{package_id}-s{index}" for index in range(len(support))]
            package_path = root / f"{package_id}.npz"
            np.savez(
                package_path,
                support_pre_relu=support,
                support_zdom=np.zeros_like(support),
                support_labels=np.asarray(labels),
                support_physical_ids=np.asarray(support_ids),
                query_pre_relu=query,
                query_physical_ids=np.asarray(query_ids),
                registered_classes=np.asarray(classes),
            )
            package_rows.append(
                {
                    "held_receiver": receiver,
                    "K": k_shot,
                    "package_id": package_id,
                    "path": package_path.name,
                    "sha256": runner._file_sha(package_path),
                }
            )
            truth_rows.append(
                {
                    "package_id": package_id,
                    "query_physical_ids": query_ids,
                    "query_truth_labels": list(classes),
                }
            )
    truth_dir = root / "truth"
    truth_dir.mkdir()
    truth = {
        "schema": "cvs.d104_r1.rxid_angq.held_truth.v2",
        "split_id": runner.SPLIT_ID,
        "package_count": 21,
        "predictor_access": False,
        "packages": truth_rows,
    }
    truth_path = truth_dir / "truth.json"
    _write_json(truth_path, truth)
    seal = {
        "split_id": runner.SPLIT_ID,
        "package_count": 21,
        "predictor_truth_access": False,
        "package_ids": [row["package_id"] for row in package_rows],
        "truth_package_root_sha256": runner._canonical_sha256(truth_rows),
    }
    seal_path = truth_dir / "truth_input_seal.json"
    _write_json(seal_path, seal)
    _write_json(
        root / "package_manifest.json",
        {
            "schema": runner.PACKAGE_SCHEMA,
            "candidate_id": runner.D104_CANDIDATE_ID,
            "split_id": runner.SPLIT_ID,
            "receiver_ids": list(receivers),
            "class_ids": list(classes),
            "query_truth_present": False,
            "target_access": False,
            "truth_input_seal_sha256": runner._file_sha(seal_path),
            "packages": package_rows,
        },
    )
    return root, truth_path, seal_path


def test_predict_then_separate_score_closes_full_matrix(tmp_path, monkeypatch) -> None:
    package_root, truth_path, seal_path = _make_package_root(tmp_path / "packages")
    wire = tmp_path / "placeholder.wire"
    wire.write_bytes(b"placeholder")
    tap_sha = runner._file_sha(wire)
    fake_asset = SimpleNamespace(
        split_id=runner.SPLIT_ID,
        checkpoint_sha256="c" * 64,
        tap_sha256=tap_sha,
        tap_receipt_sha256=tap_sha,
        asset_receipt_sha256="9" * 64,
    )

    def fake_fit(_asset, support, _labels, k_shot):
        basis = np.zeros((3, FEATURE_DIM), dtype=np.float64)
        basis[np.arange(3), np.arange(3)] = 1.0
        return {
            "basis": basis,
            "attenuation": np.full(3, np.float16(0.3), dtype=np.float64),
            "receipt": "8" * 64,
            "payload": {
                "query_rows_used_for_fit": 0,
                "query_state_updates": 0,
            },
        }

    def fake_apply(state, rows):
        return d122._d106_like_transform(
            np.asarray(rows, dtype=np.float32),
            np.asarray(state["basis"], dtype=np.float64),
            np.asarray(state["attenuation"], dtype=np.float64),
        )[0]

    monkeypatch.setattr(runner, "_parse_asset_wire", lambda *_args: fake_asset)
    monkeypatch.setattr(runner, "fit_rdce_sourceheld_state", fake_fit)
    monkeypatch.setattr(runner, "apply_rdce_state", fake_apply)
    monkeypatch.setattr(
        runner,
        "_build_d112_g1_bundle",
        lambda *_args: _fake_bundle(phase1_seal_sha256=tap_sha),
    )
    monkeypatch.setattr(
        runner,
        "_build_four_arm_predictions",
        lambda **kwargs: (
            {arm: [kwargs["registry"][0]] * len(kwargs["query_signed"]) for arm in runner.ARMS},
            {
                "student_t_lock_sha256": "1" * 64,
                "M_DA_M_JOINT_rdce_state_sha256": "2" * 64,
                "M_HEAD_state_audit": {"query_rows_used_for_fit": 0},
                "M_JOINT_state_audit": {"query_rows_used_for_fit": 0},
                "arm_logits": {},
                "new_class_logit_boundary_bit_exact": True,
            },
        ),
    )
    prediction_root = tmp_path / "prediction"
    assert runner.predict(
        Namespace(
            package_root=package_root,
            rdce_asset_wire=wire,
            rdce_wire_sha256="a" * 64,
            d106_tap_archive=wire,
            d106_tap_receipt=wire,
            d106_tap_archive_sha256=tap_sha,
            checkpoint_sha256="c" * 64,
            run_id="d122-standalone-full",
            output_dir=prediction_root,
        )
    ) == 0
    prediction = json.loads((prediction_root / "prediction_manifest.json").read_text(encoding="utf-8"))
    assert prediction["row_count"] == 63
    assert prediction["arm_row_prediction_unit_count"] == 252
    score_path = tmp_path / "scores.json"
    event_path = tmp_path / "truth-open.json"
    assert runner.score(
        Namespace(
            prediction_root=prediction_root,
            truth_json=truth_path,
            truth_input_seal_json=seal_path,
            truth_open_event_json=event_path,
            output_json=score_path,
        )
    ) == 0
    scores = json.loads(score_path.read_text(encoding="utf-8"))
    assert len(scores["performance_rows"]) == 63
    assert event_path.exists()
