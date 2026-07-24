from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

from cvsrffi.phase1_grb_jp4_cfm_bundle import canonical_array_sha256


SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "code"
    / "scripts"
    / "run_grb_jp4_cfm_phase1_held54.py"
)
SPEC = importlib.util.spec_from_file_location("run_grb_held54", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
runner = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(runner)


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    digest.update(path.read_bytes())
    return digest.hexdigest()


def _input_files(tmp_path: Path):
    count = 2
    arrays = {
        "z_id": np.ones((count, 160), dtype=np.float32),
        "hidden": np.ones((count, 320), dtype=np.float32),
        "pre_relu": np.ones((count, 160), dtype=np.float32),
        "joint_weight": np.ones((160, 320), dtype=np.float32),
        "labels": np.asarray(["a", "b"]),
        "receiver_ids": np.asarray(["r0", "r1"]),
        "day_ids": np.asarray(["d0", "d1"]),
        "physical_ids": np.asarray(["p0", "p1"]),
        "scenario_names": np.asarray(
            ["leo_clear_weak", "leo_clear_weak"]
        ),
        "class_ids": np.asarray(["a", "b"]),
        "observation_ids": np.asarray(["o0", "o1"]),
    }
    archive = tmp_path / "phase1_jp4_tap_archive.npz"
    with archive.open("xb") as handle:
        np.savez_compressed(handle, **arrays)
    checkpoint_sha = hashlib.sha256(b"checkpoint").hexdigest()
    manifest = {
        "schema": runner.TAP_SCHEMA,
        "status": "DEVELOPMENT_ONLY_NOT_FORMAL",
        "artifact_stage": "phase1_offline_before_target_access",
        "formal_phase2_eligible": False,
        "bundle_created": False,
        "target25_release_authorized": False,
        "exact_member_allowlist": list(runner.TAP_MEMBERS),
        "array_sha256": {
            name: canonical_array_sha256(value)
            for name, value in arrays.items()
        },
        "artifact": {"path": archive.name, "sha256": _sha(archive)},
        "row_count": count,
        "inputs": {"checkpoint_sha256": checkpoint_sha},
        "runtime_audit": {},
        "access_audit": {
            "source_validation_weak_iq_access": True,
            "clean_iq_access": False,
            "target_access": False,
            "query_access": False,
            "received_iq_persisted": False,
            "raw_iq_persisted": False,
        },
        "selection": {"selected_observations_per_physical_id": 1},
    }
    manifest_path = tmp_path / "phase1_jp4_tap_archive.manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, sort_keys=True), encoding="utf-8"
    )
    coverage = {
        "schema": runner.COVERAGE_SCHEMA,
        "artifact_stage": "phase1_offline_before_target_access",
        "target_access": False,
        "query_access": False,
        "held_fold_selected": False,
        "pre_registered_coverage_gate_passed": True,
    }
    coverage_path = tmp_path / "coverage.json"
    coverage_path.write_text(
        json.dumps(coverage, sort_keys=True), encoding="utf-8"
    )
    return (
        arrays,
        archive,
        manifest_path,
        checkpoint_sha,
        coverage_path,
        _sha(coverage_path),
    )


def test_phase1_input_loader_binds_legal_source_only_artifacts(tmp_path: Path):
    arrays, archive, manifest, checkpoint, coverage, coverage_sha = _input_files(
        tmp_path
    )
    loaded, binding = runner._load_phase1_inputs(
        archive_path=archive,
        manifest_path=manifest,
        checkpoint_sha256=checkpoint,
        coverage_receipt_path=coverage,
        coverage_receipt_sha256=coverage_sha,
    )
    assert set(loaded) == set(arrays)
    assert binding == {
        "archive_schema": runner.TAP_SCHEMA,
        "archive_sha256": _sha(archive),
        "manifest_sha256": _sha(manifest),
        "checkpoint_sha256": checkpoint,
        "coverage_sha256": coverage_sha,
    }


def test_phase1_input_loader_rejects_clean_or_target_access(tmp_path: Path):
    _arrays, archive, manifest, checkpoint, coverage, _coverage_sha = _input_files(
        tmp_path
    )
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    payload["access_audit"]["clean_iq_access"] = True
    manifest.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")
    coverage_sha = _sha(coverage)
    with pytest.raises(runner.Held54RunnerError, match="legality"):
        runner._load_phase1_inputs(
            archive_path=archive,
            manifest_path=manifest,
            checkpoint_sha256=checkpoint,
            coverage_receipt_path=coverage,
            coverage_receipt_sha256=coverage_sha,
        )


def test_predict_stage_publishes_rows_and_refuses_overwrite(
    tmp_path: Path, monkeypatch
):
    packet = {"packet_sha256": "a" * 64}
    query = {"query_binding_sha256": "b" * 64}
    prediction = {
        "COMMIT": "c" * 64,
        "rows": [
            {
                "row_id": "r0",
                "query_ids": ["q0"],
                "before": {"M0": {}},
                "after": {"M0": {}},
                "counterfactuals": {"ground_off": {}},
            }
        ],
    }
    monkeypatch.setattr(
        runner.held, "load_prediction_inputs", lambda _path: (packet, query)
    )
    monkeypatch.setattr(
        runner.held, "predict_packet", lambda _packet, _query: prediction
    )

    def write_prediction(path, _value):
        data = b"prediction"
        with Path(path).open("xb") as handle:
            handle.write(data)
        return hashlib.sha256(data).hexdigest()

    monkeypatch.setattr(
        runner.held,
        "write_prediction_artifact",
        write_prediction,
    )
    output = tmp_path / "prediction.json"
    rows = tmp_path / "predict.rows.jsonl"
    args = argparse.Namespace(
        build_dir=tmp_path / "build",
        output=output,
        row_receipt=rows,
    )
    result = runner.predict_stage(args)
    assert result["truth_parsed"] is False
    assert result["rows"] == 1
    assert len(rows.read_text(encoding="ascii").splitlines()) == 1
    with pytest.raises(FileExistsError):
        runner.predict_stage(args)
