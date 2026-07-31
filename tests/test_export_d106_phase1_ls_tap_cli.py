from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest


SCRIPT_PATH = (
    Path(__file__).resolve().parents[1]
    / "code"
    / "scripts"
    / "export_d106_phase1_ls_tap.py"
)
SPEC = importlib.util.spec_from_file_location("export_d106_phase1_ls_tap_cli", SCRIPT_PATH)
assert SPEC is not None and SPEC.loader is not None
SCRIPT = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(SCRIPT)


def _actions(parser, command: str) -> set[str]:
    subparsers = next(
        action for action in parser._actions if action.dest == "command"
    )
    return {action.dest for action in subparsers.choices[command]._actions}


def test_parser_exposes_only_bounded_d106_workflows() -> None:
    parser = SCRIPT.build_parser()
    subparsers = next(
        action for action in parser._actions if action.dest == "command"
    )
    assert set(subparsers.choices) == {
        "build-disjoint-receipt",
        "extract",
        "export",
        "validate",
    }
    assert _actions(parser, "build-disjoint-receipt") >= {
        "source_split_manifest",
        "source_split_manifest_sha256",
        "output",
    }
    assert _actions(parser, "export") >= {
        "selected_iq_archive",
        "selected_iq_archive_sha256",
        "selected_iq_receipt",
        "selected_iq_receipt_sha256",
        "storage_validator_receipt",
        "storage_validator_receipt_sha256",
        "ls_archive",
        "ls_archive_sha256",
        "checkpoint",
        "checkpoint_sha256",
        "runtime_manifest",
        "runtime_sha256",
        "output_dir",
        "device",
    }
    assert _actions(parser, "validate") >= {
        "archive",
        "archive_sha256",
        "receipt",
        "receipt_sha256",
    }
    assert "target" not in " ".join(_actions(parser, "export")).lower()
    assert "source_val_labels" not in _actions(parser, "export")
    assert "upstream_source_pool_cache_set" not in _actions(parser, "export")
    assert _actions(parser, "extract") >= {
        "source_split_manifest",
        "source_split_manifest_sha256",
        "disjoint_receipt",
        "disjoint_receipt_sha256",
        "upstream_source_pool_cache_set",
        "selection_salt_receipt",
        "output_dir",
    }


def test_disjoint_command_forwards_exact_manifest_binding(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    captured = {}

    def fake(**kwargs):
        captured.update(kwargs)
        return {"status": "ok", "receipt_sha256": "d" * 64}

    monkeypatch.setattr(SCRIPT, "build_d106_train_held_disjoint_receipt", fake)
    assert SCRIPT.main(
        [
            "build-disjoint-receipt",
            "--source-split-manifest",
            "split.json",
            "--source-split-manifest-sha256",
            "a" * 64,
            "--output",
            "disjoint.json",
        ]
    ) == 0
    assert captured == {
        "source_split_manifest": Path("split.json"),
        "source_split_manifest_sha256": "a" * 64,
        "output_path": Path("disjoint.json"),
    }
    assert json.loads(capsys.readouterr().out)["receipt_sha256"] == "d" * 64


def test_extract_command_is_the_only_8400_cache_consumer(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    captured = {}

    def fake(**kwargs):
        captured.update(kwargs)
        return {"status": "extracted", "row_count": 588}

    monkeypatch.setattr(SCRIPT, "extract_d106_ls_received_iq", fake)
    assert SCRIPT.main(
        [
            "extract",
            "--source-split-manifest", "split.json",
            "--source-split-manifest-sha256", "a" * 64,
            "--disjoint-receipt", "disjoint.json",
            "--disjoint-receipt-sha256", "b" * 64,
            "--upstream-source-pool-cache-set", "cache.json",
            "--selection-salt-receipt", "salt.json",
            "--output-dir", "selected",
        ]
    ) == 0
    assert captured == {
        "source_split_manifest": Path("split.json"),
        "source_split_manifest_sha256": "a" * 64,
        "disjoint_receipt": Path("disjoint.json"),
        "disjoint_receipt_sha256": "b" * 64,
        "upstream_source_pool_cache_set": Path("cache.json"),
        "selection_salt_receipt": Path("salt.json"),
        "output_dir": Path("selected"),
    }
    assert json.loads(capsys.readouterr().out)["row_count"] == 588


def test_export_command_forwards_every_frozen_input_without_tuning_knobs(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    captured = {}

    def fake(**kwargs):
        captured.update(kwargs)
        return {"status": "complete", "row_count": 588}

    monkeypatch.setattr(SCRIPT, "export_d106_phase1_ls_tap", fake)
    assert SCRIPT.main(
        [
            "export",
            "--selected-iq-archive", "selected.npz",
            "--selected-iq-archive-sha256", "a" * 64,
            "--selected-iq-receipt", "selected.json",
            "--selected-iq-receipt-sha256", "b" * 64,
            "--storage-validator-receipt", "validator.json",
            "--storage-validator-receipt-sha256", "f" * 64,
            "--ls-archive", "ls.npz",
            "--ls-archive-sha256", "c" * 64,
            "--checkpoint", "model.pth",
            "--checkpoint-sha256", "d" * 64,
            "--runtime-manifest", "runtime.json",
            "--runtime-sha256", "e" * 64,
            "--output-dir", "tap",
            "--device", "cuda:1",
        ]
    ) == 0
    assert captured == {
        "selected_iq_archive": Path("selected.npz"),
        "selected_iq_archive_sha256": "a" * 64,
        "selected_iq_receipt": Path("selected.json"),
        "selected_iq_receipt_sha256": "b" * 64,
        "storage_validator_receipt": Path("validator.json"),
        "storage_validator_receipt_sha256": "f" * 64,
        "ls_archive": Path("ls.npz"),
        "ls_archive_sha256": "c" * 64,
        "checkpoint": Path("model.pth"),
        "checkpoint_sha256": "d" * 64,
        "runtime_manifest": Path("runtime.json"),
        "runtime_sha256": "e" * 64,
        "output_dir": Path("tap"),
        "device": "cuda:1",
    }
    assert json.loads(capsys.readouterr().out) == {
        "row_count": 588,
        "status": "complete",
    }


def test_validate_command_reports_derived_zid_without_dumping_arrays(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    class Loaded:
        z_id = np.zeros((588, 160), dtype=np.float32)

    captured = {}

    def fake(archive, receipt, **kwargs):
        captured.update({"archive": archive, "receipt": receipt, **kwargs})
        return Loaded()

    monkeypatch.setattr(SCRIPT, "load_d106_phase1_ls_tap", fake)
    assert SCRIPT.main(
        [
            "validate",
            "--archive", "tap.npz",
            "--archive-sha256", "c" * 64,
            "--receipt", "tap.json",
            "--receipt-sha256", "d" * 64,
        ]
    ) == 0
    assert captured == {
        "archive": Path("tap.npz"),
        "receipt": Path("tap.json"),
        "expected_archive_sha256": "c" * 64,
        "expected_receipt_sha256": "d" * 64,
    }
    assert json.loads(capsys.readouterr().out) == {
        "received_iq_persisted": False,
        "row_count": 588,
        "status": "D106_LS_STRICT_TAP_VALID",
        "z_id_derived_from_pre_relu": True,
    }


@pytest.mark.parametrize(
    "command", ["build-disjoint-receipt", "extract", "export", "validate"]
)
def test_each_command_fails_closed_when_required_arguments_are_missing(
    command: str,
) -> None:
    with pytest.raises(SystemExit) as error:
        SCRIPT.build_parser().parse_args([command])
    assert error.value.code == 2
