from __future__ import annotations

import hashlib
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest


SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "code"
    / "scripts"
    / "run_d103_r1_phase1_fit.py"
)


def test_fit_cli_help() -> None:
    result = subprocess.run(
        [sys.executable, str(SCRIPT), "--help"],
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0
    assert "--labeled-archive" in result.stdout
    assert "--source-val-seal" in result.stdout
    assert "--held-class" in result.stdout


def test_source_manifest_requires_exact_role_and_archive_hash(tmp_path) -> None:
    sys.path.insert(0, str(SCRIPT.parent))
    import run_d103_r1_phase1_fit as fit_cli

    archive = tmp_path / "archive.npz"
    archive.write_bytes(b"archive")
    digest = hashlib.sha256(b"archive").hexdigest()
    manifest = {
        "schema": fit_cli.ARCHIVE_MANIFEST_SCHEMA,
        "candidate_id": fit_cli.CANDIDATE_ID,
        "role": "L_s",
        "fraction": 0.07,
        "tx_visibility": "visible",
        "archive_sha256": digest,
        "target_access": False,
        "formal_query_access": False,
        "source_validation_gradient_access": False,
        "physical_id_unique": True,
        "checkpoint_sha256": "a" * 64,
        "runtime_sha256": "b" * 64,
    }
    fit_cli._validate_source_manifest(
        manifest,
        role="L_s",
        fraction=0.07,
        tx_visibility="visible",
        archive_path=archive,
    )
    bad = dict(manifest)
    bad["role"] = "source_val"
    with pytest.raises(ValueError, match="semantic drift"):
        fit_cli._validate_source_manifest(
            bad,
            role="L_s",
            fraction=0.07,
            tx_visibility="visible",
            archive_path=archive,
        )


def test_failed_input_preserves_normalized_failure_receipt(tmp_path) -> None:
    output = tmp_path / "failed-fit"
    result = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--labeled-archive",
            str(tmp_path / "missing-labeled.npz"),
            "--labeled-manifest",
            str(tmp_path / "missing-labeled.json"),
            "--unlabeled-archive",
            str(tmp_path / "missing-unlabeled.npz"),
            "--unlabeled-manifest",
            str(tmp_path / "missing-unlabeled.json"),
            "--source-val-seal",
            str(tmp_path / "missing-seal.json"),
            "--source-val-manifest",
            str(tmp_path / "missing-source-val.json"),
            "--output-dir",
            str(output),
            "--device",
            "cpu",
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode != 0
    receipt = json.loads((output / "fit_failed.json").read_text(encoding="utf-8"))
    assert receipt["status"] == "PHASE1_FIT_FAILED_NO_PERFORMANCE_RESULT"
    assert receipt["performance_result"] is False
    assert len(receipt["normalized_exception_fingerprint"]) == 64


def test_exception_fingerprint_redacts_row_path_receiver_and_pid(tmp_path) -> None:
    sys.path.insert(0, str(SCRIPT.parent))
    import run_d103_r1_phase1_fit as fit_cli

    first_args = SimpleNamespace(
        held_receiver="1-1",
        held_class="tx-a",
        held_day="d1",
        output_dir=tmp_path / "row-a",
        labeled_archive=tmp_path / "l-a.npz",
        unlabeled_archive=tmp_path / "u-a.npz",
        source_val_seal=tmp_path / "s-a.json",
    )
    second_args = SimpleNamespace(
        held_receiver="14-7",
        held_class="tx-b",
        held_day="d2",
        output_dir=tmp_path / "row-b",
        labeled_archive=tmp_path / "l-b.npz",
        unlabeled_archive=tmp_path / "u-b.npz",
        source_val_seal=tmp_path / "s-b.json",
    )
    first = ValueError(
        f"systemic fault receiver=1-1 class=tx-a pid=123 "
        f"path={first_args.output_dir}"
    )
    second = ValueError(
        f"systemic fault receiver=14-7 class=tx-b pid=987 "
        f"path={second_args.output_dir}"
    )
    first_template, first_sha = fit_cli._normalized_exception(first, first_args)
    second_template, second_sha = fit_cli._normalized_exception(second, second_args)
    assert first_template == second_template
    assert first_sha == second_sha
