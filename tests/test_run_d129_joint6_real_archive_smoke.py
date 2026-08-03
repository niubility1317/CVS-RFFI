from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "code" / "scripts" / "run_d129_joint6_real_archive_smoke.py"
SPEC = importlib.util.spec_from_file_location("d129_real_archive_smoke", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
smoke = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(smoke)


def _inputs(tmp_path: Path):
    rng = np.random.default_rng(12963)
    receivers = [f"rx{index}" for index in range(7)]
    classes = [f"tx{index}" for index in range(6)]
    receiver_axes = rng.normal(size=(7, 160)).astype(np.float32) * np.float32(0.18)
    class_axes = rng.normal(size=(6, 160)).astype(np.float32) * np.float32(0.30)
    pre_relu = []
    z_dom = []
    tx_labels = []
    receiver_ids = []
    day_ids = []
    physical_ids = []
    for receiver_index, receiver in enumerate(receivers):
        for class_index, class_id in enumerate(classes):
            for row in range(14):
                feature = (
                    np.float32(0.45)
                    + receiver_axes[receiver_index]
                    + class_axes[class_index]
                    + np.float32(0.07)
                    * rng.normal(size=160).astype(np.float32)
                )
                pre_relu.append(feature)
                z_dom.append(rng.normal(size=160).astype(np.float32))
                tx_labels.append(class_id)
                receiver_ids.append(receiver)
                day_ids.append("day0")
                physical_ids.append(f"physical-{receiver}-{class_id}-{row}")
    archive = tmp_path / "features.npz"
    np.savez(
        archive,
        z_dom=np.asarray(z_dom, dtype=np.float32),
        pre_relu=np.asarray(pre_relu, dtype=np.float32),
        receiver_ids=np.asarray(receiver_ids),
        day_ids=np.asarray(day_ids),
        tx_labels=np.asarray(tx_labels),
        physical_ids=np.asarray(physical_ids),
    )
    archive_sha = hashlib.sha256(archive.read_bytes()).hexdigest()
    checkpoint_sha = "a" * 64
    fixture = tmp_path / "fixture.json"
    fixture.write_text(
        json.dumps(
            {
                "schema": "cvs.d106.real_integration_fixture.v1",
                "protocol_schema": "p2_min_v1",
                "ls_archive_sha256": archive_sha,
                "checkpoint_sha256": checkpoint_sha,
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    fixture_sha = hashlib.sha256(fixture.read_bytes()).hexdigest()
    return archive, archive_sha, fixture, fixture_sha, checkpoint_sha


def test_real_archive_smoke_executes_both_candidates_without_truth(tmp_path: Path) -> None:
    archive, archive_sha, fixture, fixture_sha, checkpoint_sha = _inputs(tmp_path)
    result = smoke.run_real_archive_smoke(
        archive_path=archive.resolve(),
        archive_sha256=archive_sha,
        fixture_path=fixture.resolve(),
        fixture_sha256=fixture_sha,
        checkpoint_sha256=checkpoint_sha,
        held_receiver="rx0",
        held_class="tx0",
        run_id="d129-test-smoke",
    )
    assert result["status"] == smoke.STATUS
    assert result["truth_loaded"] is False
    assert result["performance_result"] is False
    assert result["formal_new_registration_claim"] is False
    assert result["phase1_fit_count"] == 420
    assert result["support_count"] == 30
    assert result["query_count"] == 54
    assert result["common_r0_candidate_refit_count"] == 0
    assert set(result["candidate_receipts"]) == {"CSPAR-2", "SRDH-2"}
    assert result["passed_candidate_ids"]
    assert set(result["passed_candidate_ids"]) | set(
        result["rejected_no_function_candidate_ids"]
    ) == {"CSPAR-2", "SRDH-2"}


def test_fixture_archive_checkpoint_mismatch_fails_closed(tmp_path: Path) -> None:
    archive, archive_sha, fixture, fixture_sha, _checkpoint_sha = _inputs(tmp_path)
    with pytest.raises(smoke.D129RealArchiveSmokeError, match="provenance drift"):
        smoke.run_real_archive_smoke(
            archive_path=archive.resolve(),
            archive_sha256=archive_sha,
            fixture_path=fixture.resolve(),
            fixture_sha256=fixture_sha,
            checkpoint_sha256="b" * 64,
            held_receiver="rx0",
            held_class="tx0",
            run_id="d129-test-smoke",
        )
