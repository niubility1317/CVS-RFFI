from __future__ import annotations

import json
from pathlib import Path
import runpy

import pytest


SCRIPT = Path(__file__).resolve().parents[1] / "code" / "scripts" / "run_d127_s0.py"
HASH = "a" * 64
CHECKPOINT_HASH = "b" * 64


@pytest.fixture
def runner():
    return runpy.run_path(str(SCRIPT), run_name="d127_s0_cli_test")


def test_cli_rejects_forbidden_inputs_at_argument_boundary(runner):
    with pytest.raises(SystemExit):
        runner["parse_args"]([
            "prepare",
            "--method-lock", "lock.json",
            "--method-lock-sha256", HASH,
            "--d106-context", "context.json",
            "--d106-context-sha256", HASH,
            "--output-dir", "prepared",
            "--truth", "forbidden",
        ])


def test_candidate_worker_cli_dispatches_only_frozen_inputs(tmp_path, monkeypatch, capsys, runner):
    adapter = runner["adapter"]
    hooks = runner["checkpoint_hooks"]
    plan = {
        "method_lock_sha256": HASH,
        "checkpoint_sha256": CHECKPOINT_HASH,
    }
    payload = {
        "row_pair_count": 18,
        "state_row_count": 36,
    }
    calls = {}
    runtime = runner["main"].__globals__

    monkeypatch.setattr(adapter, "load_d127_s0_prepared_plan", lambda path, expected_sha256: (plan, expected_sha256))
    monkeypatch.setitem(runtime, "_check_method_lock_against_plan", lambda args, loaded_plan: {})
    monkeypatch.setitem(runtime, "_materialize", lambda args: "prepared-input")
    monkeypatch.setattr(hooks, "load_d127_frozen_checkpoint", lambda path, device: ("model", {"checkpoint_sha256": CHECKPOINT_HASH}))
    manifest_receipt = {"receipt": "verified"}
    monkeypatch.setattr(adapter, "load_d127_s0_candidate_asset", lambda **kwargs: ("asset", manifest_receipt))

    def fake_run(**kwargs):
        calls.update(kwargs)
        return payload

    def fake_write(path, value):
        target = Path(path)
        target.write_bytes(b"candidate-worker")
        return target

    monkeypatch.setattr(adapter, "run_d127_s0_candidate_worker_pair", fake_run)
    monkeypatch.setattr(adapter, "write_d127_s0_candidate_worker_exclusive", fake_write)
    output = tmp_path / "candidate.json"
    assert runner["main"]([
        "candidate-worker",
        "--prepared-plan", "prepared.json",
        "--prepared-plan-sha256", HASH,
        "--method-lock", "lock.json",
        "--method-lock-sha256", HASH,
        "--d106-context", "context.json",
        "--d106-context-sha256", HASH,
        "--phase1-asset-bundle", "bundle",
        "--phase1-asset-manifest-sha256", HASH,
        "--checkpoint", "frozen.pth",
        "--candidate-id", runner["entry"].CANDIDATE_IDS[0],
        "--device", "cuda:1",
        "--output", str(output),
    ]) == 0
    assert calls["candidate_id"] == runner["entry"].CANDIDATE_IDS[0]
    assert calls["prepared"] == "prepared-input"
    assert calls["checkpoint_sha256"] == CHECKPOINT_HASH
    assert calls["phase1_manifest_receipt"] is manifest_receipt
    event = json.loads(capsys.readouterr().out)
    assert event["truth_loaded"] is False
    assert event["physical_base_forwards_are_repeated_per_candidate"] is True


def test_merge_cli_requires_three_bound_worker_files(tmp_path, monkeypatch, capsys, runner):
    adapter = runner["adapter"]
    plan = {"method_lock_sha256": HASH, "checkpoint_sha256": CHECKPOINT_HASH}
    runtime = runner["main"].__globals__
    monkeypatch.setattr(adapter, "load_d127_s0_prepared_plan", lambda path, expected_sha256: (plan, expected_sha256))
    monkeypatch.setitem(runtime, "_check_method_lock_against_plan", lambda args, loaded_plan: {})
    loaded = []

    def fake_load(path, expected_sha256):
        loaded.append((Path(path).name, expected_sha256))
        return {"candidate_id": Path(path).stem}, expected_sha256

    paired = {
        "candidate_ids": ["A", "B", "C"],
        "row_pair_count": 18,
        "state_row_count": 36,
        "pair_manifest": {"pair_manifest_sha256": HASH},
    }

    def fake_write(path, value, *, prepared_plan):
        target = Path(path)
        target.write_bytes(b"paired")
        return target

    monkeypatch.setattr(adapter, "load_d127_s0_candidate_worker", fake_load)
    monkeypatch.setattr(adapter, "merge_d127_s0_candidate_workers", lambda *, prepared_plan, workers: paired)
    monkeypatch.setattr(adapter, "write_d127_s0_paired_prediction_exclusive", fake_write)
    output = tmp_path / "paired.json"
    argv = [
        "merge",
        "--prepared-plan", "prepared.json",
        "--prepared-plan-sha256", HASH,
        "--method-lock", "lock.json",
        "--method-lock-sha256", HASH,
    ]
    for name in ("a.json", "b.json", "c.json"):
        argv.extend(["--worker-prediction", name, "--worker-prediction-sha256", HASH])
    argv.extend(["--output", str(output)])
    assert runner["main"](argv) == 0
    assert loaded == [("a.json", HASH), ("b.json", HASH), ("c.json", HASH)]
    event = json.loads(capsys.readouterr().out)
    assert event["truth_loaded"] is False
    assert event["pair_manifest_sha256"] == HASH
