from __future__ import annotations

import importlib.util
from pathlib import Path

from cvsrffi import stage2_d127_da_candidates as da


def _script_module() -> object:
    path = Path(__file__).resolve().parents[1] / "code" / "scripts" / "build_d127_phase1_assets.py"
    spec = importlib.util.spec_from_file_location("test_build_d127_phase1_assets_script", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_cli_routes_only_one_frozen_candidate_and_all_bound_inputs(monkeypatch: object, capsys: object) -> None:
    module = _script_module()
    captured: dict[str, object] = {}

    def fake_build(**kwargs: object) -> dict[str, object]:
        captured.update(kwargs)
        return {"status": "D127_PHASE1_SINGLE_CANDIDATE_BUNDLE_WRITTEN", "candidate_id": kwargs["candidate_id"]}

    monkeypatch.setattr(module, "build_d127_phase1_single_candidate_from_source", fake_build)
    result = module.main(
        [
            "--candidate-id", da.CANDIDATE_B,
            "--output-dir", "out",
            "--method-lock", "lock.json",
            "--method-lock-sha256", "a" * 64,
            "--selected-iq-archive", "selected.npz",
            "--selected-iq-archive-sha256", "b" * 64,
            "--selected-iq-receipt", "selected.receipt.json",
            "--selected-iq-receipt-sha256", "c" * 64,
            "--ls-label-join-archive", "labels.npz",
            "--ls-label-join-archive-sha256", "d" * 64,
            "--checkpoint", "checkpoint.pth",
            "--checkpoint-sha256", "e" * 64,
            "--device", "cuda:1",
        ]
    )
    assert result == 0
    assert captured["candidate_id"] == da.CANDIDATE_B
    assert captured["device"] == "cuda:1"
    assert set(captured) == {
        "candidate_id", "output_dir", "method_lock_path", "method_lock_sha256",
        "selected_iq_archive", "selected_iq_archive_sha256", "selected_iq_receipt",
        "selected_iq_receipt_sha256", "ls_label_join_archive",
        "ls_label_join_archive_sha256", "checkpoint", "checkpoint_sha256", "device",
    }
    assert "D127_PHASE1_SINGLE_CANDIDATE_BUNDLE_WRITTEN" in capsys.readouterr().out


def test_cli_has_no_unfrozen_multi_candidate_or_target_option() -> None:
    module = _script_module()
    action_names = {action.dest for action in module._parser()._actions}
    assert "candidate_id" in action_names
    assert not {"candidate_ids", "merge", "target", "target_capsule", "target_data"}.intersection(action_names)
