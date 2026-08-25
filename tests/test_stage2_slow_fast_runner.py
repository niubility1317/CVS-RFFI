from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch
import torch.nn as nn

from cvsrffi.slow_fast_adapter import SlowFastAdapterState, SlowFastCandidate
from cvsrffi.slow_fast_bundle import save_slow_fast_bundle
from cvsrffi import stage2_slow_fast_runner as subject
from cvsrffi.stage2_meta_adapter_scorer import score_meta_adapter_pair
from cvsrffi.slow_fast_scorer import _score_shadow_row


class _FrozenBase(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.anchor = nn.Parameter(torch.ones(()), requires_grad=False)


def _inputs(tmp_path: Path) -> dict[str, Path]:
    prototypes = torch.eye(2)
    direction = torch.tensor([1.0, -1.0])
    direction = direction / torch.linalg.vector_norm(direction)
    basis = torch.zeros(2, 4)
    basis[:, 0] = direction
    state = SlowFastAdapterState(
        candidate=SlowFastCandidate.COMMON_SHIFT_R4,
        slow_u=basis,
        common_coeff=torch.zeros(4),
    )
    bundle = tmp_path / "bundle.pt"
    save_slow_fast_bundle(
        bundle,
        state,
        {
            "base_checkpoint_id": "ADV3B02_CORE90_SOFT_E200",
            "class_ids": torch.tensor([10, 20]),
            "prototypes": prototypes,
            "support_logit_scale": 8.0,
            "fast_step_size": 0.02,
            "trust_radius": 0.2,
        },
    )
    base = tmp_path / "base.pth"
    base.write_bytes(b"loaded through monkeypatch")
    support = tmp_path / "support.npz"
    query = tmp_path / "query.npz"
    prototype = tmp_path / "prototype.npz"
    shift = np.asarray([0.9, -0.9], dtype=np.float32)
    support_features = np.asarray(
        [[1.0, 0.0], [1.0, 0.02], [0.0, 1.0], [0.02, 1.0]],
        dtype=np.float32,
    ) + shift
    np.savez(
        support,
        received_iq=support_features[:, :, None],
        support_labels=np.asarray([10, 10, 20, 20], dtype=np.int64),
        support_physical_ids=np.asarray(["s0", "s1", "s2", "s3"]),
    )
    np.savez(
        query,
        received_iq=(np.asarray([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32) + shift)[:, :, None],
        query_ids=np.asarray(["q0", "q1"]),
    )
    np.savez(
        prototype,
        prototypes=np.eye(2, dtype=np.float32),
        class_ids=np.asarray([10, 20], dtype=np.int64),
    )
    return {"bundle": bundle, "base": base, "support": support, "query": query, "prototype": prototype}


def _config(paths: dict[str, Path]) -> dict[str, object]:
    return {
        "candidate_id": "COMMON_SHIFT_R4",
        "bundle_id": "slow-fast-common-r4",
        "protocol_schema": "p2_min_v1",
        "phase2_data_status": "VALIDATED_ONCE",
        "capsule_id": "capsule-fixed",
        "split_id": "split-fixed",
        "base_checkpoint_path": str(paths["base"]),
        "bundle_path": str(paths["bundle"]),
        "support_path": str(paths["support"]),
        "query_path": str(paths["query"]),
        "prototype_path": str(paths["prototype"]),
        "receiver": "20-1",
        "scenario": "leo_clear_weak",
        "operating_point": "K10/new10",
        "seed": 392002,
        "k_shot": 2,
        "steps": 3,
    }


def _shadow_config(paths: dict[str, Path]) -> dict[str, object]:
    return {
        **_config(paths),
        "shadow_steps": [1],
        "shadow_step_multipliers": [1.0],
        "shadow_lambdas": [0.25, 0.5],
        "crossfit_repeats": 1,
    }


def test_runner_rejects_source_fields_before_creating_output(tmp_path: Path) -> None:
    paths = _inputs(tmp_path)
    config = _config(paths)
    config["source_cache_path"] = "forbidden.npz"

    with pytest.raises(ValueError, match="allowlist"):
        subject.run_slow_fast_stage2_row(config, tmp_path / "out", device="cpu")
    assert not (tmp_path / "out").exists()


def test_runner_closes_both_states_without_truth_or_query_updates(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    paths = _inputs(tmp_path)
    monkeypatch.setattr(subject, "_load_frozen_checkpoint", lambda *args, **kwargs: _FrozenBase())
    monkeypatch.setattr(
        subject,
        "_extract_features",
        lambda _model, rows: rows[:, :, 0].float(),
    )

    receipt = subject.run_slow_fast_stage2_row(
        _config(paths), tmp_path / "out", device="cpu"
    )

    assert receipt["status"] == "PREDICTIONS_COMPLETE"
    assert receipt["states"] == ["DA0_REG0", "DA1_REG0"]
    assert receipt["query_truth_opened"] is False
    assert receipt["query_role_opened"] is False
    assert receipt["source_opened"] is False
    assert receipt["query_state_update_count"] == 0
    for name in ("predictions_DA0_REG0.npz", "predictions_DA1_REG0.npz"):
        with np.load(tmp_path / "out" / name, allow_pickle=False) as artifact:
            assert set(artifact.files) == {"query_ids", "predicted_class_ids", "scores"}
            assert artifact["query_ids"].tolist() == ["q0", "q1"]


def test_existing_truth_last_scorer_accepts_slow_fast_receipt(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    paths = _inputs(tmp_path)
    monkeypatch.setattr(subject, "_load_frozen_checkpoint", lambda *args, **kwargs: _FrozenBase())
    monkeypatch.setattr(subject, "_extract_features", lambda _model, rows: rows[:, :, 0].float())
    output = tmp_path / "out"
    subject.run_slow_fast_stage2_row(_config(paths), output, device="cpu")
    truth = tmp_path / "truth.npz"
    np.savez(
        truth,
        query_ids=np.asarray(["q0", "q1"]),
        true_class_ids=np.asarray([10, 20], dtype=np.int64),
    )

    score = score_meta_adapter_pair(
        output / "predictions_DA0_REG0.npz",
        output / "predictions_DA1_REG0.npz",
        truth,
        receipt_path=output / "receipt.json",
    )

    assert score.candidate_id == "COMMON_SHIFT_R4"
    assert score.row["scenario"] == "leo_clear_weak"


def test_shadow_runner_outputs_preregistered_states_without_reextracting_query(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    paths = _inputs(tmp_path)
    calls: list[int] = []
    monkeypatch.setattr(subject, "_load_frozen_checkpoint", lambda *args, **kwargs: _FrozenBase())

    def extract(_model, rows):
        calls.append(int(rows.shape[0]))
        return rows[:, :, 0].float()

    monkeypatch.setattr(subject, "_extract_features", extract)

    receipt = subject.run_slow_fast_stage2_row(
        _shadow_config(paths), tmp_path / "shadow", device="cpu"
    )

    assert calls == [4, 1, 1]
    assert receipt["score_type"] == "raw_cosine"
    assert receipt["support_logit_scale"] == 8.0
    assert receipt["trust_radius"] == 0.2
    assert receipt["query_state_update_count"] == 0
    assert receipt["query_truth_opened"] is False
    assert receipt["source_opened"] is False
    expected = {
        "DA0_REG0",
        "DA1_L0250_REG0",
        "DA1_L0500_REG0",
        "DA1_GATE_LEGACY_REG0",
        "DA1_GATE_CF_REG0",
    }
    assert set(receipt["states"]) == expected
    assert set(receipt["prediction_paths"]) == expected
    assert receipt["support_selection"]["crossfit_fit_count"] == 2
    assert "attempted_gradient_updates" in receipt["support_selection"]

    truth = tmp_path / "truth.npz"
    np.savez(
        truth,
        query_ids=np.asarray(["q0", "q1"]),
        true_class_ids=np.asarray([10, 20], dtype=np.int64),
    )
    score = _score_shadow_row(
        tmp_path / "shadow" / "receipt.json",
        truth,
        class_binding_path=None,
    )
    assert set(score["states"]) == expected
    assert score["truth_opened_after_all_predictions_validated"] is True
