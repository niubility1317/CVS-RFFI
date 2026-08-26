from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from cvsrffi.factored_slow_fast import FactoredSlowFastState
from cvsrffi.factored_slow_fast_bundle import save_factored_bundle
from cvsrffi import stage2_factored_slow_fast_runner as subject


class _FrozenBase(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.anchor = nn.Parameter(torch.ones(()), requires_grad=False)


def _inputs(tmp_path: Path) -> dict[str, Path]:
    prototypes = torch.eye(2)
    state = FactoredSlowFastState(
        receiver_basis=torch.tensor([[1.0], [0.0]]),
        leo_basis=torch.tensor([[0.0], [1.0]]),
        geometric_centers=prototypes,
        decision_prototypes=prototypes,
        class_ids=torch.tensor([10, 20]),
    )
    bundle = tmp_path / "factored.pt"
    save_factored_bundle(bundle, state, candidate="B3", base_checkpoint_id="ADV3B02_CORE90_SOFT_E200")
    base = tmp_path / "base.pth"
    base.write_bytes(b"monkeypatched")
    support = tmp_path / "support.npz"
    query = tmp_path / "query.npz"
    prototype = tmp_path / "prototype.npz"
    np.savez(
        support,
        received_iq=np.asarray([[1.0, 0.05], [1.0, 0.02], [0.05, 1.0], [0.02, 1.0]], dtype=np.float32)[:, :, None],
        support_labels=np.asarray([10, 10, 20, 20], dtype=np.int64),
        support_physical_ids=np.asarray(["s0", "s1", "s2", "s3"]),
    )
    np.savez(
        query,
        received_iq=np.asarray([[1.0, 0.03], [0.03, 1.0]], dtype=np.float32)[:, :, None],
        query_ids=np.asarray(["q0", "q1"]),
    )
    np.savez(prototype, prototypes=np.eye(2, dtype=np.float32), class_ids=np.asarray([10, 20], dtype=np.int64))
    return {"bundle": bundle, "base": base, "support": support, "query": query, "prototype": prototype}


def _config(paths: dict[str, Path]) -> dict[str, object]:
    return {
        "adaptation_mode": "FACTORED_CONTEXT_ADAPT",
        "candidate_id": "CVS_FSFA_V2_B3",
        "bundle_id": "fsfa-v2-b3",
        "protocol_schema": "p2_min_v1",
        "phase2_data_status": "VALIDATED_ONCE",
        "capsule_id": "capsule-fixed",
        "split_id": "split-fixed",
        "base_checkpoint_path": str(paths["base"]),
        "bundle_path": str(paths["bundle"]),
        "support_path": str(paths["support"]),
        "query_path": str(paths["query"]),
        "prototype_path": str(paths["prototype"]),
        "receiver": "target-rx",
        "scenario": "leo_clear_weak",
        "operating_point": "K2",
        "seed": 392002,
        "k_shot": 2,
    }


def test_runner_updates_only_eight_or_fewer_context_values_and_keeps_query_read_only(tmp_path, monkeypatch) -> None:
    paths = _inputs(tmp_path)
    monkeypatch.setattr(subject, "_load_frozen_checkpoint", lambda *args, **kwargs: _FrozenBase())
    monkeypatch.setattr(subject, "_extract_features", lambda _model, rows: rows[:, :, 0].float())

    receipt = subject.run_factored_slow_fast_stage2_row(_config(paths), tmp_path / "out", device="cpu")

    assert receipt["states"] == ["DA0_REG0", "DA1_REG0"]
    assert receipt["query_truth_opened"] is False
    assert receipt["query_role_opened"] is False
    assert receipt["source_opened"] is False
    assert receipt["query_state_update_count"] == 0
    assert receipt["fast_parameter_count"] == 2
    assert receipt["optimizer_state_bytes"] == 0
    assert receipt["aggregate_storage_dtype"] == "int8"
    assert receipt["support_adapter_opened"] is True
    assert set(receipt["prediction_paths"]) == {"DA0_REG0", "DA1_REG0"}
