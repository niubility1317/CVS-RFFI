from __future__ import annotations

import torch

from cvsrffi.factored_slow_fast import FactoredSlowFastState
from cvsrffi.factored_slow_fast_bundle import load_factored_bundle_strict, save_factored_bundle


def _state() -> FactoredSlowFastState:
    return FactoredSlowFastState(
        receiver_basis=torch.eye(6)[:, :2],
        leo_basis=torch.eye(6)[:, 2:4],
        geometric_centers=torch.tensor([[0.0, 0.0, 0.9, 0.1, 0.0, 0.0], [0.0, 0.0, 0.1, 0.9, 0.0, 0.0]]),
        decision_prototypes=torch.tensor([[0.0, 0.0, 1.0, 0.1, 0.0, 0.0], [0.0, 0.0, 0.1, 1.0, 0.0, 0.0]]),
        class_ids=torch.tensor([10, 20]),
        ridge_receiver=0.1,
        ridge_leo=0.2,
    )


def test_bundle_contains_only_int8_source_aggregates_and_reloads(tmp_path) -> None:
    path = tmp_path / "factored.pt"
    state = _state()
    save_factored_bundle(path, state, candidate="B3", base_checkpoint_id="ADV3B02_CORE90_SOFT_E200")

    raw = torch.load(path, map_location="cpu", weights_only=True)
    assert set(raw) == {
        "schema", "candidate", "base_checkpoint_id", "class_ids", "ridge_receiver", "ridge_leo",
        "receiver_basis_q", "receiver_basis_scale", "leo_basis_q", "leo_basis_scale",
        "geometric_centers_q", "geometric_centers_scale",
    }
    assert raw["receiver_basis_q"].dtype == torch.int8
    assert raw["leo_basis_q"].dtype == torch.int8
    assert raw["geometric_centers_q"].dtype == torch.int8
    assert "decision_prototypes" not in raw
    loaded, audit = load_factored_bundle_strict(path, decision_prototypes=state.decision_prototypes)
    assert audit["aggregate_storage_dtype"] == "int8"
    assert audit["source_samples_representable"] is False
    assert torch.allclose(loaded.receiver_basis, state.receiver_basis, atol=0.01)
    assert torch.allclose(loaded.leo_basis, state.leo_basis, atol=0.01)
    assert torch.equal(loaded.class_ids, state.class_ids)
