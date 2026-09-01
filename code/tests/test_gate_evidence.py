from __future__ import annotations

import torch

from cvsrffi.gate_evidence import GateEvidenceState


BRANCHES = ("raw", "hom", "phase", "pa", "hos")


def _embeddings() -> dict[str, torch.Tensor]:
    separated = torch.tensor([[1.0, 0.0], [1.1, 0.0], [-1.0, 0.0], [-1.1, 0.0]])
    collapsed = torch.tensor([[0.1, 0.0], [0.1, 0.0], [0.1, 0.0], [0.1, 0.0]])
    return {
        "raw": separated,
        "hom": collapsed,
        "phase": separated * 0.8,
        "pa": separated * 0.5,
        "hos": separated * 0.2,
    }


def test_discriminability_ema_is_label_permutation_invariant_and_checkpointed() -> None:
    embeddings = _embeddings()
    labels = torch.tensor([0, 0, 1, 1])
    permuted = torch.tensor([7, 7, 3, 3])
    first = GateEvidenceState(BRANCHES, momentum=0.0)
    second = GateEvidenceState(BRANCHES, momentum=0.0)

    first.update_discriminability(embeddings, labels)
    second.update_discriminability(embeddings, permuted)

    torch.testing.assert_close(first.discriminability_ema, second.discriminability_ema)
    assert first.discriminability_ema[0] > first.discriminability_ema[1]
    restored = GateEvidenceState(BRANCHES, momentum=0.9)
    restored.load_state_dict(first.state_dict())
    torch.testing.assert_close(restored.discriminability_ema, first.discriminability_ema)


def test_frozen_discriminability_state_does_not_self_reinforce() -> None:
    state = GateEvidenceState(BRANCHES, momentum=0.0)
    state.update_discriminability(_embeddings(), torch.tensor([0, 0, 1, 1]))
    before = state.discriminability_ema.clone()
    state.freeze_discriminability(True)
    state.update_discriminability(
        {name: torch.randn(4, 2) for name in BRANCHES}, torch.tensor([0, 0, 1, 1])
    )
    torch.testing.assert_close(state.discriminability_ema, before)


def test_pair_stability_and_compose_sanitize_nonfinite_evidence() -> None:
    state = GateEvidenceState(BRANCHES, momentum=0.0)
    clean = {name: torch.tensor([[1.0, 0.0], [0.0, 1.0]]) for name in BRANCHES}
    leo = {name: value.clone() for name, value in clean.items()}
    leo["phase"] = -leo["phase"]
    stability = state.paired_stability(clean, leo)
    assert torch.all(stability[:, 0] > 0.99)
    assert torch.all(stability[:, 2] < 0.01)

    state.discriminability_ema.fill_(0.5)
    evidence = state.compose(
        identifiability=torch.tensor([[float("nan"), 0.2, 0.3, 0.4, 0.5]]),
        stability=stability[:1],
        uncertainty=torch.tensor([[0.1, float("inf"), 0.2, 0.3, 0.4]]),
    )
    assert torch.isfinite(evidence["I"]).all()
    assert torch.isfinite(evidence["D"]).all()
    assert torch.isfinite(evidence["S"]).all()
    assert torch.isfinite(evidence["U"]).all()
    assert evidence["I"][0, 0].item() == 0.0
    assert evidence["U"][0, 1].item() == 1.0
