from __future__ import annotations

import math

import torch

from cvsrffi.fisher_branches import FisherBranchBank


def _iq(amplitude: torch.Tensor, phase: torch.Tensor) -> torch.Tensor:
    z = amplitude * torch.exp(1j * phase)
    return torch.stack([z.real, z.imag], dim=1)


def _inputs(batch: int = 2, length: int = 96, emb_dim: int = 16):
    phase = torch.linspace(0.0, 2.0 * math.pi, length).repeat(batch, 1)
    amplitude = torch.ones(batch, length)
    canonical = _iq(amplitude, phase)
    s_hat = torch.complex(canonical[:, 0], canonical[:, 1])
    embeddings = [torch.randn(batch, emb_dim) for _ in range(3)]
    confidence = torch.ones(batch, length)
    return canonical, s_hat, embeddings, confidence


def test_five_branch_bank_returns_finite_three_level_evidence() -> None:
    torch.manual_seed(3)
    canonical, s_hat, embeddings, confidence = _inputs()
    bank = FisherBranchBank(embedding_dim=16)
    outputs = bank(
        canonical,
        s_hat,
        raw_embedding=embeddings[0],
        hom_embedding=embeddings[1],
        pa_embedding=embeddings[2],
        content_confidence=confidence,
    )

    assert tuple(outputs) == ("raw", "hom", "phase", "pa", "hos")
    for output in outputs.values():
        assert output.embedding.shape == (2, 16)
        assert output.local_mask.shape[0] == 2
        assert output.direction_gate.shape[0] == 2
        assert output.identifiability.shape == (2,)
        assert output.stability.shape == (2,)
        assert output.uncertainty.shape == (2,)
        assert torch.isfinite(output.embedding).all()
        assert torch.all((output.local_mask >= 0.0) & (output.local_mask <= 1.0))
        assert torch.all((output.direction_gate >= 0.0) & (output.direction_gate <= 1.0))


def test_pa_branch_reports_more_identifiability_for_rich_amplitude() -> None:
    length = 96
    phase = torch.linspace(0.0, 2.0 * math.pi, length).unsqueeze(0)
    constant_amp = torch.ones(1, length)
    rich_amp = torch.linspace(0.2, 1.8, length).unsqueeze(0)
    embeddings = torch.randn(1, 16)
    bank = FisherBranchBank(embedding_dim=16).eval()

    constant = bank(
        _iq(constant_amp, phase),
        constant_amp * torch.exp(1j * phase),
        raw_embedding=embeddings,
        hom_embedding=embeddings,
        pa_embedding=embeddings,
    )["pa"]
    rich = bank(
        _iq(rich_amp, phase),
        rich_amp * torch.exp(1j * phase),
        raw_embedding=embeddings,
        hom_embedding=embeddings,
        pa_embedding=embeddings,
    )["pa"]

    assert rich.identifiability.item() > constant.identifiability.item()


def test_phase_and_hos_uncertainty_increase_under_slip_and_segment_instability() -> None:
    canonical, s_hat, embeddings, confidence = _inputs(batch=1, length=128)
    bank = FisherBranchBank(embedding_dim=16).eval()
    stable = bank(
        canonical,
        s_hat,
        raw_embedding=embeddings[0],
        hom_embedding=embeddings[1],
        pa_embedding=embeddings[2],
        content_confidence=confidence,
    )

    corrupted = canonical.clone()
    corrupted[:, :, 64:] *= -1.0
    corrupted[:, :, 96:] *= 3.0
    corrupted_s = torch.complex(corrupted[:, 0], corrupted[:, 1])
    unstable = bank(
        corrupted,
        corrupted_s,
        raw_embedding=embeddings[0],
        hom_embedding=embeddings[1],
        pa_embedding=embeddings[2],
        content_confidence=confidence,
    )

    assert unstable["phase"].uncertainty.item() > stable["phase"].uncertainty.item()
    assert unstable["hos"].uncertainty.item() > stable["hos"].uncertainty.item()
