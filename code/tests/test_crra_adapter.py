import sys
from pathlib import Path

import pytest
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[2]
CODE_ROOT = PROJECT_ROOT / "code"
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

from crra import (  # noqa: E402
    CRRAAdapter,
    LowRankDepthwiseResidual,
    SourceSupportGate,
    crra_gate_scale,
)


def test_crra_is_identity_before_gate_warmup():
    adapter = CRRAAdapter(iq_channels=8, feature_channels=16, rank=8)
    x = torch.randn(3, 16, 32)
    out = adapter(x, raw_iq=torch.randn(3, 2, 32), epoch=1)
    assert torch.allclose(out.feature, x, atol=1e-6)
    assert torch.allclose(out.gate, torch.zeros_like(out.gate))
    assert torch.allclose(out.correction_energy, torch.zeros_like(out.correction_energy))
    assert crra_gate_scale(16) == 0.0


def test_crra_preserves_iq_pairing_and_bounds_intervention():
    adapter = CRRAAdapter(
        iq_channels=8,
        feature_channels=16,
        rank=8,
        alpha_max=0.25,
    )
    out = adapter(
        torch.randn(4, 16, 32),
        raw_iq=torch.randn(4, 2, 32),
        epoch=80,
    )
    assert out.feature.shape == (4, 16, 32)
    assert float(out.alpha.max().detach()) <= 0.25 + 1e-6
    assert torch.isfinite(out.correction_energy).all()
    assert torch.isfinite(out.support_distance).all()
    assert out.q.shape[0] == 4


def test_condition_q_does_not_backpropagate_into_condition_source():
    adapter = CRRAAdapter(iq_channels=8, feature_channels=16, rank=8)
    raw = torch.randn(2, 2, 32, requires_grad=True)
    out = adapter(torch.randn(2, 16, 32), raw_iq=raw, epoch=80)
    out.feature.sum().backward()
    assert raw.grad is None or torch.allclose(raw.grad, torch.zeros_like(raw.grad))


def test_crra_gate_schedule_ramps_and_then_stays_fixed():
    assert crra_gate_scale(1) == 0.0
    assert crra_gate_scale(16) == 0.0
    assert 0.0 < crra_gate_scale(30) < 1.0
    assert crra_gate_scale(47) == 1.0
    assert crra_gate_scale(200) == 1.0


def test_crra_rejects_unpaired_feature_channels():
    with pytest.raises(ValueError, match="even"):
        CRRAAdapter(iq_channels=3, feature_channels=5, rank=2)


def test_source_support_mask_updates_only_clean_rows():
    adapter = CRRAAdapter(iq_channels=2, feature_channels=4, rank=2)
    adapter.train()
    adapter(
        torch.randn(4, 4, 16),
        raw_iq=torch.randn(4, 2, 16),
        epoch=1,
        update_source_support=True,
        source_support_mask=torch.tensor([True, True, False, False]),
    )
    assert int(adapter.support.count.item()) == 2


def test_crra_uses_an_alpha_for_each_iq_filter_pair():
    adapter = CRRAAdapter(
        iq_channels=3,
        feature_channels=6,
        rank=2,
        alpha_max=0.25,
    )
    out = adapter(
        torch.randn(4, 6, 32),
        raw_iq=torch.randn(4, 2, 32),
        epoch=47,
    )
    assert out.alpha.shape == (4, 3)
    assert torch.all(out.alpha >= 0.0)
    assert float(out.alpha.detach().max()) <= 0.25 + 1e-6


def test_low_rank_residual_is_film_conditioned_and_zero_initialized():
    residual = LowRankDepthwiseResidual(
        feature_channels=4,
        rank=2,
        condition_dim=3,
        kernel_size=5,
    )
    x = torch.randn(2, 4, 16)
    q0 = torch.zeros(2, 3)
    q1 = torch.ones(2, 3)
    assert torch.allclose(residual(x, q0), torch.zeros_like(x), atol=1e-7)
    with torch.no_grad():
        residual.up.weight.fill_(0.1)
        residual.film.weight.fill_(0.1)
    assert not torch.allclose(residual(x, q0), residual(x, q1))


def test_source_support_gate_uses_domain_centres_and_mahalanobis_kernel():
    support = SourceSupportGate(q_dim=2, num_domains=2, tau=0.5, momentum=0.0)
    support.train()
    q_source = torch.tensor([[0.0, 0.0], [0.0, 0.0], [4.0, 4.0], [4.0, 4.0]])
    support(q_source, update_source=True, update_domains=torch.tensor([0, 0, 1, 1]))
    q_eval = torch.tensor([[0.0, 0.0], [4.0, 4.0], [8.0, 8.0]])
    value, distance = support(q_eval)
    assert torch.allclose(value[:2], torch.ones(2), atol=1e-5)
    assert float(value[2]) < 0.1
    assert torch.allclose(value[2], torch.exp(-(distance[2].square()) / 0.5), atol=1e-5)
