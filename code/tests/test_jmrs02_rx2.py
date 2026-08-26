import pytest
import torch

from cvsrffi.jmrs02_rx2 import (
    build_rx2_module,
    real_core_backward_probe,
    require_finite_gradients,
)
from cvsrffi.jmrs02_j1 import J1Config


def _inputs(batch=8, length=256, classes=6):
    torch.manual_seed(29)
    clean = torch.randn(batch, 2, length)
    satellite = clean + 0.08 * torch.randn_like(clean)
    base_logits = torch.randn(batch, classes)
    receiver = torch.arange(batch) % 7
    return clean, satellite, base_logits, receiver


def test_rx2_and_global_rx0_are_exact_identity_at_initialization():
    clean, _, base_logits, _ = _inputs()
    cfg = J1Config(z_dim=24, num_classes=base_logits.shape[1])
    z_id = torch.randn(clean.size(0), cfg.z_dim)
    domain = torch.zeros(clean.size(0), dtype=torch.long)
    for row in ("RX0", "RX2"):
        model = build_rx2_module(row, cfg)
        output = model(iq=clean, z_id=z_id, base_logits=base_logits, domain=domain)
        assert torch.allclose(output.corrected_iq, clean, atol=1e-6, rtol=1e-6)
        assert torch.equal(output.final_logits, base_logits)
    assert build_rx2_module("RX0", cfg).conditioning_enabled is False
    assert build_rx2_module("RX2", cfg).conditioning_enabled is True


def test_real_core_backward_probe_reaches_estimator_with_finite_gradients():
    clean, satellite, base_logits, receiver = _inputs()
    cfg = J1Config(z_dim=24, num_classes=base_logits.shape[1])
    z_id = torch.randn(clean.size(0), cfg.z_dim)
    domain = torch.zeros(clean.size(0), dtype=torch.long)
    model = build_rx2_module("RX2", cfg)
    output = model(iq=clean, z_id=z_id, base_logits=base_logits, domain=domain)
    core = torch.nn.Sequential(
        torch.nn.Flatten(), torch.nn.Linear(2 * clean.size(-1), base_logits.size(1))
    )
    for parameter in core.parameters():
        parameter.requires_grad_(False)
    candidate_logits = core(output.corrected_iq)
    health = real_core_backward_probe(
        model,
        candidate_logits,
        receiver % base_logits.size(1),
        output.diagnostics["correction_norm"],
    )
    assert health["nonfinite_elements"] == 0
    assert health["estimator_grad_norm"] > 0.0


def test_nonfinite_gradient_is_a_hard_failure_not_sanitized():
    model = torch.nn.Linear(2, 1)
    model.weight.grad = torch.tensor([[float("nan"), 0.0]])
    with pytest.raises(FloatingPointError, match="non-finite"):
        require_finite_gradients(model)
