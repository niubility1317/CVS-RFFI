"""Focused L/T preparation, physical timing, and frozen-artifact checks."""
import argparse
import copy
import json
import math
from pathlib import Path
import tempfile

import torch
from torch import nn
from torch.nn import functional as F

from .actions import (
    fit_actions, load_actions, signal_state, logit_margin, fractional_delay,
    apply_physical_action, action_contract,
)


class TinyParent(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv1d(2, 8, 5, padding=2)
        self.bn = nn.BatchNorm1d(8)
        self.projection = nn.Linear(8 * 8, 12)
        self.weight = nn.Parameter(torch.randn(6, 12))
        self.calls = 0

    def forward(self, x):
        self.calls += 1
        h = F.adaptive_avg_pool1d(F.silu(self.bn(self.conv(x))), 8).flatten(1)
        h = F.normalize(self.projection(h), dim=1)
        return 30 * (h @ F.normalize(self.weight, dim=1).T)


def timing_checks(device):
    gen = torch.Generator(device="cpu").manual_seed(9901)
    x = torch.randn(3, 2, 256, generator=gen, dtype=torch.float64).to(device)
    zero_delay = torch.zeros(3, device=device, dtype=torch.float64, requires_grad=True)
    zero = fractional_delay(x, zero_delay)
    assert torch.equal(zero, x), "Zero timing action must be exactly identity"
    zero[:, :, 45:120].square().sum().backward()
    assert bool(torch.isfinite(zero_delay.grad).all()) and float(zero_delay.grad.abs().sum()) > 0
    assert torch.equal(apply_physical_action(x, x.new_zeros(3, 5), "temporal"), x)
    assert torch.equal(apply_physical_action(x, x.new_zeros(3, 4), "linear"), x)

    # Zero extension represents finite received packets. Constant/interior
    # sinusoid checks tolerate only the explicitly documented truncation tail.
    n = torch.arange(256, dtype=torch.float64, device=device)
    omega = 2 * math.pi * .075
    wave = torch.stack((torch.cos(omega * n), torch.sin(omega * n)))[None].repeat(2, 1, 1)
    delays = x.new_tensor([.25, -.25])
    shifted = fractional_delay(wave, delays)
    expected_phase = omega * (n[None] - delays[:, None])
    expected = torch.stack((expected_phase.cos(), expected_phase.sin()), 1)
    sinusoid_error = float((shifted[:, :, 80:176] - expected[:, :, 80:176]).abs().max())
    assert sinusoid_error < .006, sinusoid_error
    constant = x.new_zeros(2, 2, 256)
    constant[:, 0] = 1.
    constant_result = fractional_delay(constant, delays)
    constant_error = float((constant_result[:, :, 80:176] - constant[:, :, 80:176]).abs().max())
    assert constant_error < .005, constant_error
    assert bool(torch.isfinite(shifted).all() and torch.isfinite(constant_result).all())

    # A boundary impulse must not acquire the circular neighbor seen by an
    # unpadded 256 FFT; a small sinc tail is expected, not exact zero support.
    impulse = x.new_zeros(2, 2, 256)
    impulse[0, 0, 0] = 1.
    impulse[1, 0, -1] = 1.
    delayed = fractional_delay(impulse, delays)
    far_tail = max(float(delayed[0, :, -1].abs().max()), float(delayed[1, :, 0].abs().max()))
    assert far_tail < .004, far_tail
    assert bool(torch.isfinite(delayed).all())

    delay = x.new_tensor([.13, -.21, .07], requires_grad=True)
    coeff = torch.randn(x.shape, generator=gen, dtype=torch.float64).to(device)
    value = (fractional_delay(x, delay) * coeff).sum()
    value.backward()
    analytic = delay.grad.clone()
    numerical = torch.zeros_like(delay)
    eps = 1e-5
    for i in range(len(delay)):
        plus, minus = delay.detach().clone(), delay.detach().clone()
        plus[i] += eps
        minus[i] -= eps
        numerical[i] = (((fractional_delay(x, plus) - fractional_delay(x, minus)) * coeff).sum() / (2 * eps))
    assert torch.allclose(analytic, numerical, atol=2e-6, rtol=1e-6), (analytic, numerical)
    parameters = torch.tensor([[1., -1., 1., -1.]], device=device, dtype=torch.float64).expand(3, -1)
    line = apply_physical_action(x, parameters, "linear")
    complex_peak = torch.complex(x[:, 0], x[:, 1]).abs().max(1).values
    delta_peak = torch.complex(line[:, 0] - x[:, 0], line[:, 1] - x[:, 1]).abs().max(1).values
    assert bool((delta_peak <= 2 * .06 * math.sqrt(2) * complex_peak + 1e-9).all())
    return dict(zero_identity=True, zero_delay_gradient_nonzero=True,
                sinusoid_interior_max_error=sinusoid_error, constant_interior_max_error=constant_error,
                boundary_far_tail_max=far_tail, signed_delay_gradient_finite_difference=True,
                linear_amplitude_bound=True)


def run(device="cpu"):
    torch.set_num_threads(2)
    torch.manual_seed(3001)
    parent = TinyParent().to(device)
    parent.eval()
    gen = torch.Generator(device="cpu").manual_seed(12321)
    x = torch.randn(17, 2, 256, generator=gen)
    with torch.no_grad():
        logits = parent(x.to(device))
        y = logits.argmax(1).cpu()
        u0 = logit_margin(logits, y.to(device)).cpu()
    cache = dict(x=x, y=y, state=signal_state(x), ids=[f"L:{i}" for i in range(len(x))], u0=u0)
    before = copy.deepcopy(parent.state_dict())
    parent.train()
    parent.bn.eval()  # Preserve mixed module modes, not just top-level mode.
    parent.conv.weight.requires_grad_(False)
    for p in parent.parameters():
        p.grad = torch.ones_like(p) * .003
    gradient_before = [p.grad.clone() for p in parent.parameters()]
    flags = [p.requires_grad for p in parent.parameters()]
    modes = [m.training for m in parent.modules()]
    calls_before = parent.calls
    random_before = torch.get_rng_state().clone()
    payload = fit_actions(parent, cache, 2026092701, device, passes=2, batch_size=6)
    assert torch.equal(random_before, torch.get_rng_state()), "Preparation consumed global training RNG"
    assert all(torch.equal(before[k], parent.state_dict()[k]) for k in before)
    assert flags == [p.requires_grad for p in parent.parameters()]
    assert modes == [m.training for m in parent.modules()]
    assert all(torch.equal(a, p.grad) for a, p in zip(gradient_before, parent.parameters()))
    assert payload["diagnostics"]["parent_clean_forward_batches"] == 0
    assert parent.calls - calls_before == 12
    for kind in ("linear", "temporal"):
        diagnostics = payload["diagnostics"]["branches"][kind]
        assert diagnostics["optimizer_steps"] == 6
        assert diagnostics["total_samples"] == 34
        assert all(p["samples"] == p["unique_physical_ids"] == 17 for p in diagnostics["passes"])
        assert all(p["mean_action_gradient_norm"] > 0 for p in diagnostics["passes"])
    with tempfile.TemporaryDirectory(prefix="stage2-actions-check-") as directory:
        path = Path(directory) / "actions.pt"
        torch.save(payload, path)
        loaded = torch.load(path, map_location="cpu", weights_only=False)
        first, second = load_actions(loaded, device), load_actions(loaded, device)
    assert not any(p.requires_grad for p in first.networks.parameters())
    assert not hasattr(first, "parent")
    parent_calls = parent.calls
    g_first, g_second = torch.Generator().manual_seed(31), torch.Generator().manual_seed(31)
    stats = []
    for kind in ("linear", "linear", "temporal", "temporal"):
        a, diagnostic = first.transform(x.to(device), kind, g_first)
        b, _ = second.transform(x.to(device), kind, g_second)
        assert torch.equal(a, b), "Shared C2/C3 action artifact/code must produce identical IQ"
        assert not a.requires_grad
        assert diagnostic["finite"] and diagnostic["within_registered_bounds"]
        stats.append(diagnostic)
    assert parent.calls == parent_calls
    transform_parent_forwards = parent.calls - parent_calls
    assert torch.equal(random_before, torch.get_rng_state())
    again = fit_actions(parent, cache, 2026092701, device, passes=2, batch_size=6)
    fit_max_delta = max(float((payload["networks"][k][name] - again["networks"][k][name]).abs().max())
                        for k in payload["networks"] for name in payload["networks"][k])
    # Private RNG fixes the draws, not the reduction order of default CUDA
    # kernels. CPU must match bitwise; CUDA permits FP32 roundoff only.
    fit_tolerance = 0. if str(device) == "cpu" else 1e-6
    assert fit_max_delta <= fit_tolerance, dict(fit_max_delta=fit_max_delta, tolerance=fit_tolerance)
    invalid = dict(cache, ids=["duplicated"] * len(x))
    try:
        fit_actions(parent, invalid, 1, device)
        raise AssertionError("Duplicate physical IDs accepted")
    except ValueError:
        pass
    return dict(status="PASS", device=str(device), contract=action_contract(),
                fit_diagnostics=payload["diagnostics"], parent_modes_parameters_gradients_preserved=True,
                private_rng=True, deterministic_fit=True, shared_frozen_actions=True,
                repeated_fit_max_parameter_difference=fit_max_delta,
                repeated_fit_parameter_tolerance=fit_tolerance,
                transform_parent_forwards=transform_parent_forwards,
                timing=timing_checks(device), sample_transform_stats=stats)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output")
    args = parser.parse_args()
    result = run(args.device)
    text = json.dumps(result, ensure_ascii=False, indent=2)
    if args.output:
        path = Path(args.output)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text + "\n", encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
