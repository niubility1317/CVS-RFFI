import math
import sys
from dataclasses import fields
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "code"))
from cvsrffi.a1_fast_runtime import GradientSnapshot
from cvsrffi.muse_ssdg import (
    RC4Calibration, _rc4_effective_budget_batched,
    apply_rc4_quality_budget, route_fasttrust_rc4,
)


DEVICES = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])


def legacy_norm(model, predicate=None):
    total, seen = 0.0, 0
    for name, parameter in model.named_parameters():
        if parameter.grad is None or (predicate is not None and not predicate(name)):
            continue
        value = float(parameter.grad.detach().float().norm(2).item())
        total += value * value
        seen += 1
    return total ** 0.5 if seen else float("nan")


@pytest.mark.parametrize("device", DEVICES)
def test_gradient_snapshot_exact_norm_filter_missing_and_clip(device):
    model = torch.nn.Sequential(torch.nn.Linear(3, 4), torch.nn.Linear(4, 2)).to(device)
    generator = torch.Generator(device=device).manual_seed(49)
    for parameter in model.parameters():
        parameter.grad = torch.randn(parameter.shape, device=device, generator=generator)
    model[0].bias.grad = None
    snapshot = GradientSnapshot.capture(model)
    for predicate in (None, lambda name: name.startswith("0."), lambda name: "weight" in name):
        assert snapshot.norm(predicate) == legacy_norm(model, predicate)
    assert math.isnan(snapshot.norm(lambda _: False))
    assert snapshot.first_nonfinite() is None
    torch.nn.utils.clip_grad_norm_(model.parameters(), 0.1)
    after = GradientSnapshot.capture(model)
    assert after.norm() == legacy_norm(model)
    assert after.norm() < snapshot.norm()
    model.zero_grad(set_to_none=True)
    empty = GradientSnapshot.capture(model)
    assert math.isnan(empty.norm()) and empty.first_nonfinite() is None


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("bad", [float("nan"), float("inf"), -float("inf")])
def test_gradient_snapshot_first_anomaly_exact(device, bad):
    model = torch.nn.Sequential(torch.nn.Linear(3, 2), torch.nn.Linear(2, 2)).to(device)
    for parameter in model.parameters():
        parameter.grad = torch.ones_like(parameter)
    model[0].weight.grad[0] = torch.tensor([bad, float("inf"), -float("inf")], device=device)
    model[1].weight.grad.fill_(float("nan"))
    snapshot = GradientSnapshot.capture(model)
    assert snapshot.first_nonfinite() == {
        "parameter_name": "0.weight", "nonfinite_elements": 3,
        "nan_elements": int(math.isnan(bad)),
        "posinf_elements": 1 + int(bad == float("inf")),
        "neginf_elements": 1 + int(bad == -float("inf")),
    }


def legacy_effective(mask, weights, fraction):
    selected = torch.zeros_like(mask)
    budget = float(mask.numel()) * fraction
    if budget <= 0.0 or not bool(mask.any()):
        return selected
    ordered = sorted(mask.nonzero().reshape(-1).cpu().tolist(),
                     key=lambda i: (-float(weights[i].cpu().item()), i))
    used = 0.0
    for index in ordered:
        value = float(weights[index].cpu().item())
        if value <= 0.0:
            continue
        if used + value <= budget + 1e-12:
            selected[index] = True
            used += value
    if not bool(selected.any()) and ordered and budget > 0.0:
        selected[ordered[0]] = True
        weights[ordered[0]] = min(float(weights[ordered[0]].item()), budget)
    return selected


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("fraction", [0.0, 1e-14, 0.025, 0.10, 0.25, 1.0])
def test_effective_budget_boundary_random_masks_and_mutation(device, fraction):
    generator = torch.Generator().manual_seed(392005)
    for seed in range(12):
        weights = torch.rand(32, generator=generator, dtype=torch.float64)
        weights[:8] = torch.tensor([0, 0, .8, .8, .4, .4 + 1e-12, .4 - 1e-12, 2.0], dtype=torch.float64)
        mask = torch.rand(32, generator=generator) > .3
        if seed == 0:
            mask[:] = False
        weights, mask = weights.to(device), mask.to(device)
        original, optimized = weights.clone(), weights.clone()
        expected = legacy_effective(mask, original, fraction)
        actual = _rc4_effective_budget_batched(mask, optimized, fraction)
        assert torch.equal(actual, expected)
        assert torch.equal(optimized, original)


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("fraction", [0.0, 1e-14, 0.025, .1, .5, 1.0])
def test_quality_budget_exact_h_first_and_residual(device, fraction):
    generator = torch.Generator().manual_seed(85)
    for _ in range(12):
        weights = torch.rand(32, generator=generator).to(device)
        weights[:5] = torch.tensor([float("nan"), float("inf"), -float("inf"), 0, .8], device=device)
        state = torch.randint(0, 3, (32,), generator=generator).to(device)
        outputs = [apply_rc4_quality_budget(state == 1, state == 2, weights,
                   total_budget=fraction, batched_readback=mode) for mode in (False, True)]
        for expected, actual in zip(*outputs):
            assert torch.equal(actual, expected)


def calibration_fixture():
    values = {field.name: 0 for field in fields(RC4Calibration)}
    values.update(temperature=1.0, feature_mean=torch.zeros(7), feature_scale=torch.ones(7),
                  partial_feature_mean=torch.zeros(12), partial_feature_scale=torch.ones(12),
                  correctness_weight=torch.tensor([3.] + [0.] * 15),
                  partial_safety_weight=torch.tensor([3.] + [0.] * 20),
                  exclusion_safety_weight=torch.tensor([3.] + [0.] * 20),
                  aps_global=.8, aps_partial_global=.8, aps_negative_global=.8,
                  aps_by_class=torch.full((6,), .8), aps_by_domain=torch.full((2,), .8),
                  partial_threshold_scope="predicted_class", hard_risk_threshold=.7,
                  partial_safety_threshold=.6, hard_ready=True, partial_ready=True,
                  negative_ready=True, num_classes=6, num_domains=2)
    return RC4Calibration(**values)


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("total_budget", [0.0, .1])
@pytest.mark.parametrize("cell_cap", [0.0, .02])
def test_complete_route_caps_and_weights_exact(device, total_budget, cell_cap):
    generator = torch.Generator().manual_seed(19)
    logits = (torch.randn(64, 6, generator=generator) * 3).to(device)
    kwargs = dict(domains=torch.arange(64, device=device) % 2,
                  receivers=torch.arange(64, device=device) % 4,
                  z_norm=torch.ones(64, device=device), calibration=calibration_fixture(),
                  hard_max_fraction=.5, hard_effective_budget=.05,
                  partial_effective_budget=.05, negative_effective_budget=.05,
                  class_receiver_cap=True, class_receiver_effective_budget=cell_cap,
                  total_identity_effective_budget=total_budget,
                  enable_negative=total_budget == 0.0)
    legacy = route_fasttrust_rc4(logits, logits, logits, **kwargs)
    fast = route_fasttrust_rc4(logits, logits, logits, **kwargs, batched_readback=True)
    for field in fields(legacy):
        assert torch.equal(getattr(legacy, field.name), getattr(fast, field.name)), field.name
