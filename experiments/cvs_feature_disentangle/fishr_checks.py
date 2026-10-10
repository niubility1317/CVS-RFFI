"""Finite CPU/CUDA checks for the exact cosine Fishr definition and state."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F

from .fishr import (ConditionalFishr, GradientRatioCalibrator,
                    cosine_sample_gradients, gradient_norm, loss_gradients,
                    pair_penalty, sample_variance)


def run(device="cpu"):
    torch.manual_seed(83617)
    torch.set_num_threads(2)
    dtype = torch.float64
    results = {}
    z = torch.randn(48, 160, device=device, dtype=dtype, requires_grad=True)
    weight = torch.randn(6, 160, device=device, dtype=dtype, requires_grad=True)
    labels = torch.arange(6, device=device).repeat_interleave(4).repeat(2)
    rx = torch.tensor([1] * 24 + [3] * 24, device=device)
    day = torch.ones(48, device=device, dtype=torch.long)
    raw = cosine_sample_gradients(z, weight, labels, "raw")
    logits = 30 * F.normalize(z, dim=1, eps=1e-4) @ F.normalize(weight, dim=1, eps=1e-4).T
    ce = F.cross_entropy(logits, labels, reduction="none")
    automatic = torch.stack([torch.autograd.grad(ce[i], weight, retain_graph=True,
                                               create_graph=True)[0] for i in range(48)])
    torch.testing.assert_close(raw, automatic, rtol=1e-10, atol=1e-12)
    results["raw_vs_per_sample_autograd_max_error"] = float((raw - automatic).abs().max().detach())
    direction = cosine_sample_gradients(z, weight, labels)
    torch.testing.assert_close(direction, raw * weight.norm(dim=1)[None, :, None])
    scales = torch.tensor([.25, .5, 1, 2, 4, 8], device=device, dtype=dtype)[:, None]
    torch.testing.assert_close(direction, cosine_sample_gradients(z, weight * scales, labels),
                               rtol=1e-10, atol=1e-12)
    def direct_penalty(g):
        return pair_penalty(sample_variance(g[:24]), sample_variance(g[24:]))
    initial = direct_penalty(raw)
    scaled = direct_penalty(cosine_sample_gradients(z, weight * 3, labels, "raw"))
    torch.testing.assert_close(scaled, initial / 81, rtol=1e-10, atol=1e-12)
    results["raw_a_minus_four_ratio"] = float((scaled / initial).detach())
    results["direction_independent_positive_row_scale_invariant"] = True
    v = sample_variance(direction[:24])
    torch.testing.assert_close(v, direction[:24].var(dim=0, unbiased=False))
    a, b = sample_variance(direction[:24]), sample_variance(direction[24:])
    center = (a + b) / 2
    torch.testing.assert_close(pair_penalty(a, b),
                               ((a-center).square().mean()+(b-center).square().mean())/2)
    results["variance_n_denominator_and_pair_quarter"] = True
    for arg in ("z", "weight"):
        for magnitude in (0., 1e-6, 1e-4):
            zz, ww = z.detach().clone(), weight.detach().clone()
            value = zz if arg == "z" else ww
            value[0] = 0
            value[0, 0] = magnitude
            try:
                cosine_sample_gradients(zz, ww, labels)
            except ValueError:
                pass
            else:
                raise AssertionError("Clamped cosine norm must be rejected")
    results["epsilon_regime_rejected"] = True

    bank = ConditionalFishr()
    first, m1 = bank(z, weight, labels, rx, day, 8)
    assert first > 0 and m1["bucket_counts"] == [1, 1]
    torch.testing.assert_close(first, direct_penalty(direction))
    assert all(not v["m"].requires_grad for v in bank.buckets.values())
    second, m2 = bank(z, weight, labels, rx, day, 16)
    torch.testing.assert_close(second, first)
    assert m2["bucket_counts"] == [2, 2]
    g1 = torch.autograd.grad(first, z, retain_graph=True)[0]
    g2 = torch.autograd.grad(second, z, retain_graph=True)[0]
    torch.testing.assert_close(g2, g1 * (.1 / .19), rtol=1e-9, atol=1e-12)
    rx_new = torch.where(rx == 3, 4, rx)
    bank(z, weight, labels, rx_new, day, 24)
    assert [bank.buckets[(r, 1, "clean")]["count"] for r in (1, 3, 4)] == [3, 2, 1]
    assert bank.buckets[(3, 1, "clean")]["update_step"] == 16
    results["EMA_independent_count_debias_detach_coefficient"] = True

    restored = ConditionalFishr()
    restored.load_state_dict(bank.state_dict())
    def same_states(left, right):
        assert left.buckets.keys() == right.buckets.keys()
        for k in left.buckets:
            assert left.buckets[k]["count"] == right.buckets[k]["count"]
            assert left.buckets[k]["update_step"] == right.buckets[k]["update_step"]
            torch.testing.assert_close(left.buckets[k]["m"], right.buckets[k]["m"].to(left.buckets[k]["m"]))
    same_states(bank, restored)
    bank.eval()
    _, ev = bank(z, weight, labels, rx, day, 32, update=True)
    assert not ev["updated"]
    same_states(bank, restored)
    bank.train()
    bank(z, weight, labels, rx, day, 32, update=False)
    same_states(bank, restored)
    no_loss, invalid = bank(z, weight, labels, torch.ones_like(rx), day, 32)
    assert no_loss == 0 and invalid["reason"]
    same_states(bank, restored)
    try:
        bank(z, weight, labels, rx, day, 32, scene="leo")
    except ValueError:
        pass
    else:
        raise AssertionError("No-satellite Fishr cannot accept LEO")
    results["state_roundtrip_missing_bucket_and_validation_no_mutation"] = True

    encoder = nn.Linear(20, 320, device=device, dtype=dtype)
    fusion = nn.Linear(320, 160, device=device, dtype=dtype)
    classifier = nn.Parameter(weight.detach().clone())
    inp = torch.randn(48, 20, device=device, dtype=dtype)
    fused = fusion(encoder(inp).tanh())
    mature = ConditionalFishr()
    for step in range(1, 41):
        loss, info = mature(fused, classifier, labels, rx, day, step * 8)
    parameters = tuple(encoder.parameters()) + tuple(fusion.parameters()) + (classifier,)
    grads = loss_gradients(loss, parameters)
    assert all(g is not None and torch.isfinite(g).all() and g.norm() > 0 for g in grads)
    results["true_upstream_E_G_C_gradients"] = {
        "E": gradient_norm(grads[:2]), "G": gradient_norm(grads[2:4]),
        "C": gradient_norm(grads[4:])}
    assert info["bucket_counts"] == [40, 40]
    assert abs(info["buckets"][0]["current_gradient_coefficient"] - .1/(1-.9**40)) < 1e-12
    identity_loss = F.cross_entropy(30 * F.normalize(fused, dim=1) @
                                    F.normalize(classifier, dim=1).T, labels)
    results["fixed_calibration"] = []
    for target in (.03, .05, .1):
        calibration = GradientRatioCalibrator(target)
        assert not calibration.observe(identity_loss, loss, parameters, [1, 1])["accepted"]
        observed = calibration.observe(identity_loss, loss, parameters, info["bucket_counts"])
        maximum = calibration.freeze()
        achieved = maximum * observed["fishr_gradient_norm"] / observed["ce_gradient_norm"]
        assert abs(achieved - target) < 1e-12
        assert calibration.weight(20) == 0
        assert calibration.weight(21) == maximum * .05
        assert calibration.weight(40) == calibration.weight(200) == maximum
        try:
            calibration.observe(identity_loss, loss, parameters, [40, 40])
        except RuntimeError:
            pass
        else:
            raise AssertionError("Calibration cannot adapt after freeze")
        copy = GradientRatioCalibrator(target)
        copy.load_state_dict(calibration.state_dict())
        assert copy.weight(131) == maximum
        results["fixed_calibration"].append({"target": target, "achieved": achieved,
                                              "lambda_max": maximum})
    return {"status": "PASS", "device": device, "dtype": str(dtype), "checks": results}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    result = run(args.device)
    path = Path(args.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    assert json.loads(path.read_text(encoding="utf-8"))["status"] == "PASS"
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
