"""CPU synthetic checks for evidence geometry and set-mass loss only."""

from __future__ import annotations

import argparse
import json
import random

import numpy as np
import torch
from torch.nn import functional as F

from .objective import candidate_set, set_loss


def _raises(callback):
    try:
        callback()
    except ValueError:
        return
    raise AssertionError("expected malformed input/mask/constant rejection")


def _rng():
    return random.getstate(), torch.get_rng_state().clone(), np.random.get_state()


def _same_rng(left, right):
    return left[0] == right[0] and torch.equal(left[1], right[1]) and left[2][0] == right[2][0] and np.array_equal(left[2][1], right[2][1]) and left[2][2:] == right[2][2:]


def run_checks():
    rng_before = _rng()
    generator = torch.Generator(device="cpu").manual_seed(9271)
    teacher = [torch.randn(7, 160, generator=generator, dtype=torch.float64).requires_grad_() for _ in range(3)]
    weight = torch.randn(6, 160, generator=generator, dtype=torch.float64).requires_grad_()
    candidate = candidate_set(*teacher, weight)
    normalized_weight = F.normalize(weight.detach(), dim=1, eps=1e-4)
    corners = []
    for u in (0.0, 0.5):
        for v in (0.0, 0.5):
            joint = teacher[0].detach() + (1 - u) * teacher[1].detach() + (1 - v) * teacher[2].detach()
            scores = F.linear(joint, normalized_weight)
            corners.append(scores[:, :, None] - scores[:, None, :])
    corner_maximum = torch.stack(corners).amax(dim=0)
    assert torch.allclose(candidate["tensors"]["pairwise_maxima"], corner_maximum, atol=1e-12, rtol=0)
    expected_mask = (corner_maximum >= -1e-6).all(dim=2)
    assert torch.equal(candidate["candidates"], expected_mask)
    nominal = F.linear(F.normalize(sum(value.detach() for value in teacher), dim=1, eps=1e-4), normalized_weight).argmax(dim=1)
    assert bool(candidate["candidates"][torch.arange(7), nominal].all())
    assert not candidate["candidates"].requires_grad
    assert all(not tensor.requires_grad for tensor in candidate["tensors"].values() if tensor is not None)
    # beta=0 retains exactly nominal top ties, rather than choosing one label.
    basis_weight = torch.zeros(6, 160, dtype=torch.float64)
    basis_weight[0, 0] = 1; basis_weight[1, 1] = 1
    basis_weight[2, 0] = -1; basis_weight[3, 1] = -1
    basis_weight[4, 2] = 1; basis_weight[5, 2] = -1
    one = torch.zeros(1, 160, dtype=torch.float64); one[0, 0] = 1
    empty_increment = torch.zeros_like(one)
    tied_weight = basis_weight.clone(); tied_weight[1] = tied_weight[0]
    tied = candidate_set(one, empty_increment, empty_increment, tied_weight, beta=0, tolerance=0)
    assert tied["candidates"].tolist() == [[True, True, False, False, False, False]]
    zero_evidence = candidate_set(empty_increment, empty_increment, empty_increment, basis_weight)
    zero_weight = candidate_set(one, one, one, torch.zeros_like(basis_weight))
    assert bool(zero_evidence["candidates"].all()) and bool(zero_weight["candidates"].all())
    # Tolerance remains in raw score units when common packet scaling is used.
    near = one * 100.0
    near[0, 1] = 99.9999995
    near_set = candidate_set(near, empty_increment, empty_increment, basis_weight, beta=0, tolerance=1e-6)
    assert near_set["candidates"][0, 1]
    strict_set = candidate_set(near, empty_increment, empty_increment, basis_weight, beta=0, tolerance=0)
    assert not strict_set["candidates"][0, 1]
    # Pairwise-possible k=0 is never a simultaneous winner: competitors have
    # scores u-.1 and .4-u, whose sum is .3 at every point in the rectangle.
    conservative_weight = torch.zeros(6, 160, dtype=torch.float64)
    for index, dimension in enumerate((0, 1, 2, 1, 2, 1)):
        conservative_weight[index, dimension] = 1
    aa, bb, cc = [torch.zeros(1, 160, dtype=torch.float64) for _ in range(3)]
    aa[0, 1] = .9; aa[0, 2] = -.6
    bb[0, 1] = -1; bb[0, 2] = 1
    conservative = candidate_set(aa, bb, cc, conservative_weight, tolerance=0)
    assert conservative["candidates"][0, 0]
    nominal_scores = F.linear(aa + bb + cc, conservative_weight)
    assert abs((nominal_scores[0, 1] + nominal_scores[0, 2]).item() - .3) < 1e-14
    assert .3 > 0  # two competitors cannot both be <= candidate k's zero score
    # Candidate columns and W rows are equivariant to the same class permutation.
    permutation = torch.tensor([3, 5, 1, 4, 0, 2])
    permuted = candidate_set(*teacher, weight[permutation])
    assert torch.equal(permuted["candidates"], candidate["candidates"][:, permutation])
    student = torch.randn(7, 160, generator=generator, dtype=torch.float64).requires_grad_()
    original_loss = set_loss(student, weight, candidate["candidates"])
    permuted_loss = set_loss(student, weight[permutation], candidate["candidates"][:, permutation])
    assert torch.allclose(original_loss["weighted_loss"], permuted_loss["weighted_loss"], atol=1e-14, rtol=0)
    original_loss["weighted_loss"].backward()
    assert bool(torch.isfinite(student.grad).all())
    assert weight.grad is None and all(value.grad is None for value in teacher)
    # W and all teacher increments remain without gradients; student derivatives
    # exist even when student argmax disagrees with the supplied candidate set.
    active_student = (one + .2 * F.one_hot(torch.tensor([1]), 160).to(torch.float64)).requires_grad_()
    basis_weight.requires_grad_()
    singleton = torch.zeros(1, 6, dtype=torch.bool); singleton[0, 1] = True
    active = set_loss(active_student, basis_weight, singleton)
    active["weighted_loss"].backward()
    active_norm = float(active_student.grad.norm().item())
    assert active_norm > 0 and bool(torch.isfinite(active_student.grad).all())
    assert basis_weight.grad is None and weight.grad is None and all(value.grad is None for value in teacher)
    # Full set and fulfilled .9 mass both stop gradients exactly.
    for full, probability in ((True, .9), (True, 1.0), (False, .9)):
        stopped_input = (one + .01 * F.one_hot(torch.tensor([1]), 160).to(torch.float64)).requires_grad_()
        mask = torch.ones(1, 6, dtype=torch.bool) if full else torch.tensor([[True, False, False, False, False, False]])
        stopped = set_loss(stopped_input, basis_weight, mask, probability=probability)
        stopped["weighted_loss"].backward()
        assert stopped["weighted_loss"].item() == 0.0 and bool((stopped_input.grad == 0).all())
    # Eligibility uses the full batch denominator, never eligible/active count.
    batch = active_student.detach().repeat(4, 1).requires_grad_()
    batch_candidates = singleton.repeat(4, 1)
    eligibility = torch.tensor([True, False, False, True])
    masked = set_loss(batch, basis_weight, batch_candidates, eligible_mask=eligibility)
    expected = .16 * masked["tensors"]["packet_loss"][eligibility].sum() / 4
    assert torch.equal(masked["weighted_loss"], expected)
    assert masked["summary"]["eligible_count"] == masked["summary"]["active_count"] == 2
    assert masked["summary"]["eligible_coverage"] == .5
    rejected_input = batch.detach().clone().requires_grad_()
    rejected = set_loss(rejected_input, basis_weight, batch_candidates, eligible_mask=torch.zeros(4, dtype=torch.bool))
    rejected["weighted_loss"].backward()
    assert rejected["weighted_loss"].item() == 0 and bool((rejected_input.grad == 0).all())
    # Large cosine-logit magnitude stays in log space, without mass underflow.
    extreme = (one + .6 * F.one_hot(torch.tensor([1]), 160).to(torch.float64)).requires_grad_()
    opposite = torch.tensor([[False, False, True, False, False, False]])
    extreme_result = set_loss(extreme, basis_weight, opposite, scale=10000.0)
    extreme_result["weighted_loss"].backward()
    assert bool(torch.isfinite(extreme_result["weighted_loss"])) and bool(torch.isfinite(extreme.grad).all())
    fp32_teacher = [value.detach().to(torch.float32) for value in teacher]
    fp32_weight = weight.detach().to(torch.float32).requires_grad_()
    fp32_candidates = candidate_set(*fp32_teacher, fp32_weight)
    fp32_nominal = F.linear(F.normalize(sum(fp32_teacher), dim=1, eps=1e-4), F.normalize(fp32_weight.detach(), dim=1, eps=1e-4)).argmax(dim=1)
    assert bool(fp32_candidates["candidates"][torch.arange(7), fp32_nominal].all())
    extreme32 = extreme.detach().to(torch.float32).requires_grad_()
    basis32 = basis_weight.detach().to(torch.float32).requires_grad_()
    extreme32_loss = set_loss(extreme32, basis32, opposite, scale=10000.0)
    extreme32_loss["weighted_loss"].backward()
    assert bool(torch.isfinite(extreme32_loss["weighted_loss"])) and bool(torch.isfinite(extreme32.grad).all()) and basis32.grad is None
    for data in (torch.zeros(0, 160, dtype=torch.float64),):
        empty = candidate_set(data, data, data, basis_weight)
        assert empty["candidates"].shape == (0, 6)
        empty_student = data.clone().requires_grad_()
        empty_loss = set_loss(empty_student, basis_weight, empty["candidates"])
        empty_loss["weighted_loss"].backward()
        assert empty_loss["weighted_loss"].item() == 0 and empty_student.grad is not None
    # Numeric extremes of finite teacher inputs still yield nonempty sets.
    huge = torch.full((1, 160), 1e308, dtype=torch.float64)
    assert bool(candidate_set(huge, huge, huge, basis_weight)["candidates"].any(dim=1).all())
    for beta in (-.1, 1.0, float("inf")):
        _raises(lambda beta=beta: candidate_set(*teacher, weight, beta=beta))
    _raises(lambda: candidate_set(*teacher, weight, tolerance=-1))
    _raises(lambda: candidate_set(*teacher, weight, normalize_eps=0))
    _raises(lambda: candidate_set(teacher[0][:, :159], teacher[1], teacher[2], weight))
    bad = teacher[0].detach().clone(); bad[0, 0] = float("nan")
    _raises(lambda: candidate_set(bad, teacher[1], teacher[2], weight))
    bad_weight = weight.detach().clone(); bad_weight[0, 0] = float("inf")
    _raises(lambda: candidate_set(*teacher, bad_weight))
    _raises(lambda: set_loss(active_student, basis_weight, torch.zeros(1, 6, dtype=torch.bool)))
    _raises(lambda: set_loss(active_student, basis_weight, singleton.to(torch.int64)))
    _raises(lambda: set_loss(active_student, basis_weight, torch.ones(2, 6, dtype=torch.bool)))
    _raises(lambda: set_loss(active_student, basis_weight, singleton, eligible_mask=torch.ones(1, dtype=torch.int64)))
    _raises(lambda: set_loss(active_student, basis_weight, singleton, eligible_mask=torch.ones(1, 1, dtype=torch.bool)))
    for probability in (0, 1.1, float("nan")):
        _raises(lambda probability=probability: set_loss(active_student, basis_weight, singleton, probability=probability))
    _raises(lambda: set_loss(active_student, basis_weight, singleton, coefficient=-1))
    _raises(lambda: set_loss(active_student, basis_weight, singleton, scale=0))
    _raises(lambda: set_loss(active_student, basis_weight, singleton, normalize_eps=-1))
    _raises(lambda: set_loss(active_student.to(torch.float32), basis_weight, singleton))
    bad_student = active_student.detach().clone(); bad_student[0, 0] = float("inf")
    _raises(lambda: set_loss(bad_student, basis_weight, singleton))
    assert _same_rng(rng_before, _rng())
    for result in (candidate, active, masked, extreme_result):
        json.dumps(result["summary"], allow_nan=False)
    return {
        "status": "VERIFIED", "environment": {"torch": torch.__version__, "device": "cpu", "dtype": "float64 + float32 default-head/extreme-logit probes"},
        "checks": {"independent_rectangle_corner_maxima": True, "nominal_winner_retained": True,
                   "beta0_ties": True, "zero_full_ambiguity": True, "raw_tolerance_units_preserved": True,
                   "conservative_not_exact_example": True, "class_permutation_equivariance": True,
                   "teacher_W_gradients_None": True, "active_student_derivative": True,
                   "complete_set_and_high_mass_derivative_zero": True, "full_batch_eligibility_denominator": True,
                   "stable_extreme_logits": True, "empty_batch_zero": True,
                   "invalid_shapes_masks_finite_constants": True, "global_RNG_unchanged": True},
        "active_student_gradient_norm": active_norm,
        "scientific_status": "SYNTHETIC_PRIMITIVE_CHECK_ONLY_NOT_REAL_SOURCE_OR_EFFICACY",
    }


if __name__ == "__main__":
    argparse.ArgumentParser().parse_args()
    print(json.dumps(run_checks(), ensure_ascii=False, allow_nan=False, indent=2))
