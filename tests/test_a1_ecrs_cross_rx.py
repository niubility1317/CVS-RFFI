import sys
from pathlib import Path
from unittest.mock import patch

import pytest
import torch
from torch.nn import functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "code"))
from cvsrffi.a1_ecrs_cross_rx import cross_rx_triplet_loss, labeled_cross_rx_objective

DEVICES = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])


def literal_triplets(z, labels, rx, day, view, mask):
    unit = F.normalize(z.float(), dim=-1)
    dist = 1 - unit @ unit.T
    valid = mask & (labels >= 0) & (rx >= 0) & (day >= 0) & (view >= 0)
    losses = []
    for i in range(len(z)):
        terms = []
        for j in range(len(z)):
            for k in range(len(z)):
                if not (valid[i] and valid[j] and valid[k]):
                    continue
                if labels[i] != labels[j] or rx[i] == rx[j]:
                    continue
                if labels[i] == labels[k] or any(m[i] != m[k] for m in (rx, day, view)):
                    continue
                terms.append(F.relu(.2 + dist[i, j] - dist[i, k]))
        if terms:
            losses.append(torch.stack(terms).mean())
    return (torch.stack(losses).mean() if losses else z.detach().new_zeros(())), len(losses)


def fixture(device, seed=1):
    gen = torch.Generator().manual_seed(seed)
    z = torch.randn(12, 7, generator=gen).to(device).requires_grad_()
    labels = torch.arange(12, device=device) % 3
    rx = (torch.arange(12, device=device) // 3) % 2
    zeros = torch.zeros(12, device=device, dtype=torch.long)
    return z, labels, rx, zeros.clone(), zeros.clone()


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("seed", [1, 29, 392005])
def test_sorted_prefix_matches_literal_all_triplets_and_gradients(device, seed):
    z, labels, rx, day, view = fixture(device, seed)
    mask = torch.ones(12, device=device, dtype=torch.bool)
    mask[0] = False
    labels[1], rx[2], day[3], view[4] = -1, -1, -1, -1
    expected, expected_count = literal_triplets(z, labels, rx, day, view, mask)
    actual, actual_count = cross_rx_triplet_loss(z, labels, rx, day, view, mask)
    assert actual_count == expected_count and actual_count > 0
    torch.testing.assert_close(actual, expected, atol=3e-7, rtol=1e-6)
    ga = torch.autograd.grad(actual, z, retain_graph=True)[0]
    ge = torch.autograd.grad(expected, z)[0]
    torch.testing.assert_close(ga, ge, atol=3e-7, rtol=1e-5)
    assert torch.equal(ga[:5], torch.zeros_like(ga[:5]))


@pytest.mark.parametrize("device", DEVICES)
def test_amp_is_internally_fp32(device):
    z, labels, rx, day, view = fixture(device)
    full, count = cross_rx_triplet_loss(z, labels, rx, day, view)
    with torch.autocast(device_type=device, dtype=torch.float16 if device == "cuda" else torch.bfloat16):
        amp, amp_count = cross_rx_triplet_loss(z, labels, rx, day, view)
    assert amp.dtype == torch.float32 and amp_count == count
    assert torch.equal(amp, full)
    ga = torch.autograd.grad(amp, z, retain_graph=True)[0]
    gf = torch.autograd.grad(full, z)[0]
    assert torch.equal(ga, gf)


@pytest.mark.parametrize("kind", ["empty", "one_tx", "one_rx", "all_masked", "invalid_meta"])
def test_no_legal_anchor_detaches_zero(kind):
    z, labels, rx, day, view = fixture("cpu")
    mask = torch.ones(12, dtype=torch.bool)
    if kind == "empty":
        z, labels, rx, day, view, mask = (v[:0] for v in (z, labels, rx, day, view, mask))
    elif kind == "one_tx":
        labels.zero_()
    elif kind == "one_rx":
        rx.zero_()
    elif kind == "all_masked":
        mask.zero_()
    else:
        day.fill_(-1)
    loss, count = cross_rx_triplet_loss(z, labels, rx, day, view, mask)
    assert count == 0 and loss.item() == 0 and not loss.requires_grad


def test_empty_objective_cannot_move_isolated_adamw_parameters_with_old_momentum():
    branch = torch.nn.Parameter(torch.ones(12, 7))
    unrelated = torch.nn.Parameter(torch.ones(()))
    optimizer = torch.optim.AdamW([branch, unrelated], lr=.1, weight_decay=.1)
    branch.square().mean().backward()
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    before = branch.detach().clone()
    step_before = optimizer.state[branch]["step"].clone()
    labels = torch.arange(12) % 3
    rx = torch.zeros(12, dtype=torch.long)
    extra, _ = labeled_cross_rx_objective(branch, labels, rx, rx)
    (unrelated.square() + extra).backward()
    assert branch.grad is None
    optimizer.step()
    assert torch.equal(branch, before)
    assert torch.equal(optimizer.state[branch]["step"], step_before)


def test_clean_leo_only_counts_and_uses_actual_views():
    z, labels, rx, day, _ = fixture("cpu")
    clean, clean_logs = labeled_cross_rx_objective(z, labels, rx, day)
    skipped, skipped_logs = labeled_cross_rx_objective(
        z, labels, rx, day, z_leo=torch.full_like(z, float("nan")),
        leo_applied=False, scope="clean_leo")
    assert torch.equal(clean, skipped)
    assert skipped_logs["leo_count"] == 0 and skipped_logs["clean_count"] == len(z)
    leo = z.roll(1, 0)
    actual, logs = labeled_cross_rx_objective(z, labels, rx, day, z_leo=leo,
                                            leo_applied=True, scope="clean_leo")
    raw, count = cross_rx_triplet_loss(torch.cat((z, leo)), labels.repeat(2),
                                      rx.repeat(2), day.repeat(2),
                                      torch.cat((torch.zeros_like(labels), torch.ones_like(labels))))
    assert torch.equal(actual, raw * .05)
    assert logs["valid_anchors"] == count and logs["leo_count"] == len(z)
    assert logs["configured"] == logs["executed"] == 1.
    assert torch.equal(logs["weighted_loss"], actual.detach())


def test_disabled_preserves_raw_gradients_rng_and_never_calls_pair_math():
    z, labels, rx, day, _ = fixture("cpu")
    state = torch.random.get_rng_state().clone()
    with patch("cvsrffi.a1_ecrs_cross_rx.cross_rx_triplet_loss", side_effect=AssertionError):
        extra, logs = labeled_cross_rx_objective(z, labels, rx, day, weight=0.)
    assert torch.equal(state, torch.random.get_rng_state())
    assert not extra.requires_grad and logs["executed"] == logs["configured"] == 0.
    baseline = z.square().mean()
    ga = torch.autograd.grad(baseline + extra, z, retain_graph=True)[0]
    gb = torch.autograd.grad(baseline, z)[0]
    assert torch.equal(ga, gb)


@pytest.mark.parametrize("field", ["labels", "rx", "day"])
def test_missing_and_misaligned_metadata_rejected(field):
    z, labels, rx, day, _ = fixture("cpu")
    values = dict(labels=labels, rx=rx, day=day)
    values[field] = None
    with pytest.raises(ValueError, match="metadata"):
        labeled_cross_rx_objective(z, **values)
    values[field] = torch.zeros(11, dtype=torch.long)
    with pytest.raises(ValueError, match="shape"):
        labeled_cross_rx_objective(z, **values)


def test_applied_leo_requires_aligned_representation_and_class_mask():
    z, labels, rx, day, _ = fixture("cpu")
    with pytest.raises(ValueError):
        labeled_cross_rx_objective(z, labels, rx, day, leo_applied=True, scope="clean_leo")
    with pytest.raises(ValueError):
        labeled_cross_rx_objective(z, labels, rx, day, z_leo=z[:2], leo_applied=True, scope="clean_leo")
    labels[:] = 99
    loss, logs = labeled_cross_rx_objective(z, labels, rx, day)
    assert not loss.requires_grad and logs["valid_anchors"] == 0
    assert logs["configured"] == 1. and logs["executed"] == 0.
