from __future__ import annotations

import pytest
import torch

from cvsrffi.slow_fast_adapter import SlowFastCandidate, apply_slow_fast
from cvsrffi.slow_fast_cache import GroundFeatureCache
from cvsrffi.slow_fast_objectives import (
    frozen_prototype_ce,
    smooth_class_floor_loss,
    trust_region_loss,
)
from cvsrffi.slow_fast_phase15 import (
    fit_common_shift_basis,
    train_slow_fast_basis,
)


def _cache(*, forbidden_role: bool = False) -> tuple[GroundFeatureCache, torch.Tensor]:
    prototypes = torch.zeros(2, 16)
    prototypes[0, 14] = 1.0
    prototypes[1, 15] = 1.0
    shifts = torch.eye(16)[:4] * 0.8
    features = []
    labels = []
    receivers = []
    days = []
    scenes = []
    physical_ids = []
    views = []
    roles = []
    for domain_i, shift in enumerate(shifts):
        for class_id in range(2):
            for sample_i in range(2):
                physical_id = f"d{domain_i}-c{class_id}-s{sample_i}"
                for view, view_scale in (("clean", 0.0), ("leo_clear_weak", 1.0)):
                    features.append(prototypes[class_id] + view_scale * shift)
                    labels.append(class_id)
                    receivers.append(domain_i)
                    days.append(0)
                    scenes.append(view)
                    physical_ids.append(physical_id)
                    views.append(view)
                    roles.append("target" if forbidden_role else "L_s")
    return (
        GroundFeatureCache(
            features=torch.stack(features),
            labels=torch.tensor(labels),
            receivers=tuple(receivers),
            days=tuple(days),
            scenes=tuple(scenes),
            physical_sample_ids=tuple(physical_ids),
            views=tuple(views),
            roles=tuple(roles),
        ),
        prototypes,
    )


def test_ground_cache_rejects_target_or_query_roles() -> None:
    with pytest.raises(ValueError, match="L_s"):
        _cache(forbidden_role=True)


def test_class_centered_common_basis_recovers_the_four_domain_directions() -> None:
    cache, prototypes = _cache()

    basis = fit_common_shift_basis(cache, prototypes, rank=4)

    assert basis.shape == (16, 4)
    shifts = torch.eye(16)[:4] * 0.8
    residual = shifts - (shifts @ basis) @ basis.transpose(0, 1)
    assert torch.linalg.vector_norm(residual).item() < 1.0e-4


def test_floor_and_interval_trust_objectives_have_hand_checked_boundaries() -> None:
    per_sample = torch.tensor([0.2, 0.4, 1.0, 1.2])
    labels = torch.tensor([0, 0, 1, 1])
    floor = smooth_class_floor_loss(per_sample, labels, temperature=0.1)
    assert floor.item() > 1.0

    base = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    inside = base + 0.05
    outside = base + 0.5
    assert trust_region_loss(inside, base, max_relative_move=0.2).item() == 0.0
    assert trust_region_loss(outside, base, max_relative_move=0.2).item() > 0.0


@pytest.mark.parametrize(
    "candidate",
    [SlowFastCandidate.FAST_FILM_R8, SlowFastCandidate.FAST_LOWRANK_R8],
)
def test_phase15_training_reduces_frozen_prototype_loss_without_changing_prototypes(
    candidate: SlowFastCandidate,
) -> None:
    cache, prototypes = _cache()
    prototype_before = prototypes.clone()
    initial = frozen_prototype_ce(cache.features, cache.labels, prototypes, scale=8.0)

    state, audit = train_slow_fast_basis(
        cache,
        prototypes,
        candidate=candidate,
        steps=80,
        learning_rate=0.03,
        seed=17,
    )
    final = frozen_prototype_ce(
        apply_slow_fast(cache.features, state), cache.labels, prototypes, scale=8.0
    )

    assert audit["steps"] == 80
    assert audit["meta_steps"] == 80
    assert sum(audit["episode_k_counts"].values()) == 80
    assert audit["episode_k_counts"]["1"] == 80
    assert audit["stage1_final_loss"] <= audit["initial_loss"]
    assert torch.isfinite(torch.tensor(audit["final_loss"]))
    assert final.item() < initial.item()
    assert torch.equal(prototypes, prototype_before)
