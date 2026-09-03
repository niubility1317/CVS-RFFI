from __future__ import annotations

import torch

from cvsrffi.daot_unlabeled_trust import classify_unlabeled_trust, continuous_unlabeled_trust


def test_unlabeled_trust_is_monotone_in_recoverability_and_disagreement() -> None:
    trust = continuous_unlabeled_trust(
        recoverability=torch.tensor([0.9, 0.2]),
        view_js=torch.tensor([0.01, 0.40]),
        temporal_inconsistency=torch.tensor([0.02, 0.50]),
        prototype_margin=torch.tensor([0.8, 0.1]),
    )

    assert float(trust[0]) > float(trust[1])
    assert bool(((trust >= 0.0) & (trust <= 1.0)).all())


def test_unlabeled_three_state_routing_and_group_quota_are_explicit() -> None:
    result = classify_unlabeled_trust(
        trust=torch.tensor([0.95, 0.85, 0.55, 0.10]),
        predicted_class=torch.tensor([0, 0, 1, 1]),
        receiver_bin=torch.tensor([1, 1, 1, 2]),
        severity_bin=torch.tensor([0, 0, 1, 1]),
        core_threshold=0.80,
        irrecoverable_threshold=0.20,
        max_core_per_group=1,
    )

    assert result["core"].tolist() == [True, False, False, False]
    assert result["ambiguous"].tolist() == [False, True, True, False]
    assert result["irrecoverable"].tolist() == [False, False, False, True]
