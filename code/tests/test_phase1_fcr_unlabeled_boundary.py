from __future__ import annotations

from copy import deepcopy

import pytest
import torch
from torch import nn

from cvsrffi.phase1_fcr_schedule import permission_for_role
from cvsrffi.phase1_fcr_types import FCRPairBatch
from train import fcr_readonly_validation_forward, validate_fcr_pair_for_role


def _pair(*, labels: torch.Tensor, label_mask: torch.Tensor, fingerprint_valid: bool = False) -> FCRPairBatch:
    batch_size = int(labels.numel())
    invalid = torch.full((batch_size,), -1, dtype=torch.long)
    fingerprint_index = torch.roll(torch.arange(batch_size), shifts=1) if fingerprint_valid else invalid.clone()
    return FCRPairBatch(
        clean_iq=torch.zeros(batch_size, 2, 8),
        leo_iq=torch.zeros(batch_size, 2, 8),
        labels=labels,
        label_mask=label_mask,
        receiver_id=torch.zeros(batch_size, dtype=torch.long),
        day_id=torch.zeros(batch_size, dtype=torch.long),
        nuisance=torch.zeros(batch_size, 3),
        nuisance_valid=torch.zeros(batch_size, 3, dtype=torch.bool),
        physical_sample_id=tuple(f"opaque:{i}" for i in range(batch_size)),
        pair_id=tuple(f"opaque:{i}" for i in range(batch_size)),
        clean_crop_offset=torch.zeros(batch_size, dtype=torch.long),
        leo_crop_offset=torch.zeros(batch_size, dtype=torch.long),
        nuisance_pair_index=torch.arange(batch_size),
        content_pair_index=invalid,
        fingerprint_pair_index=fingerprint_index,
        pair_valid_mask={
            "nuisance": torch.ones(batch_size, dtype=torch.bool),
            "content": torch.zeros(batch_size, dtype=torch.bool),
            "fingerprint": torch.full((batch_size,), fingerprint_valid, dtype=torch.bool),
        },
    )


def test_role_permissions_are_exhaustive_and_query_is_unreachable() -> None:
    labeled = permission_for_role("L_s")
    unlabeled = permission_for_role("U_s")
    validation = permission_for_role("V")

    assert labeled.optimizer_step is True
    assert {"id", "transplant"} <= labeled.allowed
    assert unlabeled.optimizer_step is True
    assert unlabeled.allowed == frozenset({"self", "swap", "shared", "latent_cycle", "eta", "phys"})
    assert validation.optimizer_step is False
    assert validation.allowed == frozenset()
    with pytest.raises(ValueError, match="query"):
        permission_for_role("query")


def test_unlabeled_pair_requires_minus_one_labels_and_cannot_carry_transplant_capability() -> None:
    legal = _pair(labels=torch.tensor([-1, -1]), label_mask=torch.tensor([False, False]))
    validate_fcr_pair_for_role(legal, "U_s")

    leaked_label = _pair(labels=torch.tensor([0, -1]), label_mask=torch.tensor([False, False]))
    with pytest.raises(ValueError, match="labels=-1"):
        validate_fcr_pair_for_role(leaked_label, "U_s")

    leaked_mask = _pair(labels=torch.tensor([-1, -1]), label_mask=torch.tensor([True, False]))
    with pytest.raises(ValueError, match="label_mask"):
        validate_fcr_pair_for_role(leaked_mask, "U_s")

    transplant = _pair(
        labels=torch.tensor([-1, -1]),
        label_mask=torch.tensor([False, False]),
        fingerprint_valid=True,
    )
    with pytest.raises(ValueError, match="fingerprint"):
        validate_fcr_pair_for_role(transplant, "U_s")


class _ValidationModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.bn = nn.BatchNorm1d(2)
        self.linear = nn.Linear(2, 1)

    def forward(self, x: torch.Tensor, **_kwargs):
        return {"value": self.linear(self.bn(x)).sum()}


def test_validation_forward_is_read_only_and_restores_training_mode() -> None:
    model = _ValidationModel()
    model.train()
    before = deepcopy(model.state_dict())

    output = fcr_readonly_validation_forward(model, torch.randn(8, 2))

    assert model.training is True
    assert output["value"].requires_grad is False
    after = model.state_dict()
    assert set(before) == set(after)
    for key in before:
        torch.testing.assert_close(after[key], before[key], rtol=0.0, atol=0.0)
