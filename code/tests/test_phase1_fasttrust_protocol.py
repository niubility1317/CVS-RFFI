from collections import Counter
from types import SimpleNamespace

import pytest
import torch

from SSDG.train_ssdg import _partition_source_validation_roles
from cvsrffi.phase2_prototypes import (
    Phase1CalibrationError,
    audit_identity_feature_contract,
    extract_endpoint_calibration_features,
)


class _IndexDataset:
    def __init__(self, rows):
        self.index = list(rows)


def _record(tx, rx, day, sig):
    return SimpleNamespace(
        tx_i=tx,
        rx_i=rx,
        day_i=day,
        eq_i=0,
        sig_i=sig,
        base_index=10_000 + sig,
    )


def test_validation_roles_are_class_complete_receiver_day_stratified_and_disjoint():
    rows = [
        _record(tx, rx, day, sig=tx * 100 + rx * 20 + day * 8 + offset)
        for tx in (0, 1)
        for rx in (0, 1)
        for day in (0, 1)
        for offset in range(4)
    ]
    dataset = _IndexDataset(rows)

    v_cal, v_select, receipt = _partition_source_validation_roles(
        dataset,
        list(range(len(rows))),
        cal_fraction=0.5,
        min_class_samples=4,
    )

    assert set(v_cal).isdisjoint(v_select)
    assert set(v_cal) | set(v_select) == set(range(len(rows)))
    assert Counter(rows[index].tx_i for index in v_cal) == {0: 8, 1: 8}
    assert Counter(rows[index].tx_i for index in v_select) == {0: 8, 1: 8}
    assert receipt["per_role_per_class"]["V_cal"] == {"0": 8, "1": 8}
    assert receipt["per_role_per_class"]["V_select"] == {"0": 8, "1": 8}
    assert receipt["physical_id_overlap_count"] == 0


def test_validation_roles_fail_closed_when_a_class_cannot_supply_both_roles():
    rows = [_record(0, 0, 0, sig=index) for index in range(7)]

    with pytest.raises(ValueError, match="INSUFFICIENT_CLASS_SAMPLES"):
        _partition_source_validation_roles(
            _IndexDataset(rows),
            list(range(len(rows))),
            cal_fraction=0.5,
            min_class_samples=4,
        )


def _valid_contract_inputs():
    labels = torch.tensor([0] * 4 + [1] * 4)
    z_id = torch.tensor(
        [[1.0, 0.1], [0.9, 0.2], [0.8, 0.1], [0.9, 0.3]]
        + [[0.1, 1.0], [0.2, 0.9], [0.1, 0.8], [0.3, 0.9]]
    )
    feat_joint = z_id + torch.tensor([0.05, 0.02])
    logits = torch.tensor([[4.0, 0.0]] * 4 + [[0.0, 4.0]] * 4)
    domains = torch.tensor([0, 1, 0, 1, 0, 1, 0, 1])
    return z_id, feat_joint, labels, domains, logits


@pytest.mark.parametrize(
    ("mutation", "code"),
    (
        ("missing_class", "MISSING_CLASS_IN_V_CAL"),
        ("insufficient", "INSUFFICIENT_CLASS_SAMPLES"),
        ("zero", "ZERO_DIRECTION_FEATURE"),
        ("nonfinite", "NONFINITE_FEATURE"),
        ("class_order", "CLASS_ORDER_MISMATCH"),
    ),
)
def test_identity_feature_contract_classifies_failures_without_aliasing_them(mutation, code):
    z_id, feat_joint, labels, domains, logits = _valid_contract_inputs()
    expected_classes = 2
    min_class_samples = 4
    if mutation == "missing_class":
        expected_classes = 3
        logits = torch.cat([logits, torch.zeros(logits.size(0), 1)], dim=1)
    elif mutation == "insufficient":
        labels = torch.tensor([0] * 6 + [1] * 2)
    elif mutation == "zero":
        z_id[0].zero_()
    elif mutation == "nonfinite":
        feat_joint[0, 0] = float("nan")
    elif mutation == "class_order":
        logits = torch.zeros(labels.numel(), 3)

    with pytest.raises(Phase1CalibrationError) as caught:
        audit_identity_feature_contract(
            z_id,
            feat_joint,
            labels,
            domains,
            logits,
            expected_classes=expected_classes,
            min_class_samples=min_class_samples,
        )

    assert caught.value.code == code
    assert caught.value.details["class_count"] == expected_classes


def test_identity_feature_contract_reports_both_spaces_and_class_order():
    z_id, feat_joint, labels, domains, logits = _valid_contract_inputs()

    audit = audit_identity_feature_contract(
        z_id,
        feat_joint,
        labels,
        domains,
        logits,
        expected_classes=2,
        min_class_samples=4,
    )

    assert audit["status"] == "PASS"
    assert audit["feature_key_contract_pass"] is True
    assert audit["class_coverage_pass"] is True
    assert audit["spaces"]["z_id"]["zero_count"] == 0
    assert audit["spaces"]["feat_joint"]["zero_count"] == 0
    assert audit["logit_class_order"] == [0, 1]


class _DualSpaceModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.anchor = torch.nn.Parameter(torch.ones(()))

    def forward(self, x, **_kwargs):
        z_id = torch.stack([x[:, 0] + 1.0, x[:, 0] * 0.0 + 0.5], dim=1)
        feat_joint = z_id + torch.tensor([0.25, 0.0], device=x.device)
        return {
            "z_id": z_id,
            "id_feat_joint": feat_joint,
            "tx_logits": feat_joint,
        }


def test_endpoint_extraction_requires_and_returns_both_identity_spaces():
    loader = [
        (
            torch.tensor([[0.0], [1.0]]),
            torch.tensor([0, 1]),
            torch.tensor([0, 1]),
        )
    ]

    extracted = extract_endpoint_calibration_features(
        _DualSpaceModel(),
        loader,
        device=torch.device("cpu"),
        feature_key="z_id",
        require_identity_contract=True,
    )

    assert extracted["feature_key"] == "z_id"
    assert extracted["identity_feature_contract_required"] is True
    assert torch.allclose(extracted["features"], extracted["z_id_features"])
    assert not torch.allclose(extracted["z_id_features"], extracted["feat_joint_features"])
