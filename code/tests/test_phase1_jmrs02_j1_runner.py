from argparse import Namespace

import pytest

import torch

from audit_phase1_jmrs02_j1 import smoke_bypass_audit, validate_j1_args


def _args(**overrides):
    values = dict(
        target_or_query_access=False,
        train_role="L_s",
        select_role="V_select",
        cal_role="V_cal",
        audit_role="V_select",
        rows="B0,RZ0,RZ1,RX1,D1P,P0",
    )
    values.update(overrides)
    return Namespace(**values)


def test_formal_j1_args_are_source_only_and_single_module():
    assert validate_j1_args(_args()) == ("B0", "RZ0", "RZ1", "RX1", "D1P", "P0")


def test_target_access_and_joint_rows_are_rejected():
    with pytest.raises(ValueError, match="target/query"):
        validate_j1_args(_args(target_or_query_access=True))
    with pytest.raises(ValueError):
        validate_j1_args(_args(rows="B0,RDP"))


def test_rx1_smoke_uses_decision_parity_while_residual_rows_keep_logit_parity():
    base = torch.tensor([[2.0, 1.0], [0.5, 0.7]])
    numerically_shifted = base + torch.tensor([[1e-3, -1e-3], [-1e-3, 1e-3]])
    rx1 = smoke_bypass_audit("RX1", numerically_shifted, base)
    rz1 = smoke_bypass_audit("RZ1", numerically_shifted, base)
    assert rx1["epoch0_bypass_pass"] is True
    assert rx1["prediction_agreement"] == 1.0
    assert rx1["max_abs_logit_delta"] > 0.0
    assert rz1["epoch0_bypass_pass"] is False
