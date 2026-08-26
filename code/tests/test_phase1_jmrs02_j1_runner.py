from argparse import Namespace

import pytest

from audit_phase1_jmrs02_j1 import validate_j1_args


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
