import argparse
import sys
from pathlib import Path

import pytest
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[2]
CODE_ROOT = PROJECT_ROOT / "code"
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

from cvsrffi.ccoi_pa import PAChallengeEncoder, challenge_pretrain_losses  # noqa: E402
from train_phase1_ccoi_pa import build_arg_parser, validate_source_roles  # noqa: E402


def _valid_args():
    return argparse.Namespace(
        phase1_source_role_protocol="l_s_u_s_v_cal_v_select",
        split_mode="tx_rx_day_1_7_2",
        labeled_ratio=0.07,
        unlabeled_ratio=0.63,
        source_cal_ratio=0.15,
        source_select_ratio=0.15,
    )


def _valid_split():
    return {
        "rho_label": 0.10,
        "source_role_ratios": {"L_s": 0.07, "U_s": 0.63, "V_cal": 0.15, "V_select": 0.15},
    }


def test_current_phase1_roles_pass_and_rho_above_limit_fails():
    validate_source_roles(_valid_args(), _valid_split())
    bad = _valid_split()
    bad["rho_label"] = 0.1001
    with pytest.raises(ValueError, match="rho_label"):
        validate_source_roles(_valid_args(), bad)


def test_runner_parser_has_no_target_or_query_surface():
    parser = build_arg_parser()
    option_strings = {option for action in parser._actions for option in action.option_strings}

    assert not any("target" in option or "query" in option or "truth" in option for option in option_strings)


def test_unlabeled_q_pretraining_cannot_consume_tx_probe_labels():
    encoder = PAChallengeEncoder(num_tx=4, num_rx=2)
    clean = torch.randn(3, 2, 256)
    satellite = clean + 0.01 * torch.randn_like(clean)
    receiver = torch.tensor([0, 1, 0])

    losses = challenge_pretrain_losses(
        encoder,
        clean,
        satellite,
        tx_labels=None,
        rx_labels=receiver,
    )

    assert losses["tx_adversarial"].item() == 0.0
    assert losses["rx_adversarial"].item() > 0.0
