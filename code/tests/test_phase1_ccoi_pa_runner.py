import argparse
import sys
from pathlib import Path

import pytest
import torch
from torch import nn


PROJECT_ROOT = Path(__file__).resolve().parents[2]
CODE_ROOT = PROJECT_ROOT / "code"
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

from train_phase1_ccoi_pa import (  # noqa: E402
    FrozenCore90CCOI,
    build_matrix_specs,
    freeze_base_model,
    paired_challenge_batch,
    validate_output_root,
    validate_source_roles,
)
from cvsrffi.ccoi_pa import CCOIPASidecar  # noqa: E402


class _TinyBase(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.tensor(1.0))
        self.calls = 0

    def forward(self, x, **kwargs):
        del kwargs
        self.calls += 1
        logits = torch.stack((x.mean(dim=(1, 2)), -x.mean(dim=(1, 2))), dim=1) * self.weight
        return {
            "tx_logits": logits,
            "z_id": logits,
            "z_dom": logits,
            "dom_logits": logits,
            "aux_id": {"pa_token_map": x[:, :1]},
        }


def test_runner_requires_current_source_role_ratios():
    args = argparse.Namespace(
        phase1_source_role_protocol="l_s_u_s_v_cal_v_select",
        split_mode="tx_rx_day_1_7_2",
        labeled_ratio=0.08,
        unlabeled_ratio=0.62,
        source_cal_ratio=0.15,
        source_select_ratio=0.15,
    )
    with pytest.raises(ValueError, match="0.07/0.63/0.15/0.15"):
        validate_source_roles(args, {"rho_label": 0.08})


def test_c0_has_no_trainable_sidecar_and_c1_to_c4_share_capacity():
    specs = build_matrix_specs()

    assert specs["C0"].train_sidecar is False
    assert len({spec.parameter_profile for name, spec in specs.items() if name != "C0"}) == 1
    assert specs["C1"].conditioned is False
    assert specs["C2"].challenge_pairs is True
    assert specs["C3"].use_did is True
    assert specs["C4"].use_holdout is True


def test_c0_and_zero_fusion_reproduce_frozen_base_logits():
    base = freeze_base_model(_TinyBase())
    model = FrozenCore90CCOI(base, sidecar=None, row="C0", fusion_alpha=0.15)
    x = torch.randn(3, 2, 64)

    out = model(x, return_aux=True)

    torch.testing.assert_close(out["tx_logits"], base(x)["tx_logits"])
    assert all(not parameter.requires_grad for parameter in base.parameters())


def test_output_root_is_immutable(tmp_path):
    available = tmp_path / "new-run"
    validate_output_root(available)
    available.mkdir()
    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        validate_output_root(available)


def test_sidecar_training_batch_contains_clean_and_source_satellite_views():
    x = torch.randn(3, 2, 64)
    satellite = x + 1.0
    y = torch.tensor([0, 1, 2])
    domain = torch.tensor([4, 5, 6])

    paired_x, paired_y, paired_domain = paired_challenge_batch(x, satellite, y, domain)

    torch.testing.assert_close(paired_x[:3], x)
    torch.testing.assert_close(paired_x[3:], satellite)
    torch.testing.assert_close(paired_y, torch.cat((y, y)))
    torch.testing.assert_close(paired_domain, torch.cat((domain, domain)))


def test_c4_holdout_uses_separate_frozen_base_forwards():
    base = freeze_base_model(_TinyBase())
    sidecar = CCOIPASidecar(pa_channels=1, num_classes=2)
    sidecar.freeze_challenge_encoder()
    model = FrozenCore90CCOI(base, sidecar=sidecar, row="C4", fusion_alpha=0.15)

    out = model(torch.randn(2, 2, 256), return_aux=True)

    assert base.calls == 3
    assert out["ccoi"]["heldout_target"].shape[:2] == (2, 1)
