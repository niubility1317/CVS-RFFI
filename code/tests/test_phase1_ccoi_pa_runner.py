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
    _meta_value,
    accumulate_pair_audit,
    build_matrix_specs,
    calibrated_fusion_scale,
    fuse_logits,
    freeze_base_model,
    paired_challenge_batch,
    select_fusion_alpha,
    validate_output_root,
    validate_source_roles,
)
from cvsrffi.ccoi_losses import CCOILossOutput  # noqa: E402
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


class _FixedSidecar(nn.Module):
    def forward(self, x, pa_map, **kwargs):
        del pa_map, kwargs
        correction = torch.stack((2.0 * torch.ones(x.size(0)), -torch.ones(x.size(0))), dim=1)
        return {"logit_correction": correction.to(x)}


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


def test_v2_fusion_is_scale_aligned_convex_and_alpha_zero_is_exact():
    base_logits = torch.tensor([[6.0, -2.0], [1.0, 3.0]])
    operator_logits = torch.tensor([[1.0, -1.0], [-2.0, 2.0]])

    scale = calibrated_fusion_scale(base_logits, operator_logits)
    fused = fuse_logits(base_logits, operator_logits, alpha=0.25, scale=scale)

    torch.testing.assert_close(fuse_logits(base_logits, operator_logits, alpha=0.0, scale=scale), base_logits)
    torch.testing.assert_close(fused, 0.75 * base_logits + 0.25 * scale * operator_logits)
    assert 0.25 <= scale <= 20.0


def test_fusion_calibration_ties_fall_back_to_the_smaller_alpha():
    grid = {"0.00": 80.0, "0.05": 81.0, "0.10": 81.0, "0.20": 79.0}

    assert select_fusion_alpha(grid) == 0.05


def test_wrapper_uses_v2_convex_fusion():
    base = freeze_base_model(_TinyBase())
    model = FrozenCore90CCOI(base, _FixedSidecar(), row="C1", fusion_alpha=0.25, fusion_scale=2.0)
    x = torch.ones(2, 2, 64)

    out = model(x, return_aux=True)

    expected = 0.75 * out["base_tx_logits"] + 0.25 * 2.0 * out["ccoi"]["logit_correction"]
    torch.testing.assert_close(out["tx_logits"], expected)


def test_nested_wisig_metadata_returns_raw_receiver_id():
    extra = (torch.tensor([0, 1]), {"rx_i": torch.tensor([7, 11])})

    assert _meta_value(extra, "rx_i", 1, -1) == 11


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


def test_pair_audit_records_negative_and_anchor_coverage():
    pair = CCOILossOutput(
        loss=torch.tensor(0.0),
        positive_count=3,
        negative_count=5,
        anchor_count=4,
    )
    sums = {}

    accumulate_pair_audit(sums, pair, batch_size=8)

    assert sums == {
        "positive_pairs": 3.0,
        "negative_pairs": 5.0,
        "anchor_count": 4.0,
        "anchor_fraction": 0.5,
    }


def test_c4_holdout_uses_separate_frozen_base_forwards():
    base = freeze_base_model(_TinyBase())
    sidecar = CCOIPASidecar(pa_channels=1, num_classes=2)
    sidecar.freeze_challenge_encoder()
    model = FrozenCore90CCOI(base, sidecar=sidecar, row="C4", fusion_alpha=0.15)

    out = model(torch.randn(2, 2, 256), return_aux=True)

    assert base.calls == 3
    assert out["ccoi"]["heldout_target"].shape[:2] == (2, 1)
