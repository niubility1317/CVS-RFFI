import sys
from pathlib import Path

import numpy as np
import pytest
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[2]
CODE_ROOT = PROJECT_ROOT / "code"
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

from model_dual_cvsincnet import build_dual_model  # noqa: E402


def _write_mask(tmp_path: Path, fft_bins: int = 64) -> Path:
    path = tmp_path / "sid_mask.npz"
    np.savez_compressed(path, mask=np.ones(fft_bins, dtype=np.uint8))
    return path


def _tiny_model(**kwargs):
    return build_dual_model(
        num_classes=3,
        num_domains=2,
        model_size="S",
        dataset="wisig",
        input_len=64,
        sample_rate_hz=25e6,
        model_variant="lite_h",
        id_feature_key="feat_joint",
        dom_feature_key="feat_imp",
        **kwargs,
    )


def test_sid_zero_initialization_matches_raw_logits(tmp_path):
    model = _tiny_model(
        sid_fft96_mode="sid",
        sid_mask_path=str(_write_mask(tmp_path)),
        fast_infer_when_no_aux=False,
    ).eval()
    x = torch.randn(2, 2, 64)
    labels = torch.tensor([0, 1])

    with torch.no_grad():
        output = model(x, y_tx=labels, return_aux=True)

    assert torch.allclose(output["z_id_raw"], output["z_id_sid"], atol=1e-7, rtol=0)
    assert torch.allclose(output["logits_raw"], output["logits_sid"], atol=1e-7, rtol=0)
    assert torch.equal(output["z_id"], output["z_id_sid"])
    assert torch.equal(output["tx_logits"], output["logits_sid"])
    assert output["sid_fft96"].shape == (2, 96)


def test_sid_and_existing_residual_candidates_are_mutually_exclusive(tmp_path):
    mask_path = str(_write_mask(tmp_path))

    with pytest.raises(ValueError, match="independent candidates"):
        _tiny_model(sid_fft96_mode="sid", sid_mask_path=mask_path, use_ntrs=True)
    with pytest.raises(ValueError, match="independent candidates"):
        _tiny_model(sid_fft96_mode="sid", sid_mask_path=mask_path, use_crra=True)


def test_sid_off_preserves_parameter_count_and_fast_path():
    torch.manual_seed(11)
    control = _tiny_model().eval()
    torch.manual_seed(11)
    explicit_off = _tiny_model(sid_fft96_mode="off", sid_mask_path="").eval()

    assert sum(parameter.numel() for parameter in control.parameters()) == sum(
        parameter.numel() for parameter in explicit_off.parameters()
    )
    x = torch.randn(2, 2, 64)
    with torch.no_grad():
        assert torch.equal(control(x), explicit_off(x))
