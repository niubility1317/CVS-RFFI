import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[2]
CODE_ROOT = PROJECT_ROOT / "code"
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

from post_stage_common import build_baseline_model  # noqa: E402


def test_build_baseline_model_applies_sid_residual_cap(tmp_path):
    mask_path = tmp_path / "sid_mask.npz"
    np.savez_compressed(mask_path, mask=np.ones(64, dtype=np.uint8))
    args = SimpleNamespace(
        num_classes=3,
        num_domains=2,
        model_size="S",
        dataset="wisig",
        input_len=64,
        sample_rate_hz=25e6,
        model_variant="lite_h",
        id_feature_key="feat_joint",
        dom_feature_key="feat_imp",
        sid_fft96_mode="sid",
        sid_mask_path=str(mask_path),
        sid_residual_scale=1.0,
        sid_max_residual_ratio=0.10,
        fast_infer_when_no_aux=False,
    )

    model = build_baseline_model(args, device=torch.device("cpu")).eval()

    assert model.sid_fft96 is not None
    assert model.sid_fft96.max_residual_ratio == pytest.approx(0.10)
    with torch.no_grad():
        for parameter in model.sid_fft96.projector.parameters():
            parameter.fill_(100.0)
        z_raw = torch.randn(4, model.emb_dim)
        output = model.sid_fft96(torch.randn(4, 2, 64), z_raw)
    ratio = (output["z_sid"] - z_raw).norm(dim=1) / z_raw.norm(dim=1).clamp_min(1e-12)
    assert torch.all(ratio <= 0.100001)
