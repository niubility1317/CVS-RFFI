from __future__ import annotations

import sys
from pathlib import Path

import torch


PROJECT_ROOT = Path(__file__).resolve().parents[2]
CODE_ROOT = PROJECT_ROOT / "code"
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

from model_dual_cvsincnet import build_dual_model  # noqa: E402


def _small_model(**kwargs):
    return build_dual_model(
        num_classes=3,
        num_domains=2,
        model_size="S",
        dataset="wisig",
        input_len=64,
        model_variant="lite_h",
        fast_infer_when_no_aux=False,
        **kwargs,
    )


def test_fcr_off_preserves_legacy_state_dict_and_outputs() -> None:
    """Adding FCR plumbing must leave the closed legacy path bit-identical."""
    torch.manual_seed(903)
    legacy = _small_model()
    torch.manual_seed(903)
    fcr_off = _small_model(use_fcr=False)
    legacy.eval()
    fcr_off.eval()

    legacy_state = legacy.state_dict()
    fcr_off.load_state_dict(legacy_state, strict=True)
    assert legacy_state.keys() == fcr_off.state_dict().keys()
    assert not any(key.startswith("fcr.") for key in fcr_off.state_dict())
    assert fcr_off.use_fcr is False
    assert fcr_off.fcr_config is None
    assert fcr_off.fcr is None

    x = torch.randn(2, 2, 64)
    with torch.no_grad():
        legacy_out = legacy(x, return_aux=True)
        fcr_off_out = fcr_off(x, return_aux=True)
    torch.testing.assert_close(
        legacy_out["tx_logits"], fcr_off_out["tx_logits"], rtol=0.0, atol=0.0
    )
    torch.testing.assert_close(
        legacy_out["z_id"], fcr_off_out["z_id"], rtol=0.0, atol=0.0
    )
