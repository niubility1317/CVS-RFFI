from __future__ import annotations

import sys
from pathlib import Path

import torch


PROJECT_ROOT = Path(__file__).resolve().parents[2]
CODE_ROOT = PROJECT_ROOT / "code"
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

from model_dual_cvsincnet import build_dual_model  # noqa: E402


def _tiny_model(**kwargs):
    return build_dual_model(
        num_classes=6,
        num_domains=5,
        model_size="S",
        dataset="wisig",
        input_len=64,
        sample_rate_hz=25e6,
        model_variant="lite_h",
        id_feature_key="feat_joint",
        dom_feature_key="feat_imp",
        fast_infer_when_no_aux=False,
        **kwargs,
    )


def test_ecrs_off_preserves_legacy_state_and_outputs() -> None:
    torch.manual_seed(20260901)
    legacy = _tiny_model().eval()
    candidate = _tiny_model(use_ecrs=False).eval()
    candidate.load_state_dict(legacy.state_dict(), strict=True)

    assert not any(key.startswith("ecrs") for key in candidate.state_dict())
    x = torch.randn(2, 2, 64)
    with torch.no_grad():
        legacy_out = legacy(x, return_aux=True)
        candidate_out = candidate(x, return_aux=True)
    torch.testing.assert_close(
        candidate_out["tx_logits"], legacy_out["tx_logits"], rtol=0.0, atol=0.0
    )
    torch.testing.assert_close(
        candidate_out["z_id"], legacy_out["z_id"], rtol=0.0, atol=0.0
    )
