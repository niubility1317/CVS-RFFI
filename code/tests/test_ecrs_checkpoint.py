from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import torch


PROJECT_ROOT = Path(__file__).resolve().parents[2]
CODE_ROOT = PROJECT_ROOT / "code"
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

from cvsrffi.checkpoint import ECRS_FEATURE_SCHEMA, save_checkpoint  # noqa: E402
from model_dual_cvsincnet import build_dual_model  # noqa: E402


def _model(use_ecrs: bool):
    return build_dual_model(
        num_classes=3,
        num_domains=2,
        model_size="M",
        dataset="wisig",
        input_len=64,
        model_variant="lite_d",
        branch_ablation="no_dac",
        domain_branch_ablation="no_stats",
        use_ecrs=use_ecrs,
        fast_infer_when_no_aux=False,
    )


def test_ecrs_checkpoint_roundtrip_preserves_single_view_inference(tmp_path: Path) -> None:
    torch.manual_seed(41)
    model = _model(True).eval()
    model.ecrs.fusion_gate.set_active_rho_max(0.20)
    leo_iq = torch.randn(2, 2, 64)
    with torch.no_grad():
        expected = model(leo_iq, return_aux=True)["z_id_fused"]

    path = tmp_path / "ecrs.pth"
    save_checkpoint(
        str(path),
        model=model,
        optimizer=None,
        scheduler=None,
        scaler=None,
        epoch=17,
        args=SimpleNamespace(feature_schema="legacy"),
        split_info={},
        stats={},
    )
    payload = torch.load(path, map_location="cpu", weights_only=False)
    assert payload["feature_schema"] == ECRS_FEATURE_SCHEMA
    bundle = payload["ecrs_bundle"]
    for key in (
        "basis",
        "M_ref",
        "anchor_grid",
        "anchor_design",
        "normalization",
        "response_projection",
        "fusion_gate",
        "response_prototypes",
        "response_covariance",
        "single_view_inference",
    ):
        assert key in bundle
    assert bundle["single_view_inference"] is True
    assert bundle["anchor_grid"].numel() == 8
    assert bundle["anchor_encoder"]["weight"].shape == (64, 16)

    restored = _model(True).eval()
    restored.load_state_dict(payload["model"], strict=True)
    with torch.no_grad():
        actual = restored(leo_iq, return_aux=True)["z_id_fused"]
    torch.testing.assert_close(actual, expected, rtol=0.0, atol=0.0)


def test_legacy_checkpoint_has_no_ecrs_bundle_and_strictly_loads(tmp_path: Path) -> None:
    model = _model(False)
    path = tmp_path / "legacy.pth"
    save_checkpoint(
        str(path),
        model=model,
        optimizer=None,
        scheduler=None,
        scaler=None,
        epoch=1,
        args=SimpleNamespace(feature_schema="ADV3B02:z_id:legacy"),
        split_info={},
        stats={},
    )
    payload = torch.load(path, map_location="cpu", weights_only=False)
    assert payload["ecrs_bundle"] is None
    assert payload["feature_schema"] == "ADV3B02:z_id:legacy"
    _model(False).load_state_dict(payload["model"], strict=True)
