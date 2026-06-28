import sys
from pathlib import Path

import pytest
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "code"))

from cvsrffi.feature_masks import FeatureMaskRouter  # noqa: E402


def test_feature_mask_router_outputs_normalized_subspaces_and_regularization():
    router = FeatureMaskRouter(feat_dim=4, tx_ratio=0.6, rx_ratio=0.3, int_ratio=0.1)
    h = torch.randn(5, 4)

    z_tx, z_rx, z_int, masks = router(h)
    reg, metrics = router.mask_regularization()

    assert z_tx.shape == z_rx.shape == z_int.shape == (5, 4)
    assert set(masks) == {"tx", "rx", "int"}
    assert torch.allclose(masks["tx"] + masks["rx"] + masks["int"], torch.ones(4), atol=1e-6)
    assert torch.isfinite(reg)
    assert 0.0 < metrics["tx_mean"] < 1.0


def test_feature_mask_router_sanitizes_nan_inf_inputs():
    router = FeatureMaskRouter(feat_dim=2)
    h = torch.tensor([[float("nan"), float("inf")], [0.0, 0.0]])

    z_tx, z_rx, z_int, _ = router(h)

    assert torch.isfinite(z_tx).all()
    assert torch.isfinite(z_rx).all()
    assert torch.isfinite(z_int).all()


def test_feature_mask_router_rejects_dimension_mismatch():
    router = FeatureMaskRouter(feat_dim=3)

    with pytest.raises(ValueError):
        router(torch.randn(2, 4))
