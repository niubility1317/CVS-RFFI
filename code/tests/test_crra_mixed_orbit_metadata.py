import sys
from pathlib import Path
from types import SimpleNamespace

import torch


PROJECT_ROOT = Path(__file__).resolve().parents[2]
CODE_ROOT = PROJECT_ROOT / "code"
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

from baseline_origin_sat_view import BaselineOriginSatViewAugment  # noqa: E402


def _fake_apply(x, scenario, args, gen=None, return_meta=False):
    assert return_meta is True
    batch = int(x.size(0))
    meta = {
        "scenario": scenario,
        "snr_db": torch.linspace(5.0, 8.0, batch),
        "cfo_hz": torch.arange(batch, dtype=torch.float32) * 10.0,
        "residual_cfo_hz": torch.arange(batch, dtype=torch.float32) * 2.0,
        "fD_hz": torch.arange(batch, dtype=torch.float32) * 3.0,
        "pl_db": torch.full((batch,), 120.0),
        "K_db": torch.full((batch,), 4.0),
        "theta_deg": torch.full((batch,), 45.0),
        "h_km": torch.full((batch,), 550.0),
        "state": torch.arange(batch, dtype=torch.float32),
    }
    return x + 1.0, meta


def test_mixed_orbit_metadata_is_carried_with_the_same_satellite_view():
    aug = BaselineOriginSatViewAugment(
        scenarios=["mixed_orbit"],
        p=1.0,
        seed=7,
        apply_fn=_fake_apply,
    )
    view = aug.transform(
        torch.randn(3, 2, 32),
        args=SimpleNamespace(),
        epoch=1,
        batch_idx=1,
    )
    assert view.applied is True
    assert view.meta["scenario"] == "mixed_orbit"
    assert view.nuisance.shape == (3, 9)
    assert bool(view.nuisance_valid.all()) is True
    assert view.nuisance_fields[0] == "snr_db"


def test_expand_masks_clean_rows_and_keeps_satellite_nuisance_alignment():
    aug = BaselineOriginSatViewAugment(
        scenarios=["mixed_orbit"],
        p=1.0,
        seed=7,
        apply_fn=_fake_apply,
    )
    batch = aug.expand(
        torch.randn(2, 2, 32),
        torch.tensor([0, 1]),
        None,
        args=SimpleNamespace(),
        epoch=1,
        batch_idx=1,
    )
    assert batch.nuisance.shape == (4, 9)
    assert bool(batch.nuisance_valid[:2].any()) is False
    assert bool(batch.nuisance_valid[2:].all()) is True


def test_clean_duplicate_does_not_invent_nuisance_targets():
    aug = BaselineOriginSatViewAugment(
        scenarios=["mixed_orbit"],
        p=0.0,
        seed=7,
        apply_fn=_fake_apply,
    )
    view = aug.transform(
        torch.randn(2, 2, 16),
        args=SimpleNamespace(),
        epoch=1,
        batch_idx=1,
    )
    assert view.applied is False
    assert view.nuisance is None
    assert view.nuisance_valid is None
