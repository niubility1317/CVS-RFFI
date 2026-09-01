from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[2]
CODE_ROOT = PROJECT_ROOT / "code"
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

from baseline_origin_sat_view import BaselineOriginSatViewAugment  # noqa: E402
from dataset_wisig import WiSigCompactDataset, WiSigMetaSslSubsetDataset  # noqa: E402
from cvsrffi.tensors import extract_domain_from_extra  # noqa: E402


def _tiny_wisig_dataset() -> WiSigCompactDataset:
    signal = np.stack(
        [np.arange(10, dtype=np.float32), np.arange(10, dtype=np.float32) * -1.0],
        axis=-1,
    )[None, ...]
    ds = {
        "data": [[[[signal]]]],
        "tx_list": ["tx0"],
        "rx_list": ["rx0"],
        "capture_date_list": ["day0"],
        "equalized_list": [1],
    }
    return WiSigCompactDataset(ds, out_len=6, crop_mode="center", normalize=False)


def test_wisig_meta_exposes_stable_physical_sample_and_crop_identity() -> None:
    dataset = _tiny_wisig_dataset()
    _, _, _, meta = dataset[0]

    assert meta["physical_sample_id"] == "sample:0"
    assert meta["pair_id"] == meta["physical_sample_id"]
    assert meta["receiver_id"] == 0
    assert meta["day_id"] == 0
    assert meta["crop_offset"] == 2
    assert meta["synchronized_crop"] is True


def test_ecrs_metadata_extension_preserves_legacy_domain_extraction() -> None:
    expected = torch.tensor([2, 3])
    assert torch.equal(extract_domain_from_extra((expected,), torch.device("cpu")), expected)
    assert torch.equal(
        extract_domain_from_extra(([4, 5],), torch.device("cpu")), torch.tensor([4, 5])
    )


def test_unlabeled_source_metadata_contains_no_tx_truth_or_tx_bearing_id() -> None:
    unlabeled = WiSigMetaSslSubsetDataset(
        _tiny_wisig_dataset(), [0], split_source="source_unlabeled", role="U_s", tx_label_visible=False
    )
    _, y, _, meta = unlabeled[0]
    assert y == -1
    assert not ({"tx", "tx_i", "true_tx_i", "tx_label", "y_tx"} & set(meta))
    assert "tx" not in meta["physical_sample_id"].lower()
    assert "tx" not in meta["pair_id"].lower()


def test_ecrs_pair_metadata_keeps_clean_and_leo_views_synchronized() -> None:
    def fake_apply(x, scenario, args, gen=None, return_meta=False):
        return x + 1.0, {"scenario": scenario, "snr_db": 18.0}

    augment = BaselineOriginSatViewAugment(
        scenarios=["leo_clear_weak"], p=1.0, seed=7, apply_fn=fake_apply
    )
    x = torch.zeros(2, 2, 6)
    y = torch.tensor([1, -1])
    sample_meta = {
        "physical_sample_id": ["p0", "p1"],
        "pair_id": ["p0", "p1"],
        "rx_i": torch.tensor([2, 3]),
        "day_i": torch.tensor([4, 5]),
        "crop_offset": torch.tensor([7, 8]),
    }
    out = augment.expand(
        x,
        y,
        None,
        args=SimpleNamespace(),
        epoch=1,
        batch_idx=0,
        use_ecrs=True,
        sample_meta=sample_meta,
        label_mask=torch.tensor([True, False]),
    )
    meta = out.pair_meta

    assert meta is not None
    assert meta["pair_id"][:2] == meta["pair_id"][2:]
    assert meta["physical_sample_id"][:2] == meta["physical_sample_id"][2:]
    assert torch.equal(meta["crop_offset"][:2], meta["crop_offset"][2:])
    assert meta["view_type"] == ["clean", "clean", "leo", "leo"]
    assert torch.equal(meta["clean_mask"], torch.tensor([True, True, False, False]))
    assert torch.equal(meta["leo_mask"], torch.tensor([False, False, True, True]))
    assert torch.equal(meta["label_mask"], torch.tensor([True, False, True, False]))
    assert "true_tx_i" not in meta
    assert meta["synchronized_crop"] is True
