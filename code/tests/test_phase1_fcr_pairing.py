from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import torch

from dataset_wisig import WiSigCompactDataset, WiSigMetaSslSubsetDataset


def _tiny_wisig_dataset() -> dict:
    return {
        "data": [[[[np.arange(16, dtype=np.float32).reshape(1, 8, 2)]]]],
        "tx_list": ["tx-0"],
        "rx_list": ["rx-0"],
        "capture_date_list": ["2026-01-01"],
        "equalized_list": [1],
    }


def test_unlabeled_role_hides_reversible_tx_metadata() -> None:
    """Removing the label must also remove every reversible TX metadata field."""

    base = WiSigCompactDataset(_tiny_wisig_dataset(), out_len=4, normalize=False)
    hidden = WiSigMetaSslSubsetDataset(
        base,
        [0],
        split_source="unit",
        role="unlabeled_source",
        tx_label_visible=False,
    )

    _, y, _, meta = hidden[0]

    assert y == -1
    assert set(meta).isdisjoint({"true_tx_i", "tx_i", "tx"})
    assert meta["label_visible"] is False
    assert meta["physical_sample_id"].startswith("sample:")
    assert "tx0" not in meta["physical_sample_id"]


def test_visible_clean_and_leo_pair_keep_physical_id_and_crop() -> None:
    """A satellite view must inherit the already-cropped clean sample identity."""

    from baseline_origin_sat_view import BaselineOriginSatViewAugment
    from cvsrffi.phase1_fcr_interventions import InterventionCubeBatchBuilder

    base = WiSigCompactDataset(_tiny_wisig_dataset(), out_len=4, normalize=False)
    visible = WiSigMetaSslSubsetDataset(
        base,
        [0],
        split_source="unit",
        role="labeled_train",
        tx_label_visible=True,
    )
    clean_one, y_one, d_one, meta_one = visible[0]
    clean = clean_one.unsqueeze(0)
    labels = torch.tensor([y_one])
    domains = torch.tensor([d_one])
    batch_meta = {key: [value] for key, value in meta_one.items()}

    def apply_fn(x, scenario, args, gen=None, return_meta=False):
        return x + 1.0, {"scenario": scenario}

    augment = BaselineOriginSatViewAugment(
        scenarios=["clear_leo"], p=1.0, seed=1, apply_fn=apply_fn
    )
    leo_view = augment.transform(
        clean,
        args=SimpleNamespace(),
        epoch=1,
        batch_idx=0,
        batch_meta=batch_meta,
    )
    pair = InterventionCubeBatchBuilder().build(
        clean,
        leo_view,
        labels,
        domains,
        batch_meta,
    )

    assert pair.clean_iq.shape == pair.leo_iq.shape
    assert pair.pair_id == pair.physical_sample_id
    torch.testing.assert_close(pair.clean_crop_offset, pair.leo_crop_offset)
    assert pair.pair_valid_mask["nuisance"].tolist() == [True]
