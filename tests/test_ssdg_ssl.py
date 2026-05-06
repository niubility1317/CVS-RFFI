import numpy as np
import torch

from dataset_wisig import (
    WiSigCompactDataset,
    WiSigUnlabeledSubsetDataset,
    build_unlabeled_indices_from_splits,
)
from train import SSDGPseudoLabelMemory


def _tiny_wisig_dict():
    data = []
    for tx_i in range(2):
        rx_block = []
        for rx_i in range(1):
            day_block = []
            for day_i in range(1):
                eq_block = []
                samples = np.zeros((4, 8, 2), dtype=np.float32)
                samples[:, :, 0] = float(tx_i + 1)
                samples[:, :, 1] = float(rx_i + day_i)
                eq_block.append(samples)
                day_block.append(eq_block)
            rx_block.append(day_block)
        data.append(rx_block)
    return {
        "data": data,
        "tx_list": ["tx0", "tx1"],
        "rx_list": ["rx0"],
        "capture_date_list": ["day0"],
        "equalized_list": [1],
    }


def test_unlabeled_wisig_view_hides_tx_label_but_keeps_domain_and_truth_meta():
    base = WiSigCompactDataset(_tiny_wisig_dict(), out_len=8, equalized=1, domain="rx_day")
    unlabeled = WiSigUnlabeledSubsetDataset(base, [0, 5], split_source="ssdg_unlabeled_pool")

    x, y, d, meta = unlabeled[1]

    assert x.shape == (2, 8)
    assert y == -1
    assert d == 0
    assert meta["has_tx_label"] is False
    assert meta["true_tx_i"] == 1
    assert meta["global_index"] == 5
    assert meta["split_source"] == "ssdg_unlabeled_pool"


def test_build_unlabeled_indices_excludes_labeled_and_validation_indices():
    remaining = build_unlabeled_indices_from_splits(
        pool_size=8,
        train_selected=[0, 1],
        val_selected=[6, 7],
    )

    assert remaining == [2, 3, 4, 5]


def test_ssdg_pseudo_label_memory_requires_stable_high_confidence_streak():
    memory = SSDGPseudoLabelMemory(num_classes=3, momentum=0.0)
    ids = torch.tensor([9])
    probs = torch.tensor([[0.04, 0.93, 0.03]])

    pred1, conf1, streak1 = memory.update(ids, probs, confidence_threshold=0.9)
    pred2, conf2, streak2 = memory.update(ids, probs, confidence_threshold=0.9)

    assert pred1.tolist() == [1]
    assert torch.allclose(conf1, torch.tensor([0.93]), atol=1e-6)
    assert streak1.tolist() == [1]
    assert pred2.tolist() == [1]
    assert torch.allclose(conf2, torch.tensor([0.93]), atol=1e-6)
    assert streak2.tolist() == [2]
