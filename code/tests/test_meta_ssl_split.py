import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CODE_ROOT = PROJECT_ROOT / "code"
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

from dataset_wisig import make_wisig_meta_ssl_source_split


def _fake_wisig(samples_per_combo=10):
    data = []
    for tx in range(3):
        tx_rows = []
        for rx in range(3):
            rx_rows = []
            for day in range(4):
                day_rows = []
                arr = np.zeros((samples_per_combo, 16, 2), dtype=np.float32)
                arr[:, :, 0] = tx + rx * 0.1 + day * 0.01
                day_rows.append(arr)
                rx_rows.append(day_rows)
            tx_rows.append(rx_rows)
        data.append(tx_rows)
    return {
        "data": data,
        "tx_list": [f"tx{i}" for i in range(3)],
        "rx_list": [f"rx{i}" for i in range(3)],
        "capture_date_list": [f"day{i}" for i in range(4)],
        "equalized_list": [1],
    }


def test_meta_ssl_split_masks_unlabeled_tx_and_is_disjoint():
    labeled, unlabeled, source_val, info = make_wisig_meta_ssl_source_split(
        _fake_wisig(samples_per_combo=10),
        train_days=[0, 1],
        holdout_days=[2, 3],
        train_rxs=[0, 1],
        holdout_rxs=[2],
        labeled_ratio=0.1,
        unlabeled_ratio=0.7,
        val_ratio=0.2,
        seed=11,
    )

    assert info["overlap_count"] == 0
    assert len(labeled) > 0
    assert len(unlabeled) > len(labeled)
    assert len(source_val) > 0
    assert info["source_ssl_split"] == "0.1L/0.7U/0.2Val"

    _, y_labeled, _, meta_labeled = labeled[0]
    _, y_unlabeled, _, meta_unlabeled = unlabeled[0]
    _, y_val, _, meta_val = source_val[0]

    assert y_labeled >= 0
    assert meta_labeled["tx_label_visible"] is True
    assert y_unlabeled == -1
    assert meta_unlabeled["tx_label_visible"] is False
    assert "true_tx_i" in meta_unlabeled
    assert y_val >= 0
    assert meta_val["meta_ssl_role"] == "source_val"


def test_meta_ssl_split_label_matches_requested_ratios():
    _, unlabeled, source_val, info = make_wisig_meta_ssl_source_split(
        _fake_wisig(samples_per_combo=10),
        train_days=[0, 1],
        holdout_days=[2, 3],
        train_rxs=[0, 1],
        holdout_rxs=[2],
        labeled_ratio=0.1,
        unlabeled_ratio=0.6,
        val_ratio=0.3,
        seed=11,
    )

    assert len(unlabeled) > 0
    assert len(source_val) > 0
    assert info["source_ssl_split"] == "0.1L/0.6U/0.3Val"
    assert info["mode"] == "meta_ssl_source_only_0p1L_0p6U_0p3Val"
    assert unlabeled.split_source == "meta_ssl_unlabeled_source_0p6_tx_masked"
    assert source_val.split_source == "meta_ssl_source_val_0p3_tx_visible_eval_only"
