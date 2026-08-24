import numpy as np
import pytest

from dataset_wisig import WiSigIndex, wisig_capture_block_id, wisig_physical_sample_id
from dataset_wisig import WiSigCompactDataset


def test_wisig_physical_id_is_complete_and_stable():
    item = WiSigIndex(tx_i=2, rx_i=3, day_i=1, eq_i=0, sig_i=19)
    assert wisig_physical_sample_id(item) == "tx2|rx3|day1|eq0|sig19"


def test_capture_block_uses_sig_index_without_claiming_real_channel():
    item = WiSigIndex(tx_i=2, rx_i=3, day_i=1, eq_i=0, sig_i=19)
    assert wisig_capture_block_id(item, block_size=8) == 2


def test_capture_block_rejects_non_positive_block_size():
    item = WiSigIndex(tx_i=2, rx_i=3, day_i=1, eq_i=0, sig_i=19)
    for block_size in (0, -1):
        with pytest.raises(ValueError, match="capture block_size must be positive"):
            wisig_capture_block_id(item, block_size=block_size)


def test_compact_dataset_rejects_non_positive_capture_block_size():
    samples = np.zeros((1, 2, 2), dtype=np.float32)
    ds = {
        "data": [[[[samples]]]],
        "tx_list": ["tx"],
        "rx_list": ["rx"],
        "capture_date_list": ["day"],
        "equalized_list": [1],
    }
    with pytest.raises(ValueError, match="capture_block_size must be positive"):
        WiSigCompactDataset(ds, out_len=2, normalize=False, capture_block_size=0)


def test_compact_dataset_metadata_contains_stable_identity_and_proxy_block():
    samples = np.zeros((3, 2, 2), dtype=np.float32)
    ds = {
        "data": [[[[samples]]]],
        "tx_list": ["tx"],
        "rx_list": ["rx"],
        "capture_date_list": ["day"],
        "equalized_list": [1],
    }
    dataset = WiSigCompactDataset(
        ds,
        out_len=2,
        normalize=False,
        capture_block_size=2,
    )

    _, _, _, meta = dataset[2]

    assert meta["physical_sample_id"] == "tx0|rx0|day0|eq0|sig2"
    assert meta["capture_block_i"] == 1
    assert meta["capture_block_semantics"] == "sig_index_time_block_proxy"
