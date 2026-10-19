"""Safety checks for the local source cache interface, using synthetic data."""
import json

import numpy as np
import pytest

from tools.local_phase1_data import SourceRoleDataset


def cache(tmp_path, role):
    (tmp_path / "manifest.json").write_text(json.dumps(dict(status="READY", native_role_ids="EXACT_MATCH")), encoding="utf-8")
    folder = tmp_path / role
    folder.mkdir()
    np.save(folder / "iq.npy", np.ones((2, 2, 256), dtype=np.float32))
    for name in ("domain", "rx_i", "day_i"):
        np.save(folder / (name + ".npy"), np.zeros(2, dtype=np.int64))
    np.save(folder / "physical_ids.npy", np.asarray(["opaque-a", "opaque-b"]))
    if role != "U_s":
        np.save(folder / "labels.npy", np.asarray([1, 2], dtype=np.int64))
    return folder


def test_u_has_no_labels_or_tx_metadata_and_tensor_is_writable_copy(tmp_path):
    cache(tmp_path, "U_s")
    ds = SourceRoleDataset(tmp_path, "U_s")
    x, y, domain, meta = ds[0]
    assert y == -1 and meta["tx_label_visible"] is False
    assert set(meta) == {"rx_i", "day_i", "physical_id", "split_source", "tx_label_visible"}
    x.zero_()
    assert np.all(ds.x[0] == 1)


def test_u_rejects_accidentally_exported_tx_labels(tmp_path):
    folder = cache(tmp_path, "U_s")
    np.save(folder / "labels.npy", np.zeros(2, dtype=np.int64))
    with pytest.raises(ValueError, match="must not contain TX"):
        SourceRoleDataset(tmp_path, "U_s")


def test_l_retains_original_labels(tmp_path):
    cache(tmp_path, "L_s")
    ds = SourceRoleDataset(tmp_path, "L_s")
    assert ds[1][1] == 2 and ds[1][3]["tx_label_visible"] is True


def test_target_role_is_rejected_before_any_file_is_opened(tmp_path):
    with pytest.raises(ValueError, match="Only source"):
        SourceRoleDataset(tmp_path, "target")
