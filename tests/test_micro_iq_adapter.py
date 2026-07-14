from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from paper_reproduction.cvs_aligned.cvs_method_runner import SCENARIOS
from paper_reproduction.scripts.train_export_cvs_micro_iq_adapter import (
    MicroIQResidualAdapter,
    _numpy_to_tensor_compat,
    _tensor_to_numpy_compat,
    adapter_resource_audit,
    assemble_support_views,
)
from export_spaceborne_features import extract_features_with_metadata


def test_micro_iq_adapter_starts_as_exact_identity_and_is_extreme_light() -> None:
    adapter = MicroIQResidualAdapter(hidden=8, kernel_size=5, alpha=0.2)
    rows = torch.randn(4, 2, 256)
    torch.testing.assert_close(adapter(rows), rows, rtol=0.0, atol=0.0)
    audit = adapter_resource_audit(adapter, sequence_length=256)
    assert audit["trainable_parameters"] == 154
    assert audit["adapter_state_bytes_fp16"] == 308
    assert audit["adapter_macs_per_query"] == 34816
    assert audit["query_view_count"] == 1


def test_numpy_buffer_bridge_preserves_shape_dtype_and_values() -> None:
    array = np.arange(12, dtype=np.float32).reshape(2, 2, 3)
    tensor = _numpy_to_tensor_compat(
        array, numpy_dtype=np.dtype(np.float32), torch_dtype=torch.float32
    )
    assert tensor.shape == (2, 2, 3)
    assert tensor.dtype == torch.float32
    torch.testing.assert_close(tensor, torch.arange(12, dtype=torch.float32).reshape(2, 2, 3))
    restored = _tensor_to_numpy_compat(tensor, dtype=np.dtype(np.float32))
    np.testing.assert_array_equal(restored, array)


def _scenario_cache(scenario: str) -> dict[str, np.ndarray]:
    roles = []
    tx_ids = []
    rx_ids = []
    day_ids = []
    eq_ids = []
    sig_ids = []
    raw = []
    for role, tx, offset in (("target_old", "old-a", 0), ("target_new", "new-a", 100)):
        for signal in range(30):
            roles.append(role)
            tx_ids.append(tx)
            rx_ids.append("rx-a")
            day_ids.append("0")
            eq_ids.append("1")
            sig_ids.append(str(signal))
            raw.append(np.full((2, 16), offset + signal, dtype=np.float32))
    return {
        "raw_iq": np.stack(raw),
        "dataset_role": np.asarray(roles),
        "tx_ids": np.asarray(tx_ids),
        "rx_ids": np.asarray(rx_ids),
        "day_ids": np.asarray(day_ids),
        "eq_ids": np.asarray(eq_ids),
        "sig_ids": np.asarray(sig_ids),
        "sat_scenarios": np.asarray([scenario] * len(roles)),
    }


def test_support_assembly_uses_three_matched_views_and_no_query_rows() -> None:
    caches = {scenario: _scenario_cache(scenario) for scenario in SCENARIOS}
    rows, labels, manifest = assemble_support_views(
        caches,
        receiver="rx-a",
        old_labels=["old-a"],
        new_labels=["new-a"],
        seed=713101,
        k_shot=10,
        support_pool_max_k=10,
        query_per_tx=20,
    )
    assert rows.shape == (60, 2, 16)
    assert labels.shape == (60,)
    assert np.bincount(labels).tolist() == [30, 30]
    assert manifest["support_view_count"] == 3
    assert len(manifest["physical_support_ids"]) == 20
    assert len(manifest["physical_query_ids"]) == 40
    assert set(manifest["scenario_audit"]) == set(SCENARIOS)


class _TinyExportDataset(Dataset):
    def __len__(self) -> int:
        return 3

    def __getitem__(self, index: int):
        row = torch.full((2, 16), float(index))
        return row, index % 2, 0, {
            "tx": f"tx-{index % 2}",
            "rx": "rx-a",
            "day": "0",
            "equalized": "1",
            "sig_i": str(index),
        }


class _TinyExportModel(torch.nn.Module):
    def forward(self, x, **_kwargs):
        z = x.mean(dim=-1)
        return {"z_id": z, "z_dom": z, "tx_logits": z}


def test_feature_export_can_persist_exact_raw_iq_view() -> None:
    payload = extract_features_with_metadata(
        _TinyExportModel(),
        DataLoader(_TinyExportDataset(), batch_size=2, shuffle=False),
        device=torch.device("cpu"),
        feature_name="z_id",
        role="target_old",
        channel_view="clean",
        include_raw_iq=True,
    )
    assert payload["raw_iq"].shape == (3, 2, 16)
    np.testing.assert_array_equal(payload["raw_iq"][2], np.full((2, 16), 2.0, dtype=np.float32))
