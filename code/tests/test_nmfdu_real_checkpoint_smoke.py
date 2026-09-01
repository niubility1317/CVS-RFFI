from __future__ import annotations

import numpy as np
import pytest
import torch

from cvsrffi.nmfdu_real_checkpoint_smoke import (
    build_real_source_batch,
    parse_index_csv,
    validate_legacy_transfer,
)


def _tiny_wisig() -> dict:
    rows = np.stack(
        [
            np.stack([np.linspace(0.0, 1.0, 32), np.linspace(1.0, 0.0, 32)], axis=1),
            np.stack([np.linspace(1.0, 0.0, 32), np.linspace(0.0, 1.0, 32)], axis=1),
        ]
    ).astype(np.float32)
    return {
        "data": [[[[rows]]]],
        "tx_list": ["tx0"],
        "rx_list": ["rx0"],
        "capture_date_list": ["day0"],
        "equalized_list": [1],
    }


def test_real_source_batch_is_bound_to_declared_phase1_train_indices() -> None:
    x, y, metadata = build_real_source_batch(
        _tiny_wisig(),
        input_len=32,
        train_rxs=(0,),
        train_days=(0,),
        equalized=1,
        batch_size=2,
    )
    assert x.shape == (2, 2, 32)
    assert torch.equal(y, torch.zeros(2, dtype=torch.long))
    assert {row["rx_i"] for row in metadata} == {0}
    assert {row["day_i"] for row in metadata} == {0}
    assert all(row["source_role"] == "phase1_source_labeled_smoke" for row in metadata)
    assert all("query" not in key.lower() for row in metadata for key in row)


def test_legacy_transfer_allows_only_new_nmfdu_state() -> None:
    validate_legacy_transfer(
        ["id_backbone.nmfdu_gate.training_stage", "id_backbone.nmfdu_gate.x.weight"],
        [],
    )
    with pytest.raises(ValueError, match="outside NMFDU"):
        validate_legacy_transfer(["id_backbone.time_conv.weight"], [])
    with pytest.raises(ValueError, match="unexpected"):
        validate_legacy_transfer([], ["legacy.extra"])


def test_parse_index_csv_rejects_empty_source_scope() -> None:
    assert parse_index_csv("0,2,3") == (0, 2, 3)
    with pytest.raises(ValueError, match="at least one"):
        parse_index_csv("")
