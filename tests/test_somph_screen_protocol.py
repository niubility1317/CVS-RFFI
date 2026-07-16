from __future__ import annotations

import json

import numpy as np
import pytest

from paper_reproduction.scripts.screen_support_only_multiprototype_head import (
    _load_cache,
)


def test_clean_co_residence_is_rejected_before_feature_tensor_access(tmp_path) -> None:
    path = tmp_path / "leo_clear_weak.npz"
    # An object feature array would fail under allow_pickle=False if accessed.
    # The protocol check must reject the clean provenance row first.
    np.savez(
        path,
        features=np.asarray([{"forbidden": "feature"}], dtype=object),
        tx_ids=np.asarray(["14-10"]),
        rx_ids=np.asarray(["20-1"]),
        day_ids=np.asarray(["0"]),
        eq_ids=np.asarray(["0"]),
        sig_ids=np.asarray(["0"]),
        dataset_role=np.asarray(["source"]),
        channel_views=np.asarray(["clean"]),
        sat_scenarios=np.asarray([""]),
        manifest_json=np.asarray(json.dumps({})),
    )
    with pytest.raises(ValueError, match="PROTOCOL_INVALID_FOR_PHASE2"):
        _load_cache(path, scenario="leo_clear_weak")
