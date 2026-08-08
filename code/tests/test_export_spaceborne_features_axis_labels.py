from __future__ import annotations

import pytest

from export_spaceborne_features import _resolve_indices


def test_export_axis_resolution_preserves_physical_day_and_receiver_labels():
    assert _resolve_indices(
        ["2021_03_01", "2021_03_08", "2021_03_15"],
        "2021_03_01,2021_03_08",
    ) == [0, 1]
    assert _resolve_indices(
        ["1-1", "1-19", "14-7"],
        "1-1,14-7",
    ) == [0, 2]


def test_export_axis_resolution_still_accepts_numeric_indices_and_rejects_drift():
    assert _resolve_indices(["day0", "day1"], "0,1") == [0, 1]
    with pytest.raises(ValueError, match="cannot resolve"):
        _resolve_indices(["2021_03_01"], "20210301")
