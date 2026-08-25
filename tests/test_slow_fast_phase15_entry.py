from __future__ import annotations

import torch
import torch.nn as nn

from cvsrffi import slow_fast_phase15_entry as subject


class _FrozenFeatures(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.anchor = nn.Parameter(torch.ones(()), requires_grad=False)

    def forward(self, rows: torch.Tensor) -> dict[str, torch.Tensor]:
        return {"z_id": rows[:, :, 0]}


class _SourceRows:
    def __init__(self) -> None:
        self.rows = []
        for class_id in (10, 20):
            for sample in range(2):
                x = torch.tensor([[float(class_id), 0.0], [0.0, float(sample + 1)]])
                self.rows.append(
                    (
                        x,
                        class_id,
                        0,
                        {
                            "rx_i": sample,
                            "day_i": 0,
                            "physical_sample_id": f"c{class_id}-s{sample}",
                        },
                    )
                )

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int):
        return self.rows[index]


def test_ground_cache_builder_extracts_four_views_once_per_physical_source_row(
    monkeypatch,
) -> None:
    offsets = {
        "clean": 0.0,
        "leo_clear_weak": 1.0,
        "leo_low_elev_weak": 2.0,
        "leo_rain_weak": 3.0,
    }
    monkeypatch.setattr(
        subject,
        "_materialize_source_view",
        lambda x, *, physical_sample_id, view, seed: x + offsets[view],
    )

    cache = subject.build_ground_feature_cache(
        _FrozenFeatures(),
        _SourceRows(),
        class_id_to_row={10: 0, 20: 1},
        seed=392002,
        device="cpu",
        batch_size=3,
    )

    assert cache.features.shape == (16, 2)
    assert cache.labels.tolist().count(0) == 8
    assert cache.labels.tolist().count(1) == 8
    assert set(cache.views) == {
        "clean",
        "leo_clear_weak",
        "leo_low_elev_weak",
        "leo_rain_weak",
    }
    assert set(cache.roles) == {"L_s"}
    assert len(set(zip(cache.physical_sample_ids, cache.views))) == 16


def test_ground_cache_builder_rejects_unregistered_source_class() -> None:
    with torch.no_grad():
        rows = _SourceRows()
        rows.rows = [rows.rows[0]]
    try:
        subject.build_ground_feature_cache(
            _FrozenFeatures(), rows, class_id_to_row={20: 0}, seed=1, device="cpu"
        )
    except ValueError as exc:
        assert "frozen class mapping" in str(exc)
    else:
        raise AssertionError("unregistered source class must fail")
