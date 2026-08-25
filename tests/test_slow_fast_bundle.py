from __future__ import annotations

from pathlib import Path

import pytest
import torch

from cvsrffi.slow_fast_adapter import SlowFastAdapterState, SlowFastCandidate
from cvsrffi.slow_fast_bundle import (
    load_slow_fast_bundle_strict,
    save_slow_fast_bundle,
)


def _state() -> SlowFastAdapterState:
    return SlowFastAdapterState(
        candidate=SlowFastCandidate.FAST_FILM_R8,
        slow_u=torch.randn(160, 8),
        slow_v=torch.randn(160, 8),
        rho=0.1,
        gamma=torch.zeros(8),
        beta=torch.zeros(8),
    )


def _metadata() -> dict[str, object]:
    return {
        "base_checkpoint_id": "ADV3B02_CORE90_SOFT_E200",
        "class_ids": torch.arange(6, dtype=torch.long),
        "prototypes": torch.randn(6, 160),
        "support_logit_scale": 8.0,
        "fast_step_size": 0.02,
        "trust_radius": 0.15,
    }


def test_bundle_roundtrip_contains_aggregate_state_but_no_source_rows(
    tmp_path: Path,
) -> None:
    path = tmp_path / "slow_fast.pt"
    save_slow_fast_bundle(path, _state(), _metadata())

    state, audit = load_slow_fast_bundle_strict(path)

    assert state.candidate is SlowFastCandidate.FAST_FILM_R8
    assert state.feature_dim == 160
    assert audit["base_checkpoint_id"] == "ADV3B02_CORE90_SOFT_E200"
    assert audit["class_ids"].tolist() == list(range(6))
    assert audit["prototypes"].shape == (6, 160)
    payload = torch.load(path, map_location="cpu", weights_only=True)
    assert payload["schema"] == "cvs.cached_slow_fast.v2"
    assert audit["direction_gate_semantics"] == "signed_tanh_zero_centered"
    assert not {
        "source_cache",
        "source_features",
        "physical_sample_ids",
        "received_iq",
    } & set(payload)


def test_bundle_rejects_source_cache_metadata_before_writing(tmp_path: Path) -> None:
    metadata = _metadata()
    metadata["source_cache"] = "forbidden.npz"

    with pytest.raises(ValueError, match="metadata allowlist"):
        save_slow_fast_bundle(tmp_path / "forbidden.pt", _state(), metadata)
    assert not (tmp_path / "forbidden.pt").exists()


def test_legacy_sigmoid_gate_is_converted_without_changing_its_initial_effect(
    tmp_path: Path,
) -> None:
    path = tmp_path / "legacy.pt"
    legacy = SlowFastAdapterState(
        candidate=SlowFastCandidate.FAST_LOWRANK_R8,
        slow_u=torch.randn(160, 8),
        slow_v=torch.randn(160, 8),
        rho=0.1,
        gamma=torch.zeros(8),
        beta=torch.zeros(8),
        direction_gate=torch.zeros(8),
    )
    save_slow_fast_bundle(path, legacy, _metadata())
    payload = torch.load(path, map_location="cpu", weights_only=True)
    payload["schema"] = "cvs.cached_slow_fast.v1"
    torch.save(payload, path)

    converted, audit = load_slow_fast_bundle_strict(path)

    assert audit["source_schema"] == "cvs.cached_slow_fast.v1"
    assert torch.allclose(torch.tanh(converted.direction_gate), torch.full((8,), 0.5))
