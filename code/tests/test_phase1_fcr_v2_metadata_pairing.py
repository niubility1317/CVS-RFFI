from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[2]
CODE_ROOT = PROJECT_ROOT / "code"
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

from baseline_origin_sat_view import BaselineOriginSatViewAugment, CRRA_NUISANCE_FIELDS  # noqa: E402
from cvsrffi.phase1_fcr_v2_metadata import build_fcr_v2_metadata  # noqa: E402
from cvsrffi.phase1_fcr_v2_pairing import FCRV2PairBuilder  # noqa: E402


def _batch_meta() -> dict[str, object]:
    return {
        "physical_sample_id": ("physical:p0", "physical:p1", "physical:p2", "physical:p3"),
        "content_record_id": ("content:0", "content:1", "content:2", "content:3"),
        "crop_offset": torch.tensor([0, 0, 224, 224], dtype=torch.long),
        "common_preamble_id": ("pre:0", "pre:0", "pre:1", "pre:1"),
        "tx_id": torch.tensor([0, 1, 0, 1], dtype=torch.long),
        "rx_i": torch.tensor([2, 2, 2, 2], dtype=torch.long),
        "day_i": torch.tensor([5, 5, 5, 5], dtype=torch.long),
        "link_condition": ("clear", "clear", "clear", "clear"),
        "excitation_bin": torch.tensor([3, 3, 7, 7], dtype=torch.long),
    }


def _eta_meta(batch_size: int) -> dict[str, object]:
    values = torch.arange(batch_size, dtype=torch.float32)
    return {
        "scenario": "clear_leo",
        "snr_db": 12.0 + values,
        "cfo_hz": 50.0 + values,
        "residual_cfo_hz": 5.0 + values,
        "fD_hz": 100.0 + values,
        "pl_db": 130.0 + values,
        "K_db": 4.0 + values,
        "theta_deg": 20.0 + values,
        "h_km": 550.0 + values,
        "state": torch.ones(batch_size, dtype=torch.float32),
    }


def _build_augmented_view(batch_meta: dict[str, object]):
    def fake_apply(x, scenario, args, gen=None, return_meta=False):
        return x + 1.0, _eta_meta(int(x.shape[0]))

    augment = BaselineOriginSatViewAugment(
        scenarios=["clear_leo"],
        p=1.0,
        seed=392005,
        apply_fn=fake_apply,
    )
    iq = torch.arange(4 * 2 * 8, dtype=torch.float32).view(4, 2, 8)
    return augment.transform(
        iq,
        args=SimpleNamespace(),
        epoch=7,
        batch_idx=0,
        batch_meta=batch_meta,
    )


def _directed_pair_keys(metadata, pairs: torch.Tensor) -> set[tuple[str, str, str, str]]:
    return {
        (
            metadata.physical_sample_id[int(src)],
            metadata.view_type[int(src)],
            metadata.physical_sample_id[int(dst)],
            metadata.view_type[int(dst)],
        )
        for src, dst in pairs.tolist()
    }


def test_augmentation_exports_applied_eta_and_full_valid_mask() -> None:
    view = _build_augmented_view(_batch_meta())

    assert view.applied is True
    assert view.eta.shape == (4, len(CRRA_NUISANCE_FIELDS))
    assert view.eta_valid_mask.shape == view.eta.shape
    assert view.eta_valid_mask.float().mean().item() >= 0.99


def test_metadata_builder_rejects_missing_required_batch_field() -> None:
    batch_meta = _batch_meta()
    batch_meta.pop("content_record_id")

    with pytest.raises(ValueError, match="content_record_id"):
        build_fcr_v2_metadata(batch_meta, _build_augmented_view(_batch_meta()))


def test_metadata_builder_concatenates_clean_and_view_rows() -> None:
    metadata = build_fcr_v2_metadata(_batch_meta(), _build_augmented_view(_batch_meta()))

    assert metadata.batch_size == 8
    assert metadata.eta_schema_version == "fcr-v2/eta-v1"
    assert metadata.view_type[:4] == ("clean", "clean", "clean", "clean")
    assert metadata.view_type[4:] == ("clear_leo", "clear_leo", "clear_leo", "clear_leo")
    assert metadata.eta.shape == (8, len(CRRA_NUISANCE_FIELDS))
    assert metadata.eta_valid_mask.shape == metadata.eta.shape
    assert metadata.physical_sample_id[:4] == metadata.physical_sample_id[4:]
    torch.testing.assert_close(metadata.crop_offset[:4], metadata.crop_offset[4:])


def test_pair_builder_is_stateless_and_cross_tx_strict() -> None:
    metadata = build_fcr_v2_metadata(_batch_meta(), _build_augmented_view(_batch_meta()))
    builder = FCRV2PairBuilder(crop_span=256)

    forward = builder.build(metadata, epoch=8, seed=392005)
    flipped = builder.build(metadata.flip_batch(), epoch=8, seed=392005)

    assert _directed_pair_keys(metadata, forward["fingerprint"]) == _directed_pair_keys(
        metadata.flip_batch(), flipped["fingerprint"]
    )
    assert forward["nuisance"].ndim == 2
    assert forward["content"].ndim == 2
    assert forward["fingerprint"].ndim == 2
    assert all(metadata.tx_id[int(src)] != metadata.tx_id[int(dst)] for src, dst in forward["fingerprint"].tolist())
