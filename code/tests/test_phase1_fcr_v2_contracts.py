from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[2]
CODE_ROOT = PROJECT_ROOT / "code"
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

from cvsrffi import phase1_fcr_types as fcr_types  # noqa: E402


def _valid_metadata(batch: int = 2, eta_dim: int = 8) -> dict[str, object]:
    return {
        "physical_sample_id": tuple(f"physical:{index}" for index in range(batch)),
        "content_record_id": tuple(f"content:{index}" for index in range(batch)),
        "crop_offset": torch.arange(batch, dtype=torch.long),
        "common_preamble_id": tuple(f"preamble:{index}" for index in range(batch)),
        "tx_id": torch.arange(batch, dtype=torch.long),
        "rx_i": torch.zeros(batch, dtype=torch.long),
        "day_i": torch.zeros(batch, dtype=torch.long),
        "view_type": tuple("leo_clear_weak" for _ in range(batch)),
        "link_condition": tuple("clear" for _ in range(batch)),
        "excitation_bin": torch.zeros(batch, dtype=torch.long),
        "eta_schema_version": "fcr-v2/eta-v1",
        "eta": torch.zeros(batch, eta_dim, dtype=torch.float32),
        "eta_valid_mask": torch.ones(batch, eta_dim, dtype=torch.bool),
    }


def test_v2_metadata_from_mapping_accepts_valid_contract() -> None:
    meta = fcr_types.FCRV2Metadata.from_mapping(_valid_metadata(), batch_size=2)

    assert meta.batch_size == 2
    assert meta.eta_schema_version == "fcr-v2/eta-v1"
    assert meta.eta.shape == (2, 8)
    assert meta.eta_valid_mask.shape == (2, 8)
    assert meta.eta_valid_mask.dtype == torch.bool
    assert meta.physical_sample_id == ("physical:0", "physical:1")
    assert meta.view_type == ("leo_clear_weak", "leo_clear_weak")


def test_v2_metadata_shape_mismatch_fails_closed() -> None:
    meta = _valid_metadata()
    meta["eta_valid_mask"] = torch.ones(1, 8, dtype=torch.bool)

    with pytest.raises(ValueError, match="eta_valid_mask"):
        fcr_types.FCRV2Metadata.from_mapping(meta, batch_size=2)


def test_v2_metadata_unknown_schema_fails_closed() -> None:
    meta = _valid_metadata()
    meta["eta_schema_version"] = "legacy-v0"

    with pytest.raises(ValueError, match="eta_schema_version"):
        fcr_types.FCRV2Metadata.from_mapping(meta, batch_size=2)


def test_v2_factor_output_freezes_decoder_inputs() -> None:
    out = fcr_types.FCRV2FactorOutput(
        z_s=torch.zeros(2, 4, 16),
        z_f_id=torch.nn.functional.normalize(torch.randn(2, 160), dim=1),
        z_f_dev=torch.randn(2, 160, requires_grad=True),
        z_n={"alpha": torch.zeros(2, 2), "beta": torch.zeros(2, 2)},
        s_hat=torch.zeros(2, 2, 64, dtype=torch.complex64),
        delta_f=torch.zeros(2, 2, 64, dtype=torch.complex64),
    )

    assert out.z_s.shape == (2, 4, 16)
    assert out.z_f_id.shape == (2, 160)
    assert out.z_f_dev.requires_grad
    assert set(out.z_n) == {"alpha", "beta"}
    assert out.decoder_inputs() == (out.s_hat, out.delta_f, out.z_n)


def test_v2_capability_state_requires_explicit_reasons() -> None:
    state = fcr_types.FCRV2CapabilityState(
        eta_ready=True,
        decoder_ready=False,
        swap_ready=False,
        fingerprint_ready=True,
        reasons={"decoder": "warmup", "swap": "insufficient_pair_coverage"},
    )

    assert state.eta_ready is True
    assert state.decoder_ready is False
    assert state.swap_ready is False
    assert state.fingerprint_ready is True
    assert state.reasons["swap"] == "insufficient_pair_coverage"


def test_v2_loss_output_tracks_active_losses_and_weights() -> None:
    total = torch.tensor(1.25)
    shared = torch.tensor(0.25)
    self_loss = torch.tensor(1.0)
    out = fcr_types.FCRV2LossOutput(
        total=total,
        components={"self": self_loss, "shared_f": shared},
        metrics={"self": 1.0, "shared_f": 0.25},
        active_losses=frozenset({"self", "shared_f"}),
        weights={"self": 0.1, "shared_f": 0.2},
        blocked={"swap": "MECHANISM_NOT_ACTIVATED:pair_coverage"},
    )

    assert out.total is total
    assert out.active_losses == frozenset({"self", "shared_f"})
    assert out.weights["shared_f"] == 0.2
    assert out.blocked["swap"].startswith("MECHANISM_NOT_ACTIVATED:")


def test_cross_decode_uses_destination_fingerprint() -> None:
    pytest.importorskip("cvsrffi.phase1_fcr_v2_losses", reason="cross_decode lands in Task 4")
