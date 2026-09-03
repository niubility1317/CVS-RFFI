from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[2]
CODE_ROOT = PROJECT_ROOT / "code"
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

from cvsrffi.phase1_fcr_types import FCRDecodeOutput, FCRV2CapabilityState, FCRV2FactorOutput  # noqa: E402
from cvsrffi.phase1_fcr_v2_losses import compute_fcr_v2_losses, cross_decode, necessity_loss  # noqa: E402
from cvsrffi.phase1_fcr_v2_schedule import FCRV2Schedule  # noqa: E402


class _UnitNormalizer:
    def normalize(self, name: str, value: torch.Tensor) -> torch.Tensor:
        return value / value.detach().clamp_min(1e-6)


def _ready(**overrides: object) -> FCRV2CapabilityState:
    return FCRV2CapabilityState(
        eta_ready=bool(overrides.pop("eta_ready", True)),
        decoder_ready=bool(overrides.pop("decoder_ready", True)),
        swap_ready=bool(overrides.pop("swap_ready", True)),
        fingerprint_ready=bool(overrides.pop("fingerprint_ready", True)),
        reasons=dict(overrides.pop("reasons", {})),
    )


def _factors(*, fill: float, delta_fill: float, nuisance_name: str) -> FCRV2FactorOutput:
    return FCRV2FactorOutput(
        z_s=torch.full((2, 4, 16), fill),
        z_f_id=torch.nn.functional.normalize(torch.randn(2, 160), dim=1),
        z_f_dev=torch.full((2, 160), fill),
        z_n={nuisance_name: torch.full((2, 3), fill)},
        s_hat=torch.full((2, 2, 32), fill + 0.0j, dtype=torch.complex64),
        delta_f=torch.full((2, 2, 32), delta_fill + 0.0j, dtype=torch.complex64),
    )


def _components(value: float = 1.0) -> dict[str, torch.Tensor]:
    names = (
        "identity_ce",
        "prototype",
        "tail",
        "self",
        "shared_f",
        "shared_s",
        "response",
        "eta",
        "swap",
        "cycle",
        "need",
        "transplant",
        "physical",
        "factor",
    )
    return {name: torch.tensor(float(value)) for name in names}


@pytest.mark.parametrize(
    ("row", "active"),
    [
        ("S1", {"self", "shared_f"}),
        ("S2", {"self", "shared_s"}),
        ("S4", {"self", "shared_f", "shared_s", "swap"}),
    ],
)
def test_row_activates_only_registered_losses(row: str, active: set[str]) -> None:
    schedule = FCRV2Schedule()

    state = schedule.state(epoch=120, row=row, capabilities=_ready())

    assert state.active_losses == frozenset(active)


def test_failed_capability_zeroes_scale_and_records_reason() -> None:
    schedule = FCRV2Schedule()

    state = schedule.state(
        epoch=120,
        row="S4",
        capabilities=_ready(swap_ready=False, reasons={"swap": "pair_coverage"}),
    )

    assert "swap" not in state.active_losses
    assert state.scales["swap"] == 0.0
    assert state.blocked["swap"] == "MECHANISM_NOT_ACTIVATED:pair_coverage"


def test_necessity_is_relative_drop_f_gap() -> None:
    loss = necessity_loss(full_error=torch.tensor(2.0), drop_error=torch.tensor(5.0))

    torch.testing.assert_close(loss, torch.tensor(1.5))


def test_cross_decode_uses_destination_response_and_nuisance() -> None:
    source = _factors(fill=1.0, delta_fill=2.0, nuisance_name="src")
    destination = _factors(fill=3.0, delta_fill=7.0, nuisance_name="dst")

    class _Decoder:
        def __call__(self, s_hat: torch.Tensor, delta_f: torch.Tensor, z_n: dict[str, torch.Tensor]) -> FCRDecodeOutput:
            self.last_s_hat = s_hat
            self.last_delta_f = delta_f
            self.last_z_n = z_n
            return FCRDecodeOutput(
                mu_iq=torch.zeros_like(source.s_hat.real),
                log_variance=torch.zeros(source.s_hat.size(0), source.s_hat.size(-1)),
                delta_f=delta_f,
            )

    decoder = _Decoder()
    decoded = cross_decode(source, destination, decoder)

    assert decoder.last_s_hat is source.s_hat
    assert decoder.last_delta_f is destination.delta_f
    assert decoder.last_z_n is destination.z_n
    assert decoded.delta_f is destination.delta_f


def test_labeled_losses_keep_supervised_weights_after_ema_normalization() -> None:
    out = compute_fcr_v2_losses(
        {
            "epoch": 50,
            "role": "L_s",
            "capabilities": _ready(),
            "components": _components(),
        },
        row="C3",
        ema_normalizer=_UnitNormalizer(),
    )

    torch.testing.assert_close(out.total, torch.tensor(1.275))
    assert out.active_losses == frozenset({"self"})
    assert out.weights["identity_ce"] == 1.0
    assert out.weights["prototype"] == 0.10
    assert out.weights["tail"] == 0.075
    assert out.weights["self"] == 0.10


def test_unlabeled_losses_apply_fixed_total_fcr_scale() -> None:
    out = compute_fcr_v2_losses(
        {
            "epoch": 120,
            "role": "U_s",
            "capabilities": _ready(),
            "components": _components(),
        },
        row="S3",
        ema_normalizer=_UnitNormalizer(),
    )

    torch.testing.assert_close(out.total, torch.tensor(0.1225))
    assert out.active_losses == frozenset({"self", "shared_f", "shared_s"})
    assert out.weights["self"] == pytest.approx(0.035)
    assert out.weights["shared_f"] == pytest.approx(0.07)
    assert out.weights["shared_s"] == pytest.approx(0.0175)
