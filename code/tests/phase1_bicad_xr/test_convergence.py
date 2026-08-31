from __future__ import annotations

import math

import pytest

from cvsrffi.phase1_bicad_xr.convergence import (
    CoverageLedger,
    DGObservation,
    ConvergenceController,
)


def _observation(
    index: int,
    *,
    coverage_u: float | None = None,
    score: float | None = None,
    learning_rate: float = 1.0e-3,
) -> DGObservation:
    return DGObservation(
        updates=500 * (index + 1),
        coverage_u=coverage_u if coverage_u is not None else 3.0 + 0.5 * index,
        s_dg=score if score is not None else 0.8 + 0.0001 * index,
        learning_rate=learning_rate,
        d_logit=0.005,
        d_theta=0.0005,
        margin_q10=0.20 + 0.001 * index,
        elapsed_hours=1.0 + index,
    )


def test_coverage_ledger_tracks_u_cycles_unique_coverage_and_l_floor() -> None:
    ledger = CoverageLedger(
        u_sample_ids=("u0", "u1", "u2", "u3"),
        l_groups=((0, 1, 1), (0, 3, 1), (1, 1, 1)),
    )

    ledger.record_u(("u0", "u1"))
    ledger.record_l(((0, 1, 1), (1, 1, 1)))

    assert ledger.u_coverage == pytest.approx(0.5)
    assert ledger.u_unique_coverage == pytest.approx(0.5)
    assert ledger.l_coverage == pytest.approx(0.0)

    ledger.record_u(("u2", "u3", "u0"))
    ledger.record_l(((0, 3, 1),))

    assert ledger.u_coverage == pytest.approx(1.25)
    assert ledger.u_unique_coverage == pytest.approx(1.0)
    assert ledger.l_coverage == pytest.approx(1.0)


def test_candidate_activation_age_is_measured_from_its_last_core_mechanism() -> None:
    controller = ConvergenceController(last_mechanism_activation_coverage=2.0)

    decision = controller.observe(_observation(0, coverage_u=4.5))

    assert decision.activation_age == pytest.approx(2.5)
    assert decision.status == "CONTINUE"
    assert not decision.should_stop


def test_plateau_slope_and_two_lr_reductions_enable_scientific_stop() -> None:
    controller = ConvergenceController(last_mechanism_activation_coverage=0.0)
    learning_rates = (1.0e-3, 1.0e-3, 3.0e-4, 3.0e-4, 9.0e-5, 9.0e-5)

    decisions = [
        controller.observe(_observation(index, learning_rate=learning_rate))
        for index, learning_rate in enumerate(learning_rates)
    ]

    decision = decisions[-1]
    assert decision.plateau_slope == pytest.approx(0.0001)
    assert decision.lr_reduction_count == 2
    assert decision.status == "SCIENTIFICALLY_CONVERGED"
    assert decision.scientifically_converged
    assert decision.should_stop


def test_safety_coverage_limit_is_not_reported_as_scientific_convergence() -> None:
    controller = ConvergenceController(last_mechanism_activation_coverage=0.0)

    decision = controller.observe(_observation(0, coverage_u=12.0))

    assert decision.status == "NOT_CONVERGED_SAFETY_STOP"
    assert decision.should_stop
    assert not decision.scientifically_converged


@pytest.mark.parametrize(
    ("field_name", "value"),
    [
        ("s_dg", math.nan),
        ("d_logit", math.inf),
        ("elapsed_hours", -math.inf),
    ],
)
def test_dg_observation_rejects_nonfinite_values(field_name: str, value: float) -> None:
    values = {
        "updates": 500,
        "coverage_u": 0.5,
        "s_dg": 0.8,
        "learning_rate": 1.0e-3,
        "d_logit": 0.005,
        "d_theta": 0.0005,
        "margin_q10": 0.2,
        "elapsed_hours": 1.0,
    }
    values[field_name] = value

    with pytest.raises(ValueError, match="finite"):
        DGObservation(**values)
