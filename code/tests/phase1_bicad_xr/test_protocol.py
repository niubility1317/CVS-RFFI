from __future__ import annotations

import pytest

from SSDG import train_ssdg


@pytest.mark.parametrize(
    "flag",
    [
        "--use_fasttrust",
        "--use_mixstyle",
        "--sat_train_scenario=mixed_orbit",
    ],
)
def test_bicad_rejects_incompatible_flags(flag: str) -> None:
    with pytest.raises(ValueError, match="BiCAD-XR"):
        train_ssdg.parse_and_resolve(["--phase1_method", "bicad_xr", flag])


def test_bicad_protocol_surface_is_source_only() -> None:
    args = train_ssdg.parse(["--phase1_method", "bicad_xr", "--candidate_id", "D5"])
    resolved = train_ssdg.resolve_bicad_protocol(args)

    exposed = {name.lower() for name in vars(resolved)}
    forbidden = ("target_rx", "phase2", "support", "query", "truth")
    assert not any(any(token in name for token in forbidden) for name in exposed)
    assert resolved.target_access is False


def test_bicad_runtime_is_reconstructable_and_target_closed() -> None:
    args = train_ssdg.parse(["--phase1_method", "bicad_xr", "--candidate_id", "D5"])
    resolved = train_ssdg.resolve_bicad_protocol(args)

    runtime = train_ssdg.bicad_xr_runtime(resolved)

    assert runtime["phase1_method"] == "bicad_xr"
    assert runtime["candidate_id"] == "D5"
    assert runtime["target_access"] is False
    assert runtime["protocol"]["concat_sat_ce_only"] is True
    assert runtime["protocol"]["lambda_sat_cls"] == pytest.approx(0.68)
    assert runtime["protocol"]["concat_sat_start_epoch"] == 80
    assert runtime["protocol"]["sat_train_scenarios"] == [
        "leo_clear_weak",
        "leo_low_elev_weak",
        "leo_rain_weak",
    ]
