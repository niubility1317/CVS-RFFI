from __future__ import annotations

import ast
import inspect

import pytest

from SSDG import train_ssdg


def parse(argv: list[str]):
    return train_ssdg.parse(argv)


def test_bicad_entry_forces_concat_leo_weak_contract() -> None:
    args = parse(["--phase1_method", "bicad_xr", "--candidate_id", "D5"])

    resolved = train_ssdg.resolve_bicad_protocol(args)

    assert resolved.use_concat_sat_channel_aug
    assert resolved.concat_sat_ce_only
    assert resolved.concat_sat_ce_weight == pytest.approx(0.68)
    assert resolved.concat_sat_start_epoch == 80
    assert resolved.sat_train_scenarios == (
        "leo_clear_weak,leo_low_elev_weak,leo_rain_weak"
    )


def test_bicad_route_is_explicit_and_legacy_route_stays_lazy() -> None:
    bicad_args = parse(["--phase1_method", "bicad_xr", "--candidate_id", "D5"])
    legacy_args = parse([])

    assert train_ssdg.route_phase1_method(bicad_args) == "bicad_xr"
    assert train_ssdg.route_phase1_method(legacy_args) == "legacy"

    tree = ast.parse(inspect.getsource(train_ssdg))
    eager_bicad_imports = []
    for node in tree.body:
        if isinstance(node, ast.Import):
            eager_bicad_imports.extend(
                alias.name
                for alias in node.names
                if "phase1_bicad_xr" in alias.name
            )
        elif isinstance(node, ast.ImportFrom) and node.module:
            if "phase1_bicad_xr" in node.module:
                eager_bicad_imports.append(node.module)
    assert eager_bicad_imports == []
