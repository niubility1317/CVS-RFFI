from __future__ import annotations

import importlib.util
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
PROBE = ROOT / "code" / "scripts" / "probe_d87_ground_radius_sigma_margin.py"


def _load_probe():
    spec = importlib.util.spec_from_file_location("d87_probe_test_module", PROBE)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_probe_lock_uses_nonquadratic_grouped_sigma_margin() -> None:
    probe = _load_probe()
    assert probe.ARM == "ground_radius_sigma_margin_centered_head"
    assert "sqrt(2*median_class_p90_radius)" in probe.FORMULA
    assert "1/2,1/4,1/4" in probe.FORMULA
    assert "physical-rank OOF" in probe.FORMULA
    assert "single INT8 affine head" in probe.FORMULA


def test_sigma_resource_bound_is_positive_and_below_formal_scale() -> None:
    probe = _load_probe()
    inherited = probe.d78._resource_upper_bounds(
        k_shot=8,
        class_count=11,
        dimension=288,
        lda_macs=1_000_000,
        ground_statistics_macs=216_724,
    )
    bounded = probe._resource_upper_bounds(
        k_shot=8,
        class_count=11,
        dimension=288,
        lda_macs=1_000_000,
        ground_statistics_macs=216_724,
    )
    added = bounded["total_added"] - inherited["total_added"]
    assert added > 0
    assert added < 500_000_000
    assert bounded["non_lda_total"] > inherited["non_lda_total"]


def test_probe_source_contains_v2_and_query_free_fail_closed_checks() -> None:
    source = PROBE.read_text(encoding="utf-8")
    required = (
        "allow_pending_outer_joint_seal_development=True",
        "ground_target_identity_mapping_access",
        "counterfactual_views_count_as_physical_samples",
        "physical_group_crossfit_preserved",
        "d79_query_extra_mac_equivalents",
        "d79_query_extra_state_bytes",
        "d79_single_affine_state_only",
        "ground_component_bitwise_unchanged",
        "outer_joint_seal_verified",
    )
    assert all(token in source for token in required)

