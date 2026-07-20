from __future__ import annotations

import importlib.util
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
PROBE = ROOT / "code" / "scripts" / "probe_d91_crossfit_consensus_sigma_margin.py"


def _load_probe():
    spec = importlib.util.spec_from_file_location("d91_probe_test_module", PROBE)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_probe_lock_is_threshold_free_and_query_free() -> None:
    probe = _load_probe()
    assert probe.ARM == "crossfit_consensus_ground_sigma_margin_head"
    assert "physical-rank OOF fold" in probe.FORMULA
    assert "without a threshold" in probe.FORMULA
    assert "single INT8 affine head" in probe.FORMULA


def test_resource_increment_is_positive_and_small() -> None:
    probe = _load_probe()
    inherited = probe._D87_RESOURCE_UPPER_BOUNDS(
        k_shot=8, class_count=11, dimension=288,
        lda_macs=1_000_000, ground_statistics_macs=216_724,
    )
    bounded = probe._resource_upper_bounds(
        k_shot=8, class_count=11, dimension=288,
        lda_macs=1_000_000, ground_statistics_macs=216_724,
    )
    extra = bounded["total_added"] - inherited["total_added"]
    assert 0 < extra < 10_000_000


def test_probe_source_keeps_protocol_closure() -> None:
    source = PROBE.read_text(encoding="utf-8")
    required = (
        "verified_d91_query_rows_used",
        "old_new_role_specific_branch",
        "class_id_specific_formula",
        "physical_group_crossfit_preserved",
        "forced_nonpromotable",
    )
    assert all(token in source for token in required)
