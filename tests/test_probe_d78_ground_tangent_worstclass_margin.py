from __future__ import annotations

import importlib.util
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "code" / "scripts" / "probe_d78_ground_tangent_worstclass_margin.py"


def _load():
    spec = importlib.util.spec_from_file_location("test_d78_probe", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_formula_uses_ground_as_tangent_not_class_score() -> None:
    module = _load()
    formula = module.FORMULA.lower()
    assert "domain tangent basis" in formula
    assert "top-2 logistic" in formula
    assert "20 accepted steps" in formula
    assert "class-agnostic" in formula
    assert "anchor" not in formula
    assert "query" not in formula


def test_arm_uses_d62_structure_and_fixed_twenty_steps() -> None:
    module = _load()
    assert module.ARM == "ground_tangent_worstclass_top2_margin"
    assert module.STRUCTURE == module.d62.STRUCTURE
    assert module.core.OPTIMIZER_STEPS == 20
    assert module.d43.ARM_STRUCTURES[module.ARM] == module.STRUCTURE


def test_resource_upper_bound_is_complete_and_additive() -> None:
    module = _load()
    values = module._resource_upper_bounds(
        k_shot=8,
        class_count=11,
        dimension=288,
        lda_macs=12345,
        ground_statistics_macs=90000000,
    )
    non_lda_parts = (
        values["oof_gradient_mac_upper_bound"]
        + values["frank_wolfe_mac_upper_bound"]
        + values["oof_ce_audit_mac_upper_bound"]
        + values["preconditioner_application_macs"]
        + values["affine_compile_mac_equivalents"]
        + values["ground_statistics_macs"]
    )
    assert values["non_lda_total"] == non_lda_parts
    assert values["total_added"] == 12345 + non_lda_parts


def test_runner_resource_accounting_keeps_component_in_main_state() -> None:
    module = _load()

    class Runner:
        @staticmethod
        def _evaluate_d42_fold(*_args, **_kwargs):
            return {
                "resource": {
                    "d78_ground_component_logical_state_bytes": 25428,
                    "persistent_state_bytes": 8583,
                    "persistent_state_cap_bytes": 256 * 1024,
                }
            }

    module._install_runner_resource_accounting(Runner)
    resource = Runner._evaluate_d42_fold()["resource"]
    assert resource["d78_compiled_affine_state_bytes"] == 8583
    assert resource["d78_component_inclusive_persistent_state_bytes"] == 34011
    assert resource["persistent_state_bytes"] == 34011
    assert resource["persistent_state_cap_pass"] is True


def test_source_locks_protocol_and_single_affine_state() -> None:
    source = SCRIPT.read_text(encoding="utf-8")
    assert '"d78_ground_class_score_access": False' in source
    assert '"d78_ground_component_update_access": False' in source
    assert '"d78_query_extra_mac_equivalents": 0' in source
    assert '"d78_query_extra_state_bytes": 0' in source
    assert '"d78_dense_query_graph_bytes": 0' in source
    assert '"d78_single_affine_state_only": True' in source
    assert '"component_formal_phase2_eligible": False' in source
    assert "selected_only_full_k10_refit_allowed" in source
