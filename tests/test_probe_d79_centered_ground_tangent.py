from __future__ import annotations

import importlib.util
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "code" / "scripts" / "probe_d79_centered_ground_tangent.py"


def _load():
    spec = importlib.util.spec_from_file_location("test_d79_probe", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_formula_locks_centered_bias_compile_and_no_query():
    module = _load()
    formula = module.FORMULA.lower()
    assert "subtract" in formula
    assert "target-support mean" in formula
    assert "delta_b=-delta_w*support_mean" in formula
    assert "exactly zero" in formula
    assert "query" not in formula
    assert module.core.OPTIMIZER_STEPS == 20
    assert module.d43.ARM_STRUCTURES[module.ARM] == module.d62.STRUCTURE


def test_runner_resource_accounting_includes_ground_component():
    module = _load()

    class Runner:
        @staticmethod
        def _evaluate_d42_fold(*_args, **_kwargs):
            return {
                "resource": {
                    "d79_ground_component_logical_state_bytes": 25428,
                    "persistent_state_bytes": 8583,
                    "persistent_state_cap_bytes": 256 * 1024,
                }
            }

    module._install_runner_resource_accounting(Runner)
    resource = Runner._evaluate_d42_fold()["resource"]
    assert resource["d79_compiled_affine_state_bytes"] == 8583
    assert resource["d79_component_inclusive_persistent_state_bytes"] == 34011
    assert resource["persistent_state_cap_pass"] is True


def test_source_locks_single_affine_protocol_fields():
    source = SCRIPT.read_text(encoding="utf-8")
    assert "compile_with_centered_bias" in source
    assert '"d79_bias_compile_mac_equivalents"' in source
    assert '"d79_single_affine_state_only"' in source
    assert '"d79_ground_class_score_access"' in source
    assert '"d79_ground_component_update_access"' in source
    assert "D79_PROBE_METADATA.json" in source
