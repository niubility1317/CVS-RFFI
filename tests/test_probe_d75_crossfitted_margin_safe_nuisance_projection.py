from __future__ import annotations

import importlib.util
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "code" / "scripts" / "probe_d75_crossfitted_margin_safe_nuisance_projection.py"


def _load():
    spec = importlib.util.spec_from_file_location("test_d75_probe", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_formula_is_role_blind_and_support_held() -> None:
    module = _load()
    formula = module.FORMULA
    assert "leave-one-physical-rank" in formula
    assert "any registered class" in formula
    assert "query" not in formula.lower()


def test_arm_uses_existing_d62_structure() -> None:
    module = _load()
    assert module.ARM == "crossfitted_margin_safe_nuisance_projection"
    assert module.STRUCTURE == module.d62.STRUCTURE
    assert module.d43.ARM_STRUCTURES[module.ARM] == module.STRUCTURE


def test_source_locks_identity_fallback_and_no_query_state() -> None:
    source = SCRIPT.read_text(encoding="utf-8")
    assert "crossfitted_margin_rejected_exact_d62_fallback" in source
    assert '"d75_query_extra_mac_equivalents": 0' in source
    assert '"d75_persistent_state_extra_bytes": 0' in source
    assert '"d75_ground_component_input_count": 0' in source
    assert '"d75_dense_query_graph_bytes": 0' in source
