from __future__ import annotations

import importlib.util
from pathlib import Path
import sys


SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "code"
    / "scripts"
    / "probe_d88_ground_sigma_pareto_guard.py"
)


def _load():
    spec = importlib.util.spec_from_file_location("test_d88_probe", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_d88_probe_locks_single_mechanism() -> None:
    module = _load()
    assert module.ARM == "ground_sigma_pareto_guard_centered_head"
    assert "every registered class" in module.FORMULA
    assert "query" not in module.FORMULA.lower()
    parser = module.build_parser()
    known, remaining = parser.parse_known_args(
        ["--d88-arm", module.ARM, "--output", "unused"]
    )
    assert known.d88_arm == module.ARM
    assert remaining == ["--output", "unused"]
