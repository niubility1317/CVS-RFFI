from __future__ import annotations

import importlib.util
from pathlib import Path
import sys


SCRIPT = Path(__file__).resolve().parents[1] / "code" / "scripts" / "probe_d89_v2_radius_cauchy_center.py"


def test_probe_formula_and_parser_lock() -> None:
    spec = importlib.util.spec_from_file_location("test_d89_probe", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    assert module.ARM == "v2_radius_reliability_cauchy_center"
    assert "84 domain-class cells" in module.FORMULA
    assert "unchanged D81" in module.FORMULA
    known, remaining = module.build_parser().parse_known_args([
        "--d89-arm", module.ARM,
        "--ground-v2-component-dir", "component",
        "--ground-v2-manifest-sha256", "m",
        "--expected-checkpoint-sha256", "c",
        "--expected-class-handle-binding-sha256", "b",
        "--expected-pre-sign-content-root-sha256", "p",
        "--output", "unused",
    ])
    assert known.d89_arm == module.ARM
    assert remaining == ["--output", "unused"]
