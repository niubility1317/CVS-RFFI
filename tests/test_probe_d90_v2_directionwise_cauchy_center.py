from __future__ import annotations

import importlib.util
from pathlib import Path
import sys


SCRIPT = Path(__file__).resolve().parents[1] / "code" / "scripts" / "probe_d90_v2_directionwise_cauchy_center.py"


def test_probe_formula_and_parser_lock() -> None:
    spec = importlib.util.spec_from_file_location("test_d90_probe", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    assert module.ARM == "v2_directionwise_cauchy_center"
    assert "independent Cauchy robust center per retained ground direction" in module.FORMULA
    known, remaining = module.build_parser().parse_known_args([
        "--d90-arm", module.ARM,
        "--output", "unused",
    ])
    assert known.d90_arm == module.ARM
    assert remaining == ["--output", "unused"]
