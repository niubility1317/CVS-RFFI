from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from types import MappingProxyType


def _load_cli_module():
    path = Path(__file__).resolve().parents[1] / "code" / "scripts" / "run_next_r5_fa_target125.py"
    spec = importlib.util.spec_from_file_location("run_next_r5_fa_target125", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_json_plain_serializes_nested_immutable_runtime_result() -> None:
    module = _load_cli_module()
    value = MappingProxyType(
        {
            "status": "PREPARED",
            "nested": MappingProxyType({"counts": (125, 375, 1500)}),
        }
    )

    plain = module._json_plain(value)

    assert plain == {"status": "PREPARED", "nested": {"counts": [125, 375, 1500]}}
    assert json.loads(json.dumps(plain)) == plain
