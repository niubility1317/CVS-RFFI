from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from types import MappingProxyType

import pytest


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


def test_prepare_cli_requires_and_accepts_pr160_extractor_binding() -> None:
    module = _load_cli_module()
    fixed = [
        "prepare",
        "--d108-plan-manifest",
        "C:/sealed/d108_plan.json",
        "--d108-plan-manifest-sha256",
        "a" * 64,
        "--d108-context-manifest",
        "C:/sealed/d108_context.json",
        "--d108-context-manifest-sha256",
        "b" * 64,
        "--fa-asset",
        "C:/sealed/fa_asset.wire",
        "--fa-asset-sha256",
        "c" * 64,
        "--method-lock",
        "C:/sealed/method_lock.json",
        "--method-lock-sha256",
        "d" * 64,
        "--output-dir",
        "C:/output/prepared",
    ]
    with pytest.raises(SystemExit):
        module.parse_args(fixed)
    args = module.parse_args(
        fixed[:-2]
        + [
            "--pr160-extractor-runtime",
            "C:/sealed/d92_pr160_extractor_runtime.pt",
            "--pr160-extractor-runtime-sha256",
            "e" * 64,
        ]
        + fixed[-2:]
    )
    assert args.pr160_extractor_runtime == Path(
        "C:/sealed/d92_pr160_extractor_runtime.pt"
    )
    assert args.pr160_extractor_runtime_sha256 == "e" * 64
