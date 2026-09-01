from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

from cvsrffi.stage2_marc_ot_pilot import FORMAL_ARMS, SCENARIOS, validate_pilot_config


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "code" / "scripts" / "run_stage2_marc_ot_pilot.py"
CONFIG = ROOT / "configs" / "marc_ot_k10_pilot_20260901.json"


def _module():
    spec = importlib.util.spec_from_file_location("run_stage2_marc_ot_pilot", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_cli_has_exact_smoke_pilot_score_commands_and_smoke_has_no_query_path() -> None:
    module = _module()
    parser = module.parser()
    subparsers = next(
        action for action in parser._actions if hasattr(action, "choices") and action.choices
    )
    assert tuple(subparsers.choices) == ("smoke", "pilot", "score")
    smoke = subparsers.choices["smoke"]
    smoke_options = {
        option
        for action in smoke._actions
        for option in action.option_strings
    }
    assert all("query" not in option for option in smoke_options)
    score = subparsers.choices["score"]
    score_options = {
        option
        for action in score._actions
        for option in action.option_strings
    }
    assert {"--prediction-root", "--truth-sidecar", "--output-root"} <= score_options
    assert "--manifest" not in score_options
    assert "--config" not in score_options


def test_output_root_is_immutable(tmp_path) -> None:
    module = _module()
    existing = tmp_path / "existing"
    existing.mkdir()
    with pytest.raises(FileExistsError, match="immutable"):
        module.create_immutable_output_root(existing)
    created = module.create_immutable_output_root(tmp_path / "new")
    assert created.is_dir()


def test_frozen_k10_config_is_complete_and_has_no_mrior_history_fields() -> None:
    payload = json.loads(CONFIG.read_text(encoding="utf-8-sig"))
    validated = validate_pilot_config(payload)
    assert tuple(validated["arms"]) == FORMAL_ARMS
    assert tuple(validated["scenarios"]) == SCENARIOS
    assert validated["k_shot"] == 10
    controls = json.dumps(validated["mrior_controls"], sort_keys=True).lower()
    assert "historical" not in controls
    assert "mrior_sda_result" not in controls
