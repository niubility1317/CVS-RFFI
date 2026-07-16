from __future__ import annotations

import importlib.util
from pathlib import Path


SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "code"
    / "scripts"
    / "run_d13_support_only_intrusion_guard.py"
)
SPEC = importlib.util.spec_from_file_location("run_d13_support_only", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
runner = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(runner)


def test_formal_candidate_grid_is_constant_only_and_has_true_zero_fallback() -> None:
    candidates = runner._candidates()
    assert len(candidates) == 6
    assert all(value.mode == "constant" for value in candidates)
    zero = [value for value in candidates if value.force_zero]
    assert len(zero) == 1
    assert zero[0].candidate_id == "d13_delta0_base"
    assert zero[0].cap == 0.0


def test_unified_hyperparameter_lock_is_deterministic_and_three_scenario_scoped() -> None:
    first = runner._candidate_lock(runner._candidates())
    second = runner._candidate_lock(runner._candidates())
    assert first == second
    assert len(first["lock_sha256"]) == 64
    assert (
        first["selection_scope"]
        == "one_hyperparameter_arm_shared_by_all_three_scenarios"
    )
    assert all(row["quantile_method"] == "linear" for row in first["candidates"])


def test_runner_source_has_no_query_payload_or_scorer_loader() -> None:
    source = SCRIPT.read_text(encoding="utf-8")
    assert "load_verified_somph_predictor_bundle" in source
    assert "query_root" not in source
    assert "scorer_root" not in source
    assert "_select_artifact" not in source
    assert "_ARTIFACT_TOKEN" not in source
