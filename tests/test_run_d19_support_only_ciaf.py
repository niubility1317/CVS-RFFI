from __future__ import annotations

import inspect
import sys
from pathlib import Path


SCRIPTS = Path(__file__).resolve().parents[1] / "code" / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import run_d19_support_only_ciaf as runner


def test_cli_has_no_query_truth_role_quota_or_global_assignment_surface() -> None:
    destinations = {action.dest for action in runner.build_parser()._actions}
    forbidden = ("query", "truth", "role", "quota", "assignment")
    assert not any(token in name.lower() for name in destinations for token in forbidden)


def test_run_requires_preopen_component_and_class_binding_inputs() -> None:
    parameters = inspect.signature(runner.run).parameters
    for name in (
        "component_dir",
        "expected_component_manifest_sha256",
        "class_binding_path",
        "expected_class_binding_sha256",
    ):
        assert name in parameters
    assert runner.MODE == "development_select_unverified_component"
    assert runner.SUPPORT_QUERY_DISJOINTNESS_STATUS == "SUPPORT_ONLY_NO_QUERY_CLAIM"
