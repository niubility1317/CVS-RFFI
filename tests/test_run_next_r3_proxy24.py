from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest


SCRIPT = Path(__file__).resolve().parents[1] / "code" / "scripts" / "run_next_r3_proxy24.py"
SPEC = importlib.util.spec_from_file_location("run_next_r3_proxy24_under_test", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
runner = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = runner
SPEC.loader.exec_module(runner)


def test_missing_real_inputs_fail_closed(tmp_path: Path):
    absent = tmp_path / "absent.npz"
    args = SimpleNamespace(
        received_iq=absent,
        received_iq_sha256="0" * 64,
        source_held_archive=absent,
        source_held_archive_sha256="1" * 64,
        phase1_cells=absent,
        phase1_cells_sha256="2" * 64,
    )
    with pytest.raises(runner.MissingRealInputArtifacts, match=r"^MISSING_REAL_INPUT_ARTIFACTS"):
        runner._load_real_rows(args)


def test_new_run_root_refuses_overwrite(tmp_path: Path):
    root = runner._new_root(tmp_path / "run")
    assert (root / "rows").is_dir()
    with pytest.raises(runner.NextR3Proxy24Error, match="new absolute child"):
        runner._new_root(root)
