from __future__ import annotations

import importlib.util
import json
import shlex
from pathlib import Path, PurePosixPath


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = REPO_ROOT / "code" / "scripts" / "run_d106_real_integration.py"
ARTIFACT_ROOT = (
    REPO_ROOT
    / "automation_reports"
    / "CV-SincNet"
    / "d106_rdce_gtsm_20260801_r1"
)
FIXTURE_PATH = (
    ARTIFACT_ROOT
    / "artifacts"
    / "d106_real_integration_fixture_deefd57c_r5.json"
)
HANDOFF_PATH = (
    ARTIFACT_ROOT / "d106_real_integration_runner_handoff_deefd57c_r5.md"
)
RUN_ID = "d106_real_integration_deefd57c_20260801_r5"
RUN_ROOT = f"/home/szu2070436088/2510044040/CV-SincNet/runs/{RUN_ID}"
EXPECTED_FIXTURE = f"{RUN_ROOT}/input/{FIXTURE_PATH.name}"
EXPECTED_OUTPUT = f"{RUN_ROOT}/output"
EXPECTED_SCRIPT = f"{RUN_ROOT}/source/code/scripts/run_d106_real_integration.py"
RUN_LOCAL_FIELDS = {
    "source_split_manifest",
    "disjoint_receipt",
    "ls_archive",
    "runtime_manifest",
    "method_lock",
    "construction_code",
}

SPEC = importlib.util.spec_from_file_location("d106_handoff_runner", SCRIPT_PATH)
assert SPEC is not None and SPEC.loader is not None
runner = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(runner)


def test_r5_fixture_is_canonical_and_all_bound_paths_are_absolute() -> None:
    payload = FIXTURE_PATH.read_bytes()
    fixture = json.loads(payload.decode("utf-8"))
    assert payload == runner._canonical_bytes(fixture)
    assert set(fixture) == runner.FIXTURE_FIELDS
    assert fixture["release_commit"] == "deefd57c4185a5343f87772be78b5038c37e6217"
    for path_field, _sha_field in runner.PATH_HASH_FIELDS:
        path = PurePosixPath(fixture[path_field])
        assert path.is_absolute()
        assert ".." not in path.parts
        if path_field in RUN_LOCAL_FIELDS:
            assert fixture[path_field].startswith(RUN_ROOT + "/")
            assert "d106_real_integration_deefd57c_20260801_r4" not in fixture[
                path_field
            ]


def test_r5_handoff_launch_uses_exact_absolute_fixture_and_output() -> None:
    text = HANDOFF_PATH.read_text(encoding="utf-8")
    assert "--fixture ../" not in text
    commands = [
        line
        for line in text.splitlines()
        if line.startswith("CUDA_VISIBLE_DEVICES=0 ")
    ]
    assert len(commands) == 1
    command = commands[0]
    argv = shlex.split(command, posix=True)
    assert argv[2] == EXPECTED_SCRIPT
    fixture_index = argv.index("--fixture") + 1
    output_index = argv.index("--output-dir") + 1
    assert argv[fixture_index] == EXPECTED_FIXTURE
    assert argv[output_index] == EXPECTED_OUTPUT
    assert PurePosixPath(argv[fixture_index]).is_absolute()
    assert PurePosixPath(argv[output_index]).is_absolute()
    assert argv[fixture_index].startswith(RUN_ROOT + "/")
    assert argv[output_index].startswith(RUN_ROOT + "/")
    assert "retry：`NOT_AUTHORIZED`" in text
    assert "不得启动第二次" in text
    assert "d106_real_integration_deefd57c_20260801_r4/input" not in text


def test_r5_handoff_has_four_absolute_local_to_remote_mappings() -> None:
    text = HANDOFF_PATH.read_text(encoding="utf-8")
    mapping_lines = [
        line
        for line in text.splitlines()
        if line.startswith("|`E:\\type10-7\\")
    ]
    assert len(mapping_lines) == 4
    for line in mapping_lines:
        cells = [cell.strip().strip("`") for cell in line.strip("|").split("|")]
        local_path, _sha256, remote_path = cells
        assert Path(local_path).is_absolute()
        assert PurePosixPath(remote_path).is_absolute()
        assert remote_path.startswith(RUN_ROOT + "/")
        assert "d106_real_integration_deefd57c_20260801_r4" not in remote_path
