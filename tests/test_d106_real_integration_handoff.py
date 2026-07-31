from __future__ import annotations

import importlib.util
import hashlib
import json
import shlex
from pathlib import Path, PurePosixPath
import zipfile


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
R6_RUN_ID = "d106_real_integration_44e33eab_20260801_r6"
R6_RUN_ROOT = f"/home/szu2070436088/2510044040/CV-SincNet/runs/{R6_RUN_ID}"
R6_FIXTURE_PATH = (
    ARTIFACT_ROOT / "artifacts" / "d106_real_integration_fixture_44e33eab_r6.json"
)
R6_HANDOFF_PATH = ARTIFACT_ROOT / "d106_real_integration_runner_handoff_44e33eab_r6.md"
R6_SOURCE_ARCHIVE = (
    ARTIFACT_ROOT / "artifacts" / "d106_real_integration_source_44e33eab.zip"
)

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


def test_r6_fixture_and_launch_are_absolute_and_isolated() -> None:
    payload = R6_FIXTURE_PATH.read_bytes()
    fixture = json.loads(payload.decode("utf-8"))
    assert payload == runner._canonical_bytes(fixture)
    assert set(fixture) == runner.FIXTURE_FIELDS
    assert fixture["release_commit"] == "44e33eab9bcc9352456e5f3a8ae85405c603a36c"
    for path_field, _sha_field in runner.PATH_HASH_FIELDS:
        path = PurePosixPath(fixture[path_field])
        assert path.is_absolute()
        assert ".." not in path.parts
        if path_field in RUN_LOCAL_FIELDS:
            assert fixture[path_field].startswith(R6_RUN_ROOT + "/")
            assert "d106_real_integration_deefd57c_20260801_r4" not in fixture[
                path_field
            ]
            assert "d106_real_integration_deefd57c_20260801_r5" not in fixture[
                path_field
            ]

    text = R6_HANDOFF_PATH.read_text(encoding="utf-8")
    commands = [
        line
        for line in text.splitlines()
        if line.startswith("CUDA_VISIBLE_DEVICES=0 ")
    ]
    assert len(commands) == 1
    argv = shlex.split(commands[0], posix=True)
    assert argv[2] == f"{R6_RUN_ROOT}/source/code/scripts/run_d106_real_integration.py"
    assert argv[argv.index("--fixture") + 1] == (
        f"{R6_RUN_ROOT}/input/{R6_FIXTURE_PATH.name}"
    )
    assert argv[argv.index("--output-dir") + 1] == f"{R6_RUN_ROOT}/output"
    assert "--fixture ../" not in text
    assert "retry：`NOT_AUTHORIZED`" in text
    assert "不得启动第二次" in text
    assert f"`{runner.COMPLETION_NAME}`" in text
    assert "D106_REAL_INTEGRATION_COMPLETE.json" not in text


def test_r6_handoff_has_four_absolute_local_to_remote_mappings() -> None:
    text = R6_HANDOFF_PATH.read_text(encoding="utf-8")
    mapping_lines = [
        line
        for line in text.splitlines()
        if line.startswith("|`E:\\type10-7\\")
    ]
    assert len(mapping_lines) == 4
    for line in mapping_lines:
        cells = [cell.strip().strip("`") for cell in line.strip("|").split("|")]
        local_path, expected_sha256, remote_path = cells
        assert Path(local_path).is_absolute()
        assert hashlib.sha256(Path(local_path).read_bytes()).hexdigest() == expected_sha256
        assert PurePosixPath(remote_path).is_absolute()
        assert remote_path.startswith(R6_RUN_ROOT + "/")
        assert "d106_real_integration_deefd57c_20260801_r4" not in remote_path
        assert "d106_real_integration_deefd57c_20260801_r5" not in remote_path


def test_r6_archive_contains_exact_release_dependency_closure() -> None:
    assert hashlib.sha256(R6_SOURCE_ARCHIVE.read_bytes()).hexdigest() == (
        "91c5a30b156972482476b4befdae4bbbffbb66a0b1a14ad5205f58fb8f17b6fe"
    )
    expected = {
        "source/code/scripts/run_d106_real_integration.py": (
            "3bb8acb3c48ad371c6c0b51f20fbefb0821445f2b7ecfaecd54de71e8a39de27"
        ),
        "source/code/cvsrffi/stage2_d106_phase1_tap.py": (
            "5a63a5935748f17a1efcbf4069d5c80c1d99a8e813330a2c3a15895483c53e9b"
        ),
        "source/code/baseline_origin_sat_view.py": (
            "fa7221ae505a51a2afc2a51b857675ac4a5384b004d5a4f36e10dafc9d4f8ace"
        ),
        "source/code/model.py": (
            "afc6e6266a09fd5f5be967fed85254c6c92fa0241a0336fd5ffa3eb12aa1c417"
        ),
        "source/code/model_dual_cvsincnet.py": (
            "11b56f2a763eb49de21d0a566d8e4420538cb7af310c9338f4e8d4a21c42c235"
        ),
        "source/configs/d106_candidate_runtime_manifest_20260801.json": (
            "0e8bc733ce9650aea3463da90242f97e969210ca8a95983fee032f1474f87cb2"
        ),
        "source/configs/d106_rdce_method_lock_20260801.json": (
            "e7a1982b4bdeaf5b8179993ce78f4a2af26965d8f4a3239440dbe636ebf14cc1"
        ),
    }
    with zipfile.ZipFile(R6_SOURCE_ARCHIVE, "r") as archive:
        for name, expected_sha256 in expected.items():
            assert hashlib.sha256(archive.read(name)).hexdigest() == expected_sha256
