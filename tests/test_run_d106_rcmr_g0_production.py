from __future__ import annotations

import ast
import hashlib
import importlib.util
from pathlib import Path
import subprocess
import sys

import pytest


ROOT = Path(__file__).resolve().parents[1]
CODE_ROOT = ROOT / "code"
SCRIPT_PATH = CODE_ROOT / "scripts" / "run_d106_rcmr_g0_production.py"
CHILD_PATH = CODE_ROOT / "scripts" / "d106_rcmr_g0_clean_child.py"
SPEC = importlib.util.spec_from_file_location("d106_g0_clean_runner_test", SCRIPT_PATH)
assert SPEC is not None and SPEC.loader is not None
runner = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = runner
SPEC.loader.exec_module(runner)


def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _manifest_bytes() -> bytes:
    modules = sorted(
        {
            "cvsrffi.stage2_d106_rcmr_g0",
            "cvsrffi.stage2_d106_phase1_tap",
            "cvsrffi.stage2_d106_train_only_predecessor_lock",
            "cvsrffi.stage2_d106_rcmr_2v_qknn",
            "cvsrffi.stage2_zid_student_t_qknn",
        }
    )
    paths = {
        runner.RUNNER_RELATIVE_PATH: SCRIPT_PATH,
        runner.CHILD_RELATIVE_PATH: CHILD_PATH,
        "cvsrffi/__init__.py": CODE_ROOT / "cvsrffi" / "__init__.py",
        "model.py": CODE_ROOT / "model.py",
        "model_dual_cvsincnet.py": CODE_ROOT / "model_dual_cvsincnet.py",
        **{
            module.replace(".", "/") + ".py": CODE_ROOT
            / (module.replace(".", "/") + ".py")
            for module in modules
        },
    }
    document = {
        "schema": runner.RELEASE_MANIFEST_SCHEMA,
        "release_commit": "a" * 40,
        "registered_classes": [f"tx-{index}" for index in range(6)],
        "expected_d105_lock_authority_sha256": None,
        "runner_path": runner.RUNNER_RELATIVE_PATH,
        "child_entry_path": runner.CHILD_RELATIVE_PATH,
        "production_module_closure": modules,
        "code_files_sha256": {name: _digest(path) for name, path in paths.items()},
        "g0_expected_code_sha256": {"g0_executor_module": "b" * 64},
    }
    return runner._canonical_bytes(document)


def test_manifest_test_only_accepts_external_canonical_root_without_output(tmp_path):
    payload = _manifest_bytes()
    checked = runner.validate_release_manifest_test_only(
        payload, expected_manifest_sha256=hashlib.sha256(payload).hexdigest()
    )
    assert checked["test_only"] is True
    assert checked["source_file_count"] == 10
    assert not (tmp_path / "output").exists()


def test_manifest_test_only_rejects_wrong_external_root():
    payload = _manifest_bytes()
    with pytest.raises(runner.D106RCMRG0ProductionRunnerError, match="external SHA256"):
        runner.validate_release_manifest_test_only(
            payload, expected_manifest_sha256="f" * 64
        )


def test_windows_hard_refuses_before_any_path_child_or_output(tmp_path):
    output = (tmp_path / "must-not-exist").resolve()
    if runner.os.name == "posix":
        pytest.skip("this regression is asserted on the Windows development host")
    with pytest.raises(runner.D106RCMRG0ProductionRunnerError, match="POSIX_ONLY"):
        runner.run_d106_rcmr_g0_production(
            archive_path=tmp_path / "missing.npz",
            archive_sha256="a" * 64,
            receipt_path=tmp_path / "missing.json",
            receipt_sha256="b" * 64,
            release_manifest_path=tmp_path / "missing.manifest.json",
            expected_release_manifest_sha256="c" * 64,
            output_dir=output,
        )
    assert not output.exists()


def test_parent_rejects_internal_rows_schema_before_any_publish():
    internal = runner._canonical_bytes(
        {
            "schema": "cvs.phase1.d106.rcmr_2v_g0.internal_rows_test.v1",
            "status": "INTERNAL_ROWS_TEST_ONLY_NOT_A_PRODUCTION_ARTIFACT",
        }
    )
    with pytest.raises(runner.D106RCMRG0ProductionRunnerError, match="public production"):
        runner._extract_public_result(internal)


def test_parent_runner_never_imports_project_execution_code():
    tree = ast.parse(SCRIPT_PATH.read_text(encoding="utf-8"))
    project_imports = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            project_imports.extend(alias.name for alias in node.names if alias.name.startswith("cvsrffi"))
        elif isinstance(node, ast.ImportFrom) and node.module:
            if node.module.startswith("cvsrffi"):
                project_imports.append(node.module)
    assert project_imports == []
    source = SCRIPT_PATH.read_text(encoding="utf-8")
    assert '"-I"' in source
    assert "pass_fds=(root_fd, manifest_fd)" in source


def test_clean_child_accepts_only_actual_paths_or_result_bytes_and_has_two_modes():
    source = CHILD_PATH.read_text(encoding="utf-8")
    assert 'CHILD_MODE == "execute"' in source
    assert 'CHILD_MODE == "verify"' in source
    assert '"archive_path", "archive_sha256", "receipt_path", "receipt_sha256"' in source
    assert '"result_base64", "expected_result_sha256"' in source
    assert "predecessor_locks" not in source
    assert "rows=" not in source
    assert "D106RCMRG0ProductionRequest" in source
    assert "verify_d106_rcmr_g0_production_result_bytes" in source
    assert "with _verified_project_import_lifecycle(sources) as g0" in source


def test_verified_finder_rejects_unlisted_project_import_during_g0_execute_lifecycle():
    """A lazy project import remains fail-closed until the G0 call returns."""

    probe = f"""
import importlib.util
import sys
child_path = {str(CHILD_PATH)!r}
spec = importlib.util.spec_from_file_location('d106_child_lifecycle_probe', child_path)
child = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = child
spec.loader.exec_module(child)
child.CODE_ROOT_PATH = '/verified-release'
sources = {{
    'cvsrffi': ('cvsrffi/__init__.py', b''),
    'cvsrffi.stage2_d106_rcmr_g0': (
        'cvsrffi/stage2_d106_rcmr_g0.py',
        b'import importlib\\ndef execute():\\n    return importlib.import_module("cvsrffi.unlisted_during_execute")\\n',
    ),
}}
with child._verified_project_import_lifecycle(sources) as g0:
    try:
        g0.execute()
    except ImportError as error:
        assert 'unlisted project module: cvsrffi.unlisted_during_execute' in str(error)
    else:
        raise AssertionError('unlisted lazy project import was accepted')
print('LIFECYCLE_REJECTION_OK')
"""
    completed = subprocess.run(
        [sys.executable, "-I", "-c", probe],
        capture_output=True,
        text=True,
        check=False,
        timeout=30,
    )
    assert completed.returncode == 0, completed.stderr
    assert completed.stdout.strip() == "LIFECYCLE_REJECTION_OK"


def test_verified_finder_loads_fixed_d105_top_level_model_pair_from_sources():
    """The clean child must not require ambient ``PYTHONPATH`` for D105 imports."""

    probe = f"""
import importlib.util
import os
import sys
child_path = {str(CHILD_PATH)!r}
spec = importlib.util.spec_from_file_location('d106_child_model_pair_probe', child_path)
child = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = child
spec.loader.exec_module(child)
child.CODE_ROOT_PATH = '/verified-release'
sources = {{
    'cvsrffi': ('cvsrffi/__init__.py', b''),
    'cvsrffi.stage2_d106_rcmr_g0': (
        'cvsrffi/stage2_d106_rcmr_g0.py',
        b'from cvsrffi.dual_feature_forward import VALUE\\n',
    ),
    'cvsrffi.dual_feature_forward': (
        'cvsrffi/dual_feature_forward.py',
        b'from model_dual_cvsincnet import VALUE\\n',
    ),
    'model_dual_cvsincnet': (
        'model_dual_cvsincnet.py',
        b'from model import VALUE\\n',
    ),
    'model': ('model.py', b'VALUE = 17\\n'),
}}
with child._verified_project_import_lifecycle(sources) as g0:
    assert g0.VALUE == 17
    assert sys.modules['model'].__file__ == os.path.join('/verified-release', 'model.py')
    assert sys.modules['model_dual_cvsincnet'].__file__ == os.path.join('/verified-release', 'model_dual_cvsincnet.py')
print('MODEL_PAIR_OK')
"""
    completed = subprocess.run(
        [sys.executable, "-I", "-c", probe],
        capture_output=True,
        text=True,
        check=False,
        timeout=30,
    )
    assert completed.returncode == 0, completed.stderr
    assert completed.stdout.strip() == "MODEL_PAIR_OK"


def test_posix_publisher_uses_fd_relative_no_overwrite_publication_primitives():
    source = SCRIPT_PATH.read_text(encoding="utf-8")
    assert "os.O_NOFOLLOW" in source
    assert "os.O_DIRECTORY" in source
    assert "src_dir_fd=directory_fd" in source
    assert "dst_dir_fd=directory_fd" in source
    assert "os.fsync(directory_fd)" in source
    assert "marker_written_last" in source


def test_cli_requires_externally_anchored_release_manifest():
    options = {action.dest for action in runner.build_parser()._actions}
    assert {
        "archive", "archive_sha256", "receipt", "receipt_sha256",
        "release_manifest", "expected_release_manifest_sha256", "output_dir",
    }.issubset(options)
