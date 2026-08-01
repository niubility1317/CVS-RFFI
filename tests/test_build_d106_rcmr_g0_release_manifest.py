from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import textwrap
from types import ModuleType

import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]
BUILDER_PATH = REPO_ROOT / "code" / "scripts" / "build_d106_rcmr_g0_release_manifest.py"
RUNNER_PATH = REPO_ROOT / "code" / "scripts" / "run_d106_rcmr_g0_production.py"
DIRECT_MODULES = (
    "cvsrffi.stage2_d106_rcmr_g0",
    "cvsrffi.stage2_d106_phase1_tap",
    "cvsrffi.stage2_d106_train_only_predecessor_lock",
    "cvsrffi.stage2_d106_rcmr_2v_qknn",
    "cvsrffi.stage2_zid_student_t_qknn",
)
CLASSES = ("tx-06", "tx-01", "tx-03", "tx-05", "tx-02", "tx-04")


def _load_module(path: Path, name: str) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    try:
        spec.loader.exec_module(module)
    finally:
        sys.modules.pop(name, None)
    return module


@pytest.fixture(scope="module")
def builder() -> ModuleType:
    return _load_module(BUILDER_PATH, "d106_release_manifest_builder_test")


@pytest.fixture(scope="module")
def runner() -> ModuleType:
    return _load_module(RUNNER_PATH, "d106_release_manifest_runner_schema_test")


def _write(root: Path, relative: str, source: str) -> None:
    path = root / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(textwrap.dedent(source).lstrip(), encoding="utf-8", newline="\n")


def _git(cwd: Path, *args: str) -> str:
    completed = subprocess.run(
        ["git", *args], cwd=cwd, check=True, capture_output=True, text=True, encoding="utf-8"
    )
    return completed.stdout.strip()


def _fixture_public_map() -> dict[str, str]:
    names = (
        "g0_executor_module",
        "tap_loader_module",
        "tap_loader_callable",
        "predecessor_lock_module",
        "predecessor_lock_bundle_callable",
        "predecessor_lock_reconstruct_callable",
        "rcmr_module",
        "qknn_module",
    )
    return {
        name: hashlib.sha256(("fixture:" + name).encode("utf-8")).hexdigest()
        for name in names
    }


def _make_clean_release_tree(tmp_path: Path) -> tuple[Path, Path, str]:
    repository = tmp_path / "release_tree"
    code_root = repository / "code"
    public_map = repr(_fixture_public_map())
    _write(code_root, "cvsrffi/__init__.py", "__all__ = []\n")
    _write(
        code_root,
        "cvsrffi/stage2_d106_rcmr_g0.py",
        f"""
        from .stage2_d106_phase1_tap import TAP_VALUE

        __all__ = ["get_d106_rcmr_g0_release_expected_code_sha256"]

        def get_d106_rcmr_g0_release_expected_code_sha256():
            _ = TAP_VALUE
            return dict({public_map})
        """,
    )
    _write(
        code_root,
        "cvsrffi/stage2_d106_phase1_tap.py",
        """
        from .transitive_helper import TRANSITIVE_VALUE

        TAP_VALUE = TRANSITIVE_VALUE
        """,
    )
    _write(code_root, "cvsrffi/transitive_helper.py", "TRANSITIVE_VALUE = 7\n")
    _write(code_root, "cvsrffi/stage2_d106_train_only_predecessor_lock.py", "LOCK = 1\n")
    _write(
        code_root,
        "cvsrffi/stage2_d106_rcmr_2v_qknn.py",
        "import cvsrffi.stage2_zid_student_t_qknn as student\nRCMR = student.STUDENT\n",
    )
    _write(code_root, "cvsrffi/stage2_zid_student_t_qknn.py", "STUDENT = 2\n")
    _write(code_root, "cvsrffi/unrelated.py", "UNRELATED = True\n")
    _write(
        code_root,
        "scripts/d106_rcmr_g0_clean_child.py",
        """
        _REQUIRED_DIRECT_MODULES = {
            "cvsrffi.stage2_d106_rcmr_g0",
            "cvsrffi.stage2_d106_phase1_tap",
            "cvsrffi.stage2_d106_train_only_predecessor_lock",
            "cvsrffi.stage2_d106_rcmr_2v_qknn",
            "cvsrffi.stage2_zid_student_t_qknn",
        }
        """,
    )
    _write(code_root, "scripts/run_d106_rcmr_g0_production.py", "# fixture runner\n")
    _git(repository, "init")
    _git(repository, "config", "user.email", "fixture@example.invalid")
    _git(repository, "config", "user.name", "Fixture")
    _git(repository, "add", ".")
    _git(repository, "commit", "-m", "fixture release tree")
    return repository, code_root, _git(repository, "rev-parse", "HEAD")


def _build(
    builder: ModuleType, code_root: Path, commit: str, output: Path, **kwargs: object
):
    return builder.build_release_manifest(
        code_root=code_root,
        expected_release_commit=commit,
        registered_classes=CLASSES,
        output_path=output,
        **kwargs,
    )


def test_ast_fixed_point_includes_transitive_modules_and_excludes_unrelated(
    tmp_path: Path, builder: ModuleType
) -> None:
    _repository, code_root, _commit = _make_clean_release_tree(tmp_path)

    closure = builder.discover_production_module_closure(code_root)

    assert set(DIRECT_MODULES).issubset(closure)
    assert "cvsrffi.transitive_helper" in closure
    assert "cvsrffi.unrelated" not in closure
    assert closure == tuple(sorted(closure))


def test_canonical_manifest_matches_current_runner_schema_and_never_overwrites(
    tmp_path: Path, builder: ModuleType, runner: ModuleType, capsys: pytest.CaptureFixture[str]
) -> None:
    repository, code_root, commit = _make_clean_release_tree(tmp_path)
    output_parent = tmp_path / "published"
    output_parent.mkdir()
    output = output_parent / "release_manifest.json"

    exit_code = builder.main(
        [
            "--code-root",
            str(code_root),
            "--expected-release-commit",
            commit,
            *sum((["--registered-class", value] for value in CLASSES), []),
            "--output",
            str(output),
        ]
    )
    captured = capsys.readouterr()
    raw = output.read_bytes()
    document = json.loads(raw.decode("utf-8"))

    assert exit_code == 0
    assert raw == builder._canonical_bytes(document)
    manifest_sha = hashlib.sha256(raw).hexdigest()
    assert f"manifest_sha256={manifest_sha}" in captured.out
    assert document["schema"] == runner.RELEASE_MANIFEST_SCHEMA
    assert document["release_commit"] == commit
    assert document["registered_classes"] == sorted(CLASSES, key=lambda value: value.encode("utf-8"))
    assert set(DIRECT_MODULES).issubset(document["production_module_closure"])
    assert "cvsrffi.transitive_helper" in document["production_module_closure"]
    assert "cvsrffi.unrelated" not in document["production_module_closure"]
    assert {
        "scripts/run_d106_rcmr_g0_production.py",
        "scripts/d106_rcmr_g0_clean_child.py",
        "cvsrffi/__init__.py",
    }.issubset(document["code_files_sha256"])
    assert runner.validate_release_manifest_test_only(
        raw, expected_manifest_sha256=manifest_sha
    )["source_file_count"] == len(document["code_files_sha256"])
    assert _git(repository, "status", "--porcelain=v1") == ""

    with pytest.raises(builder.ReleaseManifestBuildError, match="refusing to overwrite"):
        _build(builder, code_root, commit, output)


def test_missing_transitive_module_fails_closed(tmp_path: Path, builder: ModuleType) -> None:
    _repository, code_root, _commit = _make_clean_release_tree(tmp_path)
    _write(code_root, "cvsrffi/transitive_helper.py", "from .missing_leaf import VALUE\n")

    with pytest.raises(builder.ReleaseManifestBuildError, match="missing imported cvsrffi module"):
        builder.discover_production_module_closure(code_root)


def test_relative_import_escape_and_relative_output_are_rejected(
    tmp_path: Path, builder: ModuleType
) -> None:
    _repository, code_root, _commit = _make_clean_release_tree(tmp_path)
    _write(code_root, "cvsrffi/transitive_helper.py", "from ..outside import VALUE\n")

    with pytest.raises(builder.ReleaseManifestBuildError, match="escapes cvsrffi"):
        builder.discover_production_module_closure(code_root)

    _repository, code_root, commit = _make_clean_release_tree(tmp_path / "second")
    with pytest.raises(builder.ReleaseManifestBuildError, match="explicit and absolute"):
        _build(builder, code_root, commit, Path("relative_manifest.json"))


def test_dirty_checkout_and_head_mismatch_are_rejected(
    tmp_path: Path, builder: ModuleType
) -> None:
    _repository, code_root, commit = _make_clean_release_tree(tmp_path)
    output_parent = tmp_path / "published"
    output_parent.mkdir()

    with pytest.raises(builder.ReleaseManifestBuildError, match="Git HEAD mismatch"):
        _build(builder, code_root, "0" * 40, output_parent / "wrong_head.json")

    _write(code_root, "cvsrffi/unrelated.py", "UNRELATED = False\n")
    with pytest.raises(builder.ReleaseManifestBuildError, match="not clean"):
        _build(builder, code_root, commit, output_parent / "dirty.json")


def test_g0_expected_code_requires_public_fresh_interface(
    tmp_path: Path, builder: ModuleType
) -> None:
    _repository, code_root, _commit = _make_clean_release_tree(tmp_path)
    _write(
        code_root,
        "cvsrffi/stage2_d106_rcmr_g0.py",
        """
        def get_d106_rcmr_g0_release_expected_code_sha256():
            return {"g0_executor_module": "0" * 64}
        """,
    )

    with pytest.raises(builder.ReleaseManifestBuildError, match="public packaging interface unavailable"):
        builder.read_g0_public_expected_code_sha256(code_root)

