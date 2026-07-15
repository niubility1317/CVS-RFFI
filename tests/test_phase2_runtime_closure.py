from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]
CODE_ROOT = REPO_ROOT / "code"

from cvsrffi.phase2_runtime_closure import (  # noqa: E402
    Phase2RuntimeClosureError,
    RUNTIME_ENTRYPOINT,
    RUNTIME_MEMBER_ALLOWLIST,
    RUNTIME_MOUNT_PATH,
    build_phase2_runtime_closure,
    verify_phase2_runtime_closure,
)


def _copy_reviewed_sources(destination: Path) -> Path:
    code_root = destination / "code"
    for relative in RUNTIME_MEMBER_ALLOWLIST:
        source = CODE_ROOT.joinpath(*relative.split("/"))
        target = code_root.joinpath(*relative.split("/"))
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
    return code_root


def test_builds_only_exact_runtime_members_with_bwrap_layout(tmp_path: Path) -> None:
    output = tmp_path / "closure"
    result = build_phase2_runtime_closure(CODE_ROOT, output)
    assert result["verified"] is True
    assert result["member_count"] == 9
    assert result["runtime_mount_path"] == RUNTIME_MOUNT_PATH == "/runtime/code"
    assert result["entrypoint"] == RUNTIME_ENTRYPOINT
    actual = {
        path.relative_to(output / "runtime").as_posix()
        for path in (output / "runtime").rglob("*")
        if path.is_file()
    }
    assert actual == set(RUNTIME_MEMBER_ALLOWLIST)
    assert (output / "runtime/scripts/run_cvs_stage2_predictor.py").is_file()


def test_build_refuses_any_existing_output_root(tmp_path: Path) -> None:
    output = tmp_path / "closure"
    build_phase2_runtime_closure(CODE_ROOT, output)
    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        build_phase2_runtime_closure(CODE_ROOT, output)


def test_cli_builds_verified_runtime_closure(tmp_path: Path) -> None:
    output = tmp_path / "cli_closure"
    completed = subprocess.run(
        [
            sys.executable,
            "code/scripts/build_cvs_stage2_runtime_closure.py",
            "--source-code-root",
            str(CODE_ROOT),
            "--output-root",
            str(output),
        ],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    payload = json.loads(completed.stdout)
    assert payload["verified"] is True
    assert payload["root_sha256"] == verify_phase2_runtime_closure(output)["root_sha256"]


def test_verifier_rejects_extra_file(tmp_path: Path) -> None:
    output = tmp_path / "closure"
    build_phase2_runtime_closure(CODE_ROOT, output)
    extra = output / "runtime/cvsrffi/train_legacy.py"
    extra.write_text("raise RuntimeError('must not be importable')\n", encoding="utf-8")
    with pytest.raises(Phase2RuntimeClosureError, match="file allowlist mismatch"):
        verify_phase2_runtime_closure(output)


def test_verifier_rejects_digest_tamper(tmp_path: Path) -> None:
    output = tmp_path / "closure"
    build_phase2_runtime_closure(CODE_ROOT, output)
    target = output / "runtime/cvsrffi/stage2_predictor_entry.py"
    os.chmod(target, 0o644)
    with target.open("ab") as handle:
        handle.write(b"\n# tamper\n")
    os.chmod(target, 0o444)
    with pytest.raises(Phase2RuntimeClosureError, match="digest drift"):
        verify_phase2_runtime_closure(output)


@pytest.mark.parametrize(
    "injected_import",
    [
        "from paper_reproduction.common.wisig_runtime import load_wisig_compact_pkl\n",
        "from cvsrffi.leo_weak_cache import load_verified_leo_weak_cache_set\n",
        "import SSDG.train_ssdg\n",
        "__import__('legacy.dataset.loader')\n",
        "exec('import SSDG.train_ssdg')\n",
        "import ctypes\n",
    ],
)
def test_builder_rejects_training_dataset_legacy_or_unlisted_internal_import(
    tmp_path: Path,
    injected_import: str,
) -> None:
    source_root = _copy_reviewed_sources(tmp_path / "source")
    target = source_root / "cvsrffi/stage2_predictor_entry.py"
    with target.open("a", encoding="utf-8") as handle:
        handle.write(injected_import)
    output = tmp_path / "closure"
    with pytest.raises(Phase2RuntimeClosureError):
        build_phase2_runtime_closure(source_root, output)
    assert not output.exists()


def test_builder_rejects_source_member_symlink(tmp_path: Path) -> None:
    source_root = _copy_reviewed_sources(tmp_path / "source")
    target = source_root / "cvsrffi/stage2_predictor_entry.py"
    real_target = source_root / "outside_entry.py"
    shutil.copyfile(target, real_target)
    target.unlink()
    try:
        target.symlink_to(real_target)
    except OSError as exc:
        pytest.skip(f"symlink creation is unavailable: {exc}")
    with pytest.raises(Phase2RuntimeClosureError, match="symlink"):
        build_phase2_runtime_closure(source_root, tmp_path / "closure")


def test_verifier_rejects_runtime_member_symlink(tmp_path: Path) -> None:
    output = tmp_path / "closure"
    build_phase2_runtime_closure(CODE_ROOT, output)
    target = output / "runtime/cvsrffi/stage2_predictor_entry.py"
    os.chmod(target, 0o644)
    replacement = output / "replacement.py"
    replacement.write_bytes(target.read_bytes())
    target.unlink()
    try:
        target.symlink_to(replacement)
    except OSError as exc:
        pytest.skip(f"symlink creation is unavailable: {exc}")
    with pytest.raises(Phase2RuntimeClosureError, match="symlink"):
        verify_phase2_runtime_closure(output)
