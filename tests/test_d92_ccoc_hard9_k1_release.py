from __future__ import annotations

import os
import re
import subprocess
import tarfile
import sys
import tempfile
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
RUN_ID = "d92_e0_full_ccoc_hard9k1_20260816_v1"
RELEASE = ROOT / "automation_reports" / "CV-SincNet" / RUN_ID
MIRROR = Path("E:/type10-7/automation_reports/CV-SincNet") / RUN_ID
ARCHIVE_NAME = "d92_ccoc_hard9_k1_source_07e69f1e_20260816_v1.tar.gz"
ARCHIVE = RELEASE / "runtime" / ARCHIVE_NAME
CONFIG = ROOT / "configs" / "stage2_d92_ccoc_hard9_k1_v1.json"
SOURCE_MANIFEST = "code/D92_CCOC_HARD9_K1_SOURCE_MANIFEST.sha256"


def _release_files() -> tuple[Path, ...]:
    return (
        RELEASE / "report.md",
        RELEASE / "launch.sh",
        RELEASE / "DELIVERY_MANIFEST.txt",
        ARCHIVE,
        CONFIG,
    )


def test_release_surface_exists_and_external_mirror_is_byte_identical() -> None:
    missing = [str(path) for path in _release_files() if not path.is_file()]
    assert not missing, f"missing release artifacts: {missing}"
    for relative in ("report.md", "launch.sh", "DELIVERY_MANIFEST.txt"):
        source = RELEASE / relative
        mirrored = MIRROR / relative
        assert mirrored.is_file(), mirrored
        assert mirrored.read_bytes() == source.read_bytes(), relative
    mirrored_archive = MIRROR / "runtime" / ARCHIVE_NAME
    assert mirrored_archive.is_file(), mirrored_archive
    assert mirrored_archive.read_bytes() == ARCHIVE.read_bytes()


def test_archive_is_safe_and_contains_the_runtime_surface() -> None:
    assert ARCHIVE.is_file(), ARCHIVE
    with tarfile.open(ARCHIVE, "r:gz") as bundle:
        members = bundle.getmembers()
        names = [member.name for member in members]
        assert SOURCE_MANIFEST in names
        assert all(not member.issym() and not member.islnk() for member in members)
        for name in names:
            path = Path(name)
            assert not path.is_absolute() and ".." not in path.parts
            lowered = name.lower()
            assert not any(
                token in lowered
                for token in ("/data/", "/checkpoint", "/truth", "/tests/", "/docs/")
            )
            assert not lowered.startswith(("data/", "checkpoint", "truth/", "tests/", "docs/"))
        required = {
            "code/cvsrffi/__init__.py",
            "code/cvsrffi/stage2_d92_ccoc_hard9_k1.py",
            "code/cvsrffi/stage2_d92_ccoc_hard9_k1_analysis.py",
            "code/scripts/run_d92_ccoc_hard9_k1.py",
            "code/scripts/analyze_d92_ccoc_hard9_k1.py",
            "configs/stage2_d92_ccoc_hard9_k1_v1.json",
        }
        assert required <= set(names)


def test_extracted_runtime_clis_are_importable_without_running_analyzer() -> None:
    """Exercise both CLIs from the exact archive extraction, not the worktree."""

    with tempfile.TemporaryDirectory(prefix="d92_ccoc_hard9_release_") as raw_root:
        extracted = Path(raw_root)
        with tarfile.open(ARCHIVE, "r:gz") as bundle:
            members = bundle.getmembers()
            for member in members:
                path = Path(member.name)
                assert not path.is_absolute() and ".." not in path.parts
                assert not member.issym() and not member.islnk()
            bundle.extractall(extracted)

        code_root = extracted / "code"
        environment = os.environ.copy()
        environment["PYTHONPATH"] = os.pathsep.join(
            (str(code_root), str(extracted))
        )
        commands = (
            ("runner", code_root / "scripts" / "run_d92_ccoc_hard9_k1.py"),
            ("analyzer", code_root / "scripts" / "analyze_d92_ccoc_hard9_k1.py"),
        )
        for label, script in commands:
            result = subprocess.run(
                [sys.executable, str(script), "--help"],
                check=False,
                capture_output=True,
                text=True,
                env=environment,
            )
            assert result.returncode == 0, (
                f"{label} --help failed from extracted archive\n"
                f"stdout={result.stdout}\nstderr={result.stderr}"
            )
            assert "usage:" in result.stdout.lower()


def test_launch_is_prepare_smoke_then_exactly_eight_shards_without_analyzer() -> None:
    launch = RELEASE / "launch.sh"
    assert launch.is_file(), launch
    text = launch.read_text(encoding="utf-8")
    assert "--truth" not in text.lower()
    assert "truth_sidecar" not in text
    assert '"$runner" score' not in text
    assert '"$python" -u "$code_root/scripts/analyze_d92_ccoc_hard9_k1.py"' not in text
    assert "python -m cvsrffi.stage2_d92_ccoc_hard9_k1_analysis" not in text
    prepare = text.index(" prepare ")
    smoke = text.index(" smoke ")
    shards = [match.start() for match in re.finditer(r" run-shard ", text)]
    assert prepare < smoke < min(shards)
    assert "for shard in 0 1 2 3 4 5 6 7; do" in text
    assert len(shards) == 1
    assert text.count('--shard-index "$shard"') == 1
    assert text.count('--shard-count 8') == 1
    assert text.count(" coordinator-stop ") <= 1
    assert "fresh_run_retry=false" in text
    assert "pkill" not in text and "killall" not in text


def test_report_and_delivery_manifest_are_preregistered_without_results() -> None:
    report = RELEASE / "report.md"
    manifest = RELEASE / "DELIVERY_MANIFEST.txt"
    assert report.is_file(), report
    assert manifest.is_file(), manifest
    report_text = report.read_text(encoding="utf-8")
    manifest_text = manifest.read_text(encoding="utf-8")
    expected = (
        "LOCAL_VERIFIED_READY_FOR_N607_HANDOFF / "
        "NO_HARD9_RUNTIME_RESULT / NO_PERFORMANCE_RESULT"
    )
    assert expected in report_text
    assert expected in manifest_text
    assert RUN_ID in report_text and RUN_ID in manifest_text
    assert "CCOC-16" in report_text and "CCOC-16" in manifest_text
    assert "candidate_peak_hard_max_bytes=1048576" in report_text
    assert "candidate_peak_target_max_bytes=524288" in report_text
    assert not re.search(r"(?:verdict|status)\s*[:=]\s*PASS\b", report_text, re.I)


def test_sync_and_traceability_record_release_ready_no_performance() -> None:
    sync = ROOT / "code" / "SYNC_MANIFEST.txt"
    trace = ROOT / "analysis" / "d92_ccoc_traceability_20260813.md"
    assert sync.is_file(), sync
    assert trace.is_file(), trace
    for path in (sync, trace):
        text = path.read_text(encoding="utf-8")
        assert RUN_ID in text
        assert "CCOC-16" in text
        assert "release ready" in text.lower()
        assert "NO_PERFORMANCE_RESULT" in text
