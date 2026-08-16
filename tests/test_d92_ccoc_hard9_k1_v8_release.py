from __future__ import annotations

import ast
import hashlib
import json
import os
import re
import subprocess
import sys
import tarfile
import tempfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RUN_ID = "d92_e0_full_ccoc_hard9k1_20260817_v8"
RELEASE = ROOT / "automation_reports" / "CV-SincNet" / RUN_ID
MIRROR = Path("E:/type10-7/automation_reports/CV-SincNet") / RUN_ID
ARCHIVE_NAME = "d92_ccoc_hard9_k1_source_4e267130_20260817_v8.tar.gz"
ARCHIVE = RELEASE / "runtime" / ARCHIVE_NAME
CONFIG = ROOT / "configs" / "stage2_d92_ccoc_hard9_k1_v8.json"
V7_CONFIG = ROOT / "configs" / "stage2_d92_ccoc_hard9_k1_v7.json"
SCIENTIFIC_COMMIT = "053ef7d006b05d4cb00c593e9b694669c0ecb005"
CORE_LOCK_PATH = "cvsrffi/stage2_d92_cross_class_offblock_consensus.py"
CORE_ARCHIVE_PATH = f"code/{CORE_LOCK_PATH}"
CORE_SHA256 = "6f87d4eb041ba8874182a46eb3f2a76dc3f2f075a6692ee217f19bcd2f8ff331"
QUERY_LOCK_PATH = "cvsrffi/stage2_d92_e0d_query_evaluation.py"
QUERY_ARCHIVE_PATH = f"code/{QUERY_LOCK_PATH}"
QUERY_BLOB = "03e163678bcb84ad39917f48c343347e75eb8ca5"
QUERY_SHA256 = "48e71bb8eaf9092f430e68e8cec6471a1e64d142a5282ff1b349faea336b55f1"
OUTPUT_ROOT = (
    "/home/szu2070436088/2510044040/CV-SincNet/runs/"
    "d92_ccoc_hard9_k1_20260817_v8"
)
SOURCE_MANIFEST = "code/D92_CCOC_HARD9_K1_SOURCE_MANIFEST.sha256"


def _launch_import_names(launch_text: str) -> tuple[str, ...]:
    match = re.search(r"names\s*=\s*(\([\s\S]*?\))\nfor name in names:", launch_text)
    assert match is not None, "launch import probe is missing"
    names = ast.literal_eval(match.group(1))
    assert isinstance(names, tuple) and names
    assert all(isinstance(name, str) and name for name in names)
    return names


def test_v8_surface_archive_manifest_and_external_mirror_are_exact() -> None:
    required = (
        RELEASE / "report.md",
        RELEASE / "launch.sh",
        RELEASE / "DELIVERY_MANIFEST.txt",
        ARCHIVE,
        CONFIG,
    )
    missing = [str(path) for path in required if not path.is_file()]
    assert not missing, f"missing v8 release artifacts: {missing}"
    for relative in (
        "report.md",
        "launch.sh",
        "DELIVERY_MANIFEST.txt",
        f"runtime/{ARCHIVE_NAME}",
    ):
        assert (MIRROR / relative).read_bytes() == (RELEASE / relative).read_bytes()

    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    v7_config = json.loads(V7_CONFIG.read_text(encoding="utf-8"))
    assert config["runtime"]["output_root"] == OUTPUT_ROOT
    v7_config["runtime"]["output_root"] = OUTPUT_ROOT
    v7_config["runtime_source"]["scientific_entry_commit"] = SCIENTIFIC_COMMIT
    v7_config["runtime_source"]["files"][QUERY_LOCK_PATH] = {
        "git_blob": QUERY_BLOB,
        "sha256": QUERY_SHA256,
    }
    assert config == v7_config

    with tarfile.open(ARCHIVE, "r:gz") as bundle:
        members = bundle.getmembers()
        names = [member.name for member in members]
        assert len(names) == 49
        assert SOURCE_MANIFEST in names
        assert all(not member.issym() and not member.islnk() for member in members)
        for name in names:
            path = Path(name)
            assert not path.is_absolute() and ".." not in path.parts
            lowered = name.lower()
            assert not any(
                token in lowered
                for token in (
                    "/data/",
                    "/checkpoint",
                    "/truth",
                    "/tests/",
                    "/docs/",
                )
            )
        manifest_lines = bundle.extractfile(SOURCE_MANIFEST).read().decode("utf-8")
        records = {}
        for line in manifest_lines.splitlines():
            digest, name = line.split(" *", 1)
            records[name] = digest
        assert len(records) == 48
        assert set(records) == set(names) - {SOURCE_MANIFEST}
        assert records[CORE_ARCHIVE_PATH] == CORE_SHA256
        assert records[QUERY_ARCHIVE_PATH] == QUERY_SHA256
        for name, digest in records.items():
            payload = bundle.extractfile(name).read()
            assert hashlib.sha256(payload).hexdigest() == digest, name


def test_exact_v8_archive_imports_helps_and_sha_only_prepare_pass() -> None:
    launch_text = (RELEASE / "launch.sh").read_text(encoding="utf-8")
    names = _launch_import_names(launch_text)
    with tempfile.TemporaryDirectory(prefix="d92_ccoc_hard9_v8_") as raw:
        extracted = Path(raw)
        with tarfile.open(ARCHIVE, "r:gz") as bundle:
            bundle.extractall(extracted)
        assert not (extracted / ".git").exists()
        code_root = extracted / "code"
        environment = os.environ.copy()
        environment["PYTHONPATH"] = os.pathsep.join((str(code_root), str(extracted)))

        import_probe = (
            "import importlib,json,sys;"
            "[importlib.import_module(n) for n in json.loads(sys.argv[1])]"
        )
        imported = subprocess.run(
            [sys.executable, "-c", import_probe, json.dumps(names)],
            check=False,
            capture_output=True,
            text=True,
            env=environment,
        )
        assert imported.returncode == 0, imported.stderr

        for script in (
            code_root / "scripts" / "run_d92_ccoc_hard9_k1.py",
            code_root / "scripts" / "analyze_d92_ccoc_hard9_k1.py",
        ):
            helped = subprocess.run(
                [sys.executable, str(script), "--help"],
                check=False,
                capture_output=True,
                text=True,
                env=environment,
            )
            assert helped.returncode == 0, helped.stderr
            assert "usage:" in helped.stdout.lower()

        prepare_probe = """
import argparse
import json
import sys
from pathlib import Path
root = Path(sys.argv[1]).resolve()
sys.path.insert(0, str(root / "code"))
from scripts import run_d92_ccoc_hard9_k1 as runner
config = root / "configs" / "stage2_d92_ccoc_hard9_k1_v8.json"
runner.build_hard9_k1_manifest = lambda _config, require_package_files: {
    "method_lock": str(config),
    "jobs": [],
    "output_root": str(root / "prepare_output"),
}
print(json.dumps(runner.prepare(argparse.Namespace(config=str(config))), sort_keys=True))
"""
        prepared = subprocess.run(
            [sys.executable, "-c", prepare_probe, str(extracted)],
            check=False,
            capture_output=True,
            text=True,
            env=environment,
        )
        assert prepared.returncode == 0, prepared.stderr
        receipt = json.loads(prepared.stdout)
        assert receipt["status"] == "CCOC_HARD9_K1_MATRIX_PREPARED"
        assert receipt["runtime_source_verification_mode"] == "sha256_only"
        assert (
            receipt["e0_resource_source_mode"]
            == "embedded_preregistered_projection"
        )
        assert receipt["e0_resource_fit_audit_declared_sha256"] == {}


def test_v8_launch_is_hash_closed_prepare_smoke_then_eight_shards() -> None:
    text = (RELEASE / "launch.sh").read_text(encoding="utf-8")
    assert RUN_ID in text
    assert ARCHIVE_NAME in text
    assert "stage2_d92_ccoc_hard9_k1_v8.json" in text
    assert "runtime_source_verification_mode" in text
    assert '"sha256_only"' in text
    assert "e0_resource_source_mode" in text
    assert '"embedded_preregistered_projection"' in text
    assert "e0_resource_fit_audit_declared_sha256" in text
    assert "sha256sum -c \"$source_manifest\"" in text
    assert "--truth" not in text.lower() and "truth_sidecar" not in text
    assert "analyze_d92_ccoc_hard9_k1.py" not in text
    assert "pkill" not in text and "killall" not in text
    assert text.index(" prepare ") < text.index(" smoke ") < text.index(" run-shard ")
    assert "for shard in 0 1 2 3 4 5 6 7; do" in text
    assert text.count(" run-shard ") == 1
    assert "fresh_run_retry=false" in text
    archive_sha = re.search(r"^archive_sha256=([0-9a-f]{64})$", text, re.M)
    archive_size = re.search(r"^archive_size_bytes=([0-9]+)$", text, re.M)
    assert archive_sha and archive_sha.group(1) == hashlib.sha256(ARCHIVE.read_bytes()).hexdigest()
    assert archive_size and int(archive_size.group(1)) == ARCHIVE.stat().st_size


def test_v8_report_manifest_sync_and_trace_are_preregistered_only() -> None:
    expected = (
        "LOCAL_VERIFIED_READY_FOR_N607_HANDOFF / "
        "NO_HARD9_RUNTIME_RESULT / NO_PERFORMANCE_RESULT"
    )
    paths = (
        RELEASE / "report.md",
        RELEASE / "DELIVERY_MANIFEST.txt",
        ROOT / "code" / "SYNC_MANIFEST.txt",
        ROOT / "analysis" / "d92_ccoc_traceability_20260813.md",
    )
    for path in paths:
        text = path.read_text(encoding="utf-8")
        assert RUN_ID in text
        assert "CCOC-16" in text
        assert "NO_PERFORMANCE_RESULT" in text
    report = paths[0].read_text(encoding="utf-8")
    manifest = paths[1].read_text(encoding="utf-8")
    assert expected in report and expected in manifest
    assert "candidate_peak_hard_max_bytes=1048576" in report
    assert "candidate_peak_target_max_bytes=524288" in report
    assert "runtime_source_verification_mode=sha256_only" in manifest
    assert "e0_resource_source_mode=embedded_preregistered_projection" in manifest
    assert not re.search(r"(?:verdict|status)\s*[:=]\s*PASS\b", report, re.I)
