from __future__ import annotations

import ast
import json
import os
import re
import subprocess
import sys
import tarfile
import tempfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RUN_ID = "d92_e0_full_ccoc_hard9k1_20260817_v2"
RELEASE = (
    ROOT
    / "automation_reports"
    / "CV-SincNet"
    / RUN_ID
)
ARCHIVE = (
    RELEASE
    / "runtime"
    / "d92_ccoc_hard9_k1_source_fe9033be_20260817_v2.tar.gz"
)
CONFIG = ROOT / "configs" / "stage2_d92_ccoc_hard9_k1_v2.json"
V1_CONFIG = ROOT / "configs" / "stage2_d92_ccoc_hard9_k1_v1.json"
OUTPUT_ROOT = (
    "/home/szu2070436088/2510044040/CV-SincNet/runs/"
    "d92_ccoc_hard9_k1_20260817_v2"
)


def _launch_import_names(launch_text: str) -> tuple[str, ...]:
    match = re.search(r"names\s*=\s*(\([\s\S]*?\))\nfor name in names:", launch_text)
    assert match is not None, "launch import probe is missing"
    names = ast.literal_eval(match.group(1))
    assert isinstance(names, tuple) and names
    assert all(isinstance(name, str) and name for name in names)
    return names


def test_exact_extracted_release_can_import_every_launcher_probe_module() -> None:
    launch = (RELEASE / "launch.sh").read_text(encoding="utf-8")
    names = _launch_import_names(launch)
    with tempfile.TemporaryDirectory(prefix="d92_ccoc_hard9_v2_red_") as raw:
        extracted = Path(raw)
        with tarfile.open(ARCHIVE, "r:gz") as bundle:
            members = bundle.getmembers()
            for member in members:
                path = Path(member.name)
                assert not path.is_absolute() and ".." not in path.parts
                assert not member.issym() and not member.islnk()
            bundle.extractall(extracted)
        code_root = extracted / "code"
        environment = os.environ.copy()
        environment["PYTHONPATH"] = os.pathsep.join((str(code_root), str(extracted)))
        command = (
            "import importlib, json, sys; "
            "[importlib.import_module(name) for name in json.loads(sys.argv[1])]"
        )
        result = subprocess.run(
            [sys.executable, "-c", command, json.dumps(names)],
            check=False,
            capture_output=True,
            text=True,
            env=environment,
        )
        assert result.returncode == 0, result.stderr


def test_v2_config_and_release_paths_are_new_and_non_overwriting() -> None:
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    v1_config = json.loads(V1_CONFIG.read_text(encoding="utf-8"))
    assert config["runtime"]["output_root"] == OUTPUT_ROOT
    v1_config["runtime"]["output_root"] = OUTPUT_ROOT
    assert config == v1_config

    launch = (RELEASE / "launch.sh").read_text(encoding="utf-8")
    expected_paths = (
        RUN_ID,
        "d92_ccoc_hard9_k1_source_fe9033be_20260817_v2",
        "d92_ccoc_hard9_k1_20260817_v2",
        "stage2_d92_ccoc_hard9_k1_v2.json",
    )
    for value in expected_paths:
        assert value in launch
    assert "stage2_registration_balanced_covariance" not in launch
    assert "cvsrffi.stage2_d92_registration_balanced_covariance" in launch
