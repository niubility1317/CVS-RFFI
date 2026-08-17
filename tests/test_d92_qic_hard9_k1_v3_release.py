from __future__ import annotations

import hashlib
import io
import json
import tarfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RELEASE = (
    ROOT
    / "automation_reports"
    / "CV-SincNet"
    / "d92_e0_full_d42_qic_hard9k1_20260817_v3"
)
ARCHIVE = (
    RELEASE
    / "runtime"
    / "d92_qic_hard9_k1_source_fa75cf8e_20260817_v3.tar.gz"
)


def test_v3_archive_closes_every_runtime_locked_file() -> None:
    with tarfile.open(ARCHIVE, mode="r:gz") as bundle:
        config = json.load(
            io.TextIOWrapper(
                bundle.extractfile(
                    "configs/stage2_d92_qic_hard9_k1_v3.json"
                ),
                encoding="utf-8",
            )
        )
        assert config["runtime"]["output_root"].endswith(
            "/d92_qic_hard9_k1_20260817_v3"
        )
        for relative_path, expected in config["runtime_source"]["files"].items():
            payload = bundle.extractfile(f"code/{relative_path}").read()
            assert hashlib.sha256(payload).hexdigest() == expected["sha256"]
