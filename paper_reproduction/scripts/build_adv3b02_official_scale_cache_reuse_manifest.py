#!/usr/bin/env python3
"""Generate v7 read-only reuse and same-cache new20 integrity commands."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path, PurePosixPath

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from paper_reproduction.scripts.build_adv3b02_official_scale_cache_specs import (
    NEW25,
    RECEIVERS,
    SEEDS,
    _safe,
)


def _write_new(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="\n") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())


def build(args: argparse.Namespace) -> dict:
    cache_root = PurePosixPath(str(args.reused_cache_root))
    receipt_root = PurePosixPath(str(args.remote_integrity_root))
    entries = []
    commands = []
    for receiver in RECEIVERS:
        for seed in SEEDS:
            cache_set = (
                cache_root
                / f"rx_{_safe(receiver)}"
                / f"seed_{seed}"
                / "cache_set.json"
            )
            receipt = (
                receipt_root / f"rx_{_safe(receiver)}" / f"seed_{seed}.json"
            )
            entries.append(
                {
                    "receiver": receiver,
                    "seed": seed,
                    "reused_cache_set": str(cache_set),
                    "integrity_receipt": str(receipt),
                }
            )
            commands.append(
                [
                    "python",
                    (
                        "paper_reproduction/scripts/"
                        "verify_adv3b02_official_scale_cache_parity.py"
                    ),
                    "--reference-cache-set",
                    str(cache_set),
                    "--expanded-cache-set",
                    str(cache_set),
                    "--reference-scope",
                    "external_comparison_registered",
                    "--expanded-scope",
                    "external_comparison_registered",
                    "--preserved-class-labels",
                    ",".join(NEW25[:20]),
                    "--mode",
                    "same_cache_new20_integrity",
                    "--output",
                    str(receipt),
                ]
            )
    manifest = {
        "schema": "cvs.adv3b02.official_scale_cache_reuse_manifest.v1",
        "experiment_id": str(args.experiment_id),
        "status": "REUSE_VERIFICATION_NOT_EXECUTED",
        "reuse_policy": "READ_ONLY_NO_CACHE_REBUILD",
        "source_experiment_id": str(args.source_experiment_id),
        "reused_cache_root": str(cache_root),
        "receivers": list(RECEIVERS),
        "seeds": list(SEEDS),
        "preserved_new20_class_labels": list(NEW25[:20]),
        "entries": entries,
        "integrity_commands": commands,
        "claim_boundary": (
            "same-cache integrity only; no historical-cache parity claim"
        ),
    }
    _write_new(Path(args.output), manifest)
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment-id", required=True)
    parser.add_argument("--source-experiment-id", required=True)
    parser.add_argument("--reused-cache-root", required=True)
    parser.add_argument("--remote-integrity-root", required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


if __name__ == "__main__":
    print(json.dumps(build(parse_args()), ensure_ascii=False, sort_keys=True))
