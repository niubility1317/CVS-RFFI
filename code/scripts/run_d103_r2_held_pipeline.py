#!/usr/bin/env python3
"""Run the complete frozen D103-R2 Phase1-held evidence pipeline."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
from typing import Sequence


ROOT = Path(__file__).resolve().parents[1]


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _d102_construction_code_sha() -> str:
    return hashlib.sha256(
        (
            _sha(ROOT / "cvsrffi" / "phase1_rb_metabias4_bundle.py")
            + "\0"
            + _sha(ROOT / "scripts" / "build_d103_r2_d102_comparators.py")
        ).encode("ascii")
    ).hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-split-root", type=Path, required=True)
    parser.add_argument("--checkpoint-sha256", required=True)
    parser.add_argument("--runtime-sha256", required=True)
    parser.add_argument("--method-lock-sha256", required=True)
    parser.add_argument("--python", type=Path, default=Path(sys.executable))
    parser.add_argument("--gpus", default="0,1,2,3,4,5,6,7")
    parser.add_argument("--workers-per-gpu", type=int, default=2)
    parser.add_argument("--run-root", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    source = args.source_split_root.resolve(strict=True)
    run_root = args.run_root.resolve()
    if run_root.exists() or run_root.is_symlink():
        raise FileExistsError(f"immutable held run root exists: {run_root}")
    run_root.mkdir(parents=True, exist_ok=False)
    logs = run_root / "logs"
    logs.mkdir()
    python = str(args.python.resolve())
    scripts = ROOT / "scripts"
    common = {
        "labeled_archive": source / "L_s" / "features.npz",
        "labeled_manifest": source / "L_s" / "manifest.json",
        "unlabeled_archive": source / "U_s" / "features.npz",
        "unlabeled_manifest": source / "U_s" / "manifest.json",
        "source_val_seal": source / "source_val.seal.json",
        "source_val_manifest": source / "source_val.manifest.json",
        "source_val_archive": source / "scorer_only" / "source_val" / "features.npz",
        "source_val_scorer_manifest": source
        / "scorer_only"
        / "source_val"
        / "manifest.json",
    }
    commands: list[tuple[str, Sequence[str]]] = [
        (
            "fit_matrix",
            (
                python,
                str(scripts / "run_d103_r2_fit_matrix.py"),
                "--labeled-archive",
                str(common["labeled_archive"]),
                "--labeled-manifest",
                str(common["labeled_manifest"]),
                "--unlabeled-archive",
                str(common["unlabeled_archive"]),
                "--unlabeled-manifest",
                str(common["unlabeled_manifest"]),
                "--source-val-seal",
                str(common["source_val_seal"]),
                "--source-val-manifest",
                str(common["source_val_manifest"]),
                "--python",
                python,
                "--gpus",
                args.gpus,
                "--workers-per-gpu",
                str(args.workers_per_gpu),
                "--output-root",
                str(run_root / "matrix"),
            ),
        ),
        (
            "d102_comparators_before_source_val_open",
            (
                python,
                str(scripts / "build_d103_r2_d102_comparators.py"),
                "--labeled-archive",
                str(common["labeled_archive"]),
                "--labeled-manifest",
                str(common["labeled_manifest"]),
                "--checkpoint-sha256",
                args.checkpoint_sha256,
                "--runtime-sha256",
                args.runtime_sha256,
                "--code-sha256",
                _d102_construction_code_sha(),
                "--output-dir",
                str(run_root / "d102_comparators"),
            ),
        ),
        (
            "prepare_held_packages_and_tx_probe",
            (
                python,
                str(scripts / "prepare_d103_r2_held_packages.py"),
                "--source-val-archive",
                str(common["source_val_archive"]),
                "--source-val-manifest",
                str(common["source_val_scorer_manifest"]),
                "--fits-root",
                str(run_root / "matrix" / "fits"),
                "--checkpoint-sha256",
                args.checkpoint_sha256,
                "--runtime-sha256",
                args.runtime_sha256,
                "--method-lock-sha256",
                args.method_lock_sha256,
                "--output-dir",
                str(run_root / "held_packages"),
            ),
        ),
        (
            "truth_free_predictor",
            (
                python,
                str(scripts / "run_d103_r2_held_predictor.py"),
                "--package-root",
                str(run_root / "held_packages"),
                "--fits-root",
                str(run_root / "matrix" / "fits"),
                "--d102-root",
                str(run_root / "d102_comparators"),
                "--checkpoint-sha256",
                args.checkpoint_sha256,
                "--runtime-sha256",
                args.runtime_sha256,
                "--method-lock-sha256",
                args.method_lock_sha256,
                "--output-dir",
                str(run_root / "predictions"),
            ),
        ),
        (
            "independent_truth_side_scorer",
            (
                python,
                str(scripts / "score_d103_r2_held_predictions.py"),
                "--prediction-root",
                str(run_root / "predictions"),
                "--truth-json",
                str(run_root / "held_packages" / "scorer_only" / "truth.json"),
                "--output-json",
                str(run_root / "scores" / "held_scores.json"),
                "--truth-open-event-json",
                str(run_root / "scores" / "truth_first_open.json"),
            ),
        ),
        (
            "resource_receipt",
            (
                python,
                str(scripts / "finalize_d103_r2_runner_resources.py"),
                "--matrix-status",
                str(run_root / "matrix" / "matrix_status.json"),
                "--run-root",
                str(run_root),
                "--output-json",
                str(run_root / "analysis" / "runner_resources.json"),
            ),
        ),
        (
            "assemble_receipts",
            (
                python,
                str(scripts / "assemble_d103_r2_held_receipts.py"),
                "--scores-json",
                str(run_root / "scores" / "held_scores.json"),
                "--d102-provenance-json",
                str(run_root / "d102_comparators" / "d102_provenance.json"),
                "--tx-probe-json",
                str(run_root / "held_packages" / "tx_probe_receipt.json"),
                "--runner-resource-json",
                str(run_root / "analysis" / "runner_resources.json"),
                "--matrix-status-json",
                str(run_root / "matrix" / "matrix_status.json"),
                "--source-split-manifest",
                str(source / "source_split_manifest.json"),
                "--output-json",
                str(run_root / "analysis" / "held_receipts.json"),
            ),
        ),
        (
            "held_gate",
            (
                python,
                str(scripts / "run_d103_r1_held_gate.py"),
                "--receipts-json",
                str(run_root / "analysis" / "held_receipts.json"),
                "--output-json",
                str(run_root / "analysis" / "held_gate.json"),
                "--output-sha256",
                str(run_root / "analysis" / "held_gate.json.sha256"),
            ),
        ),
    ]
    completed = []
    for name, command in commands:
        log_path = logs / f"{len(completed):02d}_{name}.log"
        with log_path.open("xb") as log:
            result = subprocess.run(
                list(command),
                cwd=str(ROOT.parent),
                stdout=log,
                stderr=subprocess.STDOUT,
                check=False,
            )
        completed.append(
            {"stage": name, "returncode": result.returncode, "log": log_path.name}
        )
        if result.returncode != 0:
            status = {
                "status": "PIPELINE_TECHNICAL_FAILURE_NO_PERFORMANCE_RESULT",
                "failed_stage": name,
                "completed": completed,
            }
            (run_root / "pipeline_status.json").write_text(
                json.dumps(status, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            return result.returncode
    gate = json.loads(
        (run_root / "analysis" / "held_gate.json").read_text(encoding="utf-8")
    )
    status = {
        "status": "ARTIFACTS_COMPLETE_ANALYZED",
        "held_gate_status": gate["status"],
        "target25_gate_eligible": gate["target25_gate_eligible"],
        "target25_authorized": False,
        "completed": completed,
    }
    (run_root / "pipeline_status.json").write_text(
        json.dumps(status, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(status["status"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
