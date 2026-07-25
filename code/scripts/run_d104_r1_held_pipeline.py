#!/usr/bin/env python3
"""Run the complete frozen D104 Phase1 source-held evidence pipeline."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
from typing import Sequence


ROOT = Path(__file__).resolve().parents[1]


def _read_object(path: Path) -> dict[str, object]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_sha256(value: object) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _require_sha256(value: object, name: str) -> str:
    text = str(value)
    if (
        len(text) != 64
        or any(character not in "0123456789abcdef" for character in text)
    ):
        raise ValueError(f"{name} must be a lowercase SHA256")
    return text


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-split-root", type=Path, required=True)
    parser.add_argument("--checkpoint-sha256", required=True)
    parser.add_argument("--runtime-sha256", required=True)
    parser.add_argument("--method-lock-sha256", required=True)
    parser.add_argument("--python", type=Path, default=Path(sys.executable))
    parser.add_argument("--gpus", default="0,1,2,3,4,5,6,7")
    parser.add_argument("--workers-per-gpu", type=int, choices=(1, 2), default=2)
    parser.add_argument("--run-root", type=Path, required=True)
    return parser.parse_args()


def _write_new(path: Path, value: object) -> None:
    payload = json.dumps(value, sort_keys=True, indent=2) + "\n"
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        stream.write(payload)


def main() -> int:
    args = parse_args()
    source = args.source_split_root.resolve(strict=True)
    run_root = args.run_root.resolve()
    if run_root.exists() or run_root.is_symlink():
        raise FileExistsError(f"immutable D104 run root exists: {run_root}")
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
        "source_val_scorer_manifest": (
            source / "scorer_only" / "source_val" / "manifest.json"
        ),
    }
    root_manifest_path = source / "source_split_manifest.json"
    root_manifest = _read_object(root_manifest_path)
    roles = root_manifest.get("roles")
    source_val = root_manifest.get("source_val")
    inputs = root_manifest.get("inputs")
    if (
        root_manifest.get("schema")
        != "cvs.d104_r1.source_split.archive.v2"
        or root_manifest.get("candidate_id") != "D104-R1-ANGQ-RXID-MB4"
        or root_manifest.get("split_id") != "d104_source_seed104713_v2"
        or root_manifest.get("status")
        != "FORMAL_PHASE1_SOURCE_SPLIT_COMPLETE"
        or root_manifest.get("protocol_schema") != "p2_min_v1"
        or root_manifest.get("target_access") is not False
        or root_manifest.get("formal_query_access") is not False
        or not isinstance(roles, dict)
        or set(roles) != {"L_s", "U_s"}
        or not isinstance(source_val, dict)
        or not isinstance(inputs, dict)
        or inputs.get("checkpoint_sha256") != args.checkpoint_sha256
        or inputs.get("runtime_sha256") != args.runtime_sha256
    ):
        raise ValueError("D104 source split root manifest closure drift")
    expected_role_paths = {
        "L_s": {
            "archive": Path("L_s") / "features.npz",
            "manifest": Path("L_s") / "manifest.json",
            "row_count": 588,
        },
        "U_s": {
            "archive": Path("U_s") / "features.npz",
            "manifest": Path("U_s") / "manifest.json",
            "row_count": 5292,
        },
    }
    for role, expected in expected_role_paths.items():
        row = roles[role]
        archive = (source / str(row.get("archive", ""))).resolve(strict=True)
        manifest = (source / str(row.get("manifest", ""))).resolve(strict=True)
        if (
            not archive.is_relative_to(source)
            or not manifest.is_relative_to(source)
            or archive != (source / expected["archive"]).resolve(strict=True)
            or manifest != (source / expected["manifest"]).resolve(strict=True)
            or row.get("row_count") != expected["row_count"]
            or row.get("archive_sha256") != _sha256_file(archive)
            or row.get("manifest_sha256") != _sha256_file(manifest)
        ):
            raise ValueError(f"D104 source split {role} binding drift")
    source_val_paths = {
        "fit_manifest": common["source_val_manifest"],
        "seal": common["source_val_seal"],
        "scorer_manifest": common["source_val_scorer_manifest"],
    }
    for field, path in source_val_paths.items():
        actual = (source / str(source_val.get(field, ""))).resolve(strict=True)
        expected = path.resolve(strict=True)
        if (
            actual != expected
            or not actual.is_relative_to(source)
            or source_val.get(f"{field}_sha256") != _sha256_file(actual)
        ):
            raise ValueError(f"D104 source-val {field} binding drift")
    scorer_archive_row = source_val.get("scorer_archive")
    if (
        not isinstance(scorer_archive_row, dict)
        or (source / str(scorer_archive_row.get("path", ""))).resolve(
            strict=True
        )
        != common["source_val_archive"].resolve(strict=True)
        or scorer_archive_row.get("sha256")
        != _sha256_file(common["source_val_archive"].resolve(strict=True))
    ):
        raise ValueError("D104 source-val scorer archive binding drift")
    method_lock_sha = _require_sha256(
        args.method_lock_sha256,
        "method lock",
    )
    run_input_binding = {
        "schema": "cvs.d104_r1.rxid_angq.run_input_binding.v1",
        "split_id": "d104_source_seed104713_v2",
        "source_split_manifest_sha256": _sha256_file(root_manifest_path),
        "historical_exclusion_manifest": root_manifest[
            "historical_exclusion_manifest"
        ],
        "checkpoint_sha256": args.checkpoint_sha256,
        "runtime_sha256": args.runtime_sha256,
        "method_lock_sha256": method_lock_sha,
        "matrix_fit_input_sha256": {
            "labeled_archive": _sha256_file(common["labeled_archive"]),
            "unlabeled_archive": _sha256_file(common["unlabeled_archive"]),
            "source_val_seal": _sha256_file(common["source_val_seal"]),
        },
        "matrix_fit_input_manifest_sha256": {
            "labeled_manifest": _sha256_file(common["labeled_manifest"]),
            "unlabeled_manifest": _sha256_file(common["unlabeled_manifest"]),
            "source_val_manifest": _sha256_file(
                common["source_val_manifest"]
            ),
        },
        "source_val_scorer_manifest_sha256": _sha256_file(
            common["source_val_scorer_manifest"]
        ),
        "source_val_scorer_archive_sha256": _sha256_file(
            common["source_val_archive"]
        ),
        "target_access": False,
        "formal_query_access": False,
    }
    run_input_binding["receipt_sha256"] = _canonical_sha256(
        run_input_binding
    )
    run_root.mkdir(parents=True, exist_ok=False)
    logs = run_root / "logs"
    logs.mkdir()
    _write_new(run_root / "run_input_binding.json", run_input_binding)
    commands: list[tuple[str, Sequence[str]]] = [
        (
            "fit_matrix_246",
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
            "prepare_21_packages_and_tx_probe",
            (
                python,
                str(scripts / "prepare_d104_r1_held_packages.py"),
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
            "seal_63_rows_252_arm_units",
            (
                python,
                str(scripts / "run_d104_r1_held_predictor.py"),
                "--package-root",
                str(run_root / "held_packages"),
                "--fits-root",
                str(run_root / "matrix" / "fits"),
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
            "independent_truth_side_score",
            (
                python,
                str(scripts / "score_d104_r1_held_predictions.py"),
                "--prediction-root",
                str(run_root / "predictions"),
                "--truth-json",
                str(run_root / "held_packages" / "scorer_only" / "truth.json"),
                "--truth-input-seal-json",
                str(
                    run_root
                    / "held_packages"
                    / "scorer_only"
                    / "truth_input_seal.json"
                ),
                "--output-json",
                str(run_root / "scores" / "held_scores.json"),
                "--truth-open-event-json",
                str(run_root / "scores" / "truth_first_open.json"),
            ),
        ),
        (
            "runner_resources",
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
            "frozen_held_gate",
            (
                python,
                str(scripts / "finalize_d104_r1_held_gate.py"),
                "--scores-json",
                str(run_root / "scores" / "held_scores.json"),
                "--tx-probe-json",
                str(run_root / "held_packages" / "tx_probe_receipt.json"),
                "--matrix-status-json",
                str(run_root / "matrix" / "matrix_status.json"),
                "--runner-resource-json",
                str(run_root / "analysis" / "runner_resources.json"),
                "--source-split-manifest",
                str(source / "source_split_manifest.json"),
                "--run-input-binding-json",
                str(run_root / "run_input_binding.json"),
                "--output-json",
                str(run_root / "analysis" / "held_gate.json"),
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
            _write_new(
                run_root / "pipeline_status.json",
                {
                    "status": "PIPELINE_TECHNICAL_FAILURE_NO_PERFORMANCE_RESULT",
                    "failed_stage": name,
                    "completed": completed,
                },
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
    _write_new(run_root / "pipeline_status.json", status)
    print(status["status"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
