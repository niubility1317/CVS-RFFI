"""Immutable development-only release wrapper for D101 Phase1 nested LODO."""

from __future__ import annotations

import argparse
from dataclasses import fields
import hashlib
import json
import os
from pathlib import Path
import sys
from typing import Any, Mapping


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT / "code") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "code"))

from cvsrffi.phase2_candidate_capsule import BASE_CHECKPOINT_SHA256
from cvsrffi import stage2_d101_phase1_lodo as d101_lodo
from scripts import run_d99_d100_phase1_lodo as d99_runner


CONFIG_SCHEMA = "cvs.release.d101_phase1_lodo.v1"
RESULT_SCHEMA = "cvs.release.d101_phase1_lodo_result.v1"
EXECUTION_MODE = "development_diagnostic"
RECEIPT_FILENAME = "d101_phase1_lodo_development_diagnostic.json"


class D101ReleaseRunnerError(ValueError):
    pass


_CONFIG_FIELDS = {
    "schema",
    "run_id",
    "seed",
    "feature_archive_path",
    "feature_archive_sha256",
    "feature_archive_manifest_path",
    "feature_archive_manifest_sha256",
    "ground_bundle_npz_path",
    "ground_bundle_npz_sha256",
    "ground_release_manifest_path",
    "ground_release_manifest_sha256",
    "base_d99_lock_path",
    "base_d99_lock_sha256",
    "d81_ground_component_dir",
    "d81_ground_manifest_sha256",
    "d81_device",
    "d81_metric_seed",
    "phase1_checkpoint_sha256",
    "d99_d100_grid",
    "d101_grid",
    "gate_lock",
    "expected_module_sha256",
    "execution_mode",
    "output_dir",
}


def _parse_gate_lock(value: Any) -> d101_lodo.D101LODOGateLock:
    expected_fields = {field.name for field in fields(d101_lodo.D101LODOGateLock)}
    if not isinstance(value, Mapping) or set(value) != expected_fields:
        raise D101ReleaseRunnerError("D101 gate_lock field registry drift")
    try:
        return d101_lodo.D101LODOGateLock(**dict(value))
    except (TypeError, ValueError) as exc:
        raise D101ReleaseRunnerError("D101 gate_lock is invalid") from exc


def validate_release_config(value: Mapping[str, Any]) -> dict[str, Any]:
    if set(value) != _CONFIG_FIELDS or value.get("schema") != CONFIG_SCHEMA:
        raise D101ReleaseRunnerError("release config must match exact schema")
    run_id = str(value["run_id"])
    if not d99_runner.RUN_ID_PATTERN.fullmatch(run_id):
        raise D101ReleaseRunnerError("run_id is not immutable-path safe")
    try:
        seed = int(value["seed"])
        metric_seed = int(value["d81_metric_seed"])
    except (TypeError, ValueError) as exc:
        raise D101ReleaseRunnerError("seed values must be integers") from exc
    if seed < 0 or metric_seed < 0:
        raise D101ReleaseRunnerError("seed values must be nonnegative")
    if value["phase1_checkpoint_sha256"] != BASE_CHECKPOINT_SHA256:
        raise D101ReleaseRunnerError("Phase1 checkpoint identity drift")
    if str(value["d81_device"]) not in {"cpu", "cuda"}:
        raise D101ReleaseRunnerError("D81 device must be explicit cpu or cuda")
    if value["execution_mode"] != EXECUTION_MODE:
        raise D101ReleaseRunnerError("D101 wrapper is development_diagnostic-only")
    d101_lodo.base.candidate_grid(value["d99_d100_grid"])
    d101_lodo.d101_candidate_grid(value["d101_grid"])
    gate_lock = _parse_gate_lock(value["gate_lock"])
    expected_code = {
        str(key): d99_runner._require_sha(item, f"module SHA {key}")
        for key, item in dict(value["expected_module_sha256"]).items()
    }
    if expected_code != d101_lodo.current_code_sha256():
        raise D101ReleaseRunnerError("D101 module source SHA registry drift")
    for key in (
        "feature_archive_sha256",
        "feature_archive_manifest_sha256",
        "ground_bundle_npz_sha256",
        "ground_release_manifest_sha256",
        "base_d99_lock_sha256",
        "d81_ground_manifest_sha256",
    ):
        d99_runner._require_sha(value[key], key)
    output = Path(value["output_dir"]).resolve()
    staging = output.with_name(f".{output.name}.{run_id}.staging")
    if output.exists():
        raise D101ReleaseRunnerError("output_dir already exists; overwrite forbidden")
    if staging.exists():
        raise D101ReleaseRunnerError("staging_dir already exists; ambiguous prior attempt")
    return {
        **dict(value),
        "run_id": run_id,
        "seed": seed,
        "d81_metric_seed": metric_seed,
        "gate_lock": gate_lock,
        "expected_module_sha256": expected_code,
        "execution_mode": EXECUTION_MODE,
        "output_dir": str(output),
        "staging_dir": str(staging),
    }


def _validate_development_receipt(receipt: Mapping[str, Any]) -> None:
    if receipt.get("schema") != d101_lodo.SCHEMA:
        raise D101ReleaseRunnerError("D101 receipt schema drift")
    if receipt.get("status") not in {
        d101_lodo.STATUS_ADMITTED,
        d101_lodo.STATUS_REJECTED,
    }:
        raise D101ReleaseRunnerError("D101 diagnostic status drift")
    if (
        receipt.get("formal_phase1_lock") is not False
        or receipt.get("formal_phase2_eligible") is not False
        or receipt.get("target_authority") is not False
        or receipt.get("n607_authority") is not False
        or receipt.get("canonical_lock_artifact_write_allowed") is not False
    ):
        raise D101ReleaseRunnerError("D101 development receipt authority drift")


def _publish_pair_atomic(
    output_dir: Path,
    staging_dir: Path,
    receipt_bytes: bytes,
    result_bytes: bytes,
) -> None:
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    staging_dir.mkdir(exist_ok=False)
    receipt_path = staging_dir / RECEIPT_FILENAME
    result_path = staging_dir / "result.json"
    try:
        d99_runner._exclusive_write(receipt_path, receipt_bytes)
        d99_runner._exclusive_write(result_path, result_bytes)
        if output_dir.exists():
            raise D101ReleaseRunnerError("output_dir appeared during publication")
        os.replace(staging_dir, output_dir)
    except Exception:
        receipt_path.unlink(missing_ok=True)
        result_path.unlink(missing_ok=True)
        try:
            staging_dir.rmdir()
        except FileNotFoundError:
            pass
        raise


def run_from_config(config_path: str | Path, config_sha256: str) -> dict[str, Any]:
    config = validate_release_config(
        d99_runner._read_bound_json(config_path, config_sha256, "release config")
    )
    d99_runner._read_bound_bytes(
        config["feature_archive_path"],
        config["feature_archive_sha256"],
        "feature archive",
    )
    bundle, ground_manifest_raw = d99_runner._load_ground_bundle(config)
    authority = d99_runner.load_ground_release_authority(
        ground_manifest_raw,
        config["ground_release_manifest_sha256"],
        bundle,
    )
    lock_payload = d99_runner._read_bound_json(
        config["base_d99_lock_path"],
        config["base_d99_lock_sha256"],
        "base D99 lock",
    )
    base_d99 = d99_runner._parse_base_d99_lock(lock_payload, EXECUTION_MODE)
    scorer = d99_runner.D81Phase1EpisodeScorer.from_component(
        config["d81_ground_component_dir"],
        config["d81_ground_manifest_sha256"],
        device=config["d81_device"],
        metric_seed=config["d81_metric_seed"],
        phase1_checkpoint_sha256=config["phase1_checkpoint_sha256"],
    )
    receipt = d101_lodo.run_phase1_d101_nested_lodo(
        config["feature_archive_path"],
        config["feature_archive_manifest_path"],
        config["feature_archive_manifest_sha256"],
        ground_bundle=bundle,
        ground_authority=authority,
        base_d99_config=base_d99,
        base_scorer=scorer,
        base_scorer_id=scorer.scorer_id,
        base_scorer_receipt_sha256=scorer.scorer_id,
        d99_d100_grid=config["d99_d100_grid"],
        d101_grid=config["d101_grid"],
        gate_lock=config["gate_lock"],
        code_sha256=config["expected_module_sha256"],
        seed=config["seed"],
    )
    if not d101_lodo.verify_receipt(receipt):
        raise D101ReleaseRunnerError("D101 LODO receipt fixed point failed")
    _validate_development_receipt(receipt)

    receipt_bytes = d99_runner._canonical_bytes(receipt) + b"\n"
    output_dir = Path(config["output_dir"])
    output_path = output_dir / RECEIPT_FILENAME
    result = {
        "schema": RESULT_SCHEMA,
        "run_id": config["run_id"],
        "status": receipt["status"],
        "development_diagnostic_only": True,
        "canonical_lock_artifact": False,
        "formal_phase1_lock": False,
        "formal_phase2_eligible": False,
        "target_authority": False,
        "n607_authority": False,
        "receipt_sha256": receipt["receipt_sha256"],
        "output_path": str(output_path),
        "output_file_sha256": hashlib.sha256(receipt_bytes).hexdigest(),
        "output_file_bytes": len(receipt_bytes),
        "config_sha256": d99_runner._require_sha(config_sha256, "config SHA"),
    }
    result_bytes = d99_runner._canonical_bytes(result) + b"\n"
    _publish_pair_atomic(
        output_dir,
        Path(config["staging_dir"]),
        receipt_bytes,
        result_bytes,
    )
    return result


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--config-sha256", required=True)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    print(json.dumps(run_from_config(args.config, args.config_sha256), sort_keys=True))


if __name__ == "__main__":
    main()
