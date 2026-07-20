"""Immutable file runner for the Phase1-only D99/D100 receiver LODO lock."""

from __future__ import annotations

import argparse
from dataclasses import fields
import hashlib
import io
import json
import os
from pathlib import Path
import re
import sys
from typing import Any, Mapping

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT / "code") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "code"))

from cvsrffi import stage2_d99_ra_cgtmk_d81 as d99
from cvsrffi.phase2_candidate_capsule import BASE_CHECKPOINT_SHA256
from cvsrffi.stage2_d81_phase1_episode_scorer import D81Phase1EpisodeScorer
from cvsrffi.stage2_d99_d100_phase1_lodo import (
    STATUS_DIAGNOSTIC,
    STATUS_FORMAL,
    candidate_grid,
    current_code_sha256,
    load_ground_release_authority,
    run_phase1_d99_d100_lodo,
    verify_receipt,
)


CONFIG_SCHEMA = "cvs.release.d99_d100_phase1_lodo.v1"
RESULT_SCHEMA = "cvs.release.d99_d100_phase1_lodo_result.v1"
DEVELOPMENT_D99_PRIOR_SCHEMA = "cvs.phase1.d99.development_prior_wrapper.v1"
DEVELOPMENT_D99_PRIOR_STATUS = "PREREGISTERED_DEVELOPMENT_PRIORS_NONFORMAL"
DEVELOPMENT_D99_PLACEHOLDER_EVIDENCE_FIELDS = (
    "phase1_receipt_sha256",
    "quantization_margin_audit_sha256",
    "validation_method_lock_sha256",
    "d81_phase1_lock_sha256",
)
GROUND_NPZ_MEMBERS = (
    "codes_qint8",
    "scales_fp16",
    "domain_class_mask",
    "physical_sample_count_floor_uint16",
    "domain_ids",
    "ground_old_registry",
)
RUN_ID_PATTERN = re.compile(r"^[a-z0-9][a-z0-9_.-]{2,127}$")


class D99D100ReleaseRunnerError(ValueError):
    pass


def _sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _require_sha(value: Any, name: str) -> str:
    text = str(value)
    if (
        text != text.lower()
        or len(text) != 64
        or any(character not in "0123456789abcdef" for character in text)
    ):
        raise D99D100ReleaseRunnerError(f"{name} must be lowercase SHA256")
    return text


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=True,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def _read_bound_bytes(path: str | Path, expected_sha256: str, name: str) -> bytes:
    resolved = Path(path).resolve()
    expected = _require_sha(expected_sha256, f"{name} expected SHA")
    if not resolved.is_file() or resolved.is_symlink():
        raise D99D100ReleaseRunnerError(f"{name} must be a regular non-symlink file")
    raw = resolved.read_bytes()
    if hashlib.sha256(raw).hexdigest() != expected:
        raise D99D100ReleaseRunnerError(f"{name} path/SHA256 drift")
    return raw


def _read_bound_json(path: str | Path, expected_sha256: str, name: str) -> dict[str, Any]:
    raw = _read_bound_bytes(path, expected_sha256, name)
    try:
        value = json.loads(raw.decode("utf-8-sig"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise D99D100ReleaseRunnerError(f"{name} is not valid JSON") from exc
    if not isinstance(value, dict):
        raise D99D100ReleaseRunnerError(f"{name} must be a JSON object")
    return value


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
    "candidate_grid",
    "expected_module_sha256",
    "execution_mode",
    "output_dir",
}


def validate_release_config(value: Mapping[str, Any]) -> dict[str, Any]:
    if set(value) != _CONFIG_FIELDS or value.get("schema") != CONFIG_SCHEMA:
        raise D99D100ReleaseRunnerError("release config must match exact schema")
    run_id = str(value["run_id"])
    if not RUN_ID_PATTERN.fullmatch(run_id):
        raise D99D100ReleaseRunnerError("run_id is not immutable-path safe")
    try:
        seed = int(value["seed"])
        metric_seed = int(value["d81_metric_seed"])
    except (TypeError, ValueError) as exc:
        raise D99D100ReleaseRunnerError("seed values must be integers") from exc
    if seed < 0 or metric_seed < 0:
        raise D99D100ReleaseRunnerError("seed values must be nonnegative")
    if value["phase1_checkpoint_sha256"] != BASE_CHECKPOINT_SHA256:
        raise D99D100ReleaseRunnerError("Phase1 checkpoint identity drift")
    if str(value["d81_device"]) not in {"cpu", "cuda"}:
        raise D99D100ReleaseRunnerError("D81 device must be explicit cpu or cuda")
    if value["execution_mode"] not in {"development_diagnostic", "formal_lock"}:
        raise D99D100ReleaseRunnerError(
            "execution_mode must be development_diagnostic or formal_lock"
        )
    candidate_grid(value["candidate_grid"])
    expected_code = {
        str(key): _require_sha(item, f"module SHA {key}")
        for key, item in dict(value["expected_module_sha256"]).items()
    }
    if expected_code != current_code_sha256():
        raise D99D100ReleaseRunnerError("module source SHA registry drift")
    for key in (
        "feature_archive_sha256",
        "feature_archive_manifest_sha256",
        "ground_bundle_npz_sha256",
        "ground_release_manifest_sha256",
        "base_d99_lock_sha256",
        "d81_ground_manifest_sha256",
    ):
        _require_sha(value[key], key)
    output = Path(value["output_dir"]).resolve()
    if output.exists():
        raise D99D100ReleaseRunnerError("output_dir already exists; overwrite forbidden")
    return {
        **dict(value),
        "run_id": run_id,
        "seed": seed,
        "d81_metric_seed": metric_seed,
        "expected_module_sha256": expected_code,
        "output_dir": str(output),
    }


def _load_ground_bundle(config: Mapping[str, Any]) -> tuple[d99.Phase1GroundAggregateBundle, bytes]:
    npz_path = Path(config["ground_bundle_npz_path"]).resolve()
    npz_raw = _read_bound_bytes(
        npz_path, config["ground_bundle_npz_sha256"], "ground bundle NPZ"
    )
    try:
        # Parse the already hashed immutable byte snapshot.  Reopening the path
        # here would create a TOCTOU window between hash and array loading.
        with np.load(io.BytesIO(npz_raw), allow_pickle=False) as payload:
            if tuple(payload.files) != GROUND_NPZ_MEMBERS:
                raise D99D100ReleaseRunnerError("ground bundle NPZ member registry drift")
            arrays = {name: np.asarray(payload[name]) for name in payload.files}
    except (OSError, ValueError) as exc:
        raise D99D100ReleaseRunnerError("ground bundle NPZ cannot be safely loaded") from exc
    manifest_raw = _read_bound_bytes(
        config["ground_release_manifest_path"],
        config["ground_release_manifest_sha256"],
        "ground release manifest",
    )
    try:
        manifest = json.loads(manifest_raw.decode("utf-8"))
        receipt_payload = dict(manifest["aggregation_receipt"])
    except (UnicodeDecodeError, json.JSONDecodeError, KeyError, TypeError) as exc:
        raise D99D100ReleaseRunnerError("ground manifest aggregation receipt missing") from exc
    expected_receipt_fields = {field.name for field in fields(d99.ExternalGroundAggregationReceipt)}
    if set(receipt_payload) != expected_receipt_fields:
        raise D99D100ReleaseRunnerError("ground aggregation receipt schema drift")
    receipt = d99.ExternalGroundAggregationReceipt(**receipt_payload)
    bundle = d99.produce_typed_ground_aggregate_bundle(
        codes_qint8=arrays["codes_qint8"],
        scales_fp16=arrays["scales_fp16"],
        domain_class_mask=arrays["domain_class_mask"],
        physical_sample_count_floor_uint16=arrays[
            "physical_sample_count_floor_uint16"
        ],
        domain_ids=arrays["domain_ids"].astype(str).tolist(),
        ground_old_registry=arrays["ground_old_registry"].astype(str).tolist(),
        aggregation_receipt=receipt,
    )
    return bundle, manifest_raw


def _exclusive_write(path: Path, payload: bytes) -> None:
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
    except Exception:
        try:
            path.unlink(missing_ok=True)
        finally:
            raise


def _validate_execution_outcome(receipt: Mapping[str, Any], execution_mode: str) -> bool:
    formal = receipt.get("status") == STATUS_FORMAL
    if not formal and execution_mode != "development_diagnostic":
        raise D99D100ReleaseRunnerError(
            "formal_lock mode refuses nonformal Phase1 authority"
        )
    if formal and execution_mode != "formal_lock":
        raise D99D100ReleaseRunnerError(
            "development_diagnostic mode cannot emit a formal canonical lock"
        )
    if not formal and receipt.get("status") != STATUS_DIAGNOSTIC:
        raise D99D100ReleaseRunnerError("nonformal diagnostic status drift")
    return formal


def _parse_base_d99_lock(
    payload: Mapping[str, Any], execution_mode: str
) -> d99.Phase1D99Lock:
    expected_lock_fields = {field.name for field in fields(d99.Phase1D99Lock)}
    wrapper = (
        isinstance(payload, Mapping)
        and payload.get("schema") == DEVELOPMENT_D99_PRIOR_SCHEMA
        and payload.get("status") == DEVELOPMENT_D99_PRIOR_STATUS
    )
    if execution_mode == "formal_lock":
        if wrapper:
            raise D99D100ReleaseRunnerError(
                "formal_lock refuses nonformal development D99 prior wrapper"
            )
        values = dict(payload)
    elif execution_mode == "development_diagnostic":
        if not wrapper or set(payload) != {
            "schema",
            "status",
            "values",
            "placeholder_evidence_fields",
        }:
            raise D99D100ReleaseRunnerError(
                "development_diagnostic requires exact nonformal D99 prior wrapper"
            )
        if tuple(payload["placeholder_evidence_fields"]) != (
            DEVELOPMENT_D99_PLACEHOLDER_EVIDENCE_FIELDS
        ):
            raise D99D100ReleaseRunnerError(
                "development D99 placeholder evidence registry drift"
            )
        if not isinstance(payload["values"], Mapping):
            raise D99D100ReleaseRunnerError("development D99 values must be an object")
        values = dict(payload["values"])
    else:
        raise D99D100ReleaseRunnerError("unsupported execution_mode")
    if set(values) != expected_lock_fields:
        raise D99D100ReleaseRunnerError("base D99 lock field registry drift")
    values["ground_old_registry"] = tuple(values["ground_old_registry"])
    return d99.Phase1D99Lock(**values)


def run_from_config(config_path: str | Path, config_sha256: str) -> dict[str, Any]:
    config = validate_release_config(
        _read_bound_json(config_path, config_sha256, "release config")
    )
    _read_bound_bytes(
        config["feature_archive_path"],
        config["feature_archive_sha256"],
        "feature archive",
    )
    bundle, ground_manifest_raw = _load_ground_bundle(config)
    authority = load_ground_release_authority(
        ground_manifest_raw,
        config["ground_release_manifest_sha256"],
        bundle,
    )
    lock_payload = _read_bound_json(
        config["base_d99_lock_path"], config["base_d99_lock_sha256"], "base D99 lock"
    )
    base_d99 = _parse_base_d99_lock(lock_payload, config["execution_mode"])
    scorer = D81Phase1EpisodeScorer.from_component(
        config["d81_ground_component_dir"],
        config["d81_ground_manifest_sha256"],
        device=config["d81_device"],
        metric_seed=config["d81_metric_seed"],
        phase1_checkpoint_sha256=config["phase1_checkpoint_sha256"],
    )
    receipt = run_phase1_d99_d100_lodo(
        config["feature_archive_path"],
        config["feature_archive_manifest_path"],
        config["feature_archive_manifest_sha256"],
        ground_bundle=bundle,
        ground_authority=authority,
        base_d99_config=base_d99,
        base_scorer=scorer,
        base_scorer_id=scorer.scorer_id,
        base_scorer_receipt_sha256=scorer.scorer_id,
        grid=config["candidate_grid"],
        code_sha256=config["expected_module_sha256"],
        seed=config["seed"],
    )
    if not verify_receipt(receipt):
        raise D99D100ReleaseRunnerError("LODO receipt fixed point failed")
    formal = _validate_execution_outcome(receipt, config["execution_mode"])
    filename = (
        "d99_d100_phase1_lodo_lock.json"
        if formal
        else "d99_d100_phase1_lodo_blocked_diagnostic.json"
    )
    receipt_bytes = _canonical_bytes(receipt) + b"\n"
    output_dir = Path(config["output_dir"])
    output_path = output_dir / filename
    result = {
        "schema": RESULT_SCHEMA,
        "run_id": config["run_id"],
        "status": receipt["status"],
        "canonical_lock_artifact": formal,
        "receipt_sha256": receipt["receipt_sha256"],
        "output_path": str(output_path),
        "output_file_sha256": hashlib.sha256(receipt_bytes).hexdigest(),
        "output_file_bytes": len(receipt_bytes),
        "config_sha256": _require_sha(config_sha256, "config SHA"),
    }
    result_bytes = _canonical_bytes(result) + b"\n"
    # Serialize both payloads before creating the output directory.  A JSON
    # conversion error can therefore never leave a partial canonical receipt.
    output_dir.mkdir(parents=True, exist_ok=False)
    _exclusive_write(output_path, receipt_bytes)
    _exclusive_write(output_dir / "result.json", result_bytes)
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
