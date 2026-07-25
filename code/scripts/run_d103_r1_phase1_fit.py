#!/usr/bin/env python3
"""Run one immutable 400-step D103-R1 Phase1 fit.

This entry point accepts only separately sealed L_s/U_s archives and a
non-readable source-validation seal. It produces a ground-side FP32 teacher
artifact for later INT8 compilation; it never creates a Phase2 deployment
bundle, reads source validation arrays, or computes performance.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
import sys
import time
from typing import Any, Mapping

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from cvsrffi.rxid_metabias4_phase1_trainer import (  # noqa: E402
    CANDIDATE_ID,
    D103R1Config,
    D103R1Phase1Trainer,
    OuterMaskSpec,
    build_training_data,
)


FIT_SCHEMA = "cvs.d103_r2.rxid_crossreceiver.phase1_fit.v1"
ARCHIVE_MANIFEST_SCHEMA = "cvs.d103_r2.rxid_dualsplit.source_feature_archive.v1"
LABELED_KEYS = {
    "z_dom",
    "pre_relu",
    "receiver_ids",
    "day_ids",
    "tx_labels",
    "physical_ids",
}
UNLABELED_KEYS = {"z_dom", "receiver_ids", "day_ids", "physical_ids"}


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical(value: Any) -> bytes:
    def convert(item: Any) -> Any:
        if isinstance(item, Mapping):
            return {str(key): convert(member) for key, member in item.items()}
        if isinstance(item, (list, tuple)):
            return [convert(member) for member in item]
        if isinstance(item, np.ndarray):
            return item.tolist()
        if isinstance(item, np.generic):
            return item.item()
        return item

    return (
        json.dumps(
            convert(value),
            ensure_ascii=False,
            sort_keys=True,
            indent=2,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def _normalized_exception(
    exc: Exception, args: argparse.Namespace
) -> tuple[str, str]:
    message = str(exc)
    for field, token in (
        ("held_receiver", "<HELD_RECEIVER>"),
        ("held_class", "<HELD_CLASS>"),
        ("held_day", "<HELD_DAY>"),
    ):
        value = getattr(args, field, None)
        if value:
            message = message.replace(str(value), token)
    for value in (
        getattr(args, "output_dir", None),
        getattr(args, "labeled_archive", None),
        getattr(args, "unlabeled_archive", None),
        getattr(args, "source_val_seal", None),
    ):
        if value:
            message = message.replace(str(Path(value)), "<PATH>")
            try:
                message = message.replace(str(Path(value).resolve()), "<PATH>")
            except OSError:
                pass
    message = re.sub(r"(?i)\bpid\s*[=:]?\s*\d+\b", "pid=<PID>", message)
    message = re.sub(r"\b0x[0-9a-fA-F]+\b", "<ADDRESS>", message)
    message = re.sub(r"\b\d{1,3}-\d{1,3}\b", "<RECEIVER>", message)
    message = re.sub(r"\s+", " ", message).strip()
    normalized = f"{type(exc).__module__}.{type(exc).__name__}:{message}"
    return normalized, hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _load_archive(path: Path, expected_keys: set[str]) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        if set(archive.files) != expected_keys:
            raise ValueError(
                f"archive member closure drift: expected={sorted(expected_keys)}, "
                f"actual={sorted(archive.files)}"
            )
        return {name: np.array(archive[name], copy=True) for name in archive.files}


def _validate_source_manifest(
    manifest: Mapping[str, Any],
    *,
    role: str,
    fraction: float,
    tx_visibility: str,
    archive_path: Path | None,
) -> None:
    exact = {
        "schema",
        "candidate_id",
        "role",
        "fraction",
        "tx_visibility",
        "archive_sha256",
        "target_access",
        "formal_query_access",
        "source_validation_gradient_access",
        "physical_id_unique",
        "checkpoint_sha256",
        "runtime_sha256",
    }
    if set(manifest) != exact:
        raise ValueError(f"{role} manifest key closure drift")
    if (
        manifest.get("schema") != ARCHIVE_MANIFEST_SCHEMA
        or manifest.get("candidate_id") != CANDIDATE_ID
        or manifest.get("role") != role
        or float(manifest.get("fraction", -1.0)) != fraction
        or manifest.get("tx_visibility") != tx_visibility
        or manifest.get("target_access") is not False
        or manifest.get("formal_query_access") is not False
        or manifest.get("source_validation_gradient_access") is not False
        or manifest.get("physical_id_unique") is not True
    ):
        raise ValueError(f"{role} manifest semantic drift")
    if archive_path is None:
        if manifest.get("archive_sha256") not in (None, ""):
            raise ValueError("source_val manifest must not expose an archive")
    elif manifest.get("archive_sha256") != _sha256_file(archive_path):
        raise ValueError(f"{role} archive SHA256 mismatch")
    for field in ("checkpoint_sha256", "runtime_sha256"):
        value = str(manifest.get(field, ""))
        if len(value) != 64 or any(token not in "0123456789abcdef" for token in value):
            raise ValueError(f"{role} {field} drift")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--labeled-archive", type=Path, required=True)
    parser.add_argument("--labeled-manifest", type=Path, required=True)
    parser.add_argument("--unlabeled-archive", type=Path, required=True)
    parser.add_argument("--unlabeled-manifest", type=Path, required=True)
    parser.add_argument("--source-val-seal", type=Path, required=True)
    parser.add_argument("--source-val-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--held-receiver")
    parser.add_argument("--held-day")
    parser.add_argument("--held-class")
    return parser.parse_args()


def _run(args: argparse.Namespace, output: Path) -> int:
    started = time.monotonic()
    if str(args.device).startswith("cuda") and torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats(torch.device(args.device))
    labeled_archive = args.labeled_archive.resolve()
    unlabeled_archive = args.unlabeled_archive.resolve()
    labeled_manifest = _read_json(args.labeled_manifest.resolve())
    unlabeled_manifest = _read_json(args.unlabeled_manifest.resolve())
    source_val_manifest = _read_json(args.source_val_manifest.resolve())
    source_val_seal = _read_json(args.source_val_seal.resolve())

    _validate_source_manifest(
        labeled_manifest,
        role="L_s",
        fraction=0.07,
        tx_visibility="visible",
        archive_path=labeled_archive,
    )
    _validate_source_manifest(
        unlabeled_manifest,
        role="U_s",
        fraction=0.63,
        tx_visibility="hidden",
        archive_path=unlabeled_archive,
    )
    _validate_source_manifest(
        source_val_manifest,
        role="source_val",
        fraction=0.30,
        tx_visibility="scorer_only",
        archive_path=None,
    )
    if (
        labeled_manifest["checkpoint_sha256"]
        != unlabeled_manifest["checkpoint_sha256"]
        or labeled_manifest["runtime_sha256"]
        != unlabeled_manifest["runtime_sha256"]
        or labeled_manifest["checkpoint_sha256"]
        != source_val_manifest["checkpoint_sha256"]
        or labeled_manifest["runtime_sha256"]
        != source_val_manifest["runtime_sha256"]
    ):
        raise ValueError("L_s/U_s/source_val checkpoint/runtime binding drift")
    if set(source_val_seal) != {"row_count", "content_sha256"}:
        raise ValueError("source-val seal key closure drift")

    labeled = _load_archive(labeled_archive, LABELED_KEYS)
    unlabeled = _load_archive(unlabeled_archive, UNLABELED_KEYS)
    data = build_training_data(
        labeled,
        unlabeled,
        source_val_seal,
        config=D103R1Config(),
    )
    spec = OuterMaskSpec(
        held_receiver=args.held_receiver,
        held_day=args.held_day,
        held_class=args.held_class,
    )
    trainer = D103R1Phase1Trainer(data, spec, device=args.device)

    steps_path = output / "step_receipts.jsonl"
    with steps_path.open("xb") as stream:
        for _ in range(data.config.total_meta_steps):
            receipt = trainer.step()
            stream.write(_canonical(receipt.__dict__ if hasattr(receipt, "__dict__") else {
                name: getattr(receipt, name)
                for name in receipt.__dataclass_fields__
            }))
    exported = trainer.export_teacher_arrays()
    teacher_path = output / "teacher_arrays_fp32_ground_only.npz"
    np.savez(
        teacher_path,
        U=exported["U"],
        B=exported["B"],
        bank_g=exported["bank_g"],
        bank_t=exported["bank_t"],
        bank_precision=exported["bank_precision"],
        bank_sigma=exported["bank_sigma"],
    )
    manifest = {
        "schema": FIT_SCHEMA,
        "candidate_id": CANDIDATE_ID,
        "checkpoint_sha256": labeled_manifest["checkpoint_sha256"],
        "runtime_sha256": labeled_manifest["runtime_sha256"],
        "status": "PHASE1_FIT_COMPLETE_GROUND_TEACHER_NOT_DEPLOYMENT",
        "performance_metrics_computed": False,
        "target_access": False,
        "formal_query_access": False,
        "source_val_rows_used_for_training": 0,
        "completed_meta_steps": trainer.completed_steps,
        "fit_elapsed_seconds": time.monotonic() - started,
        "peak_cuda_memory_bytes": (
            int(
                max(
                    torch.cuda.max_memory_allocated(torch.device(args.device)),
                    torch.cuda.max_memory_reserved(torch.device(args.device)),
                )
            )
            if str(args.device).startswith("cuda") and torch.cuda.is_available()
            else 0
        ),
        "outer_spec": {
            "held_receiver": args.held_receiver,
            "held_day": args.held_day,
            "held_class": args.held_class,
        },
        "input_sha256": {
            "labeled_archive": _sha256_file(labeled_archive),
            "unlabeled_archive": _sha256_file(unlabeled_archive),
            "source_val_seal": _sha256_file(args.source_val_seal.resolve()),
        },
        "teacher_archive": {
            "name": teacher_path.name,
            "sha256": _sha256_file(teacher_path),
            "ground_only_fp32": True,
            "phase2_eligible": False,
        },
        "aggregation_receipt": dict(exported["aggregation_receipt"]),
        "access_receipt": dict(exported["access_receipt"]),
        "step_receipts_sha256": _sha256_file(steps_path),
    }
    (output / "fit_complete.json").write_bytes(_canonical(manifest))
    print(output / "fit_complete.json")
    return 0


def main() -> int:
    args = parse_args()
    output = args.output_dir.resolve()
    if output.exists() or output.is_symlink():
        raise FileExistsError(f"immutable fit output already exists: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.mkdir(exist_ok=False)
    try:
        return _run(args, output)
    except Exception as exc:
        normalized, fingerprint = _normalized_exception(exc, args)
        failure = {
            "schema": FIT_SCHEMA,
            "candidate_id": CANDIDATE_ID,
            "status": "PHASE1_FIT_FAILED_NO_PERFORMANCE_RESULT",
            "performance_result": False,
            "partial_artifacts_preserved": True,
            "exception_type": f"{type(exc).__module__}.{type(exc).__name__}",
            "exception_message": str(exc),
            "normalized_exception_template": normalized,
            "normalized_exception_fingerprint": fingerprint,
        }
        failure_path = output / "fit_failed.json"
        if not failure_path.exists():
            failure_path.write_bytes(_canonical(failure))
        raise


if __name__ == "__main__":
    raise SystemExit(main())
