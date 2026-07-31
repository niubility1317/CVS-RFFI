#!/usr/bin/env python3
"""Run the fail-closed D105 Phase1 source-only evidence chain.

The normal order is deliberately fixed:

``tap-runtime`` or ``tap-cache`` -> ``predict-source-held`` ->
``open-truth`` -> ``score-source-held`` -> ``build`` -> ``seal``.

The predictor never receives truth labels.  The truth-side scorer refuses to
run until the complete prediction manifest is already immutable.  ``build``
does not accept caller-declared gate booleans; it replays the joined evidence.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import stat
import sys
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT / "code") not in sys.path:
    sys.path.insert(0, str(ROOT / "code"))

from cvsrffi.stage2_d105_phase1_bundle import (  # noqa: E402
    D105_STRICT_TAP_FORWARD_BATCH_CAPACITY,
    D105Phase1BundleError,
    build_d105_exact_model_from_checkpoint,
    build_d105_phase1_component,
    build_d105_source_access_receipt,
    derive_d105_source_held_gate,
    execute_d105_source_held_predictions,
    export_d105_phase1_strict_tap,
    load_d105_exact_sha_bound_checkpoint,
    load_d105_candidate_method_lock,
    load_d105_candidate_runtime_manifest,
    load_d105_strict_tap_rows,
    load_d105_tap_cache_selection_salt,
    load_d105_tap_cache_source_validation_set,
    open_d105_source_held_truth,
    score_d105_source_held_truth,
    select_d105_tap_cache_observations,
    seal_d105_phase1_component,
    sha256_file,
    validate_d105_phase1_asset,
)
from cvsrffi.stage2_d105_phase1_authority import (  # noqa: E402
    load_signed_d102_revocation_manifest,
)


SOURCE_INPUT_MEMBERS = (
    "received_iq",
    "labels",
    "receiver_ids",
    "physical_ids",
)

# The historical dual archive is consumed only as a byte-parity comparator.
# Keep its immutable member contract local so ``tap-cache`` never imports the
# legacy dual-exporter module (and therefore never loads its paper-reproduction
# implementation tree).
REFERENCE_DUAL_ARCHIVE_MEMBERS = (
    "z_id",
    "z_dom",
    "tx_logits",
    "labels",
    "receiver_ids",
    "day_ids",
    "physical_ids",
    "scenario_names",
    "class_ids",
    "observation_ids",
)


def _print_result(value: dict[str, object]) -> None:
    print(json.dumps(value, ensure_ascii=False, sort_keys=True))


def _regular(path: Path, name: str) -> Path:
    if path.is_symlink() or not path.is_file():
        raise D105Phase1BundleError(f"{name} must be a regular non-symlink file")
    return path


def _new_immutable_json(path: Path, value: dict[str, Any]) -> None:
    if path.exists() or path.is_symlink() or not path.parent.is_dir():
        raise D105Phase1BundleError("gate output must be a new child file")
    payload = json.dumps(
        value, ensure_ascii=True, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    with path.open("xb") as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())
    os.chmod(path, stat.S_IREAD)
    if path.stat().st_mode & (stat.S_IWUSR | stat.S_IWGRP | stat.S_IWOTH):
        raise D105Phase1BundleError("gate output remained writable")


def _load_source_input(path: Path) -> dict[str, np.ndarray]:
    archive = _regular(path, "source input NPZ")
    try:
        with np.load(archive, allow_pickle=False) as loaded:
            if tuple(loaded.files) != SOURCE_INPUT_MEMBERS:
                raise D105Phase1BundleError("source input NPZ member order drift")
            arrays = {
                name: np.ascontiguousarray(np.array(loaded[name], copy=True))
                for name in SOURCE_INPUT_MEMBERS
            }
    except (OSError, ValueError, KeyError) as error:
        if isinstance(error, D105Phase1BundleError):
            raise
        raise D105Phase1BundleError("source input NPZ cannot be read safely") from error
    return arrays


def _load_candidate_identity(
    *,
    candidate_runtime_manifest: Path,
    candidate_method_lock: Path,
    checkpoint_sha256: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Load the actual D105 closure before touching a source input."""

    runtime = load_d105_candidate_runtime_manifest(
        _regular(candidate_runtime_manifest, "candidate runtime manifest"),
        expected_checkpoint_sha256=checkpoint_sha256,
    )
    lock = load_d105_candidate_method_lock(
        _regular(candidate_method_lock, "candidate method lock"),
        expected_checkpoint_sha256=checkpoint_sha256,
        expected_runtime_sha256=runtime["d105_candidate_runtime_manifest_sha256"],
    )
    return runtime, lock


def _reference_dual_parity(
    *,
    reference_archive: Path,
    reference_archive_sha256: str,
    strict_tap_archive: Path,
    strict_tap_receipt: Path,
) -> dict[str, object]:
    """Use the historical dual archive only as a byte-parity comparator."""

    reference = _regular(reference_archive, "reference dual archive")
    observed_reference_sha = sha256_file(reference)
    if observed_reference_sha != reference_archive_sha256:
        raise D105Phase1BundleError("reference dual archive SHA256 drift")
    try:
        with np.load(reference, allow_pickle=False) as loaded:
            if tuple(loaded.files) != REFERENCE_DUAL_ARCHIVE_MEMBERS:
                raise D105Phase1BundleError("reference dual archive member closure drift")
            reference_arrays = {
                name: np.ascontiguousarray(np.array(loaded[name], copy=True))
                for name in ("z_id", "z_dom", "labels", "receiver_ids", "physical_ids")
            }
    except (OSError, ValueError, KeyError) as error:
        if isinstance(error, D105Phase1BundleError):
            raise
        raise D105Phase1BundleError("reference dual archive cannot be read safely") from error
    strict_rows, _ = load_d105_strict_tap_rows(strict_tap_archive, strict_tap_receipt)
    if (
        not np.array_equal(reference_arrays["labels"].astype(str), np.asarray(strict_rows.labels))
        or not np.array_equal(
            reference_arrays["receiver_ids"].astype(str), np.asarray(strict_rows.receiver_ids)
        )
        or not np.array_equal(
            reference_arrays["physical_ids"].astype(str), np.asarray(strict_rows.physical_ids)
        )
    ):
        raise D105Phase1BundleError("reference dual archive metadata parity drift")
    z_id_max_abs = float(
        np.max(
            np.abs(
                np.asarray(reference_arrays["z_id"], dtype=np.float64)
                - np.maximum(strict_rows.pre_relu, np.float32(0.0)).astype(np.float64)
            )
        )
    )
    z_dom_max_abs = float(
        np.max(
            np.abs(
                np.asarray(reference_arrays["z_dom"], dtype=np.float64)
                - np.asarray(strict_rows.z_dom, dtype=np.float64)
            )
        )
    )
    if not np.isfinite(z_id_max_abs) or not np.isfinite(z_dom_max_abs):
        raise D105Phase1BundleError("reference dual archive parity is non-finite")
    if z_id_max_abs > 1.0e-5 or z_dom_max_abs > 1.0e-5:
        raise D105Phase1BundleError(
            "strict D105 tap/reference dual archive parity failed"
        )
    return {
        "reference_dual_archive_sha256": observed_reference_sha,
        "reference_used_for_byte_parity_only": True,
        "reference_z_id_max_abs": z_id_max_abs,
        "reference_z_dom_max_abs": z_dom_max_abs,
    }


def _tap_from_runtime(args: argparse.Namespace) -> dict[str, object]:
    try:
        import torch
    except ImportError as error:
        raise D105Phase1BundleError("tap-runtime requires PyTorch") from error
    source = _load_source_input(args.source_input_npz)
    feature_runtime = _regular(args.runtime_file, "strict feature runtime")
    checkpoint = _regular(args.checkpoint, "checkpoint")
    checkpoint_sha = sha256_file(checkpoint)
    candidate_runtime, candidate_lock = _load_candidate_identity(
        candidate_runtime_manifest=args.candidate_runtime_manifest,
        candidate_method_lock=args.candidate_method_lock,
        checkpoint_sha256=checkpoint_sha,
    )
    runtime_sha = candidate_runtime["d105_candidate_runtime_manifest_sha256"]
    method_lock_sha = candidate_lock["d105_candidate_method_lock_sha256"]
    revocation = load_signed_d102_revocation_manifest(
        _regular(args.d102_revocation_manifest, "D102 revocation manifest"),
        _regular(args.d102_revocation_signature, "D102 revocation signature"),
    )
    try:
        model = torch.jit.load(str(feature_runtime), map_location=args.device)
    except (RuntimeError, OSError) as error:
        raise D105Phase1BundleError("runtime TorchScript load failed") from error
    model.eval()
    access = build_d105_source_access_receipt(
        source_received_iq=source["received_iq"],
        source_labels=source["labels"].astype(str).tolist(),
        source_receiver_ids=source["receiver_ids"].astype(str).tolist(),
        source_physical_ids=source["physical_ids"].astype(str).tolist(),
        checkpoint_sha256=checkpoint_sha,
        runtime_sha256=runtime_sha,
        method_lock_sha256=method_lock_sha,
        d102_revocation_manifest_sha256=revocation["manifest_sha256"],
    )
    return export_d105_phase1_strict_tap(
        model=model,
        source_received_iq=source["received_iq"],
        source_labels=source["labels"].astype(str).tolist(),
        source_receiver_ids=source["receiver_ids"].astype(str).tolist(),
        source_physical_ids=source["physical_ids"].astype(str).tolist(),
        checkpoint_sha256=checkpoint_sha,
        runtime_sha256=runtime_sha,
        method_lock_sha256=method_lock_sha,
        source_access_receipt=access,
        d102_revocation_manifest=args.d102_revocation_manifest,
        d102_revocation_signature=args.d102_revocation_signature,
        output_dir=args.output_dir,
        device=args.device,
        batch_size=args.batch_size,
    )


def _tap_from_cache(args: argparse.Namespace) -> dict[str, object]:
    """Create a new D105 strict tap directly from verified source weak-IQ cache."""

    if (
        type(args.batch_size) is not int
        or args.batch_size != D105_STRICT_TAP_FORWARD_BATCH_CAPACITY
    ):
        raise D105Phase1BundleError(
            "tap-cache batch_size must equal fixed forward capacity 256"
        )
    try:
        import torch
    except ImportError as error:
        raise D105Phase1BundleError("tap-cache dependencies are unavailable") from error
    cache = _regular(args.cache_set, "source cache set")
    salt_path = _regular(args.selection_salt_receipt, "selection salt receipt")
    checkpoint = _regular(args.checkpoint, "checkpoint")
    if sha256_file(cache) != args.cache_set_sha256:
        raise D105Phase1BundleError("source cache set SHA256 drift")
    if sha256_file(salt_path) != args.selection_salt_receipt_sha256:
        raise D105Phase1BundleError("selection salt SHA256 drift")
    checkpoint_sha = sha256_file(checkpoint)
    if checkpoint_sha != args.checkpoint_sha256:
        raise D105Phase1BundleError("checkpoint SHA256 drift")
    candidate_runtime, candidate_lock = _load_candidate_identity(
        candidate_runtime_manifest=args.candidate_runtime_manifest,
        candidate_method_lock=args.candidate_method_lock,
        checkpoint_sha256=checkpoint_sha,
    )
    runtime_sha = candidate_runtime["d105_candidate_runtime_manifest_sha256"]
    method_lock_sha = candidate_lock["d105_candidate_method_lock_sha256"]
    revocation = load_signed_d102_revocation_manifest(
        _regular(args.d102_revocation_manifest, "D102 revocation manifest"),
        _regular(args.d102_revocation_signature, "D102 revocation signature"),
    )
    try:
        device = torch.device(args.device)
    except (TypeError, RuntimeError) as error:
        raise D105Phase1BundleError("tap-cache device is invalid") from error
    if (
        device.type != "cuda"
        or not torch.cuda.is_available()
        or device.index is None
        or device.index >= torch.cuda.device_count()
    ):
        raise D105Phase1BundleError("tap-cache requires an explicit available CUDA device")
    checkpoint_payload, _ = load_d105_exact_sha_bound_checkpoint(
        checkpoint, checkpoint_sha
    )
    model, _ = build_d105_exact_model_from_checkpoint(
        checkpoint_payload, input_len=256, device=device
    )
    if model.training:
        raise D105Phase1BundleError("D105 exact checkpoint model remained in training mode")
    salt = load_d105_tap_cache_selection_salt(
        salt_path,
        args.selection_salt_receipt_sha256,
        checkpoint_sha256=checkpoint_sha,
    )
    arrays_by_scenario, _, _ = load_d105_tap_cache_source_validation_set(cache)
    metadata, selected_iq = select_d105_tap_cache_observations(
        arrays_by_scenario, salt["selection_salt_sha256"]
    )
    access = build_d105_source_access_receipt(
        source_received_iq=np.asarray(selected_iq, dtype=np.float32),
        source_labels=metadata["labels"].astype(str).tolist(),
        source_receiver_ids=metadata["receiver_ids"].astype(str).tolist(),
        source_physical_ids=metadata["physical_ids"].astype(str).tolist(),
        checkpoint_sha256=checkpoint_sha,
        runtime_sha256=runtime_sha,
        method_lock_sha256=method_lock_sha,
        d102_revocation_manifest_sha256=revocation["manifest_sha256"],
    )
    result = export_d105_phase1_strict_tap(
        model=model,
        source_received_iq=np.asarray(selected_iq, dtype=np.float32),
        source_labels=metadata["labels"].astype(str).tolist(),
        source_receiver_ids=metadata["receiver_ids"].astype(str).tolist(),
        source_physical_ids=metadata["physical_ids"].astype(str).tolist(),
        checkpoint_sha256=checkpoint_sha,
        runtime_sha256=runtime_sha,
        method_lock_sha256=method_lock_sha,
        source_access_receipt=access,
        d102_revocation_manifest=args.d102_revocation_manifest,
        d102_revocation_signature=args.d102_revocation_signature,
        output_dir=args.output_dir,
        device=args.device,
        batch_size=args.batch_size,
    )
    parity = _reference_dual_parity(
        reference_archive=args.reference_dual_archive,
        reference_archive_sha256=args.reference_dual_archive_sha256,
        strict_tap_archive=Path(str(result["strict_tap_archive"])),
        strict_tap_receipt=Path(str(result["strict_tap_receipt"])),
    )
    del selected_iq, model, checkpoint_payload
    torch.cuda.empty_cache()
    return {**result, **parity}


def _tap_runtime_command(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--source-input-npz", type=Path, required=True)
    parser.add_argument("--runtime-file", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--candidate-runtime-manifest", type=Path, required=True)
    parser.add_argument("--candidate-method-lock", type=Path, required=True)
    parser.add_argument("--d102-revocation-manifest", type=Path, required=True)
    parser.add_argument("--d102-revocation-signature", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--batch-size",
        type=int,
        default=D105_STRICT_TAP_FORWARD_BATCH_CAPACITY,
        help=(
            "deprecated compatibility option; execution always uses fixed "
            "forward capacity 256"
        ),
    )


def _tap_cache_command(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--cache-set", type=Path, required=True)
    parser.add_argument("--cache-set-sha256", required=True)
    parser.add_argument("--selection-salt-receipt", type=Path, required=True)
    parser.add_argument("--selection-salt-receipt-sha256", required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--checkpoint-sha256", required=True)
    parser.add_argument("--candidate-runtime-manifest", type=Path, required=True)
    parser.add_argument("--candidate-method-lock", type=Path, required=True)
    parser.add_argument("--d102-revocation-manifest", type=Path, required=True)
    parser.add_argument("--d102-revocation-signature", type=Path, required=True)
    parser.add_argument("--reference-dual-archive", type=Path, required=True)
    parser.add_argument("--reference-dual-archive-sha256", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--batch-size", type=int, default=D105_STRICT_TAP_FORWARD_BATCH_CAPACITY
    )


def _predict_command(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--strict-tap-archive", type=Path, required=True)
    parser.add_argument("--strict-tap-receipt", type=Path, required=True)
    parser.add_argument("--candidate-runtime-manifest", type=Path, required=True)
    parser.add_argument("--candidate-method-lock", type=Path, required=True)
    parser.add_argument("--output-prediction-manifest", type=Path, required=True)


def _truth_open_command(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--strict-tap-archive", type=Path, required=True)
    parser.add_argument("--strict-tap-receipt", type=Path, required=True)
    parser.add_argument("--candidate-runtime-manifest", type=Path, required=True)
    parser.add_argument("--candidate-method-lock", type=Path, required=True)
    parser.add_argument("--source-held-prediction-manifest", type=Path, required=True)
    parser.add_argument("--output-receipt", type=Path, required=True)


def _score_command(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--strict-tap-archive", type=Path, required=True)
    parser.add_argument("--strict-tap-receipt", type=Path, required=True)
    parser.add_argument("--candidate-runtime-manifest", type=Path, required=True)
    parser.add_argument("--candidate-method-lock", type=Path, required=True)
    parser.add_argument("--source-held-prediction-manifest", type=Path, required=True)
    parser.add_argument("--source-held-truth-open-receipt", type=Path, required=True)
    parser.add_argument("--output-score-artifact", type=Path, required=True)


def _gate_command(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--strict-tap-archive", type=Path, required=True)
    parser.add_argument("--strict-tap-receipt", type=Path, required=True)
    parser.add_argument("--candidate-runtime-manifest", type=Path, required=True)
    parser.add_argument("--candidate-method-lock", type=Path, required=True)
    parser.add_argument("--source-held-prediction-manifest", type=Path, required=True)
    parser.add_argument("--source-held-truth-open-receipt", type=Path, required=True)
    parser.add_argument("--source-held-score-artifact", type=Path, required=True)
    parser.add_argument("--output-receipt", type=Path, required=True)


def _build_command(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--strict-tap-archive", type=Path, required=True)
    parser.add_argument("--strict-tap-receipt", type=Path, required=True)
    parser.add_argument("--candidate-runtime-manifest", type=Path, required=True)
    parser.add_argument("--candidate-method-lock", type=Path, required=True)
    parser.add_argument("--source-held-prediction-manifest", type=Path, required=True)
    parser.add_argument("--source-held-truth-open-receipt", type=Path, required=True)
    parser.add_argument("--source-held-score-artifact", type=Path, required=True)
    parser.add_argument("--d102-revocation-manifest", type=Path, required=True)
    parser.add_argument("--d102-revocation-signature", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)


def _seal_command(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--component-dir", type=Path, required=True)
    parser.add_argument("--authority-envelope", type=Path, required=True)
    parser.add_argument("--authority-signature", type=Path, required=True)
    parser.add_argument("--independent-review-receipt", type=Path, required=True)
    parser.add_argument("--d102-revocation-manifest", type=Path, required=True)
    parser.add_argument("--d102-revocation-signature", type=Path, required=True)
    parser.add_argument("--nonce-ledger-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)


def _validate_command(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--bundle-dir", type=Path, required=True)
    parser.add_argument("--require-formal-phase2-eligible", action="store_true")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    tap_runtime_parser = subparsers.add_parser("tap-runtime")
    tap_cache_parser = subparsers.add_parser("tap-cache")
    predict_parser = subparsers.add_parser("predict-source-held")
    truth_open_parser = subparsers.add_parser("open-truth")
    score_parser = subparsers.add_parser("score-source-held")
    gate_parser = subparsers.add_parser("derive-gate")
    build_parser = subparsers.add_parser("build")
    seal_parser = subparsers.add_parser("seal")
    validate_parser = subparsers.add_parser("validate")
    _tap_runtime_command(tap_runtime_parser)
    _tap_cache_command(tap_cache_parser)
    _predict_command(predict_parser)
    _truth_open_command(truth_open_parser)
    _score_command(score_parser)
    _gate_command(gate_parser)
    _build_command(build_parser)
    _seal_command(seal_parser)
    _validate_command(validate_parser)
    args = parser.parse_args(argv)
    try:
        if args.command == "tap-runtime":
            _print_result(_tap_from_runtime(args))
        elif args.command == "tap-cache":
            _print_result(_tap_from_cache(args))
        elif args.command == "predict-source-held":
            _print_result(
                execute_d105_source_held_predictions(
                    args.strict_tap_archive,
                    args.strict_tap_receipt,
                    args.candidate_method_lock,
                    args.candidate_runtime_manifest,
                    args.output_prediction_manifest,
                )
            )
        elif args.command == "open-truth":
            _print_result(
                open_d105_source_held_truth(
                    args.strict_tap_archive,
                    args.strict_tap_receipt,
                    args.candidate_method_lock,
                    args.candidate_runtime_manifest,
                    args.source_held_prediction_manifest,
                    args.output_receipt,
                )
            )
        elif args.command == "score-source-held":
            _print_result(
                score_d105_source_held_truth(
                    args.strict_tap_archive,
                    args.strict_tap_receipt,
                    args.candidate_method_lock,
                    args.candidate_runtime_manifest,
                    args.source_held_prediction_manifest,
                    args.source_held_truth_open_receipt,
                    args.output_score_artifact,
                )
            )
        elif args.command == "derive-gate":
            result = derive_d105_source_held_gate(
                args.strict_tap_archive,
                args.strict_tap_receipt,
                args.candidate_method_lock,
                args.candidate_runtime_manifest,
                args.source_held_prediction_manifest,
                args.source_held_truth_open_receipt,
                args.source_held_score_artifact,
            )
            _new_immutable_json(args.output_receipt, result["gate"])
            _print_result(
                {
                    "gate_receipt": str(args.output_receipt),
                    "gate_receipt_sha256": result["gate_sha256"],
                    "formal_prerequisites_missing": result[
                        "formal_prerequisites_missing"
                    ],
                }
            )
        elif args.command == "build":
            _print_result(
                build_d105_phase1_component(
                    args.strict_tap_archive,
                    args.strict_tap_receipt,
                    args.candidate_method_lock,
                    args.candidate_runtime_manifest,
                    args.source_held_prediction_manifest,
                    args.source_held_truth_open_receipt,
                    args.source_held_score_artifact,
                    args.d102_revocation_manifest,
                    args.d102_revocation_signature,
                    args.output_dir,
                )
            )
        elif args.command == "seal":
            _print_result(
                seal_d105_phase1_component(
                    args.component_dir,
                    args.authority_envelope,
                    args.authority_signature,
                    args.independent_review_receipt,
                    args.d102_revocation_manifest,
                    args.d102_revocation_signature,
                    args.nonce_ledger_dir,
                    args.output_dir,
                )
            )
        elif args.command == "validate":
            _print_result(
                validate_d105_phase1_asset(
                    args.bundle_dir,
                    require_formal_phase2_eligible=bool(
                        args.require_formal_phase2_eligible
                    ),
                )
            )
        else:
            raise AssertionError("argparse subcommand closure drift")
    except (D105Phase1BundleError, FileNotFoundError, ValueError) as error:
        parser.error(str(error))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
