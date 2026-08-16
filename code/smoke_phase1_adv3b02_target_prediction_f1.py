"""File-backed F1 technical smoke for the frozen ADV blind predictor.

The smoke derives one source-only train-data config, strictly reconstructs the
ADV model, and forwards one local all-zero IQ sample.  It has no IQ-package,
target, truth, role, known-config, reference, scorer, metric, retry, fitting,
update, or selection input.
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

import evaluate_phase1_adv3b02_target_leo as _adv
from cvsrffi.leo_weak_cache import sha256_file


SMOKE_SCHEMA = "cvs.phase1.adv3b02_target_prediction_technical_smoke.v2"
SMOKE_RUN_ID = "phase1_adv3b02_target_prediction_20260816_v2"
ADV_TRAINING_RUN_ID = "phase1_adv3b02_clic6_20260816_v2"
ADV_CANDIDATE_ID = "F1_ADV3B02_CLIC"
CLIC_CLEAN_CANDIDATE_ID = "F1C_CLIC12"
SMOKE_SCENE = "leo_clear_weak"
SMOKE_INPUT_LEN = 256
SOURCE_CLASS_COUNT = 4


class ADV3B02TargetSmokeError(_adv.ADV3B02TargetProtocolError):
    """Raised when the one-shot blind technical smoke cannot be proven safe."""


def _path(value: str | Path, *, label: str) -> Path:
    if not isinstance(value, (str, Path)) or not str(value).strip():
        raise ADV3B02TargetSmokeError(f"{label} must be a path")
    return Path(value).resolve()


def _read_json(path: Path, *, label: str) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ADV3B02TargetSmokeError(f"{label} cannot be reopened") from exc
    if not isinstance(payload, Mapping):
        raise ADV3B02TargetSmokeError(f"{label} must be an object")
    return {str(key): value for key, value in payload.items()}


def _require_f1_paths(
    *, checkpoint: Path, completion: Path, clean_v4: Path | None = None
) -> None:
    if (
        checkpoint.name != "final_ssdg.pth"
        or checkpoint.parent.name != ADV_CANDIDATE_ID
        or checkpoint.parent.parent.name != ADV_TRAINING_RUN_ID
    ):
        raise ADV3B02TargetSmokeError("technical smoke requires the frozen F1 ADV checkpoint")
    if completion != checkpoint.parent / "phase1_training_completion_receipt.json":
        raise ADV3B02TargetSmokeError(
            "technical smoke requires the F1 checkpoint completion receipt"
        )
    if clean_v4 is not None and (
        clean_v4.name != "source_clean_proxy.npz"
        or clean_v4.parent.name != CLIC_CLEAN_CANDIDATE_ID
        or clean_v4.parent.parent.name != "phase1_clic_postfreeze_20260812_v4"
    ):
        raise ADV3B02TargetSmokeError(
            "technical smoke requires the canonical F1C clean-v4 authority"
        )


def _snapshot(paths: Mapping[str, Path]) -> dict[str, str]:
    result: dict[str, str] = {}
    for label, path in paths.items():
        if not path.is_file():
            raise FileNotFoundError(f"{label} is missing: {path}")
        result[label] = sha256_file(path)
    return result


def _verify_snapshot(paths: Mapping[str, Path], expected: Mapping[str, str]) -> None:
    for label, path in paths.items():
        if sha256_file(path) != expected[label]:
            raise ADV3B02TargetSmokeError(f"{label} changed during technical smoke")


def _require_exact_fields(payload: Mapping[str, Any]) -> None:
    expected = {
        "schema": SMOKE_SCHEMA,
        "completed": True,
        "claim": "NO_PERFORMANCE_RESULT",
        "run_id": SMOKE_RUN_ID,
        "adv_training_run_id": ADV_TRAINING_RUN_ID,
        "candidate_id": ADV_CANDIDATE_ID,
        "fold": "F1",
        "scene": SMOKE_SCENE,
        "input_shape": [2, SMOKE_INPUT_LEN],
        "source_class_count": SOURCE_CLASS_COUNT,
        "finite_logit_count": SOURCE_CLASS_COUNT,
        "strict_runtime_load": True,
        "synthetic_local_input_count": 1,
        "forward_count": 1,
        "target_rows_opened": 0,
        "query_rows_opened": 0,
        "target_fit_rows": 0,
        "target_update_rows": 0,
        "target_retry_count": 0,
        "target_selection_count": 0,
        "target_selection_feedback": False,
        "iq_only_package_opened": False,
        "truth_sidecar_opened": False,
        "known_test_config_opened": False,
        "reference_opened": False,
        "metrics_opened": False,
        "formal_invocation": 0,
        "smoke_invocation": 1,
        "retry_authorized": False,
        "baseline_terminal_status": "NON_PROMOTABLE_P0_DISABLED",
        "baseline_exit_code": 8,
        "baseline_promotion_ready": False,
        "formal_performance_claim": False,
    }
    for field, value in expected.items():
        observed = payload.get(field)
        if type(observed) is not type(value) or observed != value:
            raise ADV3B02TargetSmokeError(
                f"technical smoke receipt field drift: {field}"
            )


def run_f1_technical_smoke(
    *,
    checkpoint_path: str | Path,
    completion_receipt_path: str | Path,
    clean_v4_npz_path: str | Path,
    train_config_output_path: str | Path,
    receipt_output_path: str | Path,
) -> Path:
    """Run the only allowed synthetic F1 strict-forward smoke exactly once."""

    checkpoint = _path(checkpoint_path, label="F1 ADV checkpoint")
    completion = _path(
        completion_receipt_path, label="F1 ADV completion receipt"
    )
    clean_v4 = _path(clean_v4_npz_path, label="F1C clean-v4 authority")
    train_config = _path(
        train_config_output_path, label="technical smoke train-config output"
    )
    receipt = _path(receipt_output_path, label="technical smoke receipt output")
    _require_f1_paths(
        checkpoint=checkpoint, completion=completion, clean_v4=clean_v4
    )
    if receipt.exists():
        raise ADV3B02TargetSmokeError(
            f"technical smoke receipt output already exists and is immutable: {receipt}"
        )
    if train_config.exists():
        raise ADV3B02TargetSmokeError(
            "technical smoke train-config output already exists and is immutable: "
            f"{train_config}"
        )
    if not train_config.parent.is_dir() or not receipt.parent.is_dir():
        raise ADV3B02TargetSmokeError("technical smoke output parent is missing")

    source_inputs = {
        "checkpoint": checkpoint,
        "completion_receipt": completion,
        "clean_v4": clean_v4,
    }
    source_hashes = _snapshot(source_inputs)
    sealed_config = _adv.seal_adv3b02_train_data_config(
        checkpoint_path=checkpoint,
        completion_receipt_path=completion,
        clean_v4_npz_path=clean_v4,
        output_path=train_config,
    )
    if sealed_config.resolve() != train_config:
        raise ADV3B02TargetSmokeError("technical smoke train-config output path drift")
    _verify_snapshot(source_inputs, source_hashes)

    config = _read_json(train_config, label="technical smoke train-data config")
    normalized = config.get("normalized")
    if not isinstance(normalized, Mapping):
        raise ADV3B02TargetSmokeError("technical smoke normalized train config is missing")
    preprocessing = normalized.get("preprocessing")
    if not isinstance(preprocessing, Mapping) or preprocessing.get(
        "input_len"
    ) != SMOKE_INPUT_LEN:
        raise ADV3B02TargetSmokeError("technical smoke requires input_len=256")
    if config.get("fold_index") != 1 or config.get("clic_clean_arm") != "C":
        raise ADV3B02TargetSmokeError("technical smoke train config is not F1C-bound")
    if config.get("clean_v4_npz_sha256") != source_hashes["clean_v4"]:
        raise ADV3B02TargetSmokeError("technical smoke clean-v4 SHA binding drift")

    runtime = _adv.load_verified_adv3b02_runtime(
        checkpoint_path=checkpoint,
        completion_receipt_path=completion,
        train_config_manifest_path=train_config,
    )
    local_zero_iq = np.zeros((2, SMOKE_INPUT_LEN), dtype=np.float32)
    output = runtime.forward_once(local_zero_iq, scene=SMOKE_SCENE)
    if not isinstance(output, Mapping):
        raise ADV3B02TargetSmokeError("technical smoke forward output is not an object")
    try:
        logits = np.asarray(output.get("tx_logits"), dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ADV3B02TargetSmokeError("technical smoke logits are invalid") from exc
    if logits.shape != (SOURCE_CLASS_COUNT,) or not np.isfinite(logits).all():
        raise ADV3B02TargetSmokeError(
            "technical smoke requires exactly four finite logits"
        )
    runtime_order = list(getattr(runtime, "source_class_order", []))
    if runtime_order != config.get("source_class_order") or len(
        runtime_order
    ) != SOURCE_CLASS_COUNT:
        raise ADV3B02TargetSmokeError("technical smoke source class order drift")

    config_hash = sha256_file(train_config)
    all_inputs = {**source_inputs, "train_config_manifest": train_config}
    all_hashes = {**source_hashes, "train_config_manifest": config_hash}
    _verify_snapshot(all_inputs, all_hashes)
    payload = {
        "schema": SMOKE_SCHEMA,
        "completed": True,
        "claim": "NO_PERFORMANCE_RESULT",
        "run_id": SMOKE_RUN_ID,
        "adv_training_run_id": ADV_TRAINING_RUN_ID,
        "candidate_id": ADV_CANDIDATE_ID,
        "fold": "F1",
        "scene": SMOKE_SCENE,
        "input_shape": [2, SMOKE_INPUT_LEN],
        "source_class_count": SOURCE_CLASS_COUNT,
        "finite_logit_count": int(np.isfinite(logits).sum()),
        "strict_runtime_load": True,
        "synthetic_local_input_count": 1,
        "forward_count": 1,
        "target_rows_opened": 0,
        "query_rows_opened": 0,
        "target_fit_rows": 0,
        "target_update_rows": 0,
        "target_retry_count": 0,
        "target_selection_count": 0,
        "target_selection_feedback": False,
        "iq_only_package_opened": False,
        "truth_sidecar_opened": False,
        "known_test_config_opened": False,
        "reference_opened": False,
        "metrics_opened": False,
        "formal_invocation": 0,
        "smoke_invocation": 1,
        "retry_authorized": False,
        "checkpoint_sha256": source_hashes["checkpoint"],
        "completion_receipt_sha256": source_hashes["completion_receipt"],
        "clean_v4_npz_sha256": source_hashes["clean_v4"],
        "train_config_manifest_sha256": config_hash,
        "train_config_normalized_sha256": config.get("normalized_sha256"),
        "train_config_physical_axis_binding_sha256": config.get(
            "physical_axis_binding_sha256"
        ),
        "source_class_order_sha256": config.get("source_class_order_sha256"),
        "baseline_terminal_status": config.get("baseline_terminal_status"),
        "baseline_exit_code": config.get("baseline_exit_code"),
        "baseline_promotion_ready": config.get("baseline_promotion_ready"),
        "formal_performance_claim": config.get("formal_performance_claim"),
        "input_artifact_sha256": {
            "checkpoint": source_hashes["checkpoint"],
            "completion_receipt": source_hashes["completion_receipt"],
            "clean_v4": source_hashes["clean_v4"],
            "train_config_manifest": config_hash,
        },
    }
    _require_exact_fields(payload)
    _verify_snapshot(all_inputs, all_hashes)
    try:
        return _adv._write_immutable_json(
            receipt, payload, label="ADV target prediction v2 technical smoke receipt"
        ).resolve()
    except _adv.ADV3B02TargetProtocolError as exc:
        raise ADV3B02TargetSmokeError(str(exc)) from exc


def validate_f1_technical_smoke_receipt(
    *,
    checkpoint_path: str | Path,
    completion_receipt_path: str | Path,
    train_config_manifest_path: str | Path,
    receipt_path: str | Path,
) -> Path:
    """Validate a completed smoke receipt without clean, target, or query access."""

    checkpoint = _path(checkpoint_path, label="F1 ADV checkpoint")
    completion = _path(
        completion_receipt_path, label="F1 ADV completion receipt"
    )
    train_config = _path(
        train_config_manifest_path, label="technical smoke train-data config"
    )
    receipt = _path(receipt_path, label="technical smoke receipt")
    _require_f1_paths(checkpoint=checkpoint, completion=completion)
    inputs = {
        "checkpoint": checkpoint,
        "completion_receipt": completion,
        "train_config_manifest": train_config,
        "technical_smoke_receipt": receipt,
    }
    hashes = _snapshot(inputs)
    payload = _read_json(receipt, label="technical smoke receipt")
    _require_exact_fields(payload)
    expected_hash_fields = {
        "checkpoint_sha256": hashes["checkpoint"],
        "completion_receipt_sha256": hashes["completion_receipt"],
        "train_config_manifest_sha256": hashes["train_config_manifest"],
    }
    for field, expected in expected_hash_fields.items():
        if payload.get(field) != expected:
            raise ADV3B02TargetSmokeError(
                f"technical smoke receipt input SHA drift: {field}"
            )
    config = _read_json(train_config, label="technical smoke train-data config")
    config_fields = {
        "clean_v4_npz_sha256": "clean_v4_npz_sha256",
        "train_config_normalized_sha256": "normalized_sha256",
        "train_config_physical_axis_binding_sha256": (
            "physical_axis_binding_sha256"
        ),
        "source_class_order_sha256": "source_class_order_sha256",
    }
    for receipt_field, config_field in config_fields.items():
        if payload.get(receipt_field) != config.get(config_field):
            raise ADV3B02TargetSmokeError(
                f"technical smoke receipt train-config binding drift: {receipt_field}"
            )
    expected_artifacts = {
        "checkpoint": hashes["checkpoint"],
        "completion_receipt": hashes["completion_receipt"],
        "clean_v4": config.get("clean_v4_npz_sha256"),
        "train_config_manifest": hashes["train_config_manifest"],
    }
    if payload.get("input_artifact_sha256") != expected_artifacts:
        raise ADV3B02TargetSmokeError(
            "technical smoke receipt input artifact SHA map drift"
        )
    runtime = _adv.load_verified_adv3b02_runtime(
        checkpoint_path=checkpoint,
        completion_receipt_path=completion,
        train_config_manifest_path=train_config,
    )
    if list(getattr(runtime, "source_class_order", [])) != config.get(
        "source_class_order"
    ):
        raise ADV3B02TargetSmokeError(
            "technical smoke receipt strict runtime class binding drift"
        )
    _verify_snapshot(inputs, hashes)
    return receipt


def build_arg_parser() -> argparse.ArgumentParser:
    """Build the frozen smoke/receipt-validation CLI without target inputs."""

    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_mutually_exclusive_group(required=True)
    modes.add_argument("--run-smoke", action="store_true")
    modes.add_argument("--validate-receipt", action="store_true")
    parser.add_argument("--checkpoint")
    parser.add_argument("--completion-receipt-json")
    parser.add_argument("--clean-v4-npz")
    parser.add_argument("--train-config-output")
    parser.add_argument("--receipt-output")
    return parser


def _require_cli_paths(
    parser: argparse.ArgumentParser, args: argparse.Namespace, *fields: str
) -> None:
    missing = [
        f"--{field.replace('_', '-')}"
        for field in fields
        if not isinstance(getattr(args, field, None), str)
        or not str(getattr(args, field)).strip()
    ]
    if missing:
        parser.error(f"selected mode requires {', '.join(missing)}")


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    if args.run_smoke:
        _require_cli_paths(
            parser,
            args,
            "checkpoint",
            "completion_receipt_json",
            "clean_v4_npz",
            "train_config_output",
            "receipt_output",
        )
        result = run_f1_technical_smoke(
            checkpoint_path=args.checkpoint,
            completion_receipt_path=args.completion_receipt_json,
            clean_v4_npz_path=args.clean_v4_npz,
            train_config_output_path=args.train_config_output,
            receipt_output_path=args.receipt_output,
        )
    else:
        _require_cli_paths(
            parser,
            args,
            "checkpoint",
            "completion_receipt_json",
            "train_config_output",
            "receipt_output",
        )
        result = validate_f1_technical_smoke_receipt(
            checkpoint_path=args.checkpoint,
            completion_receipt_path=args.completion_receipt_json,
            train_config_manifest_path=args.train_config_output,
            receipt_path=args.receipt_output,
        )
    print(str(result))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
