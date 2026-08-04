#!/usr/bin/env python3
"""Build a truth-free capsule, then predict NEXT-R2 in a separate process."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
from typing import Any, Mapping

import numpy as np


SCRIPT_ROOT = Path(__file__).resolve().parent
CODE_ROOT = SCRIPT_ROOT.parent
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

from cvsrffi import stage2_next_r2_bssdg as bssdg  # noqa: E402
from cvsrffi import stage2_next_r2_matrix as matrix  # noqa: E402
from cvsrffi import stage2_next_r2_real as real  # noqa: E402
from cvsrffi import stage2_next_r2_runtime as runtime  # noqa: E402


SCHEMA = "cvs.stage2.next_r2.proxy24.runner.v2"
BUILDER_RECEIPT_SCHEMA = "cvs.stage2.next_r2.capsule_builder_receipt.v1"
COMPLETION_SCHEMA = "cvs.stage2.next_r2.proxy24.completion.v1"


class NextR2Proxy24Error(ValueError):
    """The immutable NEXT-R2 build or prediction closure did not hold."""


def _plain(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _plain(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_plain(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


def _sha(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha_file(path: Path) -> str:
    return _sha(path.read_bytes())


def _write_json_new(path: Path, value: Any) -> None:
    with path.open("x", encoding="utf-8", newline="\n") as handle:
        json.dump(_plain(value), handle, ensure_ascii=False, sort_keys=True, indent=2)
        handle.write("\n")


def _write_bytes_new(path: Path, value: bytes) -> None:
    with path.open("xb") as handle:
        handle.write(value)


def _require_new_absolute_file(path: Path, *, name: str) -> Path:
    resolved = path.resolve(strict=False)
    if path != resolved or not path.is_absolute() or path.exists() or not path.parent.is_dir():
        raise NextR2Proxy24Error(
            f"{name} must be a new resolved absolute child of an existing directory"
        )
    return path


def _new_root(path: Path) -> Path:
    _require_new_absolute_file(path, name="run root")
    path.mkdir()
    (path / "states").mkdir()
    return path


def _safe_stem(outer_key_id: str, state_id: str) -> str:
    if not outer_key_id.startswith("n2-") or state_id not in matrix.STATE_IDS:
        raise NextR2Proxy24Error("state artifact identity drift")
    return f"{outer_key_id}__{state_id}"


def _save_state(root: Path, result: runtime.NextR2StateResult) -> Mapping[str, Any]:
    states = root / "states"
    stem = _safe_stem(result.outer_key_id, result.state_id)
    npz_path = states / f"{stem}.npz"
    json_path = states / f"{stem}.json"
    head_path = states / f"{stem}.bssdg.wire"
    cvfr_path = states / f"{stem}.cvfr.wire"
    with npz_path.open("xb") as handle:
        np.savez_compressed(
            handle,
            query_physical_ids=np.asarray(result.query_physical_ids, dtype=np.str_),
            registered_classes=np.asarray(result.registered_classes, dtype=np.str_),
            scores=np.ascontiguousarray(result.scores, dtype=np.float32),
            predictions=np.asarray(result.predictions, dtype=np.str_),
        )
    _write_bytes_new(head_path, bssdg.serialize_bssdg_state(result.bssdg_state))
    cvfr_sha: str | None = None
    cvfr_relative: str | None = None
    if result.cvfr_state is not None:
        _write_bytes_new(cvfr_path, result.cvfr_state.to_wire())
        cvfr_sha = _sha_file(cvfr_path)
        cvfr_relative = cvfr_path.relative_to(root).as_posix()
    seal = runtime.state_seal(result)
    payload = {
        "schema": runtime.STATE_RECEIPT_SCHEMA,
        "outer_key_id": result.outer_key_id,
        "state_id": result.state_id,
        "receipt": result.receipt,
        "seal": seal,
        "npz_path": npz_path.relative_to(root).as_posix(),
        "npz_sha256": _sha_file(npz_path),
        "bssdg_wire_path": head_path.relative_to(root).as_posix(),
        "bssdg_wire_sha256": _sha_file(head_path),
        "cvfr_wire_path": cvfr_relative,
        "cvfr_wire_sha256": cvfr_sha,
        "truth_present": False,
        "score_present": False,
    }
    _write_json_new(json_path, payload)
    return {
        "outer_key_id": result.outer_key_id,
        "state_id": result.state_id,
        "json_path": json_path.relative_to(root).as_posix(),
        "json_sha256": _sha_file(json_path),
        "npz_path": npz_path.relative_to(root).as_posix(),
        "npz_sha256": _sha_file(npz_path),
        "state_seal_sha256": seal["state_seal_sha256"],
    }


def run_build_capsule(args: argparse.Namespace) -> Mapping[str, Any]:
    capsule_path = _require_new_absolute_file(args.capsule_output, name="capsule output")
    receipt_path = _require_new_absolute_file(
        args.builder_receipt_output, name="builder receipt output"
    )
    if capsule_path == receipt_path:
        raise NextR2Proxy24Error("capsule and builder receipt paths must differ")
    rows = real.load_next_r2_real_rows(
        selected_iq_archive=args.selected_iq,
        selected_iq_archive_sha256=args.selected_iq_sha256,
        selected_iq_receipt=args.selected_receipt,
        selected_iq_receipt_sha256=args.selected_receipt_sha256,
        ls_label_join_archive=args.ls_join,
        ls_label_join_archive_sha256=args.ls_join_sha256,
    )
    capsule = real.build_next_r2_prediction_capsule(
        rows,
        capsule_id=args.capsule_id,
        split_id=args.split_id,
        selected_iq_archive_sha256=args.selected_iq_sha256,
        selected_iq_receipt_sha256=args.selected_receipt_sha256,
        label_join_archive_sha256=args.ls_join_sha256,
    )
    value = real.capsule_bytes(capsule)
    _write_bytes_new(capsule_path, value)
    receipt = {
        "schema": BUILDER_RECEIPT_SCHEMA,
        "capsule_id": args.capsule_id,
        "split_id": args.split_id,
        "capsule_path": str(capsule_path),
        "capsule_file_sha256": _sha(value),
        "capsule_content_sha256": capsule["capsule_content_sha256"],
        "matrix_sha256": capsule["matrix_sha256"],
        "selected_iq_archive_sha256": args.selected_iq_sha256,
        "selected_iq_receipt_sha256": args.selected_receipt_sha256,
        "label_join_archive_sha256": args.ls_join_sha256,
        "physical_id_root_sha256": capsule["physical_id_root_sha256"],
        "truth_opened_for_capsule_build": True,
        "query_labels_persisted": False,
        "prediction_executed": False,
    }
    _write_json_new(receipt_path, receipt)
    print(json.dumps(receipt, sort_keys=True), flush=True)
    return receipt


def run_predict(args: argparse.Namespace) -> Mapping[str, Any]:
    root = _new_root(args.run_root)
    rows = real.load_next_r2_prediction_rows(
        selected_iq_archive=args.selected_iq,
        selected_iq_archive_sha256=args.selected_iq_sha256,
        selected_iq_receipt=args.selected_receipt,
        selected_iq_receipt_sha256=args.selected_receipt_sha256,
    )
    capsule = real.load_next_r2_prediction_capsule(
        args.capsule,
        capsule_sha256=args.capsule_sha256,
        rows=rows,
    )
    plan = matrix.validate_next_r2_proxy24_plan(capsule["plan"])
    bridge, model_receipt = real.load_next_r2_real_model(
        rows,
        checkpoint_path=args.checkpoint,
        checkpoint_sha256=args.checkpoint_sha256,
        device=args.device,
    )
    first_key = capsule["keys"][0]
    smoke_indices = tuple(first_key["registrations"]["REG1"]["support_indices"][:2])
    smoke = real.verified_next_r2_real_smoke(bridge, smoke_indices)
    preregistration = {
        "schema": SCHEMA,
        "run_id": args.run_id,
        "candidate_id": matrix.CANDIDATE_ID,
        "protocol_schema": matrix.PROTOCOL_SCHEMA,
        "capsule_id": capsule["capsule_id"],
        "split_id": capsule["split_id"],
        "capsule_file_sha256": args.capsule_sha256,
        "capsule_content_sha256": capsule["capsule_content_sha256"],
        "matrix_sha256": plan["matrix_sha256"],
        "outer_key_count": matrix.OUTER_KEY_COUNT,
        "state_prediction_count": matrix.STATE_PREDICTION_COUNT,
        "checkpoint_path": str(args.checkpoint),
        "checkpoint_sha256": args.checkpoint_sha256,
        "device": args.device,
        "prediction_rows_receipt": rows.receipt,
        "model_receipt": model_receipt,
        "smoke_receipt": smoke,
        "label_join_input": None,
        "query_labels_present": False,
        "truth_input": None,
        "truth_scoring_in_process": False,
        "output_overwrite": False,
    }
    _write_json_new(root / "preregistration.json", preregistration)
    _write_json_new(root / "plan.json", plan)
    artifacts: list[Mapping[str, Any]] = []
    for key_mapping in plan["keys"]:
        outer_key = matrix.outer_key_from_mapping(key_mapping)
        inputs = real.build_next_r2_four_state_inputs(
            rows,
            bridge,
            outer_key,
            capsule=capsule,
        )
        results = runtime.execute_next_r2_outer_key(outer_key, inputs)
        artifacts.extend(_save_state(root, result) for result in results)
    manifest = runtime.build_next_r2_sealed_manifest(plan, artifacts)
    _write_json_new(root / "manifest.json", manifest)
    completion = {
        "schema": COMPLETION_SCHEMA,
        "run_id": args.run_id,
        "status": "ARTIFACTS_COMPLETE_NOT_SCORED",
        "capsule_id": capsule["capsule_id"],
        "split_id": capsule["split_id"],
        "outer_keys_completed": matrix.OUTER_KEY_COUNT,
        "states_completed": matrix.STATE_PREDICTION_COUNT,
        "all_states_sealed": True,
        "label_join_opened": False,
        "query_labels_present": False,
        "truth_opened": False,
        "scoring_performed": False,
        "plan_sha256": _sha_file(root / "plan.json"),
        "manifest_sha256": _sha_file(root / "manifest.json"),
        "preregistration_sha256": _sha_file(root / "preregistration.json"),
    }
    _write_json_new(root / "completion.json", completion)
    print(json.dumps(completion, sort_keys=True), flush=True)
    return completion


def _add_selected_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--selected-iq", required=True, type=Path)
    parser.add_argument("--selected-iq-sha256", required=True)
    parser.add_argument("--selected-receipt", required=True, type=Path)
    parser.add_argument("--selected-receipt-sha256", required=True)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    commands = parser.add_subparsers(dest="command", required=True)
    build = commands.add_parser("build-capsule")
    _add_selected_arguments(build)
    build.add_argument("--ls-join", required=True, type=Path)
    build.add_argument("--ls-join-sha256", required=True)
    build.add_argument("--capsule-id", required=True)
    build.add_argument("--split-id", required=True)
    build.add_argument("--capsule-output", required=True, type=Path)
    build.add_argument("--builder-receipt-output", required=True, type=Path)
    build.set_defaults(func=run_build_capsule)
    predict = commands.add_parser("predict")
    predict.add_argument("--run-id", required=True)
    predict.add_argument("--run-root", required=True, type=Path)
    predict.add_argument("--checkpoint", required=True, type=Path)
    predict.add_argument("--checkpoint-sha256", required=True)
    _add_selected_arguments(predict)
    predict.add_argument("--capsule", required=True, type=Path)
    predict.add_argument("--capsule-sha256", required=True)
    predict.add_argument("--device", default="cuda:0")
    predict.set_defaults(func=run_predict)
    return parser


def main() -> None:
    args = _parser().parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
