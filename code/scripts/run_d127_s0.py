"""Immutable, truth-free prediction release for the frozen D127 S0 matrix.

This entry deliberately has three small modes.  ``prepare`` makes the sealed
18-pair plan, each ``candidate-worker`` runs exactly one Phase1 A/B/C asset,
and ``merge`` keeps one byte-identical M0/M_L92 public arm while retaining all
candidate-specific M_DA/M_JOINT arms.  It never accepts labels, truth, roles,
or quotas as command-line inputs.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
from typing import Any, Sequence


_ROOT = Path(__file__).resolve().parents[2]
_CODE = _ROOT / "code"
if str(_CODE) not in sys.path:
    sys.path.insert(0, str(_CODE))

from cvsrffi import stage2_d127_checkpoint_hooks as checkpoint_hooks
from cvsrffi import stage2_d127_s0_entry as entry
from cvsrffi import stage2_d127_s0_package_adapter as adapter


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _emit(status: MappingLike) -> None:
    print(json.dumps(status, ensure_ascii=True, sort_keys=True, separators=(",", ":")))


MappingLike = dict[str, Any]


def _materialize(args: argparse.Namespace) -> adapter.D127S0PreparedPackageRows:
    return adapter.materialize_d127_s0_package_rows(
        method_lock_path=args.method_lock,
        expected_method_lock_sha256=args.method_lock_sha256,
        d106_context_path=args.d106_context,
        expected_d106_context_sha256=args.d106_context_sha256,
    )


def _check_method_lock_against_plan(args: argparse.Namespace, plan: MappingLike) -> MappingLike:
    document, observed, _locks = adapter.load_d127_s0_method_lock(
        args.method_lock, expected_sha256=args.method_lock_sha256
    )
    if observed != plan["method_lock_sha256"]:
        raise adapter.D127S0PackageAdapterError("D127 method-lock/plan SHA256 drift")
    checkpoint = document.get("checkpoint")
    if not isinstance(checkpoint, dict) or checkpoint.get("sha256") != plan["checkpoint_sha256"]:
        raise adapter.D127S0PackageAdapterError("D127 method-lock checkpoint/plan drift")
    return document


def _prepare(args: argparse.Namespace) -> MappingLike:
    output_dir = Path(args.output_dir)
    if output_dir.exists() or output_dir.is_symlink():
        raise adapter.D127S0PackageAdapterError("D127 prepare output directory already exists")
    prepared = _materialize(args)
    plan = adapter.build_d127_s0_prepared_plan(prepared)
    output_dir.mkdir(parents=True, exist_ok=False)
    prefix_path = output_dir / "k5_prefix_receipt.json"
    plan_path = output_dir / "prepared_plan.json"
    adapter.write_d127_s0_prefix_receipt_exclusive(prefix_path, prepared.prefix_receipt)
    adapter.write_d127_s0_prepared_plan_exclusive(plan_path, plan)
    return {
        "status": "D127_S0_PREPARED",
        "row_pair_count": plan["row_pair_count"],
        "state_row_count": plan["state_row_count"],
        "truth_loaded": False,
        "prefix_receipt": str(prefix_path.resolve()),
        "prefix_receipt_sha256": plan["prefix_receipt_sha256"],
        "prefix_receipt_file_sha256": _sha256_file(prefix_path),
        "prepared_plan": str(plan_path.resolve()),
        "prepared_plan_sha256": plan["prepared_plan_sha256"],
        "prepared_plan_file_sha256": _sha256_file(plan_path),
    }


def _candidate_worker(args: argparse.Namespace) -> MappingLike:
    plan, _observed = adapter.load_d127_s0_prepared_plan(
        args.prepared_plan, expected_sha256=args.prepared_plan_sha256
    )
    _check_method_lock_against_plan(args, plan)
    prepared = _materialize(args)
    model, checkpoint_receipt = checkpoint_hooks.load_d127_frozen_checkpoint(
        args.checkpoint, device=args.device
    )
    if checkpoint_receipt.get("checkpoint_sha256") != plan["checkpoint_sha256"]:
        raise adapter.D127S0PackageAdapterError("D127 strict checkpoint/plan SHA256 drift")
    asset, phase1_manifest_receipt = adapter.load_d127_s0_candidate_asset(
        bundle_dir=args.phase1_asset_bundle,
        expected_manifest_sha256=args.phase1_asset_manifest_sha256,
        candidate_id=args.candidate_id,
        device=args.device,
        prepared_plan=plan,
    )
    payload = adapter.run_d127_s0_candidate_worker_pair(
        model=model,
        candidate_id=args.candidate_id,
        asset=asset,
        prepared=prepared,
        prepared_plan=plan,
        phase1_asset_manifest_sha256=args.phase1_asset_manifest_sha256,
        phase1_manifest_receipt=phase1_manifest_receipt,
        checkpoint_sha256=checkpoint_receipt["checkpoint_sha256"],
    )
    output = adapter.write_d127_s0_candidate_worker_exclusive(args.output, payload)
    return {
        "status": "D127_S0_CANDIDATE_WORKER_COMPLETE",
        "candidate_id": args.candidate_id,
        "row_pair_count": payload["row_pair_count"],
        "state_row_count": payload["state_row_count"],
        "truth_loaded": False,
        "output": str(output.resolve()),
        "output_sha256": _sha256_file(output),
        "physical_base_forwards_are_repeated_per_candidate": True,
    }


def _merge(args: argparse.Namespace) -> MappingLike:
    plan, _observed = adapter.load_d127_s0_prepared_plan(
        args.prepared_plan, expected_sha256=args.prepared_plan_sha256
    )
    _check_method_lock_against_plan(args, plan)
    if len(args.worker_prediction) != len(entry.CANDIDATE_IDS) or len(args.worker_prediction_sha256) != len(entry.CANDIDATE_IDS):
        raise adapter.D127S0PackageAdapterError("D127 merge requires exactly three worker path/hash pairs")
    workers: list[MappingLike] = []
    for path, expected_sha256 in zip(args.worker_prediction, args.worker_prediction_sha256, strict=True):
        worker, _digest = adapter.load_d127_s0_candidate_worker(path, expected_sha256=expected_sha256)
        workers.append(worker)
    paired = adapter.merge_d127_s0_candidate_workers(prepared_plan=plan, workers=workers)
    output = adapter.write_d127_s0_paired_prediction_exclusive(args.output, paired, prepared_plan=plan)
    return {
        "status": "D127_S0_PAIRED_PREDICTION_COMPLETE",
        "candidate_ids": paired["candidate_ids"],
        "row_pair_count": paired["row_pair_count"],
        "state_row_count": paired["state_row_count"],
        "truth_loaded": False,
        "output": str(output.resolve()),
        "output_sha256": _sha256_file(output),
        "pair_manifest_sha256": paired["pair_manifest"]["pair_manifest_sha256"],
        "physical_base_forwards_are_repeated_per_candidate": True,
    }


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    actions = parser.add_subparsers(dest="mode", required=True)

    prepare = actions.add_parser("prepare", help="write one immutable S0 plan and K5 prefix receipt")
    prepare.add_argument("--method-lock", required=True, type=Path)
    prepare.add_argument("--method-lock-sha256", required=True)
    prepare.add_argument("--d106-context", required=True, type=Path)
    prepare.add_argument("--d106-context-sha256", required=True)
    prepare.add_argument("--output-dir", required=True, type=Path)
    prepare.set_defaults(handler=_prepare)

    worker = actions.add_parser("candidate-worker", help="run one frozen A/B/C candidate on before and after rows")
    worker.add_argument("--prepared-plan", required=True, type=Path)
    worker.add_argument("--prepared-plan-sha256", required=True)
    worker.add_argument("--method-lock", required=True, type=Path)
    worker.add_argument("--method-lock-sha256", required=True)
    worker.add_argument("--d106-context", required=True, type=Path)
    worker.add_argument("--d106-context-sha256", required=True)
    worker.add_argument("--phase1-asset-bundle", required=True, type=Path)
    worker.add_argument("--phase1-asset-manifest-sha256", required=True)
    worker.add_argument("--checkpoint", required=True, type=Path)
    worker.add_argument("--candidate-id", required=True, choices=entry.CANDIDATE_IDS)
    worker.add_argument("--device", default="cpu")
    worker.add_argument("--output", required=True, type=Path)
    worker.set_defaults(handler=_candidate_worker)

    merge = actions.add_parser("merge", help="verify and merge exactly three candidate worker outputs")
    merge.add_argument("--prepared-plan", required=True, type=Path)
    merge.add_argument("--prepared-plan-sha256", required=True)
    merge.add_argument("--method-lock", required=True, type=Path)
    merge.add_argument("--method-lock-sha256", required=True)
    merge.add_argument("--worker-prediction", required=True, action="append", type=Path)
    merge.add_argument("--worker-prediction-sha256", required=True, action="append")
    merge.add_argument("--output", required=True, type=Path)
    merge.set_defaults(handler=_merge)
    return parser


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    return _build_parser().parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    _emit(args.handler(args))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
