#!/usr/bin/env python3
"""Predict or independently score one sealed D105 Target25.

The ``predict`` command's ``--context-manifest`` is an immutable,
schema-closed JSON document covering
all 25 rows: D92's four sealed package refs, six before/after split
authorities, the formal D105 Phase1 asset, exact checkpoint and K-specific
qKNN lock.  The launcher binds each row to ``cuda:<assigned GPU>`` and runs
sequentially; it never claims intra-launcher GPU parallelism.  It accepts no
truth labels, query roles, class quotas, or a per-state evaluator shortcut.

The ``score`` command accepts only a read-only, schema-closed truth catalog
whose complete file SHA256 is supplied out of band.  It validates the full
prediction manifest before opening that catalog and exposes no callable truth
provider.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT / "code") not in sys.path:
    sys.path.insert(0, str(ROOT / "code"))

from cvsrffi.stage2_d105_target25_launcher import (  # noqa: E402
    D105Target25LauncherError,
    execute_d105_target25_with_evaluator,
    load_d105_target25_context_factory,
)
from cvsrffi.stage2_d105_target25_inputs import (  # noqa: E402
    D105Target25InputError,
    build_d105_target25_prepare_binding,
)
from cvsrffi.stage2_d105_phase1_authority import (  # noqa: E402
    D105AuthorityError,
    TARGET25_PREPARE_SIGNATURE_DOMAIN,
    compute_d105_nonce_ledger_identity,
    consume_target25_prepare_nonce_once,
    load_signed_d105_target25_prepare_envelope,
)
from cvsrffi.stage2_d105_target25_runner import (  # noqa: E402
    D105Target25GPUSchedule,
    D105Target25RunnerError,
    load_d105_target25_plan_manifest,
    load_d105_target25_run,
    prepare_d105_target25_run,
    score_d105_target25_from_catalog_file,
)


def _gpu_ids(value: str) -> tuple[int, ...]:
    try:
        values = tuple(int(item.strip()) for item in value.split(",") if item.strip())
    except ValueError as error:
        raise argparse.ArgumentTypeError("--gpu-ids must be comma-separated integers") from error
    if not values or any(item < 0 for item in values) or len(set(values)) != len(values):
        raise argparse.ArgumentTypeError("--gpu-ids must be unique non-negative integers")
    return values


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    predict = commands.add_parser(
        "predict", help="execute the truth-free sealed Target25 prediction matrix"
    )
    predict.add_argument("--plan-manifest", type=Path, required=True)
    predict.add_argument("--context-manifest", type=Path, required=True)
    predict.add_argument("--prepare-receipt", type=Path, required=True)
    predict.add_argument("--prepare-receipt-sha256", required=True)
    predict.add_argument("--d92-matrix-index", type=Path, required=True)
    predict.add_argument("--d92-matrix-index-sha256", required=True)
    predict.add_argument("--prepare-authority-envelope", type=Path, required=True)
    predict.add_argument("--prepare-authority-signature", type=Path, required=True)
    predict.add_argument("--git-commit", required=True)
    predict.add_argument("--nonce-ledger-dir", type=Path, required=True)
    predict.add_argument("--output-root", type=Path, required=True)
    predict.add_argument("--run-id", required=True)
    predict.add_argument("--gpu-ids", type=_gpu_ids, required=True)
    predict.add_argument("--workers-per-gpu", type=int, default=1)
    predict.add_argument(
        "--dry-run",
        action="store_true",
        help="verify sealed plan/context inputs and deterministic assignments only",
    )
    score = commands.add_parser(
        "score", help="score a complete run from an externally SHA-bound truth catalog"
    )
    score.add_argument("--plan-manifest", type=Path, required=True)
    score.add_argument("--context-manifest", type=Path, required=True)
    score.add_argument("--prepare-receipt", type=Path, required=True)
    score.add_argument("--prepare-receipt-sha256", required=True)
    score.add_argument("--d92-matrix-index", type=Path, required=True)
    score.add_argument("--d92-matrix-index-sha256", required=True)
    score.add_argument("--prepare-authority-envelope", type=Path, required=True)
    score.add_argument("--prepare-authority-signature", type=Path, required=True)
    score.add_argument("--run-id", required=True)
    score.add_argument("--git-commit", required=True)
    score.add_argument("--nonce-ledger-dir", type=Path, required=True)
    score.add_argument("--run-root", type=Path, required=True)
    score.add_argument("--truth-catalog", type=Path, required=True)
    score.add_argument("--truth-catalog-sha256", required=True)
    score.add_argument("--score-root", type=Path, required=True)
    return parser.parse_args(argv)


def _verify_prepare_authority(
    args: argparse.Namespace,
    plan: object,
    *,
    consume_nonce: bool,
) -> dict[str, object]:
    ledger_identity = compute_d105_nonce_ledger_identity(
        args.nonce_ledger_dir,
        run_id=args.run_id,
        signature_domain=TARGET25_PREPARE_SIGNATURE_DOMAIN,
    )
    binding = build_d105_target25_prepare_binding(
        prepare_receipt_path=args.prepare_receipt,
        expected_prepare_receipt_file_sha256=args.prepare_receipt_sha256,
        matrix_index_path=args.d92_matrix_index,
        expected_matrix_index_sha256=args.d92_matrix_index_sha256,
        plan_manifest_path=args.plan_manifest,
        context_manifest_path=args.context_manifest,
        plan=plan,
        run_id=args.run_id,
        git_commit=args.git_commit,
        nonce_ledger_identity_sha256=ledger_identity,
    )
    signed = load_signed_d105_target25_prepare_envelope(
        args.prepare_authority_envelope,
        args.prepare_authority_signature,
        expected_binding=binding,
    )
    if consume_nonce:
        consume_target25_prepare_nonce_once(
            args.nonce_ledger_dir,
            envelope=signed["envelope"],
            envelope_sha256=signed["envelope_sha256"],
        )
    return signed


def _predict(args: argparse.Namespace) -> int:
    if args.workers_per_gpu != 1:
        raise D105Target25LauncherError(
            "this formal launcher is sequential; --workers-per-gpu must equal 1"
        )
    plan = load_d105_target25_plan_manifest(args.plan_manifest)
    _verify_prepare_authority(args, plan, consume_nonce=not args.dry_run)
    schedule = D105Target25GPUSchedule(
        gpu_ids=args.gpu_ids,
        workers_per_gpu=args.workers_per_gpu,
    )
    context_factory = load_d105_target25_context_factory(args.context_manifest, plan)
    if args.dry_run:
        capacity = len(schedule.gpu_ids) * schedule.workers_per_gpu
        assignments = [
            {
                "row_id": row.row_id,
                "gpu_id": schedule.gpu_ids[index % len(schedule.gpu_ids)],
                "worker_slot": (index % capacity) // len(schedule.gpu_ids),
            }
            for index, row in enumerate(plan.rows)
        ]
        print(
            json.dumps(
                {
                    "status": "DRY_RUN_VALIDATED",
                    "plan_receipt_sha256": plan.plan_receipt_sha256,
                    "outer_row_count": len(plan.rows),
                    "assignments": assignments,
                    "context_manifest": str(args.context_manifest),
                    "execution_mode": "SEQUENTIAL_PER_ROW_GPU_BOUND",
                    "max_concurrent_rows": 1,
                },
                ensure_ascii=False,
                sort_keys=True,
            )
        )
        return 0
    run = prepare_d105_target25_run(
        plan,
        output_root=args.output_root,
        run_id=args.run_id,
        schedule=schedule,
    )
    summary = execute_d105_target25_with_evaluator(run, context_factory)
    print(
        json.dumps(
            {
                "status": summary.status,
                "run_root": str(run.run_root),
                "manifest_path": str(summary.manifest_path),
                "outer_rows": summary.completed_outer_rows,
                "scenario_arm_pair_count": summary.scenario_arm_pair_count,
                "state_prediction_surface_count": summary.state_prediction_surface_count,
                "stop_dispatch": summary.stop_dispatch,
                "stop_fingerprint_sha256": summary.stop_fingerprint_sha256,
                "execution_mode": "SEQUENTIAL_PER_ROW_GPU_BOUND",
                "max_concurrent_rows": 1,
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )
    return 0 if summary.status == "PREDICTIONS_COMPLETE" else 3


def _score(args: argparse.Namespace) -> int:
    plan = load_d105_target25_plan_manifest(args.plan_manifest)
    _verify_prepare_authority(args, plan, consume_nonce=False)
    run = load_d105_target25_run(plan, args.run_root)
    if run.run_id != args.run_id:
        raise D105Target25RunnerError("score run ID differs from signed prepare authority")
    output = score_d105_target25_from_catalog_file(
        run,
        truth_catalog_path=args.truth_catalog,
        expected_truth_catalog_sha256=args.truth_catalog_sha256,
        score_root=args.score_root,
    )
    print(
        json.dumps(
            {
                "status": "SCORES_COMPLETE",
                "run_root": str(run.run_root),
                "score_manifest_path": str(output),
                "truth_catalog_sha256": args.truth_catalog_sha256,
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )
    return 0


def main(argv: list[str] | None = None) -> int:
    try:
        args = parse_args(argv)
        return _predict(args) if args.command == "predict" else _score(args)
    except (
        D105Target25InputError,
        D105AuthorityError,
        D105Target25LauncherError,
        D105Target25RunnerError,
        FileNotFoundError,
    ) as error:
        print(f"run_d105_target25: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
